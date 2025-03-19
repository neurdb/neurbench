"""
Code was adapted from https://github.com/Yura52/rtdl
"""

import math
from typing import Callable, List, Type, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch import Tensor

ModuleType = Union[str, Callable[..., nn.Module]]


class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def _is_glu_activation(activation: ModuleType):
    return (
        isinstance(activation, str)
        and activation.endswith("GLU")
        or activation in [ReGLU, GEGLU]
    )


def _all_or_none(values):
    assert all(x is None for x in values) or all(x is not None for x in values)


def reglu(x: Tensor) -> Tensor:
    """The ReGLU activation function from [1].
    References:
        [1] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """
    assert x.shape[-1] % 2 == 0
    a, b = x.chunk(2, dim=-1)
    return a * F.relu(b)


def geglu(x: Tensor) -> Tensor:
    """The GEGLU activation function from [1].
    References:
        [1] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """
    assert x.shape[-1] % 2 == 0
    a, b = x.chunk(2, dim=-1)
    return a * F.gelu(b)


class ReGLU(nn.Module):
    """The ReGLU activation function from [shazeer2020glu].
    Examples:
        .. testcode::
            module = ReGLU()
            x = torch.randn(3, 4)
            assert module(x).shape == (3, 2)
    References:
        * [shazeer2020glu] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """

    def forward(self, x: Tensor) -> Tensor:
        return reglu(x)


class GEGLU(nn.Module):
    """The GEGLU activation function from [shazeer2020glu].
    Examples:
        .. testcode::
            module = GEGLU()
            x = torch.randn(3, 4)
            assert module(x).shape == (3, 2)
    References:
        * [shazeer2020glu] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """

    def forward(self, x: Tensor) -> Tensor:
        return geglu(x)


def _make_nn_module(module_type: ModuleType, *args) -> nn.Module:
    return (
        (
            ReGLU()
            if module_type == "ReGLU"
            else GEGLU() if module_type == "GEGLU" else getattr(nn, module_type)(*args)
        )
        if isinstance(module_type, str)
        else module_type(*args)
    )


class MLP(nn.Module):
    """The MLP model used in [gorishniy2021revisiting].
    The following scheme describes the architecture:
    .. code-block:: text
          MLP: (in) -> Block -> ... -> Block -> Linear -> (out)
        Block: (in) -> Linear -> Activation -> Dropout -> (out)
    Examples:
        .. testcode::
            x = torch.randn(4, 2)
            module = MLP.make_baseline(x.shape[1], [3, 5], 0.1, 1)
            assert module(x).shape == (len(x), 1)
    References:
        * [gorishniy2021revisiting] Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, Artem Babenko, "Revisiting Deep Learning Models for Tabular Data", 2021
    """

    class Block(nn.Module):
        """The main building block of `MLP`."""

        def __init__(
            self,
            *,
            d_in: int,
            d_out: int,
            bias: bool,
            activation: ModuleType,
            dropout: float,
        ) -> None:
            super().__init__()
            self.linear = nn.Linear(d_in, d_out, bias)
            self.activation = _make_nn_module(activation)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x: Tensor) -> Tensor:
            return self.dropout(self.activation(self.linear(x)))

    def __init__(
        self,
        *,
        d_in: int,
        d_layers: List[int],
        dropouts: Union[float, List[float]],
        activation: Union[str, Callable[[], nn.Module]],
        d_out: int,
    ) -> None:
        """
        Note:
            `make_baseline` is the recommended constructor.
        """
        super().__init__()

        pp_dropouts: List[float] = []
        if isinstance(dropouts, list):
            pp_dropouts = dropouts
        else:
            pp_dropouts = [dropouts] * len(d_layers)

        assert len(d_layers) == len(pp_dropouts)
        assert activation not in ["ReGLU", "GEGLU"]

        self.blocks = nn.ModuleList(
            [
                MLP.Block(
                    d_in=d_layers[i - 1] if i else d_in,
                    d_out=d,
                    bias=True,
                    activation=activation,
                    dropout=dropout,
                )
                for i, (d, dropout) in enumerate(zip(d_layers, pp_dropouts))
            ]
        )
        self.head = nn.Linear(d_layers[-1] if d_layers else d_in, d_out)

    @classmethod
    def make_baseline(
        cls: Type["MLP"],
        d_in: int,
        d_layers: List[int],
        dropout: float,
        d_out: int,
    ) -> "MLP":
        """Create a "baseline" `MLP`.
        This variation of MLP was used in [gorishniy2021revisiting]. Features:
        * :code:`Activation` = :code:`ReLU`
        * all linear layers except for the first one and the last one are of the same dimension
        * the dropout rate is the same for all dropout layers
        Args:
            d_in: the input size
            d_layers: the dimensions of the linear layers. If there are more than two
                layers, then all of them except for the first and the last ones must
                have the same dimension. Valid examples: :code:`[]`, :code:`[8]`,
                :code:`[8, 16]`, :code:`[2, 2, 2, 2]`, :code:`[1, 2, 2, 4]`. Invalid
                example: :code:`[1, 2, 3, 4]`.
            dropout: the dropout rate for all hidden layers
            d_out: the output size
        Returns:
            MLP
        References:
            * [gorishniy2021revisiting] Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, Artem Babenko, "Revisiting Deep Learning Models for Tabular Data", 2021
        """
        assert isinstance(dropout, float)
        if len(d_layers) > 2:
            assert len(set(d_layers[1:-1])) == 1, (
                "if d_layers contains more than two elements, then"
                " all elements except for the first and the last ones must be equal."
            )
        return MLP(
            d_in=d_in,
            d_layers=d_layers,  # type: ignore
            dropouts=dropout,
            activation="SELU",
            d_out=d_out,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x.float()
        for block in self.blocks:
            x = block(x)
        x = self.head(x)
        return x


class ResNet(nn.Module):
    """The ResNet model used in [gorishniy2021revisiting].
    The following scheme describes the architecture:
    .. code-block:: text
        ResNet: (in) -> Linear -> Block -> ... -> Block -> Head -> (out)
                 |-> Norm -> Linear -> Activation -> Dropout -> Linear -> Dropout ->|
                 |                                                                  |
         Block: (in) ------------------------------------------------------------> Add -> (out)
          Head: (in) -> Norm -> Activation -> Linear -> (out)
    Examples:
        .. testcode::
            x = torch.randn(4, 2)
            module = ResNet.make_baseline(
                d_in=x.shape[1],
                n_blocks=2,
                d_main=3,
                d_hidden=4,
                dropout_first=0.25,
                dropout_second=0.0,
                d_out=1
            )
            assert module(x).shape == (len(x), 1)
    References:
        * [gorishniy2021revisiting] Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, Artem Babenko, "Revisiting Deep Learning Models for Tabular Data", 2021
    """

    class Block(nn.Module):
        """The main building block of `ResNet`."""

        def __init__(
            self,
            *,
            d_main: int,
            d_hidden: int,
            bias_first: bool,
            bias_second: bool,
            dropout_first: float,
            dropout_second: float,
            normalization: ModuleType,
            activation: ModuleType,
            skip_connection: bool,
        ) -> None:
            super().__init__()
            self.normalization = _make_nn_module(normalization, d_main)
            self.linear_first = nn.Linear(d_main, d_hidden, bias_first)
            self.activation = _make_nn_module(activation)
            self.dropout_first = nn.Dropout(dropout_first)
            self.linear_second = nn.Linear(d_hidden, d_main, bias_second)
            self.dropout_second = nn.Dropout(dropout_second)
            self.skip_connection = skip_connection

        def forward(self, x: Tensor) -> Tensor:
            x_input = x
            x = self.normalization(x)
            x = self.linear_first(x)
            x = self.activation(x)
            x = self.dropout_first(x)
            x = self.linear_second(x)
            x = self.dropout_second(x)
            if self.skip_connection:
                x = x_input + x
            return x

    class Head(nn.Module):
        """The final module of `ResNet`."""

        def __init__(
            self,
            *,
            d_in: int,
            d_out: int,
            bias: bool,
            normalization: ModuleType,
            activation: ModuleType,
        ) -> None:
            super().__init__()
            self.normalization = _make_nn_module(normalization, d_in)
            self.activation = _make_nn_module(activation)
            self.linear = nn.Linear(d_in, d_out, bias)

        def forward(self, x: Tensor) -> Tensor:
            if self.normalization is not None:
                x = self.normalization(x)
            x = self.activation(x)
            x = self.linear(x)
            return x

    def __init__(
        self,
        *,
        d_in: int,
        n_blocks: int,
        d_main: int,
        d_hidden: int,
        dropout_first: float,
        dropout_second: float,
        normalization: ModuleType,
        activation: ModuleType,
        d_out: int,
    ) -> None:
        """
        Note:
            `make_baseline` is the recommended constructor.
        """
        super().__init__()

        self.first_layer = nn.Linear(d_in, d_main)
        if d_main is None:
            d_main = d_in
        self.blocks = nn.Sequential(
            *[
                ResNet.Block(
                    d_main=d_main,
                    d_hidden=d_hidden,
                    bias_first=True,
                    bias_second=True,
                    dropout_first=dropout_first,
                    dropout_second=dropout_second,
                    normalization=normalization,
                    activation=activation,
                    skip_connection=True,
                )
                for _ in range(n_blocks)
            ]
        )
        self.head = ResNet.Head(
            d_in=d_main,
            d_out=d_out,
            bias=True,
            normalization=normalization,
            activation=activation,
        )

    @classmethod
    def make_baseline(
        cls: Type["ResNet"],
        *,
        d_in: int,
        n_blocks: int,
        d_main: int,
        d_hidden: int,
        dropout_first: float,
        dropout_second: float,
        d_out: int,
    ) -> "ResNet":
        """Create a "baseline" `ResNet`.
        This variation of ResNet was used in [gorishniy2021revisiting]. Features:
        * :code:`Activation` = :code:`ReLU`
        * :code:`Norm` = :code:`BatchNorm1d`
        Args:
            d_in: the input size
            n_blocks: the number of Blocks
            d_main: the input size (or, equivalently, the output size) of each Block
            d_hidden: the output size of the first linear layer in each Block
            dropout_first: the dropout rate of the first dropout layer in each Block.
            dropout_second: the dropout rate of the second dropout layer in each Block.
        References:
            * [gorishniy2021revisiting] Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, Artem Babenko, "Revisiting Deep Learning Models for Tabular Data", 2021
        """
        return cls(
            d_in=d_in,
            n_blocks=n_blocks,
            d_main=d_main,
            d_hidden=d_hidden,
            dropout_first=dropout_first,
            dropout_second=dropout_second,
            normalization="BatchNorm1d",
            activation="ReLU",
            d_out=d_out,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x.float()
        x = self.first_layer(x)
        x = self.blocks(x)
        x = self.head(x)
        return x


class MLPClassifier(nn.Module):
    def __init__(self, d_in, d_layers, num_classes, dropout, dim_t=128, t_in=False):
        super().__init__()
        self.t_in = t_in
        self.dim_t = dim_t
        self.num_classes = num_classes

        if self.t_in:
            self.time_embed = nn.Sequential(
                nn.Linear(dim_t, dim_t), nn.SiLU(), nn.Linear(dim_t, dim_t)
            )

        self.proj = nn.Linear(d_in, dim_t)

        self.mlp = MLP.make_baseline(
            d_in=dim_t, d_layers=d_layers, dropout=dropout, d_out=dim_t
        )

        if num_classes > 2:
            self.d_out = num_classes
        else:
            self.d_out = 1

        self.head = nn.Linear(dim_t, self.d_out)

    def forward(self, x, t=None):
        if self.t_in and t is not None:
            emb = self.time_embed(timestep_embedding(t, self.dim_t))
            x = self.proj(x) + emb
        else:
            x = self.proj(x)

        x = self.mlp(x)
        x = self.head(x)
        if self.num_classes > 2:
            return torch.softmax(x, dim=1)
        else:
            return torch.sigmoid(x)


# class Drifter(nn.Module):
#     def __init__(self, d_in, d_layers, dropout, dim_t=128, dim_drift=128):
#         super().__init__()
#         self.dim_t = dim_t
#         self.dim_drift = dim_drift

#         self.mlp = MLP.make_baseline(
#             d_in=dim_t, d_layers=d_layers, dropout=dropout, d_out=d_in
#         )
#         self.proj = nn.Linear(d_in, dim_t)
#         self.time_embed = nn.Sequential(
#             nn.Linear(dim_t, dim_t), nn.LeakyReLU(), nn.Linear(dim_t, dim_t)
#         )
#         self.drift_proj = nn.Sequential(
#             nn.Linear(1, dim_t), nn.LeakyReLU(), nn.Linear(dim_t, d_in)
#         )

#     def forward(self, x: Tensor, t: Tensor, drift: float):
#         # print("x", x.shape, "t", t.shape, "hist", hist.shape)

#         x = self.proj(x)

#         emb = self.time_embed(timestep_embedding(t, self.dim_t))
#         x = x + emb

#         out1 = self.mlp(x)

#         emb_drift: Tensor = self.drift_proj(torch.tensor([drift]).to(x.device))
#         x = out1 * emb_drift

#         return x


class Drifter(nn.Module):
    def __init__(self, d_in, d_layers, dropout, dim_t=128, dim_drift=128):
        super().__init__()
        self.dim_t = dim_t
        self.d_in = d_in
        self.dim_drift = dim_drift

        self.mlp = MLP.make_baseline(
            d_in=dim_t, d_layers=d_layers, dropout=dropout, d_out=d_in
        )

        self.proj = nn.Linear(d_in, dim_t)
        self.time_embed = nn.Sequential(
            nn.Linear(dim_t, dim_t), nn.SELU(), nn.Linear(dim_t, dim_t)
        )

        # self.drift_proj = nn.Linear(1, d_in)
        self.drift_proj = nn.Sequential(
            nn.Linear(1, dim_t), nn.SELU(), nn.Linear(dim_t, d_in)
        )

    def forward(self, x: Tensor, t: Tensor, drift: float):
        x = self.proj(x)

        emb_time = self.time_embed(timestep_embedding(t, self.dim_t))
        x = x + emb_time
        x = self.mlp(x)

        t_drift = torch.tensor([drift]).to(x.device)
        emb_drift = self.drift_proj(t_drift)
        # x = x * emb_drift[: self.d_in] + emb_drift[self.d_in :]

        x = x * torch.tanh(emb_drift)

        return x


class JSD(nn.Module):
    def forward(self, p: Tensor, q: Tensor):
        m = 0.5 * (p + q)

        # Compute KL divergence for each pair (row-wise)
        kl_pm = F.kl_div(p.log(), m, reduction="none").sum(dim=1)  # Sum over classes
        kl_qm = F.kl_div(q.log(), m, reduction="none").sum(dim=1)  # Sum over classes

        # Compute JS divergence for each pair
        js_div = 0.5 * (kl_pm + kl_qm)
        return js_div


class ResNetEncoder(nn.Module):
    def __init__(
        self, d_in, n_blocks, d_main, d_hidden, dropout, d_out, dim_t=128, t_in=False
    ):
        super().__init__()
        self.t_in = t_in
        self.dim_t = dim_t

        if t_in:
            self.time_embed = nn.Sequential(
                nn.Linear(dim_t, dim_t), nn.SiLU(), nn.Linear(dim_t, dim_t)
            )

        self.proj = nn.Linear(d_in, dim_t)

        self.resnet = ResNet.make_baseline(
            d_in=dim_t,
            n_blocks=n_blocks,
            d_main=d_main,
            d_hidden=d_hidden,
            dropout_first=dropout,
            dropout_second=dropout,
            d_out=dim_t,
        )

        # self.head = nn.Linear(dim_t, d_out)

    def forward(self, x, t=None):
        x = self.proj(x)
        if t is not None and self.t_in:
            emb = self.time_embed(timestep_embedding(t, self.dim_t))
            x = x + emb
        x = self.resnet(x)
        # x = self.head(x)
        return x


class MLPScorer(nn.Module):
    def __init__(self, d_in, d_layers, dropout, dim_t=128, t_in=False):
        super().__init__()
        self.dim_t = dim_t
        self.t_in = t_in
        self.mlp = MLP.make_baseline(
            d_in=dim_t, d_layers=d_layers, dropout=dropout, d_out=1
        )

        self.proj = nn.Linear(d_in, dim_t)
        if t_in:
            self.time_embed = nn.Sequential(
                nn.Linear(dim_t, dim_t), nn.SiLU(), nn.Linear(dim_t, dim_t)
            )
        # self.head = nn.Linear(dim_t, self.d_out)

    def forward(self, x, t=None):
        # x = torch.cat((c, x), dim=1)
        x = self.proj(x)
        if self.t_in and t is not None:
            emb = self.time_embed(timestep_embedding(t, self.dim_t))
            x = x + emb
        x = self.mlp(x)
        x = torch.sigmoid(x)
        return x


class TF_Predictor(nn.Module):
    def __init__(self, d_in, d_main, d_layers, dropout):
        super().__init__()
        self.mlp = MLP.make_baseline(
            d_in=d_main, d_layers=d_layers, dropout=dropout, d_out=1
        )
        self.proj = nn.Linear(d_in, d_main)

    def forward(self, x):
        x = self.proj(x)
        x = self.mlp(x)
        return x


class MLPEncoder(nn.Module):
    def __init__(self, d_in, d_layers, d_out, dropout, dim_t=128, t_in=False):
        super().__init__()
        self.dim_t = dim_t
        self.t_in = t_in
        self.mlp = MLP.make_baseline(
            d_in=dim_t, d_layers=d_layers, dropout=dropout, d_out=d_out
        )

        self.proj = nn.Linear(d_in, dim_t)
        if t_in:
            self.time_embed = nn.Sequential(
                nn.Linear(dim_t, dim_t), nn.SiLU(), nn.Linear(dim_t, dim_t)
            )
        # self.head = nn.Linear(dim_t, self.d_out)

    def forward(self, x, t=None):
        x = self.proj(x)
        if self.t_in and t is not None:
            emb = self.time_embed(timestep_embedding(t, self.dim_t))
            x = x + emb
        x = self.mlp(x)
        return x


# def compute_similarity(c, x, t, cond_encoder, data_encoder):
#     c_features = cond_encoder(c, t)
#     x_features = data_encoder(x, t)

#     c_features = c_features / c_features.norm(dim=1, keepdim=True)
#     x_features = x_features / x_features.norm(dim=1, keepdim=True)

#     logit_scale = self.logit_scale.exp()
#     logits_per_c = logit_scale * c_features @ x_features.t()
#     logits_per_x = logits_per_c.t()

#     return logits_per_c, logits_per_x


class CondScorer(nn.Module):
    def __init__(self, cond_encoder, data_encoder):
        super().__init__()
        self.cond_encoder = cond_encoder
        self.data_encoder = data_encoder

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, c, x, t):
        c_features = self.cond_encoder(c, t)
        x_features = self.data_encoder(x, t)

        c_features = c_features / c_features.norm(dim=1, keepdim=True)
        x_features = x_features / x_features.norm(dim=1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits_per_c = logit_scale * c_features @ x_features.t()
        logits_per_x = logits_per_c.t()

        return logits_per_c, logits_per_x

    def similarity(self, c, x, t):
        with torch.no_grad():
            c_features = self.cond_encoder(c, t)
            x_features = self.data_encoder(x, t)

        c_features = c_features / c_features.norm(dim=1, keepdim=True)
        x_features = x_features / x_features.norm(dim=1, keepdim=True)

        similarity = c_features @ x_features.t()
        return similarity


class MLPDiffusion(nn.Module):
    def __init__(
        self, d_in, d_layers, dropout, dim_t=128, cond_dim=0
    ):  # num_classes=0, is_y_cond=False):
        super().__init__()
        self.dim_t = dim_t
        # self.num_classes = num_classes
        self.is_cond = cond_dim > 0
        self.cond_dim = cond_dim

        # d0 = rtdl_params['d_layers'][0]

        self.mlp = MLP.make_baseline(
            d_in=dim_t, d_layers=d_layers, dropout=dropout, d_out=d_in
        )

        if self.is_cond:
            # self.cond_emd = nn.Linear(self.cond_dim, dim_t)
            self.cond_embed = nn.Linear(self.cond_dim, dim_t)

        self.proj = nn.Linear(d_in, dim_t)
        self.time_embed = nn.Sequential(
            nn.Linear(dim_t, dim_t), nn.SiLU(), nn.Linear(dim_t, dim_t)
        )

    def forward(self, x, timesteps, cond=None):
        emb = self.time_embed(timestep_embedding(timesteps, self.dim_t))
        if self.is_cond and cond is not None:
            emb += F.silu(self.cond_embed(cond))
        x = self.proj(x) + emb
        return self.mlp(x)


class ResNetDiffusion(nn.Module):
    def __init__(
        self, d_in, n_blocks, d_main, d_hidden, dropout, dim_t=128, cond_dim=0
    ):
        super().__init__()
        self.dim_t = dim_t

        self.is_cond = cond_dim > 0
        self.cond_dim = cond_dim

        if self.is_cond:
            self.cond_embed = nn.Linear(self.cond_dim, dim_t)

        self.time_embed = nn.Sequential(
            nn.Linear(dim_t, dim_t), nn.SiLU(), nn.Linear(dim_t, dim_t)
        )

        self.proj = nn.Linear(d_in, dim_t)

        self.resnet = ResNet.make_baseline(
            d_in=dim_t,
            n_blocks=n_blocks,
            d_main=d_main,
            d_hidden=d_hidden,
            dropout_first=dropout,
            dropout_second=dropout,
            d_out=dim_t,
        )

        self.head = nn.Linear(dim_t, d_in)

    def forward(self, x, timesteps, cond=None):
        emb = self.time_embed(timestep_embedding(timesteps, self.dim_t))
        if self.is_cond and cond is not None:
            emb += F.silu(self.cond_embed(cond))
        x = self.proj(x) + emb
        x = self.resnet(x)
        x = self.head(x)
        return x

    # def forward(self, x, t=None):
    #     x = self.proj(x)
    #     if t is not None and self.t_in:
    #         emb = self.time_embed(timestep_embedding(t, self.dim_t))
    #         x = x + emb
    #     x = self.resnet(x)
    #     x = self.head(x)
    #     return x


# class ResNetDiffusion(nn.Module):
#     def __init__(self, d_in, num_classes, rtdl_params, dim_t = 256):
#         super().__init__()
#         self.dim_t = dim_t
#         self.num_classes = num_classes

#         rtdl_params['d_in'] = d_in
#         rtdl_params['d_out'] = d_in
#         rtdl_params['emb_d'] = dim_t
#         self.resnet = ResNet.make_baseline(**rtdl_params)

#         if self.num_classes > 0:
#             self.label_emb = nn.Embedding(self.num_classes, dim_t)

#         self.time_embed = nn.Sequential(
#             nn.Linear(dim_t, dim_t),
#             nn.SiLU(),
#             nn.Linear(dim_t, dim_t)
#         )

#     def forward(self, x, timesteps, y=None):
#         emb = self.time_embed(timestep_embedding(timesteps, self.dim_t))
#         if y is not None and self.num_classes > 0:
#              emb += self.label_emb(y.squeeze())
#         return self.resnet(x, emb)
