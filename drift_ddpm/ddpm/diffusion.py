"""
Based on https://github.com/openai/guided-diffusion/blob/main/guided_diffusion
and https://github.com/ehoogeboom/multinomial_diffusion
"""

import math

import numpy as np
import torch

from .corr import pearson, spearman, spearman_rel
from .utils import *

"""
Based in part on: https://github.com/lucidrains/denoising-diffusion-pytorch/blob/5989f4c77eafcdc6be0fb4739f0f277a6dd7f7d8/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py#L281
"""
eps = 1e-8


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.
    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].
    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class GaussianDiffusion(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        denoise_fn,
        num_timesteps=1000,
        gaussian_loss_type="mse",
        gaussian_parametrization="eps",
        parametrization="x0",
        scheduler="cosine",
        device=torch.device("cpu"),
        lambda_p=1.0,
        lambda_s=1.0,
    ):
        super(GaussianDiffusion, self).__init__()
        assert parametrization in ("x0", "direct")

        self.input_dim = input_dim

        self._lambda_p = lambda_p
        self._lambda_s = lambda_s

        print(
            f"[{self.__class__.__name__}] lambda_p: {self._lambda_p}, lambda_s: {self._lambda_s}"
        )

        self._denoise_fn = denoise_fn
        self.gaussian_loss_type = gaussian_loss_type
        self.gaussian_parametrization = gaussian_parametrization
        self.num_timesteps = num_timesteps
        self.parametrization = parametrization
        self.scheduler = scheduler

        alphas = 1.0 - get_named_beta_schedule(scheduler, num_timesteps)
        alphas = torch.tensor(alphas.astype("float64"))
        betas = 1.0 - alphas

        log_alpha = np.log(alphas)
        log_cumprod_alpha = np.cumsum(log_alpha)

        log_1_min_alpha = log_1_min_a(log_alpha)
        log_1_min_cumprod_alpha = log_1_min_a(log_cumprod_alpha)

        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.tensor(np.append(1.0, alphas_cumprod[:-1]))
        alphas_cumprod_next = torch.tensor(np.append(alphas_cumprod[1:], 0.0))
        sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - alphas_cumprod)
        sqrt_recip_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod)
        sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod - 1)

        # Gaussian diffusion

        self.posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.posterior_log_variance_clipped = (
            torch.from_numpy(
                np.log(
                    np.append(self.posterior_variance[1], self.posterior_variance[1:])
                )
            )
            .float()
            .to(device)
        )
        self.posterior_mean_coef1 = (
            (betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
            .float()
            .to(device)
        )
        self.posterior_mean_coef2 = (
            (
                (1.0 - alphas_cumprod_prev)
                * np.sqrt(alphas.numpy())
                / (1.0 - alphas_cumprod)
            )
            .float()
            .to(device)
        )

        assert log_add_exp(log_alpha, log_1_min_alpha).abs().sum().item() < 1.0e-5
        assert (
            log_add_exp(log_cumprod_alpha, log_1_min_cumprod_alpha).abs().sum().item()
            < 1e-5
        )
        assert (np.cumsum(log_alpha) - log_cumprod_alpha).abs().sum().item() < 1.0e-5

        # Convert to float32 and register buffers.
        self.register_buffer("alphas", alphas.float().to(device))
        self.register_buffer("log_alpha", log_alpha.float().to(device))
        self.register_buffer("log_1_min_alpha", log_1_min_alpha.float().to(device))
        self.register_buffer(
            "log_1_min_cumprod_alpha", log_1_min_cumprod_alpha.float().to(device)
        )
        self.register_buffer("log_cumprod_alpha", log_cumprod_alpha.float().to(device))
        self.register_buffer("alphas_cumprod", alphas_cumprod.float().to(device))
        self.register_buffer(
            "alphas_cumprod_prev", alphas_cumprod_prev.float().to(device)
        )
        self.register_buffer(
            "alphas_cumprod_next", alphas_cumprod_next.float().to(device)
        )
        self.register_buffer(
            "sqrt_alphas_cumprod", sqrt_alphas_cumprod.float().to(device)
        )
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            sqrt_one_minus_alphas_cumprod.float().to(device),
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", sqrt_recip_alphas_cumprod.float().to(device)
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod",
            sqrt_recipm1_alphas_cumprod.float().to(device),
        )

        self.register_buffer("Lt_history", torch.zeros(num_timesteps))
        self.register_buffer("Lt_count", torch.zeros(num_timesteps))

    def variables_to_device(self, device):
        self.alphas = self.alphas.to(device)
        self.log_alpha = self.log_alpha.to(device)
        self.log_1_min_alpha = self.log_1_min_alpha.to(device)
        self.log_1_min_cumprod_alpha = self.log_1_min_cumprod_alpha.to(device)
        self.log_cumprod_alpha = self.log_cumprod_alpha.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        self.alphas_cumprod_next = self.alphas_cumprod_next.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(
            device
        )
        self.sqrt_recip_alphas_cumprod = self.sqrt_recip_alphas_cumprod.to(device)
        self.sqrt_recipm1_alphas_cumprod = self.sqrt_recipm1_alphas_cumprod.to(device)
        self.posterior_log_variance_clipped = self.posterior_log_variance_clipped.to(
            device
        )
        self.posterior_mean_coef1 = self.posterior_mean_coef1.to(device)
        self.posterior_mean_coef2 = self.posterior_mean_coef2.to(device)

    # Gaussian part
    def gaussian_q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_1_min_cumprod_alpha, t, x_start.shape)
        return mean, variance, log_variance

    def gaussian_q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def gaussian_q_posterior_mean_variance(self, x_start, x_t, t):
        assert x_start.shape == x_t.shape
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def gaussian_p_mean_variance(
        self,
        model_output,
        x,
        t,
        clip_denoised=False,
        denoised_fn=None,
        model_kwargs=None,
    ):
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        assert t.shape == (B,)

        model_variance = torch.cat(
            [
                self.posterior_variance[1].unsqueeze(0).to(x.device),
                (1.0 - self.alphas)[1:],
            ],
            dim=0,
        )
        # model_variance = self.posterior_variance.to(x.device)
        model_log_variance = torch.log(model_variance)

        model_variance = extract(model_variance, t, x.shape)
        model_log_variance = extract(model_log_variance, t, x.shape)

        if self.gaussian_parametrization == "eps":
            pred_xstart = self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
        elif self.gaussian_parametrization == "x0":
            pred_xstart = model_output
        else:
            raise NotImplementedError

        if clip_denoised:
            pred_xstart.clamp_(0.0, 1.0)

        model_mean, _, _ = self.gaussian_q_posterior_mean_variance(
            x_start=pred_xstart, x_t=x, t=t
        )

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        ), f"{model_mean.shape}, {model_log_variance.shape}, {pred_xstart.shape}, {x.shape}"

        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def _vb_terms_bpd(
        self, model_output, x_start, x_t, t, clip_denoised=False, model_kwargs=None
    ):
        (
            true_mean,
            _,
            true_log_variance_clipped,
        ) = self.gaussian_q_posterior_mean_variance(x_start=x_start, x_t=x_t, t=t)
        out = self.gaussian_p_mean_variance(
            model_output, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = torch.where((t == 0), decoder_nll, kl)
        return {
            "output": output,
            "pred_xstart": out["pred_xstart"],
            "out_mean": out["mean"],
            "true_mean": true_mean,
        }

    def _prior_gaussian(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.

        This term can't be optimized, as it only depends on the encoder.

        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = torch.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.gaussian_q_mean_variance(x_start, t)
        kl_prior = normal_kl(
            mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0
        )
        return mean_flat(kl_prior) / np.log(2.0)

    def _gaussian_loss(self, model_out, x_start, x_t, t, noise, model_kwargs=None):
        if model_kwargs is None:
            model_kwargs = {}

        terms = {}
        if self.gaussian_loss_type == "mse":
            terms["loss"] = mean_flat((noise - model_out) ** 2)
        elif self.gaussian_loss_type == "kl":
            terms["loss"] = self._vb_terms_bpd(
                model_output=model_out,
                x_start=x_start,
                x_t=x_t,
                t=t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )["output"]

        return terms["loss"]

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - pred_xstart
        ) / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def condition_mean(self, cond_fn, cond, p_mean_var, x, t, model_kwargs=None):
        gradient = cond_fn(cond, x, t)
        new_mean = (
            p_mean_var["mean"].float() + p_mean_var["variance"] * gradient.float()
        )
        return new_mean

    def gaussian_p_sample(
        self,
        model_out,
        x,
        t,
        clip_denoised=False,
        denoised_fn=None,
        model_kwargs=None,
        control_tools=None,
    ):
        out = self.gaussian_p_mean_variance(
            model_out,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = torch.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0

        if control_tools is not None:
            cond, cond_fn = control_tools
            if exists(cond):
                cond = cond.to(x.device)

            out["mean"] = self.condition_mean(
                cond_fn, cond, out, x, t, model_kwargs=model_kwargs
            )

        sample = (
            out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
        )
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def sample_time(self, b, device, method="uniform"):
        if method == "importance":
            if not (self.Lt_count > 10).all():
                return self.sample_time(b, device, method="uniform")

            Lt_sqrt = torch.sqrt(self.Lt_history + 1e-10) + 0.0001
            Lt_sqrt[0] = Lt_sqrt[1]  # Overwrite decoder term with L1.
            pt_all = (Lt_sqrt / Lt_sqrt.sum()).to(device)

            t = torch.multinomial(pt_all, num_samples=b, replacement=True).to(device)

            pt = pt_all.gather(dim=0, index=t)

            return t, pt

        elif method == "uniform":
            t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

            pt = torch.ones_like(t).float() / self.num_timesteps
            return t, pt
        else:
            raise ValueError

    def compute_loss(self, x, cond=None):
        b = x.shape[0]
        device = x.device
        t, pt = self.sample_time(b, device, "uniform")

        noise = torch.randn_like(x)
        x_t = self.gaussian_q_sample(x, t, noise=noise)

        model_out = self._denoise_fn(x_t, t)

        loss_gauss = self._gaussian_loss(model_out, x, x_t, t, noise)

        return loss_gauss

    @torch.no_grad()
    def mixed_elbo(self, x0):
        b = x0.size(0)
        device = x0.device

        gaussian_loss = []
        xstart_mse = []
        mse = []
        mu_mse = []
        out_mean = []
        true_mean = []
        for t in range(self.num_timesteps):
            t_array = (torch.ones(b, device=device) * t).long()
            noise = torch.randn_like(x0)

            x_t = self.gaussian_q_sample(x_start=x0, t=t_array, noise=noise)

            model_out = self._denoise_fn(
                torch.cat(x_t, dim=1),
                t_array,
            )

            out = self._vb_terms_bpd(
                model_out, x_start=x0, x_t=x_t, t=t_array, clip_denoised=False
            )

            gaussian_loss.append(out["output"])
            xstart_mse.append(mean_flat((out["pred_xstart"] - x0) ** 2))
            # mu_mse.append(mean_flat(out["mean_mse"]))
            out_mean.append(mean_flat(out["out_mean"]))
            true_mean.append(mean_flat(out["true_mean"]))

            eps = self._predict_eps_from_xstart(x_t, t_array, out["pred_xstart"])
            mse.append(mean_flat((eps - noise) ** 2))

        gaussian_loss = torch.stack(gaussian_loss, dim=1)
        xstart_mse = torch.stack(xstart_mse, dim=1)
        mse = torch.stack(mse, dim=1)
        # mu_mse = torch.stack(mu_mse, dim=1)
        out_mean = torch.stack(out_mean, dim=1)
        true_mean = torch.stack(true_mean, dim=1)

        prior_gauss = self._prior_gaussian(x0)

        total_gauss = gaussian_loss.sum(dim=1) + prior_gauss
        return {
            "total_gaussian": total_gauss,
            "losses_gaussian": gaussian_loss,
            "xstart_mse": xstart_mse,
            "mse": mse,
            # "mu_mse": mu_mse
            "out_mean": out_mean,
            "true_mean": true_mean,
        }

    @torch.no_grad()
    def gaussian_ddim_step(
        self, model_out, x, t, clip_denoised=False, denoised_fn=None, eta=0.0
    ):
        out = self.gaussian_p_mean_variance(
            model_out,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=None,
        )

        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])

        alpha_bar = extract(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = extract(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
            eta
            * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
        )

        noise = torch.randn_like(x)
        mean_pred = (
            out["pred_xstart"] * torch.sqrt(alpha_bar_prev)
            + torch.sqrt(1 - alpha_bar_prev - sigma**2) * eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise

        return sample

    @torch.no_grad()
    def gaussian_ddim_sample(self, noise, T, eta=0.0):
        x = noise
        b = x.shape[0]
        device = x.device
        for t in reversed(range(T)):
            print(f"Sample timestep {t:4d}", end="\r")
            t_array = (torch.ones(b, device=device) * t).long()
            out_num = self._denoise_fn(x, t_array)
            x = self.gaussian_ddim_step(out_num, x, t_array)
        print()
        return x

    @torch.no_grad()
    def gaussian_ddim_reverse_step(self, model_out, x, t, clip_denoised=False, eta=0.0):
        assert eta == 0.0, "Eta must be zero."
        out = self.gaussian_p_mean_variance(
            model_out,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=None,
            model_kwargs=None,
        )

        eps = (
            extract(self.sqrt_recip_alphas_cumprod, t, x.shape) * x - out["pred_xstart"]
        ) / extract(self.sqrt_recipm1_alphas_cumprod, t, x.shape)
        alpha_bar_next = extract(self.alphas_cumprod_next, t, x.shape)

        mean_pred = (
            out["pred_xstart"] * torch.sqrt(alpha_bar_next)
            + torch.sqrt(1 - alpha_bar_next) * eps
        )

        return mean_pred

    @torch.no_grad()
    def gaussian_ddim_reverse_sample(
        self,
        x,
        T,
    ):
        b = x.shape[0]
        device = x.device
        for t in range(T):
            print(f"Reverse timestep {t:4d}", end="\r")
            t_array = (torch.ones(b, device=device) * t).long()
            out_num = self._denoise_fn(x, t_array)
            x = self.gaussian_ddim_reverse_step(out_num, x, t_array, eta=0.0)
        print()

        return x

    @torch.no_grad()
    def sample_ddim(self, num_samples, given_z_norm=None, given_log_z=None, cond=None):
        b = num_samples
        device = self.log_alpha.device
        has_cat = self.num_classes[0] != 0

        if exists(given_z_norm):
            z_norm = given_z_norm
        else:
            z_norm = torch.randn((b, self.input_dim), device=device)

        if exists(given_log_z):
            log_z = given_log_z
        else:
            log_z = torch.zeros((b, 0), device=device).float()
            if has_cat:
                uniform_logits = torch.zeros(
                    (b, len(self.num_classes_expanded)), device=device
                )
                log_z = self.log_sample_categorical(uniform_logits)

        for i in reversed(range(0, self.num_timesteps)):
            print(f"Sample timestep {i:4d}", end="\r")
            t = torch.full((b,), i, device=device, dtype=torch.long)
            model_out = self._denoise_fn(
                torch.cat([z_norm, log_z], dim=1).float(),
                t,
            )
            model_out_num = model_out[:, : self.input_dim]
            model_out_cat = model_out[:, self.input_dim :]
            z_norm = self.gaussian_ddim_step(
                model_out_num, z_norm, t, clip_denoised=False
            )
            if has_cat:
                log_z = self.multinomial_ddim_step(model_out_cat, log_z, t)

        print()
        z_ohe = torch.exp(log_z).round()
        z_cat = log_z
        if has_cat:
            z_cat = ohe_to_categories(z_ohe, self.num_classes)
        sample = torch.cat([z_norm, z_cat], dim=1).cpu()
        return sample

    @torch.no_grad()
    def sample(self, num_samples, clip_denoised=False, control_tools=None):
        b = num_samples
        device = self.log_alpha.device
        z_norm = torch.randn((b, self.input_dim), device=device)

        for i in reversed(range(0, self.num_timesteps)):
            print(f"Sample timestep {i:4d}", end="\r")
            t = torch.full((b,), i, device=device, dtype=torch.long)

            model_out = self._denoise_fn(z_norm.float(), t)

            z_norm = self.gaussian_p_sample(
                model_out,
                z_norm,
                t,
                clip_denoised=clip_denoised,
                control_tools=control_tools,
            )["sample"]

        print()
        sample = z_norm
        return sample

    def batch_sample(
        self, num_samples, batch_size, clip_denoised=False, control_tools=None
    ):
        sample_fn = self.sample

        if control_tools is not None:
            cond, cond_fn = control_tools

        bs = batch_size
        no_batches = math.ceil(num_samples / bs)

        all_samples = []
        for batch_i in range(no_batches):
            batch_start = batch_i * bs
            batch_end = min((batch_i + 1) * bs, num_samples)

            if control_tools is not None:
                batch_cond = cond[batch_start:batch_end, :]
                batch_control_tools = (batch_cond, cond_fn)
            else:
                batch_control_tools = control_tools

            sample = sample_fn(
                batch_end - batch_start, clip_denoised, batch_control_tools
            )
            all_samples.append(sample)

        all_samples = torch.cat(all_samples, dim=0)

        return all_samples

    # def sample_all(self, num_samples, batch_size, ddim=False):
    #     if ddim:
    #         print('Sample using DDIM.')
    #         sample_fn = self.sample_ddim
    #     else:
    #         sample_fn = self.sample

    #     b = batch_size

    #     all_samples = []
    #     num_generated = 0
    #     while num_generated < num_samples:
    #         sample = sample_fn(b)
    #         mask_nan = torch.any(sample.isnan(), dim=1)
    #         sample = sample[~mask_nan]

    #         all_samples.append(sample)
    #         if sample.shape[0] != b:
    #             raise FoundNANsError
    #         num_generated += sample.shape[0]

    #     x_gen = torch.cat(all_samples, dim=0)[:num_samples]

    #     return x_gen
