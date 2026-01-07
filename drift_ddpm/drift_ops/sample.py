import sys

from ddpm import modules

sys.path.append("../")

import torch


def get_cond_fn(
    controller: modules.Drifter,
    controller_cond: modules.CondScorer,
    scale_factor: float,
    drift: float,
):
    print("[get_cond_fn] drift: ", drift)

    def cond_fn(c, x, t):
        # c = c.to(x.device)

        x = x.float()
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            epsilon_t = controller(x_in, t, drift)
            gradients = torch.autograd.grad(epsilon_t.sum(), x_in)[0] * scale_factor

            # x_features = controller_cond.data_encoder(
            #     x_in[:, 0][:, None],
            #     torch.tensor(t[0]).expand(1, 1).to(x.device),
            # ).squeeze(0)

            # rand_columns = torch.randperm(c.shape[0])[: x_in.shape[0]]
            # c_sample = c[rand_columns].to(x.device)

            # c_features = controller_cond.cond_encoder(
            #     c_sample, torch.tensor(t[0]).expand(1, 1).to(x.device)
            # )

            # c_features = c_features / c_features.norm(dim=1, keepdim=True)
            # x_features = x_features / x_features.norm(dim=1, keepdim=True)

            # logits = torch.einsum("bi,bi->b", [c_features, x_features])
            # gradients_cond = (
            #     torch.autograd.grad(torch.sum(logits), x_in)[0] * scale_factor
            # )

            # return gradients + gradients_cond
            return gradients

    return cond_fn


def oversampling(
    n_samples,
    controller,
    diffuser,
    controller_cond,
    cond_x,
    synthetic_x,
    device,
    drift=0.3,
    scale_factor=8.0,
    sample_steps=None,
):
    diffuser.to(device)
    diffuser.variables_to_device(device)

    controller.to(device)
    if controller_cond:
        controller_cond.to(device)

    cond_fn = get_cond_fn(controller, controller_cond, scale_factor, drift)
    # cond = torch.zeros(n_samples, 1)

    # cond_x = torch.as_tensor(cond_x).float()
    # synthetic_x = torch.as_tensor(synthetic_x).float()
    # print("[DEBUG] cond_x.shape:", cond_x.shape)

    # samples = diffuser.sample(n_samples, control_tools=[(cond_x, synthetic_x), cond_fn])
    # samples = diffuser.sample(n_samples, control_tools=[cond_x, cond_fn])
    samples = diffuser.sample(n_samples, control_tools=[None, cond_fn], sample_steps=sample_steps)

    return samples
