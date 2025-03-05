import sys

import torch.nn.functional as F
from kornia.enhance import histogram

from ddpm import modules

sys.path.append("../")

import torch


def get_cond_fn(controller: modules.Drifter, scale_factor: float, drift: float):
    print("[get_cond_fn] drift: ", drift)

    def cond_fn(c, x, t):
        x = x.float()
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            epsilon_t = controller(x_in, t, drift)
            gradients = torch.autograd.grad(epsilon_t.sum(), x_in)[0] * scale_factor
            # gradients *= (drift - 0.5)
            return gradients

    return cond_fn


def oversampling(
    n_samples,
    controller,
    diffuser,
    device,
    drift=0.3,
    scale_factor=8.0,
):
    diffuser.to(device)
    diffuser.variables_to_device(device)

    controller.to(device)

    cond_fn = get_cond_fn(controller, scale_factor, drift)
    cond = torch.zeros(n_samples, 1)

    samples = diffuser.sample(n_samples, control_tools=[cond, cond_fn])

    return samples
