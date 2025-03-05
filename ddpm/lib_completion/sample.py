import sys

sys.path.append("../")

import numpy as np
import pandas as pd
import torch

import data_utils as du


def get_cond_fn(controller, scale_factor):
    def cond_fn(c, x, t):
        x = x.float()
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)

            x_features = controller.data_encoder(x_in, t)
            c_features = controller.cond_encoder(c, t)

            c_features = c_features / c_features.norm(dim=1, keepdim=True)
            x_features = x_features / x_features.norm(dim=1, keepdim=True)

            logits = torch.einsum("bi,bi->b", [c_features, x_features])
            gradients = torch.autograd.grad(torch.sum(logits), x_in)[0] * scale_factor
            return gradients

    return cond_fn


def table_condition_sample(
    condition_wrapper,
    synthetic_wrapper,
    condition_data,
    diffuser,
    controller,
    device,
    scale_factor=25,
    bs=100000,
):
    condition_wrapper = du.merge_wrapper(condition_wrapper)
    synthetic_wrapper = du.merge_wrapper(synthetic_wrapper)
    condition_data = condition_data[condition_wrapper.raw_columns]

    test_cond_norm = condition_wrapper.transform(condition_data)
    test_cond_norm = torch.as_tensor(test_cond_norm).float()

    diffuser.eval()
    diffuser.to(device)
    diffuser.variables_to_device(device)

    controller.eval()
    controller.to(device)

    cond_fn = get_cond_fn(controller, scale_factor)

    sample_index = np.arange(len(test_cond_norm))
    sample_data = np.zeros([len(test_cond_norm), synthetic_wrapper.raw_dim])

    while len(sample_index) > 0:
        # print("sample_index:", sample_index)
        cond_input = test_cond_norm[sample_index, :]
        control_tools = (cond_input, cond_fn)

        sample = diffuser.batch_sample(
            len(cond_input), batch_size=bs, control_tools=control_tools
        )
        # sample = diffuser.sample(len(cond_input), control_tools=control_tools, clip_denoised=True, control_t=diffuser.num_timesteps)
        sample = sample.cpu().numpy()
        sample = synthetic_wrapper.ReverseToOrdi(sample)

        allow_index, reject_index = synthetic_wrapper.RejectSample(sample)
        sample_allow_index = sample_index[allow_index] if len(allow_index) > 0 else []
        sample_reject_index = (
            sample_index[reject_index] if len(reject_index) > 0 else []
        )
        # print(sample_allow_index, allow_index, sample_reject_index)

        if len(sample_allow_index) > 0:
            sample_data[sample_allow_index, :] = sample[allow_index, :]

        sample_index = sample_reject_index

    sample_data = synthetic_wrapper.ReverseToCat(sample_data)
    sample_data = pd.DataFrame(sample_data, columns=synthetic_wrapper.columns)
    sample_data = synthetic_wrapper.ReOrderColumns(sample_data)
    return condition_data, sample_data
