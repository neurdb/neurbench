import sys

sys.path.append("../")
import os
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
from kornia.enhance import histogram

import data_utils as du
from ddpm import diffusion, modules, train
from ddpm.corr import pearson, pearson_rel, spearman, spearman_rel
from ddpm.resample import create_named_schedule_sampler


def data_preprocessing(raw_data, save_dir=None):
    data_wrapper = du.DataWrapper()
    data_wrapper.fit(raw_data)

    # if save_dir is not None:
    #     du.save_pickle(
    #         data=data_wrapper, path=os.path.join(save_dir, "data_wrapper.pkl")
    #     )
    return data_wrapper


def set_anneal_lr(opt, init_lr, step, all_steps):
    frac_done = step / all_steps
    lr = init_lr * (1 - frac_done)
    for param_group in opt.param_groups:
        param_group["lr"] = lr


def diffuser_training(
    train_x,
    save_path,
    device,
    d_hidden=[512, 1024, 1024, 512],
    num_timesteps=1000,
    epochs=30000,
    lr=0.0018,
    drop_out=0.0,
    bs=4096,
    lambda_p=1.0,
    lambda_s=1.0,
):
    train_x = torch.from_numpy(train_x).float()
    print("train_x.shape", train_x.shape)
    print("train_x[0]", train_x[0])

    model = modules.MLPDiffusion(train_x.shape[1], d_hidden, drop_out).to(device)
    print("Model initialized")

    diff_model = diffusion.GaussianDiffusion(
        train_x.shape[1],
        model,
        num_timesteps=num_timesteps,
        device=device,
        lambda_p=lambda_p,
        lambda_s=lambda_s,
    ).to(device)
    diff_model.train()
    print("Diffusion initialized")

    ds = [train_x]
    dl = du.prepare_fast_dataloader(ds, batch_size=bs, shuffle=True)

    trainer = train.Trainer(
        diff_model, dl, lr, 1e-2, epochs, save_path=None, device=device
    )
    train_sta = time.time()
    trainer.run_loop()
    train_end = time.time()
    print(f"training time: {train_end-train_sta}")

    diff_model.to(torch.device("cpu"))
    diff_model.variables_to_device(torch.device("cpu"))
    diff_model.eval()
    torch.save(diff_model, save_path)


def validate_no_nan(x: torch.Tensor):
    if torch.isnan(x).any():
        raise ValueError("nan detected")


def controller_training(
    train_x,
    diffuser,
    save_path,
    device,
    lr=0.001,
    d_hidden=[512, 512],
    steps=10000,
    drop_out=0.0,
    bs=1024,
):
    """Train an MLP controller."""
    train_x = torch.from_numpy(train_x).float()

    model = modules.Drifter(
        d_in=train_x.shape[1],
        d_layers=d_hidden,
        dropout=drop_out,
    )
    ds = [train_x]
    dl = du.prepare_fast_dataloader(ds, batch_size=bs, shuffle=True)
    schedule_sampler = create_named_schedule_sampler("uniform", diffuser.num_timesteps)

    model.train()
    model.to(device)
    diffuser.to(device)
    diffuser.variables_to_device(device)

    jsd = modules.JSD()

    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.00001)
    sta = time.time()

    for step in range(steps):
        loss = torch.zeros(1).to(device)

        [x] = next(dl)
        x = x.to(device)

        # t, _ = schedule_sampler.sample(len(y), device)
        t, _ = schedule_sampler.sample(1, device)
        # t = torch.randint(0, diffuser.num_timesteps, (1,)).repeat(len(y)).to(device)

        expected_drift = torch.FloatTensor(1).uniform_(0.05, 0.75).to(device)
        # expected_drift = expected_drift / t[0]
        # if (
        #     expected_drift < 1e-5
        #     or expected_drift > (1 - 1e-5)
        #     or torch.isinf(expected_drift)
        # ):
        #     continue

        xt = diffuser.gaussian_q_sample(x, t)
        hist_xt = histogram(
            xt.T,
            bins=torch.linspace(-10, 10, 20 + 1).to(device),
            bandwidth=torch.tensor(0.9).to(device),
        )

        xc = model(xt, t, expected_drift)
        hist_xc = histogram(
            xc.T,
            bins=torch.linspace(-10, 10, 20 + 1).to(device),
            bandwidth=torch.tensor(0.9).to(device),
        )

        actual_drifts = jsd(hist_xt, hist_xc)
        expected_drifts = expected_drift.repeat(actual_drifts.shape[0])
        loss = F.mse_loss(actual_drifts, expected_drifts).sum()
        if torch.isinf(loss):
            continue

        p_corr_xt = pearson(xt).fill_diagonal_(0.0).nan_to_num_(1.0)
        p_corr_xc = pearson(xc).fill_diagonal_(0.0).nan_to_num_(1.0)
        p_loss_corr = F.mse_loss(p_corr_xt, p_corr_xc)

        # s_corr_xt = spearman(xt).fill_diagonal_(0.0).nan_to_num_(1.0)
        # s_corr_xc = spearman(xc).fill_diagonal_(0.0).nan_to_num_(1.0)
        # s_loss_corr = F.mse_loss(s_corr_xt, s_corr_xc)

        # p_loss_corr = pearson_rel(xt.t(), xc.t()).abs()
        # p_loss_corr = -(1.0 - p_loss_corr) * torch.log(1.0 - p_loss_corr)

        # s_loss_corr = spearman_rel(logits.t(), xt.t()).abs()
        # s_loss_corr = -(1.0 - s_loss_corr) * torch.log(1.0 - s_loss_corr)

        print(
            f"{expected_drift.item():8.6f} "
            f"{actual_drifts[0].item():8.6f} "
            f"{abs(expected_drift.item() - actual_drifts[0].item()):8.6f} "
            f"{p_loss_corr.item():8.6f} "
            # f"{s_loss_corr.item():8.6f} "
        )

        total_loss = loss + 1.0 * p_loss_corr
        # + 1.0 * s_loss_corr

        opt.zero_grad()
        total_loss.backward()
        opt.step()

        set_anneal_lr(opt, lr, step, steps)

        if (step + 1) % 100 == 0 or step == 0:
            print(
                f"Step {step+1}/{steps}: Loss {total_loss.item():.8f} "
                f"(Drift: {loss.item():.8f}, PCorr: {p_loss_corr.item():.8f})"
                # f", SCorr: {s_loss_corr.item():.8f})"
            )

    end = time.time()

    model.to(torch.device("cpu"))
    model.eval()
    torch.save(model, save_path)
