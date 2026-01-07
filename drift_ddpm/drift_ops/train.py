import sys

sys.path.append("../")
import time

import data_utils as du
import torch
import torch.nn.functional as F
import torch.optim as optim
from ddpm import diffusion, modules, train
from ddpm.corr import pearson
from ddpm.resample import create_named_schedule_sampler
from kornia.enhance import histogram


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
    print(f"training time: {train_end - train_sta}")

    diff_model.to(torch.device("cpu"))
    diff_model.variables_to_device(torch.device("cpu"))
    diff_model.eval()
    torch.save(diff_model, save_path)


def validate_no_nan(x: torch.Tensor):
    if torch.isnan(x).any():
        raise ValueError("nan detected")


def controller_training(
    train_x,
    real_x,
    # cond_x,
    # synthetic_x,
    diffuser,
    save_path,
    cond_save_path,
    device,
    lr=0.001,
    d_hidden=[512, 512],
    steps=10000,
    drop_out=0.0,
    bs=1024,
    # New parameters for better control
    drift_range=(0.05, 0.75),  # Range for expected drift during training
    loss_weight_corr=0.8,      # Weight for correlation loss
    loss_weight_real=0.1,      # Weight for RealMSE loss
):
    """Train an MLP controller."""
    train_x = torch.from_numpy(train_x).float()
    real_x = torch.from_numpy(real_x).float()

    # train_cond_norm = torch.as_tensor(cond_x).float()  ## condition id
    # train_data_norm = torch.as_tensor(synthetic_x).float()  ## real_x

    diffuser.to(device)
    diffuser.variables_to_device(device)
    diffuser.eval()

    # print(f"train_cond_norm.shape: {train_cond_norm.shape}")
    # print(f"train_data_norm.shape: {train_data_norm.shape}")

    # cond_encoder = modules.MLPEncoder(
    #     train_cond_norm.shape[1], d_hidden, 128, 0.0, 128, t_in=False
    # )
    # data_encoder = modules.MLPEncoder(
    #     train_cond_norm.shape[1], d_hidden, 128, 0.0, 128, t_in=True
    # )
    # controller = modules.CondScorer(cond_encoder, data_encoder)
    # controller.to(device)

    model = modules.Drifter(
        d_in=train_x.shape[1],
        d_layers=d_hidden,
        dropout=drop_out,
    )
    ds = [train_x, real_x]
    print(train_x.shape, real_x.shape)

    # ds = [train_x, real_x, train_cond_norm]
    # print(train_x.shape, real_x.shape, train_cond_norm.shape)

    dl = du.prepare_fast_dataloader(ds, batch_size=bs, shuffle=True)
    schedule_sampler = create_named_schedule_sampler("uniform", diffuser.num_timesteps)

    model.train()
    model.to(device)
    diffuser.to(device)
    diffuser.variables_to_device(device)

    jsd = modules.JSD()

    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.00001)
    # opt_cond = optim.AdamW(controller.parameters(), lr=lr, weight_decay=0.0)

    # Pre-compute FULL reference correlation (not batch) for stable target
    # This solves the batch vs full-dataset correlation discrepancy
    full_ref_corr = pearson(real_x).fill_diagonal_(0.0).nan_to_num_(1.0).to(device)
    print(f"Pre-computed full reference correlation matrix: {full_ref_corr.shape}")

    sta = time.time()

    for step in range(steps):
        loss = torch.zeros(1).to(device)

        # [x, real, tcond] = next(dl)
        [x, real] = next(dl)
        x = x.to(device)
        real = real.to(device)
        # tcond = tcond.to(device)

        t, _ = schedule_sampler.sample(1, device)

        expected_drift = torch.FloatTensor(1).uniform_(drift_range[0], drift_range[1]).to(device)
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

        # Use pre-computed FULL reference correlation as target (not batch)
        # This ensures training target matches evaluation target
        p_corr_xc = pearson(xc).fill_diagonal_(0.0).nan_to_num_(1.0)
        p_loss_corr = F.mse_loss(full_ref_corr, p_corr_xc)

        mse_real = F.mse_loss(xc, real)

        # xt_prim_column = xt[:, 0][:, None]
        # logits_c, logits_x = controller(tcond, xt_prim_column, t)
        # cond_loss = F.mse_loss(logits_c, logits_x)

        # print(
        #     f"{expected_drift.item():8.6f} "
        #     f"{actual_drifts[0].item():8.6f} "
        #     f"{abs(expected_drift.item() - actual_drifts[0].item()):8.6f} "
        #     f"{p_loss_corr.item():8.6f} "
        #     f"{mse_real.item():8.6f} "
        #     # f"{cond_loss.item():8.6f}"
        #     # f"{s_loss_corr.item():8.6f} "
        # )

        # total_loss = loss + 0.3 * p_loss_corr + 0.3 * mse_real + 0.1 * cond_loss
        # total_loss = loss + 0.3 * p_loss_corr + 0.3 * mse_real
        total_loss = loss + loss_weight_corr * p_loss_corr + loss_weight_real * mse_real

        opt.zero_grad()
        # opt_cond.zero_grad()

        total_loss.backward()

        opt.step()
        # opt_cond.step()

        set_anneal_lr(opt, lr, step, steps)
        # set_anneal_lr(opt_cond, lr, step, steps)

        if (step + 1) % 500 == 0 or step == 0:
            print(
                f"Step {step + 1}/{steps}: Loss {total_loss.item():.8f} "
                # f"(Drift: {loss.item():.8f}, PCorr: {p_loss_corr.item():.8f})"
                f"(Drift: {loss.item():.8f}, PCorr: {p_loss_corr.item():.8f}, "
                f"RealMSE: {mse_real.item():.8f}, "
                # f"Cond: {cond_loss.item():.8f}"
                # f", SCorr: {s_loss_corr.item():.8f})"
            )

    end = time.time()

    train_elapse = end - sta
    print(f"training time: {train_elapse}")

    model.to(torch.device("cpu"))
    model.eval()
    torch.save(model, save_path)

    # controller.to(torch.device("cpu"))
    # controller.eval()
    # torch.save(controller, cond_save_path)
