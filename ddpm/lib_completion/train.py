import sys

sys.path.append("../")
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim

import data_utils as du
from ddpm import diffusion, modules, train
from ddpm.resample import create_named_schedule_sampler


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
):
    train_x = torch.from_numpy(train_x).float()
    model = modules.MLPDiffusion(train_x.shape[1], d_hidden, drop_out)
    model.to(device)
    print("Model Initialization")

    diffuser = diffusion.GaussianDiffusion(
        train_x.shape[1], model, device=device, num_timesteps=num_timesteps
    )
    diffuser.to(device)
    diffuser.train()
    print("Diffusion Initialization")

    ds = [train_x]
    dl = du.prepare_fast_dataloader(ds, batch_size=bs, shuffle=True)

    trainer = train.Trainer(
        diffuser, dl, lr, 0.0, epochs, save_path=None, device=device
    )
    train_sta = time.time()
    trainer.run_loop()
    train_end = time.time()
    print(f"training time: {train_end-train_sta}")

    diffuser.to(torch.device("cpu"))
    diffuser.variables_to_device(torch.device("cpu"))
    diffuser.eval()
    torch.save(diffuser, save_path)


def controller_training(
    raw_train_data,
    condition_wrapper,
    synthetic_wrapper,
    diffuser,
    save_path,
    device,
    d_hidden=[128, 128],
    lr=0.0015,
    steps=2000,
    batch_size=4096,
):
    condition_data = [
        raw_train_data[wrapper.raw_columns] for wrapper in condition_wrapper
    ]
    condition_data = pd.concat(condition_data, axis=1)
    synthetic_data = [
        raw_train_data[wrapper.raw_columns] for wrapper in synthetic_wrapper
    ]
    synthetic_data = pd.concat(synthetic_data, axis=1)

    condition_wrapper = du.merge_wrapper(condition_wrapper)
    synthetic_wrapper = du.merge_wrapper(synthetic_wrapper)

    condition_data = condition_wrapper.transform(condition_data)
    synthetic_data = synthetic_wrapper.transform(synthetic_data)

    train_cond_norm = torch.as_tensor(condition_data).float()
    train_data_norm = torch.as_tensor(synthetic_data).float()

    diffuser.to(device)
    diffuser.variables_to_device(device)
    diffuser.eval()

    cond_encoder = modules.MLPEncoder(
        train_cond_norm.shape[1], d_hidden, 128, 0.0, 128, t_in=False
    )
    data_encoder = modules.MLPEncoder(
        train_data_norm.shape[1], d_hidden, 128, 0.0, 128, t_in=True
    )
    controller = modules.CondScorer(cond_encoder, data_encoder)
    controller.to(device)

    ds = [train_cond_norm, train_data_norm]
    dl = du.prepare_fast_dataloader(ds, batch_size=batch_size, shuffle=True)

    schedule_sampler = create_named_schedule_sampler("uniform", diffuser.num_timesteps)

    opt = optim.AdamW(controller.parameters(), lr=lr, weight_decay=0.0)

    sta = time.time()
    losses = []
    for step in range(steps):
        c, x = next(dl)
        c = c.to(device)
        x = x.to(device)

        t, _ = schedule_sampler.sample(len(x), device)
        x_t = diffuser.gaussian_q_sample(x, t)
        # c_t = diffuser.gaussian_q_sample(c, t)

        logits_c, logits_x = controller(c, x_t, t)
        labels = np.arange(logits_c.shape[0])
        labels = torch.as_tensor(labels).to(device)

        loss_1 = F.cross_entropy(logits_c, labels)
        loss_2 = F.cross_entropy(logits_x, labels)
        loss = (loss_1 + loss_2) / 2

        opt.zero_grad()
        loss.backward()
        opt.step()

        set_anneal_lr(opt, lr, step, steps)

        if (step + 1) % 1000 == 0 or step == 0:
            print(
                f"Step {step+1}/{steps} : Loss {loss.data}, loss1 {loss_1.data}, loss2 {loss_2.data}"
            )
            losses.append(loss.detach().cpu().numpy())
    end = time.time()
    train_elapse = end - sta
    print(f"training time: {train_elapse}")

    controller.eval()
    controller.to(torch.device("cpu"))
    torch.save(controller, save_path)


def diffuser_tuning(
    train_x, diffuser, save_path, device, epochs, lr=0.0018, batch_size=4096
):
    train_x = torch.from_numpy(train_x).float()
    ds = [train_x]
    dl = du.prepare_fast_dataloader(ds, batch_size=batch_size, shuffle=True)

    diffuser.to(device)
    diffuser.train()
    diffuser.variables_to_device(device)

    trainer = train.Trainer(
        diffuser, dl, lr, 0.0, epochs, save_path=None, device=device
    )
    train_sta = time.time()
    trainer.run_loop()
    train_end = time.time()
    print(f"tuning time: {train_end-train_sta}")

    diffuser.to(torch.device("cpu"))
    diffuser.variables_to_device(device)
    diffuser.eval()
    torch.save(diffuser, save_path)
