import os
import time
from copy import deepcopy

import numpy as np
import pandas as pd
import torch


def update_ema(target_params, source_params, rate=0.999):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.
    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src.detach(), alpha=1 - rate)


class Trainer:
    def __init__(
        self,
        diffusion,
        train_iter,
        lr,
        weight_decay,
        steps,
        save_path=None,
        num_checkpoints=1,
        device=torch.device("cuda:1"),
    ):
        self.diffusion = diffusion
        self.ema_model = deepcopy(self.diffusion._denoise_fn)
        for param in self.ema_model.parameters():
            param.detach_()
        # self.n_dim = n_dim
        self.is_cond = self.diffusion._denoise_fn.is_cond
        if self.is_cond:
            print("Conditional Training!")

        self.train_iter = train_iter
        self.steps = steps
        self.init_lr = lr
        self.optimizer = torch.optim.AdamW(
            self.diffusion.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.device = device
        self.loss_history = pd.DataFrame(columns=["step", "loss"])
        self.log_every = 100
        self.print_every = 500
        self.ema_every = 1000
        self.step_per_check = steps // num_checkpoints
        self.save_path = save_path

        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
            os.makedirs(os.path.join(self.save_path, "checkpoints/"), exist_ok=True)

    def _anneal_lr(self, step):
        frac_done = step / self.steps
        lr = self.init_lr * (1 - frac_done)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _step(self, x, cond=None):
        x = x.to(self.device)
        if self.is_cond and cond is not None:
            cond = cond.to(self.device)

        self.optimizer.zero_grad()

        loss = self.diffusion.compute_loss(x, cond)
        loss.backward(torch.ones_like(loss))

        # for p in self.diffusion.parameters():
        # #     print(p)
        # #     print(p.grad)
        #      p.grad.clamp_(-0.1, 0.1)

        self.optimizer.step()

        return loss

    def run_loop(self):
        step = 0
        # curr_loss = 0.0
        curr_count = 0

        train_start = time.time()

        while step < self.steps:
            if self.is_cond:
                x, cond = next(self.train_iter)
            else:
                x = next(self.train_iter)[0]
                cond = None

            batch_loss = self._step(x, cond)

            # print(f"Step {step}:", (batch_loss_gauss + batch_loss_multi).data)
            self._anneal_lr(step)

            curr_count += len(x)
            # curr_loss += batch_loss.item() * len(x)

            if (step + 1) % self.log_every == 0:
                # loss = curr_loss / curr_count
                batch_loss = batch_loss.mean()

                if (step + 1) % self.print_every == 0:
                    print(
                        f"Step {(step + 1)}/{self.steps} Loss: {batch_loss.item():8.6f}"
                    )

                self.loss_history.loc[len(self.loss_history)] = [
                    step + 1,
                    batch_loss.item(),
                ]

                curr_count = 0
                # curr_loss = 0.0

            update_ema(
                self.ema_model.parameters(), self.diffusion._denoise_fn.parameters()
            )

            step += 1
            if step % self.step_per_check == 0 and self.save_path is not None:
                torch.save(
                    self.diffusion,
                    os.path.join(self.save_path, f"checkpoints/diff_model_{step}.pt"),
                )

        train_end = time.time()

        self.loss_history.loc[len(self.loss_history)] = [step, train_end - train_start]

        if self.save_path is not None:
            self.loss_history.to_csv(
                os.path.join(self.save_path, "loss_history.csv"), index=False
            )
