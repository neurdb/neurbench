import os
import numpy as np
from datetime import datetime

from stable_baselines3 import DDPG
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor

import config
from rl_env import ALEXDistEnv


class TuningCallback(BaseCallback):

    def __init__(self, verbose=1):
        super().__init__(verbose)
        self.best_throughput = 0
        self.best_params = None
        self.baseline = None

    def _on_step(self) -> bool:
        info = self.locals.get("infos", [{}])[0]

        throughput = info.get("throughput", 0)
        best = info.get("best_throughput", 0)
        params = info.get("params", [0] * 8)
        crashed = info.get("crashed", False)
        dataset = info.get("dataset", "?")
        ep_step = info.get("episode_step", 0)

        if self.baseline is None:
            self.baseline = info.get("baseline_throughput", 1)

        if best > self.best_throughput:
            self.best_throughput = best
            self.best_params = params

        best_improvement = ((self.best_throughput - self.baseline) / self.baseline * 100
                            if self.baseline > 0 else 0)

        ds_name = os.path.basename(dataset)

        if crashed:
            print(f"Step {self.num_timesteps} [{ds_name} ep{ep_step}]: "
                  f"node={params[0]:.0f}MB d=({params[1]:.2f},{params[2]:.2f},{params[3]:.2f}) "
                  f"w=(exp={params[4]:.1f},sh={params[5]:.2f},nl={params[6]:.1f},ms={params[7]:.1e}) "
                  f"-> CRASHED | best: {self.best_throughput:.0f} ({best_improvement:+.1f}%)")
        else:
            improvement = (throughput - self.baseline) / self.baseline * 100 if self.baseline > 0 else 0
            print(f"Step {self.num_timesteps} [{ds_name} ep{ep_step}]: "
                  f"node={params[0]:.0f}MB d=({params[1]:.2f},{params[2]:.2f},{params[3]:.2f}) "
                  f"w=(exp={params[4]:.1f},sh={params[5]:.2f},nl={params[6]:.1f},ms={params[7]:.1e}) "
                  f"-> {throughput:.0f} ({improvement:+.1f}%) | best: {self.best_throughput:.0f} ({best_improvement:+.1f}%)")

        return True


def train(
    total_timesteps=None,
    max_steps=10,
    datasets=None,
):
    total_timesteps = total_timesteps or config.TOTAL_TIMESTEPS

    print("=" * 60)
    print("a-li: DDPG")
    print("=" * 60)
    print(f"datasets: {datasets or config.DATASETS}")
    print(f"init_count: {config.INIT_COUNT}")
    print(f"op_count: {config.OP_COUNT}")
    print(f"read_ratio: {config.READ_RATIO}")
    print(f"total_timesteps: {total_timesteps}")
    print(f"max_steps/episode: {max_steps}")
    print(f"buffer_size: {config.BUFFER_SIZE}")
    print(f"batch_size: {config.BATCH_SIZE}")
    print(f"learning_rate: {config.LEARNING_RATE}")
    print(f"noise_sigma: {config.NOISE_SIGMA}")
    print()

    env = ALEXDistEnv(
        datasets=datasets,
        max_steps=max_steps,
    )
    env = Monitor(env)

    n_actions = env.action_space.shape[0]
    action_noise = OrnsteinUhlenbeckActionNoise(
        mean=np.zeros(n_actions),
        sigma=config.NOISE_SIGMA * np.ones(n_actions),
    )

    model = DDPG(
        "MlpPolicy",
        env,
        learning_rate=config.LEARNING_RATE,
        buffer_size=config.BUFFER_SIZE,
        batch_size=config.BATCH_SIZE,
        tau=config.TAU,
        gamma=config.GAMMA,
        learning_starts=config.LEARNING_STARTS,
        action_noise=action_noise,
        verbose=1,
        tensorboard_log=config.LOG_DIR,
    )

    tuning_callback = TuningCallback()

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    checkpoint_callback = CheckpointCallback(
        save_freq=500,
        save_path=config.CHECKPOINT_DIR,
        name_prefix="ddpg_dist",
    )

    callback = CallbackList([tuning_callback, checkpoint_callback])

    print("Training...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"ddpg_dist_{timestamp}"
    model.save(model_path)
    print(f"\nModel saved: {model_path}")

    print("\n" + "=" * 60)
    print("Training done!")
    print("=" * 60)

    best_config = env.unwrapped.get_best_config()
    if best_config:
        print(f"\nBest config:")
        print(f"  node_size: {best_config['node_size_mb']:.0f} MB")
        print(f"  init_density: {best_config['init_density']:.2f}")
        print(f"  max_density: {best_config['max_density']:.2f}")
        print(f"  min_density: {best_config['min_density']:.2f}")
        print(f"  exp_search_w: {best_config['exp_search_iterations_weight']:.1f}")
        print(f"  shifts_w: {best_config['shifts_weight']:.2f}")
        print(f"  node_lookups_w: {best_config['node_lookups_weight']:.1f}")
        print(f"  model_size_w: {best_config['model_size_weight']:.2e}")
        print(f"  throughput: {best_config['throughput']:.0f} ops/s ({best_config['improvement']:+.2f}%)")

    return model, env


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=10)
    parser.add_argument("--datasets", nargs="+", default=None)
    args = parser.parse_args()

    train(
        total_timesteps=args.timesteps,
        max_steps=args.max_steps,
        datasets=args.datasets,
    )
