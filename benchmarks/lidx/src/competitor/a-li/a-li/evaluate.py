import os
import sys
import argparse
import numpy as np

from stable_baselines3 import DDPG
from stable_baselines3.common.monitor import Monitor

import config
from rl_env import ALEXDistEnv


def evaluate_single(model_path: str, dataset: str, n_episodes: int = 3, max_steps: int = 10):
    env = ALEXDistEnv(datasets=[dataset], max_steps=max_steps)
    model = DDPG.load(model_path)

    all_best = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        print(f"\n--- Episode {ep+1} ---")

        for step in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            p = info["params"]
            print(f"  Step {step+1}: node={p[0]:.0f}MB d=({p[1]:.2f},{p[2]:.2f},{p[3]:.2f}) "
                  f"-> {info['throughput']:.0f} ({info['improvement']:+.1f}%) r={reward:.4f}")

            if terminated or truncated:
                break

        best = env.get_best_config()
        all_best.append(best)
        print(f"  Best: {best['throughput']:.0f} ops/s ({best['improvement']:+.1f}%)")

    throughputs = [b["throughput"] for b in all_best]
    improvements = [b["improvement"] for b in all_best]
    print(f"\nSummary ({n_episodes} episodes):")
    print(f"  avg throughput: {np.mean(throughputs):.0f} ops/s")
    print(f"  avg improvement: {np.mean(improvements):+.1f}%")
    print(f"  best improvement: {np.max(improvements):+.1f}%")

    return all_best


def evaluate_cross(model_path: str, datasets: list = None, max_steps: int = 10):
    datasets = datasets or config.DATASETS

    model = DDPG.load(model_path)
    results = {}

    for dataset in datasets:
        ds_name = os.path.basename(dataset)
        print(f"\n{'='*40}")
        print(f"Dataset: {ds_name}")
        print(f"{'='*40}")

        env = ALEXDistEnv(datasets=[dataset], max_steps=max_steps)
        obs, _ = env.reset()

        for step in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            print(f"  Step {step+1}: {info['throughput']:.0f} ({info['improvement']:+.1f}%)")

            if terminated or truncated:
                break

        best = env.get_best_config()
        results[ds_name] = best
        print(f"  Best: {best['throughput']:.0f} ops/s ({best['improvement']:+.1f}%)")

    print(f"\n{'='*60}")
    print("Cross-dataset summary:")
    print(f"{'='*60}")
    print(f"{'Dataset':<15} {'Baseline':>12} {'Best':>12} {'Improv':>10}")
    print("-" * 50)
    for ds_name, best in results.items():
        print(f"{ds_name:<15} {best['baseline_throughput']:>12.0f} {best['throughput']:>12.0f} {best['improvement']:>+9.1f}%")

    return results


def evaluate_compare(model_path: str, dataset: str, max_steps: int = 10):
    env = ALEXDistEnv(datasets=[dataset], max_steps=max_steps)
    model = DDPG.load(model_path)

    obs, _ = env.reset()
    baseline = env.baseline_throughput

    for step in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    ddpg_best = env.get_best_config()

    print(f"\n{'Config':<20} {'Throughput':>15} {'vs Baseline':>12}")
    print("-" * 50)
    print(f"{'Default':<20} {baseline:>15.0f} {'---':>12}")
    print(f"{'DDPG Best':<20} {ddpg_best['throughput']:>15.0f} {ddpg_best['improvement']:>+11.1f}%")

    print(f"\nDDPG best params:")
    print(f"  node_size: {ddpg_best['node_size_mb']:.0f} MB")
    print(f"  init_density: {ddpg_best['init_density']:.2f}")
    print(f"  max_density: {ddpg_best['max_density']:.2f}")
    print(f"  min_density: {ddpg_best['min_density']:.2f}")
    print(f"  exp_search_w: {ddpg_best['exp_search_iterations_weight']:.1f}")
    print(f"  shifts_w: {ddpg_best['shifts_weight']:.2f}")
    print(f"  node_lookups_w: {ddpg_best['node_lookups_weight']:.1f}")
    print(f"  model_size_w: {ddpg_best['model_size_weight']:.2e}")

    return ddpg_best


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--mode", type=str, default="compare", choices=["single", "cross", "compare"])
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--max_steps", type=int, default=10)
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()

    if args.mode == "single":
        dataset = args.dataset or config.DATASETS[0]
        evaluate_single(args.model, dataset, n_episodes=args.episodes, max_steps=args.max_steps)
    elif args.mode == "cross":
        evaluate_cross(args.model, datasets=args.datasets, max_steps=args.max_steps)
    elif args.mode == "compare":
        dataset = args.dataset or config.DATASETS[0]
        evaluate_compare(args.model, dataset, max_steps=args.max_steps)
