import sys
import os
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import multiprocessing as mp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "drl"))

import config
from data_distribution import extract_features, load_keys

try:
    import alex_mem_wrapper
    WRAPPER_AVAILABLE = True
except ImportError:
    WRAPPER_AVAILABLE = False


def _run_benchmark_worker(params, keys_path, init_count, op_count, read_ratio, insert_frac, result_queue):
    try:
        import alex_mem_wrapper

        node_size_mb, init_d, max_d, min_d, exp_search_w, shifts_w, node_lookups_w, model_size_w = params

        action = alex_mem_wrapper.ALEXMemAction()
        action.expected_insert_frac = insert_frac
        action.max_node_size = int(node_size_mb * 1024 * 1024)
        action.init_density = float(init_d)
        action.max_density = float(max_d)
        action.min_density = float(min_d)
        action.exp_search_iterations_weight = float(exp_search_w)
        action.shifts_weight = float(shifts_w)
        action.node_lookups_weight = float(node_lookups_w)
        action.model_size_weight = float(model_size_w)

        wrapper = alex_mem_wrapper.ALEXMemWrapper()
        result = wrapper.run_benchmark(action, keys_path, init_count, op_count, read_ratio)

        if result.get("success", 0) > 0:
            result_queue.put((result["throughput"], result.get("total_smo", 0)))
        else:
            result_queue.put((0.0, 0))
    except Exception as e:
        result_queue.put((0.0, 0))


class ALEXDistEnv(gym.Env):

    NODE_SIZE_MIN = 4
    NODE_SIZE_MAX = 64
    DENSITY_MIN = 0.20
    DENSITY_MAX = 0.90

    EXP_SEARCH_WEIGHT_MIN = 5.0
    EXP_SEARCH_WEIGHT_MAX = 50.0
    SHIFTS_WEIGHT_MIN = 0.1
    SHIFTS_WEIGHT_MAX = 2.0
    NODE_LOOKUPS_WEIGHT_MIN = 5.0
    NODE_LOOKUPS_WEIGHT_MAX = 50.0
    MODEL_SIZE_WEIGHT_LOG_MIN = -8.0
    MODEL_SIZE_WEIGHT_LOG_MAX = -5.0

    def __init__(
        self,
        datasets=None,
        init_count=None,
        op_count=None,
        read_ratio=None,
        max_steps=10,
    ):
        super().__init__()

        if not WRAPPER_AVAILABLE:
            raise ImportError("alex_mem_wrapper not available")

        self.datasets = datasets or config.DATASETS
        self.init_count = init_count or config.INIT_COUNT
        self.op_count = op_count or config.OP_COUNT
        self.read_ratio = read_ratio if read_ratio is not None else config.READ_RATIO
        self.insert_frac = 1.0 - self.read_ratio
        self.max_steps = max_steps

        self.action_space = spaces.Box(
            low=np.zeros(8, dtype=np.float32),
            high=np.ones(8, dtype=np.float32),
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=np.zeros(config.FEATURE_DIM, dtype=np.float32),
            high=np.ones(config.FEATURE_DIM, dtype=np.float32),
            dtype=np.float32,
        )

        self.current_dataset = None
        self.dist_features = None
        self.current_params = None
        self.current_throughput = 0.0
        self.baseline_throughput = 0.0
        self.best_throughput = 0.0
        self.best_params = None
        self.episode_step = 0
        self.total_steps = 0
        self._cache = {}

    def _get_default_params(self):
        return np.array([16.0, 0.7, 0.8, 0.6, 20.0, 0.5, 20.0, 5e-7], dtype=np.float32)

    def _clip_params(self, params):
        node_size, init_d, max_d, min_d, exp_search_w, shifts_w, node_lookups_w, model_size_w = params
        gap = 0.05

        node_size = np.clip(node_size, self.NODE_SIZE_MIN, self.NODE_SIZE_MAX)

        sorted_densities = sorted([min_d, init_d, max_d])
        min_d, init_d, max_d = sorted_densities

        max_d = np.clip(max_d, self.DENSITY_MIN + 2 * gap, self.DENSITY_MAX)
        init_d = np.clip(init_d, self.DENSITY_MIN + gap, max_d - gap)
        min_d = np.clip(min_d, self.DENSITY_MIN, init_d - gap)

        if min_d >= init_d:
            min_d = init_d - gap
        if init_d >= max_d:
            init_d = max_d - gap
        if min_d >= init_d:
            min_d = init_d - gap

        exp_search_w = np.clip(exp_search_w, self.EXP_SEARCH_WEIGHT_MIN, self.EXP_SEARCH_WEIGHT_MAX)
        shifts_w = np.clip(shifts_w, self.SHIFTS_WEIGHT_MIN, self.SHIFTS_WEIGHT_MAX)
        node_lookups_w = np.clip(node_lookups_w, self.NODE_LOOKUPS_WEIGHT_MIN, self.NODE_LOOKUPS_WEIGHT_MAX)
        model_size_w = np.clip(model_size_w, 10 ** self.MODEL_SIZE_WEIGHT_LOG_MIN, 10 ** self.MODEL_SIZE_WEIGHT_LOG_MAX)

        return np.array([node_size, init_d, max_d, min_d, exp_search_w, shifts_w, node_lookups_w, model_size_w],
                        dtype=np.float32)

    def _run_benchmark(self, params, timeout=120):
        result_queue = mp.Queue()
        proc = mp.Process(
            target=_run_benchmark_worker,
            args=(params.tolist(), self.current_dataset, self.init_count, self.op_count,
                  self.read_ratio, self.insert_frac, result_queue),
        )
        proc.start()
        proc.join(timeout=timeout)

        if proc.is_alive():
            print("Benchmark timeout, killing worker...")
            proc.terminate()
            proc.join()
            return 0.0, 0

        if proc.exitcode != 0:
            print(f"Benchmark worker crashed (exitcode={proc.exitcode})")
            return 0.0, 0

        try:
            if not result_queue.empty():
                return result_queue.get_nowait()
        except Exception:
            pass
        return 0.0, 0

    def _params_to_obs(self):
        node_size, init_d, max_d, min_d, exp_search_w, shifts_w, node_lookups_w, model_size_w = self.current_params

        node_size_norm = (node_size - self.NODE_SIZE_MIN) / (self.NODE_SIZE_MAX - self.NODE_SIZE_MIN)
        init_d_norm = (init_d - self.DENSITY_MIN) / (self.DENSITY_MAX - self.DENSITY_MIN)
        max_d_norm = (max_d - self.DENSITY_MIN) / (self.DENSITY_MAX - self.DENSITY_MIN)
        min_d_norm = (min_d - self.DENSITY_MIN) / (self.DENSITY_MAX - self.DENSITY_MIN)

        exp_search_w_norm = (exp_search_w - self.EXP_SEARCH_WEIGHT_MIN) / (self.EXP_SEARCH_WEIGHT_MAX - self.EXP_SEARCH_WEIGHT_MIN)
        shifts_w_norm = (shifts_w - self.SHIFTS_WEIGHT_MIN) / (self.SHIFTS_WEIGHT_MAX - self.SHIFTS_WEIGHT_MIN)
        node_lookups_w_norm = (node_lookups_w - self.NODE_LOOKUPS_WEIGHT_MIN) / (self.NODE_LOOKUPS_WEIGHT_MAX - self.NODE_LOOKUPS_WEIGHT_MIN)
        model_size_w_log = np.log10(model_size_w) if model_size_w > 0 else self.MODEL_SIZE_WEIGHT_LOG_MIN
        model_size_w_norm = (model_size_w_log - self.MODEL_SIZE_WEIGHT_LOG_MIN) / (self.MODEL_SIZE_WEIGHT_LOG_MAX - self.MODEL_SIZE_WEIGHT_LOG_MIN)

        return np.array([
            node_size_norm, init_d_norm, max_d_norm, min_d_norm,
            exp_search_w_norm, shifts_w_norm, node_lookups_w_norm, model_size_w_norm,
        ], dtype=np.float32)

    def _get_obs(self):
        params_obs = self._params_to_obs()

        if self.baseline_throughput > 0:
            throughput_norm = np.clip(self.current_throughput / self.baseline_throughput / 2.0, 0.0, 1.0)
        else:
            throughput_norm = 0.5

        obs = np.concatenate([self.dist_features, params_obs, [throughput_norm]])
        return np.clip(obs, 0.0, 1.0).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.episode_step = 0

        idx = self.np_random.integers(0, len(self.datasets))
        self.current_dataset = self.datasets[idx]

        if self.current_dataset in self._cache:
            self.dist_features, self.baseline_throughput = self._cache[self.current_dataset]
            print(f"[reset] cached: {self.current_dataset} (baseline={self.baseline_throughput:.0f})")
        else:
            print(f"[reset] extracting features: {self.current_dataset}")
            keys = load_keys(self.current_dataset, self.init_count)
            self.dist_features = extract_features(keys)
            del keys

            default_params = self._get_default_params()
            print(f"[reset] getting baseline...")
            throughput, smo = self._run_benchmark(default_params)
            self.baseline_throughput = throughput
            print(f"[reset] Baseline: {throughput:.0f} ops/s, SMO: {smo}")

            self._cache[self.current_dataset] = (self.dist_features.copy(), self.baseline_throughput)

        self.current_params = self._get_default_params()
        self.current_throughput = self.baseline_throughput
        self.best_throughput = self.baseline_throughput
        self.best_params = self.current_params.copy()

        return self._get_obs(), {}

    def step(self, action):
        self.episode_step += 1
        self.total_steps += 1
        prev_throughput = self.current_throughput

        node_size = self.NODE_SIZE_MIN + action[0] * (self.NODE_SIZE_MAX - self.NODE_SIZE_MIN)
        init_d = self.DENSITY_MIN + action[1] * (self.DENSITY_MAX - self.DENSITY_MIN)
        max_d = self.DENSITY_MIN + action[2] * (self.DENSITY_MAX - self.DENSITY_MIN)
        min_d = self.DENSITY_MIN + action[3] * (self.DENSITY_MAX - self.DENSITY_MIN)

        exp_search_w = self.EXP_SEARCH_WEIGHT_MIN + action[4] * (self.EXP_SEARCH_WEIGHT_MAX - self.EXP_SEARCH_WEIGHT_MIN)
        shifts_w = self.SHIFTS_WEIGHT_MIN + action[5] * (self.SHIFTS_WEIGHT_MAX - self.SHIFTS_WEIGHT_MIN)
        node_lookups_w = self.NODE_LOOKUPS_WEIGHT_MIN + action[6] * (self.NODE_LOOKUPS_WEIGHT_MAX - self.NODE_LOOKUPS_WEIGHT_MIN)
        model_size_w_log = self.MODEL_SIZE_WEIGHT_LOG_MIN + action[7] * (self.MODEL_SIZE_WEIGHT_LOG_MAX - self.MODEL_SIZE_WEIGHT_LOG_MIN)
        model_size_w = 10 ** model_size_w_log

        new_params = np.array(
            [node_size, init_d, max_d, min_d, exp_search_w, shifts_w, node_lookups_w, model_size_w],
            dtype=np.float32,
        )
        new_params = self._clip_params(new_params)

        throughput, smo = self._run_benchmark(new_params)
        crashed = throughput <= 0

        self.current_params = new_params
        self.current_throughput = throughput if not crashed else prev_throughput

        if crashed:
            reward = -1.0
        elif self.baseline_throughput > 0:
            reward = (throughput - self.baseline_throughput) / self.baseline_throughput
        else:
            reward = 0.0

        if throughput > self.best_throughput:
            self.best_throughput = throughput
            self.best_params = new_params.copy()

        terminated = self.episode_step >= self.max_steps
        truncated = False

        obs = self._get_obs()

        improvement = ((throughput - self.baseline_throughput) / self.baseline_throughput * 100
                       if self.baseline_throughput > 0 and not crashed else -100.0)
        info = {
            "throughput": throughput,
            "smo": smo,
            "params": new_params.tolist(),
            "best_throughput": self.best_throughput,
            "baseline_throughput": self.baseline_throughput,
            "improvement": improvement,
            "crashed": crashed,
            "dataset": self.current_dataset,
            "episode_step": self.episode_step,
        }

        return obs, reward, terminated, truncated, info

    def get_best_config(self):
        if self.best_params is None:
            return None
        return {
            "node_size_mb": self.best_params[0],
            "init_density": self.best_params[1],
            "max_density": self.best_params[2],
            "min_density": self.best_params[3],
            "exp_search_iterations_weight": self.best_params[4],
            "shifts_weight": self.best_params[5],
            "node_lookups_weight": self.best_params[6],
            "model_size_weight": self.best_params[7],
            "throughput": self.best_throughput,
            "baseline_throughput": self.baseline_throughput,
            "improvement": (self.best_throughput - self.baseline_throughput) / self.baseline_throughput * 100
            if self.baseline_throughput > 0 else 0.0,
        }


def test_env():
    env = ALEXDistEnv(max_steps=3)

    print(f"action_space: {env.action_space}")
    print(f"observation_space: {env.observation_space}")

    obs, info = env.reset()
    print(f"\ninit obs shape={obs.shape}, range=[{obs.min():.4f}, {obs.max():.4f}]")

    for i in range(3):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"\nStep {i+1}: throughput={info['throughput']:.0f} "
              f"({info['improvement']:+.1f}%) reward={reward:.4f} "
              f"terminated={terminated}")
        if terminated:
            obs, _ = env.reset()

    best = env.get_best_config()
    if best:
        print(f"\nbest: {best['throughput']:.0f} ops/s ({best['improvement']:+.1f}%)")


if __name__ == "__main__":
    test_env()
