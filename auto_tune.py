"""
Auto-tuning module for dbproc.py parameters.

This module provides automatic parameter optimization to achieve target drift
and maintain correlation quality for data generation.
"""

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import distance

# Add drift_ddpm to path for imports
sys.path.append("drift_ddpm")


@dataclass
class TuningParams:
    """Parameters that can be tuned for data generation."""
    # Diffuser parameters
    diffuser_lr: float = 0.0018
    diffuser_steps: int = 30000
    diffuser_bs: int = 2048
    diffuser_timesteps: int = 1000

    # Controller parameters
    controller_lr: float = 0.001
    controller_steps: int = 10000
    controller_bs: int = 512
    controller_dim: tuple = (512, 512)  # Hidden layer dimensions

    # Sampling parameters
    scale_factor: float = 8.0

    # Lambda parameters for diffuser training
    lambda_p: float = 1.0
    lambda_s: float = 1.0

    # Controller training improvement parameters
    drift_range_min: float = 0.05
    drift_range_max: float = 0.75
    loss_weight_corr: float = 0.8
    loss_weight_real: float = 0.1

    def to_cmd_args(self) -> str:
        """Convert to command line arguments."""
        args = [
            f"--diffuser-lr={self.diffuser_lr}",
            f"--diffuser-steps={self.diffuser_steps}",
            f"--diffuser-bs={self.diffuser_bs}",
            f"--diffuser-timesteps={self.diffuser_timesteps}",
            f"--controller-lr={self.controller_lr}",
            f"--controller-steps={self.controller_steps}",
            f"--controller-bs={self.controller_bs}",
            f"--controller-dim {' '.join(str(d) for d in self.controller_dim)}",
            f"--scale-factor={self.scale_factor}",
            f"--lambda-p={self.lambda_p}",
            f"--lambda-s={self.lambda_s}",
            f"--drift-range-min={self.drift_range_min}",
            f"--drift-range-max={self.drift_range_max}",
            f"--loss-weight-corr={self.loss_weight_corr}",
            f"--loss-weight-real={self.loss_weight_real}",
        ]
        return " ".join(args)


@dataclass
class TuningResult:
    """Result of a tuning run."""
    params: TuningParams
    target_drift: float
    actual_drift: float
    correlation_loss: float  # Mean absolute correlation loss
    drift_error: float  # Relative error: |actual_drift - target_drift| / target_drift
    score: float  # Combined score (lower is better)
    timestamp: str = ""
    validation_passed: bool = False  # Whether DB validation passed
    validation_ratio: float = 0.0  # Query time ratio from validation

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.strftime("%Y-%m-%d %H:%M:%S")


@dataclass
class TableTuningCache:
    """Cache of tuning results for a specific table."""
    dataset_name: str
    table_name: str
    # Map from target_drift (rounded to 2 decimals) to best params
    best_params: Dict[str, TuningResult] = field(default_factory=dict)
    # History of all tuning runs
    history: List[TuningResult] = field(default_factory=list)

    def get_best_params(self, target_drift: float) -> Optional[TuningResult]:
        """Get best params for a target drift."""
        key = f"{target_drift:.2f}"
        return self.best_params.get(key)

    def update(self, result: TuningResult):
        """Update cache with new result."""
        self.history.append(result)
        key = f"{result.target_drift:.2f}"

        if key not in self.best_params or result.score < self.best_params[key].score:
            self.best_params[key] = result

    def update_validation_status(self, target_drift: float, passed: bool, ratio: float):
        """Update validation status for a cached result."""
        key = f"{target_drift:.2f}"
        if key in self.best_params:
            self.best_params[key].validation_passed = passed
            self.best_params[key].validation_ratio = ratio


class DataEvaluator:
    """Evaluates generated data quality."""

    CORR_TYPES = ["pearson", "spearman"]

    def __init__(self, dataset_name: str, table_name: str):
        self.dataset_name = dataset_name
        self.table_name = table_name
        self._csv_cache = {}  # Cache loaded CSVs to avoid repeated parsing warnings

    def _load_dataset_info(self) -> dict:
        """Load dataset info for the table."""
        info_path = f"datasets/{self.dataset_name}/dataset_info.json"
        with open(info_path, "r") as f:
            info = json.load(f)
        return info.get(self.table_name, {})

    def _numerical_dist(self, series: pd.Series, n_bins: int = 20):
        """Get distribution of numerical values."""
        if all(isinstance(x, (int, np.integer)) for x in series.dropna()):
            nunique = series.nunique()
            if nunique < n_bins:
                n_bins = max(nunique - 1, 1)
            edges = np.linspace(series.min(), series.max(), n_bins + 1)
            edges = np.unique(edges.astype(int))
            if len(edges) <= 1:
                return series.value_counts(normalize=True)
            return pd.cut(series, bins=edges, precision=0, include_lowest=True).value_counts(normalize=True, sort=False)
        else:
            return series.value_counts(bins=n_bins, normalize=True, sort=False)

    def _numerical_dist_on_bins(self, series: pd.Series, bins: list):
        """Get distribution on predefined bins."""
        indices = pd.IntervalIndex.from_tuples([(x.left, x.right) for x in bins])
        return pd.cut(series, bins=indices).value_counts(normalize=True, sort=False)

    def _categorical_dist(self, series: pd.Series):
        """Get distribution of categorical values."""
        return series.value_counts(normalize=True)

    def _categorical_dist_on_bins(self, series: pd.Series, bins: list):
        """Get distribution on predefined bins."""
        distribution = series.value_counts().reindex(bins, fill_value=0)
        return distribution / max(len(series), 1)

    def _is_numerical(self, series: pd.Series) -> bool:
        """Check if column is numerical."""
        for x in series.dropna().head(100):
            if isinstance(x, float):
                return True
        if all(isinstance(x, (int, np.integer)) for x in series.dropna().head(100)):
            return series.nunique() >= 20
        return False

    def _load_csv(self, path: str, use_cache: bool = True) -> pd.DataFrame:
        """Load CSV with multiple parsing strategies (same as dbproc.py).

        Args:
            path: Path to CSV file
            use_cache: If True, cache the result to avoid repeated parsing warnings.
                       Set to False for files that change between calls (e.g., drifted.csv)
        """
        if use_cache and path in self._csv_cache:
            return self._csv_cache[path]

        for strategy in [
            {"doublequote": True},                        # Standard CSV (most common)
            {"doublequote": False, "escapechar": "\\"},   # Backslash escaped
        ]:
            try:
                df = pd.read_csv(
                    path, low_memory=False, on_bad_lines='warn', **strategy
                )
                if use_cache:
                    self._csv_cache[path] = df
                return df
            except Exception:
                continue
        raise RuntimeError(f"Failed to load {path}")

    def evaluate(self, original_path: str, drifted_path: str) -> Tuple[float, float]:
        """
        Evaluate drifted data quality.

        Returns:
            (mean_drift, mean_corr_loss): Mean JS divergence and mean correlation loss
        """
        # Cache original data (doesn't change), but not drifted (regenerated each iteration)
        original_data = self._load_csv(original_path, use_cache=True)
        drifted_data = self._load_csv(drifted_path, use_cache=False)

        dataset_info = self._load_dataset_info()
        drifted_columns = dataset_info.get("applicable_columns", [])

        # Calculate drift (JS divergence)
        divergences = []
        for col in drifted_columns:
            if col not in original_data.columns or col not in drifted_data.columns:
                continue

            col_data = original_data[col].dropna()
            if len(col_data) == 0:
                continue

            try:
                if self._is_numerical(col_data):
                    original_dist = self._numerical_dist(col_data)
                    bins = original_dist.index
                    drifted_dist = self._numerical_dist_on_bins(drifted_data[col].dropna(), bins)
                else:
                    original_dist = self._categorical_dist(col_data)
                    bins = sorted(original_dist.index)
                    drifted_dist = self._categorical_dist_on_bins(drifted_data[col], bins)

                jsd = distance.jensenshannon(original_dist.values, drifted_dist.values)
                if np.isnan(jsd):
                    jsd = 1.0
                divergences.append(jsd)
            except Exception as e:
                print(f"Warning: Error computing drift for column {col}: {e}")
                continue

        mean_drift = np.mean(divergences) if divergences else 0.0

        # Calculate correlation loss
        corr_losses = []
        for corr_type in self.CORR_TYPES:
            try:
                original_corr = original_data.corr(method=corr_type, numeric_only=True)
                drifted_corr = drifted_data.corr(method=corr_type, numeric_only=True)

                # Align columns
                common_cols = original_corr.columns.intersection(drifted_corr.columns)
                if len(common_cols) > 0:
                    original_corr = original_corr.loc[common_cols, common_cols]
                    drifted_corr = drifted_corr.loc[common_cols, common_cols]

                    loss = (drifted_corr - original_corr).abs()
                    mean_loss = loss.mean().mean()
                    if not np.isnan(mean_loss):
                        corr_losses.append(mean_loss)
            except Exception as e:
                print(f"Warning: Error computing {corr_type} correlation: {e}")
                continue

        mean_corr_loss = np.mean(corr_losses) if corr_losses else 0.0

        return mean_drift, mean_corr_loss


class AutoTuner:
    """Automatic parameter tuning for data generation."""

    CACHE_DIR = "tuning_cache"

    def __init__(
        self,
        dataset_name: str,
        table_name: str,
        reference_dataset: str = None,
        device: int = 0,
        verbose: bool = True,
        target_corr_loss: float = None,  # Target correlation loss from reference
    ):
        self.dataset_name = dataset_name
        self.table_name = table_name
        self.reference_dataset = reference_dataset
        self.device = device
        self.verbose = verbose
        self.target_corr_loss = target_corr_loss  # If set, use as target instead of minimizing
        self.evaluator = DataEvaluator(dataset_name, table_name)
        self.cache = self._load_cache()

    def _cache_path(self) -> str:
        """Get path to cache file."""
        os.makedirs(self.CACHE_DIR, exist_ok=True)
        # Include reference dataset in cache filename to separate different reference configs
        if self.reference_dataset:
            return os.path.join(self.CACHE_DIR, f"{self.dataset_name}_{self.table_name}_ref_{self.reference_dataset}.json")
        return os.path.join(self.CACHE_DIR, f"{self.dataset_name}_{self.table_name}.json")

    def _load_cache(self) -> TableTuningCache:
        """Load tuning cache from disk."""
        path = self._cache_path()
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                cache = TableTuningCache(
                    dataset_name=data["dataset_name"],
                    table_name=data["table_name"],
                )
                # Reconstruct best_params
                for key, result_data in data.get("best_params", {}).items():
                    params_data = result_data["params"]
                    # Convert list back to tuple for controller_dim
                    if "controller_dim" in params_data and isinstance(params_data["controller_dim"], list):
                        params_data["controller_dim"] = tuple(params_data["controller_dim"])
                    params = TuningParams(**params_data)
                    result = TuningResult(
                        params=params,
                        target_drift=result_data["target_drift"],
                        actual_drift=result_data["actual_drift"],
                        correlation_loss=result_data["correlation_loss"],
                        drift_error=result_data["drift_error"],
                        score=result_data["score"],
                        timestamp=result_data.get("timestamp", ""),
                        validation_passed=result_data.get("validation_passed", False),
                        validation_ratio=result_data.get("validation_ratio", 0.0),
                    )
                    cache.best_params[key] = result
                return cache
            except Exception as e:
                print(f"Warning: Failed to load cache: {e}")
        return TableTuningCache(dataset_name=self.dataset_name, table_name=self.table_name)

    def _save_cache(self):
        """Save tuning cache to disk."""
        path = self._cache_path()
        data = {
            "dataset_name": self.cache.dataset_name,
            "table_name": self.cache.table_name,
            "best_params": {},
        }
        for key, result in self.cache.best_params.items():
            data["best_params"][key] = {
                "params": asdict(result.params),
                "target_drift": result.target_drift,
                "actual_drift": result.actual_drift,
                "correlation_loss": result.correlation_loss,
                "drift_error": result.drift_error,
                "score": result.score,
                "timestamp": result.timestamp,
                "validation_passed": result.validation_passed,
                "validation_ratio": result.validation_ratio,
            }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def _save_last_generation_metadata(
        self,
        params: TuningParams,
        target_drift: float,
        actual_drift: float,
        correlation_loss: float,
    ):
        """
        Save metadata about the last generation for manual validation.
        This allows 'vd' command to know which cache entry to update.
        """
        metadata_dir = os.path.join("expdir", self.dataset_name, self.table_name)
        os.makedirs(metadata_dir, exist_ok=True)
        metadata_path = os.path.join(metadata_dir, "last_generation.json")

        metadata = {
            "dataset_name": self.dataset_name,
            "table_name": self.table_name,
            "reference_dataset": self.reference_dataset,
            "target_drift": target_drift,
            "actual_drift": actual_drift,
            "correlation_loss": correlation_loss,
            "scale_factor": params.scale_factor,
            "target_corr_loss": self.target_corr_loss,
            "cache_key": f"{target_drift:.2f}",
            "cache_path": self._cache_path(),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        if self.verbose:
            print(f"Saved generation metadata to: {metadata_path}")

    def _run_generation(
        self,
        params: TuningParams,
        target_drift: float,
        retrain: bool = True,
    ) -> Tuple[bool, str]:
        """
        Run data generation with given parameters.

        Returns:
            (success, output_path)
        """
        cmd = f"python3 dbproc.py --dataset-name={self.dataset_name} --table-name={self.table_name}"
        cmd += f" --drift={target_drift} --device={self.device}"
        cmd += f" {params.to_cmd_args()}"

        if self.reference_dataset:
            cmd += f" --reference-dataset={self.reference_dataset}"

        if retrain:
            cmd += " --retrain-diffuser --retrain-controller"
        else:
            cmd += " --reuse"

        if self.verbose:
            print(f"Running: {cmd}")

        try:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=not self.verbose,
                text=True,
                timeout=7200,  # 2 hour timeout
            )

            if result.returncode != 0:
                if not self.verbose and result.stderr:
                    print(f"Generation failed: {result.stderr}")
                return False, ""

            output_path = os.path.join(
                "expdir", self.dataset_name, self.table_name,
                f"{self.table_name}.drifted.csv"
            )

            if not os.path.exists(output_path):
                print(f"Output file not found: {output_path}")
                return False, ""

            return True, output_path

        except subprocess.TimeoutExpired:
            print("Generation timed out")
            return False, ""
        except Exception as e:
            print(f"Generation error: {e}")
            return False, ""

    def _evaluate_params(
        self,
        params: TuningParams,
        target_drift: float,
        retrain: bool = True,
    ) -> Optional[TuningResult]:
        """Evaluate a set of parameters."""
        success, output_path = self._run_generation(params, target_drift, retrain)

        if not success:
            return None

        original_path = os.path.join(
            "datasets", self.dataset_name, f"{self.table_name}.csv"
        )

        actual_drift, corr_loss = self.evaluator.evaluate(original_path, output_path)
        # Relative error: |actual - target| / target
        drift_error = abs(actual_drift - target_drift) / target_drift if target_drift > 0 else abs(actual_drift)

        # Score: weighted combination of drift error and correlation loss/error
        # Lower is better
        if self.target_corr_loss is not None:
            # If we have a target correlation loss, score based on how close we are to it
            corr_error = abs(self.target_corr_loss - corr_loss)
            score = drift_error * 2.0 + corr_error * 1.5
        else:
            # Otherwise, just minimize correlation loss
            score = drift_error * 2.0 + corr_loss * 1.0

        result = TuningResult(
            params=params,
            target_drift=target_drift,
            actual_drift=actual_drift,
            correlation_loss=corr_loss,
            drift_error=drift_error,
            score=score,
        )

        # Save last generation metadata for manual validation
        self._save_last_generation_metadata(params, target_drift, actual_drift, corr_loss)

        if self.verbose:
            print(f"\n--- Evaluation Result ---")
            print(f"Target drift: {target_drift:.4f}")
            print(f"Actual drift: {actual_drift:.4f}")
            print(f"Drift error: {drift_error:.2%}")  # Relative error
            print(f"Correlation loss: {corr_loss:.4f}")
            if self.target_corr_loss is not None:
                corr_error = abs(self.target_corr_loss - corr_loss)  # Absolute error
                print(f"Target corr loss: {self.target_corr_loss:.4f}")
                print(f"Corr error: {corr_error:.4f} (tolerance: 0.20)")
            print(f"Score: {score:.4f}")
            print("-" * 25)

        return result

    def _get_initial_scale_factor(self, target_drift: float) -> float:
        """Estimate initial scale_factor based on target drift."""
        # Empirical mapping: higher drift needs higher scale_factor
        # This is a rough estimate, will be refined by adaptive search
        if target_drift < 0.05:
            return 0.1
        elif target_drift < 0.1:
            return 0.3
        elif target_drift < 0.2:
            return 1.0
        elif target_drift < 0.3:
            return 3.0
        elif target_drift < 0.5:
            return 6.0
        else:
            return 10.0

    def tune(
        self,
        target_drift: float,
        max_iterations: int = 100,
        tolerance: float = 0.20,
        use_cache: bool = True,
        stop_only_on_tolerance: bool = True,
        stagnation_limit: int = 5,
        retrain_interval: int = 10,
        require_validation: bool = False,
    ) -> Optional[TuningResult]:
        """
        Automatically tune parameters for target drift using adaptive binary search.

        Args:
            target_drift: Target drift value (0.0 - 1.0)
            max_iterations: Maximum number of iterations (safety limit)
            tolerance: Acceptable drift error tolerance
            use_cache: Whether to use cached results
            stop_only_on_tolerance: If True, only stop when tolerance is met (up to max_iterations)
            stagnation_limit: Number of iterations without improvement before trying retrain
            retrain_interval: Retrain models every N iterations if not improving
            require_validation: If True, only use cache if validation_passed=True

        Returns:
            Best TuningResult found
        """
        print(f"\n{'='*60}")
        print(f"Auto-tuning for {self.dataset_name}.{self.table_name}")
        print(f"Target drift: {target_drift}")
        if self.target_corr_loss is not None:
            print(f"Target corr loss: {self.target_corr_loss:.4f}")
        print(f"Reference dataset: {self.reference_dataset or f'{self.dataset_name}_2014 (default)'}")
        print(f"Max iterations: {max_iterations}")
        print(f"Tolerance: {tolerance}")
        print(f"Stop only on tolerance: {stop_only_on_tolerance}")
        if require_validation:
            print(f"Require validation: True")
        print(f"{'='*60}\n")

        # Check cache first
        if use_cache:
            cached = self.cache.get_best_params(target_drift)
            if cached and cached.drift_error <= tolerance:
                # Also check correlation error if target_corr_loss is set
                # drift_error: relative (<=20%), corr_error: absolute (<=0.20)
                corr_ok = True
                corr_error = 0.0
                corr_tolerance = 0.20
                if self.target_corr_loss is not None:
                    corr_error = abs(cached.correlation_loss - self.target_corr_loss)  # Absolute
                    corr_ok = corr_error <= corr_tolerance

                if corr_ok:
                    if require_validation:
                        # Validation mode: must have passed validation to use cache
                        if cached.validation_passed:
                            print(f"Using cached parameters (drift_error={cached.drift_error:.2%}, corr_error={corr_error:.4f}, validated=✓)")
                            return cached
                        else:
                            print(f"Cache found but validation not passed (ratio={cached.validation_ratio:.2f}x), re-tuning...")
                    else:
                        # No validation mode: use cache if within tolerance
                        print(f"Using cached parameters (drift_error={cached.drift_error:.2%}, corr_error={corr_error:.4f})")
                        return cached
                else:
                    print(f"Cache found but corr_error too high ({corr_error:.4f} > {corr_tolerance}), re-tuning...")

        best_result = None

        # Adaptive search bounds for scale_factor
        # Lower bound needs to be small enough for low drift targets (e.g., 0.05-0.1)
        sf_low = 0.01
        sf_high = 30.0
        current_sf = self._get_initial_scale_factor(target_drift)

        # Track history for smarter adjustments
        history = []  # List of (scale_factor, actual_drift, drift_error)

        # Stagnation tracking
        iterations_without_improvement = 0
        last_best_score = float('inf')

        # Correlation focus mode: when drift is OK but correlation is bad
        corr_focus_mode = False
        corr_focus_weight = 0.8  # Starting correlation weight
        corr_focus_attempt = 0   # Number of attempts in correlation focus mode

        # Drift insensitivity detection: when scale_factor changes but drift doesn't respond
        drift_insensitive_count = 0  # Count of consecutive iterations with tiny gradient
        drift_insensitive_retrain_count = 0  # Number of retrains due to drift insensitivity

        for i in range(max_iterations):
            # Determine if we should retrain
            # Retrain on: first iteration, after stagnation, or at retrain_interval
            should_retrain = (i == 0) or \
                             (iterations_without_improvement >= stagnation_limit) or \
                             (i > 0 and i % retrain_interval == 0 and best_result and best_result.drift_error > tolerance)

            if should_retrain and i > 0:
                if corr_focus_mode:
                    print(f"\n*** Retraining controller with correlation focus (weight={corr_focus_weight:.2f}) ***")
                else:
                    print(f"\n*** Retraining models (stagnation={iterations_without_improvement}, interval check) ***")
                # Reset search bounds when retraining (only if not in corr_focus_mode)
                if not corr_focus_mode:
                    sf_low = 0.01
                    sf_high = 30.0
                iterations_without_improvement = 0

            # Set controller params based on mode
            if corr_focus_mode and should_retrain:
                # In correlation focus mode, use higher correlation weight
                ctrl_corr_weight = corr_focus_weight

                # 根据尝试次数调整策略
                # 渐进式: 先增大两层宽度, 然后加层
                if corr_focus_attempt >= 5:
                    # 多次尝试失败,用更极端的参数
                    ctrl_lr = 0.0003
                    ctrl_steps = 30000
                    ctrl_real_weight = 0.01  # 几乎忽略real loss
                    ctrl_dim = (1024, 1024, 512)  # 三层最大
                elif corr_focus_attempt >= 3:
                    # 中等尝试次数
                    ctrl_lr = 0.0004
                    ctrl_steps = 25000
                    ctrl_real_weight = 0.02
                    ctrl_dim = (1024, 768, 512)  # 三层
                else:
                    # 初始尝试
                    ctrl_lr = 0.0005
                    ctrl_steps = 20000
                    ctrl_real_weight = 0.05
                    ctrl_dim = (768, 768)  # 两层更宽
            elif should_retrain:
                # Check if we're retraining due to drift insensitivity
                if drift_insensitive_retrain_count > 0:
                    print(f"*** Drift insensitivity retrain #{drift_insensitive_retrain_count} ***")
                    # Use progressively stronger controller for drift-insensitive tables
                    # 渐进式: 先增大两层宽度, 然后加层
                    if drift_insensitive_retrain_count >= 3:
                        ctrl_dim = (1024, 1024, 512)  # 三层最大
                        ctrl_steps = 25000
                        ctrl_lr = 0.0005
                    elif drift_insensitive_retrain_count >= 2:
                        ctrl_dim = (1024, 768, 512)  # 三层
                        ctrl_steps = 20000
                        ctrl_lr = 0.0006
                    else:
                        ctrl_dim = (768, 768)  # 两层更宽
                        ctrl_steps = 18000
                        ctrl_lr = 0.0007
                    ctrl_corr_weight = 0.5  # Lower corr weight to focus more on drift control
                    ctrl_real_weight = 0.2  # Higher real weight for better distribution match
                else:
                    ctrl_lr = 0.0008
                    ctrl_steps = 15000
                    ctrl_corr_weight = 0.8
                    ctrl_real_weight = 0.1
                    ctrl_dim = (512, 512)  # 默认网络
            else:
                ctrl_lr = 0.001
                ctrl_steps = 12000
                ctrl_corr_weight = 0.8
                ctrl_real_weight = 0.1
                ctrl_dim = (512, 512)  # 默认网络

            params = TuningParams(
                scale_factor=current_sf,
                controller_lr=ctrl_lr,
                controller_steps=ctrl_steps,
                controller_dim=ctrl_dim,
                # Focus drift_range around target
                drift_range_min=max(0.01, target_drift - 0.15),
                drift_range_max=min(0.95, target_drift + 0.15),
                loss_weight_corr=ctrl_corr_weight,
                loss_weight_real=ctrl_real_weight,
            )

            print(f"\n[Iteration {i+1}/{max_iterations}]")
            if corr_focus_mode:
                print(f"*** Correlation Focus Mode (attempt={corr_focus_attempt}) ***")
                print(f"    corr_weight={params.loss_weight_corr:.2f}, "
                      f"real_weight={params.loss_weight_real:.3f}, "
                      f"ctrl_dim={params.controller_dim}")
            print(f"Params: scale_factor={current_sf:.2f}, "
                  f"drift_range=({params.drift_range_min:.2f}, {params.drift_range_max:.2f}), "
                  f"lr={params.controller_lr:.4f}, steps={params.controller_steps}, "
                  f"retrain={should_retrain}")

            result = self._evaluate_params(params, target_drift, retrain=should_retrain)

            if result is None:
                print("Evaluation failed, trying different scale_factor...")
                current_sf = (sf_low + sf_high) / 2
                iterations_without_improvement += 1
                continue

            # Track history
            history.append((current_sf, result.actual_drift, result.drift_error))

            # Update cache
            self.cache.update(result)
            self._save_cache()

            # Update best and track improvement
            if best_result is None or result.score < best_result.score:
                best_result = result
                print(f"New best! Score: {result.score:.4f}, Drift error: {result.drift_error:.4f}")
                iterations_without_improvement = 0
                last_best_score = result.score
            else:
                iterations_without_improvement += 1
                print(f"No improvement ({iterations_without_improvement}/{stagnation_limit})")

            # Success: found params within tolerance
            # drift_error: relative (<=20%), corr_error: absolute (<=0.20)
            drift_ok = result.drift_error <= tolerance
            corr_tolerance = 0.20  # Absolute tolerance for correlation
            if self.target_corr_loss is not None:
                corr_error = abs(result.correlation_loss - self.target_corr_loss)  # Absolute error
                corr_ok = corr_error <= corr_tolerance
                if drift_ok and corr_ok:
                    print(f"\n✓ Found params within tolerance! (drift_error={result.drift_error:.2%}, corr_error={corr_error:.4f})")
                    # Reset insensitivity counters on success
                    drift_insensitive_count = 0
                    drift_insensitive_retrain_count = 0
                    break
                elif drift_ok and not corr_ok:
                    # Drift is good but correlation is bad - need to focus on controller training
                    print(f"\n⚠ Drift OK but correlation error too high ({corr_error:.4f} > {corr_tolerance})")

                    # Enter or continue correlation focus mode
                    if not corr_focus_mode:
                        corr_focus_mode = True
                        corr_focus_attempt = 0
                        # Set initial weight based on how bad the correlation error is
                        # 误差越大,初始权重越高
                        if corr_error > 0.4:
                            corr_focus_weight = 3.0
                        elif corr_error > 0.3:
                            corr_focus_weight = 2.0
                        else:
                            corr_focus_weight = 1.2
                        print(f"  Entering correlation focus mode (weight={corr_focus_weight:.2f})")
                    else:
                        corr_focus_attempt += 1
                        # 根据尝试次数和误差大小动态调整
                        # 误差大 -> 更激进的增长
                        increment = 0.5 if corr_error > 0.3 else 0.3
                        corr_focus_weight = min(5.0, corr_focus_weight + increment)
                        print(f"  Attempt {corr_focus_attempt}: Increasing correlation weight to {corr_focus_weight:.2f}")

                        # 如果尝试多次还是不行,考虑其他策略
                        if corr_focus_attempt >= 5:
                            print(f"  Warning: {corr_focus_attempt} attempts without success, trying different approach")
                            # 尝试降低real weight到接近0
                            # 这会在下一次retrain时生效

                    print(f"  Keeping scale_factor={current_sf:.2f}, will retrain controller")
                    # Force retrain next iteration with correlation focus
                    iterations_without_improvement = stagnation_limit
                    continue
            elif drift_ok:
                print(f"\n✓ Found params within tolerance! (drift_error={result.drift_error:.2%})")
                drift_insensitive_count = 0
                drift_insensitive_retrain_count = 0
                break

            # If we're in corr_focus_mode but correlation is now OK, exit the mode
            if corr_focus_mode and self.target_corr_loss is not None:
                corr_error = abs(result.correlation_loss - self.target_corr_loss)
                if corr_error <= corr_tolerance:
                    print(f"Correlation now OK (error={corr_error:.4f}), exiting correlation focus mode")
                    corr_focus_mode = False
                    corr_focus_weight = 0.8  # Reset weight
                    corr_focus_attempt = 0   # Reset attempt counter

            # Adaptive adjustment of scale_factor based on actual vs target drift
            drift_diff = result.actual_drift - target_drift

            print(f"Drift diff: {drift_diff:+.4f} (actual={result.actual_drift:.4f}, target={target_drift:.4f})")

            # Estimate gradient from history (how much drift changes per unit scale_factor)
            if len(history) >= 2:
                # Use last two points to estimate gradient
                sf1, drift1, _ = history[-2]
                sf2, drift2, _ = history[-1]
                sf_change = sf2 - sf1
                drift_change = drift2 - drift1

                if abs(sf_change) > 0.01:
                    gradient = drift_change / sf_change
                    print(f"Estimated gradient: {gradient:.4f} (drift/sf)")

                    if abs(gradient) > 0.001:
                        # Gradient is OK, reset insensitivity counter
                        drift_insensitive_count = 0

                        # Use gradient to calculate needed scale_factor change
                        drift_needed = target_drift - result.actual_drift
                        sf_change_needed = drift_needed / gradient

                        # Limit the change to avoid wild jumps
                        max_change = current_sf * 1.5  # Max 150% change
                        sf_change_needed = max(-max_change, min(max_change, sf_change_needed))

                        new_sf = current_sf + sf_change_needed
                        new_sf = max(0.1, min(50.0, new_sf))  # Absolute bounds

                        print(f"Gradient-based: sf_change_needed={sf_change_needed:+.2f} -> new_sf={new_sf:.2f}")

                        if drift_diff > 0:
                            sf_high = current_sf
                        else:
                            sf_low = current_sf

                        current_sf = new_sf
                    else:
                        # Gradient too small - table is insensitive, try larger jump
                        drift_insensitive_count += 1
                        print(f"Gradient too small ({gradient:.6f}), drift_insensitive_count={drift_insensitive_count}")

                        # If drift is consistently insensitive, force retrain with different params
                        if drift_insensitive_count >= 3:
                            drift_insensitive_retrain_count += 1
                            print(f"\n⚠ Drift insensitive to scale_factor for {drift_insensitive_count} iterations")
                            print(f"  Current model cannot control drift effectively")
                            print(f"  Forcing retrain #{drift_insensitive_retrain_count} with stronger controller")
                            iterations_without_improvement = stagnation_limit  # Force retrain
                            drift_insensitive_count = 0  # Reset counter
                            # Reset scale_factor and search bounds for fresh start
                            sf_low = 0.01
                            sf_high = 30.0
                            current_sf = self._get_initial_scale_factor(target_drift)
                            continue

                        if drift_diff > 0:
                            sf_high = current_sf
                            current_sf = current_sf * 0.5
                        else:
                            sf_low = current_sf
                            current_sf = current_sf * 2.0
                else:
                    # sf didn't change much, use default adjustment
                    adjustment = min(0.5, abs(drift_diff) * 2)
                    if drift_diff > 0:
                        sf_high = current_sf
                        current_sf = current_sf * (1 - adjustment)
                    else:
                        sf_low = current_sf
                        current_sf = current_sf * (1 + adjustment)
            else:
                # First iteration: use default proportional adjustment
                adjustment = min(0.5, abs(drift_diff) * 2)
                if drift_diff > 0:
                    sf_high = current_sf
                    current_sf = current_sf * (1 - adjustment)
                else:
                    sf_low = current_sf
                    current_sf = current_sf * (1 + adjustment)
                print(f"Initial adjustment: {adjustment:.2%}")

            # Clamp to bounds
            current_sf = max(sf_low, min(sf_high, current_sf))

            # If search range is too narrow and not improving
            if sf_high - sf_low < 0.5 and iterations_without_improvement >= stagnation_limit:
                print(f"Search range converged: [{sf_low:.2f}, {sf_high:.2f}]")
                if not stop_only_on_tolerance:
                    print("Stopping due to convergence (stop_only_on_tolerance=False)")
                    break
                else:
                    # Reset and try with different parameters
                    print("Resetting search bounds and retraining with different settings...")
                    sf_low = 0.01
                    sf_high = 30.0
                    current_sf = self._get_initial_scale_factor(target_drift) * 1.5  # Try different starting point
                    iterations_without_improvement = 0

        if best_result:
            print(f"\n{'='*60}")
            print(f"Best result:")
            print(f"  Scale factor: {best_result.params.scale_factor}")
            print(f"  Controller LR: {best_result.params.controller_lr}")
            print(f"  Controller steps: {best_result.params.controller_steps}")
            print(f"  Actual drift: {best_result.actual_drift:.4f}")
            print(f"  Drift error: {best_result.drift_error:.4f}")
            print(f"  Correlation loss: {best_result.correlation_loss:.4f}")
            print(f"{'='*60}\n")

        return best_result

    def quick_tune(
        self,
        target_drift: float,
        use_cache: bool = True,
    ) -> Optional[TuningResult]:
        """
        Quick tuning - only adjust scale_factor using existing models.
        Much faster than full tune() as it doesn't retrain models.

        Args:
            target_drift: Target drift value
            use_cache: Whether to use cached results

        Returns:
            Best TuningResult found
        """
        print(f"\n{'='*60}")
        print(f"Quick-tuning for {self.dataset_name}.{self.table_name}")
        print(f"Target drift: {target_drift}")
        print(f"Reference dataset: {self.reference_dataset or f'{self.dataset_name}_2014 (default)'}")
        print(f"{'='*60}\n")

        # Check cache
        if use_cache:
            cached = self.cache.get_best_params(target_drift)
            if cached:
                print(f"Using cached parameters")
                return cached

        # Check if models exist
        model_dir = os.path.join("expdir", self.dataset_name, self.table_name)
        diffuser_path = os.path.join(model_dir, "diffuser.pt")
        controller_path = os.path.join(model_dir, "controller.pt")

        if not os.path.exists(diffuser_path) or not os.path.exists(controller_path):
            print("Models not found. Running full tune instead.")
            return self.tune(target_drift, max_iterations=5)

        # Quick search over scale factors only
        if target_drift < 0.2:
            scale_factors = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        elif target_drift < 0.4:
            scale_factors = [4.0, 6.0, 8.0, 10.0, 12.0]
        else:
            scale_factors = [8.0, 10.0, 12.0, 14.0, 16.0, 20.0]

        best_result = None

        for sf in scale_factors:
            params = TuningParams(scale_factor=sf)
            print(f"Trying scale_factor={sf}")

            result = self._evaluate_params(params, target_drift, retrain=False)

            if result is None:
                continue

            self.cache.update(result)
            self._save_cache()

            if best_result is None or result.score < best_result.score:
                best_result = result

        return best_result

    def set_validation_status(self, target_drift: float, passed: bool, ratio: float):
        """Update validation status for a target drift and save cache."""
        self.cache.update_validation_status(target_drift, passed, ratio)
        self._save_cache()
        if self.verbose:
            status = "✓ PASSED" if passed else "✗ FAILED"
            print(f"Validation status saved: {status} (ratio={ratio:.2f}x)")

    def generate_with_best_params(
        self,
        target_drift: float,
        use_cache: bool = True,
    ) -> Tuple[bool, str]:
        """
        Generate data using best known parameters.

        Args:
            target_drift: Target drift value
            use_cache: Whether to use cached parameters

        Returns:
            (success, output_path)
        """
        cached = self.cache.get_best_params(target_drift) if use_cache else None

        if cached:
            params = cached.params
            actual_drift = cached.actual_drift
            corr_loss = cached.correlation_loss
            print(f"Using cached params: scale_factor={params.scale_factor}")
        else:
            # Use default params
            params = TuningParams()
            actual_drift = 0.0
            corr_loss = 0.0
            print("Using default params")

        success, output_path = self._run_generation(params, target_drift, retrain=False)

        if success:
            # Save metadata for manual validation
            self._save_last_generation_metadata(params, target_drift, actual_drift, corr_loss)

        return success, output_path


def list_cached_params(dataset_name: Optional[str] = None) -> Dict[str, Dict]:
    """List all cached tuning parameters."""
    cache_dir = AutoTuner.CACHE_DIR
    if not os.path.exists(cache_dir):
        return {}

    results = {}
    for filename in os.listdir(cache_dir):
        if not filename.endswith(".json"):
            continue

        if dataset_name and not filename.startswith(dataset_name):
            continue

        path = os.path.join(cache_dir, filename)
        try:
            with open(path, "r") as f:
                data = json.load(f)
            key = f"{data['dataset_name']}.{data['table_name']}"
            results[key] = {
                "file": filename,
                "drifts": list(data.get("best_params", {}).keys()),
            }
        except Exception:
            continue

    return results


def get_cached_params_for_table(
    dataset_name: str,
    table_name: str,
    target_drift: float,
    reference_dataset: str = None,
) -> Optional[TuningParams]:
    """Get cached parameters for a specific table and drift."""
    tuner = AutoTuner(dataset_name, table_name, reference_dataset=reference_dataset, verbose=False)
    result = tuner.cache.get_best_params(target_drift)
    if result:
        return result.params
    return None


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Auto-tune dbproc parameters")
    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument("--table-name", type=str, required=True)
    parser.add_argument("--target-drift", type=float, required=True)
    parser.add_argument("--target-corr-loss", type=float, default=None,
                        help="Target correlation loss (from reference, e.g., drift_reference.csv)")
    parser.add_argument("--reference-dataset", type=str, default=None,
                        help="Reference dataset for drift direction (default: {dataset_name}_2014)")
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--tolerance", type=float, default=0.20, help="Drift error tolerance (relative, default 20%)")
    parser.add_argument("--no-stop-on-tolerance", action="store_true",
                        help="Allow early stop even if tolerance not met")
    parser.add_argument("--stagnation-limit", type=int, default=5,
                        help="Iterations without improvement before retrain")
    parser.add_argument("--retrain-interval", type=int, default=10,
                        help="Retrain models every N iterations if not improving")
    parser.add_argument("--quick", action="store_true", help="Quick tune (no retraining)")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--no-cache", action="store_true", help="Ignore cached params, force retrain")
    parser.add_argument("--require-validation", action="store_true",
                        help="Only use cache if validation_passed=True")
    parser.add_argument("--list-cache", action="store_true", help="List cached parameters")

    args = parser.parse_args()

    if args.list_cache:
        cached = list_cached_params(args.dataset_name)
        print("Cached parameters:")
        for key, info in cached.items():
            print(f"  {key}: drifts={info['drifts']}")
        sys.exit(0)

    tuner = AutoTuner(
        dataset_name=args.dataset_name,
        table_name=args.table_name,
        reference_dataset=args.reference_dataset,
        device=args.device,
        target_corr_loss=args.target_corr_loss,
    )

    if args.quick:
        result = tuner.quick_tune(args.target_drift, use_cache=not args.no_cache)
    else:
        result = tuner.tune(
            args.target_drift,
            max_iterations=args.max_iterations,
            tolerance=args.tolerance,
            use_cache=not args.no_cache,
            stop_only_on_tolerance=not args.no_stop_on_tolerance,
            stagnation_limit=args.stagnation_limit,
            retrain_interval=args.retrain_interval,
            require_validation=args.require_validation,
        )

    if result:
        print(f"\nFinal result: drift_error={result.drift_error:.4f}, corr_loss={result.correlation_loss:.4f}")

        # Generate data using the best parameters (needed when using cached result)
        output_path = os.path.join("expdir", args.dataset_name, args.table_name, f"{args.table_name}.drifted.csv")
        if not os.path.exists(output_path):
            print(f"\nGenerating data with cached parameters...")
            tuner.generate_with_best_params(args.target_drift, use_cache=True)

        # Exit code 2 means validated cache was used (skip validation)
        if result.validation_passed:
            print("Used validated cache, skip validation")
            sys.exit(2)
