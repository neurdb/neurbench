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
from scipy.spatial.distance import jensenshannon

# Add drift_ddpm to path for imports
sys.path.append("drift_ddpm")

# Tolerance constants
DRIFT_RELATIVE_TOLERANCE = 0.20  # 20% relative error
DRIFT_ABSOLUTE_TOLERANCE = 0.05  # ±0.05 absolute difference
CORR_TOLERANCE = 0.15            # Correlation error tolerance (when target_corr_loss is set)
CORR_TOLERANCE_BASE = 0.10       # Base correlation tolerance (when no reference)
CORR_TOLERANCE_SCALE = 0.4       # Scale factor: tolerance = base + scale * drift


def get_corr_tolerance(target_drift: float, has_target_corr: bool) -> float:
    """Get correlation tolerance, scaled by drift when no reference dataset.

    When has_target_corr=True (have reference): use fixed CORR_TOLERANCE
    When has_target_corr=False (no reference): scale tolerance with drift
        - drift=0.1 → tolerance = 0.10 + 0.04 = 0.14
        - drift=0.3 → tolerance = 0.10 + 0.12 = 0.22
        - drift=0.5 → tolerance = 0.10 + 0.20 = 0.30
    """
    if has_target_corr:
        return CORR_TOLERANCE
    else:
        return CORR_TOLERANCE_BASE + CORR_TOLERANCE_SCALE * target_drift


def is_drift_ok(actual_drift: float, target_drift: float, rel_tol: float = DRIFT_RELATIVE_TOLERANCE, abs_tol: float = DRIFT_ABSOLUTE_TOLERANCE) -> bool:
    """Check if drift meets tolerance: either relative error <= 20% OR absolute diff <= 0.05."""
    if target_drift > 0:
        drift_error = abs(actual_drift - target_drift) / target_drift
    else:
        drift_error = abs(actual_drift)
    abs_diff = abs(actual_drift - target_drift)
    return (drift_error <= rel_tol) or (abs_diff <= abs_tol)


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
    # Three loss weights for controller training
    loss_weight_drift: float = 1.0   # Weight for drift loss
    loss_weight_corr: float = 0.8    # Weight for correlation loss
    loss_weight_real: float = 0.1    # Weight for RealMSE loss

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
            f"--loss-weight-drift={self.loss_weight_drift}",
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
    fallback_type: str = ""  # Type of fallback used: "year_offset", "row_mixing", or "" for normal

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
        """Update cache with new result (only if better)."""
        self.history.append(result)
        key = f"{result.target_drift:.2f}"

        if key not in self.best_params or result.score < self.best_params[key].score:
            self.best_params[key] = result

    def clear(self, target_drift: float):
        """Clear cached result for a specific target_drift (for ops=all mode)."""
        key = f"{target_drift:.2f}"
        if key in self.best_params:
            del self.best_params[key]

    def update_validation_status(self, target_drift: float, passed: bool, ratio: float):
        """Update validation status for a cached result."""
        key = f"{target_drift:.2f}"
        if key in self.best_params:
            self.best_params[key].validation_passed = passed
            self.best_params[key].validation_ratio = ratio


class DataEvaluator:
    """Evaluates generated data quality."""

    CORR_TYPES = ["pearson"]  # Only use pearson to match calc_drift.py

    def __init__(self, dataset_name: str, table_name: str, reference_dataset: str = None):
        self.dataset_name = dataset_name
        self.table_name = table_name
        self.reference_dataset = reference_dataset  # Use reference for correlation comparison
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

    def _load_csv(self, path: str, use_cache: bool = True, **kwargs) -> pd.DataFrame:
        """Load CSV with smart strategy auto-detection (same as calc_drift.py).

        Args:
            path: Path to CSV file
            use_cache: If True, cache the result to avoid repeated parsing warnings.
                       Set to False for files that change between calls (e.g., drifted.csv)
            **kwargs: Additional arguments to pass to pd.read_csv (e.g., usecols)
        """
        import warnings

        if use_cache and path in self._csv_cache:
            return self._csv_cache[path]

        # Use escapechar first for: .drifted.csv files OR imdb dataset (without year)
        use_escapechar_first = path.endswith(".drifted.csv") or "/imdb/" in path or "\\imdb\\" in path

        if use_escapechar_first:
            strategies = [
                {"doublequote": False, "escapechar": "\\"},   # Backslash escaped
                {"doublequote": True},                        # Standard CSV (fallback)
            ]
        else:
            strategies = [
                {"doublequote": True},                        # Standard CSV (most common)
                {"doublequote": False, "escapechar": "\\"},   # Backslash escaped
            ]

        last_df = None
        for strategy in strategies:
            try:
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    df = pd.read_csv(path, low_memory=False, on_bad_lines='warn', **strategy, **kwargs)

                # If no warnings, this strategy works - use it
                if len(w) == 0:
                    if use_cache:
                        self._csv_cache[path] = df
                    return df
                # Otherwise save and try next strategy
                last_df = df
            except Exception:
                continue

        # All strategies had warnings, return last successful one
        if last_df is not None:
            if use_cache:
                self._csv_cache[path] = last_df
            return last_df
        raise RuntimeError(f"Failed to load {path}")

    def evaluate(self, original_path: str, drifted_path: str) -> Tuple[float, float, float, float]:
        """
        Evaluate drifted data quality.

        Returns:
            (mean_drift, mean_corr_loss, abs_corr_base, abs_corr_gen):
            - mean_drift: Mean JS divergence
            - mean_corr_loss: Mean correlation loss (|base - generated|)
            - abs_corr_base: Mean absolute correlation value of base data
            - abs_corr_gen: Mean absolute correlation value of generated data
        """
        # Use calc_drift module for consistent drift calculation with dbproc.py
        from calc_drift import calc_drift as measure_drift, calc_correlation

        # Cache original data (doesn't change), but not drifted (regenerated each iteration)
        original_data = self._load_csv(original_path, use_cache=True)
        drifted_data = self._load_csv(drifted_path, use_cache=False)

        dataset_info = self._load_dataset_info()
        drifted_columns = dataset_info.get("applicable_columns", [])

        # Calculate drift using calc_drift (same as dbproc.py) for consistency
        mean_drift = measure_drift(original_data, drifted_data, drifted_columns, verbose=False)

        # Calculate correlation using calc_correlation (same as dbproc.py) for consistency
        corr_result = calc_correlation(original_data, drifted_data, verbose=False)
        mean_corr_loss = corr_result.get('pearson', 0.0)
        mean_abs_corr_base = corr_result.get('pearson_abs', 0.0)
        mean_abs_corr_gen = corr_result.get('pearson_abs_gen', 0.0)

        return mean_drift, mean_corr_loss, mean_abs_corr_base, mean_abs_corr_gen


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
        num_gpus: int = 1,  # Number of GPUs for parallel batch generation
        sample_start: int = 0,  # For chunked generation
        sample_count: int = -1,  # -1 means all samples
        sample_steps: int = None,  # DDIM sampling steps (None = use diffuser-timesteps)
        force_cache_update: bool = False,  # Force update cache (for ops=all mode)
        skip_freq_preservation: bool = False,  # Skip frequency preservation (for non-drift-ref mode)
        variant_id: int = -1,  # Variant ID for separate output directories
    ):
        self.dataset_name = dataset_name
        self.table_name = table_name
        self.reference_dataset = reference_dataset
        self.device = device
        self.verbose = verbose
        self.target_corr_loss = target_corr_loss  # If set, use as target instead of minimizing
        self.num_gpus = num_gpus
        self.sample_start = sample_start
        self.sample_count = sample_count
        self.sample_steps = sample_steps
        self.force_cache_update = force_cache_update
        self.skip_freq_preservation = skip_freq_preservation
        self.variant_id = variant_id
        self.evaluator = DataEvaluator(dataset_name, table_name, reference_dataset)
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
        abs_corr_base: float = 0.0,
        abs_corr_gen: float = 0.0,
    ):
        """
        Save metadata about the last generation for manual validation.
        This allows 'vd' command to know which cache entry to update.
        """
        # Build path matching dbproc.py logic
        dataset_dir = self.dataset_name
        if self.reference_dataset and self.reference_dataset != self.dataset_name:
            dataset_dir = f"{self.dataset_name}_ref_{self.reference_dataset}"
        if self.variant_id > 0:
            dataset_dir += f"-{self.variant_id}"
        metadata_dir = os.path.join("expdir", dataset_dir, self.table_name)
        os.makedirs(metadata_dir, exist_ok=True)
        metadata_path = os.path.join(metadata_dir, "last_generation.json")

        corr_ratio = correlation_loss / abs_corr_base if abs_corr_base > 0 else 0.0
        metadata = {
            "dataset_name": self.dataset_name,
            "table_name": self.table_name,
            "reference_dataset": self.reference_dataset,
            "target_drift": target_drift,
            "actual_drift": actual_drift,
            "correlation_loss": correlation_loss,
            "abs_corr_base": abs_corr_base,
            "abs_corr_gen": abs_corr_gen,
            "corr_loss_ratio": corr_ratio,
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
        retrain_controller_only: bool = False,
        train_only: bool = False,
    ) -> Tuple[bool, str]:
        """
        Run data generation with given parameters.

        Args:
            retrain: If True, retrain both diffuser and controller
            retrain_controller_only: If True, only retrain controller (used when adjusting weight params)
            train_only: If True, only train models, skip generation

        Returns:
            (success, output_path)
        """
        cmd = f"python3 dbproc.py --dataset-name={self.dataset_name} --table-name={self.table_name}"
        cmd += f" --drift={target_drift} --device={self.device}"
        cmd += f" {params.to_cmd_args()}"

        # Multi-GPU parallel batch generation
        if self.num_gpus > 1:
            cmd += f" --num-gpus={self.num_gpus}"

        # Chunked generation
        if self.sample_start > 0 or self.sample_count > 0:
            cmd += f" --sample-start={self.sample_start} --sample-count={self.sample_count}"

        # DDIM sampling steps
        if self.sample_steps:
            cmd += f" --sample-steps={self.sample_steps}"

        # Tables that need fillna (have NULL values that cause issues)
        FILLNA_TABLES = {"aka_title", "title"}
        if self.table_name in FILLNA_TABLES:
            cmd += " --fillna"

        if self.reference_dataset:
            cmd += f" --reference-dataset={self.reference_dataset}"

        if self.skip_freq_preservation:
            cmd += " --skip-freq-preservation"

        if retrain_controller_only:
            cmd += " --retrain-controller"  # Only retrain controller (weight params don't affect diffuser)
            print("*** RETRAINING CONTROLLER ONLY (weight adjustment) ***")
        elif retrain:
            cmd += " --retrain-diffuser --retrain-controller"
        else:
            cmd += " --reuse"

        if train_only:
            cmd += " --train-only"  # Only train, skip generation

        if self.variant_id > 0:
            cmd += f" --variant-id={self.variant_id}"

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

            # Determine output filename based on chunked mode
            if self.sample_start > 0 or self.sample_count > 0:
                sample_end = self.sample_start + self.sample_count
                output_filename = f"{self.table_name}.drifted.chunk_{self.sample_start}_{sample_end}.csv"
            else:
                output_filename = f"{self.table_name}.drifted.csv"

            # Build output path matching dbproc.py logic
            dataset_dir = self.dataset_name
            if self.reference_dataset and self.reference_dataset != self.dataset_name:
                dataset_dir = f"{self.dataset_name}_ref_{self.reference_dataset}"
            if self.variant_id > 0:
                dataset_dir += f"-{self.variant_id}"
            output_path = os.path.join(
                "expdir", dataset_dir, self.table_name,
                output_filename
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
        retrain_controller_only: bool = False,
    ) -> Tuple[Optional[TuningResult], bool]:
        """Evaluate a set of parameters.

        Returns:
            (result, False): TuningResult and placeholder for compatibility
        """
        success, output_path = self._run_generation(params, target_drift, retrain, retrain_controller_only)

        if not success:
            return None, False

        original_path = os.path.join(
            "datasets", self.dataset_name, f"{self.table_name}.csv"
        )

        actual_drift, corr_loss, abs_corr_base, abs_corr_gen = self.evaluator.evaluate(original_path, output_path)
        corr_ratio = corr_loss / abs_corr_base if abs_corr_base > 0 else 0
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
        self._save_last_generation_metadata(params, target_drift, actual_drift, corr_loss, abs_corr_base, abs_corr_gen)

        if self.verbose:
            print(f"\n--- Evaluation Result ---")
            print(f"Target drift: {target_drift:.4f}")
            print(f"Actual drift: {actual_drift:.4f}")
            print(f"Drift error: {drift_error:.2%}")  # Relative error
            print(f"Correlation: base={abs_corr_base:.4f}, gen={abs_corr_gen:.4f}, loss={corr_loss:.4f}, ratio={corr_ratio:.4f}")
            if self.target_corr_loss is not None:
                corr_error = abs(self.target_corr_loss - corr_loss)  # Absolute error
                print(f"Target corr loss: {self.target_corr_loss:.4f}")
                print(f"Corr error: {corr_error:.4f} (tolerance: 0.10)")
            print(f"Score: {score:.4f}")
            print("-" * 25)

        return result, False

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

    def _is_single_year_column_table(self) -> Tuple[bool, Optional[str]]:
        """Check if this table has only one drifting column and it's a year column.

        Returns:
            (is_single_year, column_name): True and column name if single year column, else (False, None)
        """
        info_path = f"datasets/{self.dataset_name}/dataset_info.json"
        try:
            with open(info_path, "r") as f:
                info = json.load(f)
            table_info = info.get(self.table_name, {})
            if not table_info:
                return False, None

            columns = table_info.get("applicable_columns", [])
            if len(columns) != 1:
                return False, None

            col = columns[0]
            # Check if column name suggests it's a year column
            year_keywords = ["year", "yr", "date"]
            if any(kw in col.lower() for kw in year_keywords):
                return True, col

            return False, None
        except Exception as e:
            print(f"Warning: Could not check table info: {e}")
            return False, None

    def _compute_year_offset(self, year_column: str) -> int:
        """Compute year offset based on source and reference dataset distributions.

        Returns:
            Offset to add to years (e.g., +3 means shift years forward by 3)
        """
        src_path = f"datasets/{self.dataset_name}/{self.table_name}.csv"
        ref_dataset = self.reference_dataset or f"{self.dataset_name}_2017"
        ref_path = f"datasets/{ref_dataset}/{self.table_name}.csv"

        try:
            # Load source data
            src_df = self.evaluator._load_csv(src_path, usecols=[year_column])
            src_years = src_df[year_column].dropna()
            src_years = src_years[src_years > 1800]  # Filter valid years

            # Load reference data
            ref_df = self.evaluator._load_csv(ref_path, usecols=[year_column])
            ref_years = ref_df[year_column].dropna()
            ref_years = ref_years[ref_years > 1800]  # Filter valid years

            # Compute offset based on median difference
            src_median = int(src_years.median())
            ref_median = int(ref_years.median())
            offset = ref_median - src_median

            print(f"Year offset computation:")
            print(f"  Source median year: {src_median}")
            print(f"  Reference median year: {ref_median}")
            print(f"  Computed offset: {offset:+d}")

            return offset
        except Exception as e:
            print(f"Warning: Could not compute year offset: {e}")
            # Default offset based on dataset names (e.g., 2014 -> 2017 = +3)
            return 3

    def _apply_year_offset_fallback(
        self,
        year_column: str,
        target_drift: float,
        tolerance: float = 0.20,
        max_attempts: int = 10,
    ) -> Optional[TuningResult]:
        """Apply year offset fallback for single-column year tables.

        This is used when DDPM fails to control drift for tables with only
        a year column (which has very limited value space).

        Uses adaptive search to find the best offset that achieves target drift.

        Returns:
            TuningResult if successful, None otherwise
        """
        print(f"\n{'='*60}")
        print(f"FALLBACK: Year Offset Method (Adaptive)")
        print(f"{'='*60}")
        print(f"Table {self.table_name} has single year column: {year_column}")
        print(f"DDPM cannot effectively control drift for narrow value space.")
        print(f"Falling back to adaptive year offset method...")
        print()

        # Compute initial offset based on dataset difference
        initial_offset = self._compute_year_offset(year_column)

        # Load original data (from reference dataset, same as dbproc.py)
        ref_dataset = self.reference_dataset or f"{self.dataset_name}_2017"
        ref_path = f"datasets/{ref_dataset}/{self.table_name}.csv"

        try:
            # Load reference data
            try:
                df_original = self.evaluator._load_csv(ref_path)
            except RuntimeError:
                print(f"Failed to load {ref_path}")
                return None

            print(f"Loaded {len(df_original)} rows from {ref_path}")

            original_path = f"datasets/{self.dataset_name}/{self.table_name}.csv"
            # Build path matching dbproc.py logic
            dataset_dir = self.dataset_name
            if self.reference_dataset and self.reference_dataset != self.dataset_name:
                dataset_dir = f"{self.dataset_name}_ref_{self.reference_dataset}"
            if self.variant_id > 0:
                dataset_dir += f"-{self.variant_id}"
            output_dir = os.path.join("expdir", dataset_dir, self.table_name)
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{self.table_name}.drifted.csv")

            # Adaptive search for best offset
            best_result = None
            best_offset = initial_offset

            # Search range: try offsets around the initial estimate
            offset_low = -10
            offset_high = 20
            current_offset = initial_offset

            history = []  # (offset, drift, drift_error)

            for attempt in range(max_attempts):
                print(f"\n[Fallback Attempt {attempt + 1}/{max_attempts}] offset={current_offset:+d}")

                # Apply offset
                df = df_original.copy()
                original_years = df[year_column].copy()
                df[year_column] = df[year_column].apply(
                    lambda x, off=current_offset: x + off if pd.notna(x) and x > 1800 else x
                )

                # Clamp to reasonable range
                max_year = 2025
                df[year_column] = df[year_column].apply(
                    lambda x: min(x, max_year) if pd.notna(x) and x > 1800 else x
                )

                # Save
                df.to_csv(output_path, index=False, doublequote=False, escapechar="\\")

                # Evaluate
                actual_drift, corr_loss, abs_corr_base, abs_corr_gen = self.evaluator.evaluate(original_path, output_path)
                corr_ratio = corr_loss / abs_corr_base if abs_corr_base > 0 else 0
                drift_error = abs(actual_drift - target_drift) / target_drift if target_drift > 0 else abs(actual_drift)

                # Compute score - corr_loss = |base - generated|, compare with target from drift_ref
                if self.target_corr_loss is not None:
                    corr_error = abs(self.target_corr_loss - corr_loss)
                else:
                    corr_error = corr_loss  # Target is 0
                score = drift_error * 2.0 + corr_error * 1.5

                history.append((current_offset, actual_drift, drift_error))

                print(f"  Actual drift: {actual_drift:.4f}, Target: {target_drift:.4f}")
                print(f"  Drift error: {drift_error:.2%}, corr base={abs_corr_base:.4f}, gen={abs_corr_gen:.4f}, loss={corr_loss:.4f}, ratio={corr_ratio:.4f}")

                # Check if we meet tolerance (relative <= 20% OR absolute <= 0.05)
                drift_ok = is_drift_ok(actual_drift, target_drift)
                corr_tol = get_corr_tolerance(target_drift, self.target_corr_loss is not None)
                corr_ok = corr_error <= corr_tol

                result = TuningResult(
                    params=TuningParams(scale_factor=float(current_offset)),  # Store offset in scale_factor field
                    target_drift=target_drift,
                    actual_drift=actual_drift,
                    correlation_loss=corr_loss,
                    drift_error=drift_error,
                    score=score,
                    fallback_type="year_offset",
                )

                if best_result is None or result.score < best_result.score:
                    best_result = result
                    best_offset = current_offset
                    print(f"  New best! Score: {score:.4f}")

                if drift_ok and corr_ok:
                    print(f"\n✓ Fallback SUCCESS! offset={current_offset:+d}")
                    break

                # Adjust offset based on drift difference
                drift_diff = actual_drift - target_drift

                if len(history) >= 2:
                    # Estimate gradient: how much drift changes per unit offset
                    off1, drift1, _ = history[-2]
                    off2, drift2, _ = history[-1]
                    off_change = off2 - off1
                    drift_change = drift2 - drift1

                    if abs(off_change) > 0 and abs(drift_change) > 0.001:
                        gradient = drift_change / off_change
                        drift_needed = target_drift - actual_drift
                        offset_change = int(round(drift_needed / gradient))
                        # Limit change
                        offset_change = max(-5, min(5, offset_change))
                        current_offset = current_offset + offset_change
                    else:
                        # Gradient too small, try larger jump
                        if drift_diff > 0:  # Actual > target, need less offset
                            current_offset -= 2
                        else:
                            current_offset += 2
                else:
                    # First iteration, adjust based on drift difference
                    if drift_diff > 0:  # Actual > target
                        current_offset -= 1
                    else:
                        current_offset += 1

                # Clamp offset to reasonable range
                current_offset = max(offset_low, min(offset_high, current_offset))

            # Final result
            if best_result:
                print(f"\n--- Fallback Best Result ---")
                print(f"Best offset: {best_offset:+d}")
                print(f"Target drift: {target_drift:.4f}")
                print(f"Actual drift: {best_result.actual_drift:.4f}")
                print(f"Drift error: {best_result.drift_error:.2%}")
                print(f"Correlation loss: {best_result.correlation_loss:.4f}")
                if self.target_corr_loss is not None:
                    print(f"Target corr loss: {self.target_corr_loss:.4f}")
                    corr_error = abs(best_result.correlation_loss - self.target_corr_loss)
                    print(f"Corr error: {corr_error:.4f}")
                print(f"Score: {best_result.score:.4f}")
                print("-" * 25)

                # Re-apply best offset and save
                if best_offset != current_offset:
                    df = df_original.copy()
                    df[year_column] = df[year_column].apply(
                        lambda x, off=best_offset: x + off if pd.notna(x) and x > 1800 else x
                    )
                    df[year_column] = df[year_column].apply(
                        lambda x: min(x, 2025) if pd.notna(x) and x > 1800 else x
                    )
                    df.to_csv(output_path, index=False, doublequote=False, escapechar="\\")

                # Save metadata
                self._save_last_generation_metadata(
                    best_result.params, target_drift, best_result.actual_drift, best_result.correlation_loss
                )

            # Check if best result meets tolerance (relative <= 20% OR absolute <= 0.05)
            if best_result:
                drift_ok = is_drift_ok(best_result.actual_drift, target_drift)
                corr_ok = True
                if self.target_corr_loss is not None:
                    corr_error = abs(best_result.correlation_loss - self.target_corr_loss)
                    corr_ok = corr_error <= 0.10

                if not (drift_ok and corr_ok):
                    print(f"\nYear offset fallback: best effort (drift_error={best_result.drift_error:.2%})")

            return best_result

        except Exception as e:
            print(f"Fallback failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def tune(
        self,
        target_drift: float,
        max_iterations: int = 50,
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
            require_validation: Deprecated - cache is now used if it meets tolerance regardless of validation status

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
        print(f"{'='*60}\n")

        # Clear cache for this target_drift if force_cache_update (ops=all mode)
        if self.force_cache_update:
            self.cache.clear(target_drift)
            self._save_cache()
            print(f"Cleared cache for drift={target_drift:.2f} (ops=all mode)")

        # Check cache first
        if use_cache:
            cached = self.cache.get_best_params(target_drift)
            if cached and is_drift_ok(cached.actual_drift, target_drift):
                # Check correlation error
                # correlation_loss = |base - generated|, compare with target_corr_loss from drift_ref
                if self.target_corr_loss is not None:
                    corr_error = abs(cached.correlation_loss - self.target_corr_loss)
                else:
                    corr_error = cached.correlation_loss  # Target is 0, so error = loss itself
                corr_tol = get_corr_tolerance(target_drift, self.target_corr_loss is not None)
                corr_ok = corr_error <= corr_tol

                if corr_ok:
                    # Cache meets tolerance - use it regardless of validation status
                    # In --validate mode, validation will happen later
                    if cached.validation_passed:
                        print(f"Using cached parameters (drift_error={cached.drift_error:.2%}, corr_error={corr_error:.4f}, validated=✓)")
                    else:
                        print(f"Using cached parameters (drift_error={cached.drift_error:.2%}, corr_error={corr_error:.4f}, pending validation)")
                    return cached
                else:
                    print(f"Cache found but corr_error too high ({corr_error:.4f} > {corr_tol:.2f}), re-tuning...")

        best_result = None

        # Adaptive search bounds for scale_factor
        # Lower bound needs to be small enough for low drift targets (e.g., 0.05-0.1)
        sf_low = 0.001
        sf_high = 30.0
        current_sf = self._get_initial_scale_factor(target_drift)

        # Track history for smarter adjustments
        history = []  # List of (scale_factor, actual_drift, drift_error)

        # Stagnation tracking
        iterations_without_improvement = 0
        last_best_score = float('inf')

        # === MODE TRACKING ===
        # "normal": just adjust scale_factor
        # "corr_focus": drift OK but corr bad → strengthen correlation (step-by-step)
        # "drift_focus": corr OK but drift bad → strengthen drift (step-by-step)
        mode = "normal"

        # History tracking
        drift_history = []  # List of {sf, drift, drift_error, corr_loss}

        # Drift insensitivity detection (for entering drift_focus mode)
        consecutive_small_gradient = 0
        gradient_history = []  # Track gradient values for diagnostics

        # === SIMPLIFIED TUNING: Each mode adjusts ONE parameter ===
        # normal: adjust scale_factor
        # corr_focus: adjust loss_weight_corr (increase to improve correlation)
        # drift_focus: adjust loss_weight_drift (increase to improve drift control)
        #
        # Focus mode has two phases:
        # 1. "scan": try candidate values until no improvement for 2 consecutive attempts
        # 2. "bisect": binary search within the best range found

        # Candidate values for focus modes (scan phase)
        CORR_WEIGHT_VALUES = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
        # Drift weight: increase when drift too low, decrease when drift too high
        DRIFT_WEIGHT_UP_VALUES = [2.0, 4.0, 8.0, 16.0, 32.0]    # Increase from default 1.0
        DRIFT_WEIGHT_DOWN_VALUES = [0.5, 0.2, 0.1, 0.05, 0.02]  # Decrease from default 1.0

        # Corr focus state
        corr_focus_phase = "scan"  # "scan" or "bisect"
        corr_focus_attempt = 0
        corr_focus_no_improve = 0
        corr_focus_bisect_low = 0.8
        corr_focus_bisect_high = CORR_WEIGHT_VALUES[-1]  # Start with full range
        best_corr_weight = 0.8
        best_corr_loss_in_focus = float('inf')

        # Drift focus state
        drift_focus_phase = "scan"
        drift_focus_direction = "up"  # "up" (drift too low) or "down" (drift too high)
        drift_focus_attempt = 0
        drift_focus_no_improve = 0
        drift_focus_bisect_low = 1.0
        drift_focus_bisect_high = DRIFT_WEIGHT_UP_VALUES[-1]  # Will be updated based on direction
        best_drift_weight = 1.0
        best_drift_error_in_focus = float('inf')

        # Flag to force retrain on next iteration (set when search range resets)
        force_retrain_next = False

        # Counter to prevent infinite resets
        reset_count = 0
        max_resets = 3  # Maximum number of search range resets allowed

        # Track retrain variants - try different hyperparams each time (one param at a time)
        retrain_variant = 0
        # Different training configurations to try when stuck
        # Each variant changes ONE parameter from default to isolate the effect
        # Default: diffuser_lr=0.0018, controller_lr=0.001, controller_steps=10000
        RETRAIN_CONFIGS = [
            # variant 0: default
            {"diffuser_lr": 0.0018, "controller_lr": 0.001, "controller_steps": 10000, "desc": "default"},
            # variant 1: higher diffuser_lr
            {"diffuser_lr": 0.004, "controller_lr": 0.001, "controller_steps": 10000, "desc": "higher diffuser_lr"},
            # variant 2: lower diffuser_lr
            {"diffuser_lr": 0.0008, "controller_lr": 0.001, "controller_steps": 10000, "desc": "lower diffuser_lr"},
            # variant 3: higher controller_lr
            {"diffuser_lr": 0.0018, "controller_lr": 0.003, "controller_steps": 10000, "desc": "higher controller_lr"},
            # variant 4: more controller_steps
            {"diffuser_lr": 0.0018, "controller_lr": 0.001, "controller_steps": 20000, "desc": "more controller_steps"},
            # variant 5: even more controller_steps
            {"diffuser_lr": 0.0018, "controller_lr": 0.001, "controller_steps": 30000, "desc": "even more controller_steps"},
        ]

        for i in range(max_iterations):
            # === DETERMINE PARAMS BASED ON MODE ===
            # Default values
            ctrl_drift_weight = best_drift_weight
            ctrl_corr_weight = best_corr_weight
            # RealMSE only useful when reference dataset is different from source
            has_reference = self.reference_dataset and self.reference_dataset != self.dataset_name
            ctrl_real_weight = 0.1 if has_reference else 0.0

            # Get current training config based on retrain_variant
            current_config = RETRAIN_CONFIGS[retrain_variant % len(RETRAIN_CONFIGS)]

            if mode == "normal":
                retrain_controller_only = False
                if i == 0 or force_retrain_next:
                    # First iteration or after reset: train with current variant's params
                    should_retrain = True
                    if force_retrain_next:
                        retrain_variant += 1
                        current_config = RETRAIN_CONFIGS[retrain_variant % len(RETRAIN_CONFIGS)]
                        print(f"*** Retrain variant {retrain_variant} ({current_config['desc']}): "
                              f"diffuser_lr={current_config['diffuser_lr']}, "
                              f"controller_lr={current_config['controller_lr']}, "
                              f"controller_steps={current_config['controller_steps']} ***")
                    force_retrain_next = False
                else:
                    # Normal: just adjust scale_factor, no retrain
                    should_retrain = False

            elif mode == "corr_focus":
                # Corr Focus: adjust loss_weight_corr only
                should_retrain = False
                retrain_controller_only = True
                if corr_focus_phase == "scan":
                    ctrl_corr_weight = CORR_WEIGHT_VALUES[min(corr_focus_attempt, len(CORR_WEIGHT_VALUES) - 1)]
                    print(f"*** Corr Focus [scan]: trying weight_corr={ctrl_corr_weight} (attempt {corr_focus_attempt + 1}/{len(CORR_WEIGHT_VALUES)}) ***")
                else:  # bisect
                    ctrl_corr_weight = (corr_focus_bisect_low + corr_focus_bisect_high) / 2
                    print(f"*** Corr Focus [bisect]: trying weight_corr={ctrl_corr_weight:.2f} (range [{corr_focus_bisect_low:.2f}, {corr_focus_bisect_high:.2f}]) ***")

            elif mode == "drift_focus":
                # Drift Focus: adjust loss_weight_drift only
                # Direction: "up" when drift too low, "down" when drift too high
                should_retrain = False
                retrain_controller_only = True
                drift_weight_values = DRIFT_WEIGHT_UP_VALUES if drift_focus_direction == "up" else DRIFT_WEIGHT_DOWN_VALUES
                if drift_focus_phase == "scan":
                    ctrl_drift_weight = drift_weight_values[min(drift_focus_attempt, len(drift_weight_values) - 1)]
                    dir_str = "↑" if drift_focus_direction == "up" else "↓"
                    print(f"*** Drift Focus [scan {dir_str}]: trying weight_drift={ctrl_drift_weight} (attempt {drift_focus_attempt + 1}/{len(drift_weight_values)}) ***")
                else:  # bisect
                    ctrl_drift_weight = (drift_focus_bisect_low + drift_focus_bisect_high) / 2
                    dir_str = "↑" if drift_focus_direction == "up" else "↓"
                    print(f"*** Drift Focus [bisect {dir_str}]: trying weight_drift={ctrl_drift_weight:.4f} (range [{drift_focus_bisect_low:.4f}, {drift_focus_bisect_high:.4f}]) ***")

            params = TuningParams(
                scale_factor=current_sf,
                loss_weight_drift=ctrl_drift_weight,
                loss_weight_corr=ctrl_corr_weight,
                loss_weight_real=ctrl_real_weight,
                # Use variant-specific training params when retraining
                diffuser_lr=current_config["diffuser_lr"],
                controller_lr=current_config["controller_lr"],
                controller_steps=current_config["controller_steps"],
            )

            print(f"\n[Iteration {i+1}/{max_iterations}] mode={mode}")
            train_mode = "controller_only" if retrain_controller_only else ("full" if should_retrain else "none")
            print(f"Params: sf={current_sf:.4f}, drift_w={ctrl_drift_weight:.2f}, "
                  f"corr_w={ctrl_corr_weight:.2f}, real_w={ctrl_real_weight:.2f}, train={train_mode}")
            if should_retrain:
                print(f"Training config ({current_config['desc']}): "
                      f"diffuser_lr={current_config['diffuser_lr']}, "
                      f"controller_lr={current_config['controller_lr']}, "
                      f"controller_steps={current_config['controller_steps']}")

            result, _ = self._evaluate_params(params, target_drift, retrain=should_retrain, retrain_controller_only=retrain_controller_only)

            if result is None:
                print("Evaluation failed, trying different scale_factor...")
                current_sf = (sf_low + sf_high) / 2
                iterations_without_improvement += 1
                continue

            # Track history
            history.append((current_sf, result.actual_drift, result.drift_error))
            drift_history.append({
                'sf': current_sf,
                'drift': result.actual_drift,
                'drift_error': result.drift_error,
                'corr_loss': result.correlation_loss,
            })

            # Update cache (don't force during iteration)
            self.cache.update(result)
            self._save_cache()

            # Update best result
            if best_result is None or result.score < best_result.score:
                best_result = result
                print(f"New best! Score: {result.score:.4f}, Drift error: {result.drift_error:.4f}")
                iterations_without_improvement = 0
            else:
                iterations_without_improvement += 1

            # === EVALUATE RESULT AND DETERMINE MODE ===
            # Drift OK if relative error <= 20% OR absolute diff <= 0.05
            drift_ok = is_drift_ok(result.actual_drift, target_drift)
            # correlation_loss = |base - generated|, compare with target_corr_loss from drift_ref
            if self.target_corr_loss is not None:
                corr_error = abs(result.correlation_loss - self.target_corr_loss)
            else:
                corr_error = result.correlation_loss  # Target is 0
            corr_tol = get_corr_tolerance(target_drift, self.target_corr_loss is not None)
            corr_ok = corr_error <= corr_tol

            print(f"Result: drift_error={result.drift_error:.2%} ({'OK' if drift_ok else 'BAD'}), "
                  f"corr_error={corr_error:.4f} ({'OK' if corr_ok else 'BAD'})")

            # === TRACK BEST IN FOCUS MODES ===
            if mode == "corr_focus":
                prev_best = best_corr_loss_in_focus
                if result.correlation_loss < best_corr_loss_in_focus:
                    # Record the previous best weight before updating
                    prev_best_weight = best_corr_weight
                    best_corr_loss_in_focus = result.correlation_loss
                    best_corr_weight = ctrl_corr_weight
                    corr_focus_no_improve = 0
                    # Update bisect range: best is between prev and current
                    if corr_focus_phase == "scan" and corr_focus_attempt > 0:
                        corr_focus_bisect_low = prev_best_weight
                        corr_focus_bisect_high = ctrl_corr_weight
                    print(f"  → New best corr_weight={best_corr_weight:.2f} (corr_loss={result.correlation_loss:.4f})")
                else:
                    corr_focus_no_improve += 1
                    print(f"  → No improvement (corr_loss={result.correlation_loss:.4f}, best={best_corr_loss_in_focus:.4f}, no_improve={corr_focus_no_improve})")
                    if corr_focus_phase == "bisect":
                        # Bisect: narrow range based on which half is better
                        mid = (corr_focus_bisect_low + corr_focus_bisect_high) / 2
                        if best_corr_weight < mid:
                            corr_focus_bisect_high = mid
                        else:
                            corr_focus_bisect_low = mid

            elif mode == "drift_focus":
                prev_best = best_drift_error_in_focus
                if result.drift_error < best_drift_error_in_focus:
                    prev_best_weight = best_drift_weight
                    best_drift_error_in_focus = result.drift_error
                    best_drift_weight = ctrl_drift_weight
                    drift_focus_no_improve = 0
                    if drift_focus_phase == "scan" and drift_focus_attempt > 0:
                        # Ensure bisect_low < bisect_high regardless of direction
                        if drift_focus_direction == "up":
                            drift_focus_bisect_low = prev_best_weight
                            drift_focus_bisect_high = ctrl_drift_weight
                        else:  # down direction: values decrease
                            drift_focus_bisect_low = ctrl_drift_weight
                            drift_focus_bisect_high = prev_best_weight
                    print(f"  → New best drift_weight={best_drift_weight:.4f} (drift_error={result.drift_error:.2%})")
                else:
                    drift_focus_no_improve += 1
                    print(f"  → No improvement (drift_error={result.drift_error:.2%}, best={best_drift_error_in_focus:.2%}, no_improve={drift_focus_no_improve})")
                    if drift_focus_phase == "bisect":
                        mid = (drift_focus_bisect_low + drift_focus_bisect_high) / 2
                        if best_drift_weight < mid:
                            drift_focus_bisect_high = mid
                        else:
                            drift_focus_bisect_low = mid

            # === SUCCESS ===
            if drift_ok and corr_ok:
                print(f"\n✓ SUCCESS! Both drift and correlation within tolerance!")
                break

            # === MODE TRANSITIONS ===
            if drift_ok and not corr_ok:
                # Drift OK, correlation bad → Corr Focus Mode
                if mode == "normal":
                    print(f"\n→ Drift OK but correlation bad, entering Corr Focus Mode")
                    mode = "corr_focus"
                    corr_focus_phase = "scan"
                    corr_focus_attempt = 0
                    corr_focus_no_improve = 0
                    corr_focus_bisect_low = 0.8
                    corr_focus_bisect_high = CORR_WEIGHT_VALUES[-1]  # Max candidate value
                    best_corr_loss_in_focus = result.correlation_loss
                    best_corr_weight = 0.8
                elif mode == "drift_focus":
                    print(f"\n✓ Drift Focus succeeded! Switching to Corr Focus Mode")
                    mode = "corr_focus"
                    corr_focus_phase = "scan"
                    corr_focus_attempt = 0
                    corr_focus_no_improve = 0
                    corr_focus_bisect_low = 0.8
                    corr_focus_bisect_high = CORR_WEIGHT_VALUES[-1]
                    best_corr_loss_in_focus = result.correlation_loss
                    best_corr_weight = 0.8
                elif mode == "corr_focus":
                    if corr_focus_phase == "scan":
                        # Check if should switch to bisect or continue scan
                        if corr_focus_no_improve >= 2 or corr_focus_attempt >= len(CORR_WEIGHT_VALUES) - 1:
                            if corr_focus_bisect_high - corr_focus_bisect_low > 0.5:
                                print(f"\n→ Scan done, switching to bisect in range [{corr_focus_bisect_low:.2f}, {corr_focus_bisect_high:.2f}]")
                                corr_focus_phase = "bisect"
                                corr_focus_no_improve = 0
                            else:
                                print(f"\n✗ GIVING UP: Corr focus exhausted")
                                is_year_table, year_col = self._is_single_year_column_table()
                                if is_year_table and year_col:
                                    fallback_result = self._apply_year_offset_fallback(year_col, target_drift)
                                    if fallback_result and (best_result is None or fallback_result.score < best_result.score):
                                        best_result = fallback_result
                                break
                        else:
                            corr_focus_attempt += 1
                    else:  # bisect phase
                        if corr_focus_bisect_high - corr_focus_bisect_low < 0.5 or corr_focus_no_improve >= 3:
                            print(f"\n✗ GIVING UP: Bisect converged at corr_weight={best_corr_weight:.2f}")
                            break
                continue

            elif not drift_ok and corr_ok:
                # Correlation OK, drift bad
                if mode == "normal":
                    if consecutive_small_gradient >= 3:
                        # Determine direction: drift too high → decrease weight, drift too low → increase weight
                        drift_too_high = result.actual_drift > target_drift
                        drift_focus_direction = "down" if drift_too_high else "up"
                        dir_str = "↓ (drift too high)" if drift_too_high else "↑ (drift too low)"
                        print(f"\n→ Corr OK but drift not responding to sf, entering Drift Focus Mode {dir_str}")
                        mode = "drift_focus"
                        drift_focus_phase = "scan"
                        drift_focus_attempt = 0
                        drift_focus_no_improve = 0
                        # Set bisect range based on direction
                        if drift_focus_direction == "up":
                            drift_focus_bisect_low = 1.0
                            drift_focus_bisect_high = DRIFT_WEIGHT_UP_VALUES[-1]
                        else:
                            drift_focus_bisect_low = DRIFT_WEIGHT_DOWN_VALUES[-1]
                            drift_focus_bisect_high = 1.0
                        best_drift_error_in_focus = result.drift_error
                        best_drift_weight = 1.0
                        consecutive_small_gradient = 0
                        sf_low, sf_high = 0.001, 30.0
                        current_sf = self._get_initial_scale_factor(target_drift)
                        continue
                elif mode == "corr_focus":
                    print(f"\n✓ Corr Focus succeeded! Returning to normal mode for sf adjustment")
                    mode = "normal"
                    consecutive_small_gradient = 0
                    sf_low, sf_high = 0.001, 30.0
                    current_sf = self._get_initial_scale_factor(target_drift)
                elif mode == "drift_focus":
                    drift_weight_values = DRIFT_WEIGHT_UP_VALUES if drift_focus_direction == "up" else DRIFT_WEIGHT_DOWN_VALUES
                    # Threshold for bisect: use relative threshold for small values
                    bisect_threshold = 0.5 if drift_focus_direction == "up" else 0.05
                    if drift_focus_phase == "scan":
                        if drift_focus_no_improve >= 2 or drift_focus_attempt >= len(drift_weight_values) - 1:
                            bisect_range = abs(drift_focus_bisect_high - drift_focus_bisect_low)
                            if bisect_range > bisect_threshold:
                                print(f"\n→ Scan done, switching to bisect in range [{drift_focus_bisect_low:.4f}, {drift_focus_bisect_high:.4f}]")
                                drift_focus_phase = "bisect"
                                drift_focus_no_improve = 0
                            else:
                                print(f"\n✗ GIVING UP: Drift focus exhausted")
                                is_year_table, year_col = self._is_single_year_column_table()
                                if is_year_table and year_col:
                                    fallback_result = self._apply_year_offset_fallback(year_col, target_drift)
                                    if fallback_result and (best_result is None or fallback_result.score < best_result.score):
                                        best_result = fallback_result
                                break
                        else:
                            drift_focus_attempt += 1
                    else:  # bisect phase
                        bisect_range = abs(drift_focus_bisect_high - drift_focus_bisect_low)
                        if bisect_range < bisect_threshold or drift_focus_no_improve >= 3:
                            print(f"\n✗ GIVING UP: Bisect converged at drift_weight={best_drift_weight:.4f}")
                            break
                    continue

            elif not drift_ok and not corr_ok:
                # Both bad - return to normal mode
                if mode != "normal":
                    print(f"Warning: Was in {mode} but both are bad, returning to normal")
                    mode = "normal"
                    consecutive_small_gradient = 0
                    sf_low, sf_high = 0.001, 30.0
                    current_sf = self._get_initial_scale_factor(target_drift)

            # === ADJUST SCALE_FACTOR (only in normal mode or when drift is bad) ===
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
                        consecutive_small_gradient = 0

                        # Use gradient to calculate needed scale_factor change
                        drift_needed = target_drift - result.actual_drift
                        sf_change_needed = drift_needed / gradient

                        # Limit the change to avoid wild jumps
                        # Ensure minimum max_change so we can still move when sf is very small
                        max_change = max(0.05, current_sf * 1.5)  # Max 150% change, but at least 0.05
                        sf_change_needed = max(-max_change, min(max_change, sf_change_needed))

                        new_sf = current_sf + sf_change_needed
                        new_sf = max(sf_low, min(sf_high, new_sf))  # Use sf_low/sf_high as bounds

                        print(f"Gradient-based: sf_change_needed={sf_change_needed:+.4f} -> new_sf={new_sf:.4f}")

                        # Detect if stuck at lower bound with drift still too high
                        if new_sf <= sf_low * 1.01 and current_sf <= sf_low * 1.01 and drift_diff > 0:
                            consecutive_small_gradient += 1
                            print(f"  ⚠️ Stuck at lower bound (sf={sf_low:.4f}), drift still too high")
                            if consecutive_small_gradient >= 3:
                                print(f"  → Triggering full retrain in next iteration")
                                force_retrain_next = True
                                consecutive_small_gradient = 0  # Reset counter after triggering
                        else:
                            if drift_diff > 0:
                                sf_high = current_sf
                            else:
                                sf_low = current_sf

                        current_sf = new_sf
                    else:
                        # Gradient too small - table is insensitive to scale_factor
                        consecutive_small_gradient += 1
                        gradient_history.append(gradient)
                        print(f"Gradient too small ({gradient:.6f}), consecutive_small_gradient={consecutive_small_gradient}")

                        # Analyze history for adaptive sf range adjustment
                        if len(drift_history) >= 3:
                            recent_drifts = [h['drift'] for h in drift_history[-6:]]
                            drift_mean = np.mean(recent_drifts)
                            drift_std = np.std(recent_drifts)

                            # Pattern: Drift stuck low -> try larger scale_factor
                            if drift_mean < 0.1 and drift_std < 0.02:
                                print(f"  → Pattern: Drift stuck low ({drift_mean:.2%}), expanding sf range")
                                sf_high = 50.0
                                current_sf = max(current_sf * 2.0, 15.0)
                            # Pattern: Drift stuck high -> try smaller scale_factor
                            elif drift_mean > 0.7 and drift_std < 0.02:
                                print(f"  → Pattern: Drift stuck high ({drift_mean:.2%}), trying smaller sf")
                                sf_low = 0.001
                                current_sf = min(current_sf * 0.3, 1.0)
                            else:
                                # Default: try larger jump in direction needed
                                if drift_diff > 0:
                                    sf_high = current_sf
                                    current_sf = current_sf * 0.5
                                else:
                                    sf_low = current_sf
                                    current_sf = current_sf * 2.0
                        else:
                            # Not enough history, use default jump
                            if drift_diff > 0:
                                sf_high = current_sf
                                current_sf = current_sf * 0.5
                            else:
                                sf_low = current_sf
                                current_sf = current_sf * 2.0

                        # If gradient is small for too long, trigger retrain with different params
                        if consecutive_small_gradient >= 5:
                            print(f"  → Gradient stuck for {consecutive_small_gradient} iterations, triggering retrain")
                            force_retrain_next = True
                            consecutive_small_gradient = 0
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
                print(f"Search range converged: [{sf_low:.4f}, {sf_high:.4f}]")
                if not stop_only_on_tolerance:
                    print("Stopping due to convergence (stop_only_on_tolerance=False)")
                    break
                else:
                    reset_count += 1
                    if reset_count > max_resets:
                        print(f"Max resets ({max_resets}) reached, stopping search")
                        break
                    # Reset and try with different parameters
                    # Use different multipliers for each reset to explore different regions
                    reset_multipliers = [1.5, 0.5, 2.5, 0.25]
                    multiplier = reset_multipliers[min(reset_count - 1, len(reset_multipliers) - 1)]
                    print(f"Resetting search bounds (reset {reset_count}/{max_resets}) with multiplier {multiplier}...")
                    sf_low = 0.001
                    sf_high = 30.0
                    current_sf = self._get_initial_scale_factor(target_drift) * multiplier
                    iterations_without_improvement = 0
                    force_retrain_next = True  # Ensure next iteration does full retrain

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
        retrain: bool = False,
        train_only: bool = False,
    ) -> Tuple[bool, str]:
        """
        Generate data using best known parameters.

        Args:
            target_drift: Target drift value
            use_cache: Whether to use cached parameters
            retrain: If True, force retrain with cached/default params (for ops=retrain mode)
            train_only: If True, only train models, skip generation

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

        success, output_path = self._run_generation(params, target_drift, retrain=retrain, train_only=train_only)

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
    parser.add_argument("--max-iterations", type=int, default=50)
    parser.add_argument("--tolerance", type=float, default=0.20, help="Drift error tolerance (relative, default 20%)")
    parser.add_argument("--no-stop-on-tolerance", action="store_true",
                        help="Allow early stop even if tolerance not met")
    parser.add_argument("--stagnation-limit", type=int, default=5,
                        help="Iterations without improvement before retrain")
    parser.add_argument("--retrain-interval", type=int, default=10,
                        help="Retrain models every N iterations if not improving")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--num-gpus", type=int, default=1,
                        help="Number of GPUs for parallel batch generation")
    parser.add_argument("--no-cache", action="store_true", help="Ignore cached params, force retrain")
    parser.add_argument("--force-regenerate", action="store_true",
                        help="Force regenerate using cached params (no skip), train only if no cache")
    parser.add_argument("--require-validation", action="store_true",
                        help="Only use cache if validation_passed=True")
    parser.add_argument("--list-cache", action="store_true", help="List cached parameters")
    parser.add_argument("--sample-start", type=int, default=0,
                        help="Start index for sample generation (for chunked generation)")
    parser.add_argument("--sample-count", type=int, default=-1,
                        help="Number of samples to generate (-1 for all)")
    parser.add_argument("--sample-steps", type=int, default=None,
                        help="Number of DDIM sampling steps (default: use diffuser-timesteps)")
    parser.add_argument("--skip-freq-preservation", action="store_true", default=False,
                        help="Skip frequency preservation (for non-drift-ref mode)")
    parser.add_argument("--retrain-only", action="store_true", default=False,
                        help="Use cached/default params and force retrain (no tuning, for ops=retrain)")
    parser.add_argument("--variant-id", type=int, default=-1,
                        help="Variant ID for creating separate output directories")

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
        num_gpus=args.num_gpus,
        sample_start=args.sample_start,
        sample_count=args.sample_count,
        sample_steps=args.sample_steps,
        skip_freq_preservation=args.skip_freq_preservation,
        variant_id=args.variant_id,
    )

    # --retrain-only: use cached/default params and force retrain (no tuning)
    if args.retrain_only:
        print(f"[--retrain-only] Using cached/default params, force retrain, no tuning...")
        success, output_path = tuner.generate_with_best_params(args.target_drift, use_cache=True, retrain=True)
        if success:
            print(f"Retrain completed: {output_path}")
            sys.exit(0)
        else:
            print("Retrain failed")
            sys.exit(1)

    # --force-regenerate: use cached params to regenerate, train only if no cache
    if args.force_regenerate:
        cached = tuner.cache.get_best_params(args.target_drift)
        if cached:
            print(f"Using cached params (drift_error={cached.drift_error:.2%}) to force regenerate...")
            success, output_path = tuner.generate_with_best_params(args.target_drift, use_cache=True)
            if success:
                print(f"Force regenerate completed: {output_path}")
                sys.exit(0)  # Don't skip validation
            else:
                print("Force regenerate failed")
                sys.exit(1)
        else:
            print("No cached params found, will train...")
            # Fall through to normal tune

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

        # Always ensure data is generated with best params
        # The last iteration might not be the best one, so we need to regenerate
        # Build path matching dbproc.py logic
        dataset_dir = args.dataset_name
        if args.reference_dataset and args.reference_dataset != args.dataset_name:
            dataset_dir = f"{args.dataset_name}_ref_{args.reference_dataset}"
        if args.variant_id > 0:
            dataset_dir += f"-{args.variant_id}"
        metadata_path = os.path.join("expdir", dataset_dir, args.table_name, "last_generation.json")
        need_regenerate = True

        if os.path.exists(metadata_path):
            try:
                with open(metadata_path) as f:
                    metadata = json.load(f)
                last_sf = metadata.get("scale_factor")
                best_sf = result.params.scale_factor
                if last_sf is not None and abs(last_sf - best_sf) < 0.01:
                    print(f"Last generation used best params (scale_factor={best_sf:.4f}), skipping regeneration")
                    need_regenerate = False
                else:
                    print(f"Last generation used scale_factor={last_sf}, best is {best_sf:.4f}, regenerating...")
            except Exception as e:
                print(f"Could not read metadata: {e}, will regenerate")

        if need_regenerate:
            print(f"\nGenerating data with best parameters...")
            tuner.generate_with_best_params(args.target_drift, use_cache=True)

        # Exit code 2 means validated cache was used (skip validation)
        if result.validation_passed:
            print("Used validated cache, skip validation")
            sys.exit(2)
