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

# Import from nrbench for find_q fallback
try:
    from nrbench.drift import find_q
    from nrbench.sample import sample_from_distribution
    from nrbench import dist as nrbench_dist
    NRBENCH_AVAILABLE = True
except ImportError:
    NRBENCH_AVAILABLE = False


# Tolerance constants
DRIFT_RELATIVE_TOLERANCE = 0.20  # 20% relative error
DRIFT_ABSOLUTE_TOLERANCE = 0.05  # ±0.05 absolute difference
CORR_TOLERANCE = 0.15            # Correlation error tolerance


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
    # Balanced defaults: Drift~0.005, PCorr~0.001, RealMSE~0.65
    # weight_corr=5 -> PCorr*5 ≈ 0.005, weight_real=0.008 -> RealMSE*0.008 ≈ 0.005
    loss_weight_corr: float = 5.0
    loss_weight_real: float = 0.008

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
    fallback_type: str = ""  # Type of fallback used: "year_offset", "find_q", "row_mixing", or "" for normal

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

        # Calculate correlation loss - compare BASE (original) vs GENERATED (drifted)
        # This measures how much correlation changed from base to generated
        # Then we compare this with target (base vs ref) from drift_ref.csv
        corr_losses = []
        abs_corrs_base = []
        abs_corrs_gen = []

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

                    # Calculate absolute correlation values (excluding diagonal)
                    orig_no_diag = original_corr.copy()
                    drift_no_diag = drifted_corr.copy()
                    np.fill_diagonal(orig_no_diag.values, 0)
                    np.fill_diagonal(drift_no_diag.values, 0)
                    abs_corr_base = orig_no_diag.abs().mean().mean()
                    abs_corr_gen = drift_no_diag.abs().mean().mean()
                    if not np.isnan(abs_corr_base):
                        abs_corrs_base.append(abs_corr_base)
                    if not np.isnan(abs_corr_gen):
                        abs_corrs_gen.append(abs_corr_gen)
            except Exception as e:
                print(f"Warning: Error computing {corr_type} correlation: {e}")
                continue

        mean_corr_loss = np.mean(corr_losses) if corr_losses else 0.0
        mean_abs_corr_base = np.mean(abs_corrs_base) if abs_corrs_base else 0.0
        mean_abs_corr_gen = np.mean(abs_corrs_gen) if abs_corrs_gen else 0.0

        return mean_drift, mean_corr_loss, mean_abs_corr_base, mean_abs_corr_gen

    def detect_mode_collapse(self, original_path: str, drifted_path: str, threshold: float = 0.1) -> Tuple[bool, List[str]]:
        """
        Detect mode collapse in generated data.

        Mode collapse is detected when:
        1. A column has much fewer unique values than expected
        2. A single value dominates the column (>90% of rows)

        Args:
            original_path: Path to original CSV
            drifted_path: Path to drifted CSV
            threshold: Ratio threshold for unique values (default 0.1 = 10%)

        Returns:
            (is_collapsed, collapsed_columns): Whether collapse detected and which columns
        """
        original_data = self._load_csv(original_path, use_cache=True)
        drifted_data = self._load_csv(drifted_path, use_cache=False)

        dataset_info = self._load_dataset_info()
        applicable_columns = dataset_info.get("applicable_columns", [])

        collapsed_columns = []

        for col in applicable_columns:
            if col not in original_data.columns or col not in drifted_data.columns:
                continue

            orig_nunique = original_data[col].nunique()
            drift_nunique = drifted_data[col].nunique()

            if orig_nunique == 0:
                continue

            # Check 1: Unique value ratio dropped significantly
            ratio = drift_nunique / orig_nunique
            if ratio < threshold:
                collapsed_columns.append(f"{col} (unique: {drift_nunique}/{orig_nunique}, ratio={ratio:.2%})")
                continue

            # Check 2: Single value dominates (>90%)
            value_counts = drifted_data[col].value_counts(normalize=True)
            if len(value_counts) > 0 and value_counts.iloc[0] > 0.9:
                dominant_val = value_counts.index[0]
                dominant_pct = value_counts.iloc[0]
                collapsed_columns.append(f"{col} (dominant: {dominant_val} @ {dominant_pct:.1%})")

        is_collapsed = len(collapsed_columns) > 0
        return is_collapsed, collapsed_columns


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
        metadata_dir = os.path.join("expdir", self.dataset_name, self.table_name)
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
    ) -> Tuple[bool, str]:
        """
        Run data generation with given parameters.

        Args:
            retrain: If True, retrain both diffuser and controller
            retrain_controller_only: If True, only retrain controller (used when adjusting weight params)

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

        if retrain_controller_only:
            cmd += " --retrain-controller"  # Only retrain controller (weight params don't affect diffuser)
            print("*** RETRAINING CONTROLLER ONLY (weight adjustment) ***")
        elif retrain:
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

            # Determine output filename based on chunked mode
            if self.sample_start > 0 or self.sample_count > 0:
                sample_end = self.sample_start + self.sample_count
                output_filename = f"{self.table_name}.drifted.chunk_{self.sample_start}_{sample_end}.csv"
            else:
                output_filename = f"{self.table_name}.drifted.csv"

            output_path = os.path.join(
                "expdir", self.dataset_name, self.table_name,
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
        check_mode_collapse: bool = True,
    ) -> Tuple[Optional[TuningResult], bool]:
        """Evaluate a set of parameters.

        Returns:
            (result, mode_collapsed): TuningResult and whether mode collapse was detected
        """
        success, output_path = self._run_generation(params, target_drift, retrain, retrain_controller_only)

        if not success:
            return None, False

        original_path = os.path.join(
            "datasets", self.dataset_name, f"{self.table_name}.csv"
        )

        # Check for mode collapse first
        mode_collapsed = False
        if check_mode_collapse:
            is_collapsed, collapsed_cols = self.evaluator.detect_mode_collapse(original_path, output_path)
            if is_collapsed:
                mode_collapsed = True
                print(f"\n⚠️  MODE COLLAPSE DETECTED!")
                for col_info in collapsed_cols:
                    print(f"    - {col_info}")

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
            print(f"Correlation loss: {corr_loss:.4f}, abs_corr(base)={abs_corr_base:.4f}, abs_corr(gen)={abs_corr_gen:.4f}, loss/abs_ratio={corr_ratio:.4f}")
            if self.target_corr_loss is not None:
                corr_error = abs(self.target_corr_loss - corr_loss)  # Absolute error
                print(f"Target corr loss: {self.target_corr_loss:.4f}")
                print(f"Corr error: {corr_error:.4f} (tolerance: 0.10)")
            print(f"Score: {score:.4f}")
            if mode_collapsed:
                print(f"⚠️  Mode collapse detected - results may be unreliable")
            print("-" * 25)

        return result, mode_collapsed

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

    def _is_all_id_columns_table(self) -> Tuple[bool, List[str]]:
        """Check if all applicable columns are ID columns (foreign keys).

        ID columns are problematic for DDPM because:
        1. They're discrete identifiers, not continuous values
        2. They have very large value ranges (millions)
        3. There's no natural "drift" direction

        Returns:
            (is_all_id, columns): True and list of columns if all are IDs, else (False, [])
        """
        info_path = f"datasets/{self.dataset_name}/dataset_info.json"
        try:
            with open(info_path, "r") as f:
                info = json.load(f)
            table_info = info.get(self.table_name, {})
            if not table_info:
                return False, []

            columns = table_info.get("applicable_columns", [])
            if not columns:
                return False, []

            # Check if all columns look like ID columns
            id_patterns = ["_id", "id"]
            all_ids = True
            for col in columns:
                col_lower = col.lower()
                is_id = any(col_lower.endswith(p) or col_lower == "id" for p in id_patterns)
                if not is_id:
                    all_ids = False
                    break

            if all_ids and len(columns) >= 2:
                return True, columns
            return False, []
        except Exception as e:
            print(f"Warning: Could not check table info: {e}")
            return False, []

    def _get_conservative_id_params(self) -> TuningParams:
        """Get conservative parameters for all-ID tables.

        These tables are prone to mode collapse with aggressive parameters.
        Use very low scale_factor and correlation weight.
        """
        return TuningParams(
            diffuser_lr=0.0018,
            diffuser_steps=30000,
            diffuser_bs=2048,
            diffuser_timesteps=1000,
            controller_lr=0.0005,
            controller_steps=15000,
            controller_bs=1024,
            controller_dim=(512, 512),
            scale_factor=1.5,  # Very low to prevent mode collapse
            lambda_p=1.0,
            lambda_s=1.0,
            drift_range_min=0.05,
            drift_range_max=0.50,
            loss_weight_corr=0.5,  # Low to preserve diversity
            loss_weight_real=0.2,
        )

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
            src_df = self._load_csv(src_path, usecols=[year_column])
            src_years = src_df[year_column].dropna()
            src_years = src_years[src_years > 1800]  # Filter valid years

            # Load reference data
            ref_df = self._load_csv(ref_path, usecols=[year_column])
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
                df_original = self._load_csv(ref_path)
            except RuntimeError:
                print(f"Failed to load {ref_path}")
                return None

            print(f"Loaded {len(df_original)} rows from {ref_path}")

            original_path = f"datasets/{self.dataset_name}/{self.table_name}.csv"
            output_dir = os.path.join("expdir", self.dataset_name, self.table_name)
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
                print(f"  Drift error: {drift_error:.2%}, Corr loss: {corr_loss:.4f}, abs_corr(base)={abs_corr_base:.4f}, loss/abs_ratio={corr_ratio:.4f}")

                # Check if we meet tolerance (relative <= 20% OR absolute <= 0.05)
                drift_ok = is_drift_ok(actual_drift, target_drift)
                corr_ok = corr_error <= CORR_TOLERANCE

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
                    print(f"\nYear offset fallback didn't meet tolerance (drift_error={best_result.drift_error:.2%})")
                    print("Trying find_q distribution-based fallback...")
                    find_q_result = self._apply_find_q_fallback(
                        year_column, target_drift, tolerance
                    )
                    if find_q_result and find_q_result.drift_error < best_result.drift_error:
                        print("find_q fallback achieved better result!")
                        return find_q_result
                    else:
                        print("Keeping year offset result (better or find_q failed)")

            return best_result

        except Exception as e:
            print(f"Fallback failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _apply_find_q_fallback(
        self,
        year_column: str,
        target_drift: float,
        tolerance: float = 0.20,
    ) -> Optional[TuningResult]:
        """Apply find_q distribution-based fallback.

        Uses scipy.optimize to find a target distribution q where
        jensenshannon(p, q) = target_drift, then samples from q.

        This is an alternative to year offset when more precise drift control is needed.
        """
        if not NRBENCH_AVAILABLE:
            print("nrbench not available, skipping find_q fallback")
            return None

        print(f"\n{'='*60}")
        print(f"FALLBACK: find_q Distribution Method")
        print(f"{'='*60}")

        try:
            # Load original data
            ref_dataset = self.reference_dataset or f"{self.dataset_name}_2017"
            ref_path = f"datasets/{ref_dataset}/{self.table_name}.csv"
            original_path = f"datasets/{self.dataset_name}/{self.table_name}.csv"
            output_dir = os.path.join("expdir", self.dataset_name, self.table_name)
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{self.table_name}.drifted.csv")

            # Load reference data
            try:
                df = self._load_csv(ref_path)
            except RuntimeError:
                print(f"Failed to load {ref_path}")
                return None

            print(f"Loaded {len(df)} rows from {ref_path}")

            # Get year column series
            series = df[year_column].dropna()
            series = series[series > 1800]  # Filter valid years

            # Compute distribution using categorical dist (each year is a category)
            year_counts = series.value_counts().sort_index()
            p = (year_counts / year_counts.sum()).values
            values = year_counts.index.values

            print(f"Year range: {values.min()} - {values.max()} ({len(values)} unique years)")
            print(f"Finding target distribution q for drift={target_drift}...")

            # Find target distribution
            q = find_q(p, target_drift, skewed=True)

            # Verify achieved drift
            achieved_drift = jensenshannon(p, q)
            print(f"Achieved JS divergence: {achieved_drift:.4f}")

            # Sample from target distribution
            num_samples = len(df)
            sampled_values, _ = sample_from_distribution(list(q), list(values), num_samples)

            # Replace year column with sampled values
            # Only replace valid years, keep NaN and invalid years as-is
            df_new = df.copy()
            valid_mask = df_new[year_column].notna() & (df_new[year_column] > 1800)

            # Truncate or extend sampled values to match valid count
            valid_count = valid_mask.sum()
            if len(sampled_values) >= valid_count:
                new_years = sampled_values[:valid_count]
            else:
                # Repeat if not enough samples
                new_years = (sampled_values * (valid_count // len(sampled_values) + 1))[:valid_count]

            df_new.loc[valid_mask, year_column] = new_years

            # Save
            df_new.to_csv(output_path, index=False, doublequote=False, escapechar="\\")

            # Evaluate
            actual_drift, corr_loss, abs_corr_base, abs_corr_gen = self.evaluator.evaluate(original_path, output_path)
            corr_ratio = corr_loss / abs_corr_base if abs_corr_base > 0 else 0
            drift_error = abs(actual_drift - target_drift) / target_drift if target_drift > 0 else abs(actual_drift)

            print(f"Actual drift: {actual_drift:.4f}, Target: {target_drift:.4f}")
            print(f"Drift error: {drift_error:.2%}, Corr loss: {corr_loss:.4f}, abs_corr(base)={abs_corr_base:.4f}, loss/abs_ratio={corr_ratio:.4f}")

            # Compute score
            if self.target_corr_loss is not None:
                corr_error = abs(self.target_corr_loss - corr_loss)
                score = drift_error * 2.0 + corr_error * 1.5
            else:
                score = drift_error * 2.0 + corr_loss * 1.0

            result = TuningResult(
                params=TuningParams(),  # Default params for fallback
                target_drift=target_drift,
                actual_drift=actual_drift,
                correlation_loss=corr_loss,
                drift_error=drift_error,
                score=score,
            )

            # Check if meets tolerance (relative <= 20% OR absolute <= 0.05)
            drift_ok = is_drift_ok(actual_drift, target_drift)
            if drift_ok:
                print(f"✓ find_q fallback SUCCESS!")
            else:
                print(f"find_q fallback: drift_error={drift_error:.2%}, abs_diff={abs(actual_drift - target_drift):.4f}")

            # Save metadata
            self._save_last_generation_metadata(
                result.params, target_drift, actual_drift, corr_loss, abs_corr_base, abs_corr_gen
            )

            return result

        except Exception as e:
            print(f"find_q fallback failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _apply_id_table_fallback(
        self,
        id_columns: List[str],
        target_drift: float,
        tolerance: float = 0.20,
        max_attempts: int = 20,
    ) -> Optional[TuningResult]:
        """Apply row-mixing fallback for all-ID tables.

        For tables where all columns are IDs (foreign keys), DDPM is not suitable.
        Instead, we mix rows from source and reference datasets at certain ratios
        to achieve target drift while preserving correlations.

        This works by:
        1. Sample some rows from source (original values)
        2. Sample remaining rows from reference (drifted values)
        3. Binary search to find the mixing ratio that achieves target drift
        """
        print(f"\n{'='*60}")
        print(f"FALLBACK: Row-Mixing Method for All-ID Table")
        print(f"{'='*60}")
        print(f"ID columns: {id_columns}")

        try:
            # Load source and reference data
            ref_dataset = self.reference_dataset or f"{self.dataset_name}_2017"
            src_path = f"datasets/{self.dataset_name}/{self.table_name}.csv"
            ref_path = f"datasets/{ref_dataset}/{self.table_name}.csv"

            output_dir = os.path.join("expdir", self.dataset_name, self.table_name)
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{self.table_name}.drifted.csv")

            # Load datasets
            try:
                src_df = self._load_csv(src_path)
                ref_df = self._load_csv(ref_path)
            except RuntimeError:
                print(f"Failed to load datasets")
                return None

            print(f"Source: {len(src_df)} rows, Reference: {len(ref_df)} rows")

            # Use source row count as target
            target_rows = len(src_df)

            # Get applicable columns for drift calculation
            info_path = f"datasets/{self.dataset_name}/dataset_info.json"
            with open(info_path, "r") as f:
                info = json.load(f)
            applicable_cols = info.get(self.table_name, {}).get("applicable_columns", id_columns)

            def compute_drift_for_ratio(ref_ratio: float) -> Tuple[float, pd.DataFrame]:
                """Compute JS divergence for a given reference ratio."""
                # Sample rows from each dataset
                n_ref = int(target_rows * ref_ratio)
                n_src = target_rows - n_ref

                # Sample with replacement if needed
                if n_src > 0:
                    src_sample = src_df.sample(n=n_src, replace=(n_src > len(src_df)))
                else:
                    src_sample = pd.DataFrame()

                if n_ref > 0:
                    ref_sample = ref_df.sample(n=n_ref, replace=(n_ref > len(ref_df)))
                else:
                    ref_sample = pd.DataFrame()

                # Combine
                mixed_df = pd.concat([src_sample, ref_sample], ignore_index=True)

                # Shuffle to mix rows
                mixed_df = mixed_df.sample(frac=1.0).reset_index(drop=True)

                # Reset ID column to sequential
                mixed_df['id'] = range(1, len(mixed_df) + 1)

                # Compute JS divergence on applicable columns
                drifts = []
                for col in applicable_cols:
                    if col not in mixed_df.columns or col not in ref_df.columns:
                        continue

                    mixed_vals = mixed_df[col].dropna()
                    ref_vals = ref_df[col].dropna()

                    if len(mixed_vals) == 0 or len(ref_vals) == 0:
                        continue

                    # Compute distributions
                    all_vals = np.concatenate([mixed_vals, ref_vals])
                    bins = min(100, len(np.unique(all_vals)))
                    range_min, range_max = all_vals.min(), all_vals.max()

                    mixed_hist, _ = np.histogram(mixed_vals, bins=bins, range=(range_min, range_max), density=True)
                    ref_hist, _ = np.histogram(ref_vals, bins=bins, range=(range_min, range_max), density=True)

                    # Add small epsilon to avoid division by zero
                    mixed_hist = mixed_hist + 1e-10
                    ref_hist = ref_hist + 1e-10

                    # Normalize
                    mixed_hist = mixed_hist / mixed_hist.sum()
                    ref_hist = ref_hist / ref_hist.sum()

                    js = jensenshannon(mixed_hist, ref_hist)
                    drifts.append(js)

                avg_drift = np.mean(drifts) if drifts else 0.0
                return avg_drift, mixed_df

            # Binary search to find the right mixing ratio
            low_ratio, high_ratio = 0.0, 1.0
            best_result = None
            best_df = None
            best_drift_error = float('inf')

            print(f"Searching for mixing ratio to achieve drift={target_drift}...")

            for attempt in range(max_attempts):
                mid_ratio = (low_ratio + high_ratio) / 2
                actual_drift, mixed_df = compute_drift_for_ratio(mid_ratio)
                drift_error = abs(actual_drift - target_drift) / target_drift if target_drift > 0 else abs(actual_drift)

                print(f"  Attempt {attempt+1}: ref_ratio={mid_ratio:.3f}, drift={actual_drift:.4f}, error={drift_error:.2%}")

                if drift_error < best_drift_error:
                    best_drift_error = drift_error
                    best_df = mixed_df
                    best_result = (mid_ratio, actual_drift)

                if is_drift_ok(actual_drift, target_drift):
                    print(f"  ✓ Found acceptable solution!")
                    break

                # Adjust search bounds
                if actual_drift < target_drift:
                    # Need more drift -> more reference data
                    low_ratio = mid_ratio
                else:
                    # Need less drift -> less reference data
                    high_ratio = mid_ratio

            if best_df is None:
                print("Failed to find acceptable mixing ratio")
                return None

            ref_ratio, actual_drift = best_result

            # Save the mixed data
            best_df.to_csv(output_path, index=False, doublequote=False, escapechar="\\")
            print(f"Saved mixed data to {output_path}")

            # Compute correlation (for ID tables, this is typically 0)
            corr_loss = 0.0  # ID correlations don't have semantic meaning

            result = TuningResult(
                params=TuningParams(),  # Default params for fallback
                target_drift=target_drift,
                actual_drift=actual_drift,
                correlation_loss=corr_loss,
                drift_error=best_drift_error,
                score=best_drift_error,  # Simple score for fallback
            )

            # Update cache
            self.cache.update(result)
            self._save_cache()

            print(f"\nRow-mixing fallback result:")
            print(f"  Reference ratio: {ref_ratio:.2%}")
            print(f"  Actual drift: {actual_drift:.4f}")
            print(f"  Drift error: {best_drift_error:.2%}")

            if is_drift_ok(actual_drift, target_drift):
                print(f"✓ Row-mixing fallback SUCCESS!")
            else:
                print(f"Row-mixing fallback: best effort (error={best_drift_error:.2%}, abs_diff={abs(actual_drift - target_drift):.4f})")

            # Save metadata
            self._save_last_generation_metadata(
                result.params, target_drift, actual_drift, corr_loss
            )

            return result

        except Exception as e:
            print(f"Row-mixing fallback failed: {e}")
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
                corr_ok = corr_error <= CORR_TOLERANCE

                if corr_ok:
                    # Cache meets tolerance - use it regardless of validation status
                    # In --validate mode, validation will happen later
                    if cached.validation_passed:
                        print(f"Using cached parameters (drift_error={cached.drift_error:.2%}, corr_error={corr_error:.4f}, validated=✓)")
                    else:
                        print(f"Using cached parameters (drift_error={cached.drift_error:.2%}, corr_error={corr_error:.4f}, pending validation)")
                    return cached
                else:
                    print(f"Cache found but corr_error too high ({corr_error:.4f} > {CORR_TOLERANCE}), re-tuning...")

        # Check if this is an all-ID columns table (prone to mode collapse)
        is_all_id, id_columns = self._is_all_id_columns_table()
        if is_all_id:
            print(f"\n⚠️  NOTE: All-ID columns table detected: {id_columns}")
            print(f"    Will use conservative parameters and monitor for mode collapse.\n")

        # Track mode collapse retries
        mode_collapse_retries = 0
        max_mode_collapse_retries = 3

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

        # Adaptive drift_range adjustment
        drift_range_expansion = 0.0  # Start with ±0.15, can expand up to ±0.40

        # === STEP-BY-STEP TUNING PHASES (Adaptive) ===
        # Each phase adjusts one parameter with trend detection:
        # - If improving: continue in same direction (or expand range)
        # - If worsening: stop phase and use best value found

        # Corr Focus phases: parameter name, initial candidates, can_expand_up, can_expand_down
        # Defaults: weight_corr=5, weight_real=0.008, batch_size=512, dim=(512,512), steps=10000
        # To improve correlation: increase weight_corr or decrease weight_real
        CORR_FOCUS_PHASES = [
            ("weight_corr", [8, 12, 20, 30, 50, 80], True, False),  # Increase from default 5
            ("weight_real", [0.005, 0.003, 0.001, 0.0005], False, True),  # Decrease from default 0.008
            ("batch_size", [1024, 2048, 4096, 8192], True, False),  # Skip default 512
            ("dim", [(768, 768), (1024, 768), (1024, 1024), (1024, 1024, 512), (1024, 1024, 1024)], True, False),  # Skip default (512,512)
            ("steps", [12000, 15000, 20000, 25000, 30000, 40000], True, False),
        ]

        # Drift Focus phases
        # Defaults: weight_corr=5, weight_real=0.008, dim=(512,512)
        # To improve drift: decrease weight_corr or increase weight_real
        DRIFT_FOCUS_PHASES = [
            ("weight_corr", [3, 2, 1, 0.5, 0.2, 0.1], False, True),  # Decrease from default 5
            ("weight_real", [0.01, 0.02, 0.05, 0.1], True, False),  # Increase from default 0.008
            ("drift_range", [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50], True, False),
            ("dim", [(768, 768), (1024, 768), (1024, 1024), (1024, 1024, 512)], True, False),  # Skip default (512,512)
        ]

        # Phase tracking for focus modes
        corr_focus_phase = 0
        corr_focus_attempt = 0
        corr_focus_best_params = {
            "weight_corr": 5.0, "weight_real": 0.008, "batch_size": 512, "dim": (512, 512),
            "steps": 15000, "lr": 0.001,
        }
        # Trend tracking within phase
        corr_focus_phase_best_value = None
        corr_focus_phase_best_corr = float('inf')
        corr_focus_phase_history = []  # List of (value, corr_loss) for trend detection
        corr_focus_consecutive_worse = 0  # Count consecutive worsening results

        drift_focus_phase = 0
        drift_focus_attempt = 0
        drift_focus_best_params = {
            "weight_corr": 5.0, "weight_real": 0.008, "batch_size": 512, "dim": (512, 512),
            "steps": 15000, "lr": 0.001,
        }
        drift_focus_phase_best_value = None
        drift_focus_phase_best_drift_error = float('inf')
        drift_focus_phase_history = []
        drift_focus_consecutive_worse = 0

        for i in range(max_iterations):
            # === DETERMINE PARAMS BASED ON MODE ===

            if mode == "normal":
                retrain_controller_only = False
                if i == 0:
                    # First iteration: train with default params
                    should_retrain = True
                    ctrl_dim = (512, 512)
                    ctrl_steps = 15000
                    ctrl_lr = 0.0008
                    ctrl_corr_weight = 5.0  # Balanced default: PCorr*5 ≈ 0.005
                    ctrl_real_weight = 0.008  # Balanced default: RealMSE*0.008 ≈ 0.005
                    ctrl_bs = 512
                else:
                    # Normal: just adjust scale_factor, no retrain
                    should_retrain = False
                    ctrl_dim = (512, 512)
                    ctrl_steps = 12000
                    ctrl_lr = 0.001
                    ctrl_corr_weight = 5.0  # Balanced default
                    ctrl_real_weight = 0.008  # Balanced default
                    ctrl_bs = 512

            elif mode == "corr_focus":
                # Corr Focus: step-by-step adjustment to strengthen correlation
                # Only retrain controller - diffuser is not affected by weight/dim/steps params
                should_retrain = False
                retrain_controller_only = True

                # Get current phase info
                phase_name, phase_values, can_expand_up, can_expand_down = CORR_FOCUS_PHASES[corr_focus_phase]
                current_value = phase_values[min(corr_focus_attempt, len(phase_values) - 1)]

                # Use best params found so far, override the current phase param
                ctrl_corr_weight = corr_focus_best_params["weight_corr"]
                ctrl_real_weight = corr_focus_best_params["weight_real"]
                ctrl_bs = corr_focus_best_params["batch_size"]
                ctrl_dim = corr_focus_best_params["dim"]
                ctrl_steps = corr_focus_best_params["steps"]
                ctrl_lr = corr_focus_best_params["lr"]

                # Override the current phase parameter
                if phase_name == "weight_corr":
                    ctrl_corr_weight = current_value
                elif phase_name == "weight_real":
                    ctrl_real_weight = current_value
                elif phase_name == "batch_size":
                    ctrl_bs = current_value
                elif phase_name == "dim":
                    ctrl_dim = current_value
                elif phase_name == "steps":
                    ctrl_steps = current_value

                print(f"*** Corr Focus Mode (phase {corr_focus_phase + 1}/{len(CORR_FOCUS_PHASES)}: {phase_name}) ***")
                print(f"  Trying {phase_name}={current_value} (attempt {corr_focus_attempt + 1})")
                print(f"  Current best: weight_corr={corr_focus_best_params['weight_corr']}, "
                      f"weight_real={corr_focus_best_params['weight_real']}, "
                      f"bs={corr_focus_best_params['batch_size']}, dim={corr_focus_best_params['dim']}")
                if corr_focus_phase_history:
                    trend = "improving" if len(corr_focus_phase_history) < 2 or corr_focus_phase_history[-1][1] < corr_focus_phase_history[-2][1] else "worsening"
                    print(f"  Trend: {trend} (consecutive_worse={corr_focus_consecutive_worse})")

            elif mode == "drift_focus":
                # Drift Focus: step-by-step adjustment to strengthen drift control
                # Only retrain controller - diffuser is not affected by weight/dim/drift_range params
                should_retrain = False
                retrain_controller_only = True

                # Get current phase info
                phase_name, phase_values, can_expand_up, can_expand_down = DRIFT_FOCUS_PHASES[drift_focus_phase]
                current_value = phase_values[min(drift_focus_attempt, len(phase_values) - 1)]

                # Use best params found so far, override the current phase param
                ctrl_corr_weight = drift_focus_best_params["weight_corr"]
                ctrl_real_weight = drift_focus_best_params["weight_real"]
                ctrl_bs = drift_focus_best_params["batch_size"]
                ctrl_dim = drift_focus_best_params["dim"]
                ctrl_steps = drift_focus_best_params["steps"]
                ctrl_lr = drift_focus_best_params["lr"]

                # Override the current phase parameter
                if phase_name == "weight_corr":
                    ctrl_corr_weight = current_value
                elif phase_name == "weight_real":
                    ctrl_real_weight = current_value
                elif phase_name == "drift_range":
                    drift_range_expansion = current_value
                elif phase_name == "dim":
                    ctrl_dim = current_value

                print(f"*** Drift Focus Mode (phase {drift_focus_phase + 1}/{len(DRIFT_FOCUS_PHASES)}: {phase_name}) ***")
                print(f"  Trying {phase_name}={current_value} (attempt {drift_focus_attempt + 1})")
                print(f"  Current best: weight_corr={drift_focus_best_params['weight_corr']}, "
                      f"weight_real={drift_focus_best_params['weight_real']}, dim={drift_focus_best_params['dim']}")
                if drift_focus_phase_history:
                    trend = "improving" if len(drift_focus_phase_history) < 2 or drift_focus_phase_history[-1][1] < drift_focus_phase_history[-2][1] else "worsening"
                    print(f"  Trend: {trend} (consecutive_worse={drift_focus_consecutive_worse})")

            # Set drift_range centered on target (with adaptive expansion)
            base_range = 0.15 + drift_range_expansion
            drift_range_min = max(0.01, target_drift - base_range)
            drift_range_max = min(0.95, target_drift + base_range)

            params = TuningParams(
                scale_factor=current_sf,
                controller_lr=ctrl_lr,
                controller_steps=ctrl_steps,
                controller_bs=ctrl_bs,
                controller_dim=ctrl_dim,
                drift_range_min=drift_range_min,
                drift_range_max=drift_range_max,
                loss_weight_corr=ctrl_corr_weight,
                loss_weight_real=ctrl_real_weight,
            )

            print(f"\n[Iteration {i+1}/{max_iterations}] mode={mode}")
            train_mode = "controller_only" if retrain_controller_only else ("full" if should_retrain else "none")
            print(f"Params: sf={current_sf:.4f}, dim={ctrl_dim}, bs={ctrl_bs}, "
                  f"corr_w={ctrl_corr_weight:.2f}, real_w={ctrl_real_weight:.2f}, "
                  f"train={train_mode}")

            result, mode_collapsed = self._evaluate_params(params, target_drift, retrain=should_retrain, retrain_controller_only=retrain_controller_only)

            if result is None:
                print("Evaluation failed, trying different scale_factor...")
                current_sf = (sf_low + sf_high) / 2
                iterations_without_improvement += 1
                continue

            # Handle mode collapse - reduce scale_factor and corr_weight
            if mode_collapsed:
                mode_collapse_retries += 1
                print(f"\n⚠️  Mode collapse detected (retry {mode_collapse_retries}/{max_mode_collapse_retries})")
                if mode_collapse_retries <= max_mode_collapse_retries:
                    # Reduce aggressive parameters
                    current_sf = max(sf_low, current_sf * 0.5)
                    ctrl_corr_weight = max(0.5, ctrl_corr_weight * 0.5)
                    print(f"    Reducing: scale_factor -> {current_sf:.4f}, corr_weight -> {ctrl_corr_weight:.2f}")
                    should_retrain = True  # Force retrain with new params
                    retrain_controller_only = False  # Need full retrain for mode collapse
                    continue
                else:
                    print(f"    Max retries reached. Trying row-mixing fallback...")
                    is_all_id, id_columns = self._is_all_id_columns_table()
                    if is_all_id:
                        fallback_result = self._apply_id_table_fallback(id_columns, target_drift, tolerance)
                        if fallback_result:
                            return fallback_result
                    print("    Continuing with best effort...")

            # Track history
            history.append((current_sf, result.actual_drift, result.drift_error))
            drift_history.append({
                'sf': current_sf,
                'drift': result.actual_drift,
                'drift_error': result.drift_error,
                'corr_loss': result.correlation_loss,
            })

            # Update cache
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
            corr_ok = corr_error <= CORR_TOLERANCE

            print(f"Result: drift_error={result.drift_error:.2%} ({'OK' if drift_ok else 'BAD'}), "
                  f"corr_error={corr_error:.4f} ({'OK' if corr_ok else 'BAD'})")

            # === TRACK BEST VALUE AND TREND WITHIN CURRENT PHASE ===
            if mode == "corr_focus":
                phase_name, phase_values, _, _ = CORR_FOCUS_PHASES[corr_focus_phase]
                current_value = phase_values[min(corr_focus_attempt, len(phase_values) - 1)]

                # Record in history for trend detection
                corr_focus_phase_history.append((current_value, result.correlation_loss))

                # Track best corr_loss in this phase
                if result.correlation_loss < corr_focus_phase_best_corr:
                    corr_focus_phase_best_corr = result.correlation_loss
                    corr_focus_phase_best_value = current_value
                    corr_focus_consecutive_worse = 0  # Reset worse counter
                    print(f"  → New best in phase {phase_name}: {current_value} (corr_loss={result.correlation_loss:.4f})")
                else:
                    corr_focus_consecutive_worse += 1
                    print(f"  → No improvement (corr_loss={result.correlation_loss:.4f}, best={corr_focus_phase_best_corr:.4f})")

            elif mode == "drift_focus":
                phase_name, phase_values, _, _ = DRIFT_FOCUS_PHASES[drift_focus_phase]
                current_value = phase_values[min(drift_focus_attempt, len(phase_values) - 1)]

                # Record in history for trend detection
                drift_focus_phase_history.append((current_value, result.drift_error))

                # Track best drift_error in this phase
                if result.drift_error < drift_focus_phase_best_drift_error:
                    drift_focus_phase_best_drift_error = result.drift_error
                    drift_focus_phase_best_value = current_value
                    drift_focus_consecutive_worse = 0  # Reset worse counter
                    print(f"  → New best in phase {phase_name}: {current_value} (drift_error={result.drift_error:.2%})")
                else:
                    drift_focus_consecutive_worse += 1
                    print(f"  → No improvement (drift_error={result.drift_error:.2%}, best={drift_focus_phase_best_drift_error:.2%})")

            # === SUCCESS ===
            if drift_ok and corr_ok:
                print(f"\n✓ SUCCESS! Both drift and correlation within tolerance!")
                break

            # === MODE TRANSITIONS ===
            if drift_ok and not corr_ok:
                # Drift is OK, need to improve correlation → Corr Focus Mode
                if mode == "normal":
                    print(f"\n→ Drift OK but correlation bad, entering Corr Focus Mode")
                    mode = "corr_focus"
                    corr_focus_phase = 0
                    corr_focus_attempt = 0
                    corr_focus_phase_best_value = None
                    corr_focus_phase_best_corr = float('inf')
                    corr_focus_phase_history = []
                    corr_focus_consecutive_worse = 0
                    # Initialize best params from current normal mode params
                    corr_focus_best_params = {
                        "weight_corr": 5.0, "weight_real": 0.008, "batch_size": 512, "dim": (512, 512),
                        "steps": 15000, "lr": 0.001,
                    }
                elif mode == "drift_focus":
                    # Drift Focus succeeded! But now corr is bad.
                    print(f"\n✓ Drift Focus succeeded! Drift is now OK.")
                    print(f"  But correlation is bad - switching to Corr Focus Mode")
                    mode = "corr_focus"
                    corr_focus_phase = 0
                    corr_focus_attempt = 0
                    corr_focus_phase_best_value = None
                    corr_focus_phase_best_corr = float('inf')
                    corr_focus_phase_history = []
                    corr_focus_consecutive_worse = 0
                    # Carry over successful drift_focus params as starting point
                    corr_focus_best_params = drift_focus_best_params.copy()
                elif mode == "corr_focus":
                    # Still in corr_focus, check if should advance or stop phase
                    phase_name, phase_values, can_expand_up, can_expand_down = CORR_FOCUS_PHASES[corr_focus_phase]

                    # Check if we should stop this phase early (trend is worsening)
                    should_stop_phase = corr_focus_consecutive_worse >= 2
                    at_end_of_list = corr_focus_attempt >= len(phase_values) - 1

                    if should_stop_phase:
                        print(f"\n  ⚡ Early stop: {phase_name} worsening for {corr_focus_consecutive_worse} attempts")

                    if should_stop_phase or at_end_of_list:
                        # Save best value and move to next phase
                        best_val = corr_focus_phase_best_value or phase_values[0]
                        if phase_name == "weight_corr":
                            corr_focus_best_params["weight_corr"] = best_val
                        elif phase_name == "weight_real":
                            corr_focus_best_params["weight_real"] = best_val
                        elif phase_name == "batch_size":
                            corr_focus_best_params["batch_size"] = best_val
                        elif phase_name == "dim":
                            corr_focus_best_params["dim"] = best_val
                        elif phase_name == "steps":
                            corr_focus_best_params["steps"] = best_val

                        print(f"\n  Phase {phase_name} done. Best value: {best_val} (corr_loss={corr_focus_phase_best_corr:.4f})")
                        corr_focus_phase += 1
                        corr_focus_attempt = 0
                        # Reset phase tracking for next phase
                        corr_focus_phase_best_value = None
                        corr_focus_phase_best_corr = float('inf')
                        corr_focus_phase_history = []
                        corr_focus_consecutive_worse = 0

                        if corr_focus_phase >= len(CORR_FOCUS_PHASES):
                            print(f"\n✗ GIVING UP: Tried all corr_focus phases")

                            # Check if this is a single year column table - try fallback
                            is_year_table, year_col = self._is_single_year_column_table()
                            if is_year_table and year_col:
                                print(f"\nDetected single year column table, trying fallback...")
                                fallback_result = self._apply_year_offset_fallback(year_col, target_drift)
                                if fallback_result:
                                    self.cache.update(fallback_result)
                                    self._save_cache()
                                    if best_result is None or fallback_result.score < best_result.score:
                                        best_result = fallback_result
                                        print(f"Fallback improved result! Score: {fallback_result.score:.4f}")
                            break
                    else:
                        # Continue to next attempt in this phase
                        corr_focus_attempt += 1
                # Keep scale_factor fixed, will retrain with adjusted params
                continue

            elif not drift_ok and corr_ok:
                # Correlation is OK, need to improve drift
                if mode == "normal":
                    # Check if drift is insensitive to scale_factor
                    if consecutive_small_gradient >= 3:
                        print(f"\n→ Corr OK but drift not responding to sf (gradient too small for {consecutive_small_gradient} iterations)")
                        print(f"  Entering Drift Focus Mode to retrain with different controller params")
                        mode = "drift_focus"
                        drift_focus_phase = 0
                        drift_focus_attempt = 0
                        drift_focus_phase_best_value = None
                        drift_focus_phase_best_drift_error = float('inf')
                        drift_focus_phase_history = []
                        drift_focus_consecutive_worse = 0
                        drift_focus_best_params = {
                            "weight_corr": 5.0, "weight_real": 0.008, "batch_size": 512, "dim": (512, 512),
                            "steps": 15000, "lr": 0.001,
                        }
                        consecutive_small_gradient = 0
                        # Reset sf search bounds for fresh start after retrain
                        sf_low = 0.001
                        sf_high = 30.0
                        current_sf = self._get_initial_scale_factor(target_drift)
                        continue
                    # Otherwise, stay in normal mode - sf adjustment will handle it
                elif mode == "corr_focus":
                    # Correlation was fixed by corr_focus! But now drift is bad.
                    print(f"\n✓ Corr Focus succeeded! Correlation is now OK.")
                    print(f"  But drift is bad - returning to normal mode to try sf adjustment")
                    mode = "normal"
                    consecutive_small_gradient = 0
                    # Reset sf search bounds since model changed
                    sf_low = 0.001
                    sf_high = 30.0
                    current_sf = self._get_initial_scale_factor(target_drift)
                    # Fall through to sf adjustment below
                elif mode == "drift_focus":
                    # Still in drift_focus, check if should advance or stop phase
                    phase_name, phase_values, can_expand_up, can_expand_down = DRIFT_FOCUS_PHASES[drift_focus_phase]

                    # Check if we should stop this phase early (trend is worsening)
                    should_stop_phase = drift_focus_consecutive_worse >= 2
                    at_end_of_list = drift_focus_attempt >= len(phase_values) - 1

                    if should_stop_phase:
                        print(f"\n  ⚡ Early stop: {phase_name} worsening for {drift_focus_consecutive_worse} attempts")

                    if should_stop_phase or at_end_of_list:
                        # Save best value and move to next phase
                        best_val = drift_focus_phase_best_value or phase_values[0]
                        if phase_name == "weight_corr":
                            drift_focus_best_params["weight_corr"] = best_val
                        elif phase_name == "weight_real":
                            drift_focus_best_params["weight_real"] = best_val
                        elif phase_name == "drift_range":
                            drift_range_expansion = best_val  # Keep the best drift_range
                        elif phase_name == "dim":
                            drift_focus_best_params["dim"] = best_val

                        print(f"\n  Phase {phase_name} done. Best value: {best_val} (drift_error={drift_focus_phase_best_drift_error:.2%})")
                        drift_focus_phase += 1
                        drift_focus_attempt = 0
                        # Reset phase tracking for next phase
                        drift_focus_phase_best_value = None
                        drift_focus_phase_best_drift_error = float('inf')
                        drift_focus_phase_history = []
                        drift_focus_consecutive_worse = 0

                        if drift_focus_phase >= len(DRIFT_FOCUS_PHASES):
                            print(f"\n✗ GIVING UP: Tried all drift_focus phases")

                            # Check if this is a single year column table - try fallback
                            is_year_table, year_col = self._is_single_year_column_table()
                            if is_year_table and year_col:
                                print(f"\nDetected single year column table, trying fallback...")
                                fallback_result = self._apply_year_offset_fallback(year_col, target_drift)
                                if fallback_result:
                                    self.cache.update(fallback_result)
                                    self._save_cache()
                                    if best_result is None or fallback_result.score < best_result.score:
                                        best_result = fallback_result
                                        print(f"Fallback improved result! Score: {fallback_result.score:.4f}")
                            break
                    else:
                        # Continue to next attempt in this phase
                        drift_focus_attempt += 1
                    continue

            elif not drift_ok and not corr_ok:
                # Both bad - stay in normal mode, adjust scale_factor
                # If we were in a focus mode but now both are bad, something went wrong
                if mode != "normal":
                    print(f"Warning: Was in {mode} but now both drift and corr are bad, returning to normal")
                    mode = "normal"
                    consecutive_small_gradient = 0
                    # Reset sf search bounds since model changed
                    sf_low = 0.001
                    sf_high = 30.0
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
                                print(f"  → Triggering retrain with different controller params")
                                should_retrain = True
                                retrain_controller_only = False  # Need full retrain
                                # Try expanding drift_range for controller training
                                drift_range_expansion = min(0.25, drift_range_expansion + 0.05)
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

                        # Note: if consecutive_small_gradient >= 3, next iteration will
                        # transition to drift_focus mode (handled in mode transition section)
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
                    # Reset and try with different parameters
                    print("Resetting search bounds and retraining with different settings...")
                    sf_low = 0.001
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

        # Check cache - same logic as tune()
        if use_cache:
            cached = self.cache.get_best_params(target_drift)
            if cached and is_drift_ok(cached.actual_drift, target_drift):
                # Check correlation error
                # correlation_loss = |base - generated|, compare with target_corr_loss from drift_ref
                if self.target_corr_loss is not None:
                    corr_error = abs(cached.correlation_loss - self.target_corr_loss)
                else:
                    corr_error = cached.correlation_loss  # Target is 0
                corr_ok = corr_error <= CORR_TOLERANCE

                if corr_ok:
                    if cached.validation_passed:
                        print(f"Using cached parameters (drift_error={cached.drift_error:.2%}, corr_error={corr_error:.4f}, validated=✓)")
                    else:
                        print(f"Using cached parameters (drift_error={cached.drift_error:.2%}, corr_error={corr_error:.4f}, pending validation)")
                    return cached
                else:
                    print(f"Cache found but corr_error too high ({corr_error:.4f} > {CORR_TOLERANCE}), searching...")

        # Check if this is an all-ID columns table (just warn, mode collapse detection will handle it)
        is_all_id, id_columns = self._is_all_id_columns_table()
        if is_all_id:
            print(f"\n⚠️  NOTE: All-ID columns table detected: {id_columns}")
            print(f"    Will monitor for mode collapse.\n")

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

            result, mode_collapsed = self._evaluate_params(params, target_drift, retrain=False)

            if result is None:
                continue

            # Skip mode-collapsed results in quick_tune
            if mode_collapsed:
                print(f"  Skipping due to mode collapse")
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
    parser.add_argument("--max-iterations", type=int, default=50)
    parser.add_argument("--tolerance", type=float, default=0.20, help="Drift error tolerance (relative, default 20%)")
    parser.add_argument("--no-stop-on-tolerance", action="store_true",
                        help="Allow early stop even if tolerance not met")
    parser.add_argument("--stagnation-limit", type=int, default=5,
                        help="Iterations without improvement before retrain")
    parser.add_argument("--retrain-interval", type=int, default=10,
                        help="Retrain models every N iterations if not improving")
    parser.add_argument("--quick", action="store_true", help="Quick tune (no retraining)")
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
    )

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
            # Fall through to normal tune/quick_tune

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

        # Always ensure data is generated with best params
        # The last iteration might not be the best one, so we need to regenerate
        metadata_path = os.path.join("expdir", args.dataset_name, args.table_name, "last_generation.json")
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
