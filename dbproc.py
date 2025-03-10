import argparse
import time
from functools import cached_property
from typing import List, Optional

import pandas as pd

import neurbench
from neurbench import config, dist, fileop, sample
from neurbench.drift import find_q, jensenshannon
from neurbench.util import formatted_list


def is_numerical_column(series: pd.Series, threshold: int = 20):
    # if there is float, => numerical
    for x in series:
        if isinstance(x, float):
            return True

    # if all int + (unique values < thresholw) => cate
    if all(isinstance(x, int) for x in series):
        unique_count = series.nunique()
        if unique_count < threshold:
            return False
        else:
            return True

    # If the series contains non-integer/non-float types, return False by default
    return False


class NumericalDistributionHelpers:
    def __init__(
        self,
        series: pd.Series,
        n_bins: Optional[int] = None,
        config: Optional[dict] = None,
    ):
        self._series = series
        self._n_bins = n_bins
        self._config = config

    @cached_property
    def dist(self):
        print("called dist on numerical_dist_on_predefined_bins")
        if self._config:
            d = dist.numerical_dist_on_predefined_bins(
                self._series, bins=self._config["values"]
            )
            return d

        if not self._n_bins:
            raise ValueError("n_bins is required for numerical columns")

        d = dist.numerical_dist(self._series, n_bins=self._n_bins)
        print(d)
        return d

    @property
    def bin_values(self):
        return {
            "type": "numerical",
            "values": sorted(
                [{"start": x.left, "end": x.right} for x in self.dist.index],
                key=lambda x: x["start"],
            ),
        }


class CategoricalDistributionHelpers:
    def __init__(self, series: pd.Series, config: Optional[dict] = None):
        self._series = series
        self._config = config

    @cached_property
    def dist(self):
        if self._config:
            d = dist.categorical_dist_on_predefined_bins(
                self._series, bins=self._config["values"]
            )
        else:
            d = dist.categorical_dist(self._series)

        return d

    @property
    def bin_values(self):
        return {"type": "categorical", "values": sorted(self.dist.index)}


class SeriesDistribution:
    def __init__(
        self,
        series: pd.Series,
        n_bins: Optional[int] = None,
        config: Optional[dict] = None,
    ):
        self._series = series
        self._n_bins = n_bins
        self._config = config
        self._is_numerical = is_numerical_column(series)
        if self._is_numerical:
            self._helpers = NumericalDistributionHelpers(
                series, n_bins=n_bins, config=config
            )
        else:
            self._helpers = CategoricalDistributionHelpers(series, config=config)

    def get(self) -> pd.Series:
        return self._helpers.dist

    @property
    def bin_values(self):
        return self._helpers.bin_values

    def __repr__(self):
        return f"""Dist of Column {self._series.name}:
{self.get()}
"""


class TableProcessor(neurbench.Processor):
    def __init__(
        self,
        dbname: str,
        table: str,
        config_path: str,
        n_bins: int,
        skewed: int,
    ):
        self.dbname = dbname
        self.table = table
        self.config_path = config_path
        self.n_bins = n_bins
        self.skewed = skewed

        self.applicable_columns_list = config.DB_MAP[dbname][
            "drift_applicable_columns"
        ][table]
        self.predefined_bins = None
        self.dists = {}
        self.new_data = {}

        self._config, err = neurbench.load_config(self.config_path)
        if err is not None:
            print("WARN  loading config: ", err)

    @property
    def config(self):
        return self._config

    def load(self, input_path: str):
        if input_path.endswith(".csv"):
            sep = ","
        elif input_path.endswith(".tbl"):
            sep = "|"
        else:
            raise ValueError(f"Unknown file type: {input_path}")

        df = pd.read_csv(
            input_path,
            sep=sep,
            header=None if self.dbname == "job" else "infer",
            doublequote=False,
            escapechar="\\",
            low_memory=False,
        )

        def test_and_convert_column_dtypes(series):
            """Convert series to correct dtype based on the first 10 values.

            If all values in the series are integers, convert the series to integers.
            If all values in the series are floats, convert the series to floats.
            If all values in the series are strings, convert the series to strings.
            If some values in the series are integers and some are floats, convert the series to floats.
            """
            # print(series.head())

            if all(isinstance(x, float) for x in series.head()):
                if any(series.head().isna()):
                    return series
                elif all(abs(x - int(x)) < 1e-5 for x in series.head()):
                    return series.fillna(value=0).apply(int)
                else:
                    return series

            if all(isinstance(x, str) for x in series.head()):
                return series.apply(str)
            else:
                return series

        for column, dtype in df.dtypes.items():
            print(f"{column}: {dtype}")
        print()

        df = df.apply(test_and_convert_column_dtypes, axis=0)

        for column, dtype in df.dtypes.items():
            print(f"{column}: {dtype}")
        print()

        df.columns = df.columns.astype(str)
        self.df = df

    def compute_dists(self):
        for i in range(len(self.applicable_columns_list)):
            if not self.applicable_columns_list[i]:
                continue

            ### Ver 1
            # i = str(i)
            # series = self.df[i]
            
            ### Ver 2
            series = self.df.iloc[:, i]
            
            d = self._get_dist(series)
            print(d)

            self._update_config(series.name, d.bin_values)

            self.dists[str(series.name)] = d

    def _get_dist(self, series):
        if self.config:
            c = self.config.get(str(series.name))
        else:
            c = None

        return SeriesDistribution(series, n_bins=self.n_bins, config=c)

    def _update_config(self, series_name, bin_values):
        self.config[str(series_name)] = bin_values

    def apply_drift(self, drift: float, n_samples: Optional[int]):
        if n_samples is not None:
            raise NotImplementedError("n_samples is not supported")

        for k in self.dists.keys():
            dist = self.dists[k].get()
            # dist.values: frequence of bin/value
            p = dist.values
            
            start_time = time.time()
            q = find_q(p, drift, self.skewed == 1)
            print(f"find_q time: {time.time() - start_time}")

            print(formatted_list(p))
            print(formatted_list(q))
            print(f"JS divergence={jensenshannon(p, q)}")

            self.new_data[k] = self._sample_data(
                q, dist.index.values, len(self.df.index)
            )

    def _sample_data(self, dist: List[float], index: list, size: int):
        return sample.sample_from_distribution(dist, index, size)[0]

    def save(self, output_path: str):
        df = self.df.copy()

        for k in self.new_data.keys():
            df[k] = self.new_data[k]

        if output_path.endswith(".tbl"):
            fileop.dump_tbl(df, output_path)
        else:
            fileop.dump_csv(df, output_path)


def main():
    parser = argparse.ArgumentParser(description="Drift data")

    parser.add_argument(
        "-d", "--dbname", default="tpch", help="Database name (default: tpch)"
    )
    parser.add_argument("-t", "--table", required=True, help="Table name")
    parser.add_argument(
        "-i",
        "--input",
        default="./{table}.tbl",
        help="Path to the input file (default: ./{table}.tbl)",
    )
    parser.add_argument(
        "-D", "--drift", type=float, default=0.2, help="Drift factor (default: 0.2)"
    )
    parser.add_argument(
        "-o",
        "--output",
        default="./{table}-drifted.tbl",
        help="Path to the output CSV file (default: ./{table}-drifted.tbl)",
    )
    parser.add_argument(
        "-b",
        "--n-bins",
        type=int,
        default=10,
        help="Number of bins for numerical data (default: 10)",
    )
    parser.add_argument(
        "-c",
        "--config",
        default="./{dbname}-table-{table}-config.json",
        help="Path to table config file, including bin values (default: ./{dbname}-table-{table}-config.json)",
    )
    parser.add_argument(
        "-s",
        "--skewed",
        type=int,
        default=1,
        help="Whether to distribution shifts towards more skewed. 1 = yes, 0 = no (default: 1)",
    )

    args = parser.parse_args()

    for k, v in args.__dict__.items():
        if isinstance(v, str) and ("{dbname}" in v or "{table}" in v):
            args.__dict__[k] = v.format(dbname=args.dbname, table=args.table)

    print(args)

    # Validate drift factor
    if not 0.0 <= args.drift <= 1.0:
        parser.error("Drift factor must be between 0.0 and 1.0")

    tp: neurbench.Processor = TableProcessor(
        args.dbname,
        args.table,
        args.config,
        args.n_bins,
        args.skewed,
    )

    neurbench.make_drift(tp, "", args.input, args.output, args.config, args.drift)


if __name__ == "__main__":
    main()
