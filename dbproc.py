import argparse
from functools import cached_property
import json
import os
from typing import List, Optional

import pandas as pd
from neurbench import config, dist, deterministic, sample, fileop
from neurbench.drift import find_q, jensenshannon
from neurbench.util import formatted_list

deterministic.seed_everything(42)


def is_numerical_column(series: pd.Series):
    return all(isinstance(x, int) or isinstance(x, float) for x in series)


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


class TableProcessor:
    def __init__(
            self,
            dbname: str,
            table: str,
            input_path: str,
            output_path: str,
            config_path: str,
            n_bins: int,
            skewed: int,
    ):
        # TODO: Support other databases than TPC-H
        self.dbname = dbname
        self.table = table
        self.input_path = input_path
        self.output_path = output_path
        self.config_path = config_path
        self.n_bins = n_bins
        self.skewed = skewed

        self.applicable_columns_list = config.TPCH_DB_APPLICABLE_COLUMNS[table]
        self.predefined_bins = None
        self.config = {}
        self.dists = {}
        self.new_data = {}

        self.load_bin_values()
        self.df = self.load_data()
        self.compute_dists()

    def load_bin_values(self):
        if not os.path.exists(self.config_path):
            return

        try:
            with open(self.config_path, "r") as f:
                self.config = json.loads(f.read())
        except:
            print("WARN  Invalid config. ignoring.")

        print("config loaded from: ", self.config_path)

    def load_data(self):
        df = pd.read_csv(self.input_path, sep="|", header=None)
        df.columns = df.columns.astype(str)

        return df

    def compute_dists(self):
        for i in range(len(self.applicable_columns_list)):
            if not self.applicable_columns_list[i]:
                continue

            i = str(i)

            d = self._get_dist(self.df[i])
            print(d)

            self.dists[str(self.df[i].name)] = d

    def dump_config(self):
        fileop.dump_json(self.config, self.config_path)

    def _get_dist(self, series):
        if self.config:
            c = self.config.get(str(series.name))
        else:
            c = None

        d = SeriesDistribution(series, n_bins=self.n_bins, config=c)
        self.config[str(series.name)] = d.bin_values

        return d

    def apply_drift(self, drift: float):
        for k in self.dists.keys():
            dist = self.dists[k].get()
            # dist.values: frequence of bin/value
            p = dist.values
            q = find_q(p, drift, self.skewed == 1)

            print(formatted_list(p))
            print(formatted_list(q))
            print(f"JS divergence={jensenshannon(p, q)}")

            self.new_data[k] = self._sample_data(
                q, dist.index.values, len(self.df.index)
            )

    def _sample_data(self, dist: List[float], index: list, size: int):
        return sample.sample_from_distribution(dist, index, size)[0]

    def save_data(self):
        df = self.df.copy()

        for k in self.new_data.keys():
            df[k] = self.new_data[k]

        fileop.dump_tbl(df, self.output_path)


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
        default="./{table}-config.json",
        help="Path to table config file, including bin values (default: ./{table}-config.json)",
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
        if isinstance(v, str) and "{table}" in v:
            args.__dict__[k] = v.format(table=args.table)

    print(args)

    # Validate drift factor
    if not 0.0 <= args.drift <= 1.0:
        parser.error("Drift factor must be between 0.0 and 1.0")

    tp = TableProcessor(
        args.dbname,
        args.table,
        args.input,
        args.output,
        args.config,
        args.n_bins,
        args.skewed,
    )

    tp.dump_config()
    print(f"Table config dumped to {args.config}")

    tp.apply_drift(args.drift)
    print(f"Processed data with drift factor {args.drift}")

    tp.save_data()
    print(f"Data saved to {args.output}")


if __name__ == "__main__":
    main()
