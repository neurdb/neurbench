import argparse
import glob
import os
from functools import cached_property
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
import pglast
from scipy.spatial.distance import jensenshannon

import neurbench
from neurbench import deterministic, dist, fileop, sample
from neurbench.drift import find_q
from neurbench.query import SQLInfoExtractor
from neurbench.util import formatted_list, tuple_to_list

TYPES = ["tables", "predicates", "joins", "aliasname_fullname"]


class MetadataDistributionHelpers:
    def __init__(self, series: pd.Series, slots: Optional[List[str]] = None):
        self._series = series
        self._slots = slots

    @cached_property
    def dist(self) -> pd.Series:
        if self._slots:
            series = self._series.astype(str)
            d = dist.categorical_dist_on_predefined_bins(series, bins=self._slots)
        else:
            d = dist.categorical_dist(self._series)

        return d


class MetadataDistribution:
    def __init__(
            self,
            type: str,
            data: Sequence[Any],
            slots: Optional[List[str]] = None,
    ):
        self._type = type
        self._data = data
        self._series = pd.Series(self._data)
        self._helpers = MetadataDistributionHelpers(series=self._series, slots=slots)

    def get(self) -> pd.Series:
        return self._helpers.dist

    def __len__(self):
        return len(self._data)

    @property
    def bin_values(self):
        return self._helpers.dist.index.values

    def __repr__(self):
        return f"""Dist of Metadata {self._type}:
{self.get()}
"""


class QueryProcessor(neurbench.Processor):
    def __init__(
            self,
            dbname: str,
            type: str,
            config_path: str,
            skewed: int,
            dump_feature_table: bool = False,
    ):
        self.dbname = dbname
        self.config_path = config_path
        self.skewed = skewed
        self.dump_feature_table = dump_feature_table

        self.type = type

        self.predefined_bins = None

        self.data: Dict[str, List[Any]] = {}
        self.dists: Dict[str, MetadataDistribution] = {}
        self.new_data: Dict[str, List[Any]] = {}

        self._create = False

        self._config, err = neurbench.load_config(
            self.config_path, {"bin_values": {}, "map": {}}
        )
        if err is not None:
            print("WARN  loading config: ", err)

            if "no such file" in err:
                self._create = True

    @property
    def config(self):
        return self._config

    def _slots(self, data_dict: Optional[dict] = None):
        if not data_dict:
            data_dict = self.data

        return [str(k) for k in data_dict]

    def load_from_file(self, input_file: str):
        with open(input_file, "r") as f:
            sqls = f.readlines()
            for s in sqls:
                node = pglast.parse_sql(s.split("#####")[1])
                extractor = SQLInfoExtractor()
                extractor(node)

                info = extractor.info
                for k in info.keys():
                    if k not in self.data:
                        self.data[k] = []

                    info_k_values = sorted(info[k])
                    info_k_values = tuple_to_list(info_k_values)

                    self.data[k].append(info_k_values)

                    if self._create:
                        if k not in self.config["map"]:
                            self.config["map"][k] = {}

                        info_k_values_str = str(info_k_values)
                        if info_k_values_str not in self.config["map"][k]:
                            self.config["map"][k][info_k_values_str] = []

                        self.config["map"][k][info_k_values_str].append(s)

        # self.compute_dists()

    def load(self, input_path: str):
        sql_files = glob.glob(os.path.join(input_path, "*.sql"))
        for sql_file in sql_files:
            with open(sql_file, "r") as f:
                text = f.read()

            sqls = text.split(";")
            sqls = [s.strip() for s in sqls if s.strip()]

            for s in sqls:
                node = pglast.parse_sql(s)
                extractor = SQLInfoExtractor()
                extractor(node)

                info = extractor.info
                for k in info.keys():
                    if k not in self.data:
                        self.data[k] = []

                    info_k_values = sorted(info[k])
                    info_k_values = tuple_to_list(info_k_values)

                    self.data[k].append(info_k_values)

                    if self._create:
                        if k not in self.config["map"]:
                            self.config["map"][k] = {}

                        info_k_values_str = str(info_k_values)
                        if info_k_values_str not in self.config["map"][k]:
                            self.config["map"][k][info_k_values_str] = []

                        self.config["map"][k][info_k_values_str].append(s)

        # self.compute_dists()
        
        if self.dump_feature_table:
            self._dump_feature_table()
            
    def _dump_feature_table(self):
        df = pd.DataFrame(self.data)
        df = df.applymap(lambda x: str(x).replace("'", "").replace(", ", " AND "))
        df.to_csv(f"{self.dbname}_feature_table.csv", index=False)

    def compute_dists(self):
        for t in TYPES:
            d = self._get_dist(t, self.data[t])
            print(d)

            if self._create:
                self._update_config(t, d.bin_values)

            self.dists[t] = d

    def _update_config(self, type, bin_values):
        self.config["bin_values"][type] = bin_values

    def _get_dist(self, type: str, data: Sequence[Any]):
        if self._create:
            return MetadataDistribution(type, data)
        else:
            return MetadataDistribution(
                type, data, slots=self._slots(self.config["bin_values"][type])
            )

    def apply_drift(self, drift: float, n_samples: Optional[int]):
        dist = self.dists[self.type].get()

        if n_samples is None:
            n_samples = len(dist)

        # dist.values: frequence of bin/value
        p = dist.values
        q = find_q(p, drift, self.skewed == 1)

        print("previous dist", formatted_list(p))
        print("after dist", formatted_list(q))
        print(f"JS divergence={jensenshannon(p, q)}")

        self.new_data[self.type] = self._sample_data(q, dist.index.values, n_samples)

    def _sample_data(self, dist: List[float], index: list, size: int):
        result = []

        sampled_infos = sample.sample_from_distribution(dist, index, size)[0]
        for info in sampled_infos:
            selections = self.config["map"][self.type][str(info)]
            result.append(deterministic.sample_rng.choice(selections))

        return result

    def save_to_single_file(self, output_path: str):
        text = ";\n".join(self.new_data[self.type])
        fileop.dump_text(text, output_path)

    def save(self, output_path: str):
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        for i, sql in enumerate(self.new_data[self.type]):
            fileop.dump_text(sql, os.path.join(output_path, f"{i}.sql"))


def main():
    parser = argparse.ArgumentParser(description="Drift data")

    parser.add_argument(
        "-d", "--dbname", help="Database name (default: tpch)", required=True
    )

    parser.add_argument(
        "-i",
        "--input_file",
        default="",
        help=".sql file",
    )

    parser.add_argument(
        "-I",
        "--input_dir",
        default="",
        help="Path to the directory of input files",
    )
    parser.add_argument(
        "-D", "--drift", type=float, default=0.2, help="Drift factor (default: 0.2)"
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Path to the output CSV file (default: ./{dbname}-query-drifted.sql)",
        required = True
    )
    parser.add_argument(
        "-t",
        "--type",
        help='Metadata type (available: "tables", "predicates", "joins", "aliasname_fullname". default: "tables")',
        required=True
    )
    parser.add_argument(
        "-c",
        "--config",
        default="./{dbname}-query-config.json",
        help="Path to DB config file (default: ./{dbname}-query-config.json)",
    )
    parser.add_argument(
        "-n",
        "--n-samples",
        default=1000,
        type=int,
        help="Number of sampled queries (default: 1000)",
    )
    parser.add_argument(
        "-s",
        "--skewed",
        type=int,
        default=1,
        help="Whether to distribution shifts towards more skewed. 1 = yes, 0 = no (default: 1)",
    )
    parser.add_argument(
        "-F",
        "--dump_feature_table",
        action="store_true",
        help="Dump feature table",
    )

    args = parser.parse_args()

    for k, v in args.__dict__.items():
        if isinstance(v, str) and "{dbname}" in v:
            args.__dict__[k] = v.format(dbname=args.dbname)

    print(args)

    if args.type not in TYPES:
        parser.error(f"Type must be one of {TYPES}")

    # Validate drift factor
    if not 0.0 <= args.drift <= 1.0:
        parser.error("Drift factor must be between 0.0 and 1.0")

    p = QueryProcessor(args.dbname, args.type, args.config, args.skewed, args.dump_feature_table)

    neurbench.make_drift(
        p, args.input_file, args.input_dir, args.output, args.config, args.drift, args.n_samples
    )


if __name__ == "__main__":
    main()
