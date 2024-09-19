import argparse

import pandas as pd
import config
import dist


def is_numerical_column(series: pd.Series):
    return all(isinstance(x, int) or isinstance(x, float) for x in series)


class DataProcessor:
    def __init__(
        self,
        dbname: str,
        table: str,
        input_path: str,
        output_path: str,
        drift: float,
        n_bins: int,
    ):
        # TODO: Support other databases than TPC-H
        self.dbname = dbname
        self.table = table
        self.input_path = input_path
        self.output_path = output_path
        self.drift = drift
        self.n_bins = n_bins

        self.applicable_columns_list = config.TPCH_DB_APPLICABLE_COLUMNS[table]

    def process(self):
        with open(self.input_path, "r") as f:
            df = pd.read_csv(f, sep="|", header=None)

        for i in range(len(self.applicable_columns_list)):
            if not self.applicable_columns_list[i]:
                continue

            # check if the column contains numerical values
            self._get_dist(df[i])

    def _get_dist(self, series):
        if is_numerical_column(series):
            distribution = dist.numerical_dist(series, n_bins=self.n_bins)
        else:
            distribution = dist.categorical_dist(series)

        print(f"Dist of Column {i}")
        print(distribution)


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

    args = parser.parse_args()
    if "{table}" in args.input:
        args.input = args.input.format(table=args.table)
    if "{table}" in args.output:
        args.output = args.output.format(table=args.table)

    print(args)

    # Validate drift factor
    if not 0.0 <= args.drift <= 1.0:
        parser.error("Drift factor must be between 0.0 and 1.0")

    DataProcessor(
        args.dbname, args.table, args.input, args.output, args.drift, args.n_bins
    ).process()
    print(f"Processed data with drift factor {args.drift}")


if __name__ == "__main__":
    main()
