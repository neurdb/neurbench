import argparse


def main():
    parser = argparse.ArgumentParser(description="Drift data")

    parser.add_argument(
        "-d", "--dbname", default="tpch", help="Database name (default: tpch)"
    )
    parser.add_argument(
        "-i",
        "--input",
        default="./{dbname}.sql",
        help="Path to the input file (default: ./{dbname}.tbl)",
    )
    parser.add_argument(
        "-D", "--drift", type=float, default=0.2, help="Drift factor (default: 0.2)"
    )
    parser.add_argument(
        "-o",
        "--output",
        default="./{dbname}-drifted.sql",
        help="Path to the output CSV file (default: ./{dbname}-drifted.sql)",
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
        default="./{dbname}-config.json",
        help="Path to DB config file (default: ./{dbname}-query-config.json)",
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
        if isinstance(v, str) and "{dbname}" in v:
            args.__dict__[k] = v.format(dbname=args.dbname)

    print(args)

    # Validate drift factor
    if not 0.0 <= args.drift <= 1.0:
        parser.error("Drift factor must be between 0.0 and 1.0")


if __name__ == "__main__":
    main()
