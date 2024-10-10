"""qpre: Preprocessing query SQLs"""

import argparse
import os
import glob


def remove_comments(sql):
    return "\n".join(
        [line for line in sql.split("\n") if not line.strip().startswith("--")]
    )


def compress_sql_to_single_line(sql):
    return " ".join(sql.split())


def preprocess(sql):
    sql = remove_comments(sql)
    sql = compress_sql_to_single_line(sql)
    sql = sql.lower()
    return sql


def main():
    parser = argparse.ArgumentParser(description="Preprocessing query SQLs")

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
        "-I",
        "--input_dir",
        default="./q_train_0",
        help="Path to the folder of input files",
    )

    parser.add_argument(
        "-o",
        "--output",
        default="./{dbname}-pp.sql",
        help="Path to the output CSV file (default: ./{dbname}-drifted.sql)",
    )
    parser.add_argument(
        "-c",
        "--config",
        default="./{dbname}-query-config.json",
        help="Path to DB config file (default: ./{dbname}-query-config.json)",
    )

    args = parser.parse_args()

    for k, v in args.__dict__.items():
        if isinstance(v, str) and "{dbname}" in v:
            args.__dict__[k] = v.format(dbname=args.dbname)

    print(args)

    if args.input_dir:
        if os.path.exists(args.output):
            os.remove(args.output)

        sql_files = glob.glob(os.path.join(args.input_dir, "*.sql"))
        for sql_file in sql_files:
            with open(sql_file, "r") as f:
                sqls = f.read().split(";")
                sqls = [preprocess(sql) for sql in sqls if sql]
                sqls = [sql for sql in sqls if sql.strip()]
                sqls = [sql for sql in sqls if not sql.startswith("limit")]
                sqls = [f"{sql};" for sql in sqls]

                with open(args.output, "a+") as fo:
                    fo.write(sqls[0])
                    fo.write("\n")
        return

    with open(args.input, "r") as f:
        sqls = f.read().split(";")
        sqls = [preprocess(sql) for sql in sqls if sql]
        sqls = [sql for sql in sqls if sql.strip()]
        sqls = [sql for sql in sqls if not sql.startswith("limit")]
        sqls = [f"{sql};" for sql in sqls]

        with open(args.output, "w") as f:
            f.write("\n".join(sqls))


if __name__ == "__main__":
    main()
