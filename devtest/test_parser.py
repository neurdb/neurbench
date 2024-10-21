import argparse
import glob
import os
from pprint import pprint

import pglast
from neuralbench.query import SQLInfoExtractor


def print_info(extractor: SQLInfoExtractor):
    # Print extracted information
    print("Tables used:", extractor.tables)
    print("Alias to full table name mapping:", extractor.aliasname_fullname)
    print("Joins:", extractor.joins)
    print("Predicates:", extractor.predicates)
    print("-" * 80)


def main():
    parser = argparse.ArgumentParser(description="Preprocessing query SQLs")

    parser.add_argument(
        "-d", "--dbname", default="tpch", help="Database name (default: tpch)"
    )

    parser.add_argument(
        "-I",
        "--input_dir",
        default="",
        help="Path to the folder of input files",
    )

    parser.add_argument(
        "-i",
        "--input",
        default="./testdata/tpch-pp.sql",
        help="Path to the input file (default: ./{dbname}.tbl)",
    )

    parser.add_argument(
        "-o",
        "--output",
        default="./{dbname}-pp.sql",
        help="Path to the output sql, metadata pair",
    )

    args = parser.parse_args()

    for k, v in args.__dict__.items():
        if isinstance(v, str) and "{dbname}" in v:
            args.__dict__[k] = v.format(dbname=args.dbname)

    print(args)

    if args.input_dir:
        sql_files = glob.glob(os.path.join(args.input_dir, "*.sql"))
        for sql_file in sql_files:
            with open(sql_file, "r") as f:
                sql = f.read().split(";")
                node = pglast.parse_sql(sql)
                extractor = SQLInfoExtractor()
                extractor(node)

                info = extractor.info
                pprint(info)
    else:
        with open(args.input, "r") as f:
            sqls = f.read().split(";")
            for sql_id, sql in enumerate(sqls):
                if sql.strip() and "--select" not in sql:
                    try:
                        print(f"processing {sql_id}, {sql} \n")
                        node = pglast.parse_sql(sql)
                        extractor = SQLInfoExtractor()
                        extractor(node)

                        info = extractor.info
                        pprint(info)
                    except Exception as e:
                        print("ERROR", sql)
                        print(e)
                        print("-" * 80)


if __name__ == "__main__":
    main()
