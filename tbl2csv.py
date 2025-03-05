import argparse
import os
from typing import List, Optional

from neurbench.config import DB_MAP


def process(input_path: str, output_path: str, columns: List[str]) -> Optional[str]:
    try:
        tbl = open(input_path + ".tbl", "r")
        csv = open(output_path + ".csv", "w+")
        lines = tbl.readlines()

        header_line = ",".join(columns) + "\n"
        csv.write(header_line)

        for line in lines:
            line = line.replace(",", ";")
            line = line.replace("|", ",")
            csv.write(line)

        tbl.close()
        csv.close()

    except OSError as e:
        return f"processing {input_path}: {e}"

    return None


def check_and_makedirs(dirs: List[str]):
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Preprocessing query SQLs")

    parser.add_argument(
        "-d", "--dbname", default="tpch", help="Database name (default: tpch)"
    )
    parser.add_argument(
        "-i",
        "--input",
        default=".",
        help="Directory to the input files (default: .)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=".",
        help="Directory to output CSV files (default: .)",
    )

    args = parser.parse_args()

    for k, v in args.__dict__.items():
        if isinstance(v, str) and "{table}" in v:
            args.__dict__[k] = v.format(dbname=args.dbname)

    print(args)

    check_and_makedirs([args.input, args.output])

    column_names_map = DB_MAP[args.dbname]["column_names"]
    for table in column_names_map.keys():
        in_file_path = os.path.join(args.input, f"{table}")
        out_file_path = os.path.join(args.output, f"{table}")

        print(f"INFO  processing table {table} from {in_file_path} to {out_file_path} ...")
        err = process(in_file_path, out_file_path, column_names_map[table])
        if err:
            print(f"ERROR {err}")
        else:
            print("INFO  done!")


if __name__ == "__main__":
    main()
