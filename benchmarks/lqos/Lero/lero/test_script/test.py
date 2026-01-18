import argparse
import traceback
import json

from utils import *

# python test.py --query_path ../reproduce/test_query/stats.txt --output_query_latency_file stats.test
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Model training helper")
    parser.add_argument("--query_path",
                        metavar="PATH",
                        help="Load the queries")
    parser.add_argument("--output_query_latency_file", metavar="PATH")

    args = parser.parse_args()
    test_queries = []
    with open(args.query_path, 'r') as f:
        for line in f.readlines():
            arr = line.strip().split("#####")
            test_queries.append((arr[0], arr[1]))
    print("Read", len(test_queries), "test queries.")

    import time
    total_exec_time = 0.0
    query_count = 0

    for (fp, q) in test_queries:
        try:
            do_run_query(q, fp, ["SET enable_lero TO True"], args.output_query_latency_file, True, None, None)
        except:
            print('Error', fp)
            tb_str = traceback.format_exc()  # Returns the traceback as a string
            print(tb_str)  # Print the string

            # waiting for db restart ec.
            time.sleep(20)

        time.sleep(3)

    try:
        with open(args.output_query_latency_file, 'r') as f:
            for line in f:
                parts = line.strip().split("#####")
                if len(parts) >= 2:
                    try:
                        latency_json = json.loads(parts[1])
                        exec_time = latency_json[0].get("Execution Time", 0)
                        total_exec_time += exec_time
                        query_count += 1
                    except (json.JSONDecodeError, IndexError, KeyError):
                        pass
    except Exception as e:
        print(f"Warning: Could not read latency file: {e}")

    print("=" * 60)
    print(f"Completed all queries, total time: {total_exec_time:.1f}ms")
    print(f"Query count: {query_count}")
    print(f"Average time per query: {total_exec_time / query_count:.1f}ms" if query_count > 0 else "N/A")
    print("=" * 60)
