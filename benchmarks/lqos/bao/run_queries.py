import psycopg2
import os
import sys
import random
from time import time, sleep
import datetime
import argparse
import glob

USE_BAO = True
TIMEOUT_LIMIT = 6 * 60 * 1000
NUM_EXECUTIONS = 3

# PostgreSQL connection defaults
PG_HOST = "172.17.0.1"
PG_USER = "postgres"
PG_PASSWORD = "postgres"


# https://stackoverflow.com/questions/312443/
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def pg_connection_string(db_name, port=5430):
    return f"dbname={db_name} user={PG_USER} password={PG_PASSWORD} host={PG_HOST} port={port}"


def prewarm_database(db_name, port=5430):
    """Prewarm all tables and indexes using pg_prewarm."""
    print(f"Prewarming database {db_name}...", flush=True)
    try:
        conn = psycopg2.connect(pg_connection_string(db_name=db_name, port=port))
        conn.autocommit = True
        cursor = conn.cursor()

        # Ensure pg_prewarm extension exists
        cursor.execute("CREATE EXTENSION IF NOT EXISTS pg_prewarm;")

        # Get all tables in public schema
        cursor.execute("""
            SELECT schemaname, tablename
            FROM pg_tables
            WHERE schemaname = 'public'
            ORDER BY tablename
        """)
        tables = [(r[0], r[1]) for r in cursor.fetchall()]

        # Get all indexes in public schema
        cursor.execute("""
            SELECT schemaname, indexname
            FROM pg_indexes
            WHERE schemaname = 'public'
            ORDER BY indexname
        """)
        indexes = [(r[0], r[1]) for r in cursor.fetchall()]

        # Prewarm tables
        table_count = 0
        for schemaname, tablename in tables:
            try:
                rel = f"{schemaname}.{tablename}"
                cursor.execute("SELECT pg_prewarm(%s::regclass);", (rel,))
                cursor.fetchone()
                table_count += 1
            except Exception as e:
                print(f"Warning: Could not prewarm table {tablename}: {e}", flush=True)

        # Prewarm indexes
        index_count = 0
        for schemaname, indexname in indexes:
            try:
                rel = f"{schemaname}.{indexname}"
                cursor.execute("SELECT pg_prewarm(%s::regclass);", (rel,))
                cursor.fetchone()
                index_count += 1
            except Exception as e:
                print(f"Warning: Could not prewarm index {indexname}: {e}", flush=True)

        cursor.close()
        conn.close()
        print(f"Prewarmed {table_count} tables and {index_count} indexes", flush=True)
        return True
    except Exception as e:
        print(f"Error prewarming database: {e}", flush=True)
        return False


def run_query(sql, bao_select=False, bao_reward=False, db_name='imdbload', port=5430):
    while True:
        conn = None
        try:
            conn = psycopg2.connect(pg_connection_string(db_name=db_name, port=port))
            cur = conn.cursor()

            # Hardcode bao_host to fixed IP given in docker-compose
            # 172.17.0.1, the default Docker bridge gateway
            cur.execute("SET bao_host TO '172.17.0.1'")
            cur.execute(f"SET enable_bao TO {bao_select or bao_reward}")
            cur.execute(f"SET enable_bao_selection TO {bao_select}")
            cur.execute(f"SET enable_bao_rewards TO {bao_reward}")
            cur.execute("SET bao_num_arms TO 25")
            cur.execute(f"SET statement_timeout TO {TIMEOUT_LIMIT}")

            # As visible in the #should_report_reward method of the pg_extension
            # found in pg_extension/bao_util.h, EXPLAIN (and ANALYZE) queries are not
            # put into the experience buffer and need to be run without EXPLAIN to
            # ensure that they are used to train Bao
            #
            if bao_reward:
                cur.execute(sql)
                cur.fetchall()

            # Execute once more to extract planning (+= Bao inference) and execution times
            cur.execute(f"EXPLAIN (ANALYZE, VERBOSE, FORMAT JSON) {sql}")
            result = cur.fetchall()[0][0][-1]

            measurement = {
                'execution_time': result['Execution Time'],
                'planning_time': result['Planning Time']
            }

            conn.close()
            break
        except Exception as e:
            print("An unexpected exception OR timeout occured during database querying:", e, flush=True)
            if conn is not None:
                conn.close()
            return {
                'execution_time': 2 * TIMEOUT_LIMIT,
                'planning_time': 2 * TIMEOUT_LIMIT
            }

    return measurement


def current_timestamp_str():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')


def write_to_file(file_path, output_string):
    # print(output_string, flush=True)
    with open(file_path, 'a') as f:
        f.write(output_string)
        f.write(os.linesep)


def main(args):
    # Look for .sql files
    pattern = os.path.join(args.query_dir, '**/*.sql')
    query_paths = sorted(glob.glob(pattern, recursive=True))
    print(f"Found {len(query_paths)} queries in {args.query_dir} and its subdirectories.", flush=True)

    queries = []
    for fp in query_paths:
        with open(fp) as f:
            query = f.read()
        queries.append((fp, query))
    print("Using Bao:", USE_BAO, flush=True)

    db_name = args.database_name
    print("Running against DB:", db_name, flush=True)

    # Prewarm database before training
    prewarm_database(db_name, args.db_port)

    random.seed(42)

    queries_to_run = 500 if len(queries) < 500 else len(queries)
    query_sequence = random.choices(queries, k=queries_to_run)
    pg_chunks, *bao_chunks = list(chunks(query_sequence, 25))

    print("Executing queries using PG optimizer for initial training", flush=True)

    if os.path.exists(args.output_file):
        raise FileExistsError(f"The file {args.output_file} already exists, stopping.")

    for q_idx, (fp, q) in enumerate(pg_chunks):
        # Warm up the cache
        for iteration in range(NUM_EXECUTIONS - 1):
            measurement = run_query(q, db_name=db_name, port=args.db_port)
            output_string = f"x, {q_idx}, {iteration}, {current_timestamp_str()}, {fp}, {measurement['planning_time']}, {measurement['execution_time']}, PG"
            write_to_file(args.output_file, output_string)

        measurement = run_query(q, bao_reward=True, db_name=db_name, port=args.db_port)
        output_string = f"x, {q_idx}, {NUM_EXECUTIONS - 1}, {current_timestamp_str()}, {fp}, {measurement['planning_time']}, {measurement['execution_time']}, PG"
        write_to_file(args.output_file, output_string)

    for c_idx, chunk in enumerate(bao_chunks):
        print("===" * 30, flush=True)
        print(f"Iteration over chunk {c_idx + 1}/{len(bao_chunks)}...", flush=True)
        if USE_BAO:
            print(f"[{current_timestamp_str()}]\t[{c_idx + 1}/{len(bao_chunks)}]\tRetraining Bao...", flush=True)
            os.system("cd bao_server && CUDA_VISIBLE_DEVICES="" python3 baoctl.py --retrain")
            # os.system("cd bao_server && python3 baoctl.py --retrain")
            os.system("sync")
            print(f"[{current_timestamp_str()}]\t[{c_idx + 1}/{len(bao_chunks)}]\tRetraining done.", flush=True)

        for q_idx, (fp, q) in enumerate(chunk):
            # Warm up the cache
            for iteration in range(NUM_EXECUTIONS - 1):
                measurement = run_query(q, bao_reward=False, bao_select=USE_BAO, db_name=db_name, port=args.db_port)
                output_string = f"{c_idx}, {q_idx}, {iteration}, {current_timestamp_str()}, {fp}, {measurement['planning_time']}, {measurement['execution_time']}, Bao"
                write_to_file(args.output_file, output_string)

            measurement = run_query(q, bao_reward=USE_BAO, bao_select=USE_BAO, db_name=db_name, port=args.db_port)
            output_string = f"{c_idx}, {q_idx}, {NUM_EXECUTIONS - 1}, {current_timestamp_str()}, {fp}, {measurement['planning_time']}, {measurement['execution_time']}, Bao"
            write_to_file(args.output_file, output_string)


# Example Call:
#
# python3 run_queries.py --query_dir queries/job__base_query_split_1/train --output_file train__bao__base_query_split_1.txt
#
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--database_name', type=str, default='imdbload', help='Database name to query against')
    parser.add_argument('--query_dir', type=str, required=True,
                        help='Directory which contains all the *training* queries')
    parser.add_argument('--output_file', type=str, required=True, help='File in which to store the results')
    parser.add_argument('--db-port', type=int, default=5430, help='PostgreSQL port (default: 5430)')

    args = parser.parse_args()
    main(args)
