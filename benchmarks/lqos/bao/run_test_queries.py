import psycopg2
import os
import sys
import random
from time import time, sleep
import datetime
import argparse
import glob

TIMEOUT_LIMIT = 6 * 60 * 1000

# PostgreSQL connection defaults
PG_HOST = "172.17.0.1"
PG_USER = "postgres"
PG_PASSWORD = "postgres"


def current_timestamp_str():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')


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


def extract_q_errors(plan):
    est = plan.get('Plan Rows')
    act = plan.get('Actual Rows')
    if est is not None and act not in (None, 0):
        return [max(est / act, act / est)]
    else:
        return []
    # q_errors = []

    # def traverse(node):
    #     est = node.get('Plan Rows')
    #     act = node.get('Actual Rows')
    #     if est is not None and act not in (None, 0):
    #         q_error = max(est / act, act / est)
    #         q_errors.append(q_error)
    #     for subplan in node.get('Plans', []):
    #         traverse(subplan)

    # traverse(plan)
    # return q_errors

def run_query(sql, bao_select=False, bao_reward=False, db_name='imdbload', use_geqo=True, use_bao=True, port=5430, num_executions=1):
    """
    Execute a query with EXPLAIN ANALYZE.

    Args:
        num_executions: Number of times to execute the query (default: 1)
    """
    measurements = []
    try:
        conn = psycopg2.connect(pg_connection_string(db_name=db_name, port=port))
        cur = conn.cursor()

        # Hardcode bao_host to fixed IP given in docker-compose
        cur.execute("SET bao_host TO '172.17.0.1'")
        cur.execute(f"SET enable_bao TO {bao_select or bao_reward}")
        cur.execute(f"SET enable_bao_selection TO {bao_select}")
        cur.execute(f"SET enable_bao_rewards TO {bao_reward}")
        cur.execute("SET bao_num_arms TO 25")
        cur.execute(f"SET statement_timeout TO {TIMEOUT_LIMIT}")

        # cur.execute("SET enable_mergejoin TO off")
        # cur.execute("SET enable_seqscan TO off")
        # cur.execute("SET enable_indexonlyscan TO off")
        # SET enable_nestloop TO off


        if not use_geqo:
            print("disable geqo", flush=True)
            cur.execute(f"SET geqo TO off")

        for i in range(num_executions):
            cur.execute(f"EXPLAIN (ANALYZE, VERBOSE, FORMAT JSON) {sql}")
            result = cur.fetchall()[0][0]

            q_errors = []
            if not use_bao:
                q_errors = extract_q_errors(result[0]['Plan'])

            # Get actual execution time
            actual_time = result[-1]['Execution Time']

            # Get Bao's predicted time if using Bao
            if use_bao:
                bao_hint = result[0]['Bao']['Bao recommended hint']
                predicted_time = result[0]['Bao']['Bao prediction']
                if predicted_time == 'NaN':
                    predicted_time = float('nan')
                else:
                    predicted_time = float(predicted_time)
            else:
                bao_hint = None
                predicted_time = None

            measurements.append({
                'execution_time': actual_time,
                'planning_time': result[-1]['Planning Time'],
                'hint': bao_hint,
                'predicted_time': predicted_time,
                'q_errors': q_errors
            })

            # print(
            #     f"\t{i}: Execution Time: {measurements[-1]['execution_time']:.4f}\tPlanning Time: {measurements[-1]['planning_time']:.4f}\tPredicted Time: {measurements[-1]['predicted_time'] if measurements[-1]['predicted_time'] is not None else 'N/A'}", flush=True)

        conn.close()
    except Exception as e:
        print("An unexpected exception OR timeout occurred:", e, flush=True)
        conn.close()

        tmp = []
        for _ in range(num_executions):
            tmp.append({
                'execution_time': 2 * TIMEOUT_LIMIT,
                'planning_time': 2 * TIMEOUT_LIMIT,
                'hint': None,
                'predicted_time': None,
                'q_errors': []
            })
        return tmp

    return measurements


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

    use_bao = args.use_bao and (not args.use_postgres)
    print("Using Bao:", use_bao, flush=True)

    use_geqo = args.use_geqo
    print("Using GEQO:", use_geqo, flush=True)

    db_name = args.database_name
    print("Running against DB:", db_name, flush=True)

    # Execution mode: single (prewarm + 1 run) or triple (3 runs, use 3rd)
    execution_mode = getattr(args, 'execution_mode', 'single')
    print(f"Execution mode: {execution_mode}", flush=True)

    if execution_mode == 'single':
        num_executions = 1
        result_index = 0  # Use the only execution
        # Prewarm before running queries
        prewarm_database(db_name, args.db_port)
    else:  # triple
        num_executions = 3
        result_index = 2  # Use 3rd execution (index 2)

    random.seed(42)

    print(f"Start running {len(queries)} queries for evaluation...", flush=True)

    if os.path.exists(args.output_file):
        raise FileExistsError(f"The file {args.output_file} already exists, stopping.")

    # Early stop tracking
    early_stop_time = getattr(args, 'early_stop_time', None)
    cumulative_time = 0.0
    early_stopped = False

    for fp, q in queries:
        measurements = run_query(q, bao_select=use_bao, bao_reward=False, db_name=db_name, use_geqo=use_geqo, use_bao=use_bao, port=args.db_port, num_executions=num_executions)

        for i, measurement in enumerate(measurements):
            avg_q_error = sum(measurement['q_errors']) / len(measurement['q_errors']) if measurement['q_errors'] else 'N/A'
            output_string = f"{'x' if measurement['hint'] is None else measurement['hint']}, {i}, {current_timestamp_str()}, {fp}, {measurement['planning_time']}, {measurement['execution_time']}, {measurement['predicted_time'] if measurement['predicted_time'] is not None else 'N/A'}, {'Bao' if use_bao else 'PG'}, {avg_q_error}"
            # print(output_string, flush=True)
            with open(args.output_file, 'a') as f:
                f.write(output_string)
                f.write(os.linesep)

            # Track cumulative time (only the result_index execution)
            if i == result_index:
                cumulative_time += measurement['execution_time']

        # Check early stop after each query
        if early_stop_time and cumulative_time > early_stop_time:
            print(f"[EARLY STOP] Cumulative time ({cumulative_time:.1f}ms) > threshold ({early_stop_time:.1f}ms)", flush=True)
            early_stopped = True
            break

    if early_stopped:
        print(f"Early stopped after {cumulative_time:.1f}ms", flush=True)
    else:
        print(f"Completed all queries, total time: {cumulative_time:.1f}ms", flush=True)


# Example Call:
#
# python3 run_test_queries.py --use_postgres --database_name imdbload --query_dir queries/job --output_file test__postgres__job.txt
#
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_bao', action='store_true', help='Should Bao be used?')
    parser.add_argument('--use_postgres', action='store_true', help='Should Postgres be used?')
    parser.add_argument('--database_name', type=str, default='imdbload', help='Database name to query against')
    parser.add_argument('--query_dir', type=str, required=True, help='Directory which contains all the queries to evaluate')
    parser.add_argument('--output_file', type=str, required=True, help='File in which to store the results')
    parser.add_argument('--use_geqo', action='store_true', default=True, help='Should GEQO be used? (default=True)')
    parser.add_argument('--disable_geqo', action='store_false', dest='use_geqo', help='Should GEQO be disabled?')
    parser.add_argument('--db-port', type=int, default=5430, help='PostgreSQL port (default: 5430)')
    parser.add_argument('--early-stop-time', type=float, default=None, help='Stop when cumulative execution time (ms) exceeds this threshold')
    parser.add_argument('--execution-mode', type=str, default='single', choices=['single', 'triple'],
                        help='Execution mode: single (prewarm + 1 run, default) or triple (3 runs, use 3rd)')

    args = parser.parse_args()

    if (not args.use_bao) and (not args.use_postgres):
        print(f"Need to either select Bao or Postgres to be used. (--use_bao or --use_postgres)", flush=True)
    if args.use_bao and args.use_postgres:
        print("Need to only select Bao or Postgres, not both. (--use_bao or --use_postgres)", flush=True)

    main(args)
