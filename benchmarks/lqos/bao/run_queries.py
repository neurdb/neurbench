import psycopg2
import os
import sys
import random
import shutil
from time import time, sleep
import datetime
import argparse
import glob

USE_BAO = True
TIMEOUT_LIMIT = 6 * 60 * 1000
NUM_EXECUTIONS = 3

# Execution mode: "single" (prewarm + 1 run) or "triple" (3 runs)
EXECUTION_MODE = "single"

# Early stopping and checkpoint settings
CHECKPOINT_INTERVAL = 20  # Save model every N iterations
TEST_INTERVAL = 20  # Run test every N iterations
EARLY_STOP_PATIENCE = 2  # Stop if no improvement for N consecutive tests
IMPROVEMENT_THRESHOLD = 0.03  # Minimum improvement ratio to count as "improved" (3%)

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


def save_checkpoint(iteration, checkpoint_dir):
    """Save model checkpoint at current iteration."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_iter_{iteration}")

    # Copy current model to checkpoint
    default_model = "bao_server/bao_default_model"
    if os.path.exists(default_model):
        if os.path.exists(checkpoint_path):
            shutil.rmtree(checkpoint_path)
        shutil.copytree(default_model, checkpoint_path)
        print(f"[Checkpoint] Saved model at iteration {iteration} to {checkpoint_path}", flush=True)
        return checkpoint_path
    else:
        print(f"[Checkpoint] Warning: No model to save at iteration {iteration}", flush=True)
        return None


def run_test_evaluation(test_queries, db_name, db_port, iteration=None, save_dir=None):
    """Run test queries and return total execution time.

    If save_dir is provided, save detailed results to a file.
    """
    total_time = 0
    timeout_count = 0
    query_results = []

    for fp, sql in test_queries:
        query_name = os.path.basename(fp).replace('.sql', '')
        exec_time = None
        status = 'ok'

        try:
            conn = psycopg2.connect(pg_connection_string(db_name=db_name, port=db_port))
            cur = conn.cursor()

            cur.execute("SET bao_host TO '172.17.0.1'")
            cur.execute("SET enable_bao TO on")
            cur.execute("SET enable_bao_selection TO on")
            cur.execute("SET enable_bao_rewards TO off")
            cur.execute("SET bao_num_arms TO 25")
            cur.execute(f"SET statement_timeout TO {TIMEOUT_LIMIT}")

            cur.execute(f"EXPLAIN (ANALYZE, FORMAT JSON) {sql}")
            result = cur.fetchall()[0][0][-1]
            exec_time = result['Execution Time']
            total_time += exec_time

            conn.close()
        except Exception as e:
            print(f"[Test] Query timeout or error: {query_name}", flush=True)
            exec_time = 2 * TIMEOUT_LIMIT
            total_time += exec_time
            timeout_count += 1
            status = 'timeout'
            if conn:
                conn.close()

        query_results.append({
            'query': query_name,
            'time_ms': exec_time,
            'status': status
        })

    # Save results to file if save_dir is provided
    if save_dir and iteration is not None:
        os.makedirs(save_dir, exist_ok=True)
        result_file = os.path.join(save_dir, f"test_iter_{iteration}.csv")
        with open(result_file, 'w') as f:
            f.write("query,time_ms,status\n")
            for r in query_results:
                f.write(f"{r['query']},{r['time_ms']:.3f},{r['status']}\n")
            f.write(f"\n# Total: {total_time:.3f}ms, Timeouts: {timeout_count}\n")
        print(f"[Test] Results saved to {result_file}", flush=True)

    return total_time, timeout_count


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

    # Determine number of queries to run
    min_queries = getattr(args, 'min_queries', None)
    if min_queries is not None:
        # Use min-queries as minimum (sample up if needed, use all if more available)
        # This matches Lero's behavior
        if len(queries) < min_queries:
            queries_to_run = min_queries
            print(f"Using --min-queries={min_queries} (sampling up from {len(queries)})", flush=True)
        else:
            queries_to_run = len(queries)
            print(f"Using all {len(queries)} queries (>= min-queries={min_queries})", flush=True)
    elif EXECUTION_MODE == "single":
        # single mode default: 1500 queries
        queries_to_run = 1500 if len(queries) < 1500 else len(queries)
    else:
        # triple mode default: 500 queries
        queries_to_run = 500 if len(queries) < 500 else len(queries)

    print(f"Will run {queries_to_run} queries (from {len(queries)} unique queries)", flush=True)
    query_sequence = random.choices(queries, k=queries_to_run)
    pg_chunks, *bao_chunks = list(chunks(query_sequence, 25))

    print("Executing queries using PG optimizer for initial training", flush=True)

    if os.path.exists(args.output_file):
        raise FileExistsError(f"The file {args.output_file} already exists, stopping.")

    # Determine number of warmup iterations based on execution mode
    warmup_iterations = 0 if EXECUTION_MODE == "single" else NUM_EXECUTIONS - 1

    for q_idx, (fp, q) in enumerate(pg_chunks):
        # Warm up the cache (skipped in single mode since we prewarm the database)
        for iteration in range(warmup_iterations):
            measurement = run_query(q, db_name=db_name, port=args.db_port)
            output_string = f"x, {q_idx}, {iteration}, {current_timestamp_str()}, {fp}, {measurement['planning_time']}, {measurement['execution_time']}, PG"
            write_to_file(args.output_file, output_string)

        measurement = run_query(q, bao_reward=True, db_name=db_name, port=args.db_port)
        output_string = f"x, {q_idx}, {warmup_iterations}, {current_timestamp_str()}, {fp}, {measurement['planning_time']}, {measurement['execution_time']}, PG"
        write_to_file(args.output_file, output_string)

    # Early stopping state
    best_test_time = float('inf')
    no_improvement_count = 0
    test_history = []

    # Get parameters from args or use defaults
    test_interval = getattr(args, 'test_interval', TEST_INTERVAL) or TEST_INTERVAL
    checkpoint_interval = getattr(args, 'checkpoint_interval', CHECKPOINT_INTERVAL) or CHECKPOINT_INTERVAL
    early_stop_patience = getattr(args, 'early_stop_patience', EARLY_STOP_PATIENCE) or EARLY_STOP_PATIENCE

    # Checkpoint directory (priority: args > env var > default)
    checkpoint_dir = getattr(args, 'checkpoint_dir', None)
    if checkpoint_dir is None:
        checkpoint_dir = os.environ.get('BAO_CHECKPOINT_DIR')
    if checkpoint_dir is None:
        # Default: use output file directory with timestamp
        output_dir = os.path.dirname(args.output_file)
        checkpoint_dir = os.path.join(output_dir, 'bao_checkpoints')

    print(f"[Config] Test interval: {test_interval}, Checkpoint interval: {checkpoint_interval}, Early stop patience: {early_stop_patience}", flush=True)
    print(f"[Config] Checkpoint directory: {checkpoint_dir}", flush=True)

    # Test queries (use all training queries for testing, or a subset)
    test_queries = queries  # Use original queries for testing

    early_stopped = False
    for c_idx, chunk in enumerate(bao_chunks):
        print("===" * 30, flush=True)
        print(f"Iteration over chunk {c_idx + 1}/{len(bao_chunks)}...", flush=True)
        iteration_num = c_idx + 1  # 1-indexed

        if USE_BAO:
            print(f"[{current_timestamp_str()}]\t[{iteration_num}/{len(bao_chunks)}]\tRetraining Bao...", flush=True)
            os.system("cd bao_server && CUDA_VISIBLE_DEVICES=2 python3 baoctl.py --retrain")
            os.system("sync")
            print(f"[{current_timestamp_str()}]\t[{iteration_num}/{len(bao_chunks)}]\tRetraining done.", flush=True)

            # Check if we should run test and save checkpoint (right after retrain)
            if iteration_num % test_interval == 0:
                print(f"\n[Test] Running test evaluation at iteration {iteration_num}...", flush=True)

                # Save checkpoint first
                if iteration_num % checkpoint_interval == 0:
                    save_checkpoint(iteration_num, checkpoint_dir)

                # Run test and save results
                test_time, timeout_count = run_test_evaluation(
                    test_queries, db_name, args.db_port,
                    iteration=iteration_num, save_dir=checkpoint_dir
                )
                test_history.append((iteration_num, test_time, timeout_count))

                print(f"[Test] Iteration {iteration_num}: Total time = {test_time/1000:.2f}s, Timeouts = {timeout_count}", flush=True)
                print(f"[Test] Best so far: {best_test_time/1000:.2f}s", flush=True)

                # Check for improvement (must be at least IMPROVEMENT_THRESHOLD faster)
                improvement_ratio = (best_test_time - test_time) / best_test_time if best_test_time > 0 else 0
                if test_time < best_test_time * (1 - IMPROVEMENT_THRESHOLD):
                    print(f"[Test] Improvement! {best_test_time/1000:.2f}s -> {test_time/1000:.2f}s ({improvement_ratio*100:.1f}% faster)", flush=True)
                    best_test_time = test_time
                    no_improvement_count = 0

                    # Save best model (overwrite previous best)
                    save_checkpoint("best", checkpoint_dir)
                else:
                    no_improvement_count += 1
                    if test_time < best_test_time:
                        print(f"[Test] Minor improvement ({improvement_ratio*100:.1f}% < {IMPROVEMENT_THRESHOLD*100:.0f}%), not counted. Count: {no_improvement_count}/{early_stop_patience}", flush=True)
                    else:
                        print(f"[Test] No improvement. Count: {no_improvement_count}/{early_stop_patience}", flush=True)

                # Early stopping check
                if no_improvement_count >= early_stop_patience:
                    print(f"\n[Early Stop] No improvement for {early_stop_patience} consecutive tests. Stopping training.", flush=True)
                    print(f"[Early Stop] Best test time: {best_test_time/1000:.2f}s", flush=True)
                    early_stopped = True
                    break

        # Execute queries in chunk (collect training data)
        for q_idx, (fp, q) in enumerate(chunk):
            # Warm up the cache (skipped in single mode since we prewarm the database)
            for iteration in range(warmup_iterations):
                measurement = run_query(q, bao_reward=False, bao_select=USE_BAO, db_name=db_name, port=args.db_port)
                output_string = f"{c_idx}, {q_idx}, {iteration}, {current_timestamp_str()}, {fp}, {measurement['planning_time']}, {measurement['execution_time']}, Bao"
                write_to_file(args.output_file, output_string)

            measurement = run_query(q, bao_reward=USE_BAO, bao_select=USE_BAO, db_name=db_name, port=args.db_port)
            output_string = f"{c_idx}, {q_idx}, {warmup_iterations}, {current_timestamp_str()}, {fp}, {measurement['planning_time']}, {measurement['execution_time']}, Bao"
            write_to_file(args.output_file, output_string)

        if early_stopped:
            break

    # Save final model
    save_checkpoint("final", checkpoint_dir)

    # Restore best model to bao_server/bao_default_model for testing
    best_model_path = os.path.join(checkpoint_dir, "checkpoint_iter_best")
    default_model_path = "bao_server/bao_default_model"
    if os.path.exists(best_model_path):
        if os.path.exists(default_model_path):
            shutil.rmtree(default_model_path)
        shutil.copytree(best_model_path, default_model_path)
        print(f"[Model] Restored best model to {default_model_path} (from {best_model_path})", flush=True)
    else:
        print(f"[Model] No best model found, using final model for testing", flush=True)

    # Print final summary
    print("\n" + "===" * 30, flush=True)
    print("Training completed!", flush=True)
    print(f"Test history:", flush=True)
    for iter_num, test_time, timeouts in test_history:
        print(f"  Iteration {iter_num}: {test_time/1000:.2f}s (timeouts: {timeouts})", flush=True)
    print(f"Best test time: {best_test_time/1000:.2f}s", flush=True)

    # Save test history summary to file
    if test_history:
        summary_file = os.path.join(checkpoint_dir, "test_summary.csv")
        os.makedirs(checkpoint_dir, exist_ok=True)
        with open(summary_file, 'w') as f:
            f.write("iteration,total_time_ms,total_time_s,timeouts\n")
            for iter_num, test_time, timeouts in test_history:
                f.write(f"{iter_num},{test_time:.3f},{test_time/1000:.2f},{timeouts}\n")
            f.write(f"\n# Best: {best_test_time:.3f}ms = {best_test_time/1000:.2f}s\n")
        print(f"Test summary saved to {summary_file}", flush=True)

    print(f"\nModel locations:", flush=True)
    print(f"  Final model: {checkpoint_dir}/checkpoint_iter_final/", flush=True)
    print(f"  Best model:  {checkpoint_dir}/checkpoint_iter_best/", flush=True)


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
    parser.add_argument('--min-queries', type=int, default=None,
                        help='Minimum number of training queries (sample with replacement if needed)')
    parser.add_argument('--checkpoint-dir', type=str, default=None,
                        help='Directory to save model checkpoints (default: output_file_dir/bao_checkpoints)')
    parser.add_argument('--test-interval', type=int, default=20,
                        help='Run test every N iterations (default: 20)')
    parser.add_argument('--checkpoint-interval', type=int, default=20,
                        help='Save checkpoint every N iterations (default: 20)')
    parser.add_argument('--early-stop-patience', type=int, default=2,
                        help='Stop if no improvement for N consecutive tests (default: 2)')

    args = parser.parse_args()
    main(args)
