#!/usr/bin/env python3
"""
Bao Learned Query Optimizer Training Script
This script automates the training process for Bao LQO
Replicates the functionality of run_queries.py with sample-based query execution
"""

import os
import sys
import time
import subprocess
import signal
import psutil
import json
import glob
import random
import psycopg2
import datetime
from pathlib import Path

class BaoTrainer:
    def __init__(self, bao_dir=".", query_dir=None, database_name='imdbload', output_file="bao_training_results.log", db_port=5432):
        self.bao_dir = Path(bao_dir).resolve()
        self.bao_server_dir = self.bao_dir / "bao_server"
        self.server_process = None
        self.original_dir = os.getcwd()

        # Query execution parameters (required)
        if not query_dir:
            raise ValueError("query_dir is required for training")

        self.query_dir = query_dir
        self.database_name = database_name
        self.db_port = db_port
        self.output_file = output_file
        self.num_executions = 3
        self.timeout_limit = 3 * 60 * 1000  # 3 minutes in milliseconds
        self.use_bao = True

        # Configurable parameters
        self.training_timeout = 300
        self.server_startup_delay = 5
        
    def __init_environment(self):
        """Initialize the Bao environment"""
        print("Initializing Bao environment...")
        
        # Check if we're in the right directory
        if not (self.bao_dir / "bao_server").exists():
            raise RuntimeError(f"Bao server directory not found in {self.bao_dir}")
        
        # Activate virtual environment if it exists
        # Note: Skipping env activation via os.system as it doesn't affect Python process
        # If you need virtual env, activate it before running this script
        # env_script = self.bao_dir / "activate_env.sh"
        # if env_script.exists():
        #     print("Activating virtual environment...")
        #     os.system(f"bash -c 'source {env_script}'")
        
        # Install requirements if needed
        requirements_file = self.bao_dir / "requirements.txt"
        if requirements_file.exists():
            print("Checking dependencies...")
            try:
                import psycopg2
                import torch
                print("Dependencies already installed")
            except ImportError:
                print("Installing requirements...")
                subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(requirements_file)], 
                             check=True, cwd=self.bao_dir)
        
    def start_server(self):
        """Start the Bao server"""
        print("Starting Bao server...")
        
        # Kill any existing Bao server processes
        self.kill_existing_servers()
        
        # Start new server
        server_script = self.bao_server_dir / "main.py"
        if not server_script.exists():
            raise RuntimeError(f"Server script not found: {server_script}")
        
        # Start server in background
        self.server_process = subprocess.Popen(
            [sys.executable, str(server_script)],
            cwd=self.bao_server_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait for server to start
        print(f"Waiting {self.server_startup_delay} seconds for server to start...")
        time.sleep(self.server_startup_delay)
        
        # Check if server is running
        if self.server_process.poll() is not None:
            raise RuntimeError("Failed to start Bao server")
        
        print(f"Bao server started with PID: {self.server_process.pid}")
        
    def kill_existing_servers(self):
        """Kill any existing Bao server processes"""
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if proc.info['cmdline'] and 'main.py' in proc.info['cmdline']:
                    print(f"Killing existing Bao server process: {proc.info['pid']}")
                    proc.terminate()
                    proc.wait(timeout=5)
            except (psutil.NoSuchProcess, psutil.TimeoutExpired):
                pass

    def pg_connection_string(self):
        """Generate PostgreSQL connection string"""
        return f"dbname={self.database_name} user=postgres password=postgres host=172.17.0.1 port={self.db_port}"

    def run_query(self, sql, bao_select=False, bao_reward=False):
        """Execute a single query and measure performance - matches run_queries.py logic"""
        while True:
            conn = None
            try:
                # Add connection timeout to prevent hanging
                conn = psycopg2.connect(
                    self.pg_connection_string(),
                    connect_timeout=30  # 30 seconds connection timeout
                )
                cur = conn.cursor()

                # Configure Bao settings
                cur.execute("SET bao_host TO '172.17.0.1'")
                cur.execute(f"SET enable_bao TO {bao_select or bao_reward}")
                cur.execute(f"SET enable_bao_selection TO {bao_select}")
                cur.execute(f"SET enable_bao_rewards TO {bao_reward}")
                cur.execute("SET bao_num_arms TO 25")
                cur.execute(f"SET statement_timeout TO {self.timeout_limit}")

                # Execute query for reward collection
                if bao_reward:
                    cur.execute(sql)
                    cur.fetchall()

                # Execute with EXPLAIN ANALYZE to get timing
                cur.execute(f"EXPLAIN (ANALYZE, VERBOSE, FORMAT JSON) {sql}")
                result = cur.fetchall()[0][0][-1]

                measurement = {
                    'execution_time': result['Execution Time'],
                    'planning_time': result['Planning Time']
                }

                conn.close()
                break

            except Exception as e:
                print(f"An unexpected exception OR timeout occured during database querying: {e}")
                if conn is not None:
                    try:
                        conn.close()
                    except:
                        pass
                return {
                    'execution_time': 2 * self.timeout_limit,
                    'planning_time': 2 * self.timeout_limit
                }

        return measurement

    def current_timestamp_str(self):
        """Get current timestamp string"""
        return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')

    def write_to_file(self, file_path, output_string):
        """Write output to file and print"""
        print(output_string)
        with open(file_path, 'a') as f:
            f.write(output_string)
            f.write(os.linesep)

    def chunks(self, lst, n):
        """Yield successive n-sized chunks from lst"""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    def load_queries(self):
        """Load queries from query directory"""
        if not self.query_dir:
            print("No query directory specified, skipping query execution")
            return []

        pattern = os.path.join(self.query_dir, '**/*.sql')
        query_paths = sorted(glob.glob(pattern, recursive=True))
        print(f"Found {len(query_paths)} queries in {self.query_dir}")

        queries = []
        for fp in query_paths:
            with open(fp) as f:
                query = f.read()
            queries.append((fp, query))

        return queries

    def execute_training_queries(self):
        """Execute queries following run_queries.py pattern"""
        print("="*80)
        print("Executing training queries with sample-based approach...")
        print("="*80)

        # Load queries
        queries = self.load_queries()
        if not queries:
            print("No queries found")
            return False

        # Sample 500 queries (or all if less than 500)
        random.seed(42)
        queries_to_run = 500 if len(queries) < 500 else len(queries)
        query_sequence = random.choices(queries, k=queries_to_run)
        print(f"Sampled {queries_to_run} queries for training")

        # Split into chunks of 25
        pg_chunks, *bao_chunks = list(self.chunks(query_sequence, 25))

        # Check if output file exists
        output_path = self.bao_dir / self.output_file
        if output_path.exists():
            print(f"Removing existing output file: {output_path}")
            output_path.unlink()

        # Execute initial chunk with PG optimizer
        print("\n" + "="*80)
        print("Phase 1: Executing queries using PG optimizer for initial training")
        print("="*80)

        # Track failures in Phase 1
        phase1_failures = 0
        consecutive_failures = 0

        for q_idx, (fp, q) in enumerate(pg_chunks):
            # Warm up the cache (first NUM_EXECUTIONS-1 runs)
            for iteration in range(self.num_executions - 1):
                measurement = self.run_query(q)
                output_string = f"x, {q_idx}, {iteration}, {self.current_timestamp_str()}, {fp}, {measurement['planning_time']}, {measurement['execution_time']}, PG"
                self.write_to_file(output_path, output_string)

            # Final execution with reward collection
            measurement = self.run_query(q, bao_reward=True)
            output_string = f"x, {q_idx}, {self.num_executions - 1}, {self.current_timestamp_str()}, {fp}, {measurement['planning_time']}, {measurement['execution_time']}, PG"
            self.write_to_file(output_path, output_string)

            # Check if query failed
            query_failed = measurement['execution_time'] >= 2 * self.timeout_limit
            if query_failed:
                phase1_failures += 1
                consecutive_failures += 1
                print(f"⚠ Query failed: {fp}")

                # Abort if too many consecutive failures
                if consecutive_failures >= 5:
                    print(f"\n✗ ERROR: {consecutive_failures} consecutive queries failed in Phase 1!")
                    print("This usually indicates a database connection problem.")
                    print("Aborting training to prevent further issues.")
                    return False
            else:
                consecutive_failures = 0

        if phase1_failures > 0:
            print(f"⚠ Phase 1 completed with {phase1_failures} failed queries")
        else:
            print(f"✓ Completed initial training with {len(pg_chunks)} queries")

        # Execute subsequent chunks with Bao
        print("\n" + "="*80)
        print(f"Phase 2: Executing {len(bao_chunks)} chunks with Bao (25 queries each)")
        print("="*80)

        # Track failures in Phase 2
        phase2_failures = 0
        consecutive_failures = 0

        for c_idx, chunk in enumerate(bao_chunks):
            print("===" * 30, flush=True)
            print(f"Iteration over chunk {c_idx + 1}/{len(bao_chunks)}...")
            if self.use_bao:
                print(f"[{self.current_timestamp_str()}]\t[{c_idx + 1}/{len(bao_chunks)}]\tRetraining Bao...", flush=True)
                # Retrain the model - using os.system to match run_queries.py exactly
                retrain_cmd = f"cd {self.bao_server_dir} && {sys.executable} baoctl.py --retrain"
                return_code = os.system(retrain_cmd)
                os.system("sync")
                print(f"[{self.current_timestamp_str()}]\t[{c_idx + 1}/{len(bao_chunks)}]\tRetraining done.", flush=True)

            # Execute chunk queries
            for q_idx, (fp, q) in enumerate(chunk):
                # Warm up the cache
                for iteration in range(self.num_executions - 1):
                    measurement = self.run_query(q, bao_reward=False, bao_select=self.use_bao)
                    output_string = f"{c_idx}, {q_idx}, {iteration}, {self.current_timestamp_str()}, {fp}, {measurement['planning_time']}, {measurement['execution_time']}, Bao"
                    self.write_to_file(output_path, output_string)

                # Final execution with reward collection
                measurement = self.run_query(q, bao_reward=self.use_bao, bao_select=self.use_bao)
                output_string = f"{c_idx}, {q_idx}, {self.num_executions - 1}, {self.current_timestamp_str()}, {fp}, {measurement['planning_time']}, {measurement['execution_time']}, Bao"
                self.write_to_file(output_path, output_string)

                # Check if query failed
                query_failed = measurement['execution_time'] >= 2 * self.timeout_limit
                if query_failed:
                    phase2_failures += 1
                    consecutive_failures += 1
                    print(f"⚠ Query failed: {fp}")

                    # Abort if too many consecutive failures
                    if consecutive_failures >= 5:
                        print(f"\n✗ ERROR: {consecutive_failures} consecutive queries failed in Phase 2!")
                        print("This usually indicates a database connection problem.")
                        print("Aborting training to prevent further issues.")
                        return False
                else:
                    consecutive_failures = 0

        # Final statistics
        total_failures = phase1_failures + phase2_failures
        total_queries = len(pg_chunks) + sum(len(chunk) for chunk in bao_chunks)

        print("\n" + "="*80)
        if total_failures > 0:
            print(f"⚠ Training completed with {total_failures} failed queries out of {total_queries}")
            print(f"  Phase 1: {phase1_failures} failures")
            print(f"  Phase 2: {phase2_failures} failures")
        else:
            print(f"✓ Training completed successfully! All queries passed.")
        print(f"Results saved to: {output_path}")
        print("="*80)

        # Return False if too many failures (>50%)
        success_rate = (total_queries - total_failures) / total_queries if total_queries > 0 else 0
        return success_rate >= 0.5
    
    def cleanup(self):
        """Clean up resources"""
        if self.server_process:
            print("Stopping Bao server...")
            try:
                self.server_process.terminate()
                self.server_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.server_process.kill()
            finally:
                self.server_process = None
    
    def run_training_pipeline(self):
        """Run the complete training pipeline"""
        try:
            # Change to bao directory
            os.chdir(self.bao_dir)

            # Initialize environment
            self.__init_environment()

            # Start server
            self.start_server()

            print("\n" + "="*80)
            print("Using query-based training (run_queries.py approach)")
            print("="*80)

            # Execute training queries
            if not self.execute_training_queries():
                raise RuntimeError("Query-based training failed")

            print("[DONE] Bao training with queries completed successfully!")
            return True

        except Exception as e:
            print(f"[FAILED] Training failed: {e}")
            return False

        finally:
            # Cleanup
            self.cleanup()
            # Return to original directory
            os.chdir(self.original_dir)

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Train Bao Learned Query Optimizer using query-based approach (like run_queries.py)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python train_bao.py --query-dir ./queries/job/train --database-name imdbload

  # Custom output file
  python train_bao.py --query-dir ./queries/job/train --output-file training_results.log

  # With timeout and startup delay settings
  python train_bao.py --query-dir ./queries/job/train --training-timeout 600 --server-startup-delay 10

How it works:
  - Samples 500 queries from the query directory (or all if less than 500)
  - Splits queries into 25-query chunks
  - First chunk: executes with PG optimizer to collect initial training data
  - Subsequent chunks: retrains Bao model every 25 queries
  - Each query executes 3 times (2 for warm-up, 1 for reward collection)
        """
    )

    parser.add_argument(
        "--query-dir",
        type=str,
        required=True,
        help="Path to directory containing training queries (REQUIRED)"
    )

    parser.add_argument(
        "--database-name",
        type=str,
        default="imdbload",
        help="PostgreSQL database name (default: imdbload)"
    )

    parser.add_argument(
        "--training-timeout",
        type=int,
        default=300,
        help="Timeout for training steps in seconds (default: 300)"
    )

    parser.add_argument(
        "--server-startup-delay",
        type=int,
        default=5,
        help="Delay after starting server in seconds (default: 5)"
    )

    parser.add_argument(
        "--bao-dir",
        type=str,
        default=".",
        help="Path to Bao directory (default: current directory)"
    )

    parser.add_argument(
        "--output-file",
        type=str,
        default="bao_training_results.log",
        help="Output file for training results (default: bao_training_results.log)"
    )

    parser.add_argument(
        "--db-port",
        type=int,
        default=5432,
        help="PostgreSQL port (default: 5432)"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )

    args = parser.parse_args()

    # Create trainer with custom settings
    trainer = BaoTrainer(
        bao_dir=args.bao_dir,
        query_dir=args.query_dir,
        database_name=args.database_name,
        output_file=args.output_file,
        db_port=args.db_port
    )

    # Update trainer settings
    trainer.training_timeout = args.training_timeout
    trainer.server_startup_delay = args.server_startup_delay

    if args.verbose:
        print(f"Configuration:")
        print(f"  Bao directory: {args.bao_dir}")
        print(f"  Query directory: {args.query_dir}")
        print(f"  Database name: {args.database_name}")
        print(f"  Output file: {args.output_file}")
        print(f"  Training timeout: {args.training_timeout}s")
        print(f"  Server startup delay: {args.server_startup_delay}s")

    # Run query-based training
    success = trainer.run_training_pipeline()

    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
