#!/usr/bin/env python3
"""
Bao Learned Query Optimizer Testing Script
This script tests the trained Bao model on test queries
Replicates the functionality of run_test_queries.py
"""

import os
import sys
import time
import subprocess
import psutil
import glob
import random
import psycopg2
import datetime
from pathlib import Path

class BaoTester:
    def __init__(self, bao_dir=".", query_dir=None, database_name='imdbload',
                 output_file="bao_test_results.log", use_bao=True, use_postgres=False, use_geqo=True, db_port=5432):
        self.bao_dir = Path(bao_dir).resolve()
        self.bao_server_dir = self.bao_dir / "bao_server"
        self.server_process = None
        self.original_dir = os.getcwd()

        # Query execution parameters (required)
        if not query_dir:
            raise ValueError("query_dir is required for testing")

        self.query_dir = query_dir
        self.database_name = database_name
        self.db_port = db_port
        self.output_file = output_file
        self.num_executions = 3
        self.timeout_limit = 3 * 60 * 1000  # 3 minutes in milliseconds

        # Test mode configuration
        self.use_bao = use_bao and (not use_postgres)
        self.use_postgres = use_postgres
        self.use_geqo = use_geqo

        # Configurable parameters
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

    def extract_q_errors(self, plan):
        """Extract Q-errors from query plan"""
        est = plan.get('Plan Rows')
        act = plan.get('Actual Rows')
        if est is not None and act not in (None, 0):
            return [max(est / act, act / est)]
        else:
            return []

    def run_query(self, sql):
        """Execute a single query and measure performance"""
        measurements = []
        conn = None
        try:
            conn = psycopg2.connect(self.pg_connection_string())
            cur = conn.cursor()

            # Configure Bao settings
            cur.execute("SET bao_host TO '172.17.0.1'")
            cur.execute(f"SET enable_bao TO {self.use_bao}")
            cur.execute(f"SET enable_bao_selection TO {self.use_bao}")
            cur.execute(f"SET enable_bao_rewards TO False")
            cur.execute("SET bao_num_arms TO 25")
            cur.execute(f"SET statement_timeout TO {self.timeout_limit}")

            # Disable certain operations
            cur.execute("SET enable_mergejoin TO off")
            cur.execute("SET enable_seqscan TO off")
            cur.execute("SET enable_indexonlyscan TO off")

            # GEQO configuration
            if not self.use_geqo:
                print("disable geqo")
                cur.execute("SET geqo TO off")

            # Execute query multiple times
            for i in range(self.num_executions):
                cur.execute(f"EXPLAIN (ANALYZE, VERBOSE, FORMAT JSON) {sql}")
                result = cur.fetchall()[0][0]

                q_errors = []
                if not self.use_bao:
                    q_errors = self.extract_q_errors(result[0]['Plan'])

                # Get actual execution time
                actual_time = result[-1]['Execution Time']

                # Get Bao's predicted time if using Bao
                if self.use_bao:
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

                print(
                    f"\t{i}: Execution Time: {measurements[-1]['execution_time']:.4f}\t"
                    f"Planning Time: {measurements[-1]['planning_time']:.4f}\t"
                    f"Predicted Time: {measurements[-1]['predicted_time'] if measurements[-1]['predicted_time'] is not None else 'N/A'}")

            conn.close()
        except Exception as e:
            print(f"Query error: {e}")
            if conn is not None:
                conn.close()

            tmp = []
            for _ in range(self.num_executions):
                tmp.append({
                    'execution_time': 2 * self.timeout_limit,
                    'planning_time': 2 * self.timeout_limit,
                    'hint': None,
                    'predicted_time': None,
                    'q_errors': []
                })
            return tmp

        return measurements

    def current_timestamp_str(self):
        """Get current timestamp string"""
        return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')

    def load_queries(self):
        """Load queries from query directory"""
        if not self.query_dir:
            print("No query directory specified")
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

    def execute_test_queries(self):
        """Execute test queries following run_test_queries.py pattern"""
        print("="*80)
        print(f"Testing {'Bao' if self.use_bao else 'PostgreSQL'} on test queries...")
        print("="*80)

        # Load queries
        queries = self.load_queries()
        if not queries:
            print("No queries found")
            return False

        print(f"Start running {len(queries)} queries for evaluation...")

        # Check if output file exists
        output_path = self.bao_dir / self.output_file
        if output_path.exists():
            print(f"Removing existing output file: {output_path}")
            output_path.unlink()

        # Track failures
        failed_queries = 0
        consecutive_failures = 0
        max_consecutive_failures = 5  # Exit if 5 consecutive queries fail
        max_failure_rate = 0.5  # Exit if >50% queries fail

        # Execute all test queries
        for query_idx, (fp, q) in enumerate(queries):
            print(f"\nExecuting query: {fp}")
            measurements = self.run_query(q)

            # Check if query failed (execution time is timeout value)
            query_failed = measurements[0]['execution_time'] >= 2 * self.timeout_limit

            if query_failed:
                failed_queries += 1
                consecutive_failures += 1
                print(f"⚠ Query failed: {fp}")

                # Check consecutive failures
                if consecutive_failures >= max_consecutive_failures:
                    print(f"\n✗ ERROR: {consecutive_failures} consecutive queries failed!")
                    print("This usually indicates a database connection problem.")
                    print("Aborting test to prevent further issues.")
                    return False
            else:
                consecutive_failures = 0  # Reset on success

            for i, measurement in enumerate(measurements):
                # Format: hint, iteration, timestamp, filepath, planning_time, execution_time, Bao/PG
                output_string = (
                    f"{'x' if measurement['hint'] is None else measurement['hint']}, "
                    f"{i}, {self.current_timestamp_str()}, {fp}, "
                    f"{measurement['planning_time']}, {measurement['execution_time']}, "
                    f"{'Bao' if self.use_bao else 'PG'}"
                )

                with open(output_path, 'a') as f:
                    f.write(output_string)
                    f.write(os.linesep)

            # Check overall failure rate
            failure_rate = failed_queries / (query_idx + 1)
            if query_idx >= 10 and failure_rate > max_failure_rate:
                print(f"\n✗ ERROR: Failure rate too high ({failure_rate:.1%})")
                print(f"Failed {failed_queries} out of {query_idx + 1} queries")
                print("Aborting test to prevent wasting resources.")
                return False

        # Final statistics
        total_queries = len(queries)
        success_rate = (total_queries - failed_queries) / total_queries if total_queries > 0 else 0

        print("\n" + "="*80)
        if failed_queries > 0:
            print(f"⚠ Testing completed with {failed_queries} failed queries ({success_rate:.1%} success rate)")
        else:
            print(f"✓ Testing completed successfully! All queries passed.")
        print(f"Results saved to: {output_path}")
        print("="*80)

        # Return False if too many failures
        return success_rate >= 0.5  # At least 50% must succeed

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

    def run_test_pipeline(self):
        """Run the complete testing pipeline"""
        try:
            # Change to bao directory
            os.chdir(self.bao_dir)

            # Initialize environment
            self.__init_environment()

            # Start server if using Bao
            if self.use_bao:
                self.start_server()

            print("\n" + "="*80)
            print(f"Using {'Bao' if self.use_bao else 'PostgreSQL'} for testing")
            print(f"GEQO: {'enabled' if self.use_geqo else 'disabled'}")
            print("="*80)

            # Execute test queries
            if not self.execute_test_queries():
                raise RuntimeError("Testing failed")

            print("[DONE] Bao testing completed successfully!")
            return True

        except Exception as e:
            print(f"[FAILED] Testing failed: {e}")
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
        description="Test Bao Learned Query Optimizer on test queries",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with Bao
  python test_bao.py --query-dir ./queries/job/test --database-name imdbload --use-bao

  # Test with PostgreSQL
  python test_bao.py --query-dir ./queries/job/test --database-name imdbload --use-postgres

  # Custom output file
  python test_bao.py --query-dir ./queries/job/test --output-file test_results.log --use-bao

  # Disable GEQO
  python test_bao.py --query-dir ./queries/job/test --use-bao --disable-geqo

How it works:
  - Loads all queries from the query directory
  - Executes each query 3 times for evaluation
  - Records execution time, planning time, predicted time, and Q-errors
  - Can test with either Bao or PostgreSQL optimizer
        """
    )

    parser.add_argument(
        "--query-dir",
        type=str,
        required=True,
        help="Path to directory containing test queries (REQUIRED)"
    )

    parser.add_argument(
        "--database-name",
        type=str,
        default="imdbload",
        help="PostgreSQL database name (default: imdbload)"
    )

    parser.add_argument(
        "--use-bao",
        action="store_true",
        help="Use Bao for query optimization"
    )

    parser.add_argument(
        "--use-postgres",
        action="store_true",
        help="Use PostgreSQL optimizer for query optimization"
    )

    parser.add_argument(
        "--use-geqo",
        action="store_true",
        default=True,
        help="Enable GEQO (default: True)"
    )

    parser.add_argument(
        "--disable-geqo",
        action="store_false",
        dest="use_geqo",
        help="Disable GEQO"
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
        default="bao_test_results.log",
        help="Output file for test results (default: bao_test_results.log)"
    )

    parser.add_argument(
        "--server-startup-delay",
        type=int,
        default=5,
        help="Delay after starting server in seconds (default: 5)"
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

    # Validate arguments
    if (not args.use_bao) and (not args.use_postgres):
        print("ERROR: Need to either select Bao or Postgres to be used. (--use-bao or --use-postgres)")
        sys.exit(1)
    if args.use_bao and args.use_postgres:
        print("ERROR: Need to only select Bao or Postgres, not both. (--use-bao or --use-postgres)")
        sys.exit(1)

    # Create tester with custom settings
    tester = BaoTester(
        bao_dir=args.bao_dir,
        query_dir=args.query_dir,
        database_name=args.database_name,
        output_file=args.output_file,
        use_bao=args.use_bao,
        use_postgres=args.use_postgres,
        use_geqo=args.use_geqo,
        db_port=args.db_port
    )

    # Update tester settings
    tester.server_startup_delay = args.server_startup_delay

    if args.verbose:
        print(f"Configuration:")
        print(f"  Bao directory: {args.bao_dir}")
        print(f"  Query directory: {args.query_dir}")
        print(f"  Database name: {args.database_name}")
        print(f"  Output file: {args.output_file}")
        print(f"  Use Bao: {args.use_bao}")
        print(f"  Use PostgreSQL: {args.use_postgres}")
        print(f"  Use GEQO: {args.use_geqo}")
        print(f"  Server startup delay: {args.server_startup_delay}s")

    # Run testing pipeline
    success = tester.run_test_pipeline()

    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
