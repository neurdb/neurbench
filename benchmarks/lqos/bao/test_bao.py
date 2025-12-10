#!/usr/bin/env python3
"""
Bao Learned Query Optimizer Testing Script
This script manages the Bao server lifecycle and delegates query execution to run_test_queries.py
"""

import os
import sys
import time
import subprocess
import psutil
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

    def execute_test_queries(self):
        """Execute test queries by calling run_test_queries.py"""
        print("="*80)
        print(f"Testing {'Bao' if self.use_bao else 'PostgreSQL'} with run_test_queries.py...")
        print("="*80)

        if not self.query_dir:
            print("No query directory specified")
            return False

        # Prepare output file path
        output_path = self.bao_dir / self.output_file
        if output_path.exists():
            print(f"Removing existing output file: {output_path}")
            output_path.unlink()

        # Build command to call run_test_queries.py
        run_test_script = self.bao_dir / "run_test_queries.py"
        if not run_test_script.exists():
            print(f"Error: run_test_queries.py not found at {run_test_script}")
            return False

        cmd = [
            sys.executable,
            str(run_test_script),
            "--query_dir", self.query_dir,
            "--database_name", self.database_name,
            "--output_file", str(output_path),
            "--db-port", str(self.db_port)
        ]

        # Add mode flags
        if self.use_bao:
            cmd.append("--use_bao")
        elif self.use_postgres:
            cmd.append("--use_postgres")

        # Add GEQO flag
        if not self.use_geqo:
            cmd.append("--disable_geqo")

        print(f"Calling run_test_queries.py with:")
        print(f"  query_dir: {self.query_dir}")
        print(f"  database: {self.database_name}")
        print(f"  output: {output_path}")
        print(f"  port: {self.db_port}")
        print(f"  mode: {'Bao' if self.use_bao else 'PostgreSQL'}")
        print(f"  geqo: {'enabled' if self.use_geqo else 'disabled'}")
        print(flush=True)

        # Run the test script
        try:
            result = subprocess.run(
                cmd,
                cwd=self.bao_dir,
                check=False,
                stdout=sys.stdout,
                stderr=sys.stderr
            )

            if result.returncode == 0:
                print("\n" + "="*80)
                print("✓ Testing completed successfully!")
                print(f"Results saved to: {output_path}")
                print("="*80)
                return True
            else:
                print("\n" + "="*80)
                print(f"✗ Testing failed with exit code {result.returncode}")
                print("="*80)
                return False

        except Exception as e:
            print(f"\n✗ Error running run_test_queries.py: {e}")
            return False

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
            print(f"Delegating query execution to run_test_queries.py")
            print(f"Mode: {'Bao' if self.use_bao else 'PostgreSQL'}")
            print(f"GEQO: {'enabled' if self.use_geqo else 'disabled'}")
            print("="*80)

            # Execute test queries via run_test_queries.py
            if not self.execute_test_queries():
                raise RuntimeError("Testing failed")

            print("[DONE] Testing completed successfully!")
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
        description="Test Bao Learned Query Optimizer - manages server lifecycle and calls run_test_queries.py",
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
  - Starts Bao server (if using Bao mode)
  - Calls run_test_queries.py to execute test queries
  - run_test_queries.py handles all query execution logic
  - Stops Bao server and cleans up when done
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
