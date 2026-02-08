#!/usr/bin/env python3
"""
Bao Learned Query Optimizer Training Script
This script manages the Bao server lifecycle and delegates query execution to run_queries.py
"""

import os
import sys
import time
import psutil
from pathlib import Path

class BaoTrainer:
    def __init__(self, bao_dir=".", query_dir=None, database_name='imdbload', output_file="bao_training_results.log", db_port=5432, min_queries=None, test_interval=None, checkpoint_interval=None):
        self.bao_dir = Path(bao_dir).resolve()
        self.bao_server_dir = self.bao_dir / "bao_server"
        self.server_process = None
        self.original_dir = os.getcwd()

        # Query execution parameters (required)
        if not query_dir:
            raise ValueError("query_dir is required for training")

        self.query_dir = Path(query_dir).resolve()  # Convert to absolute path
        self.database_name = database_name
        self.db_port = db_port
        self.output_file = Path(output_file).resolve()  # Convert to absolute path
        self.min_queries = min_queries
        self.test_interval = test_interval
        self.checkpoint_interval = checkpoint_interval

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
                os.system(f"cd {self.bao_dir} && python3 -m pip install -r {requirements_file}")
        
    def start_server(self):
        """Start the Bao server"""
        print("Starting Bao server...")

        # Kill any existing Bao server processes
        self.kill_existing_servers()

        # Check if main.py exists
        server_script = self.bao_server_dir / "main.py"
        if not server_script.exists():
            raise RuntimeError(f"Server script not found: {server_script}")

        # Use output_file's directory for server log
        output_path = Path(self.output_file)
        if output_path.is_absolute():
            log_dir = output_path.parent
        else:
            log_dir = self.bao_dir / output_path.parent
        log_dir.mkdir(parents=True, exist_ok=True)

        # Server log file
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f"server_bao_train_{timestamp}.log"
        pid_file = self.bao_server_dir / "server.pid"

        # Start server with nohup
        cmd = f"cd {self.bao_server_dir} && nohup python3 -u main.py >> {log_file} 2>&1 & echo $! > server.pid"
        print(f"Server log: {log_file}")

        result = os.system(cmd)
        if result != 0:
            raise RuntimeError(f"Failed to start Bao server, exit code: {result}")

        # Wait for server to start
        print(f"Waiting {self.server_startup_delay} seconds for server to start...")
        time.sleep(self.server_startup_delay)

        # Read PID from file
        if pid_file.exists():
            with open(pid_file) as f:
                pid = f.read().strip()
            print(f"Bao server started with PID: {pid}")
        else:
            print("Warning: Could not find server.pid file")
        
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


    def execute_training_queries(self):
        """Execute queries by calling run_queries.py"""
        print("="*80)
        print("Executing training queries with run_queries.py...")
        print("="*80)

        if not self.query_dir:
            print("No query directory specified")
            return False

        # Prepare output file path (already absolute)
        output_path = self.output_file
        if output_path.exists():
            print(f"Removing existing output file: {output_path}")
            output_path.unlink()

        # Build command to call run_queries.py
        run_queries_script = self.bao_dir / "run_queries.py"
        if not run_queries_script.exists():
            print(f"Error: run_queries.py not found at {run_queries_script}")
            return False

        print(f"Calling run_queries.py with:")
        print(f"  query_dir: {self.query_dir}")
        print(f"  database: {self.database_name}")
        print(f"  output: {output_path}")
        print(f"  port: {self.db_port}")
        print(flush=True)

        # Run the training script
        cmd = f"cd {self.bao_dir} && python3 run_queries.py --query_dir {self.query_dir} --database_name {self.database_name} --output_file {output_path} --db-port {self.db_port}"
        if self.min_queries is not None:
            cmd += f" --min-queries {self.min_queries}"
        if self.test_interval is not None:
            cmd += f" --test-interval {self.test_interval}"
        if self.checkpoint_interval is not None:
            cmd += f" --checkpoint-interval {self.checkpoint_interval}"
        result = os.system(cmd)

        if result == 0:
            print("\n" + "="*80)
            print("✓ Training completed successfully!")
            print(f"Results saved to: {output_path}")
            print("="*80)
            return True
        else:
            print("\n" + "="*80)
            print(f"✗ Training failed with exit code {result}")
            print("="*80)
            return False
    
    def cleanup(self):
        """Clean up resources"""
        print("Stopping Bao server...")

        # Kill all Bao server processes (handles parent and child processes)
        os.system("pkill -f 'python3.*main.py' || true")
        time.sleep(2)
        # Try harder if still running
        os.system("pkill -9 -f 'python3.*main.py' || true")

        # Clean up PID file
        pid_file = self.bao_server_dir / "server.pid"
        if pid_file.exists():
            pid_file.unlink()

        print("Bao server stopped")
    
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
            print("Delegating query execution to run_queries.py")
            print("="*80)

            # Execute training queries via run_queries.py
            if not self.execute_training_queries():
                raise RuntimeError("Training failed")

            print("[DONE] Bao training completed successfully!")
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
        description="Train Bao Learned Query Optimizer - manages server lifecycle and calls run_queries.py",
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
  - Starts Bao server
  - Calls run_queries.py to execute training queries
  - run_queries.py handles all query execution and Bao retraining logic
  - Stops Bao server and cleans up when done
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

    parser.add_argument(
        "--min-queries",
        type=int,
        default=None,
        help="Minimum number of training queries (sample with replacement if needed)"
    )

    parser.add_argument(
        "--test-interval",
        type=int,
        default=None,
        help="Run test every N iterations (default: 20)"
    )

    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=None,
        help="Save checkpoint every N iterations (default: 20)"
    )

    args = parser.parse_args()

    # Create trainer with custom settings
    trainer = BaoTrainer(
        bao_dir=args.bao_dir,
        query_dir=args.query_dir,
        database_name=args.database_name,
        output_file=args.output_file,
        db_port=args.db_port,
        min_queries=args.min_queries,
        test_interval=args.test_interval,
        checkpoint_interval=args.checkpoint_interval
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
