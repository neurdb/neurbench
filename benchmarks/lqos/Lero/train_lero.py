#!/usr/bin/env python3
"""
Lero Learned Query Optimizer Training Script
This script automates the training process for Lero LQO
"""

import os
import sys
import subprocess
import time
import argparse
from pathlib import Path

class LeroTrainer:
    def __init__(self, lero_dir="."):
        self.lero_dir = Path(lero_dir).resolve()
        self.original_dir = os.getcwd()
        
    def check_environment(self):
        """Check if Lero environment is properly set up"""
        print("Checking Lero environment...")
        
        # Check required files
        required_files = [
            "lero/",
            "lero/server.py",
            "lero/train.py",
            "lero/model.py",
        ]
        
        for file_path in required_files:
            if not (self.lero_dir / file_path).exists():
                print(f"[FAILED] Required file/directory not found: {file_path}")
                return False
            else:
                print(f"[SUCCESS] Found: {file_path}")
        
        return True
    
    def check_postgresql_setup(self):
        """Check if PostgreSQL is properly set up for Lero"""
        print("Checking PostgreSQL setup...")
        
        try:
            # Check if PostgreSQL is running
            result = subprocess.run(
                ["pg_isready"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                print("[SUCCESS] PostgreSQL is running")
                return True
            else:
                print("⚠ PostgreSQL is not running or not accessible")
                print("[INFO] Please ensure PostgreSQL is running and accessible")
                return False
                
        except FileNotFoundError:
            print("⚠ pg_isready command not found")
            print("[INFO] Please ensure PostgreSQL is installed and in PATH")
            return False
        except Exception as e:
            print(f"⚠ PostgreSQL check error: {e}")
            return False
    
    def start_lero_server(self):
        """Start the Lero server"""
        print("Starting Lero server...")
        
        try:
            # Start Lero server in background
            server_script = self.lero_dir / "lero" / "server.py"
            
            # Check if server is already running
            try:
                result = subprocess.run(
                    ["pgrep", "-f", "server.py"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                if result.returncode == 0:
                    print("⚠ Lero server is already running")
                    return True
                    
            except FileNotFoundError:
                pass  # pgrep not available, continue
            
            # Start new server
            server_process = subprocess.Popen(
                [sys.executable, str(server_script)],
                cwd=self.lero_dir / "lero",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Wait for server to start
            print("Waiting 5 seconds for server to start...")
            time.sleep(5)
            
            # Check if server is running
            if server_process.poll() is None:
                print(f"[SUCCESS] Lero server started with PID: {server_process.poll()}")
                return True
            else:
                print("[FAILED] Failed to start Lero server")
                return False
                
        except Exception as e:
            print(f"[FAILED] Error starting Lero server: {e}")
            return False
    
    def run_training(self, query_path=None, test_query_path=None):
        """Run Lero training"""
        print("Starting Lero training...")
        
        if not query_path:
            # Use default TPC-H queries if available
            query_path = self.lero_dir / "lero" / "test_script" / "tpch_train.txt"
            if not query_path.exists():
                print("⚠ No training query file specified and default not found")
                print("[INFO] Please provide --query-path argument")
                return False
        
        if not test_query_path:
            # Use default test queries if available
            test_query_path = self.lero_dir / "lero" / "test_script" / "tpch_test.txt"
            if not test_query_path.exists():
                print("⚠ No test query file specified and default not found")
                print("[INFO] Please provide --test-query-path argument")
                return False
        
        print(f"Using training queries: {query_path}")
        print(f"Using test queries: {test_query_path}")
        
        try:
            # Run Lero training
            train_script = self.lero_dir / "lero" / "train.py"
            
            cmd = [
                sys.executable, str(train_script),
                "--query_path", str(query_path),
                "--test_query_path", str(test_query_path),
                "--algo", "lero",
                "--query_num_per_chunk", "20",
                "--output_query_latency_file", "lero_tpch.log",
                "--model_prefix", "tpch_test_model",
                "--topK", "3"
            ]
            
            result = subprocess.run(
                cmd,
                cwd=self.lero_dir / "lero",
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            if result.returncode == 0:
                print("[SUCCESS] Lero training completed successfully")
                print("Training output:")
                print(result.stdout[-1000:])  # Last 1000 characters
                return True
            else:
                print(f"⚠ Lero training had issues: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("⚠ Lero training timed out (1 hour)")
            return False
        except Exception as e:
            print(f"[FAILED] Lero training error: {e}")
            return False
    
    def run_training_pipeline(self, query_path=None, test_query_path=None):
        """Run the complete training pipeline"""
        try:
            # Change to lero directory
            os.chdir(self.lero_dir)
            
            # Check environment
            if not self.check_environment():
                raise RuntimeError("Lero environment check failed")
            
            # Check PostgreSQL
            if not self.check_postgresql_setup():
                print("⚠ PostgreSQL check failed, but continuing...")
            
            # Start Lero server
            if not self.start_lero_server():
                raise RuntimeError("Failed to start Lero server")
            
            # Run training
            if not self.run_training(query_path, test_query_path):
                raise RuntimeError("Lero training failed")
            
            print("[DONE] Lero training completed successfully!")
            return True
            
        except Exception as e:
            print(f"[FAILED] Training failed: {e}")
            return False
            
        finally:
            # Return to original directory
            os.chdir(self.original_dir)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Train Lero Learned Query Optimizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_lero.py                                    # Train with default settings
  python train_lero.py --query-path queries.txt           # Use specific training queries
  python train_lero.py --test-query-path test_queries.txt # Use specific test queries
  python train_lero.py --verbose                          # Enable verbose output
        """
    )
    
    parser.add_argument(
        "--query-path", 
        type=str,
        help="Path to training query file"
    )
    
    parser.add_argument(
        "--test-query-path", 
        type=str,
        help="Path to test query file"
    )
    
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = LeroTrainer(".")
    
    if args.verbose:
        print(f"Configuration:")
        print(f"  Query path: {args.query_path}")
        print(f"  Test query path: {args.test_query_path}")
        print(f"  Lero directory: {trainer.lero_dir}")
    
    # Run training
    success = trainer.run_training_pipeline(args.query_path, args.test_query_path)
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
