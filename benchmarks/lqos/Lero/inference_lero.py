#!/usr/bin/env python3
"""
Lero Learned Query Optimizer Inference Script
This script runs inference using a trained Lero model
"""

import os
import sys
import subprocess
import time
import argparse
from pathlib import Path

class LeroInference:
    def __init__(self, lero_dir="."):
        self.lero_dir = Path(lero_dir).resolve()
        self.original_dir = os.getcwd()
        
    def check_model_exists(self):
        """Check if trained Lero model exists"""
        print("Checking for trained Lero model...")
        
        # Check for model files (Lero typically saves models with specific prefixes)
        possible_model_files = [
            "tpch_test_model_0.pt",
            "tpch_test_model_1.pt",
            "tpch_test_model_2.pt",
            "lero_tpch.log",
            "lero_tpch.log.training",
        ]
        
        model_found = False
        for model_file in possible_model_files:
            model_path = self.lero_dir / "lero" / model_file
            if model_path.exists():
                print(f"✅ Found model file: {model_file}")
                model_found = True
            else:
                print(f"⚠ Model file not found: {model_file}")
        
        if not model_found:
            print("❌ No trained model found")
            print("💡 Please run 'tqo lero' first to train a model")
            return False
        
        return True
    
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
                print(f"❌ Required file/directory not found: {file_path}")
                return False
            else:
                print(f"✅ Found: {file_path}")
        
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
                print("✅ PostgreSQL is running")
                return True
            else:
                print("⚠ PostgreSQL is not running or not accessible")
                print("💡 Please ensure PostgreSQL is running and accessible")
                return False
                
        except FileNotFoundError:
            print("⚠ pg_isready command not found")
            print("💡 Please ensure PostgreSQL is installed and in PATH")
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
                print(f"✅ Lero server started with PID: {server_process.poll()}")
                return True
            else:
                print("❌ Failed to start Lero server")
                return False
                
        except Exception as e:
            print(f"❌ Error starting Lero server: {e}")
            return False
    
    def run_inference_tests(self, query_path=None):
        """Run inference tests with trained model"""
        print("Running Lero inference tests...")
        
        if not query_path:
            # Use default test queries if available
            query_path = self.lero_dir / "lero" / "test_script" / "tpch_test.txt"
            if not query_path.exists():
                print("⚠ No test query file specified and default not found")
                print("💡 Please provide --query-path argument")
                return False
        
        print(f"Using test queries: {query_path}")
        
        try:
            # Run Lero inference using the trained model
            train_script = self.lero_dir / "lero" / "train.py"
            
            cmd = [
                sys.executable, str(train_script),
                "--query_path", str(query_path),
                "--algo", "lero",
                "--output_query_latency_file", "lero_inference_test.log",
                "--model_prefix", "tpch_test_model",
                "--topK", "3"
            ]
            
            result = subprocess.run(
                cmd,
                cwd=self.lero_dir / "lero",
                capture_output=True,
                text=True,
                timeout=1800  # 30 minutes timeout
            )
            
            if result.returncode == 0:
                print("✅ Lero inference tests completed successfully")
                print("Test output:")
                print(result.stdout[-1000:])  # Last 1000 characters
                return True
            else:
                print(f"⚠ Lero inference tests had issues: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("⚠ Lero inference tests timed out (30 minutes)")
            return False
        except Exception as e:
            print(f"❌ Lero inference tests error: {e}")
            return False
    
    def run_postgresql_comparison(self, query_path=None):
        """Run PostgreSQL comparison for benchmarking"""
        print("Running PostgreSQL comparison...")
        
        if not query_path:
            # Use default test queries if available
            query_path = self.lero_dir / "lero" / "test_script" / "tpch_test.txt"
            if not query_path.exists():
                print("⚠ No test query file specified and default not found")
                return False
        
        try:
            # Run PostgreSQL baseline
            train_script = self.lero_dir / "lero" / "train.py"
            
            cmd = [
                sys.executable, str(train_script),
                "--query_path", str(query_path),
                "--algo", "pg",
                "--output_query_latency_file", "pg_inference_test.log"
            ]
            
            result = subprocess.run(
                cmd,
                cwd=self.lero_dir / "lero",
                capture_output=True,
                text=True,
                timeout=1800  # 30 minutes timeout
            )
            
            if result.returncode == 0:
                print("✅ PostgreSQL comparison completed successfully")
                return True
            else:
                print(f"⚠ PostgreSQL comparison had issues: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("⚠ PostgreSQL comparison timed out (30 minutes)")
            return False
        except Exception as e:
            print(f"❌ PostgreSQL comparison error: {e}")
            return False
    
    def run_inference_pipeline(self, query_path=None, run_comparison=False):
        """Run the complete inference pipeline"""
        try:
            # Change to lero directory
            os.chdir(self.lero_dir)
            
            # Check environment
            if not self.check_environment():
                raise RuntimeError("Lero environment check failed")
            
            # Check if model exists
            if not self.check_model_exists():
                raise RuntimeError("No trained model found")
            
            # Check PostgreSQL
            if not self.check_postgresql_setup():
                print("⚠ PostgreSQL check failed, but continuing...")
            
            # Start Lero server
            if not self.start_lero_server():
                raise RuntimeError("Failed to start Lero server")
            
            # Run inference tests
            if not self.run_inference_tests(query_path):
                raise RuntimeError("Lero inference tests failed")
            
            # Run PostgreSQL comparison if requested
            if run_comparison:
                if not self.run_postgresql_comparison(query_path):
                    print("⚠ PostgreSQL comparison failed, but inference tests passed")
            
            print("🎉 Lero inference completed successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Inference failed: {e}")
            return False
            
        finally:
            # Return to original directory
            os.chdir(self.original_dir)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Run Lero Learned Query Optimizer Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python inference_lero.py                                    # Run inference with default settings
  python inference_lero.py --query-path test_queries.txt      # Use specific test queries
  python inference_lero.py --run-comparison                   # Also run PostgreSQL comparison
  python inference_lero.py --verbose                          # Enable verbose output
        """
    )
    
    parser.add_argument(
        "--query-path", 
        type=str,
        help="Path to test query file"
    )
    
    parser.add_argument(
        "--run-comparison", 
        action="store_true",
        help="Also run PostgreSQL comparison for benchmarking"
    )
    
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Create inference runner
    inference = LeroInference(".")
    
    if args.verbose:
        print(f"Configuration:")
        print(f"  Query path: {args.query_path}")
        print(f"  Run comparison: {args.run_comparison}")
        print(f"  Lero directory: {inference.lero_dir}")
    
    # Run inference
    success = inference.run_inference_pipeline(args.query_path, args.run_comparison)
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
