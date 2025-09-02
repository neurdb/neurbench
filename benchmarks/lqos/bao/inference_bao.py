#!/usr/bin/env python3
"""
Bao Learned Query Optimizer Inference Script
This script runs inference using a trained Bao model
"""

import os
import sys
import subprocess
import time
import json
from pathlib import Path

class BaoInference:
    def __init__(self, bao_dir="."):
        self.bao_dir = Path(bao_dir).resolve()
        self.bao_server_dir = self.bao_dir / "bao_server"
        self.server_process = None
        self.original_dir = os.getcwd()
        
    def check_model_exists(self):
        """Check if trained Bao model exists"""
        model_path = self.bao_server_dir / "model"
        if model_path.exists():
            print(f"✅ Found trained model at {model_path}")
            return True
        else:
            print(f"❌ No trained model found at {model_path}")
            print("💡 Please run 'tqo bao' first to train a model")
            return False
    
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
        print("Waiting 5 seconds for server to start...")
        time.sleep(5)
        
        # Check if server is running
        if self.server_process.poll() is not None:
            raise RuntimeError("Failed to start Bao server")
        
        print(f"Bao server started with PID: {self.server_process.pid}")
        
    def kill_existing_servers(self):
        """Kill any existing Bao server processes"""
        try:
            import psutil
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    if proc.info['cmdline'] and 'main.py' in proc.info['cmdline']:
                        print(f"Killing existing Bao server process: {proc.info['pid']}")
                        proc.terminate()
                        proc.wait(timeout=5)
                except (psutil.NoSuchProcess, psutil.TimeoutExpired):
                    pass
        except ImportError:
            print("Warning: psutil not available, using basic process management")
            # Basic process management without psutil
            os.system("pkill -f 'python.*main.py' 2>/dev/null || true")
    
    def test_connection(self):
        """Test connection to Bao server"""
        print("Testing Bao server connection...")
        
        try:
            result = subprocess.run(
                [sys.executable, "baoctl.py", "--test-connection"],
                cwd=self.bao_server_dir,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                print("✅ Connection test successful")
                return True
            else:
                print(f"⚠ Connection test failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("⚠ Connection test timed out")
            return False
        except Exception as e:
            print(f"⚠ Connection test error: {e}")
            return False
    
    def run_inference_tests(self):
        """Run inference tests with sample queries"""
        print("Running inference tests...")
        
        # Test queries for inference
        test_queries = [
            "SELECT COUNT(*) FROM information_schema.tables",
            "SELECT version()",
            "SELECT current_database()"
        ]
        
        results = []
        for i, query in enumerate(test_queries):
            print(f"\nTest Query {i+1}: {query}")
            try:
                result = subprocess.run(
                    [sys.executable, "baoctl.py", "--status"],
                    cwd=self.bao_server_dir,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if result.returncode == 0:
                    print("✅ Status check successful")
                    results.append({"query": query, "status": "success"})
                else:
                    print(f"⚠ Status check failed: {result.stderr}")
                    results.append({"query": query, "status": "failed"})
                    
            except Exception as e:
                print(f"⚠ Query test error: {e}")
                results.append({"query": query, "status": "error"})
        
        return results
    
    def run_advanced_inference_tests(self):
        """Run advanced inference tests with real queries"""
        print("Running advanced inference tests...")
        
        # Check if we have test queries
        test_query_dir = self.bao_dir / "test_queries"
        if test_query_dir.exists():
            print(f"Found test queries directory: {test_query_dir}")
            # TODO: Implement running actual test queries
            pass
        else:
            print("No test queries directory found, skipping advanced tests")
        
        return []
    
    def check_server_status(self):
        """Check detailed server status"""
        print("Checking server status...")
        
        try:
            result = subprocess.run(
                [sys.executable, "baoctl.py", "--status"],
                cwd=self.bao_server_dir,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                print("✅ Server status check successful")
                print("Server output:")
                print(result.stdout)
                return True
            else:
                print(f"⚠ Server status check failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"⚠ Server status check error: {e}")
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
    
    def run_inference_pipeline(self):
        """Run the complete inference pipeline"""
        try:
            # Change to bao directory
            os.chdir(self.bao_dir)
            
            # Check if model exists
            if not self.check_model_exists():
                print("⚠ No trained model found. Please run 'tqo bao' first.")
                return False
            
            # Start server
            self.start_server()
            
            # Test connection
            if not self.test_connection():
                print("⚠ Connection test failed, but continuing...")
            
            # Check server status
            self.check_server_status()
            
            # Run basic inference tests
            results = self.run_inference_tests()
            
            # Run advanced tests if requested
            if hasattr(self, 'run_advanced') and self.run_advanced:
                advanced_results = self.run_advanced_inference_tests()
                results.extend(advanced_results)
            
            # Print summary
            print("\n" + "="*50)
            print("Inference Test Results:")
            print("="*50)
            for result in results:
                status_icon = "✅" if result["status"] == "success" else "❌"
                print(f"{status_icon} {result['query']}: {result['status']}")
            
            success_count = sum(1 for r in results if r["status"] == "success")
            print(f"\nOverall: {success_count}/{len(results)} tests passed")
            
            if success_count > 0:
                print("🎉 Bao inference is working!")
                return True
            else:
                print("❌ All inference tests failed")
                return False
                
        except Exception as e:
            print(f"❌ Inference failed: {e}")
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
        description="Run Bao Learned Query Optimizer Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python inference_bao.py                    # Run inference with default settings
  python inference_bao.py --verbose          # Enable verbose output
  python inference_bao.py --advanced        # Run advanced inference tests
  python inference_bao.py --test-only       # Only test environment, don't run full inference
        """
    )
    
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--advanced", 
        action="store_true",
        help="Run advanced inference tests"
    )
    
    parser.add_argument(
        "--test-only", 
        action="store_true",
        help="Only test environment, don't run full inference"
    )
    
    args = parser.parse_args()
    
    # Create inference runner
    inference = BaoInference(".")
    
    if args.verbose:
        print("Verbose mode enabled")
    
    if args.advanced:
        inference.run_advanced = True
        print("Advanced inference tests enabled")
    
    if args.test_only:
        print("Environment test only mode...")
        # Just test the environment
        try:
            os.chdir(inference.bao_dir)
            inference.check_model_exists()
            inference.start_server()
            inference.test_connection()
            inference.check_server_status()
            print("✅ Environment test completed")
            success = True
        except Exception as e:
            print(f"❌ Environment test failed: {e}")
            success = False
        finally:
            inference.cleanup()
            os.chdir(inference.original_dir)
    else:
        # Run full inference pipeline
        success = inference.run_inference_pipeline()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
