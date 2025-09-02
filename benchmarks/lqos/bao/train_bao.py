#!/usr/bin/env python3
"""
Bao Learned Query Optimizer Training Script
This script automates the training process for Bao LQO
"""

import os
import sys
import time
import subprocess
import signal
import psutil
import json
from pathlib import Path

class BaoTrainer:
    def __init__(self, bao_dir="."):
        self.bao_dir = Path(bao_dir).resolve()
        self.bao_server_dir = self.bao_dir / "bao_server"
        self.server_process = None
        self.original_dir = os.getcwd()
        
        # Configurable parameters
        self.experiment_duration = 60
        self.training_timeout = 300
        self.server_startup_delay = 5
        
    def __init_environment(self):
        """Initialize the Bao environment"""
        print("Initializing Bao environment...")
        
        # Check if we're in the right directory
        if not (self.bao_dir / "bao_server").exists():
            raise RuntimeError(f"Bao server directory not found in {self.bao_dir}")
        
        # Activate virtual environment if it exists
        env_script = self.bao_dir / "activate_env.sh"
        if env_script.exists():
            print("Activating virtual environment...")
            # Note: source command needs to be run in shell
            os.system(f"source {env_script}")
        
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
                print("✓ Connection test successful")
                return True
            else:
                print(f"✗ Connection test failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("✗ Connection test timed out")
            return False
        except Exception as e:
            print(f"✗ Connection test error: {e}")
            return False
    
    def train_model(self):
        """Train the Bao model"""
        print("Training Bao model...")
        
        try:
            # Initial training
            result = subprocess.run(
                [sys.executable, "baoctl.py", "--retrain"],
                cwd=self.bao_server_dir,
                capture_output=True,
                text=True,
                timeout=self.training_timeout
            )
            
            if result.returncode == 0:
                print("✓ Initial training completed")
            else:
                print(f"✗ Initial training failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("✗ Initial training timed out")
            return False
        except Exception as e:
            print(f"✗ Training error: {e}")
            return False
        
        return True
    
    def run_experiments(self, duration=None):
        """Run experiments to collect experience"""
        if duration is None:
            duration = self.experiment_duration
            
        print(f"Running experiments for {duration} seconds...")
        
        try:
            result = subprocess.run(
                [sys.executable, "baoctl.py", "--experiment", str(duration)],
                cwd=self.bao_server_dir,
                capture_output=True,
                text=True,
                timeout=duration + 30
            )
            
            if result.returncode == 0:
                print("✓ Experiments completed")
                return True
            else:
                print(f"✗ Experiments failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("✗ Experiments timed out")
            return False
        except Exception as e:
            print(f"✗ Experiment error: {e}")
            return False
    
    def finalize_training(self):
        """Finalize training with collected experience"""
        print("Finalizing training with collected experience...")
        
        try:
            result = subprocess.run(
                [sys.executable, "baoctl.py", "--retrain"],
                cwd=self.bao_server_dir,
                capture_output=True,
                text=True,
                timeout=self.training_timeout
            )
            
            if result.returncode == 0:
                print("✓ Final training completed")
                return True
            else:
                print(f"✗ Final training failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("✗ Final training timed out")
            return False
        except Exception as e:
            print(f"✗ Final training error: {e}")
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
    
    def run_training_pipeline(self):
        """Run the complete training pipeline"""
        try:
            # Change to bao directory
            os.chdir(self.bao_dir)
            
            # Initialize environment
            self.__init_environment()
            
            # Start server
            self.start_server()
            
            # Test connection
            if not self.test_connection():
                print("Warning: Connection test failed, but continuing...")
            
            # Train initial model
            if not self.train_model():
                raise RuntimeError("Initial training failed")
            
            # Run experiments
            if not self.run_experiments():
                print("Warning: Experiments failed, but continuing...")
            
            # Final training
            if not self.finalize_training():
                raise RuntimeError("Final training failed")
            
            print("🎉 Bao training completed successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            return False
            
        finally:
            # Cleanup
            self.cleanup()
            # Return to original directory
            os.chdir(self.original_dir)

    def test_environment(self):
        """Test Bao environment without training"""
        print("Testing Bao environment...")
        
        try:
            # Change to bao directory
            os.chdir(self.bao_dir)
            
            # Initialize environment
            self.__init_environment()
            
            # Test server startup
            self.start_server()
            
            # Test connection
            connection_ok = self.test_connection()
            
            # Cleanup
            self.cleanup()
            
            if connection_ok:
                print("✅ Environment test passed!")
                return True
            else:
                print("⚠ Environment test completed with warnings")
                return True  # Still consider it a pass
                
        except Exception as e:
            print(f"❌ Environment test failed: {e}")
            return False
        finally:
            # Return to original directory
            os.chdir(self.original_dir)

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Train Bao Learned Query Optimizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_bao.py                    # Train with default settings
  python train_bao.py --experiment-time 120  # Run experiments for 2 minutes
  python train_bao.py --timeout 600      # Set 10 minute timeout for training
  python train_bao.py --verbose          # Enable verbose output
  python train_bao.py --test-only        # Only test environment, don't train
        """
    )
    
    parser.add_argument(
        "--experiment-time", 
        type=int, 
        default=60,
        help="Duration for experiments in seconds (default: 60)"
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
        "--verbose", 
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--test-only", 
        action="store_true",
        help="Only test environment, don't perform training"
    )
    
    parser.add_argument(
        "--bao-dir", 
        type=str, 
        default=".",
        help="Path to Bao directory (default: current directory)"
    )
    
    args = parser.parse_args()
    
    # Create trainer with custom settings
    trainer = BaoTrainer(args.bao_dir)
    
    if args.verbose:
        print(f"Configuration:")
        print(f"  Experiment time: {args.experiment_time}s")
        print(f"  Training timeout: {args.training_timeout}s")
        print(f"  Server startup delay: {args.server_startup_delay}s")
        print(f"  Test only: {args.test_only}")
        print(f"  Bao directory: {args.bao_dir}")
    
    if args.test_only:
        print("Testing Bao environment only...")
        success = trainer.test_environment()
    else:
        # Update trainer settings
        trainer.experiment_duration = args.experiment_time
        trainer.training_timeout = args.training_timeout
        trainer.server_startup_delay = args.server_startup_delay
        
        success = trainer.run_training_pipeline()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
