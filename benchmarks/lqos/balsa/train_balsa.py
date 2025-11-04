#!/usr/bin/env python3
"""
Balsa Learned Query Optimizer Training Script
This script automates the training process for Balsa LQO
"""

import os
import sys
import subprocess
import time
import argparse
from pathlib import Path

class BalsaTrainer:
    def __init__(self, balsa_dir="."):
        self.balsa_dir = Path(balsa_dir).resolve()
        self.original_dir = os.getcwd()
        
    def check_environment(self):
        """Check if Balsa environment is properly set up"""
        print("Checking Balsa environment...")
        
        # Check required files
        required_files = [
            "run.py",
            "setup.py",
            "requirements.txt",
            "balsa/",
        ]
        
        for file_path in required_files:
            if not (self.balsa_dir / file_path).exists():
                print(f"[FAILED] Required file/directory not found: {file_path}")
                return False
            else:
                print(f"[SUCCESS] Found: {file_path}")
        
        return True
    
    def install_dependencies(self):
        """Install Balsa dependencies"""
        print("Installing Balsa dependencies...")
        
        try:
            # Install Balsa package
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-e", "."],
                cwd=self.balsa_dir,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode == 0:
                print("[SUCCESS] Balsa package installed successfully")
            else:
                print(f"⚠ Balsa package installation had issues: {result.stderr}")
            
            # Install pg_executor
            pg_executor_dir = self.balsa_dir / "pg_executor"
            if pg_executor_dir.exists():
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", "-e", "."],
                    cwd=pg_executor_dir,
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                
                if result.returncode == 0:
                    print("[SUCCESS] pg_executor installed successfully")
                else:
                    print(f"⚠ pg_executor installation had issues: {result.stderr}")
            
            # Install requirements
            requirements_file = self.balsa_dir / "requirements.txt"
            if requirements_file.exists():
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)],
                    cwd=self.balsa_dir,
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                
                if result.returncode == 0:
                    print("[SUCCESS] Requirements installed successfully")
                else:
                    print(f"⚠ Requirements installation had issues: {result.stderr}")
            
            return True
            
        except Exception as e:
            print(f"[FAILED] Error installing dependencies: {e}")
            return False
    
    def run_training(self, experiment_name="Balsa_JOBRandSplit"):
        """Run Balsa training"""
        print(f"Starting Balsa training with experiment: {experiment_name}")
        
        try:
            # Run Balsa training
            result = subprocess.run(
                [sys.executable, "run.py", "--run", experiment_name],
                cwd=self.balsa_dir,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            if result.returncode == 0:
                print("[SUCCESS] Balsa training completed successfully")
                print("Training output:")
                print(result.stdout[-1000:])  # Last 1000 characters
                return True
            else:
                print(f"⚠ Balsa training had issues: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("⚠ Balsa training timed out (1 hour)")
            return False
        except Exception as e:
            print(f"[FAILED] Balsa training error: {e}")
            return False
    
    def run_training_pipeline(self, experiment_name="Balsa_JOBRandSplit"):
        """Run the complete training pipeline"""
        try:
            # Change to balsa directory
            os.chdir(self.balsa_dir)
            
            # Check environment
            if not self.check_environment():
                raise RuntimeError("Balsa environment check failed")
            
            # Install dependencies
            if not self.install_dependencies():
                print("⚠ Dependency installation had issues, but continuing...")
            
            # Run training
            if not self.run_training(experiment_name):
                raise RuntimeError("Balsa training failed")
            
            print("[DONE] Balsa training completed successfully!")
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
        description="Train Balsa Learned Query Optimizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_balsa.py                    # Train with default experiment
  python train_balsa.py --experiment exp_job_light  # Use specific experiment
  python train_balsa.py --verbose          # Enable verbose output
        """
    )
    
    parser.add_argument(
        "--experiment", 
        type=str, 
        default="Balsa_JOBRandSplit",
        help="Experiment name to run (default: Balsa_JOBRandSplit)"
    )
    
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = BalsaTrainer(".")
    
    if args.verbose:
        print(f"Configuration:")
        print(f"  Experiment: {args.experiment}")
        print(f"  Balsa directory: {trainer.balsa_dir}")
    
    # Run training
    success = trainer.run_training_pipeline(args.experiment)
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
