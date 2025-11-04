#!/usr/bin/env python3
"""
HybridQO Learned Query Optimizer Training Script
This script automates the training process for HybridQO LQO
"""

import os
import sys
import subprocess
import time
import argparse
from pathlib import Path

class HybridQOTrainer:
    def __init__(self, hybridqo_dir="."):
        self.hybridqo_dir = Path(hybridqo_dir).resolve()
        self.original_dir = os.getcwd()
        
    def check_environment(self):
        """Check if HybridQO environment is properly set up"""
        print("Checking HybridQO environment...")
        
        # Check required files
        required_files = [
            "run_mcts.py",
            "Hinter.py",
            "NET.py",
            "mcts.py",
            "requirements.txt",
        ]
        
        for file_path in required_files:
            if not (self.hybridqo_dir / file_path).exists():
                print(f"[FAILED] Required file not found: {file_path}")
                return False
            else:
                print(f"[SUCCESS] Found: {file_path}")
        
        return True
    
    def install_dependencies(self):
        """Install HybridQO dependencies"""
        print("Installing HybridQO dependencies...")
        
        try:
            # Install requirements
            requirements_file = self.hybridqo_dir / "requirements.txt"
            if requirements_file.exists():
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)],
                    cwd=self.hybridqo_dir,
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                
                if result.returncode == 0:
                    print("[SUCCESS] Requirements installed successfully")
                else:
                    print(f"⚠ Requirements installation had issues: {result.stderr}")
            
            # Install additional dependencies that HybridQO needs
            additional_deps = [
                "torch",
                "torchfold",
                "psqlparse",
                "tqdm",
                "numpy",
                "pandas",
            ]
            
            for dep in additional_deps:
                try:
                    result = subprocess.run(
                        [sys.executable, "-m", "pip", "install", dep],
                        capture_output=True,
                        text=True,
                        timeout=60
                    )
                    
                    if result.returncode == 0:
                        print(f"[SUCCESS] {dep} installed successfully")
                    else:
                        print(f"⚠ {dep} installation had issues: {result.stderr}")
                        
                except Exception as e:
                    print(f"⚠ Error installing {dep}: {e}")
            
            return True
            
        except Exception as e:
            print(f"[FAILED] Error installing dependencies: {e}")
            return False
    
    def check_configuration(self):
        """Check and configure HybridQO settings"""
        print("Checking HybridQO configuration...")
        
        config_file = self.hybridqo_dir / "ImportantConfig.py"
        if not config_file.exists():
            print("[FAILED] ImportantConfig.py not found")
            return False
        
        print("[SUCCESS] Configuration file found")
        
        # Check if model directory exists
        model_dir = self.hybridqo_dir / "model"
        if not model_dir.exists():
            print("Creating model directory...")
            model_dir.mkdir(exist_ok=True)
            print("[SUCCESS] Model directory created")
        else:
            print("[SUCCESS] Model directory exists")
        
        return True
    
    def run_training(self):
        """Run HybridQO training"""
        print("Starting HybridQO training...")
        
        try:
            # Run HybridQO training using MCTS
            result = subprocess.run(
                [sys.executable, "run_mcts.py"],
                cwd=self.hybridqo_dir,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            if result.returncode == 0:
                print("[SUCCESS] HybridQO training completed successfully")
                print("Training output:")
                print(result.stdout[-1000:])  # Last 1000 characters
                return True
            else:
                print(f"⚠ HybridQO training had issues: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("⚠ HybridQO training timed out (1 hour)")
            return False
        except Exception as e:
            print(f"[FAILED] HybridQO training error: {e}")
            return False
    
    def run_training_pipeline(self):
        """Run the complete training pipeline"""
        try:
            # Change to hybridqo directory
            os.chdir(self.hybridqo_dir)
            
            # Check environment
            if not self.check_environment():
                raise RuntimeError("HybridQO environment check failed")
            
            # Install dependencies
            if not self.install_dependencies():
                print("⚠ Dependency installation had issues, but continuing...")
            
            # Check configuration
            if not self.check_configuration():
                raise RuntimeError("HybridQO configuration check failed")
            
            # Run training
            if not self.run_training():
                raise RuntimeError("HybridQO training failed")
            
            print("[DONE] HybridQO training completed successfully!")
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
        description="Train HybridQO Learned Query Optimizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_hybridqo.py                    # Train with default settings
  python train_hybridqo.py --verbose          # Enable verbose output
        """
    )
    
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = HybridQOTrainer(".")
    
    if args.verbose:
        print(f"Configuration:")
        print(f"  HybridQO directory: {trainer.hybridqo_dir}")
    
    # Run training
    success = trainer.run_training_pipeline()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
