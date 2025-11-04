#!/usr/bin/env python3
"""
Balsa Learned Query Optimizer Inference Script
This script runs inference using a trained Balsa model
"""

import os
import sys
import subprocess
import time
import argparse
from pathlib import Path

class BalsaInference:
    def __init__(self, balsa_dir="."):
        self.balsa_dir = Path(balsa_dir).resolve()
        self.original_dir = os.getcwd()
        
    def check_model_exists(self):
        """Check if trained Balsa model exists"""
        print("Checking for trained Balsa model...")
        
        # Check for model files (Balsa typically saves models in various locations)
        possible_model_locations = [
            "checkpoints/",
            "models/",
            "outputs/",
            "logs/",
        ]
        
        model_found = False
        for location in possible_model_locations:
            model_path = self.balsa_dir / location
            if model_path.exists():
                # Check if it contains any files
                model_files = list(model_path.glob("*"))
                if model_files:
                    print(f"[SUCCESS] Found model files in {location}: {len(model_files)} files")
                    model_found = True
                else:
                    print(f"⚠ Directory {location} exists but is empty")
            else:
                print(f"⚠ Directory {location} not found")
        
        if not model_found:
            print("[FAILED] No trained model found")
            print("[INFO] Please run 'tqo balsa' first to train a model")
            return False
        
        return True
    
    def check_environment(self):
        """Check if Balsa environment is properly set up"""
        print("Checking Balsa environment...")
        
        # Check required files
        required_files = [
            "run.py",
            "balsa/",
        ]
        
        for file_path in required_files:
            if not (self.balsa_dir / file_path).exists():
                print(f"[FAILED] Required file/directory not found: {file_path}")
                return False
            else:
                print(f"[SUCCESS] Found: {file_path}")
        
        return True
    
    def run_inference_tests(self):
        """Run inference tests with trained model"""
        print("Running Balsa inference tests...")
        
        try:
            # Run Balsa with test_all flag to test the trained model
            result = subprocess.run(
                [sys.executable, "run.py", "--run", "Balsa_JOBRandSplit", "--test_all"],
                cwd=self.balsa_dir,
                capture_output=True,
                text=True,
                timeout=1800  # 30 minutes timeout
            )
            
            if result.returncode == 0:
                print("[SUCCESS] Balsa inference tests completed successfully")
                print("Test output:")
                print(result.stdout[-1000:])  # Last 1000 characters
                return True
            else:
                print(f"⚠ Balsa inference tests had issues: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("⚠ Balsa inference tests timed out (30 minutes)")
            return False
        except Exception as e:
            print(f"[FAILED] Balsa inference tests error: {e}")
            return False
    
    def run_custom_inference(self, experiment_name="Balsa_JOBRandSplit"):
        """Run custom inference with specific experiment"""
        print(f"Running Balsa inference with experiment: {experiment_name}")
        
        try:
            # Run Balsa inference
            result = subprocess.run(
                [sys.executable, "run.py", "--run", experiment_name, "--test_all"],
                cwd=self.balsa_dir,
                capture_output=True,
                text=True,
                timeout=1800  # 30 minutes timeout
            )
            
            if result.returncode == 0:
                print("[SUCCESS] Balsa inference completed successfully")
                print("Inference output:")
                print(result.stdout[-1000:])  # Last 1000 characters
                return True
            else:
                print(f"⚠ Balsa inference had issues: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("⚠ Balsa inference timed out (30 minutes)")
            return False
        except Exception as e:
            print(f"[FAILED] Balsa inference error: {e}")
            return False
    
    def run_inference_pipeline(self, experiment_name="Balsa_JOBRandSplit"):
        """Run the complete inference pipeline"""
        try:
            # Change to balsa directory
            os.chdir(self.balsa_dir)
            
            # Check environment
            if not self.check_environment():
                raise RuntimeError("Balsa environment check failed")
            
            # Check if model exists
            if not self.check_model_exists():
                raise RuntimeError("No trained model found")
            
            # Run inference tests
            if not self.run_inference_tests():
                raise RuntimeError("Balsa inference tests failed")
            
            # Run custom inference if specified
            if experiment_name != "Balsa_JOBRandSplit":
                if not self.run_custom_inference(experiment_name):
                    print("⚠ Custom inference failed, but basic tests passed")
            
            print("[DONE] Balsa inference completed successfully!")
            return True
            
        except Exception as e:
            print(f"[FAILED] Inference failed: {e}")
            return False
            
        finally:
            # Return to original directory
            os.chdir(self.original_dir)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Run Balsa Learned Query Optimizer Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python inference_balsa.py                    # Run inference with default experiment
  python inference_balsa.py --experiment exp_job_light  # Use specific experiment
  python inference_balsa.py --verbose          # Enable verbose output
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
    
    # Create inference runner
    inference = BalsaInference(".")
    
    if args.verbose:
        print(f"Configuration:")
        print(f"  Experiment: {args.experiment}")
        print(f"  Balsa directory: {inference.balsa_dir}")
    
    # Run inference
    success = inference.run_inference_pipeline(args.experiment)
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
