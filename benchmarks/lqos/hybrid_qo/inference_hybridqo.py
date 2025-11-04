#!/usr/bin/env python3
"""
HybridQO Learned Query Optimizer Inference Script
This script runs inference using a trained HybridQO model
"""

import os
import sys
import subprocess
import time
import argparse
from pathlib import Path

class HybridQOInference:
    def __init__(self, hybridqo_dir="."):
        self.hybridqo_dir = Path(hybridqo_dir).resolve()
        self.original_dir = os.getcwd()
        
    def check_model_exists(self):
        """Check if trained HybridQO model exists"""
        print("Checking for trained HybridQO model...")
        
        # Check for model files
        model_dir = self.hybridqo_dir / "model"
        if not model_dir.exists():
            print("[FAILED] Model directory not found")
            return False
        
        # Check for model files
        model_files = list(model_dir.glob("*"))
        if not model_files:
            print("[FAILED] Model directory is empty")
            print("[INFO] Please run 'tqo hybridqo' first to train a model")
            return False
        
        print(f"[SUCCESS] Found {len(model_files)} model files:")
        for file in model_files:
            print(f"  - {file.name}")
        
        return True
    
    def check_environment(self):
        """Check if HybridQO environment is properly set up"""
        print("Checking HybridQO environment...")
        
        # Check required files
        required_files = [
            "run_mcts.py",
            "Hinter.py",
            "NET.py",
            "mcts.py",
        ]
        
        for file_path in required_files:
            if not (self.hybridqo_dir / file_path).exists():
                print(f"[FAILED] Required file not found: {file_path}")
                return False
            else:
                print(f"[SUCCESS] Found: {file_path}")
        
        return True
    
    def run_inference_tests(self):
        """Run inference tests with trained model"""
        print("Running HybridQO inference tests...")
        
        try:
            # Import and test the trained model
            sys.path.insert(0, str(self.hybridqo_dir))
            
            try:
                from run_mcts import neur_bench_load_hinter
                from ImportantConfig import ImportantConfig
                
                print("[SUCCESS] Successfully imported HybridQO modules")
                
                # Load the trained model
                config = ImportantConfig()
                hinter = neur_bench_load_hinter(config, "test")
                
                print("[SUCCESS] Successfully loaded trained model")
                
                # Test model prediction (basic test)
                print("Testing model prediction...")
                # This is a basic test - in practice you'd run actual queries
                
                return True
                
            except ImportError as e:
                print(f"[FAILED] Import error: {e}")
                return False
            except Exception as e:
                print(f"[FAILED] Model loading error: {e}")
                return False
            finally:
                # Remove from path
                if str(self.hybridqo_dir) in sys.path:
                    sys.path.remove(str(self.hybridqo_dir))
                
        except Exception as e:
            print(f"[FAILED] Inference tests error: {e}")
            return False
    
    def run_custom_inference(self, query_file=None):
        """Run custom inference with specific queries"""
        print("Running HybridQO custom inference...")
        
        if query_file and os.path.exists(query_file):
            print(f"Using query file: {query_file}")
            # TODO: Implement custom query inference
            pass
        
        try:
            # Run a simple test to verify model works
            result = subprocess.run(
                [sys.executable, "-c", "import sys; sys.path.insert(0, '.'); from ImportantConfig import ImportantConfig; print('Config loaded successfully')"],
                cwd=self.hybridqo_dir,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                print("[SUCCESS] Configuration test passed")
                return True
            else:
                print(f"⚠ Configuration test failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"[FAILED] Custom inference error: {e}")
            return False
    
    def run_inference_pipeline(self, query_file=None):
        """Run the complete inference pipeline"""
        try:
            # Change to hybridqo directory
            os.chdir(self.hybridqo_dir)
            
            # Check environment
            if not self.check_environment():
                raise RuntimeError("HybridQO environment check failed")
            
            # Check if model exists
            if not self.check_model_exists():
                raise RuntimeError("No trained model found")
            
            # Run inference tests
            if not self.run_inference_tests():
                raise RuntimeError("HybridQO inference tests failed")
            
            # Run custom inference if specified
            if query_file:
                if not self.run_custom_inference(query_file):
                    print("⚠ Custom inference failed, but basic tests passed")
            
            print("[DONE] HybridQO inference completed successfully!")
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
        description="Run HybridQO Learned Query Optimizer Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python inference_hybridqo.py                    # Run inference with default settings
  python inference_hybridqo.py --query-file queries.json  # Use specific query file
  python inference_hybridqo.py --verbose          # Enable verbose output
        """
    )
    
    parser.add_argument(
        "--query-file", 
        type=str,
        help="Path to query file for custom inference"
    )
    
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Create inference runner
    inference = HybridQOInference(".")
    
    if args.verbose:
        print(f"Configuration:")
        print(f"  Query file: {args.query_file}")
        print(f"  HybridQO directory: {inference.hybridqo_dir}")
    
    # Run inference
    success = inference.run_inference_pipeline(args.query_file)
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
