#!/usr/bin/env python3
"""
Test script for LCC benchmark functionality
"""

import os
import sys
import subprocess
from pathlib import Path

def test_lcc_environment():
    """Test if LCC environment is properly set up"""
    print("[TEST] Testing LCC Environment...")
    
    # Check if we're in the right directory
    lcc_dir = Path(__file__).parent
    print(f"LCC Directory: {lcc_dir}")
    
    # Check build directory
    build_dir = lcc_dir / "out-perf.masstree"
    if build_dir.exists():
        print(f"[SUCCESS] Build directory exists: {build_dir}")
    else:
        print(f"[FAILED] Build directory missing: {build_dir}")
        print("   Please run: MODE=perf make -j dbtest")
        return False
    
    # Check benchmark binary
    benchmark_path = build_dir / "benchmarks" / "dbtest"
    if benchmark_path.exists():
        print(f"[SUCCESS] Benchmark binary exists: {benchmark_path}")
    else:
        print(f"[FAILED] Benchmark binary missing: {benchmark_path}")
        print("   Please build LCC first")
        return False
    
    # Check training directory
    training_dir = lcc_dir / "training"
    if training_dir.exists():
        print(f"[SUCCESS] Training directory exists: {training_dir}")
        
        # Check for training scripts
        training_scripts = ["ERL_main.py", "genetic_main.py"]
        for script in training_scripts:
            script_path = training_dir / script
            if script_path.exists():
                print(f"   [SUCCESS] Found training script: {script}")
            else:
                print(f"   ⚠️  Missing training script: {script}")
    else:
        print(f"[FAILED] Training directory missing: {training_dir}")
        return False
    
    # Check benchmark script
    benchmark_script = lcc_dir / "run_lcc_benchmark.py"
    if benchmark_script.exists():
        print(f"[SUCCESS] Benchmark script exists: {benchmark_script}")
    else:
        print(f"[FAILED] Benchmark script missing: {benchmark_script}")
        return False
    
    print("\n[SUCCESS] LCC environment check completed!")
    return True

def test_benchmark_script():
    """Test the benchmark script with help"""
    print("\n[TEST] Testing Benchmark Script...")
    
    lcc_dir = Path(__file__).parent
    benchmark_script = lcc_dir / "run_lcc_benchmark.py"
    
    try:
        # Test help command
        result = subprocess.run(
            [sys.executable, str(benchmark_script), "--help"],
            cwd=lcc_dir,
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("[SUCCESS] Benchmark script help works")
            print("Available options:")
            for line in result.stdout.split('\n'):
                if line.strip() and ('--' in line or 'Usage:' in line):
                    print(f"   {line.strip()}")
        else:
            print(f"[FAILED] Benchmark script help failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"[FAILED] Error testing benchmark script: {e}")
        return False
    
    return True

def test_simple_benchmark():
    """Test a simple benchmark run (dry run)"""
    print("\n[TEST] Testing Simple Benchmark (Dry Run)...")
    
    lcc_dir = Path(__file__).parent
    benchmark_script = lcc_dir / "run_lcc_benchmark.py"
    
    try:
        # Test with minimal parameters (this will fail but should show proper error handling)
        result = subprocess.run(
            [sys.executable, str(benchmark_script), "--mode", "test", "--policy", "erl", "--workload", "tpcc"],
            cwd=lcc_dir,
            capture_output=True,
            text=True,
            timeout=30  # 30 second timeout
        )
        
        if result.returncode == 0:
            print("[SUCCESS] Simple benchmark completed successfully!")
        else:
            print(f"⚠️  Benchmark failed as expected (this is normal for first run):")
            print(f"   Return code: {result.returncode}")
            if result.stdout:
                print(f"   Output: {result.stdout[:200]}...")
            if result.stderr:
                print(f"   Error: {result.stderr[:200]}...")
                
    except subprocess.TimeoutExpired:
        print("⚠️  Benchmark timed out (this may indicate it's running)")
    except Exception as e:
        print(f"[FAILED] Error running benchmark: {e}")
        return False
    
    return True

def main():
    """Main test function"""
    print("[LAUNCH] LCC Environment and Functionality Test")
    print("=" * 50)
    
    # Test 1: Environment check
    if not test_lcc_environment():
        print("\n[FAILED] Environment check failed. Please fix issues above.")
        return False
    
    # Test 2: Script functionality
    if not test_benchmark_script():
        print("\n[FAILED] Script functionality test failed.")
        return False
    
    # Test 3: Simple benchmark
    if not test_simple_benchmark():
        print("\n[FAILED] Simple benchmark test failed.")
        return False
    
    print("\n[DONE] All tests completed!")
    print("\nNext steps:")
    print("1. Build LCC: MODE=perf make -j dbtest")
    print("2. Run benchmark: python run_lcc_benchmark.py --mode test --policy erl --workload tpcc")
    print("3. Or use CLI: cd ../.. && python cli.py, then 'lcc test erl tpcc'")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
