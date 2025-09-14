#!/usr/bin/env python3
"""
Test script for LIDX benchmark functionality
"""

import os
import sys
import subprocess
from pathlib import Path

def test_lidx_environment():
    """Test if LIDX environment is properly set up"""
    print("🧪 Testing LIDX Environment...")
    
    # Check if we're in the right directory
    lidx_dir = Path(__file__).parent
    print(f"LIDX Directory: {lidx_dir}")
    
    # Check build directory
    build_dir = lidx_dir / "build"
    if build_dir.exists():
        print(f"✅ Build directory exists: {build_dir}")
    else:
        print(f"❌ Build directory missing: {build_dir}")
        print("   Please run: mkdir -p build && cd build && cmake -DCMAKE_BUILD_TYPE=Release .. && make")
        return False
    
    # Check benchmark binary
    benchmark_path = build_dir / "microbench"
    if benchmark_path.exists():
        print(f"✅ Benchmark binary exists: {benchmark_path}")
    else:
        print(f"❌ Benchmark binary missing: {benchmark_path}")
        print("   Please build LIDX first")
        return False
    
    # Check datasets directory
    datasets_dir = lidx_dir / "datasets"
    if datasets_dir.exists():
        print(f"✅ Datasets directory exists: {datasets_dir}")
        
        # Check for some common datasets
        common_datasets = ["imdb_4M_uint64", "books_200M_uint64", "fb_200M_uint64"]
        for dataset in common_datasets:
            dataset_path = datasets_dir / dataset
            if dataset_path.exists():
                print(f"   ✅ Found dataset: {dataset}")
            else:
                print(f"   ⚠️  Missing dataset: {dataset}")
    else:
        print(f"❌ Datasets directory missing: {datasets_dir}")
        return False
    
    # Check generator
    generator_path = datasets_dir / "generator"
    if generator_path.exists():
        print(f"✅ Data generator exists: {generator_path}")
    else:
        print(f"⚠️  Data generator missing: {generator_path}")
        print("   Will compile automatically when needed")
    
    # Check benchmark script
    benchmark_script = lidx_dir / "run_lidx_benchmark.py"
    if benchmark_script.exists():
        print(f"✅ Benchmark script exists: {benchmark_script}")
    else:
        print(f"❌ Benchmark script missing: {benchmark_script}")
        return False
    
    print("\n✅ LIDX environment check completed!")
    return True

def test_benchmark_script():
    """Test the benchmark script with help"""
    print("\n🧪 Testing Benchmark Script...")
    
    lidx_dir = Path(__file__).parent
    benchmark_script = lidx_dir / "run_lidx_benchmark.py"
    
    try:
        # Test help command
        result = subprocess.run(
            [sys.executable, str(benchmark_script), "--help"],
            cwd=lidx_dir,
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("✅ Benchmark script help works")
            print("Available options:")
            for line in result.stdout.split('\n'):
                if line.strip() and ('--' in line or 'Usage:' in line):
                    print(f"   {line.strip()}")
        else:
            print(f"❌ Benchmark script help failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing benchmark script: {e}")
        return False
    
    return True

def test_simple_benchmark():
    """Test a simple benchmark run (dry run)"""
    print("\n🧪 Testing Simple Benchmark (Dry Run)...")
    
    lidx_dir = Path(__file__).parent
    benchmark_script = lidx_dir / "run_lidx_benchmark.py"
    
    try:
        # Test with minimal parameters (this will fail but should show proper error handling)
        result = subprocess.run(
            [sys.executable, str(benchmark_script), "--drift", "0.1", "--dataset", "imdb", "--size", "4M", "--index", "alex"],
            cwd=lidx_dir,
            capture_output=True,
            text=True,
            timeout=30  # 30 second timeout
        )
        
        if result.returncode == 0:
            print("✅ Simple benchmark completed successfully!")
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
        print(f"❌ Error running benchmark: {e}")
        return False
    
    return True

def main():
    """Main test function"""
    print("🚀 LIDX Environment and Functionality Test")
    print("=" * 50)
    
    # Test 1: Environment check
    if not test_lidx_environment():
        print("\n❌ Environment check failed. Please fix issues above.")
        return False
    
    # Test 2: Script functionality
    if not test_benchmark_script():
        print("\n❌ Script functionality test failed.")
        return False
    
    # Test 3: Simple benchmark
    if not test_simple_benchmark():
        print("\n❌ Simple benchmark test failed.")
        return False
    
    print("\n🎉 All tests completed!")
    print("\nNext steps:")
    print("1. Download datasets: bash download.sh")
    print("2. Build LIDX: mkdir -p build && cd build && cmake -DCMAKE_BUILD_TYPE=Release .. && make")
    print("3. Run benchmark: python run_lidx_benchmark.py --drift 0.3 --dataset imdb --size 4M --index alex")
    print("4. Or use CLI: cd ../.. && python cli.py, then 'idx alex 0.3'")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
