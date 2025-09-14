#!/usr/bin/env python3
"""
LIDX Benchmark Runner Script

This script automates the LIDX benchmark process:
1. Generates synthetic data with specified drift
2. Runs the benchmark with insert operations
3. Runs the benchmark with read operations
4. Collects and reports results

Usage:
    python run_lidx_benchmark.py --drift 0.3 --dataset imdb --size 4M --index alex
"""

import os
import sys
import subprocess
import argparse
import time
import json
from pathlib import Path


class LIDXBenchmarkRunner:
    def __init__(self, drift: float, dataset: str, size: str, index: str, 
                 operations: int = 1000000, threads: int = 4, verbose: bool = False):
        self.drift = drift
        self.dataset = dataset
        self.size = size
        self.index = index
        self.operations = operations
        self.threads = threads
        self.verbose = verbose
        
        # Paths
        self.lidx_dir = Path(__file__).parent
        self.build_dir = self.lidx_dir / "build"
        self.datasets_dir = self.lidx_dir / "datasets"
        self.generator_path = self.datasets_dir / "generator"
        self.benchmark_path = self.build_dir / "microbench"
        
        # Dataset file names
        self.dataset_name = f"{dataset}_{size}_uint64"
        self.drift_dataset_name = f"{dataset}_{size}_uint64_drift_{int(drift * 100):02d}"
        self.dataset_file = self.datasets_dir / self.dataset_name
        self.drift_dataset_file = self.datasets_dir / self.drift_dataset_name
        
    def log(self, message: str):
        """Log message with timestamp"""
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] {message}")
        
    def check_prerequisites(self):
        """Check if all required components are available"""
        self.log("Checking prerequisites...")
        
        # Check if build directory exists
        if not self.build_dir.exists():
            self.log("❌ Build directory not found. Please build LIDX first:")
            self.log("   cd benchmarks/lidx")
            self.log("   mkdir -p build && cd build")
            self.log("   cmake -DCMAKE_BUILD_TYPE=Release .. && make")
            return False
            
        # Check if benchmark binary exists
        if not self.benchmark_path.exists():
            self.log("❌ Benchmark binary not found. Please build LIDX first.")
            return False
            
        # Check if generator exists
        if not self.generator_path.exists():
            self.log("❌ Data generator not found. Compiling generator...")
            if not self.compile_generator():
                return False
                
        self.log("✅ Prerequisites check passed")
        return True
        
    def compile_generator(self):
        """Compile the data generator"""
        try:
            generator_cpp = self.datasets_dir / "generator.cpp"
            if not generator_cpp.exists():
                self.log("❌ Generator source not found")
                return False
                
            cmd = ["g++", "--std=c++17", str(generator_cpp), "-o", str(self.generator_path)]
            self.log(f"Compiling generator: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, cwd=self.datasets_dir, capture_output=True, text=True)
            if result.returncode != 0:
                self.log(f"❌ Generator compilation failed: {result.stderr}")
                return False
                
            self.log("✅ Generator compiled successfully")
            return True
            
        except Exception as e:
            self.log(f"❌ Generator compilation error: {e}")
            return False
            
    def generate_drift_data(self):
        """Generate synthetic data with specified drift"""
        self.log(f"Generating drift data with drift factor {self.drift}...")
        
        # Check if drift dataset already exists
        if self.drift_dataset_file.exists():
            self.log(f"✅ Drift dataset already exists: {self.drift_dataset_file}")
            return True
            
        # Check if base dataset exists
        if not self.dataset_file.exists():
            self.log(f"❌ Base dataset not found: {self.dataset_file}")
            self.log("Please download datasets first using: bash download.sh")
            return False
            
        # Generate drift data using generator
        # Parameters: le, ge, lv, gv, num, path
        # le: local epsilon (drift factor)
        # ge: global epsilon (drift factor) 
        # lv: local non-linearity (number of models)
        # gv: global non-linearity (number of models)
        # num: number of keys to generate
        # path: output path
        
        # Estimate number of keys from file size
        file_size = self.dataset_file.stat().st_size
        num_keys = file_size // 8  # uint64 = 8 bytes
        
        # Generate drift parameters
        le = int(self.drift * 100)  # local epsilon
        ge = int(self.drift * 100)  # global epsilon
        lv = max(1, int(1 / self.drift))  # local models
        gv = max(1, int(1 / self.drift))  # global models
        
        cmd = [
            str(self.generator_path),
            str(le), str(ge), str(lv), str(gv), str(num_keys),
            str(self.drift_dataset_file)
        ]
        
        self.log(f"Running generator: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, cwd=self.datasets_dir, capture_output=True, text=True)
            if result.returncode != 0:
                self.log(f"❌ Data generation failed: {result.stderr}")
                return False
                
            self.log("✅ Drift data generated successfully")
            return True
            
        except Exception as e:
            self.log(f"❌ Data generation error: {e}")
            return False
            
    def run_benchmark(self, operation_type: str, read_ratio: float = 0.0, 
                      insert_ratio: float = 1.0, scan_ratio: float = 0.0):
        """Run the benchmark with specified parameters"""
        self.log(f"Running {operation_type} benchmark...")
        
        # Prepare benchmark command
        cmd = [
            str(self.benchmark_path),
            f"--keys_file={self.drift_dataset_file}",
            "--keys_file_type=binary",
            f"--read={read_ratio}",
            f"--insert={insert_ratio}",
            f"--scan_ratio={scan_ratio}",
            f"--operations_num={self.operations}",
            "--table_size=-1",  # Infer from file
            "--init_table_ratio=0.5",  # Bulk load 50% initially
            f"--thread_num={self.threads}",
            f"--index={self.index}",
            "--latency_sample",
            "--latency_sample_ratio=0.01",
            "--memory",
            f"--output_path=results_{operation_type}_{self.index}_{int(self.drift * 100)}.csv"
        ]
        
        if self.verbose:
            self.log(f"Benchmark command: {' '.join(cmd)}")
            
        try:
            result = subprocess.run(cmd, cwd=self.build_dir, capture_output=True, text=True)
            
            if result.returncode != 0:
                self.log(f"❌ Benchmark failed: {result.stderr}")
                return False
                
            # Parse and display results
            self.parse_benchmark_output(result.stdout, operation_type)
            return True
            
        except Exception as e:
            self.log(f"❌ Benchmark error: {e}")
            return False
            
    def parse_benchmark_output(self, output: str, operation_type: str):
        """Parse benchmark output and display key metrics"""
        self.log(f"📊 {operation_type.title()} Benchmark Results:")
        
        # Extract key metrics from output
        lines = output.split('\n')
        for line in lines:
            if 'throughput' in line.lower() or 'latency' in line.lower():
                print(f"   {line.strip()}")
                
        # Look for result file
        result_files = list(self.build_dir.glob(f"results_{operation_type}_{self.index}_{int(self.drift * 100)}.csv"))
        if result_files:
            self.log(f"📁 Results saved to: {result_files[0]}")
            
    def run_full_benchmark(self):
        """Run the complete benchmark workflow"""
        self.log("🚀 Starting LIDX Benchmark Workflow")
        self.log(f"Dataset: {self.dataset} ({self.size})")
        self.log(f"Index: {self.index}")
        self.log(f"Drift: {self.drift}")
        self.log(f"Operations: {self.operations:,}")
        self.log(f"Threads: {self.threads}")
        
        # Step 1: Check prerequisites
        if not self.check_prerequisites():
            return False
            
        # Step 2: Generate drift data
        if not self.generate_drift_data():
            return False
            
        # Step 3: Run insert benchmark
        self.log("\n📝 Phase 1: Insert Operations")
        if not self.run_benchmark("insert", read_ratio=0.0, insert_ratio=1.0):
            return False
            
        # Step 4: Run read benchmark
        self.log("\n📖 Phase 2: Read Operations")
        if not self.run_benchmark("read", read_ratio=1.0, insert_ratio=0.0):
            return False
            
        # Step 5: Run mixed workload benchmark
        self.log("\n🔄 Phase 3: Mixed Workload (80% read, 20% insert)")
        if not self.run_benchmark("mixed", read_ratio=0.8, insert_ratio=0.2):
            return False
            
        # Step 6: Run scan benchmark
        self.log("\n🔍 Phase 4: Range Scan Operations")
        if not self.run_benchmark("scan", read_ratio=0.0, insert_ratio=0.0, scan_ratio=1.0):
            return False
            
        self.log("\n✅ LIDX Benchmark completed successfully!")
        return True
        
    def cleanup(self):
        """Clean up temporary files"""
        self.log("🧹 Cleaning up temporary files...")
        
        # Remove result files
        result_pattern = f"results_*_{self.index}_{int(self.drift * 100)}.csv"
        for result_file in self.build_dir.glob(result_pattern):
            try:
                result_file.unlink()
                self.log(f"Removed: {result_file}")
            except Exception as e:
                self.log(f"Failed to remove {result_file}: {e}")


def main():
    parser = argparse.ArgumentParser(description="LIDX Benchmark Runner")
    parser.add_argument("--drift", type=float, required=True, 
                       help="Drift factor (0.0-1.0)")
    parser.add_argument("--dataset", type=str, default="imdb",
                       choices=["imdb", "books", "fb", "osm", "wiki"],
                       help="Dataset name")
    parser.add_argument("--size", type=str, default="4M",
                       choices=["4M", "200M", "400M"],
                       help="Dataset size")
    parser.add_argument("--index", type=str, default="alex",
                       choices=["alex", "art", "btree", "pgm", "xindex", "finedex"],
                       help="Index type to test")
    parser.add_argument("--operations", type=int, default=1000000,
                       help="Number of operations to perform")
    parser.add_argument("--threads", type=int, default=4,
                       help="Number of threads")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose output")
    parser.add_argument("--cleanup", action="store_true",
                       help="Clean up result files after completion")
    
    args = parser.parse_args()
    
    # Validate drift factor
    if not 0.0 <= args.drift <= 1.0:
        print("❌ Error: Drift factor must be between 0.0 and 1.0")
        sys.exit(1)
        
    # Create and run benchmark
    runner = LIDXBenchmarkRunner(
        drift=args.drift,
        dataset=args.dataset,
        size=args.size,
        index=args.index,
        operations=args.operations,
        threads=args.threads,
        verbose=args.verbose
    )
    
    try:
        success = runner.run_full_benchmark()
        
        if args.cleanup:
            runner.cleanup()
            
        if success:
            sys.exit(0)
        else:
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️  Benchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
