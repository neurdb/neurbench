#!/usr/bin/env python3
"""
LCC (Learned Concurrency Control) Benchmark Runner Script

This script automates the LCC benchmark process:
1. Trains learned concurrency control policies
2. Runs benchmarks with different workloads
3. Collects and reports results

Usage:
    python run_lcc_benchmark.py --mode train --policy erl --workload tpcc
    python run_lcc_benchmark.py --mode test --policy erl --workload tpcc
"""

import os
import sys
import subprocess
import argparse
import time
import json
from pathlib import Path


class LCCBenchmarkRunner:
    def __init__(self, mode: str, policy: str, workload: str, 
                 scale_factor: int = 1, nworkers: int = 8, eval_time: float = 1.0,
                 verbose: bool = False):
        self.mode = mode
        self.policy = policy
        self.workload = workload
        self.scale_factor = scale_factor
        self.nworkers = nworkers
        self.eval_time = eval_time
        self.verbose = verbose
        
        # Paths
        self.lcc_dir = Path(__file__).parent
        self.training_dir = self.lcc_dir / "training"
        self.benchmarks_dir = self.lcc_dir / "benchmarks"
        self.build_dir = self.lcc_dir / "out-perf.masstree"
        
        # Policy files
        self.policy_files = {
            "erl": "input-RL-ic3-new-tpcc.txt",
            "genetic": "input-ic3-tpcc.txt",
            "occ": "input-occ-tpcc.txt"
        }
        
    def log(self, message: str):
        """Log message with timestamp"""
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] {message}")
        
    def check_prerequisites(self):
        """Check if all required components are available"""
        self.log("Checking prerequisites...")
        
        # Check if build directory exists
        if not self.build_dir.exists():
            self.log("❌ Build directory not found. Please build LCC first:")
            self.log("   cd benchmarks/lcc")
            self.log("   MODE=perf make -j dbtest")
            return False
            
        # Check if benchmark binary exists
        benchmark_binary = self.build_dir / "benchmarks" / "dbtest"
        if not benchmark_binary.exists():
            self.log("❌ Benchmark binary not found. Please build LCC first.")
            return False
            
        # Check if training directory exists
        if not self.training_dir.exists():
            print(f"❌ Training directory not found at {self.training_dir}")
            return False
            
        self.log("✅ Prerequisites check passed")
        return True
        
    def train_policy(self):
        """Train a learned concurrency control policy"""
        self.log(f"Training {self.policy.upper()} policy for {self.workload.upper()} workload...")
        
        if self.policy == "erl":
            return self.train_erl_policy()
        elif self.policy == "genetic":
            return self.train_genetic_policy()
        else:
            self.log(f"❌ Unknown policy type: {self.policy}")
            return False
            
    def train_erl_policy(self):
        """Train ERL policy"""
        self.log("Training ERL (Evolutionary Reinforcement Learning) policy...")
        
        # Check if ERL training script exists
        erl_script = self.training_dir / "ERL_main.py"
        if not erl_script.exists():
            self.log(f"❌ ERL training script not found at {erl_script}")
            return False
            
        # Prepare training command
        cmd = [
            sys.executable, str(erl_script),
            "--expr-name", f"erl-{self.workload}-{self.policy}",
            "--workload-type", self.workload,
            "--scale-factor", str(self.scale_factor),
            "--nworkers", str(self.nworkers),
            "--eval-time", str(self.eval_time),
            "--max-iterations", "50",  # Reduced for demo
            "--samples-per-distribution", "16",
            "--psize", "4"
        ]
        
        if self.verbose:
            self.log(f"Training command: {' '.join(cmd)}")
            
        try:
            self.log("Starting ERL training...")
            result = subprocess.run(cmd, cwd=self.training_dir, capture_output=True, text=True)
            
            if result.returncode == 0:
                self.log("✅ ERL training completed successfully!")
                return True
            else:
                self.log(f"❌ ERL training failed: {result.stderr}")
                return False
                
        except Exception as e:
            self.log(f"❌ ERL training error: {e}")
            return False
            
    def train_genetic_policy(self):
        """Train genetic policy"""
        self.log("Training genetic policy...")
        
        # Check if genetic training script exists
        genetic_script = self.training_dir / "genetic_main.py"
        if not genetic_script.exists():
            self.log(f"❌ Genetic training script not found at {genetic_script}")
            return False
            
        # Prepare training command
        cmd = [
            sys.executable, str(genetic_script),
            "--expr-name", f"genetic-{self.workload}",
            "--workload-type", self.workload,
            "--scale-factor", str(self.scale_factor),
            "--nworkers", str(self.nworkers),
            "--eval-time", str(self.eval_time),
            "--max-iterations", "30",  # Reduced for demo
            "--psize", "4"
        ]
        
        if self.verbose:
            self.log(f"Training command: {' '.join(cmd)}")
            
        try:
            self.log("Starting genetic training...")
            result = subprocess.run(cmd, cwd=self.training_dir, capture_output=True, text=True)
            
            if result.returncode == 0:
                self.log("✅ Genetic training completed successfully!")
                return True
            else:
                self.log(f"❌ Genetic training failed: {result.stderr}")
                return False
                
        except Exception as e:
            self.log(f"❌ Genetic training error: {e}")
            return False
            
    def run_benchmark(self):
        """Run benchmark with trained policy"""
        self.log(f"Running {self.workload.upper()} benchmark with {self.policy.upper()} policy...")
        
        # Check if benchmark binary exists
        benchmark_binary = self.build_dir / "benchmarks" / "dbtest"
        if not benchmark_binary.exists():
            self.log(f"❌ Benchmark binary not found at {benchmark_binary}")
            return False
            
        # Prepare benchmark command
        cmd = [
            str(benchmark_binary),
            "--bench", self.workload,
            "--retry-aborted-transactions",
            "--parallel-loading",
            "--db-type", "ndb-ic3",
            "--backoff-aborted-transactions",
            "--scale-factor", str(self.scale_factor),
            "--num-threads", str(self.nworkers),
            "--runtime", str(self.eval_time)
        ]
        
        # Add workload-specific options
        if self.workload == "tpcc":
            cmd.extend(["--bench-opt", "--partition --length 10"])
        elif self.workload == "tpce":
            cmd.extend(["--bench-opt", "--partition"])
        elif self.workload == "ycsb":
            cmd.extend(["--bench-opt", "--partition --length 10"])
            
        if self.verbose:
            self.log(f"Benchmark command: {' '.join(cmd)}")
            
        try:
            self.log("Starting benchmark...")
            result = subprocess.run(cmd, cwd=self.build_dir, capture_output=True, text=True)
            
            if result.returncode == 0:
                self.log("✅ Benchmark completed successfully!")
                self.parse_benchmark_output(result.stdout)
                return True
            else:
                self.log(f"❌ Benchmark failed: {result.stderr}")
                return False
                
        except Exception as e:
            self.log(f"❌ Benchmark error: {e}")
            return False
            
    def parse_benchmark_output(self, output: str):
        """Parse benchmark output and display key metrics"""
        self.log(f"📊 {self.workload.upper()} Benchmark Results:")
        
        # Extract key metrics from output
        lines = output.split('\n')
        for line in lines:
            if any(keyword in line.lower() for keyword in ['throughput', 'abort', 'commit', 'latency']):
                print(f"   {line.strip()}")
                
    def run_full_workflow(self):
        """Run the complete LCC workflow"""
        self.log("🚀 Starting LCC Benchmark Workflow")
        self.log(f"Mode: {self.mode}")
        self.log(f"Policy: {self.policy}")
        self.log(f"Workload: {self.workload}")
        self.log(f"Scale Factor: {self.scale_factor}")
        self.log(f"Workers: {self.nworkers}")
        self.log(f"Eval Time: {self.eval_time}s")
        
        # Step 1: Check prerequisites
        if not self.check_prerequisites():
            return False
            
        # Step 2: Train policy (if in training mode)
        if self.mode == "train":
            if not self.train_policy():
                return False
                
        # Step 3: Run benchmark
        if not self.run_benchmark():
            return False
            
        self.log("\n✅ LCC benchmark completed successfully!")
        return True
        
    def cleanup(self):
        """Clean up temporary files"""
        self.log("🧹 Cleaning up temporary files...")
        
        # Remove training artifacts
        training_artifacts = [
            "*.log",
            "*.out",
            "saved_model/*",
            "kids/*"
        ]
        
        for pattern in training_artifacts:
            for artifact in self.training_dir.glob(pattern):
                try:
                    if artifact.is_file():
                        artifact.unlink()
                    elif artifact.is_dir():
                        import shutil
                        shutil.rmtree(artifact)
                    self.log(f"Removed: {artifact}")
                except Exception as e:
                    self.log(f"Failed to remove {artifact}: {e}")


def main():
    parser = argparse.ArgumentParser(description="LCC Benchmark Runner")
    parser.add_argument("--mode", type=str, required=True, 
                       choices=["train", "test"],
                       help="Mode: train (train policy) or test (run benchmark)")
    parser.add_argument("--policy", type=str, required=True,
                       choices=["erl", "genetic", "occ"],
                       help="Policy type to use")
    parser.add_argument("--workload", type=str, required=True,
                       choices=["tpcc", "tpce", "ycsb"],
                       help="Workload type to test")
    parser.add_argument("--scale-factor", type=int, default=1,
                       help="Scale factor for the workload")
    parser.add_argument("--nworkers", type=int, default=8,
                       help="Number of worker threads")
    parser.add_argument("--eval-time", type=float, default=1.0,
                       help="Evaluation time in seconds")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose output")
    parser.add_argument("--cleanup", action="store_true",
                       help="Clean up training artifacts after completion")
    
    args = parser.parse_args()
    
    # Create and run benchmark
    runner = LCCBenchmarkRunner(
        mode=args.mode,
        policy=args.policy,
        workload=args.workload,
        scale_factor=args.scale_factor,
        nworkers=args.nworkers,
        eval_time=args.eval_time,
        verbose=args.verbose
    )
    
    try:
        success = runner.run_full_workflow()
        
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
