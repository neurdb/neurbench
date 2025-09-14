# LCC (Learned Concurrency Control) Framework

This document describes how to use the LCC benchmark functionality within the NeurBench framework.

## Overview

LCC provides benchmarking for Polyjuice, a learned concurrency control framework. The system automatically handles:

1. **Policy Training**: ERL (Evolutionary Reinforcement Learning) training
2. **Benchmark Execution**: TPCC workload testing
3. **Performance Measurement**: Basic concurrency control metrics
4. **Integration**: Seamless CLI integration with NeurBench

## Quick Start

### 1. Environment Setup

```bash
cd benchmarks/lcc

# Build LCC
MODE=perf make -j dbtest

# Test environment
python test_lcc.py
```

### 2. Run Benchmark via CLI

```bash
# From NeurBench root directory
python cli.py

# Test Polyjuice with ERL training
[neurbench]> lcc
```

### 3. Run Test Script Directly

```bash
cd benchmarks/lcc

# Test Polyjuice functionality
python test_polyjuice.py
```

## Supported Policies

| Policy | Description | Algorithm |
|--------|-------------|-----------|
| **ERL** | Evolutionary Reinforcement Learning | Hybrid RL + Genetic |

## Supported Workloads

| Workload | Description | Characteristics |
|----------|-------------|-----------------|
| **TPCC** | TPC-C Benchmark | OLTP workload with complex transactions |

## Command Line Options

### CLI Command
- `lcc`: Test Polyjuice with ERL training (no parameters needed)

### Test Script
- `python test_polyjuice.py`: Run comprehensive Polyjuice test

## Workflow

### Polyjuice Testing
1. **Environment Check**: Verify LCC setup and training scripts
2. **ERL Training**: Run minimal ERL training for TPCC workload
3. **Performance Test**: Execute basic concurrency control benchmark
4. **Result Analysis**: Display training and benchmark results

## Performance Metrics

### Key Measurements
- **Throughput**: Transactions per second
- **Abort Rate**: Percentage of aborted transactions
- **Commit Rate**: Percentage of committed transactions
- **Latency**: Transaction response time
- **Resource Usage**: CPU, memory, and I/O utilization

### Output Format
```
📊 TPCC Benchmark Results:
   throughput = 1250000
   abort_rate = 0.05
   commit_rate = 0.95
   avg_latency = 0.002
```

## Environment Requirements

### System Requirements
- **OS**: Linux (tested on Ubuntu 18.04+)
- **Memory**: 8GB+ RAM
- **Storage**: 20GB+ free space
- **CPU**: Multi-core recommended

### Dependencies
- **Compiler**: gcc 8.3.0+
- **Build Tools**: make
- **Python**: 3.7+
- **Libraries**: Standard library modules

## Troubleshooting

### Common Issues

#### 1. Build Failures
```bash
# Error: Build directory not found
MODE=perf make -j dbtest
```

#### 2. Missing Training Scripts
```bash
# Error: Training script not found
cd training
ls -la *.py
```

#### 3. Benchmark Binary Missing
```bash
# Error: Benchmark binary not found
cd out-perf.masstree/benchmarks
ls -la dbtest
```

#### 4. Permission Issues
```bash
# Error: Cannot execute binary
chmod +x out-perf.masstree/benchmarks/dbtest
```

### Performance Tips

1. **Scale Factor**: Start with scale factor 1 for testing
2. **Worker Threads**: Match thread count to available CPU cores
3. **Evaluation Time**: Use longer eval times for stable results
4. **Memory**: Ensure sufficient RAM for large workloads

### Debug Mode

Enable verbose output for detailed debugging:

```bash
python run_lcc_benchmark.py --mode test --policy erl --workload tpcc --verbose
```

## Testing

Run the test script to verify your setup:

```bash
cd benchmarks/lcc
python test_lcc.py
```

This will check:
- Environment setup
- Script functionality  
- Basic benchmark execution

## Integration with NeurBench

The LCC framework integrates seamlessly with NeurBench:

1. **CLI Integration**: Use `lcc` command from main CLI
2. **Global Configuration**: Leverages existing drift and dataset settings
3. **Result Analysis**: Compatible with NeurBench analysis tools
4. **Workflow**: Fits into complete NeurBench evaluation pipeline

## Advanced Usage

### Custom Workloads
Modify the benchmark parameters for custom workloads:

```python
# In run_lcc_benchmark.py
def run_benchmark(self):
    cmd = [
        str(benchmark_binary),
        "--bench", self.workload,
        "--scale-factor", str(self.scale_factor),
        "--num-threads", str(self.nworkers),
        # ... other parameters
    ]
```

### Batch Processing
Run multiple benchmarks in sequence:

```bash
#!/bin/bash
# batch_lcc.sh
for policy in erl genetic occ; do
    for workload in tpcc tpce ycsb; do
        echo "Testing $policy with $workload"
        python run_lcc_benchmark.py --mode test --policy $policy --workload $workload
    done
done
```

## Contributing

To add new policies:

1. Implement the policy interface in `training/`
2. Add the policy to the benchmark system
3. Update the CLI choices in `cli.py`
4. Test with various workloads and scale factors

## References

- **Polyjuice**: [Original LCC Framework](https://github.com/stephentu/polyjuice)
- **ERL**: Evolutionary Reinforcement Learning for concurrency control
- **Genetic Algorithms**: Evolutionary optimization for policy tuning

## Support

For issues and questions:

1. Check the troubleshooting section above
2. Run `python test_lcc.py` to diagnose problems
3. Review console output for error messages
4. Check system requirements and dependencies
