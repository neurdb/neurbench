# LIDX (Learned Index) Benchmark Framework

This document describes how to use the LIDX benchmark functionality within the NeurBench framework.

## Overview

LIDX provides comprehensive benchmarking for learned indexes with drift-aware data generation and testing. The framework supports multiple index types and automatically handles:

1. **Data Generation**: Synthetic data with controllable drift factors
2. **Index Testing**: Multiple operation types (insert, read, mixed, scan)
3. **Performance Measurement**: Throughput, latency, and memory usage
4. **Result Collection**: CSV output for analysis

## Quick Start

### 1. Environment Setup

```bash
cd benchmarks/lidx

# Build LIDX
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release .. && make
cd ..

# Download datasets (optional, will be generated if missing)
bash datasets/download.sh
```

### 2. Run Benchmark via CLI

```bash
# From NeurBench root directory
python cli.py

# Test ALEX index with 0.3 drift on IMDB 4M dataset
[neurbench]> idx alex 0.3

# Test ART index with 0.5 drift on Books 200M dataset  
[neurbench]> idx art 0.5 books 200M

# Test PGM index with 0.2 drift on Facebook 200M dataset
[neurbench]> idx pgm 0.2 fb 200M
```

### 3. Run Benchmark Directly

```bash
cd benchmarks/lidx

# Basic usage
python run_lidx_benchmark.py --drift 0.3 --dataset imdb --size 4M --index alex

# Advanced usage
python run_lidx_benchmark.py \
    --drift 0.5 \
    --dataset books \
    --size 200M \
    --index pgm \
    --operations 5000000 \
    --threads 8 \
    --verbose
```

## Supported Index Types

| Index | Description | Paper |
|-------|-------------|-------|
| **alex** | ALEX: Updatable Adaptive Learned Index | SIGMOD 2020 |
| **art** | Adaptive Radix Tree | ICDE 2013 |
| **btree** | B+ Tree (traditional baseline) | - |
| **pgm** | PGM-Index | VLDB 2020 |
| **xindex** | XIndex: A Scalable Learned Index | SIGMOD 2019 |
| **finedex** | FineDB: Fine-grained Learned Index | - |

## Supported Datasets

| Dataset | Size | Description | Source |
|---------|------|-------------|---------|
| **imdb** | 4M | Movie database keys | IMDB |
| **books** | 200M/400M | Book ISBN keys | Goodreads |
| **fb** | 200M | Facebook user IDs | Facebook |
| **osm** | 200M | OpenStreetMap cell IDs | OSM |
| **wiki** | 200M | Wikipedia page IDs | Wikipedia |

## Drift Factors

The drift factor controls how much the synthetic data deviates from the original:

- **0.0**: No drift (original data)
- **0.1**: Low drift (10% deviation)
- **0.3**: Medium drift (30% deviation)  
- **0.5**: High drift (50% deviation)
- **0.7**: Very high drift (70% deviation)

## Benchmark Phases

Each benchmark run executes four phases:

### Phase 1: Insert Operations
- **Purpose**: Build the index structure
- **Operations**: 100% insert, 0% read
- **Metrics**: Insert throughput, memory usage

### Phase 2: Read Operations  
- **Purpose**: Test lookup performance
- **Operations**: 100% read, 0% insert
- **Metrics**: Read throughput, latency distribution

### Phase 3: Mixed Workload
- **Purpose**: Test concurrent read/write performance
- **Operations**: 80% read, 20% insert
- **Metrics**: Mixed workload throughput

### Phase 4: Range Scan Operations
- **Purpose**: Test range query performance
- **Operations**: 100% scan (range=100)
- **Metrics**: Scan throughput, scan efficiency

## Command Line Options

### Required Parameters
- `--drift`: Drift factor (0.0-1.0)
- `--dataset`: Dataset name
- `--size`: Dataset size
- `--index`: Index type to test

### Optional Parameters
- `--operations`: Number of operations (default: 1M)
- `--threads`: Number of threads (default: 4)
- `--verbose`: Enable detailed output
- `--cleanup`: Remove result files after completion

## Output and Results

### Console Output
```
Starting LIDX Benchmark Workflow
Dataset: imdb (4M)
Index: alex
Drift: 0.3
Operations: 1,000,000
Threads: 4

Phase 1: Insert Operations
[10:30:15] Running insert benchmark...
Insert Benchmark Results:
   throughput = 1250000
   memory = 52428800

Phase 2: Read Operations
[10:30:45] Running read benchmark...
Read Benchmark Results:
   throughput = 2500000
   latency_sample = 0.01
```

### Result Files
Results are saved as CSV files in the `build/` directory:

- `results_insert_alex_30.csv` - Insert phase results
- `results_read_alex_30.csv` - Read phase results  
- `results_mixed_alex_30.csv` - Mixed workload results
- `results_scan_alex_30.csv` - Scan phase results

### CSV Format
Each result file contains:
- Operation counts (success/failure)
- Throughput metrics
- Latency percentiles
- Memory consumption
- Index-specific metrics

## Environment Requirements

### System Requirements
- **OS**: Linux (tested on Ubuntu 18.04+)
- **Memory**: 4GB+ RAM
- **Storage**: 10GB+ free space
- **CPU**: Multi-core recommended

### Dependencies
- **Compiler**: gcc 8.3.0+
- **Build Tools**: cmake 3.14.0+
- **Libraries**: 
  - Intel MKL 2018.4.274
  - Intel TBB 2020.3
  - jemalloc

### Python Requirements
- Python 3.7+
- Standard library modules (no external packages)

## Troubleshooting

### Common Issues

#### 1. Build Failures
```bash
# Error: Build directory not found
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release .. && make
```

#### 2. Missing Datasets
```bash
# Error: Base dataset not found
cd datasets
bash download.sh
```

#### 3. Generator Compilation
```bash
# Error: Data generator not found
cd datasets
g++ --std=c++17 generator.cpp -o generator
```

#### 4. Permission Issues
```bash
# Error: Cannot write file
chmod +x run_lidx_benchmark.py
chmod +x test_lidx.py
```

### Performance Tips

1. **Dataset Size**: Start with smaller datasets (4M) for testing
2. **Thread Count**: Match thread count to available CPU cores
3. **Memory**: Ensure sufficient RAM for large datasets
4. **Storage**: Use SSD for better I/O performance

### Debug Mode

Enable verbose output for detailed debugging:

```bash
python run_lidx_benchmark.py --drift 0.3 --dataset imdb --size 4M --index alex --verbose
```

## Testing

Run the test script to verify your setup:

```bash
cd benchmarks/lidx
python test_lidx.py
```

This will check:
- Environment setup
- Script functionality  
- Basic benchmark execution

## Integration with NeurBench

The LIDX framework integrates seamlessly with NeurBench:

1. **CLI Integration**: Use `idx` command from main CLI
2. **Data Generation**: Leverages existing drift generation
3. **Result Analysis**: Compatible with NeurBench analysis tools
4. **Workflow**: Fits into complete NeurBench evaluation pipeline

## Advanced Usage

### Custom Workloads
Modify the benchmark parameters for custom workloads:

```python
# In run_lidx_benchmark.py
def run_benchmark(self, operation_type, read_ratio=0.0, 
                  insert_ratio=1.0, scan_ratio=0.0):
    # Customize operation ratios
    cmd = [
        str(self.benchmark_path),
        f"--read={read_ratio}",
        f"--insert={insert_ratio}", 
        f"--scan_ratio={scan_ratio}",
        # ... other parameters
    ]
```

### Batch Processing
Run multiple benchmarks in sequence:

```bash
#!/bin/bash
# batch_benchmark.sh
for drift in 0.1 0.3 0.5 0.7; do
    for index in alex art pgm; do
        echo "Testing $index with drift $drift"
        python run_lidx_benchmark.py --drift $drift --index $index --dataset imdb --size 4M
    done
done
```

## Contributing

To add new index types:

1. Implement the index interface in `src/competitor/`
2. Add the index to the benchmark system
3. Update the CLI choices in `cli.py`
4. Test with various datasets and drift factors


## Support

For issues and questions:

1. Check the troubleshooting section above
2. Run `python test_lidx.py` to diagnose problems
3. Review console output for error messages
4. Check system requirements and dependencies


This part of codes is updated based on GRE.
GRE is a benchmark suite for learned indexes and traditional indexes to measure throughput and latency with custom workload (read / write ratio) and any dataset. GRE quantifies datasets using local and global hardness, and includes a synthetic data generator to generate data with various hardness.

See details in GRE's VLDB 2022 paper below. 
```
Chaichon Wongkham, Baotong Lu, Chris Liu, Zhicong Zhong, Eric Lo, and Tianzheng Wang. Are Updatable Learned Indexes Ready?. PVLDB, 15(11): 3004 - 3017, 2022.
```

## Requirements
- gcc 8.3.0+
- cmake 3.14.0+

## Dependencies
- intel-mkl 2018.4.274
- intel-tbb 2020.3
- jemalloc

## Build
```
git submodule update --init # only for the first time
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release .. && make
```

## Basic usage
To calculate throughput:
```
./build/microbench \
--keys_file=./data/dataset \
--keys_file_type={binary,text} \
--read=0.5 --insert=0.5 \
--operations_num=800000000 \
--table_size=-1 \
--init_table_ratio=0.5 \
--thread_num=24 \
--index=index_name \
```
table_size=-1 is to infer from the first line of the file.
init_table_ratio is to specify the proportion of the dataset to bulkload.

For additional features, add additional flags:
- Latency
```
--latency_sample --latency_sample_ratio=0.01
```
- Range query (eg. range = 100)
```
--scan_ratio=1 --scan_num=100
```
- To use Zipfian distribution for lookup
```
--sample_distribution=zipf
```
- To perform data-shift experiment. Note that the key file needs to be generated like so (changing from one dataset to another). This flag just simply prevent the keys be shuffled and preserving the order in the key file
```
--data_shift
```
- Calculate data hardness (PLA-metric) with specified model error bound of the input dataset
```
--dataset_statistic --error_bound=32
```
- If the index implement memory consumption interface
```
--memory
```
All the result will be output to the csv file specified in --output_path flag.



