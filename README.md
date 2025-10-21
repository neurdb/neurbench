# NRBench

**NRBench** is a benchmark suite designed to evaluate end-to-end learned DBMSs containing all learned components under controllable data and workload drift.

## Dependencies

We provide an `environment.yml` configured for CUDA 11.8. You may modify this file to match your local CUDA version if needed.  
To create the conda environment, run:

```
conda env create -f environment.yml
```

## Tools & Utilities

NRBench provides a drift-aware data and workload generation tool that effectively simulates real-world drift while preserving inherent correlations.

```
conda activate NRBench
```

### Data and Workload Generator

Run the code to generate data according to a specified drift factor with the following command:

```
python dbproc.py --dataset-name=[dataset] --table-name=[table] --drift=[drift factor]
```

For example, to generate a drifted `Name` table for the default dataset (`IMDB`) with a drift factor of `0.1`, we can run the following command:

```
python dbproc.py --dataset-name=imdb --table-name=name --drift=0.1
```

Run the code to generate workloads according to a specified drift factor with the following command:

```
python qproc.py --input_file=[original workload] --output=[drifted workload] --drift=[drift factor]
```

For example, to generate default workloads with a drift factor of 0.1, we can can run this command:

```
python qproc.py --input_file=orig_queries.sql --output=drifted_01_queries.sql --drift=0.1
```

## Benchmarks

We employ NRBench to evaluate state-of-the-art learned query optimizers, learned indexes, and learned concurrency control. We include the codes of evaluators that we used in `benchmark` folder.

## Interactive CLI Interface

NRBench provides an interactive command-line interface for managing learned query optimizers (LQOs) and data generation workflows.

### Quick Start with CLI

```bash
cd benchmarks/lqos
python cli.py
```

### Available Commands

| Command | Description | Example |
|---------|-------------|---------|
| `h`, `help` | Show help message | `h` |
| `q`, `quit` | Exit CLI | `q` |
| `set [KEY] [VALUE]` | Set global configuration | `set drift 0.5` |
| `set` | Show current configuration | `set` |
| `gd DATASET [TABLE] DRIFT [SCALE]` | Generate drifted data | `gd imdb movie 0.3 8.0` |
| `dd DATASET [TABLE]` | Delete data generator models | `dd imdb movie` |
| `tqo LQO_NAME` | Train learned query optimizer | `tqo bao` |
| `iqo LQO_NAME` | Run LQO inference | `iqo bao` |
| `idx [IDX_NAME]` | Test learned index | `idx alex` |
| `lcc` | Test learned concurrency control | `lcc` |

### Supported Learned Query Optimizers (LQOs)

- **Bao** - PostgreSQL-based learned optimizer with experience-based learning
- **Balsa** - Deep reinforcement learning optimizer using sim-to-real learning
- **HybridQO** - Hybrid cost-based and learning-based approach with MCTS
- **Lero** - Learning-to-rank optimizer for query plan selection

### Supported Learned Indexes (LIDX)

- **ALEX** - Updatable Adaptive Learned Index (SIGMOD 2020)
- **ART** - Adaptive Radix Tree (ICDE 2013)
- **B+ Tree** - Traditional baseline index
- **PGM-Index** - Piecewise Geometric Model Index (VLDB 2020)
- **XIndex** - Scalable Learned Index (SIGMOD 2019)

### Supported Learned Concurrency Control (LCC)

- **Polyjuice** - Learned concurrency control framework with ERL training

### Global Configuration

The CLI supports global configuration for common parameters:

```bash
# Set global configuration
[nrbench]> set dataset books
[nrbench]> set drift 0.5

# Show current configuration
[nrbench]> set

# Use simplified commands with global settings
[nrbench]> tqo bao       # Train Bao with current settings
[nrbench]> iqo bao       # Test Bao with current settings
[nrbench]> idx alex      # Test ALEX with current settings
[nrbench]> idx art       # Test ART with current settings
[nrbench]> lcc           # Test Polyjuice 
```

### Complete Workflow Example

```bash
# 1. Generate training data
[nrbench]> gd imdb 0.3 8.0

# 2. Train LQO
[nrbench]> tqo bao

# 3. Run inference
[nrbench]> iqo bao

# 4. Cleanup (optional)
[nrbench]> dd imdb
```

### Advanced CLI Usage

Each LQO supports custom parameters and direct script execution:

```bash
# Direct script execution
cd benchmarks/lqos/bao
python train_bao.py --verbose
python inference_bao.py --test-only

# Custom training parameters
cd benchmarks/lqos/balsa
python train_balsa.py --experiment exp_job_light --verbose
```

## Documentation

### Main Documentation
- **[LQO Framework README](benchmarks/lqos/README.md)** - Complete framework overview and setup guide
- **[CLI Quick Reference](benchmarks/lqos/CLI_QUICK_REFERENCE.md)** - Command reference and examples

### Individual LQO Documentation
- **Bao**: [Original Documentation](benchmarks/lqos/bao/README.md) - Complete Bao workflow
- **Balsa**: [Original Documentation](benchmarks/lqos/balsa/README.md) - Balsa setup and usage
- **HybridQO**: [Original Documentation](benchmarks/lqos/hybrid_qo/README.md) - HybridQO configuration
- **Lero**: [Original Documentation](benchmarks/lqos/Lero/README.md) - Lero implementation details

### Learned Index (LIDX) Documentation
- **[LIDX Framework Guide](benchmarks/lidx/README.md)** - Complete LIDX benchmark framework

### Learned Concurrency Control (LCC) Documentation
- **[LCC Framework Guide](benchmarks/lcc/README.md)** - Complete LCC benchmark framework

## Environment Setup

### Prerequisites
- **Python 3.7+** - Required for all LQOs
- **PostgreSQL 12+** - Required for Bao, Balsa, and Lero
- **CUDA Support** - Optional for GPU acceleration (Balsa, HybridQO)

### Python Dependencies
```bash
pip install torch numpy pandas psycopg2-binary
```

LQO-specific packages are automatically installed during training.

### PostgreSQL Setup
1. Install PostgreSQL (version 12+ recommended)
2. Configure using provided config files in `benchmarks/lqos/conf/`
3. Install required extensions (pg_hint_plan for Bao, custom extensions for others)

## Troubleshooting

### Common Issues
1. **Missing Dependencies** - Training scripts auto-install most dependencies
2. **PostgreSQL Connection** - Check `pg_isready` and config files
3. **Model Training Failures** - Verify disk space and GPU availability
4. **Inference Errors** - Ensure training completed successfully

### Getting Help
- **CLI Help**: Use `h` or `help` command in CLI
- **Script Help**: Run scripts with `--help` flag
- **Error Messages**: Most errors include helpful guidance

## Related Work

This benchmark suite builds upon and integrates with several state-of-the-art learned query optimization approaches:

- **Bao** (SIGMOD 2021) - Making learned query optimization practical
- **Balsa** (SIGMOD 2022) - Learning a query optimizer without expert demonstrations  
- **HybridQO** (VLDB 2022) - Cost-based or learning-based hybrid approach
- **Lero** (VLDB 2023) - Learning-to-rank query optimizer

## License

This project includes code from multiple sources. Please refer to individual LQO directories for specific licensing information.
