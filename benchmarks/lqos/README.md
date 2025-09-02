This part of the test code is cloned and updated based on the codebase used in the paper "Is Your Learned Query Optimizer Behaving As You Expect? A Machine Learning Perspective", VLDB 2024.

# NeurBench LQO Framework

This repository includes a comprehensive framework for learned query optimizer methods with automated training and inference capabilities. The framework provides a unified CLI interface for managing multiple LQO implementations.

## Supported Learned Query Optimizers (LQOs)

### 1. Bao
- **Description**: PostgreSQL-based learned query optimizer that learns to steer the PostgreSQL optimizer by issuing coarse-grained query hints
- **Paper**: "Bao: Making learned query optimization practical" (SIGMOD 2021)
- **Directory**: `bao/`
- **Key Features**: Experience-based learning, PostgreSQL integration, automatic hint generation

### 2. Balsa
- **Description**: End-to-end learned optimizer using deep reinforcement learning and sim-to-real learning
- **Paper**: "Balsa: Learning a Query Optimizer Without Expert Demonstrations" (SIGMOD 2022)
- **Directory**: `balsa/`
- **Key Features**: Deep RL approach, multiple experiment configurations, PostgreSQL integration

### 3. HybridQO
- **Description**: Hybrid cost-based and learning-based approach for query plan selection
- **Paper**: "Cost-based or learning-based? A hybrid query optimizer for query plan selection" (VLDB 2022)
- **Directory**: `hybrid_qo/`
- **Key Features**: MCTS search algorithm, TreeLSTM models, hybrid optimization strategy

### 4. Lero
- **Description**: Learning-to-rank query optimizer that learns the relative order of query plans
- **Paper**: "Lero: A Learning-to-Rank Query Optimizer" (VLDB 2023)
- **Directory**: `Lero/`
- **Key Features**: Rank-based learning, PostgreSQL integration, server-client architecture

## Quick Start

### 1. Start NeurBench CLI
```bash
cd benchmarks/lqos
python cli.py
```

### 2. Available Commands
```bash
[neurbench]> h                    # Show help
[neurbench]> tqo [LQO_NAME]      # Train LQO
[neurbench]> iqo [LQO_NAME]      # Run LQO inference
[neurbench]> gd [DATASET] [TABLE] [DRIFT] [SCALE]  # Generate drifted data
[neurbench]> dd [DATASET] [TABLE] # Delete data generator models
```

### 3. Training Examples
```bash
# Train Bao LQO
[neurbench]> tqo bao

```

### 4. Inference Examples
```bash
# Run Bao inference
[neurbench]> iqo bao

```

## CLI Usage Guide

### Training Commands (`tqo`)

The `tqo` command automates the complete training pipeline for any supported LQO:

```bash
tqo [LQO_NAME]
```

**Supported LQO Names:**
- `bao` - PostgreSQL-based learned optimizer
- `balsa` - Deep reinforcement learning optimizer
- `hybridqo` - Hybrid cost-based and learning-based optimizer
- `lero` - Learning-to-rank optimizer

**What Happens During Training:**
1. **Environment Check** - Validates LQO directory structure and dependencies
2. **Dependency Installation** - Automatically installs required packages
3. **Configuration Setup** - Prepares training environment
4. **Model Training** - Executes the LQO-specific training pipeline
5. **Model Persistence** - Saves trained models for later use
6. **Cleanup** - Manages resources and temporary files

### Inference Commands (`iqo`)

The `iqo` command runs inference tests using trained models:

```bash
iqo [LQO_NAME]
```

**What Happens During Inference:**
1. **Model Validation** - Checks for trained model existence
2. **Server Startup** - Starts required services (if needed)
3. **Connection Testing** - Verifies system connectivity
4. **Inference Execution** - Runs performance tests
5. **Results Reporting** - Summarizes test outcomes
6. **Resource Cleanup** - Manages server processes and resources

### Data Generation Commands (`gd`)

Generate drifted data for training and evaluation:

```bash
gd [DATASET] [TABLE] [DRIFT] [SCALE]
```

**Parameters:**
- `DATASET` - Dataset name (e.g., `imdb`, `stack`)
- `TABLE` - Table name (e.g., `movie`, `posts`)
- `DRIFT` - Drift amount (e.g., `0.3`, `0.5`)
- `SCALE` - Scale factor (e.g., `8.0`, `10.0`)

**Examples:**
```bash
gd imdb movie 0.3 8.0      # Generate drifted IMDB movie data
gd stack posts 0.5 10.0    # Generate drifted STACK posts data
```

### Data Management Commands (`dd`)

Delete data generator models and cleanup resources:

```bash
dd [DATASET] [TABLE]
```

**Examples:**
```bash
dd imdb movie               # Delete IMDB movie data generator
dd stack                    # Delete all STACK data generators
```

## Direct Script Execution

You can also run training and inference scripts directly without the CLI:

### Bao
```bash
cd bao
python train_bao.py         # Train Bao
python inference_bao.py     # Run inference
python test_bao.py          # Test environment
```

### Balsa
```bash
cd balsa
python train_balsa.py       # Train Balsa
python inference_balsa.py   # Run inference
```

### HybridQO
```bash
cd hybrid_qo
python train_hybridqo.py    # Train HybridQO
python inference_hybridqo.py # Run inference
```

### Lero
```bash
cd Lero
python train_lero.py        # Train Lero
python inference_lero.py    # Run inference
```

## Advanced Usage

### Custom Training Parameters

Each LQO supports custom training parameters:

#### Balsa
```bash
cd balsa
python train_balsa.py --experiment exp_job_light --verbose
```

#### HybridQO
```bash
cd hybrid_qo
python train_hybridqo.py --verbose
```

#### Lero
```bash
cd Lero
python train_lero.py --query-path custom_queries.txt --verbose
```

### Custom Inference Options

#### Bao
```bash
cd bao
python inference_bao.py --test-only          # Test environment only
python inference_bao.py --verbose            # Verbose output
```

#### Balsa
```bash
cd balsa
python inference_balsa.py --experiment exp_job_light --verbose
```

#### HybridQO
```bash
cd hybrid_qo
python inference_hybridqo.py --query-file queries.json --verbose
```

#### Lero
```bash
cd Lero
python inference_lero.py --run-comparison --verbose  # Include PostgreSQL comparison
```

## Environment Setup

### Prerequisites

- **Python 3.7+** - All LQOs require Python 3.7 or higher
- **PostgreSQL** - Required for Bao, Balsa, and Lero
- **CUDA Support** - Optional for GPU acceleration (Balsa, HybridQO)

### Python Dependencies

Common packages required across all LQOs:
```bash
pip install torch numpy pandas psycopg2-binary
```

LQO-specific packages are automatically installed during training.

### PostgreSQL Setup

1. **Install PostgreSQL** (version 12+ recommended)
2. **Configure PostgreSQL** using the provided config files in `conf/`
3. **Install Extensions**:
   - `pg_hint_plan` for Bao
   - Custom extensions for Balsa and Lero

## Workflow Examples

### Complete Training and Inference Workflow

```bash
# 1. Start CLI
python cli.py

# 2. Generate training data
[neurbench]> gd imdb movie 0.3 8.0

# 3. Train LQO
[neurbench]> tqo bao

# 4. Run inference
[neurbench]> iqo bao

# 5. Compare with other LQOs
[neurbench]> tqo balsa
[neurbench]> iqo balsa
```

### Development Workflow

```bash
# 1. Test environment
cd bao && python test_bao.py

# 2. Train model
python train_bao.py

# 3. Test inference
python inference_bao.py

# 4. Cleanup (if needed)
cd .. && python cli.py
[neurbench]> dd imdb movie
```

## Troubleshooting

### Common Issues

1. **Missing Dependencies**
   - Run training command - automatic installation will handle most dependencies
   - Check individual LQO READMEs for specific requirements

2. **PostgreSQL Connection Issues**
   - Ensure PostgreSQL is running: `pg_isready`
   - Check connection settings in config files
   - Verify extension installation

3. **Model Training Failures**
   - Check available disk space
   - Verify GPU availability (if using CUDA)
   - Review error logs for specific issues

4. **Inference Errors**
   - Ensure training completed successfully
   - Check model file existence
   - Verify server connectivity

### Getting Help

- **CLI Help**: Use `h` or `help` command in CLI
- **Script Help**: Run scripts with `--help` flag
- **Individual READMEs**: Check LQO-specific documentation
- **Error Messages**: Most errors include helpful guidance

## Contributing

### Adding New LQOs

To add support for a new LQO:

1. **Create Directory**: Add LQO implementation to `benchmarks/lqos/`
2. **Training Script**: Implement `train_[lqo_name].py`
3. **Inference Script**: Implement `inference_[lqo_name].py`
4. **CLI Integration**: Update `handle_tqo()` and `handle_iqo()` functions
5. **Documentation**: Add to this README and create LQO-specific docs

### Code Structure

Follow the established pattern:
- Environment validation
- Dependency management
- Training/inference execution
- Error handling and cleanup
- Progress reporting

## Citations

Since we include the code bases from recent publications, please make sure to also include their citations. We thank the authors of the previous work for making their research available:

>Marcus, Ryan, et al. "**Neo: A Learned Query Optimizer.**" Proceedings of the VLDB Endowment 12.11.

>Marcus, Ryan, et al. "**Bao: Making learned query optimization practical.**" Proceedings of the 2021 International Conference on Management of Data. 2021.

>Yang, Zongheng, et al. "**Balsa: Learning a Query Optimizer Without Expert Demonstrations.**" Proceedings of the 2022 International Conference on Management of Data. 2022.

>Yu, Xiang, et al. "**Cost-based or learning-based? A hybrid query optimizer for query plan selection.**" Proceedings of the VLDB Endowment 15.13 (2022): 3924-3936.

Additionally, we use the Join Order Benchmark published by Leis et al.:

>Leis, Viktor, et al. "**How good are query optimizers, really?.**" Proceedings of the VLDB Endowment 9.3 (2015): 204-215.

And the STACK benchmark published by Marcus et al.:

>Marcus, Ryan, et al. "**Bao: Making learned query optimization practical.**" Proceedings of the 2021 International Conference on Management of Data. 2021.

## License

This project includes code from multiple sources. Please refer to individual LQO directories for specific licensing information.

## 📚 Additional Resources

### Quick Reference
- **[CLI Quick Reference](CLI_QUICK_REFERENCE.md)** - Command reference and examples
- **[Implementation Summary](ALL_LQO_IMPLEMENTATION_SUMMARY.md)** - Technical implementation details

### Individual LQO Documentation
- **Bao**: `bao/README_TQO.md` - Training and inference guide
- **Balsa**: `balsa/README.md` - Original Balsa documentation
- **HybridQO**: `hybrid_qo/README.md` - Original HybridQO documentation  
- **Lero**: `Lero/README.md` - Original Lero documentation

### Getting Help
- **CLI Help**: Use `h` or `help` command in the CLI
- **Script Help**: Run any script with `--help` flag
- **Error Messages**: Most errors include helpful guidance and suggestions
