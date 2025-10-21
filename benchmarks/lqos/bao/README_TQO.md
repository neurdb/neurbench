# Bao Learned Query Optimizer Training Guide

This guide explains how to use the `tqo bao` command to train the Bao learned query optimizer and the `iqo bao` command to run inference.

## Overview

The Bao LQO system provides two main commands:
- **`tqo bao`**: Trains the Bao learned query optimizer
- **`iqo bao`**: Runs inference using a trained Bao model

### Training (`tqo bao`)

The `tqo bao` command automates the complete training pipeline for Bao, including:
- Environment setup and dependency installation
- Bao server startup and connection testing
- Model training with collected experience
- Experimentation to gather training data
- Final model training and deployment

### Inference (`iqo bao`)

The `iqo bao` command runs inference using a trained Bao model:
- Model validation and server startup
- Connection testing and status checking
- Inference test execution
- Performance evaluation and reporting

## Prerequisites

1. **PostgreSQL Database**: A running PostgreSQL instance with Bao extension installed
2. **Python Environment**: Python 3.7+ with required dependencies
3. **Bao Extension**: PostgreSQL extension for Bao must be properly configured
4. **Trained Model**: For inference, a trained model must exist (run `tqo bao` first)

## Usage

### From NRBench CLI

```bash
# Start NRBench interactive shell
python cli.py

# Train Bao LQO
[nrbench]> tqo bao

# Run Bao inference
[nrbench]> iqo bao
```

### Direct Script Execution

```bash
# Navigate to bao directory
cd benchmarks/lqos/bao

# Train Bao
python train_bao.py

# Run inference
python inference_bao.py

# Test environment first
python test_bao.py

# Test inference specifically
python test_inference.py
```

## Training Process

The training process follows these steps:

1. **Environment Check**: Verifies Bao directory structure and dependencies
2. **Server Startup**: Starts the Bao server process
3. **Connection Test**: Tests connection to PostgreSQL
4. **Initial Training**: Trains initial model with available data
5. **Experience Collection**: Runs experiments to gather query execution data
6. **Final Training**: Retrains model with collected experience
7. **Cleanup**: Stops server and cleans up resources

## Inference Process

The inference process follows these steps:

1. **Model Validation**: Checks if trained model exists
2. **Server Startup**: Starts Bao server with trained model
3. **Connection Test**: Tests server connectivity
4. **Status Check**: Verifies server and model status
5. **Inference Tests**: Runs test queries to validate model performance
6. **Results Summary**: Reports test results and overall performance
7. **Cleanup**: Stops server and cleans up resources

## Configuration

### Database Connection

The Bao server expects to connect to PostgreSQL at `172.17.0.1:5432` with:
- Database: `imdbload` (default)
- User: `postgres`
- Password: `postgres`

To modify these settings, edit the connection strings in:
- `bao_server/main.py`
- `bao_server/baoctl.py`
- `run_queries.py`

### Training Parameters

Training parameters can be adjusted in:
- `bao_server/bao.cfg`: Server configuration
- `bao_server/train.py`: Training algorithm parameters
- `bao_server/model.py`: Model architecture

### Inference Options

Inference options include:
- `--verbose`: Enable detailed output
- `--advanced`: Run advanced inference tests
- `--test-only`: Only test environment, don't run full inference

## Testing

### Environment Testing

Before training or inference, test the environment:
```bash
python test_bao.py
```

### Inference Testing

Test inference functionality:
```bash
python test_inference.py
```

These tests verify:
- File structure and dependencies
- Server startup capabilities
- Basic command functionality
- Model availability

## Troubleshooting

### Common Issues

1. **Connection Failed**: Check if PostgreSQL is running and accessible
2. **Dependencies Missing**: Run `pip install -r requirements.txt`
3. **Server Won't Start**: Check for port conflicts or permission issues
4. **Training Timeout**: Increase timeout values in training scripts
5. **No Trained Model**: Run `tqo bao` before `iqo bao`

### Debug Mode

Run with verbose output:
```bash
cd bao_server
python baoctl.py --status
python baoctl.py --test-connection
```

### Logs

Check server logs for detailed error information:
```bash
# Server output is captured in training/inference scripts
# Check for error messages in stderr
```

## Architecture

```
bao/
├── bao_server/          # Core Bao server
│   ├── main.py         # Server main process
│   ├── train.py        # Training logic
│   ├── model.py        # Neural network model
│   └── baoctl.py       # Control interface
├── train_bao.py         # Automated training script
├── inference_bao.py     # Automated inference script
├── test_bao.py          # Environment testing
├── test_inference.py    # Inference testing
└── run_queries.py       # Query execution utilities
```

## Integration with NRBench

Both commands integrate with the NRBench framework:
- Automatically handle directory navigation
- Provide consistent error handling
- Integrate with CLI help system
- Support future extensions for other LQOs

## Workflow

### Complete Workflow

1. **Setup**: Ensure Bao environment is ready
2. **Training**: Run `tqo bao` to train the model
3. **Validation**: Run `iqo bao` to test inference
4. **Production**: Use trained model for query optimization

### Development Workflow

1. **Test Environment**: `python test_bao.py`
2. **Train Model**: `python train_bao.py`
3. **Test Inference**: `python test_inference.py`
4. **Run Inference**: `python inference_bao.py`

## Future Extensions

The framework is designed to support additional LQOs:
- **Balsa**: Alternative learned query optimizer
- **Lero**: Another learning-based approach
- **Custom**: User-defined LQO implementations

## References

- [Bao Paper](https://dl.acm.org/doi/abs/10.1145/3448016.3452838)
- [Bao Documentation](https://rmarcus.info/bao_docs/)
- [Original Repository](https://github.com/learnedsystems/BaoForPostgreSQL)
