# NRBench CLI Quick Reference

## 🚀 Quick Start

```bash
cd benchmarks/lqos
python cli.py
```

## 📋 Core Commands

| Command | Description | Example |
|---------|-------------|---------|
| `h`, `help` | Show help message | `h` |
| `q`, `quit` | Exit CLI | `q` |

## 🎯 LQO Training (`tqo`)

| Command | LQO Type | Description |
|---------|----------|-------------|
| `tqo bao` | PostgreSQL-based | Experience-based learning with hints |
| `tqo balsa` | Deep RL | End-to-end learning without expert demos |
| `tqo hybridqo` | Hybrid | Cost-based + learning-based approach |
| `tqo lero` | Learning-to-rank | Rank-based query optimization |

## 🔍 LQO Inference (`iqo`)

| Command | Description | What It Does |
|---------|-------------|--------------|
| `iqo bao` | Test trained Bao model | Run inference tests, validate performance |
| `iqo balsa` | Test trained Balsa model | Execute inference pipeline, report results |
| `iqo hybridqo` | Test trained HybridQO model | Load model, run inference tests |
| `iqo lero` | Test trained Lero model | Start server, run tests, benchmark |

## 📊 Data Management

### Generate Data (`gd`)
```bash
gd [DATASET] [TABLE] [DRIFT] [SCALE]
```

| Parameter | Description | Examples |
|-----------|-------------|----------|
| `DATASET` | Dataset name | `imdb`, `stack` |
| `TABLE` | Table name (optional) | `movie`, `posts` |
| `DRIFT` | Drift amount | `0.3`, `0.5` |
| `SCALE` | Scale factor | `8.0`, `10.0` |

**Examples:**
```bash
gd imdb movie 0.3 8.0      # Specific table
gd imdb 0.2 5.0            # All tables
gd stack posts 0.5 10.0    # STACK dataset
```

### Delete Data (`dd`)
```bash
dd [DATASET] [TABLE]
```

**Examples:**
```bash
dd imdb movie               # Delete specific table
dd stack                    # Delete all STACK data
```

## 🔄 Complete Workflow

### 1. Generate Training Data
```bash
[nrbench]> gd imdb movie 0.3 8.0
```

### 2. Train LQO
```bash
[nrbench]> tqo bao
```

### 3. Run Inference
```bash
[nrbench]> iqo bao
```

### 4. Cleanup (Optional)
```bash
[nrbench]> dd imdb movie
```

## 🛠️ Advanced Usage

### Direct Script Execution

#### Bao
```bash
cd bao
python train_bao.py --verbose
python inference_bao.py --test-only
```

#### Balsa
```bash
cd balsa
python train_balsa.py --experiment exp_job_light
python inference_balsa.py --verbose
```

#### HybridQO
```bash
cd hybrid_qo
python train_hybridqo.py --verbose
python inference_hybridqo.py --query-file queries.json
```

#### Lero
```bash
cd Lero
python train_lero.py --query-path custom_queries.txt
python inference_lero.py --run-comparison
```

### Environment Testing
```bash
# Test Bao environment
cd bao && python test_bao.py

# Test inference environment
cd bao && python test_inference.py
```

## 📁 Directory Structure

```
benchmarks/lqos/
├── bao/                    # Bao LQO
│   ├── train_bao.py       # Training script
│   ├── inference_bao.py   # Inference script
│   └── test_bao.py        # Environment test
├── balsa/                  # Balsa LQO
│   ├── train_balsa.py     # Training script
│   └── inference_balsa.py # Inference script
├── hybrid_qo/              # HybridQO LQO
│   ├── train_hybridqo.py  # Training script
│   └── inference_hybridqo.py # Inference script
├── Lero/                   # Lero LQO
│   ├── train_lero.py      # Training script
│   └── inference_lero.py  # Inference script
├── cli.py                  # Main CLI interface
└── README.md               # Detailed documentation
```

## ⚡ Performance Tips

### Training Time Estimates
- **Bao**: 5-30 minutes
- **Balsa**: 30-120 minutes
- **HybridQO**: 30-90 minutes
- **Lero**: 30-120 minutes

### Resource Requirements
- **Memory**: 2-8 GB
- **CPU**: Moderate to high
- **GPU**: Optional (Balsa, HybridQO)
- **Storage**: 100MB-2GB

## 🚨 Troubleshooting

### Common Issues

1. **"No trained model found"**
   - Run `tqo [LQO_NAME]` first
   - Check model directory exists

2. **"PostgreSQL connection failed"**
   - Ensure PostgreSQL is running: `pg_isready`
   - Check config files in `conf/` directory

3. **"Dependencies missing"**
   - Training scripts auto-install dependencies
   - Check individual LQO READMEs for specific requirements

4. **"Server won't start"**
   - Check for port conflicts
   - Verify permissions and firewall settings

### Getting Help

- **CLI Help**: `h` or `help` command
- **Script Help**: `python script.py --help`
- **Detailed Docs**: See `README.md`
- **Error Messages**: Most include helpful guidance

## 🔗 Related Documentation

- **Main README**: `README.md` - Complete framework overview
- **Bao README**: `bao/README_TQO.md` - Bao-specific details
- **Implementation Summary**: `ALL_LQO_IMPLEMENTATION_SUMMARY.md` - Technical details

## 📝 Notes

- All commands are case-insensitive
- Use `Ctrl+C` to interrupt long-running operations
- Check individual LQO directories for specific configuration options
- Most scripts support `--verbose` flag for detailed output
- Training creates models that inference commands require
