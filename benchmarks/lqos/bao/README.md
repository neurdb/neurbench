> This repository has been taken from [Bao](https://github.com/learnedsystems/BaoForPostgreSQL). Please refer to the original repository and cite their [publication](https://dl.acm.org/doi/abs/10.1145/3448016.3452838).


# Bao

## README

### Installation

```
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

After installation, you can always enter the environment using the `activate_env.sh` scirpt.

### Changelog from original repository

- Added more logging
- Dockerized the setup
- Changed the postgres config to use the same parallelized setup that the Balsa paper used
- Changed the timeout of queries sent to Postgres
    - Timeout increased from 5min to 15min
    - In case of a timeout or an error, we set the execution time to twice the timeout limit



### Original README from the [author's repository](https://github.com/learnedsystems/BaoForPostgreSQL)

This is a prototype implementation of Bao for PostgreSQL. Bao is a learned query optimizer that learns to "steer" the PostgreSQL optimizer by issuing coarse-grained query hints. For more information about Bao, [check out the paper](https://rm.cab/bao).

Documentation, including a tutorial, is available here: https://rmarcus.info/bao_docs/

While this repository contains working prototype implementations of many of the pieces required to build a production-ready learned query optimizer, this code itself should not be used in production in its current form. Notable limitations include:

* The reward function is currently restricted to being a user-supplied value or the query latency in wall time. Thus, results may be inconsistent with high degrees of parallelism.
* The Bao server component does not perform any level of authentication or encryption. Do not run it on a machine directly accessible from an untrusted network.
* The code has not been audited for security issues. Since the PostgreSQL integration is written using the C hooks system, there are almost certainly issues.

This software is available under the AGPLv3 license.

---

## Automated Training and Testing Pipeline

This directory includes automated scripts for training and testing Bao on custom query workloads.

### Quick Start

#### 1. Check Environment

```bash
./check_bao_env.sh ab_test
```

Verifies:
- Query directories exist
- Python dependencies installed
- Database connection working
- All required scripts present

#### 2. Run Training and Testing

```bash
./schedule_bao_simple.sh ab_test
```

Where `ab_test` is the name of your query set directory under `../../../queries/`.

This will:
1. Train Bao on training queries from `queries/ab_test/train/`
2. Test with Bao optimizer on queries from `queries/ab_test/test/`
3. Test with PostgreSQL baseline optimizer
4. Save models and logs to `../../../bao_logs_all/`
5. Clean environment for next run (clear experience database)

### File Structure

**Active Scripts:**
```
benchmarks/lqos/bao/
├── schedule_bao_simple.sh   # Main training and testing pipeline
├── check_bao_env.sh          # Environment verification script
├── train_bao.py              # Training script
├── test_bao.py               # Testing script
├── bao_server/               # Bao server and model code
└── README.md                 # This file
```

**Archived files:**
- See `_archived/` directory for scripts/files not used in current pipeline

### Configuration

Edit `schedule_bao_simple.sh` to change:
- `DATASET`: Database name (default: `imdb_ori`)
- `QUERY_SET`: Query set directory name (default: `job`)
- PostgreSQL port is configured via CLI module (default: 5430)

### Output Files

All results saved to `../../../bao_logs_all/`:

```
bao_logs_all/
├── TIMESTAMP_DATASET_QUERYSET/
│   ├── bao_default_model/        # Trained model
│   ├── bao_training_results.log  # Training log
│   └── bao_test_results.log      # Testing log
├── DATE_train_DATASET_QUERYSET.log    # Full training output
├── DATE_test_bao_DATASET_QUERYSET.log # Bao test output
└── DATE_test_pg_DATASET_QUERYSET.log  # PostgreSQL test output
```

### Usage Examples

```bash
# Use default query set
./schedule_bao_simple.sh

# Use specific query set
./schedule_bao_simple.sh job

# Use custom query set (ab_test is an example)
./schedule_bao_simple.sh ab_test
```

### Key Features

- Clean Runs: Each run starts with fresh experience database
- Automatic Backup: Models saved before cleaning
- CPU Mode: Runs on CPU to avoid CUDA multiprocessing issues
- Complete Pipeline: Train -> Test Bao -> Test PG -> Save -> Clean
- Detailed Logs: All steps logged for debugging

### Notes

- Models are saved after each run, old models backed up
- Experience database (`bao.db`) is cleared after each run for clean training
- Bao server automatically started/stopped for training and testing
- PostgreSQL connection uses port 5430 (configurable in `cli.py`)

### Troubleshooting

**Check latest logs:**
```bash
tail -f ../../../bao_logs_all/*_train_*.log
```

**Check if server is running:**
```bash
ps aux | grep bao_server
```

**Check database connection:**
```bash
psql -h 172.17.0.1 -p 5430 -U postgres -d imdb_ori -c "SELECT 1"
```

**Common Issues:**
- Server won't start: CUDA multiprocessing issue (fixed with BAO_FORCE_CPU=1)
- Model not found: Need to train first
- Path errors: Run from `benchmarks/lqos/bao/` directory

### Adding New Query Sets

1. Create query directories:
   ```
   queries/YOUR_QUERY_SET/
   ├── train/  # Training queries (.sql files)
   └── test/   # Testing queries (.sql files)
   ```

2. Run pipeline:
   ```bash
   ./schedule_bao_simple.sh YOUR_QUERY_SET
   ```

