# Bao Log Format

## Log Format

### Test Logs (bao_test_under_non_drift.log, pg_test.log)
Format: `hint, iteration, timestamp, filename, planning_time, execution_time, Bao/PG`

- 7 columns total
- **Execution time**: Column -2 (2nd from end)
- Planning time: Column -3 (3rd from end)

### Train Logs (bao_train_under_non_drift.log)
Format: `chunk_idx, query_idx, iteration, timestamp, filename, planning_time, execution_time, Bao/PG`

- 8 columns total
- **Execution time**: Column -2 (2nd from end)
- Planning time: Column -3 (3rd from end)

All times are in milliseconds (ms).
