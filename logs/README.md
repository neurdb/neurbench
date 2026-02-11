# Log Files

This folder contains the logs. For illustration purpose, we proviede the log file mapping for Figure 10(a).

### ORIG (Original IMDB Dataset)

| System     | Log File                                        | Total Time |
|------------|-------------------------------------------------|------------|
| PostgreSQL | `test_pg_job_20260113_150750.txt`               | 134.44s    |
| Bao        | `test_bao_job_20260113_174743.txt`              | 110.79s    |
| Balsa      | `20260117_011241_test_balsa_imdb_job.txt`       | 110.64s    |
| Lero       | `20260116_025828_test_lero_output_imdb_job.txt` | 132.63s    |

### d=0.1 (Drift Factor 0.1)

| System     | Log File                                                 | Total Time |
|------------|----------------------------------------------------------|------------|
| PostgreSQL | `test_pg_job_20260115_030411.txt`                        | 213.11s    |
| Bao        | `test_bao_job_20260114_182558.txt`                       | 213.23s    |
| Balsa      | `20260117_065737_test_balsa_imdb_1_gen_job.txt`          | 252.76s    |
| Lero       | `20260115_101952_test_lero_output_imdb_1_gen_job.txt`    | 210.66s    |

### d=0.3 (Drift Factor 0.3)

| System     | Log File                                              | Total Time |
|------------|-------------------------------------------------------|------------|
| PostgreSQL | `test_pg_job_20260116_174214.txt`                     | 237.22s    |
| Bao        | `test_bao_job_20260116_174627.txt`                    | 256.28s    |
| Balsa      | `20260117_063644_test_balsa_imdb_3_gen_job.txt`       | 275.18s    |
| Lero       | `20260117_031044_test_lero_output_imdb_3_gen_job.txt` | 251.33s    |

### d=0.5 (Drift Factor 0.5)

| System     | Log File                                                  | Total Time |
|------------|-----------------------------------------------------------|------------|
| PostgreSQL | `test_pg_job_20260115_152936.txt`                         | 308.38s    |
| Bao        | `test_bao_job_20260114_183755.txt`                        | 353.38s    |
| Balsa      | `20260117_093539_test_balsa_imdb_5_gen_job.txt`           | 461.69s    |
| Lero       | `20260117_023547_test_lero_output_imdb_5_gen_job.txt`     | 325.55s    |


### Bao Training Query Log Format
```
x, index, iteration, timestamp, query_file, planning_time_ms, execution_time_ms, optimizer_type
```

## Log File Formats

### PostgreSQL & Bao (CSV Format)
```
query_id,pg_latency,bao_latency,pg_plan_time,bao_plan_time,execution_time
```
- Column 6 (`execution_time`) contains the execution time in milliseconds

### Balsa (Semicolon-Delimited)
```
query_id;inference_time;planning_time;execution_time;col5;col6
```
- Column 4 (`execution_time`) contains the execution time in milliseconds

### Lero (Log Format)
```
after writting write_latency_file <timestamp> <query_id> <execution_time>
```
- The last field contains the execution time in milliseconds
