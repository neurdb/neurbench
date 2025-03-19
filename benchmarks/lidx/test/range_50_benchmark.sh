#!/bin/bash

# Define base directories
data_dir="/users/lingze/TLI/data/imdb_4M_uint64"
workload_base="/users/lingze/neurbench/data/workload"
benchmark_exec="./build/benchmark"

# Define drift variants
drift_variants=(
    "imdb_4M_uint64_drift_00"
    "imdb_4M_uint64_drift_01"
    "imdb_4M_uint64_drift_03"
    "imdb_4M_uint64_drift_05"
)

# Define index type
index_type="$1"

# Run benchmark for each drift variant
for drift in "${drift_variants[@]}"; do
    workload_file="$workload_base/$drift/workload_ops_10M_0.500000rq_50rs_0.000000nl_0.500000i_2m"
    echo "Running benchmark for $drift with index $index_type..."
    $benchmark_exec "$data_dir" "$workload_file" --only "$index_type" --through
done
