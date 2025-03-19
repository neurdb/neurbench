#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

cd /app/bao/bao_server
nohup python3 -u main.py > ./run_shift_data_01v2.txt 2>&1 &

# job 1
cd /app/bao
python3 run_queries.py --query_dir /data/datasets/imdb_v2/query/q_train_sample_80% --output_file log_vldb_data_shifts/train__bao__shift_data_01v2.txt --database_name imdb_ori
python3 run_test_queries.py --use_bao --query_dir /data/datasets/imdb_v2/query/q_test_sample_20% --output_file log_vldb_data_shifts/test__bao__shift_data_01v2.txt --database_name imdb_01v2
cd /app/bao/bao_server
mv bao_default_model bao_default_model_shift_data_01v2
mv bao_previous_model bao_previous_model_shift_data_01v2
rm bao.db
# Stop the previous bao server
pkill -f "python3 -u main.py"


cd /app/bao/bao_server
nohup python3 -u main.py > ./run_shift_data_05v2.txt 2>&1 &

# job 1
cd /app/bao
python3 run_queries.py --query_dir /data/datasets/imdb_v2/query/q_train_sample_80% --output_file log_vldb_data_shifts/train__bao__shift_data_05v2.txt --database_name imdb_ori
python3 run_test_queries.py --use_bao --query_dir /data/datasets/imdb_v2/query/q_test_sample_20% --output_file log_vldb_data_shifts/test__bao__shift_data_05v2.txt --database_name imdb_05v2
cd /app/bao/bao_server
mv bao_default_model bao_default_model_shift_data_05v2
mv bao_previous_model bao_previous_model_shift_data_05v2
rm bao.db
# Stop the previous bao server
pkill -f "python3 -u main.py"


cd /app/bao/bao_server
nohup python3 -u main.py > ./run_shift_data_07v2.txt 2>&1 &

# job 1
cd /app/bao
python3 run_queries.py --query_dir /data/datasets/imdb_v2/query/q_train_sample_80% --output_file log_vldb_data_shifts/train__bao__shift_data_07v2.txt --database_name imdb_ori
python3 run_test_queries.py --use_bao --query_dir /data/datasets/imdb_v2/query/q_test_sample_20% --output_file log_vldb_data_shifts/test__bao__shift_data_07v2.txt --database_name imdb_07v2
cd /app/bao/bao_server
mv bao_default_model bao_default_model_shift_data_07v2
mv bao_previous_model bao_previous_model_shift_data_07v2
rm bao.db
# Stop the previous bao server
pkill -f "python3 -u main.py"


cd /app/bao/bao_server
nohup python3 -u main.py > ./run_shift_data_imdb_ori.txt 2>&1 &

# job 1
cd /app/bao
python3 run_queries.py --query_dir /data/datasets/imdb_v2/query/q_train_sample_80% --output_file log_vldb_data_shifts/train__bao__shift_data_imdb_ori.txt --database_name imdb_ori
python3 run_test_queries.py --use_bao --query_dir /data/datasets/imdb_v2/query/q_test_sample_20% --output_file log_vldb_data_shifts/test__bao__shift_data_imdb_ori.txt --database_name imdb_ori
cd /app/bao/bao_server
mv bao_default_model bao_default_model_shift_data_imdb_ori
mv bao_previous_model bao_previous_model_shift_data_imdb_ori
rm bao.db
# Stop the previous bao server
pkill -f "python3 -u main.py"