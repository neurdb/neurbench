#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

cd /app/bao/bao_server
rm bao.db
pkill -f "python3 -u main.py"

cd /app/bao/bao_server
nohup python3 -u main.py > ./run_query_shift_join_03.txt 2>&1 &

# job 1
cd /app/bao
python3 run_queries.py --query_dir /data/datasets/imdb_queries/q_train_imdb_join_03 --output_file log_vldb_query_shifts/train__bao__query_shift_imdb_join_03.txt --database_name imdb_ori
python3 run_test_queries.py --use_bao --query_dir /data/datasets/imdb_queries/q_test_sample_20% --output_file log_vldb_query_shifts/test__bao__query_shift_imdb_join_03.txt --database_name imdb_ori
cd /app/bao/bao_server
mv bao_default_model bao_default_model_query_shift_join_03
mv bao_previous_model bao_previous_model_query_shift_join_03
rm bao.db
# Stop the previous bao server
pkill -f "python3 -u main.py"


cd /app/bao/bao_server
nohup python3 -u main.py > ./run_query_shift_join_05.txt 2>&1 &

# job 1
cd /app/bao
python3 run_queries.py --query_dir /data/datasets/imdb_queries/q_train_imdb_join_05 --output_file log_vldb_query_shifts/train__bao__query_shift_imdb_join_05.txt --database_name imdb_ori
python3 run_test_queries.py --use_bao --query_dir /data/datasets/imdb_queries/q_test_sample_20% --output_file log_vldb_query_shifts/test__bao__query_shift_imdb_join_05.txt --database_name imdb_ori
cd /app/bao/bao_server
mv bao_default_model bao_default_model_query_shift_join_05
mv bao_previous_model bao_previous_model_query_shift_join_05
rm bao.db
# Stop the previous bao server
pkill -f "python3 -u main.py"


cd /app/bao/bao_server
nohup python3 -u main.py > ./run_query_shift_join_07.txt 2>&1 &

# job 1
cd /app/bao
python3 run_queries.py --query_dir /data/datasets/imdb_queries/q_train_imdb_join_07 --output_file log_vldb_query_shifts/train__bao__query_shift_imdb_join_07.txt --database_name imdb_ori
python3 run_test_queries.py --use_bao --query_dir /data/datasets/imdb_queries/q_test_sample_20% --output_file log_vldb_query_shifts/test__bao__query_shift_imdb_join_07.txt --database_name imdb_ori
cd /app/bao/bao_server
mv bao_default_model bao_default_model_query_shift_join_07
mv bao_previous_model bao_previous_model_query_shift_join_07
rm bao.db
# Stop the previous bao server
pkill -f "python3 -u main.py"

