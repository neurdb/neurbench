cd /app/bao/bao_server
nohup python3 -u main.py > ./run_shift_join_03.txt 2>&1 &

# job 1
cd /app/bao
python3 run_queries.py --query_dir /data/datasets/q_train_join_03 --output_file log_sigmod/train__bao__base_query_join_03.txt --database_name tpch
python3 run_test_queries.py --use_bao --query_dir /data/datasets/q_test_0 --output_file log_sigmod/test__bao__query_join_03.txt --database_name tpch
cd /app/bao/bao_server
mv bao_default_model bao_default_model_query_shift_join_03
mv bao_previous_model bao_previous_model_query_shift_join_03
# Stop the previous bao server
pkill -f "python3 -u main.py"


cd /app/bao/bao_server
nohup python3 -u main.py > ./run_predicates_03.txt 2>&1 &

# job 2
cd /app/bao
python3 run_queries.py --query_dir /data/datasets/q_train_predicates_03 --output_file log_sigmod/train__bao__base_query_predicates_03.txt --database_name tpch
python3 run_test_queries.py --use_bao --query_dir /data/datasets/q_test_0 --output_file log_sigmod/test__bao__query_predicates_03.txt --database_name tpch
cd /app/bao/bao_server
mv bao_default_model bao_default_model_query_shift_predicates_03
mv bao_previous_model bao_previous_model_query_shift_predicates_03
# Stop the previous bao server
pkill -f "python3 -u main.py"


cd /app/bao/bao_server
nohup python3 -u main.py > ./run_join_07.txt 2>&1 &

# job 3
cd /app/bao
python3 run_queries.py --query_dir /data/datasets/q_train_join_07 --output_file log_sigmod/train__bao__base_query_join_07.txt --database_name tpch
python3 run_test_queries.py --use_bao --query_dir /data/datasets/q_test_0 --output_file log_sigmod/test__bao__query_join_07.txt --database_name tpch
cd /app/bao/bao_server
mv bao_default_model bao_default_model_query_shift_join_07
mv bao_previous_model bao_previous_model_query_shift_join_07

# Stop the previous bao server
pkill -f "python3 -u main.py"

cd /app/bao/bao_server
nohup python3 -u main.py > ./run_predicates_07.txt 2>&1 &

# job 4
cd /app/bao
python3 run_queries.py --query_dir /data/datasets/q_train_predicates_07 --output_file log_sigmod/train__bao__base_query_predicates_07.txt --database_name tpch
python3 run_test_queries.py --use_bao --query_dir /data/datasets/q_test_0 --output_file log_sigmod/test__bao__query_predicates_07.txt --database_name tpch
cd /app/bao/bao_server
mv bao_default_model bao_default_model_query_shift_predicates_07
mv bao_previous_model bao_previous_model_query_shift_predicates_07
