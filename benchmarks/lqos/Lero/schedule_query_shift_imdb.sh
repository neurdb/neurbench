

# Query shift join 01
pkill -f "python -u server.py"
sleep 2
cd /code/AI4QueryOptimizer/baseline/Lero/lero
nohup python -u server.py > ./run_server_log_01.txt 2>&1 &
sleep 2
cd /code/AI4QueryOptimizer/baseline/Lero/lero/test_script
python train_model.py --query_path imdb_queries/q_train_queryshift_join_01.txt --test_query_path imdb_queries/q_test_small_set.txt --algo lero --query_num_per_chunk 20 --output_query_latency_file lero_imdb_ori_train_queryshift_01.log --model_prefix imdb_ori_query_shift_model_01 --topK 3

# Query shift join 03
pkill -f "python -u server.py"
sleep 2
cd /code/AI4QueryOptimizer/baseline/Lero/lero
nohup python -u server.py > ./run_server_log_03.txt 2>&1 &
sleep 2
cd /code/AI4QueryOptimizer/baseline/Lero/lero/test_script
python train_model.py --query_path imdb_queries/q_train_queryshift_join_03.txt --test_query_path imdb_queries/q_test_small_set.txt --algo lero --query_num_per_chunk 20 --output_query_latency_file lero_imdb_ori_train_queryshift_03.log --model_prefix imdb_ori_query_shift_model_03 --topK 3

# Query shift join 05
pkill -f "python -u server.py"
sleep 2
cd /code/AI4QueryOptimizer/baseline/Lero/lero
nohup python -u server.py > ./run_server_log_05.txt 2>&1 &
sleep 2
cd /code/AI4QueryOptimizer/baseline/Lero/lero/test_script
python train_model.py --query_path imdb_queries/q_train_queryshift_join_05.txt --test_query_path imdb_queries/q_test_small_set.txt --algo lero --query_num_per_chunk 20 --output_query_latency_file lero_imdb_ori_train_queryshift_05.log --model_prefix imdb_ori_query_shift_model_05 --topK 3

