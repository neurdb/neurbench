

# Query shift join 01
pkill -f "python -u server.py"
sleep 2
cd /code/AI4QueryOptimizer/baseline/Lero/lero
nohup python -u server.py > ./run_server_log_predict_01.txt 2>&1 &
sleep 2
cd /code/AI4QueryOptimizer/baseline/Lero/lero/test_script
python train_model.py --query_path imdb_queries/q_train_queryshift_predicates_01.txt --test_query_path imdb_queries/q_test_small_set.txt --algo lero --query_num_per_chunk 20 --output_query_latency_file lero_imdb_ori_train_queryshift_predict_01.log --model_prefix imdb_ori_query_shift_model_predict_01 --topK 3


pkill -f "python -u server.py"
sleep 2
cd /code/AI4QueryOptimizer/baseline/Lero/lero
nohup python -u server.py > ./run_server_log_table_01.txt 2>&1 &
sleep 2
cd /code/AI4QueryOptimizer/baseline/Lero/lero/test_script
python train_model.py --query_path imdb_queries/q_train_queryshift_table_01.txt --test_query_path imdb_queries/q_test_small_set.txt --algo lero --query_num_per_chunk 20 --output_query_latency_file lero_imdb_ori_train_queryshift_table_01.log --model_prefix imdb_ori_query_shift_model_table_01 --topK 3

