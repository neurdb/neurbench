mkdir logs && \
mkdir model && \
python run_mcts.py --query_file workload/query_sample_join_03__train.json --train_database imdb_ori --test_database imdb_ori && \
mv model model_imdb_query_shift_01 && \
mv logs logs_imdb_query_shift_01 && \


mkdir logs && \
mkdir model && \
python run_mcts.py --query_file workload/query_sample_join_05__train.json --train_database imdb_ori --test_database imdb_ori && \
mv model model_imdb_query_shift_05 && \
mv logs logs_imdb_query_shift_05

