mkdir logs && \
mkdir model && \
python run_mcts.py --query_file workload/query_sample__train.json --train_database imdb_ori --test_database imdb_01v2 && \
mv model model_imdb_01v2 && \
mv logs logs_imdb_01v2 && \


mkdir logs && \
mkdir model && \
python run_mcts.py --query_file workload/query_sample__train.json --train_database imdb_ori --test_database imdb_05v2 && \
mv model model_imdb_05v2 && \
mv logs logs_imdb_05v2 && \


mkdir logs && \
mkdir model && \
python run_mcts.py --query_file workload/query_sample__train.json --train_database imdb_ori --test_database imdb_07v2 && \
mv model model_imdb_07v2 && \
mv logs logs_imdb_07v2
#
#
#mkdir logs && \
#mkdir model && \
#python run_mcts.py --query_file workload/query_sample__train.json --train_database imdb_ori --test_database imdb_ori && \
#mv model model_imdb_ori && \
#mv logs logs_imdb_ori
#
