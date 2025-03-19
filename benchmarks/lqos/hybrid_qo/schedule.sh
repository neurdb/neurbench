

python run_mcts.py --query_file workload/query_join__train.json
mv model model_join
mv logs logs_join

mkdir logs
mkdir model
python run_mcts.py --query_file workload/query_predicates__train.json
mv model model_predicates
mv logs logs_predicates


mkdir logs
mkdir model
python run_mcts.py --query_file workload/query_tables__train.json
mv model model_tables
mv logs logs_tables

