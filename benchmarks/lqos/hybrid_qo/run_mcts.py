import random

random.seed(113)

from datetime import datetime
import json
from tqdm.auto import tqdm

from sql2fea import TreeBuilder, value_extractor
from NET import TreeNet
from sql2fea import Sql2Vec
from TreeLSTM import SPINN

import torch
import os
import pickle


def neur_bench_save_hinter(hinter, config, train_or_test, epoch, save_dir="model"):
    os.makedirs(save_dir, exist_ok=True)
    """Save the Hinter and its components to disk."""
    if config.n_epochs != epoch:
        file_prefix = f"{config.train_database}_{epoch}"
        checkpoint_path = os.path.join(save_dir, f"hinter_{file_prefix}.pt")
        knn_path = os.path.join(save_dir, f"knn_{file_prefix}.pkl")
    else:
        file_prefix = f"{config.train_database}"
        checkpoint_path = os.path.join(save_dir, f"hinter_{file_prefix}.pt")
        knn_path = os.path.join(save_dir, f"knn_{file_prefix}.pkl")

    # Save PyTorch model state dictionaries and config
    from mcts import predictionNet  # Import global predictionNet
    checkpoint = {
        'tree_net_state_dict': hinter.model.value_network.state_dict(),  # Save SPINN parameters
        'mcts_searcher_state_dict': predictionNet.state_dict(),  # Save predictionNet parameters
    }

    torch.save(checkpoint, checkpoint_path)

    # Save KNN data
    with open(knn_path, 'wb') as f:
        pickle.dump(hinter.knn, f)

    print(f"Saved checkpoint to {checkpoint_path}, KNN to {knn_path}")
    return checkpoint_path, knn_path


def neur_bench_load_hinter(config, train_or_test):
    """Load the Hinter and its components from disk."""
    save_dir = "model"
    file_prefix = f"{config.train_database}"
    checkpoint_path = os.path.join(save_dir, f"hinter_{file_prefix}.pt")
    knn_path = os.path.join(save_dir, f"knn_{file_prefix}.pkl")

    if not os.path.exists(checkpoint_path) or not os.path.exists(knn_path):
        raise FileNotFoundError(f"Checkpoint {checkpoint_path} or KNN {knn_path} not found")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=config.device)

    # Initialize components
    tree_builder = TreeBuilder()
    sql2vec = Sql2Vec()
    value_network = SPINN(
        head_num=config.head_num,
        input_size=config.input_size,
        hidden_size=config.hidden_size,
        table_num=50,
        sql_size=config.max_alias_num * config.max_alias_num + config.max_column
    ).to(config.device)
    net = TreeNet(tree_builder=tree_builder, value_network=value_network)
    mcts_searcher = MCTSHinterSearch()

    # Load model state dictionaries
    net.value_network.load_state_dict(checkpoint['tree_net_state_dict'])  # Load SPINN parameters
    from mcts import predictionNet  # Import global predictionNet
    print("Now load the predictionNet saved parameter")
    predictionNet.load_state_dict(checkpoint['mcts_searcher_state_dict'])  # Load predictionNet parameters

    # Load KNN
    with open(knn_path, 'rb') as f:
        knn = pickle.load(f)

    # Create Hinter
    hinter = Hinter(
        model=net,
        sql2vec=sql2vec,
        value_extractor=value_extractor,  # Use global value_extractor
        mcts_searcher=mcts_searcher
    )
    hinter.knn = knn

    print(f"Loaded checkpoint from {checkpoint_path}, KNN from {knn_path}")
    return hinter


def load_queries(queries_path):
    with open(queries_path) as f:
        queries = json.load(f)
    return queries


def main(config, train_or_test):
    run_name = datetime.now().strftime('%Y_%m_%d__%H%M%S')
    # wandb.init(
    #     project='hybrid_qo',
    #     entity='FILL_IN_YOUR_WANDB_ENTITY_HERE',
    #     name=run_name,
    #     config=config.__dict__
    # )
    print("---train query files", config.queries_file)
    train_queries = load_queries(config.queries_file)
    print("---test query files", config.queries_file.replace('__train', '__test'))
    test_queries = load_queries(config.queries_file.replace('__train', '__test'))

    if train_or_test == "train":
        tree_builder = TreeBuilder()
        sql2vec = Sql2Vec()
        # table_num = config.max_alias_num? or actually number of tables?
        value_network = SPINN(head_num=config.head_num, input_size=config.input_size, hidden_size=config.hidden_size,
                              table_num=50,
                              sql_size=config.max_alias_num * config.max_alias_num + config.max_column).to(
            config.device)
        for name, param in value_network.named_parameters():
            from torch.nn import init
            if len(param.shape) == 2:
                init.xavier_normal_(param)
            else:
                init.uniform_(param)

        net = TreeNet(tree_builder=tree_builder, value_network=value_network)
        mcts_searcher = MCTSHinterSearch()
        hinter = Hinter(model=net, sql2vec=sql2vec, value_extractor=value_extractor, mcts_searcher=mcts_searcher)

        print(len(train_queries))

        # Prepare query log file
        query_log_file_path \
            = f"logs/{run_name}__query_log_{config.train_database}_{config.test_database}_{train_or_test}.csv"

        columns = ['epoch', 'test_query', 'query_ident', 'pg_plan_time', 'pg_latency', 'mcts_time', 'hinter_plan_time',
                   'MPHE_time', 'hinter_latency', 'hinter_query_ratio', 'mse', 'variance']
        with open(query_log_file_path, 'w') as f:
            f.write(','.join(columns) + '\n')

        # Since the splits provided by Lehmann, Sulimov & Stockinger do not include 20'000 queries,
        # we instead run the approx. 80-90 queries in repeated epochs to achieve a roughly similar
        # amount of executed queries, though these include the same queries multiple times!

        print("training ...")
        for epoch in range(config.n_epochs):
            train_epoch(hinter, train_queries, epoch, query_log_file_path)
            print(f"Epoch {epoch + 1}/{config.n_epochs} completed")

            if epoch % 10 == 0:
                checkpoint_path, knn_path = neur_bench_save_hinter(hinter, config, train_or_test, epoch)
                print(f"Epoch Done {epoch}, Saved model to {checkpoint_path}, KNN to {knn_path}")

        # Save the trained hinter
        checkpoint_path, knn_path = neur_bench_save_hinter(hinter, config, train_or_test, config.n_epochs)
        print(f"Training complete. Saved model to {checkpoint_path}, KNN to {knn_path}")

        # this is to verify only!!!!
        # print("testing after training ...")
        # test_epoch(hinter, test_queries, 0, query_log_file_path)
        # print("Testing after Training complete.")

    # Final eval
    if train_or_test == "test":
        # Load the trained hinter
        hinter = neur_bench_load_hinter(config, train_or_test)

        # Prepare query log file
        query_log_file_path \
            = f"logs/{run_name}__query_log_{config.train_database}_{config.test_database}_{train_or_test}.csv"

        columns = ['epoch', 'test_query', 'query_ident', 'pg_plan_time', 'pg_latency', 'mcts_time', 'hinter_plan_time',
                   'MPHE_time', 'hinter_latency', 'hinter_query_ratio', 'loss', 'real_mse', "predicted_time"]
        with open(query_log_file_path, 'w') as f:
            f.write(','.join(columns) + '\n')

        print("testing ...")
        test_epoch(hinter, test_queries, 0, query_log_file_path)
        print("Testing complete.")


def train_epoch(hinter, queries, epoch, query_log_file_path):
    s_pg = 0
    s_hinter = 0

    # Because of multiple epochs, make sure that queries are randomly shuffled at each time
    random.shuffle(queries)

    # pbar = tqdm(enumerate(queries[:]), total=len(queries), leave=False, desc='Iterating over training queries...')
    # for idx, (sql, query_ident, _) in pbar:
    #     pbar.set_description(f"Iterating over training query {query_ident}...")
    for idx, (sql, query_ident, _) in enumerate(queries[:]):
        print(f"Processing training query {query_ident} ({idx + 1}/{len(queries)})")
        try:
            pg_plan_time, pg_latency, mcts_time, hinter_plan_time, MPHE_time, hinter_latency, actual_plans, actual_time, mse, real_mse, predicted_time = hinter.hinterRun(
                sql, is_train=True)
            pg_latency /= 1000
            hinter_latency /= 1000
            pg_plan_time /= 1000
            hinter_plan_time /= 1000

            s_pg += pg_latency
            s_hinter += sum(actual_time) / 1000

            # wandb.log({
            #     'epoch': epoch,
            #     'pg_plan_time': pg_plan_time,
            #     'pg_lateny': pg_latency,
            #     'mcts_time': mcts_time,
            #     'hinter_plan_time': hinter_plan_time,
            #     'MPHE_time': MPHE_time,
            #     'hinter_latency': hinter_latency,
            #     'hinter_global_ratio': s_hinter / s_pg,
            #     'hinter_query_ratio': pg_latency / (sum(actual_time) / 1000),
            #     'query_ident': query_ident,
            #     'test_query': 0
            # })
            print(f"Training query {query_ident} takes {actual_time}")

            with open(query_log_file_path, 'a') as f:
                f.write(
                    f"{epoch},0,{query_ident},{pg_plan_time},{pg_latency},{mcts_time},{hinter_plan_time},{MPHE_time},{hinter_latency},{pg_latency / (sum(actual_time) / 1000)},{mse},{real_mse.item() if isinstance(real_mse, torch.Tensor) else real_mse}, {predicted_time}\n")
        except Exception as e:
            print(f"[Error] when running query {query_ident}, error: {e}")


def test_epoch(hinter, queries, epoch, query_log_file_path):
    s_pg = 0
    s_hinter = 0

    for idx, (sql, query_ident, _) in enumerate(queries[:]):
        try:
            print(f"Processing test query {query_ident} ({idx + 1}/{len(queries)})")

            pg_plan_time, pg_latency, mcts_time, hinter_plan_time, MPHE_time, hinter_latency, actual_plans, actual_time, mse, real_mse, predicted_time = hinter.hinterRun(
                sql, is_train=False)
            pg_latency /= 1000
            hinter_latency /= 1000
            pg_plan_time /= 1000
            hinter_plan_time /= 1000

            s_pg += pg_latency
            s_hinter += sum(actual_time) / 1000

            with open(query_log_file_path, 'a') as f:
                f.write(
                    f"{epoch},1,{query_ident},{pg_plan_time},{pg_latency},{mcts_time},{hinter_plan_time},{MPHE_time},{hinter_latency},{pg_latency / (sum(actual_time) / 1000)},{mse},{real_mse}, {predicted_time}\n")
        except Exception as e:
            print(f"[Error] when running test query {query_ident}, error: {e}")


if __name__ == '__main__':
    from ImportantConfig import Config
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, required=False, default=0, help="epoch to train")
    parser.add_argument('--train_test', type=str, required=True, help="is train or test")
    parser.add_argument('--query_file', type=str, required=True, help="Path to the queries file")
    parser.add_argument('--train_database', type=str, required=True, default="", help="DB anme")
    parser.add_argument('--test_database', type=str, required=True, default="", help="DB anme")

    args = parser.parse_args()
    config = Config()
    Config.train_database = args.train_database
    Config.test_database = args.test_database

    config.queries_file = args.query_file
    config.n_epochs = args.epoch
    config.train_database = args.train_database
    config.test_database = args.test_database
    print("Updating config done")

    from Hinter import Hinter
    from mcts import MCTSHinterSearch

    # Run the main function
    print(config.__dict__)
    main(config, args.train_test)
