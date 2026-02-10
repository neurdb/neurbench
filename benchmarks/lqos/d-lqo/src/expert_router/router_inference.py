from expert_router.controller_offline import ModelBuilder
from expert_router.logger import plogger
from collections import Counter
from parser.table_parser import load_db_info_json
from expert_router.encoder import Sql2VecEmbeddingV2
from expert_router.dataset import get_data_loader
import time
from data_collector.router_dataset_custome import load_workload_train_test_datasets
import json
import pandas as pd
import os
from utils.io import set_global_seed
from common import get_config, BaseConfig
from typing import Tuple


def map_back_to_method(predictions: dict, cfg: BaseConfig) -> dict:
    id_to_method = {idx: method for method, idx in cfg.FIXED_LABEL_MAPPING.items()}
    predicted_methods = {
        query: id_to_method[pred] for query, pred in predictions.items()
    }
    return predicted_methods


def compute_total_time_for_predictions(test_df, query_method: dict, workload_name: str,
                                       avg_inference_time_per_query_cpu, cfg: BaseConfig) -> Tuple[float, dict]:
    """
    Compute total time from test_df directly, using execution_time_ms from the test data.
    
    Args:
        test_df: Test DataFrame with columns: query_ident, execution_time_ms, etc.
        query_method: Dictionary mapping query_id to predicted method name
        workload_name: Name of the workload
        avg_inference_time_per_query_cpu: Average inference time per query
        cfg: Configuration object with FIXED_LABEL_MAPPING
    """
    total_time_sum = 0.0
    per_query_time = {}
    
    # Create a mapping from query_ident to row for faster lookup
    test_df_indexed = test_df.set_index('query_ident')
    
    for query_id, predicted_method in query_method.items():
        if query_id not in test_df_indexed.index:
            print(f"Warning: Query {query_id} not found in test data")
            continue
            
        row = test_df_indexed.loc[query_id]
        execution_time_ms = row['execution_time_ms']
        
        # Get execution time for the predicted method
        if isinstance(execution_time_ms, dict):
            execution_time = execution_time_ms.get(predicted_method)
        elif isinstance(execution_time_ms, (int, float)):
            # If it's already a number, use it directly
            execution_time = execution_time_ms
        else:
            raise
        
        # For inference, we only have execution_time from test data
        # We add the inference time to get total prepare time
        prepare_time = avg_inference_time_per_query_cpu
        total_time = prepare_time + execution_time
        
        total_time_sum += total_time
        per_query_time[query_id] = {
            "prepare_time": prepare_time,
            "execution_time": execution_time
        }

    print(f"Total time for workload '{workload_name}': {total_time_sum}")
    return total_time_sum, per_query_time

def get_pg_sum_res(test_df):
    """
    Calculate the sum of PostgreSQL execution times from test_df.
    execution_time_ms is already parsed as dict by json.loads() during data loading.
    """
    # execution_time_ms is already a dict (parsed by json.loads in load_workload_train_test_datasets)
    # So we can directly access the 'pg' key
    pg_sum = 0.0
    for idx, row in test_df.iterrows():
        execution_time_ms = row['execution_time_ms']
        if isinstance(execution_time_ms, dict):
            pg_time = execution_time_ms.get('pg')
            pg_sum += pg_time
        elif isinstance(execution_time_ms, str):
            exec_dict = json.loads(execution_time_ms)
            pg_sum += exec_dict.get('pg')
   
    return pg_sum


def inference_single_workload(
        model_path, train_test_data_path: str, batch_size: int, dataset: str,
        cfg: BaseConfig):
    SysOnworkloadCollector = {}
    query_per_workload = {}

    # from arg_utils import shared_args
    # from dataset.data_process import DataProcessor
    # _args = shared_args.get_stats_arg_parser()
    # _, all_paths = shared_args.init_system_dirs(_args)
    # data_processor = DataProcessor(_args)
    # data_processor.dataset_profiling(all_paths)

    # feature instance
    db_profile_res = load_db_info_json(cfg.DB_INFO_DICT)
    sql_vec = Sql2VecEmbeddingV2(db_profile_res=db_profile_res,
                                 checkpoint_file=cfg.EMBED_FILE)

    fq_instance = sql_vec

    # Use the same data loading function as training
    datasets = load_workload_train_test_datasets(folder_name=train_test_data_path)
    sorted_exps = sorted(datasets.keys())
    print(sorted_exps)
    
    for workload_name in sorted_exps:
        # Extract test data from the same structure as training
        test_df = datasets[workload_name]['test']
        print(test_df)
        data_loaders, input_dim, output_dim = get_data_loader(
            cfg=cfg,
            datasets={"test": test_df},
            batch_size=batch_size,
            fq_instance=fq_instance,
            threshold=None,
            workload_name=workload_name
        )

        builder = ModelBuilder(
            num_tables=len(set(db_profile_res.table_no_map.values())),
            num_columns=108,
            output_dim=output_dim,
            model_path_prefix=f"{model_path}/{workload_name}_model",
            embedding_path=None,  # load the whole model
            num_heads=4,
            embedding_dim=256,
            is_fix_emb=None,
            num_layers=2,
            dataset=dataset,
            cfg=cfg

        )

        builder.load_model("hypered_2")
        start_time = time.time()  # Start timing
        predictions_dict, predictions_time = builder.inference(data_loaders['test'], save_embedding=True)
        end_time = time.time()  # End timing

        total_inference_time = end_time - start_time
        avg_inference_time_per_query_cpu = total_inference_time / len(test_df)

        print("avg_inference_time_per_query", avg_inference_time_per_query_cpu)
        print(predictions_time)
        print(predictions_dict)
        query_method = map_back_to_method(predictions_dict, cfg=cfg)
        print(query_method)
        
        # Save query_method to CSV
        query_method_df = pd.DataFrame(list(query_method.items()), columns=['query_ident', 'predicted_method'])
        output_dir = "./experiment_result/result_data"
        os.makedirs(output_dir, exist_ok=True)
        csv_filename = os.path.join(output_dir, f"query_method_{workload_name}_{dataset}_{cfg.DB_NAME}.csv")
        query_method_df.to_csv(csv_filename, index=False)
        print(f"Saved query_method to {csv_filename}")
        
        pg_sum_res = get_pg_sum_res(test_df)

        total_time_sum, per_query_time = compute_total_time_for_predictions(
            test_df, query_method, workload_name, avg_inference_time_per_query_cpu, cfg)

        # Calculate model execution time only (without inference overhead)
        model_execution_time_sum = sum([per_query_time[qid]["execution_time"] for qid in per_query_time.keys()])
        # Calculate improvement percentage over PostgreSQL
        improvement = ((pg_sum_res - model_execution_time_sum) / pg_sum_res * 100) if pg_sum_res > 0 else 0.0
        print(f"Improvement over PostgreSQL: {improvement:.2f}%")

        # single query
        query_per_workload[workload_name] = per_query_time

        # sum query
        SysOnworkloadCollector[workload_name] = total_time_sum + total_inference_time

        plogger.info(f"{workload_name}: {dict(Counter(query_method.values()))}")
        plogger.info(f"{workload_name}: {total_time_sum}")
        plogger.info(f"\n")

    with open(f"{cfg.RESULT_DATA_BASE}/inference_res_{dataset}_{cfg.DB_NAME}_for_figure", "w") as f:
        json.dump(query_per_workload, f)
    print("inference_single_workload", SysOnworkloadCollector)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Figure")
    parser.add_argument("--dataset", help="imdb or stack")
    args = parser.parse_args()

    cfg = get_config(args.dataset)

    set_global_seed(2550)

    inference_single_workload(
        model_path=f"./experiment_result/models/{cfg.DB_NAME}",
        train_test_data_path=cfg.TRAIN_TEST,  # Use the same path as training
        batch_size=160,
        dataset=args.dataset,
        cfg=cfg
    )
