import pandas as pd
from data_collector.router_dataset import load_workload_train_test_datasets, load_and_process_data
from common import get_config, BaseConfig
from utils.io import set_global_seed
import json
import os


def get_max_query_set(easy_df, mid_df, hard_df, scale_number):
    a = easy_df.shape[0]
    b = mid_df.shape[0]
    c = hard_df.shape[0]

    print(f"ori line numbers: easy={a}, mid={b}, hard={c}")

    target_len = max(a, b, c) * scale_number

    easy_mul = round(target_len / a)
    mid_mul = round(target_len / b)
    hard_mul = round(target_len / c)

    easy_final = a * easy_mul
    mid_final = b * mid_mul
    hard_final = c * hard_mul

    print(f"TARGET_QUERY_COUNT: {target_len}")
    print(f"easy_df * {easy_mul} → {easy_final}")
    print(f"mid_df * {mid_mul} → {mid_final}")
    print(f"hard_df * {hard_mul} → {hard_final}")
    return target_len


def compute_needed(num_selected, target_ratio):
    return int(num_selected * ((1 - target_ratio) / target_ratio))


def combine_train_test_dfs(cfg):
    datasets = load_workload_train_test_datasets(cfg.TRAIN_TEST)
    result = {}
    for exp in cfg.TEST_QUERIES.keys():
        fixed_train_df = datasets[exp]['train'].copy()
        train_queries = sorted(list(set(fixed_train_df['query_ident'].tolist())))
        print(f"Fixed training set from '{exp}': {len(fixed_train_df)} queries")

        df = datasets[exp]['test'][
            ~datasets[exp]['test']["query_ident"].isin(fixed_train_df["query_ident"])
        ].copy()
        df["source_exp"] = exp

        # verify that the id is not exist in the train query set.
        cur_test_query_set = sorted(list(set(df['query_ident'].tolist())))
        for query_id in cur_test_query_set:
            if query_id in train_queries:
                raise
        result[exp] = df

    all_dfs = [df for df in result.values()]
    max_length = get_max_query_set(all_dfs[0], all_dfs[1], all_dfs[2], 150)
    for key, value in result.items():
        result[key] = pd.concat([value] * round(max_length / value.shape[0]), ignore_index=True)

    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("Figure")
    parser.add_argument("--dataset", help="imdb or stack", default="imdb")
    args = parser.parse_args()
    dataset = args.dataset

    cfg = get_config(dataset)
    df_processed = load_and_process_data(cfg=cfg)

    seed = 2025
    set_global_seed(seed)

    rest_dfs = combine_train_test_dfs(cfg)
    print("done")

    os.makedirs(cfg.TRAIN_TEST_ONLINE_CONVARIATE, exist_ok=True)
    for key, df_value in rest_dfs.items():
        df_value_shfulle = df_value.sample(frac=1, random_state=seed).reset_index(drop=True)
        df_value_shfulle["execution_time_ms"] = df_value_shfulle["execution_time_ms"].apply(json.dumps)

        save_path = os.path.join(cfg.TRAIN_TEST_ONLINE_CONVARIATE, f"workload_convariate_{key}.csv")
        if not os.path.exists(save_path):
            df_value_shfulle.to_csv(save_path, index=False)
