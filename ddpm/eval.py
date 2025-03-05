import deterministic

deterministic.seed_everything(42)

import argparse
import os
import warnings

import pandas as pd
import torch

import data_utils as du
import eval_utils as eu
import lib_completion as lc

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--task-name", type=str, default="oversampling")
parser.add_argument("--dataset-name", type=str, default="default")
parser.add_argument("--save-name", type=str, default="output")
parser.add_argument("--device", type=int, default=1)
args = parser.parse_args()

save_dir = os.path.join("expdir", args.save_name)
assert os.path.exists(save_dir)
device = torch.device(f"cuda:{args.device}")

if args.task_name == "oversampling":
    data_path = "datasets/minority_class_oversampling"

    config = du.load_json(os.path.join(data_path, "dataset_info.json"))[
        args.dataset_name
    ]
    train_real = pd.read_csv(os.path.join(data_path, f"{args.dataset_name}_train.csv"))
    test_data = pd.read_csv(os.path.join(data_path, f"{args.dataset_name}_test.csv"))

    result = pd.DataFrame(
        [], columns=["Methods", "DT10", "DT30", "RF10", "RF20", "Adaboost", "LR", "MLP"]
    )

    print("Evaluating the Identity ...")
    f1_results = eu.F1_eval(
        train_real,
        test_data,
        None,
        config["label"],
        config["epochs"],
        task="Clf",
        average=config["average"],
    )
    result.loc[len(result)] = ["Identity"] + f1_results

    print("Evaluating the RelDDPM ...")
    oversample_data = pd.read_csv(os.path.join(save_dir, "oversample_data.csv"))
    train_data = pd.concat((train_real, oversample_data))
    f1_results = eu.F1_eval(
        train_data,
        test_data,
        None,
        config["label"],
        config["epochs"],
        task="Clf",
        average=config["average"],
    )
    result.loc[len(result)] = ["RelDDPM"] + f1_results

    result.to_csv(os.path.join(save_dir, "f1_results.csv"), index=False)

if args.task_name == "completion":
    if args.dataset_name == "heart":
        data_path = "datasets/missing_tuple_completion/heart"

        schema = lc.schemas.HeartSchema(data_path)
        incomplete_tables = ["cardio"]
        schema.load_setup(incomplete_tables, keep_rate=0.4)
        queries = du.load_pickle(os.path.join(data_path, "aqp_queries.pkl"))

        real_data = schema.remove_pk(schema.ground_truth)
        incomplete_data = schema.remove_pk(schema.joined_data)
        complete_data = pd.read_csv(os.path.join(save_dir, "complete_data.csv"))

        print("Compute RE for incomplete data ...")
        incomplete_re = eu.RE_evaluate(real_data, incomplete_data, queries)
        print("Compute RE for complete data ...")
        complete_re = eu.RE_evaluate(real_data, complete_data, queries)

        err_rd = eu.error_reduction(incomplete_re[-1], complete_re[-1])

        attr = "maximum_heart_rate_achieved"
        bias_rd = eu.bias_reduction(
            real_data[attr].mean(),
            incomplete_data[attr].mean(),
            complete_data[attr].mean(),
        )

        results = {"bias_reduction": bias_rd, "re_reduction": err_rd}
        du.save_json(data=results, path=os.path.join(save_dir, "results.json"))

    if args.dataset_name == "airbnb":
        data_path = "datasets/missing_tuple_completion/airbnb"

        schema = lc.schemas.AirbnbSchema(data_path)
        incomplete_tables = ["apartment"]
        schema.load_setup(incomplete_tables, keep_rate=0.4)
        queries = du.load_pickle(os.path.join(data_path, "aqp_queries.pkl"))

        real_data = schema.remove_pk(schema.ground_truth)
        incomplete_data = schema.remove_pk(schema.joined_data)
        complete_data = pd.read_csv(os.path.join(save_dir, "complete_data.csv"))

        print("Compute RE for incomplete data ...")
        incomplete_re = eu.RE_evaluate(real_data, incomplete_data, queries)
        print("Compute RE for complete data ...")
        complete_re = eu.RE_evaluate(real_data, complete_data, queries)

        err_rd = eu.error_reduction(incomplete_re[-1], complete_re[-1])

        attr = "hosts.host_response_rate"
        bias_rd = eu.bias_reduction(
            real_data[attr].mean(),
            incomplete_data[attr].mean(),
            complete_data[attr].mean(),
        )

        results = {"bias_reduction": bias_rd, "re_reduction": err_rd}
        du.save_json(data=results, path=os.path.join(save_dir, "results.json"))

    if args.dataset_name == "imdb":
        data_path = "datasets/missing_tuple_completion/imdb"

        schema = lc.schemas.ImdbSchema(data_path)
        schema.load_setup(keep_rate=0.4)
        ma_queries = du.load_pickle(
            os.path.join(data_path, "aqp_queries_movie_actor.pkl")
        )
        md_queries = du.load_pickle(
            os.path.join(data_path, "aqp_queries_movie_director.pkl")
        )

        real_data_ma = schema.remove_pk(schema.ma_ground_truth)
        real_data_md = schema.remove_pk(schema.md_ground_truth)
        incomplete_data_ma = schema.remove_pk(schema.ma_joined_data)
        incomplete_data_md = schema.remove_pk(schema.md_joined_data)
        complete_data_ma = pd.read_csv(
            os.path.join(save_dir, "movie_actor_complete_data.csv")
        )
        complete_data_md = pd.read_csv(
            os.path.join(save_dir, "movie_director_complete_data.csv")
        )

        print("Compute RE for incomplete data ...")
        incomplete_re_ma = eu.RE_evaluate(real_data_ma, incomplete_data_ma, ma_queries)
        incomplete_re_md = eu.RE_evaluate(real_data_md, incomplete_data_md, md_queries)
        print("Compute RE for complete data ...")
        complete_re_ma = eu.RE_evaluate(real_data_ma, complete_data_ma, ma_queries)
        complete_re_md = eu.RE_evaluate(real_data_md, complete_data_md, md_queries)

        err_rd_ma = eu.error_reduction(incomplete_re_ma[-1], complete_re_ma[-1])
        err_rd_md = eu.error_reduction(incomplete_re_md[-1], complete_re_md[-1])

        attr = "movie.rating"
        bias_rd_ma = eu.bias_reduction(
            real_data_ma[attr].mean(),
            incomplete_data_ma[attr].mean(),
            complete_data_ma[attr].mean(),
        )
        bias_rd_md = eu.bias_reduction(
            real_data_md[attr].mean(),
            incomplete_data_md[attr].mean(),
            complete_data_md[attr].mean(),
        )

        results = {
            "bias_reduction": (bias_rd_ma + bias_rd_md) / 2,
            "re_reduction": (err_rd_ma + err_rd_md) / 2,
        }
        du.save_json(data=results, path=os.path.join(save_dir, "results.json"))
