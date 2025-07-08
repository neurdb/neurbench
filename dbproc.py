import pickle
import sys

sys.path.append("drift_ddpm")

import deterministic
import numpy as np
from tqdm import tqdm

deterministic.seed_everything(42)

import argparse
import os
import warnings

import data_utils as du
import lib_oversampling as lo
import pandas as pd
import torch

warnings.filterwarnings("ignore")


INFERENCE_BATCH_SIZE = 262144  # 524288


def main(args: argparse.Namespace):
    save_dir = os.path.join("expdir", args.dataset_name, args.table_name)
    if args.variant_id > 0:
        save_dir += f"-{args.variant_id}"

    print("save_dir", save_dir)
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device(f"cuda:{args.device}")

    base_dir = os.path.join("datasets", args.dataset_name)

    config: dict = du.load_json(os.path.join(base_dir, "dataset_info.json"))
    config = config[args.table_name]

    train_data_path = os.path.join(base_dir, f"{args.table_name}.csv")
    original_data = pd.read_csv(
        ### imdb
        # train_data_path, doublequote=False, escapechar="\\", low_memory=False
        ### stack
        train_data_path, doublequote=True, low_memory=False,
    )
    print("Original data")
    print(original_data)

    train_data = original_data[config["applicable_columns"]]
    print("Data with drifting columns")
    print(train_data)

    real_base_dir = os.path.join("datasets", args.dataset_name + "_2009")

    real_drift_data_path = os.path.join(real_base_dir, f"{args.table_name}.csv")
    original_real_drift_data = pd.read_csv(
        real_drift_data_path,
        ### imdb
        # doublequote=True,
        # quotechar='"',
        # escapechar="\\",
        # low_memory=False,
        ### stack
        doublequote=True,
        low_memory=False,
    )
    print("Real Drift data")
    print(original_real_drift_data)

    real_data = original_real_drift_data[config["applicable_columns"]]
    print("Real Drift Data with drifting columns")
    print(real_data)

    real_cond_data_path = os.path.join(
        real_base_dir, f"{args.table_name}.csv"
    )  # {args.table_name}
    original_real_cond_data = pd.read_csv(
        real_cond_data_path,
        ### imdb
        # doublequote=True,
        # quotechar='"',
        # escapechar="\\",
        # low_memory=False,
        ### stack
        doublequote=True,
        low_memory=False,
    )
    print("Conditional data")
    print(original_real_cond_data)

    # condition_data = original_real_cond_data[["id"]] ## TODO: for different table need to load different column
    # synthetic_data = original_data[["movie_id"]]

    if args.fillna:
        train_data.fillna(0.0, inplace=True)
        real_data.fillna(0.0, inplace=True)

    if args.reuse and os.path.exists(os.path.join(save_dir, "data_wrapper.pkl")):
        with open(os.path.join(save_dir, "data_wrapper.pkl"), "rb") as f:
            data_wrapper = pickle.load(f)
    else:
        data_wrapper = du.DataWrapper()
        data_wrapper.fit(train_data)

        with open(os.path.join(save_dir, "data_wrapper.pkl"), "wb") as f:
            pickle.dump(data_wrapper, f)

    if args.reuse and os.path.exists(os.path.join(save_dir, "train_x.npy")):
        with open(os.path.join(save_dir, "train_x.npy"), "rb") as f:
            train_x = np.load(f)
    else:
        train_x = data_wrapper.transform(train_data)

        with open(os.path.join(save_dir, "train_x.npy"), "wb") as f:
            np.save(f, train_x)

    if args.reuse and os.path.exists(os.path.join(save_dir, "real_data_wrapper.pkl")):
        with open(os.path.join(save_dir, "real_data_wrapper.pkl"), "rb") as f:
            real_data_wrapper = pickle.load(f)
    else:
        real_data_wrapper = du.DataWrapper()
        real_data_wrapper.fit(real_data)

        with open(os.path.join(save_dir, "real_data_wrapper.pkl"), "wb") as f:
            pickle.dump(real_data_wrapper, f)

    if args.reuse and os.path.exists(os.path.join(save_dir, "real_x.npy")):
        with open(os.path.join(save_dir, "real_x.npy"), "rb") as f:
            real_x = np.load(f)
    else:
        real_x = real_data_wrapper.transform(real_data)

        with open(os.path.join(save_dir, "real_x.npy"), "wb") as f:
            np.save(f, real_x)

    # if args.reuse and os.path.exists(os.path.join(save_dir, "cond_data_wrapper.pkl")):
    #     with open(os.path.join(save_dir, "cond_data_wrapper.pkl"), "rb") as f:
    #         cond_data_wrapper = pickle.load(f)
    # else:
    #     cond_data_wrapper = du.DataWrapper()
    #     cond_data_wrapper.fit(condition_data)

    #     with open(os.path.join(save_dir, "cond_data_wrapper.pkl"), "wb") as f:
    #         pickle.dump(cond_data_wrapper, f)

    # if args.reuse and os.path.exists(os.path.join(save_dir, "cond_x.npy")):
    #     with open(os.path.join(save_dir, "cond_x.npy"), "rb") as f:
    #         cond_x = np.load(f)
    # else:
    #     cond_x = cond_data_wrapper.transform(condition_data)

    #     with open(os.path.join(save_dir, "cond_x.npy"), "wb") as f:
    #         np.save(f, cond_x)

    # synthetic_data_wrapper = du.DataWrapper()
    # synthetic_data_wrapper.fit(synthetic_data)
    # synthetic_x = synthetic_data_wrapper.transform(synthetic_data)

    """ diffuser training. To avoid randomness, reseed everything. """
    deterministic.seed_everything(args.random_state)

    if not args.retrain_diffuser and os.path.exists(
        os.path.join(save_dir, "diffuser.pt")
    ):
        print("Load existing diffuser")
    else:
        print("Train diffuser")
        lo.diffuser_training(
            train_x=train_x,
            save_path=os.path.join(save_dir, "diffuser.pt"),
            device=device,
            d_hidden=args.diffuser_dim,
            num_timesteps=args.diffuser_timesteps,
            epochs=args.diffuser_steps,
            lr=args.diffuser_lr,
            drop_out=0.0,
            bs=args.diffuser_bs,
            lambda_p=args.lambda_p,
            lambda_s=args.lambda_s,
        )

    diffuser = torch.load(os.path.join(save_dir, "diffuser.pt"))

    """ controller training. To avoid randomness, reseed everything. """
    deterministic.seed_everything(args.random_state)

    if not args.retrain_controller and os.path.exists(
        os.path.join(save_dir, "controller.pt")
    ):
        print("Load existing controller")
    else:
        print("Train controller")
        lo.controller_training(
            train_x=train_x,
            real_x=real_x,
            # cond_x=cond_x,
            # synthetic_x=synthetic_x,
            diffuser=diffuser,
            save_path=os.path.join(save_dir, "controller.pt"),
            cond_save_path=os.path.join(save_dir, "controller_cond.pt"),
            device=device,
            lr=args.controller_lr,
            d_hidden=args.controller_dim,
            steps=args.controller_steps,
            drop_out=0.0,
            bs=args.controller_bs,
        )

    controller = torch.load(os.path.join(save_dir, "controller.pt"))
    # controller_cond = torch.load(os.path.join(save_dir, "controller_cond.pt"))

    """ oversampling. To avoid randomness, reseed everything. """
    deterministic.seed_everything(args.random_state)

    ids = range(config["n_samples"])
    batched_ids = [
        ids[x : x + INFERENCE_BATCH_SIZE]
        for x in range(0, len(ids), INFERENCE_BATCH_SIZE)
    ]

    all_data = []
    for b in tqdm(batched_ids):
        sample_data = lo.oversampling(
            len(b),
            controller,
            diffuser,
            None,  # controller_cond,
            None,  # cond_x,
            None,  # synthetic_x,
            device,
            args.drift,
            args.scale_factor,
        )
        all_data.append(sample_data)

    sample_data = torch.cat(all_data, dim=0)

    sample_data = sample_data.cpu().numpy()
    sample_data = data_wrapper.Reverse(sample_data)
    sample_data = sample_data[config["applicable_columns"]]

    # if len(original_data.index) > len(sample_data.index):
    # original_data = original_data[:len(sample_data.index)]

    if len(original_real_drift_data.index) > len(sample_data.index):
        original_real_drift_data = original_real_drift_data[: len(sample_data.index)]

    print("Drifted columns")
    print(sample_data)

    # original_data[config["applicable_columns"]] = sample_data
    original_real_drift_data[config["applicable_columns"]] = sample_data
    print("Drifted data")
    print(original_real_drift_data)

    original_real_drift_data.to_csv(
        os.path.join(save_dir, f"{args.table_name}.drifted.csv"),
        index=False,
        doublequote=False,
        escapechar="\\",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", type=str, default="imdb")
    parser.add_argument("--table-name", type=str, default="nosuchtable")

    parser.add_argument(
        "--diffuser-dim", nargs="+", type=int, default=(512, 1024, 1024, 512)
    )
    parser.add_argument("--diffuser-lr", type=float, default=0.0018)
    parser.add_argument("--diffuser-steps", type=int, default=30000)
    parser.add_argument("--diffuser-bs", type=int, default=2048)
    parser.add_argument("--diffuser-timesteps", type=int, default=1000)

    parser.add_argument(
        "--controller-dim", nargs="+", type=int, default=(512, 512)
    )  ## TODO: higher --controller-dim 1024 1024  512 512 512
    parser.add_argument("--controller-lr", type=float, default=0.001)  ## TODO: lower
    parser.add_argument("--controller-steps", type=int, default=10000)  ## TODO: higher
    parser.add_argument("--controller-bs", type=int, default=512)

    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--scale-factor", type=float, default=8.0)
    # parser.add_argument("--save-name", type=str, default="output")

    parser.add_argument("--lambda-p", type=float, default=1.0)
    parser.add_argument("--lambda-s", type=float, default=1.0)

    parser.add_argument("--retrain-diffuser", action="store_true")
    parser.add_argument("--retrain-controller", action="store_true")

    parser.add_argument("--reuse", action="store_true")

    parser.add_argument("--variant-id", type=int, default=-1)

    parser.add_argument("--drift", type=float, default=0.3)

    parser.add_argument("--random-state", type=int, default=42)

    parser.add_argument("--fillna", action="store_true", default=False)

    # parser.add_argument("--cond", action="store_true", default=False)

    args = parser.parse_args()

    main(args)
