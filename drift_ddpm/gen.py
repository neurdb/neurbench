import pickle

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


INFERENCE_BATCH_SIZE = 524288


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
        train_data_path, doublequote=False, escapechar="\\", low_memory=False
    )
    print("Original data")
    print(original_data)

    train_data = original_data[config["applicable_columns"]]
    print("Data with drifting columns")
    print(train_data)

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
            diffuser=diffuser,
            save_path=os.path.join(save_dir, "controller.pt"),
            device=device,
            lr=args.controller_lr,
            d_hidden=args.controller_dim,
            steps=args.controller_steps,
            drop_out=0.0,
            bs=args.controller_bs,
        )

    controller = torch.load(os.path.join(save_dir, "controller.pt"))

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
            len(b), controller, diffuser, device, args.drift, args.scale_factor
        )
        all_data.append(sample_data)

    sample_data = torch.cat(all_data, dim=0)

    sample_data = sample_data.cpu().numpy()
    sample_data = data_wrapper.Reverse(sample_data)
    sample_data = sample_data[config["applicable_columns"]]
    
    if len(original_data.index) > len(sample_data.index):
        original_data = original_data[:len(sample_data.index)]

    print("Drifted columns")
    print(sample_data)

    original_data[config["applicable_columns"]] = sample_data
    print("Drifted data")
    print(original_data)

    original_data.to_csv(
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

    parser.add_argument("--controller-dim", nargs="+", type=int, default=(512, 512))
    parser.add_argument("--controller-lr", type=float, default=0.001)
    parser.add_argument("--controller-steps", type=int, default=10000)
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

    args = parser.parse_args()

    main(args)
