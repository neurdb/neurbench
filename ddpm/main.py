import deterministic

deterministic.seed_everything(42)

import argparse
import os
import warnings

import pandas as pd
import torch

import data_utils as du
import lib_completion as lc
import lib_oversampling as lo

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--task-name", type=str, default="oversampling")
parser.add_argument("--dataset-name", type=str, default="default")

parser.add_argument(
    "--diffuser-dim", nargs="+", type=int, default=(512, 1024, 1024, 512)
)
parser.add_argument("--diffuser-lr", type=float, default=0.0018)
parser.add_argument("--diffuser-steps", type=int, default=30000)
parser.add_argument("--diffuser-bs", type=int, default=4096)
parser.add_argument("--diffuser-timesteps", type=int, default=1000)

parser.add_argument("--controller-dim", nargs="+", type=int, default=(512, 512))
parser.add_argument("--controller-lr", type=float, default=0.001)
parser.add_argument("--controller-steps", type=int, default=10000)
parser.add_argument("--controller-bs", type=int, default=512)

parser.add_argument("--device", type=int, default=1)
parser.add_argument("--scale-factor", type=float, default=8.0)
parser.add_argument("--save-name", type=str, default="output")

parser.add_argument("--lambda-p", type=float, default=1.0)
parser.add_argument("--lambda-s", type=float, default=1.0)

parser.add_argument("--retrain-diffuser", action="store_true")
parser.add_argument("--retrain-controller", action="store_true")

parser.add_argument("--drift", type=float, default=0.3)

args = parser.parse_args()

save_dir = os.path.join("expdir", args.save_name)
os.makedirs(save_dir, exist_ok=True)
device = torch.device(f"cuda:{args.device}")

if args.task_name == "oversampling":
    config = du.load_json(f"datasets/minority_class_oversampling/dataset_info.json")[
        args.dataset_name
    ]
    train_data = pd.read_csv(
        f"datasets/minority_class_oversampling/{args.dataset_name}_train.csv"
    )
    test_data = pd.read_csv(
        f"datasets/minority_class_oversampling/{args.dataset_name}_test.csv"
    )

    all_data = pd.concat((train_data, test_data))
    data_wrapper, label_wrapper = lo.data_preprocessing(
        all_data, config["label"], save_dir
    )

    """ diffuser training. To avoid randomness, reseed everything. """
    deterministic.seed_everything(42)

    train_x = data_wrapper.transform(train_data)

    if not args.retrain_diffuser and os.path.exists(
        os.path.join(save_dir, "diffuser.pt")
    ):
        print("Load existing diffuser")
    else:
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
    label = config["label"]
    n_classes = len(pd.unique(train_data[label]))
    train_x = data_wrapper.transform(train_data)
    train_y = label_wrapper.transform(train_data[[label]])

    """ controller training. To avoid randomness, reseed everything. """
    deterministic.seed_everything(42)

    if not args.retrain_controller and os.path.exists(
        os.path.join(save_dir, "controller.pt")
    ):
        print("Load existing controller")
    else:
        lo.controller_training(
            train_x=train_x,
            train_y=train_y,
            diffuser=diffuser,
            save_path=os.path.join(save_dir, "controller.pt"),
            device=device,
            # n_classes=n_classes,
            lr=args.controller_lr,
            d_hidden=args.controller_dim,
            steps=args.controller_steps,
            drop_out=0.0,
            bs=args.controller_bs,
        )

    diffuser = torch.load(os.path.join(save_dir, "diffuser.pt"))
    controller = torch.load(os.path.join(save_dir, "controller.pt"))

    """ oversampling. To avoid randomness, reseed everything. """
    deterministic.seed_everything(42)

    sample_data = []
    for i in range(len(config["minority_classes"])):
        samples = lo.oversampling(
            config["n_samples"][i],
            controller,
            diffuser,
            config["minority_classes"][i],
            device,
            # n_classes,
            args.drift,
            args.scale_factor,
        )
        sample_data.append(samples)

    sample_data = torch.cat(sample_data, dim=0)
    sample_data = sample_data.cpu().numpy()
    sample_data = data_wrapper.Reverse(sample_data)
    sample_data.to_csv(os.path.join(save_dir, "oversample_data.csv"), index=False)


if args.task_name == "completion":
    if args.dataset_name == "heart":
        schema = lc.schemas.HeartSchema("datasets/missing_tuple_completion/heart/")
        incomplete_tables = ["cardio"]
        schema.load_setup(incomplete_tables, keep_rate=0.4)

        """ diffuser training """
        if len(schema.cond_patient) > 0:
            print("Train diffuser for cardio")
            data_wrapper = schema.cardio_wrapper
            train_data = schema.join_cardio
            train_x = data_wrapper.transform(train_data)
            save_path = os.path.join(save_dir, "diffuser_cardio.pt")
            lc.diffuser_training(
                train_x=train_x,
                save_path=save_path,
                device=device,
                d_hidden=args.diffuser_dim,
                num_timesteps=args.diffuser_timesteps,
                epochs=args.diffuser_steps,
                lr=args.diffuser_lr,
                drop_out=0.0,
                bs=args.diffuser_bs,
            )

        """ controller training """
        if "cardio" in incomplete_tables:
            print("Train controller for cardio")
            diffuser = torch.load(os.path.join(save_dir, "diffuser_cardio.pt"))
            condition_wrapper = [schema.patient_wrapper]
            synthetic_wrapper = [schema.cardio_wrapper]
            raw_train_data = schema.joined_data.drop(["pid"], axis=1)
            save_path = os.path.join(save_dir, "controller_patient->cardio.pt")
            lc.controller_training(
                raw_train_data,
                condition_wrapper,
                synthetic_wrapper,
                diffuser,
                save_path,
                device=device,
                batch_size=4096,
                steps=2000,
            )

        """ Missing Tuple Completion """
        complete_data = schema.joined_data.drop(["pid"], axis=1)
        if len(schema.cond_patient) > 0:
            print("Completion ...")
            print("n samples: ", len(schema.cond_patient))
            condition_wrapper = [schema.patient_wrapper]
            synthetic_wrapper = [schema.cardio_wrapper]
            cond_data = schema.cond_patient

            diffuser = torch.load(os.path.join(save_dir, "diffuser_cardio.pt"))
            controller = torch.load(
                os.path.join(save_dir, "controller_patient->cardio.pt")
            )

            cond_data, sample_data = lc.table_condition_sample(
                condition_wrapper,
                synthetic_wrapper,
                cond_data,
                diffuser,
                controller,
                device=device,
                scale_factor=25,
            )
            sample_data = pd.concat((cond_data, sample_data), axis=1)
            sample_data = sample_data[complete_data.columns]
            complete_data = pd.concat((complete_data, sample_data), axis=0)
        complete_data.to_csv(os.path.join(save_dir, "complete_data.csv"), index=False)

    if args.dataset_name == "airbnb":
        schema = lc.schemas.AirbnbSchema("datasets/missing_tuple_completion/airbnb/")
        incomplete_tables = ["apartment"]
        schema.load_setup(incomplete_tables, keep_rate=0.4)

        """ diffuser training """
        print("Train diffuser for apartment")
        train_data = schema.join_apartment
        train_x = schema.apartment_wrapper.transform(train_data)
        save_path = os.path.join(save_dir, "diffuser_apartment.pt")
        lc.diffuser_training(
            train_x=train_x,
            save_path=save_path,
            device=device,
            d_hidden=args.diffuser_dim,
            num_timesteps=args.diffuser_timesteps,
            epochs=args.diffuser_steps,
            lr=args.diffuser_lr,
            drop_out=0.0,
            bs=args.diffuser_bs,
        )

        print("Train diffuser for landlord")
        train_data = schema.join_landlord
        train_x = schema.landlord_wrapper.transform(train_data)
        save_path = os.path.join(save_dir, "diffuser_landlord.pt")
        lc.diffuser_training(
            train_x=train_x,
            save_path=save_path,
            device=device,
            d_hidden=args.diffuser_dim,
            num_timesteps=args.diffuser_timesteps,
            epochs=args.diffuser_steps,
            lr=args.diffuser_lr,
            drop_out=0.0,
            bs=args.diffuser_bs,
        )

        print("Compute condition tuples ...")
        schema.get_condition_tuples()

        print("Tune diffuser for landlord")
        train_data = pd.concat(
            (schema.joined_data[schema.join_landlord.columns], schema.cond_landlord),
            axis=0,
        )
        train_x = schema.landlord_wrapper.transform(train_data)
        diffuser = torch.load(os.path.join(save_dir, f"diffuser_landlord.pt"))
        save_path = os.path.join(save_dir, f"diffuser_landlord_tuned.pt")
        lc.diffuser_tuning(train_x, diffuser, save_path, device, epochs=3000)

        """ controller training """
        if (
            len(schema.cond_neighborhoods) >= len(schema.cond_landlord)
            and len(schema.cond_neighborhoods) > 0
        ):
            print("Train controller neighborhoods->apartment")
            diffuser = torch.load(os.path.join(save_dir, "diffuser_apartment.pt"))
            condition_wrapper = [schema.neighborhoods_wrapper]
            synthetic_wrapper = [schema.apartment_wrapper]
            raw_train_data = schema.joined_data.copy()
            save_path = os.path.join(save_dir, "controller_neighborhoods->apartment.pt")
            lc.controller_training(
                raw_train_data,
                condition_wrapper,
                synthetic_wrapper,
                diffuser,
                save_path,
                device=device,
                batch_size=4096,
                steps=2000,
            )

            if (
                len(schema.cond_apartment) == 0
                and len(schema.cond_apartment_neighborhoods) == 0
            ):
                print("Train controller neighborhoods_apartment->landlord")
                diffuser = torch.load(
                    os.path.join(save_dir, "diffuser_landlord_tuned.pt")
                )
                condition_wrapper = [
                    schema.neighborhoods_wrapper,
                    schema.apartment_wrapper,
                ]
                synthetic_wrapper = [schema.landlord_wrapper]
                raw_train_data = schema.joined_data.copy()
                save_path = os.path.join(
                    save_dir, "controller_neighborhoods_apartment->landlord.pt"
                )
                lc.controller_training(
                    raw_train_data,
                    condition_wrapper,
                    synthetic_wrapper,
                    diffuser,
                    save_path,
                    device=device,
                    batch_size=4096,
                    steps=2000,
                )

        """ Missing Tuple Completion """
        complete_data = schema.joined_data.copy()
        complete_data = schema.remove_pk(complete_data)

        if (
            len(schema.cond_neighborhoods) >= len(schema.cond_landlord)
            and len(schema.cond_neighborhoods) > 0
        ):
            print("Generate apartment ..")
            condition_wrapper = [schema.neighborhoods_wrapper]
            synthetic_wrapper = [schema.apartment_wrapper]
            cond_data = schema.cond_neighborhoods
            diffuser = torch.load(os.path.join(save_dir, "diffuser_apartment.pt"))
            controller = torch.load(
                os.path.join(save_dir, "controller_neighborhoods->apartment.pt")
            )
            cond_data, sample_data = lc.table_condition_sample(
                condition_wrapper,
                synthetic_wrapper,
                cond_data,
                diffuser,
                controller,
                device=device,
                scale_factor=30,
            )

            print("Generate landlord ..")
            condition_wrapper = [schema.neighborhoods_wrapper, schema.apartment_wrapper]
            synthetic_wrapper = [schema.landlord_wrapper]
            cond_data = pd.concat((cond_data, sample_data), axis=1)
            diffuser = torch.load(os.path.join(save_dir, f"diffuser_landlord_tuned.pt"))
            controller = torch.load(
                os.path.join(
                    save_dir, f"controller_neighborhoods_apartment->landlord.pt"
                )
            )
            cond_data, sample_data = lc.table_condition_sample(
                condition_wrapper,
                synthetic_wrapper,
                cond_data,
                diffuser,
                controller,
                device=device,
                scale_factor=20,
            )

            sample_data = pd.concat((cond_data, sample_data), axis=1)
            sample_data = sample_data[complete_data.columns]
            complete_data = pd.concat((complete_data, sample_data), axis=0)

        complete_data.to_csv(os.path.join(save_dir, "complete_data.csv"), index=False)

    if args.dataset_name == "imdb":
        schema = lc.schemas.ImdbSchema("datasets/missing_tuple_completion/imdb/")
        schema.load_setup(keep_rate=0.4)

        """ diffuser training """
        print("Train diffuser for movie")
        train_data = schema.join_movie
        train_x = schema.movie_wrapper.transform(train_data)
        save_path = os.path.join(save_dir, "diffuser_movie.pt")
        lc.diffuser_training(
            train_x=train_x,
            save_path=save_path,
            device=device,
            d_hidden=args.diffuser_dim,
            num_timesteps=args.diffuser_timesteps,
            epochs=args.diffuser_steps,
            lr=args.diffuser_lr,
            drop_out=0.0,
            bs=args.diffuser_bs,
        )

        print("Compute condition tuples ...")
        schema.get_condition_tuples()

        print("Tune diffuser for movie_actor")
        train_data = pd.concat(
            (schema.ma_joined_data[schema.join_movie.columns], schema.ma_cond_movie),
            axis=0,
        )
        train_x = schema.movie_wrapper.transform(train_data)
        diffuser = torch.load(os.path.join(save_dir, f"diffuser_movie.pt"))
        save_path = os.path.join(save_dir, f"diffuser_movie_tuned_ma.pt")
        lc.diffuser_tuning(train_x, diffuser, save_path, device, epochs=3000)

        print("Tune diffuser for movie_director")
        train_data = pd.concat(
            (schema.md_joined_data[schema.join_movie.columns], schema.md_cond_movie),
            axis=0,
        )
        train_x = schema.movie_wrapper.transform(train_data)
        diffuser = torch.load(os.path.join(save_dir, f"diffuser_movie.pt"))
        save_path = os.path.join(save_dir, f"diffuser_movie_tuned_md.pt")
        lc.diffuser_tuning(train_x, diffuser, save_path, device, epochs=3000)

        """ controller training """
        print("Train controller actor->movie")
        diffuser = torch.load(os.path.join(save_dir, "diffuser_movie_tuned_ma.pt"))
        condition_wrapper = [schema.actor_wrapper]
        synthetic_wrapper = [schema.movie_wrapper]
        raw_train_data = schema.ma_joined_data.copy()
        save_path = os.path.join(save_dir, "controller_actor->movie.pt")
        lc.controller_training(
            raw_train_data,
            condition_wrapper,
            synthetic_wrapper,
            diffuser,
            save_path,
            device=device,
            batch_size=4096,
            steps=2000,
        )

        print("Train controller director->movie")
        diffuser = torch.load(os.path.join(save_dir, "diffuser_movie_tuned_md.pt"))
        condition_wrapper = [schema.director_wrapper]
        synthetic_wrapper = [schema.movie_wrapper]
        raw_train_data = schema.md_joined_data.copy()
        save_path = os.path.join(save_dir, "controller_director->movie.pt")
        lc.controller_training(
            raw_train_data,
            condition_wrapper,
            synthetic_wrapper,
            diffuser,
            save_path,
            device=device,
            batch_size=4096,
            steps=2000,
        )

        """ Missing Tuple Completion """
        ma_complete_data = schema.ma_joined_data.copy()
        ma_complete_data = schema.remove_pk(ma_complete_data)

        md_complete_data = schema.md_joined_data.copy()
        md_complete_data = schema.remove_pk(md_complete_data)

        print("Generate actor->movie ..")
        condition_wrapper = [schema.actor_wrapper]
        synthetic_wrapper = [schema.movie_wrapper]
        cond_data = schema.cond_actor
        diffuser = torch.load(os.path.join(save_dir, "diffuser_movie_tuned_ma.pt"))
        controller = torch.load(os.path.join(save_dir, "controller_actor->movie.pt"))
        cond_data, sample_data = lc.table_condition_sample(
            condition_wrapper,
            synthetic_wrapper,
            cond_data,
            diffuser,
            controller,
            device=device,
            scale_factor=35,
        )
        sample_data = pd.concat((cond_data, sample_data), axis=1)
        sample_data = sample_data[ma_complete_data.columns]
        ma_complete_data = pd.concat((ma_complete_data, sample_data), axis=0)
        ma_complete_data.to_csv(
            os.path.join(save_dir, "movie_actor_complete_data.csv"), index=False
        )

        print("Generate director->movie ..")
        condition_wrapper = [schema.director_wrapper]
        synthetic_wrapper = [schema.movie_wrapper]
        cond_data = schema.cond_director
        diffuser = torch.load(os.path.join(save_dir, "diffuser_movie_tuned_md.pt"))
        controller = torch.load(os.path.join(save_dir, "controller_director->movie.pt"))
        cond_data, sample_data = lc.table_condition_sample(
            condition_wrapper,
            synthetic_wrapper,
            cond_data,
            diffuser,
            controller,
            device=device,
            scale_factor=20,
        )
        sample_data = pd.concat((cond_data, sample_data), axis=1)
        sample_data = sample_data[md_complete_data.columns]
        md_complete_data = pd.concat((md_complete_data, sample_data), axis=0)
        md_complete_data.to_csv(
            os.path.join(save_dir, "movie_director_complete_data.csv"), index=False
        )
