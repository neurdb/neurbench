import json
import os
import shutil
import time
from typing import List

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.shortcuts import button_dialog


def load_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data


def is_integer(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def print_args(**kwargs):
    print("-" * 20)
    for key, value in kwargs.items():
        print(f"{key}: {value}")
    print("-" * 20)


# class CommandPrompter(Validator):
#     def validate(self, document):
#         text = document.text

#         if text and not text.isdigit():
#             i = 0

#             # Get index of first non numeric character.
#             # We want to move the cursor here.
#             for i, c in enumerate(text):
#                 if not c.isdigit():
#                     break

#             raise ValidationError(
#                 message="This input contains non-numeric characters", cursor_position=i
#             )


print(
    "Welcome to NeurBench interactive shell. Type 'h' to view a list of commands. Type 'q' to quit."
)


def show_help():
    print(
        """Available commands:
  h, help                            Show this help message
  q, quit                            Exit the interactive shell
  gd DATASET [TABLE] DRIFT [SCALE]   Generate data that drifts DRIFT on DATASET
  gq DATASET DRIFT                   Generate query that drifts DRIFT on DATASET
  dd DATASET [TABLE]                 Delete data generator model for DATASET
  dq DATASET                         Delete query generator model for DATASET
  tqo                                Train learned query optimizer
  tid                                Train learned index
  tcc                                Train learned concurrency control
  iqo                                Inference learned query optimizer
  iid                                Inference learned index
  icc                                Inference learned concurrency control
"""
    )


def handle_generate_data(tokens: List[str]):
    tokens = tokens[1:]

    if len(tokens) < 2:
        print("Error: Not enough arguments")
        return

    dataset_name = tokens[0]
    table_name = ""
    drift = 0.0
    scale = 1.0

    if is_float(tokens[1]) or is_integer(tokens[1]):
        drift = tokens[1]
        if len(tokens) > 2:
            scale = tokens[2]
    else:
        if len(tokens) < 3:
            print("Error: Not enough arguments")
            return

        table_name = tokens[1]
        if not is_float(tokens[2]) and not is_integer(tokens[2]):
            print("Invalid drift value:", tokens[2])
            return
        else:
            drift = tokens[2]
            if len(tokens) > 3:
                scale = tokens[3]

    if table_name:
        print_args(
            dataset_name=dataset_name,
            table_name=table_name,
            drift=drift,
            scale=scale,
        )

        os.system(
            f"python dbproc.py --dataset-name={dataset_name} --table-name={table_name} --drift={drift} --scale={scale}"
        )
    else:
        # run on all tables
        ## get table names from dataset_info.json
        base_dir = os.path.join("datasets", dataset_name)
        config: dict = load_json(os.path.join(base_dir, "dataset_info.json"))
        table_names = [t for t in config.keys() if config[t]]
        print_args(
            dataset_name=dataset_name,
            table_names=table_names,
            drift=drift,
            scale=scale,
        )

        for t in table_names:
            print(f"Generating data for {dataset_name}.{t}...")
            os.system(
                f"python dbproc.py --dataset-name={dataset_name} --table-name={t} --drift={drift} --scale={scale}"
            )
            print(f"Data generation complete for {dataset_name}.{t}")

    print("Data generation complete")


def handle_delete_data(tokens: List[str]):
    tokens = tokens[1:]

    if len(tokens) < 1:
        print("Error: Not enough arguments")
        return

    dataset_name = tokens[0]
    table_name = tokens[1] if len(tokens) > 1 else ""

    exp_dir = "expdir"
    src_dir = os.path.join(exp_dir, dataset_name)
    dest_base_dir = os.path.join(".trash", exp_dir)

    if table_name:
        print_args(
            dataset_name=dataset_name,
            table_name=table_name,
        )

        src = os.path.join(src_dir, table_name)
        if not os.path.exists(src):
            print(
                f"Error: No data generator model found for table {table_name} in dataset {dataset_name}"
            )
            return

        if button_dialog(
            title="Delete data generator model",
            text=f"Do you want to move data generator model for table {table_name} in dataset {dataset_name} to the trash folder (.trash)?",
            buttons=[
                ("No", False),
                ("Yes", True),
            ],
        ).run():
            dest_dir = os.path.join(dest_base_dir, dataset_name)
            os.makedirs(dest_dir, exist_ok=True)

            dst = os.path.join(dest_dir, f"{table_name}_{time.time()}")
            shutil.move(src, dst)
            print(f"Moved {src} to {dst}")
    else:
        print_args(dataset_name=dataset_name)

        if button_dialog(
            title="Delete data generator model",
            text=f"Do you want to move data generator model for all tables in dataset {dataset_name} to the trash folder (.trash)?",
            buttons=[
                ("No", False),
                ("Yes", True),
            ],
        ).run():
            src = src_dir
            dst = os.path.join(dest_base_dir, f"{dataset_name}_{time.time()}")
            shutil.move(src, dst)
            print(f"Moved {src} to {dst}")


def main():
    session = PromptSession(
        history=FileHistory(os.path.join(os.path.dirname(__file__), ".cli_history"))
    )

    while True:
        try:
            text = session.prompt("[neurbench]> ")
            text = text.strip()

        except KeyboardInterrupt:
            continue
        except EOFError:
            break
        else:
            if text in ["q", "quit"]:
                break

            elif text in ["h", "help"]:
                show_help()
                continue

            tokens = text.split()
            main_command = tokens[0]

            if main_command == "gd":
                # Generate data command
                handle_generate_data(tokens)
                continue

            if main_command == "gq":
                # Generate query command
                print("Error: Generate query command is not yet implemented.")
                continue

            if main_command == "dd":
                handle_delete_data(tokens)
                continue

            if main_command in ["tqo", "tid", "tcc", "iqo", "iid", "icc"]:
                print("Error: Training or inference commands are not yet implemented.")
                continue

            print("Unknown command:", text)


if __name__ == "__main__":
    main()
