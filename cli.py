import json
import os
import shutil
import time
from typing import List

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.shortcuts import button_dialog

# Global configuration for all components
GLOBAL_CONFIG = {
    "dataset": "imdb",
    "drift": 0.3
}


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
    "Welcome to NRBench interactive shell. Type 'h' to view a list of commands. Type 'q' to quit."
)


def show_help():
    print(
        """NRBench - Available Commands
"======================================================"

Core Commands:
  h, help                            Show this help message
  q, quit                            Exit the interactive shell
  set [KEY] [VALUE]                  Set global configuration parameters
  set                                Show current configuration
  gd DATASET [TABLE] DRIFT [SCALE]   Generate data that drifts DRIFT on DATASET
  gq DATASET DRIFT                   Generate query that drifts DRIFT on DATASET
  dd DATASET [TABLE]                 Delete data generator model for DATASET
  dq DATASET                         Delete query generator model for DATASET
  tqo [LQO_NAME]                     Train learned query optimizer
  iqo [LQO_NAME]                     Test learned query optimizer
  idx [INDEX_NAME]                   Test learned index
  lcc                                Test learned concurrency control

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


def handle_tqo(tokens: List[str]):
    """Handle training learned query optimizer command"""
    tokens = tokens[1:]
    
    if len(tokens) < 1:
        print("Error: Please specify LQO_NAME")
        print("Usage: tqo [LQO_NAME]")
        print("Available LQO: bao, balsa, hybridqo, lero")
        print("Current global settings:")
        print(f"  Dataset: {GLOBAL_CONFIG['dataset']}")
        print(f"  Drift: {GLOBAL_CONFIG['drift']}")
        print("Use 'set' command to change global settings")
        print("Example: set drift 0.5")
        return
    
    lqo_name = tokens[0].lower()
    
    # Use global configuration
    dataset = GLOBAL_CONFIG["dataset"]
    drift = GLOBAL_CONFIG["drift"]
    
    print("Training learned query optimizer...")
    print_args(
        lqo_name=lqo_name,
        dataset=dataset,
        drift=drift
    )
    
    if lqo_name == "bao":
        print("Training Bao learned query optimizer...")
        
        # Check if bao directory exists
        bao_dir = os.path.join("benchmarks", "lqos", "bao")
        if not os.path.exists(bao_dir):
            print(f"Error: Bao directory not found at {bao_dir}")
            return
        
        # Check if train_bao.py script exists
        train_script = os.path.join(bao_dir, "train_bao.py")
        if not os.path.exists(train_script):
            print(f"Error: Bao training script not found at {train_script}")
            return
        
        # Run the training script
        print("Starting Bao training pipeline...")
        result = os.system(f"cd {bao_dir} && python train_bao.py")
        
        if result == 0:
            print("✅ Bao training completed successfully!")
        else:
            print(f"❌ Bao training failed with exit code {result}")
    
    elif lqo_name == "balsa":
        print("Training Balsa learned query optimizer...")
        
        # Check if balsa directory exists
        balsa_dir = os.path.join("benchmarks", "lqos", "balsa")
        if not os.path.exists(balsa_dir):
            print(f"Error: Balsa directory not found at {balsa_dir}")
            return
        
        # Check if train_balsa.py script exists
        train_script = os.path.join(balsa_dir, "train_balsa.py")
        if not os.path.exists(train_script):
            print(f"Error: Balsa training script not found at {train_script}")
            return
        
        # Run the training script
        print("Starting Balsa training pipeline...")
        result = os.system(f"cd {balsa_dir} && python train_balsa.py")
        
        if result == 0:
            print("✅ Balsa training completed successfully!")
        else:
            print(f"❌ Balsa training failed with exit code {result}")
    
    elif lqo_name == "hybridqo":
        print("Training HybridQO learned query optimizer...")
        
        # Check if hybrid_qo directory exists
        hybridqo_dir = os.path.join("benchmarks", "lqos", "hybrid_qo")
        if not os.path.exists(hybridqo_dir):
            print(f"Error: HybridQO directory not found at {hybridqo_dir}")
            return
        
        # Check if train_hybridqo.py script exists
        train_script = os.path.join(hybridqo_dir, "train_hybridqo.py")
        if not os.path.exists(train_script):
            print(f"Error: HybridQO training script not found at {train_script}")
            return
        
        # Run the training script
        print("Starting HybridQO training pipeline...")
        result = os.system(f"cd {hybridqo_dir} && python train_hybridqo.py")
        
        if result == 0:
            print("✅ HybridQO training completed successfully!")
        else:
            print(f"❌ HybridQO training failed with exit code {result}")
    
    elif lqo_name == "lero":
        print("Training Lero learned query optimizer...")
        
        # Check if Lero directory exists
        lero_dir = os.path.join("benchmarks", "lqos", "Lero")
        if not os.path.exists(lero_dir):
            print(f"Error: Lero directory not found at {lero_dir}")
            return
        
        # Check if train_lero.py script exists
        train_script = os.path.join(lero_dir, "train_lero.py")
        if not os.path.exists(train_script):
            print(f"Error: Lero training script not found at {train_script}")
            return
        
        # Run the training script
        print("Starting Lero training pipeline...")
        result = os.system(f"cd {lero_dir} && python train_lero.py")
        
        if result == 0:
            print("✅ Lero training completed successfully!")
        else:
            print(f"❌ Lero training failed with exit code {result}")
            
    else:
        print(f"Error: Unknown LQO '{lqo_name}'")
        print("Available LQO: bao, balsa, hybridqo, lero")
        return


def handle_iqo(tokens: List[str]):
    """Handle inference learned query optimizer command"""
    tokens = tokens[1:]
    
    if len(tokens) < 1:
        print("Error: Please specify LQO_NAME")
        print("Usage: iqo [LQO_NAME]")
        print("Available LQO: bao, balsa, hybridqo, lero")
        print("Current global settings:")
        print(f"  Dataset: {GLOBAL_CONFIG['dataset']}")
        print(f"  Drift: {GLOBAL_CONFIG['drift']}")
        print("Use 'set' command to change global settings")
        print("Example: set drift 0.5")
        return
    
    lqo_name = tokens[0].lower()
    
    # Use global configuration
    dataset = GLOBAL_CONFIG["dataset"]
    drift = GLOBAL_CONFIG["drift"]
    
    print("Running learned query optimizer inference...")
    print_args(
        lqo_name=lqo_name,
        dataset=dataset,
        drift=drift
    )
    
    if lqo_name == "bao":
        print("Running Bao learned query optimizer inference...")
        print_args(lqo_name=lqo_name)
        
        # Check if bao directory exists
        bao_dir = os.path.join("benchmarks", "lqos", "bao")
        if not os.path.exists(bao_dir):
            print(f"Error: Bao directory not found at {bao_dir}")
            return
        
        # Check if inference script exists
        inference_script = os.path.join(bao_dir, "inference_bao.py")
        if not os.path.exists(inference_script):
            print(f"Error: Bao inference script not found at {inference_script}")
            return
        
        # Run the inference script
        print("Starting Bao inference...")
        result = os.system(f"cd {bao_dir} && python inference_bao.py")
        
        if result == 0:
            print("✅ Bao inference completed successfully!")
        else:
            print(f"❌ Bao inference failed with exit code {result}")
    
    elif lqo_name == "balsa":
        print("Running Balsa learned query optimizer inference...")
        print_args(lqo_name=lqo_name)
        
        # Check if balsa directory exists
        balsa_dir = os.path.join("benchmarks", "lqos", "balsa")
        if not os.path.exists(balsa_dir):
            print(f"Error: Balsa directory not found at {balsa_dir}")
            return
        
        # Check if inference script exists
        inference_script = os.path.join(balsa_dir, "inference_balsa.py")
        if not os.path.exists(inference_script):
            print(f"Error: Balsa inference script not found at {inference_script}")
            return
        
        # Run the inference script
        print("Starting Balsa inference...")
        result = os.system(f"cd {balsa_dir} && python inference_balsa.py")
        
        if result == 0:
            print("✅ Balsa inference completed successfully!")
        else:
            print(f"❌ Balsa inference failed with exit code {result}")
    
    elif lqo_name == "hybridqo":
        print("Running HybridQO learned query optimizer inference...")
        print_args(lqo_name=lqo_name)
        
        # Check if hybrid_qo directory exists
        hybridqo_dir = os.path.join("benchmarks", "lqos", "hybrid_qo")
        if not os.path.exists(hybridqo_dir):
            print(f"Error: HybridQO directory not found at {hybridqo_dir}")
            return
        
        # Check if inference script exists
        inference_script = os.path.join(hybridqo_dir, "inference_hybridqo.py")
        if not os.path.exists(inference_script):
            print(f"Error: HybridQO inference script not found at {inference_script}")
            return
        
        # Run the inference script
        print("Starting HybridQO inference...")
        result = os.system(f"cd {hybridqo_dir} && python inference_hybridqo.py")
        
        if result == 0:
            print("✅ HybridQO inference completed successfully!")
        else:
            print(f"❌ HybridQO inference failed with exit code {result}")
    
    elif lqo_name == "lero":
        print("Running Lero learned query optimizer inference...")
        print_args(lqo_name=lqo_name)
        
        # Check if Lero directory exists
        lero_dir = os.path.join("benchmarks", "lqos", "Lero")
        if not os.path.exists(lero_dir):
            print(f"Error: Lero directory not found at {lero_dir}")
            return
        
        # Check if inference script exists
        inference_script = os.path.join(lero_dir, "inference_lero.py")
        if not os.path.exists(inference_script):
            print(f"Error: Lero inference script not found at {inference_script}")
            return
        
        # Run the inference script
        print("Starting Lero inference...")
        result = os.system(f"cd {lero_dir} && python inference_lero.py")
        
        if result == 0:
            print("✅ Lero inference completed successfully!")
        else:
            print(f"❌ Lero inference failed with exit code {result}")
            
    else:
        print(f"Error: Unknown LQO '{lqo_name}'")
        print("Available LQO: bao, balsa, hybridqo, lero")
        return


def handle_set(tokens: List[str]):
    """Handle set command for global configuration"""
    tokens = tokens[1:]
    
    if len(tokens) == 0:
        # Show current configuration
        print("Current global configuration:")
        print("-" * 30)
        for key, value in GLOBAL_CONFIG.items():
            print(f"{key}: {value}")
        return
    
    if len(tokens) < 2:
        print("Error: Please specify KEY and VALUE")
        print("Usage: set [KEY] [VALUE]")
        print("Available keys:")
        print("  dataset: imdb, books, fb, osm, wiki")
        print("  drift: 0.0-1.0")
        print("Examples:")
        print("  set dataset books")
        print("  set drift 0.5")
        return
    
    key = tokens[0].lower()
    value = tokens[1]
    
    # Validate and set configuration
    if key == "dataset":
        valid_datasets = ["imdb", "books", "fb", "osm", "wiki"]
        if value.lower() not in valid_datasets:
            print(f"Error: Invalid dataset '{value}'")
            print(f"Available datasets: {', '.join(valid_datasets)}")
            return
        GLOBAL_CONFIG[key] = value.lower()
        print(f"✅ Set {key} = {value.lower()}")
        
    elif key == "drift":
        try:
            drift_float = float(value)
            if not 0.0 <= drift_float <= 1.0:
                print("Error: Drift must be between 0.0 and 1.0")
                return
            GLOBAL_CONFIG[key] = drift_float
            print(f"✅ Set {key} = {drift_float}")
        except ValueError:
            print("Error: Drift must be a valid number")
            return
            
    else:
        print(f"Error: Unknown configuration key '{key}'")
        print("Available keys: dataset, drift")
        return


def handle_idx(tokens: List[str]):
    """Handle learned index testing command"""
    tokens = tokens[1:]
    
    if len(tokens) < 1:
        print("Error: Please specify INDEX_NAME")
        print("Usage: idx [INDEX_NAME]")
        print("Available INDEX: alex, art, btree, pgm, xindex, finedex")
        print("Current global settings:")
        print(f"  Dataset: {GLOBAL_CONFIG['dataset']}")
        print(f"  Drift: {GLOBAL_CONFIG['drift']}")
        print("Use 'set' command to change global settings")
        print("Example: set drift 0.5")
        return
    
    index_name = tokens[0].lower()
    
    # Use global configuration
    dataset = GLOBAL_CONFIG["dataset"]
    drift = GLOBAL_CONFIG["drift"]
    
    # Set default values for LIDX-specific parameters
    size = "4M"  # Default size for LIDX
    operations = 1000000  # Default operations
    threads = 4  # Default threads
    
    print("Testing learned index...")
    print_args(
        index_name=index_name,
        drift=drift,
        dataset=dataset,
        size=size,
        operations=operations,
        threads=threads
    )
    
    # Check if lidx directory exists
    lidx_dir = os.path.join("benchmarks", "lidx")
    if not os.path.exists(lidx_dir):
        print(f"Error: LIDX directory not found at {lidx_dir}")
        return
    
    # Check if benchmark script exists
    benchmark_script = os.path.join(lidx_dir, "run_lidx_benchmark.py")
    if not os.path.exists(benchmark_script):
        print(f"Error: LIDX benchmark script not found at {benchmark_script}")
        return
    
    # Run the benchmark script with global configuration
    print("Starting LIDX benchmark...")
    cmd = f"cd {lidx_dir} && python run_lidx_benchmark.py --drift {drift} --dataset {dataset} --size {size} --index {index_name} --operations {operations} --threads {threads} --verbose"
    result = os.system(cmd)
    
    if result == 0:
        print("✅ LIDX benchmark completed successfully!")
    else:
        print(f"❌ LIDX benchmark failed with exit code {result}")


def handle_lcc(tokens: List[str]):
    """Handle learned concurrency control command - test Polyjuice"""
    tokens = tokens[1:]
    
    if len(tokens) > 0:
        print("Error: LCC command takes no parameters")
        print("Usage: lcc")
        print("This will test Polyjuice using training scripts")
        return
    
    print("Testing Polyjuice (Learned Concurrency Control)...")
    print("Current global settings:")
    print(f"  Dataset: {GLOBAL_CONFIG['dataset']}")
    print(f"  Drift: {GLOBAL_CONFIG['drift']}")
    
    # Check if lcc directory exists
    lcc_dir = os.path.join("benchmarks", "lcc")
    if not os.path.exists(lcc_dir):
        print(f"Error: LCC directory not found at {lcc_dir}")
        return
    
    # Check if training directory exists
    training_dir = os.path.join(lcc_dir, "training")
    if not os.path.exists(training_dir):
        print(f"Error: LCC training directory not found at {training_dir}")
        return
    
    # Check if ERL training script exists
    erl_script = os.path.join(training_dir, "ERL_main.py")
    if not os.path.exists(erl_script):
        print(f"Error: ERL training script not found at {erl_script}")
        return
    
    # Use global configuration for scale factor (convert drift to scale)
    scale_factor = max(1, int(GLOBAL_CONFIG["drift"] * 10))
    
    print("Starting Polyjuice test with ERL training...")
    print(f"Scale factor: {scale_factor}")
    
    # Run the ERL training script with minimal parameters
    cmd = f"cd {training_dir} && python ERL_main.py --workload-type tpcc --scale-factor {scale_factor} --nworkers 8 --eval-time 1.0 --max-iterations 10 --samples-per-distribution 8 --psize 4"
    result = os.system(cmd)
    
    if result == 0:
        print("✅ Polyjuice test completed successfully!")
    else:
        print(f"❌ Polyjuice test failed with exit code {result}")
        print("Note: This is normal if LCC is not built yet.")
        print("To build LCC, run: cd benchmarks/lcc && MODE=perf make -j dbtest")


def main():
    session = PromptSession(
        history=FileHistory(os.path.join(os.path.dirname(__file__), ".cli_history"))
    )

    while True:
        try:
            text = session.prompt("[NRBench]> ")
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

            if main_command == "tqo":
                # Train learned query optimizer command
                handle_tqo(tokens)
                continue

            if main_command == "iqo":
                # Inference learned query optimizer command
                handle_iqo(tokens)
                continue

            if main_command == "set":
                # Set global configuration command
                handle_set(tokens)
                continue
                
            if main_command == "idx":
                # Test learned index command
                handle_idx(tokens)
                continue
                
            if main_command == "lcc":
                # Test learned concurrency control command
                handle_lcc(tokens)
                continue

            print("Unknown command:", text)


if __name__ == "__main__":
    main()
