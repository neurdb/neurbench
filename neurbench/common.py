import json
import os
import time
from abc import ABCMeta, abstractmethod
from typing import Any, Optional, Tuple

from . import fileop


class Processor(metaclass=ABCMeta):
    def load_from_file(self, input_file: str):
        pass

    @abstractmethod
    def load(self, input_path: str):
        raise NotImplementedError

    @abstractmethod
    def compute_dists(self):
        raise NotImplementedError

    @abstractmethod
    def apply_drift(self, drift: float, n_samples: Optional[int]):
        raise NotImplementedError

    @abstractmethod
    def save(self, output_path: str):
        raise NotImplementedError

    @property
    @abstractmethod
    def config(self):
        raise NotImplementedError


def load_config(
        config_path: str, default: Optional[Any] = None
) -> Tuple[dict, Optional[str]]:
    if not os.path.exists(config_path):
        return default if default is not None else {}, "no such file: " + config_path

    try:
        with open(config_path, "r") as f:
            return json.loads(f.read()), None
    except:
        return default if default is not None else {}, "invalid config. ignoring"

    print("config loaded from: ", config_path)


def dump_config(config: dict, config_path: str):
    fileop.dump_json(config, config_path)


def make_drift(
        processor: Processor,
        input_file: str,
        input_path: str,
        output_path: str,
        config_path: str,
        drift: float,
        n_samples: Optional[int] = None,
):
    start_time = time.time()

    if input_path != "":
        processor.load(input_path)
    elif input_file != "":
        print(input_file)
        processor.load_from_file(input_file)
    else:
        raise ValueError("no file or folder provided!")
    
    print(f"[Profile] Load data: {time.time() - start_time}")
    
    start_time = time.time()
    processor.compute_dists()
    print(f"[Profile] Compute dists: {time.time() - start_time}")

    dump_config(processor.config, config_path)
    print(f"Table config dumped to {config_path}")

    start_time = time.time()
    processor.apply_drift(drift, n_samples)
    print(f"[Profile] Apply drift: {time.time() - start_time}")
    
    print(f"Processed data with drift factor {drift}")

    start_time = time.time()
    processor.save(output_path)
    print(f"[Profile] Save data: {time.time() - start_time}")
    
    print(f"Data saved to {output_path}")
