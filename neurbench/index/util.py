import enum
import numpy as np

from pathlib import Path
from typing import Optional, Union


class KeyType(enum.Enum):
    UINT32 = "uint32"
    UINT64 = "uint64"

    def to_numpy_type(self) -> Optional[np.dtype]:
        if self == KeyType.UINT32:
            return np.uint32
        elif self == KeyType.UINT64:
            return np.uint64
        else:
            return None

    @staticmethod
    def resolve_type_from_filename(filename: str) -> Optional["KeyType"]:
        if "uint32" in filename:
            return KeyType.UINT32
        elif "uint64" in filename:
            return KeyType.UINT64
        else:
            raise ValueError(f"Unknown key type in file {filename}.")

    @staticmethod
    def resolve_type(type: str) -> Optional["KeyType"]:
        if type == "uint32":
            return KeyType.UINT32
        elif type == "uint64":
            return KeyType.UINT64
        else:
            raise ValueError("Unknown key type.")


def load_key_set(filepath: Union[str, Path]) -> Optional[np.ndarray]:
    """
    Load the numerical key set from a binary file.
    """
    if isinstance(filepath, str):
        filepath = Path(filepath)

    if not filepath.exists():
        print(f"File {filepath} does not exist.")
        return None

    try:
        data_type = KeyType.resolve_type_from_filename(filepath.name)
    except ValueError as e:
        print(e)
        return None

    key_set = None
    np_type = data_type.to_numpy_type()

    key_set = np.fromfile(filepath, dtype=np_type)

    return key_set
