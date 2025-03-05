import enum
import struct
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

    # key_set = np.fromfile(filepath, dtype=np_type)
    try:
        # Read the file as binary
        with open(filepath, 'rb') as file:
            # Read the first 8 bytes (uint64) as the size of the key set
            key_set_size = np.frombuffer(file.read(8), dtype=np.uint64)[0]

            # Read the remaining data based on the size and type of the key set
            key_set = np.frombuffer(file.read(), dtype=np_type)

            # Ensure the key set size matches the actual data length
            if len(key_set) != key_set_size:
                print(
                    f"Key set size mismatch: expected {key_set_size}, got {len(key_set)}.")
                return None

    except Exception as e:
        print(f"Error reading the key set: {e}")
        return None

    return key_set


def save_file(filepath: Union[str, Path], data: np.ndarray, dtype: KeyType) -> None:
    """
    Save the data into binary file according to specific format
    @param filepath: the path to save the file
    @param data: the data to save
    @param dtype: the data type
    """

    assert data.ndim == 1
    # first sort the data
    data = np.sort(data)

    if isinstance(filepath, str):
        filepath = Path(filepath)

    data_np_type = dtype.to_numpy_type()
    data = np.ndarray.astype(data, data_np_type)

    with open(filepath, "wb") as f:
        size = data.size
        f.write(struct.pack("Q", size))
        f.write(data.tobytes())
