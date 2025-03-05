import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from neurbench.index.util import load_key_set

key_set_load_test_input = [
    ("uint32_1.bin", np.array(range(10), dtype=np.uint32)),
    ("uint32_2.bin", np.array(range(10, 20, 2), dtype=np.uint32)),
    ("uint64_1.bin", np.array(range(20, 30, 3), dtype=np.uint64)),
    ("uint64_2.bin", np.array(range(30, 40, 4), dtype=np.uint64)),
]


@pytest.mark.parametrize(
    "file_path, data_array",
    key_set_load_test_input
)
@pytest.mark.index
def test_load_key_set(file_path: str, data_array: np.ndarray):
    # create a file
    data_array.tofile(file_path)

    assert Path(file_path).exists()

    key_set = load_key_set(file_path)

    assert key_set is not None
    assert np.array_equal(key_set, data_array)
    assert key_set.dtype == data_array.dtype

    os.remove(file_path)

@pytest.mark.index
def test_load_key_set_file_not_exist():
    fake_file_path = "test_uint32.bin"
    with patch("os.path.exists", return_value=False):
        key_set = load_key_set(fake_file_path)
        assert key_set is None


real_file_paths = [
    "/users/lingze/TLI/data/fb_200M_uint64",
    # "/users/lingze/TLI/data/books_800M_uint64",
    "/users/lingze/TLI/data/lognormal_200M_uint32",
]

@pytest.mark.skip(reason="To test the real data file, we need to download the data file from the website.")
@pytest.mark.parametrize("filepath", real_file_paths)
@pytest.mark.index
def test_real_key_set_load(filepath: str):
    data = load_key_set(filepath)
    assert data is not None
    del data


if __name__ == "__main__":
    pytest.main()
