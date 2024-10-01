import json
import numpy as np
import cudf as pd


class NumpyEncoder(json.JSONEncoder):
    """Special json encoder for numpy types"""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def dump_tbl(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False, header=False, sep="|")


def dump_json(obj, path):
    with open(path, "w") as f:
        f.write(json.dumps(obj, indent=4, cls=NumpyEncoder))
