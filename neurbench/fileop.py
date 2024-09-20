import pandas as pd


def dump_tbl(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False, header=False, sep="|")
