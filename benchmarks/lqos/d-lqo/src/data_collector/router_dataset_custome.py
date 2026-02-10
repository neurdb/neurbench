import os
import re
import json
import pandas as pd
from typing import Dict, List, Callable, Optional
import glob


def _normalize_query_ident(q: str) -> str:
    s = str(q).strip()
    m = re.match(r"^(\d+)([A-Za-z].*)$", s)
    if not m:
        return s
    return m.group(1).zfill(2) + m.group(2).lower()


def _safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s)).replace("-", "_")


def _build_system_name_map_from_csv(df: pd.DataFrame, system_col: str) -> Dict[str, str]:
    raw = df[system_col].astype(str).str.strip()
    lower = raw.str.lower()
    tmp = pd.DataFrame({"lower": lower, "raw": raw})
    tmp = tmp[tmp["lower"].notna() & (tmp["lower"] != "")]
    canon = tmp.groupby("lower", as_index=False)["raw"].first()
    return dict(zip(canon["lower"], canon["raw"]))


def generate_workload_csvs_from_query_times(
    query_times_csv: str,
    out_dir: str,
    test_queries_dict: Optional[Dict[str, List[str]]] = None,
    load_sql_fn: Optional[Callable[[str], str]] = None,
    drift_col: str = "drift",
    system_col: str = "system_name",
    query_col: str = "query_name",
    time_col: str = "execution_time",
    agg: str = "mean",
) -> List[str]:
    """
    特殊case：train == test == 全部SQL
    => 每个 drift 只输出一个文件（沿用旧命名 train）：

      out_dir/<drift>/workload_<workload>_train_data.csv

    不再生成 test 文件。
    """
    if test_queries_dict is None:
        test_queries_dict = {"all_train": []}

    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(query_times_csv)

    for c in [drift_col, system_col, query_col, time_col]:
        if c not in df.columns:
            raise ValueError(f"query_times.csv 缺少列 '{c}'，当前列：{df.columns.tolist()}")

    system_name_map = _build_system_name_map_from_csv(df, system_col)

    df["query_ident"] = df[query_col].apply(_normalize_query_ident)

    sys_raw = df[system_col].astype(str).str.strip()
    sys_lower = sys_raw.str.lower()
    df["method"] = sys_lower.map(system_name_map)
    df.loc[df["method"].isna(), "method"] = sys_raw[df["method"].isna()]

    agg_func = {"mean": "mean", "median": "median", "min": "min", "max": "max"}.get(agg, "mean")
    g = (
        df.groupby([drift_col, "query_ident", "method"], dropna=False)[time_col]
          .agg(agg_func)
          .reset_index()
    )

    created: List[str] = []

    for workload in test_queries_dict.keys():
        wl_safe = _safe_name(workload)

        for drift in sorted(g[drift_col].astype(str).unique()):
            drift_safe = _safe_name(drift)
            if not drift_safe.endswith("_gen"):
                drift_safe = drift_safe + "_gen"

            drift_dir = os.path.join(out_dir, drift_safe)
            os.makedirs(drift_dir, exist_ok=True)

            # ✅ 只输出一个文件：train（包含全部query）
            f_out = os.path.join(drift_dir, f"workload_{wl_safe}_train_data.csv")
            if os.path.isfile(f_out):
                print(f"[SKIP] drift={drift_safe} workload={workload} -> already exists")
                print(f"     {f_out}")
                created.append(f_out)
                continue

            gd = g[g[drift_col].astype(str) == str(drift)].copy()

            rows = []
            for q in sorted(gd["query_ident"].unique()):
                sub = gd[gd["query_ident"] == q].sort_values(time_col, ascending=True)
                methods = sub["method"].tolist()
                times = sub[time_col].tolist()
                if not methods:
                    continue

                exec_dict = {m: float(t) for m, t in zip(methods, times)}
                sql = load_sql_fn(q) if load_sql_fn is not None else ""

                rows.append({
                    "query_ident": q,
                    "top_1_method": methods[0],
                    "top_2_methods": str(methods[:2]),
                    "execution_time_ms": json.dumps(exec_dict),
                    "query_sql": sql,
                })

            out_all = pd.DataFrame(rows)
            out_all = out_all[["query_ident", "top_1_method", "top_2_methods", "execution_time_ms", "query_sql"]]
            out_all = out_all.sort_values("query_ident")

            out_all.to_csv(f_out, index=False)

            created.append(f_out)
            print(f"[OK] drift={drift_safe} workload={workload} -> saved {len(out_all)} rows")
            print(f"     {f_out}")

    return created

def load_workload_train_test_datasets(folder_name: str) -> Dict[str, Dict[str, pd.DataFrame]]:
    datasets = {}

    # Find all workload train/test files
    train_files = glob.glob(os.path.join(folder_name, "workload_*_train_data.csv"))
    if len(train_files) == 0:
        raise FileNotFoundError(
            f"No workload train data files found in '{folder_name}'. "
            "Generate data first by running this module with __main__ or generate_workload_csvs_from_query_times()."
        )
    for train_file in train_files:
        # Extract workload name from filename
        workload = os.path.basename(train_file).replace("workload_", "").replace("_train_data.csv", "")

        datasets[workload] = {
            'train': pd.read_csv(train_file),
            'test': pd.read_csv(train_file)
        }
        for split in ['train', 'test']:
            datasets[workload][split]['execution_time_ms'] = datasets[workload][split]['execution_time_ms'].apply(
                json.loads)

    return datasets



if __name__ == "__main__":
    from common import get_config
    cfg = get_config("imdb")

    generate_workload_csvs_from_query_times(
        query_times_csv="./datasets/query_times.csv",
        out_dir="./experiment/datasets/workload_data_train_test/",
        test_queries_dict=None,          # 这里 None 就会用 all_train
        load_sql_fn=cfg.load_sql_query,
        drift_col="drift",
        system_col="system_name",
        query_col="query_name",
        time_col="execution_time",
        agg="mean",
    )
