import argparse
import json
import os
from typing import Dict, List, Tuple

import bm25s
import pandas as pd
import numpy as np

from neurbench.query import make_query_features_map, build_feature_retriever, tokenize


FEATURES = ["tables", "joins", "predicates"]


def make_qf_map() -> Tuple[List[str], dict]:
    fpaths = [f for f in os.listdir(args.input_dir) if f.endswith(".sql")]
    queries = []

    for fpath in fpaths:
        with open(os.path.join(args.input_dir, fpath)) as f:
            queries.append(f.read())

    qf_map = make_query_features_map(queries)
    # print(qf_map[list(qf_map.keys())[0]])

    return queries, qf_map


def main(args: argparse.Namespace):
    feature_retrievers: Dict[str, bm25s.BM25] = {}
    feature_corpora: Dict[str, List[str]] = {}

    if args.input_file:
        raise NotImplementedError("TODO: implement single .sql file input")

    if args.input_dir:
        queries, qf_map = make_qf_map()

    for f in FEATURES:
        feature_corpora[f] = [qf[f] for qf in qf_map.values()]
        feature_retrievers[f] = build_feature_retriever(feature_corpora[f])

    query_features = pd.read_csv(args.feature_table).to_dict(orient="records")

    for qf in query_features:
        total_scores = [0.0] * len(queries)

        for f in FEATURES:
            tokens = tokenize(qf[f])
            results, scores = feature_retrievers[f].retrieve(tokens, k=len(queries))

            num_ranks = results.shape[1]
            for i in range(num_ranks):
                f_id, norm_scores = results[0, i], scores[0, i] / max(scores[0])
                # print(
                #     f"Rank {i+1} (score: {norm_scores:.2f}): ID={f_id}, fv={feature_corpora[f][f_id]}"
                # )

                total_scores[f_id] += norm_scores

        total_scores = np.array(total_scores)
        ranks = np.argsort(total_scores, axis=0)[::-1]
        
        # sorted_values = total_scores[ranks]
        # print(sorted_values)

        print("original features: ", json.dumps(qf, indent=4))
        # print(total_scores)
        # print(ranks)
        print("Best matched queries:")
        for i in range(3):
            print("-" * 40)
            q_id = ranks[i]
            print(f"[Rank {i+1}] ID={q_id}, score={total_scores[q_id]:.2f}, query=")
            print(queries[q_id])
            print("features: ", json.dumps(qf_map[queries[q_id]], indent=4))
            
        print("=" * 40)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query Postprocessing")

    parser.add_argument(
        "-d", "--dbname", help="Database name (default: tpch)", required=True
    )

    parser.add_argument(
        "-i",
        "--input_file",
        default="",
        help="Path to file containing original queries (in .sql)",
    )
    parser.add_argument(
        "-I",
        "--input_dir",
        default="",
        help="Path to the directory of input files with original queries",
    )
    parser.add_argument(
        "-f",
        "--feature_table",
        required=True,
        help="Path to drifted feature table file (in .csv)",
    )

    args = parser.parse_args()

    if not args.input_file and not args.input_dir:
        raise ValueError("Need to specify either input_file or input_dir")

    for k, v in args.__dict__.items():
        if isinstance(v, str) and "{dbname}" in v:
            args.__dict__[k] = v.format(dbname=args.dbname)

    print(args)

    main(args)
