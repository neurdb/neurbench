from typing import List, Union

import bm25s
import pglast

from .metadata import SQLInfoExtractor


def make_query_features_map(queries: List[str]):
    result = {}

    for text in queries:
        sqls = text.split(";")
        sqls = [s.strip() for s in sqls if s.strip()]

        for s in sqls:
            node = pglast.parse_sql(s)
            extractor = SQLInfoExtractor()
            extractor(node)

            info = extractor.info
            result[text] = _make_printable_info(_filter(info))

    return result


def _filter(info):
    if "aliasname_fullname" in info:
        del info["aliasname_fullname"]

    return info


def _make_printable_info(info):
    result = {}

    for k in info.keys():
        result[k] = (
            str(info[k])
            .replace("'", "")
            .replace(", ", " AND ")
            .replace("(", "[")
            .replace(")", "]")
            .replace("{", "[")
            .replace("}", "]")
        )

    return result


def tokenize(feature_value: Union[str, List[str]]):
    return bm25s.tokenize(feature_value, stopwords=["AND", "[", "]"])

def build_feature_retriever(feature_values: List[str]):
    corpus_tokens = tokenize(feature_values)
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)

    return retriever