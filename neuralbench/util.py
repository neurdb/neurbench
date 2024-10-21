from typing import List, Union


def formatted_list(l: List[float]) -> str:
    return "[" + " ".join("{:.5f}".format(f) for f in l) + "]"


def tuple_to_list(t: List[Union[str, tuple]]) -> List[Union[str, list]]:
    return [list(tup) if isinstance(tup, tuple) else tup for tup in t]
