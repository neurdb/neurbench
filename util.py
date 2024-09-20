from typing import List


def formatted_list(l: List[float]) -> str:
    return "[" + ' '.join('{:.5f}'.format(f) for f in l) + "]"