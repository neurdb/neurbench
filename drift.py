from typing import List
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import jensenshannon
from dist import zipf_dist


def _objective(q, p, alpha):
    return (jensenshannon(p, q) - alpha) ** 2


def _constraint(q):
    return np.sum(q) - 1.0


def find_q(p: List[float], d: float) -> List[float]:
    n = len(p)
    if max(p) > 0.9:
        q0 = np.ones(n) / n
    else:
        q0 = zipf_dist(n, 5.0)

    cons = {"type": "eq", "fun": _constraint}
    bounds = [(0, 1) for _ in range(n)]

    result = minimize(
        _objective, q0, args=(p, d), method="SLSQP", constraints=cons, bounds=bounds
    )

    return result.x


if __name__ == "__main__":
    p = [0.1, 0.2, 0.3, 0.4]
    print(f"p={p}")
    q = find_q(p, 0.2)
    print(f"q={q}")
    print(f"JS divergence={jensenshannon(p, q)}")
