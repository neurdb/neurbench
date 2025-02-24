from typing import List
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import jensenshannon
from .dist import zipf_dist


def _objective(q, p, alpha):
    # alphaL: target JS divergence between original dist p and target dist q
    return (jensenshannon(p, q) - alpha) ** 2


def _constraint(q):
    return np.sum(q) - 1.0


def find_q(p: List[float], d: float, skewed: bool = True) -> List[float]:
    # find a distribution q which is as close as p (measured by JS divergence)
    n = len(p)
    if skewed:
        # initial guess for the distribution
        q0 = zipf_dist(n, 1.0)
    else:
        q0 = np.ones(n) / n

    # sum of probabilities must equal 1
    cons = {"type": "eq", "fun": _constraint}
    # each pribability is from 0 to 1
    bounds = [(0, 1) for _ in range(n)]

    result = minimize(
        _objective,
        q0,
        args=(p, d),
        method="L-BFGS-B",
        constraints=cons,
        bounds=bounds,
        tol=1e-6,
    )

    return result.x / sum(result.x)


if __name__ == "__main__":
    p = [0.1, 0.2, 0.3, 0.4]
    print(f"p={p}")
    q = find_q(p, 0.2)
    print(f"q={q}")
    print(f"JS divergence={jensenshannon(p, q)}")
