import math
import sys
import time
from pathlib import Path

import numpy as np
import pytest

# Add the src directory to the module search path so we can import simulator
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from simulator import _poisson_ppf  # type: ignore


def _naive_poisson_ppf(u: float, lam: float) -> int:
    k = 0
    p = math.exp(-lam)
    cdf = p
    while u > cdf:
        k += 1
        p *= lam / k
        cdf += p
    return k


def test_poisson_ppf_matches_scipy():
    scipy_stats = pytest.importorskip("scipy.stats")
    for lam in [1, 5, 20, 1000]:
        us = np.linspace(0.01, 0.99, 5)
        for u in us:
            assert _poisson_ppf(u, lam) == int(scipy_stats.poisson.ppf(u, lam))


def test_poisson_ppf_speed():
    pytest.importorskip("scipy.stats")
    lam = 1000.0
    us = np.random.default_rng(0).random(1000)

    start = time.time()
    for u in us:
        _poisson_ppf(u, lam)
    new_time = time.time() - start

    start = time.time()
    for u in us:
        _naive_poisson_ppf(u, lam)
    old_time = time.time() - start

    # SciPy-backed implementation should be faster than naive fallback
    assert new_time < old_time
