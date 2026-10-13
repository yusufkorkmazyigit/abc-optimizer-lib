"""
abcopt must reproduce the official ABC implementation exactly.

Both implementations receive the same stream of U(0, 1) numbers. If they follow the
same algorithm and draw random numbers in the same order, they evaluate exactly the
same sequence of points and return the same result.
"""
import numpy as np
import pytest

from abcopt import CanonicalABCSolver
from reference_abc import ReferenceABC


class SharedStream:
    """A reproducible U(0, 1) stream that can be replayed for a second consumer."""

    def __init__(self, seed):
        self.values = np.random.default_rng(seed).random(1_000_000)
        self.pos = 0

    def random(self):
        v = self.values[self.pos]
        self.pos += 1
        return v

    __call__ = random


def sphere(x):
    return float(np.sum(x ** 2))


def rastrigin(x):
    return float(10 * len(x) + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x)))


def shifted(x):  # takes negative values, exercising the 1 + |f| fitness branch
    return float(np.sum((x - 1.5) ** 2) - 3.0)


def recorder(f, log):
    def wrapped(x):
        log.append(np.copy(x))
        return f(x)
    return wrapped


@pytest.mark.parametrize("func,bound", [(sphere, 5.12), (rastrigin, 5.12), (shifted, 4.0)])
@pytest.mark.parametrize("dim,sn,limit,budget", [
    (2, 5, 4, 300),      # small limit: many scout restarts
    (5, 10, 50, 1000),   # limit = SN * D
    (10, 20, 30, 2003),  # budget not a multiple of anything: stops mid-phase
])
@pytest.mark.parametrize("seed", [0, 1])
def test_identical_trajectory_to_reference(func, bound, dim, sn, limit, budget, seed):
    ref_log, our_log = [], []

    ref_stream = SharedStream(seed)
    ref = ReferenceABC(recorder(func, ref_log), dim, -bound, bound, sn, budget, limit, ref_stream)
    ref_params, ref_cost = ref.run()

    our_stream = SharedStream(seed)
    ours = CanonicalABCSolver(recorder(func, our_log), dim, [-bound] * dim, [bound] * dim,
                              pop_size=sn, max_evals=budget, limit=limit, random_state=0)
    ours._rng_override = our_stream
    params, cost, _ = ours.solve()

    assert len(our_log) == len(ref_log) == budget
    np.testing.assert_array_equal(np.array(our_log), np.array(ref_log))
    assert our_stream.pos == ref_stream.pos
    assert cost == ref_cost
    np.testing.assert_array_equal(params, ref_params)
    np.testing.assert_array_equal(ours.trial, ref.trial)
    np.testing.assert_array_equal(ours.foods, ref.foods)
