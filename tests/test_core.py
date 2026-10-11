import numpy as np
import pytest

from abc_algorithm import CanonicalABCSolver


def sphere(x):
    return float(np.sum(x ** 2))


def rastrigin(x):
    return float(10 * len(x) + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x)))


def make(func=sphere, d=3, **kw):
    kw.setdefault("max_evals", 2000)
    kw.setdefault("random_state", 0)
    return CanonicalABCSolver(func, d, [-5.0] * d, [5.0] * d, **kw)


def test_sphere_converges_near_zero():
    _, cost, _ = make().solve()
    assert cost < 1e-3


def test_rastrigin_improves_over_random_start():
    solver = make(rastrigin, d=2, max_evals=4000)
    _, cost, history = solver.solve()
    assert cost < history[0][1]
    assert cost < 1.0


def test_same_seed_gives_identical_results():
    p1, c1, h1 = make(random_state=42).solve()
    p2, c2, h2 = make(random_state=42).solve()
    np.testing.assert_array_equal(p1, p2)
    assert c1 == c2 and h1 == h2


def test_different_seeds_differ():
    p1, _, _ = make(random_state=1).solve()
    p2, _, _ = make(random_state=2).solve()
    assert not np.array_equal(p1, p2)


def test_respects_evaluation_budget():
    calls = []

    def counted(x):
        calls.append(1)
        return sphere(x)

    make(counted, max_evals=137).solve()
    assert len(calls) == 137


def test_solutions_stay_within_bounds():
    seen = []

    def recorder(x):
        seen.append(np.copy(x))
        return sphere(x)

    CanonicalABCSolver(recorder, 2, [1, -2], [3, 0], max_evals=500, random_state=0).solve()
    seen = np.array(seen)
    assert np.all(seen >= [1, -2]) and np.all(seen <= [3, 0])


def test_history_is_monotonic_non_increasing():
    _, _, history = make().solve()
    costs = [c for _, c in history]
    assert all(b <= a for a, b in zip(costs, costs[1:]))


def test_solve_twice_resets_history():
    solver = make(max_evals=300)
    _, _, h1 = solver.solve()
    _, _, h2 = solver.solve()
    assert h1 == h2


def test_negative_costs_are_minimized():
    # minimum of -(x-1)^2 - 5 ... use f(x) = (x-1)^2 - 5, minimum -5 at x=1
    f = lambda x: float(np.sum((x - 1) ** 2) - 5)
    params, cost, _ = make(f, d=2).solve()
    assert cost < -4.99
    np.testing.assert_allclose(params, [1, 1], atol=0.05)


def test_nan_objective_is_treated_as_worst():
    f = lambda x: float("nan") if x[0] > 0 else sphere(x)
    params, cost, _ = make(f, d=2).solve()
    assert np.isfinite(cost) and params[0] <= 0


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(lb=[0, 0], ub=[1]), "length"),
        (dict(lb=[1, 1], ub=[0, 0]), "lower bound"),
        (dict(pop_size=1), "pop_size"),
        (dict(max_evals=5, pop_size=10), "max_evals"),
    ],
)
def test_invalid_arguments_raise(kwargs, match):
    base = dict(objective_func=sphere, n_params=2, lb=[0, 0], ub=[1, 1])
    base.update(kwargs)
    with pytest.raises(ValueError, match=match):
        CanonicalABCSolver(**base)
