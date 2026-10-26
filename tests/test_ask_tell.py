import numpy as np
import pytest

from abcopt import CanonicalABCSolver, EarlyStopping


def sphere(x):
    return float(np.sum(x ** 2))


def rastrigin(x):
    return float(10 * len(x) + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x)))


def shifted(x):  # negative costs exercise the 1 + |f| fitness branch
    return float(np.sum((x - 1.5) ** 2) - 3.0)


def make(func=None, d=3, **kw):
    kw.setdefault("max_evals", 600)
    kw.setdefault("random_state", 0)
    return CanonicalABCSolver(func, d, [-5.0] * d, [5.0] * d, **kw)


def run_ask_tell(solver, func):
    points = []
    while not solver.done:
        x = solver.ask()
        points.append(x)
        solver.tell(x, func(x))
    return points


def recorder(func, log):
    def wrapped(x):
        log.append(np.copy(x))
        return func(x)
    return wrapped


# ---------------------------------------------------------------------- #
# Same trajectory as solve()
# ---------------------------------------------------------------------- #
@pytest.mark.parametrize("func", [sphere, rastrigin, shifted])
@pytest.mark.parametrize("d,sn,limit,budget", [
    (2, 5, 4, 300),      # small limit: many scout restarts
    (5, 10, None, 1000),
    (10, 20, 30, 2003),  # budget stops mid-phase
])
@pytest.mark.parametrize("seed", [0, 7])
def test_ask_tell_matches_solve(func, d, sn, limit, budget, seed):
    kw = dict(d=d, pop_size=sn, limit=limit, max_evals=budget, random_state=seed)

    solve_log = []
    ref = make(recorder(func, solve_log), **kw)
    p0, c0, h0 = ref.solve()

    solver = make(**kw)
    points = run_ask_tell(solver, func)

    assert len(points) == len(solve_log) == budget == solver.n_evals
    np.testing.assert_array_equal(np.array(points), np.array(solve_log))
    np.testing.assert_array_equal(solver.best_params, p0)
    assert solver.best_cost == c0 and solver.history == h0
    assert solver.n_cycles == ref.n_cycles


def test_matches_solve_with_a_generator_seed():
    p0, c0, h0 = make(sphere, random_state=np.random.default_rng(3)).solve()
    solver = make(random_state=np.random.default_rng(3))
    run_ask_tell(solver, sphere)
    np.testing.assert_array_equal(solver.best_params, p0)
    assert solver.best_cost == c0 and solver.history == h0


# ---------------------------------------------------------------------- #
# Protocol
# ---------------------------------------------------------------------- #
def test_ask_repeats_the_pending_candidate():
    solver = make()
    a = solver.ask()
    b = solver.ask()
    np.testing.assert_array_equal(a, b)
    assert solver.n_evals == 0
    solver.tell(a, sphere(a))
    assert solver.n_evals == 1
    assert not np.array_equal(solver.ask(), a)


def test_ask_returns_a_copy():
    solver = make()
    x = solver.ask()
    x[:] = 123.0
    y = solver.ask()
    assert not np.array_equal(x, y)
    solver.tell(y, sphere(y))  # the original point is still accepted


def test_tell_accepts_lists_and_numpy_scalars():
    solver = make()
    x = solver.ask()
    solver.tell(list(x), np.float32(sphere(x)))
    assert solver.n_evals == 1


def test_tell_without_ask_raises():
    with pytest.raises(RuntimeError, match="ask"):
        make().tell(np.zeros(3), 1.0)


def test_tell_with_another_point_raises():
    solver = make()
    x = solver.ask()
    with pytest.raises(ValueError, match="pending"):
        solver.tell(x + 1.0, 1.0)
    with pytest.raises(ValueError, match="pending"):
        solver.tell(x[:2], 1.0)
    solver.tell(x, sphere(x))  # the solver is unaffected by the rejected calls
    assert solver.n_evals == 1


@pytest.mark.parametrize("bad", [None, "abc", [1.0, 2.0]])
def test_tell_rejects_non_scalar_cost(bad):
    solver = make()
    x = solver.ask()
    with pytest.raises(TypeError, match="scalar"):
        solver.tell(x, bad)
    solver.tell(x, 0.0)


def test_nan_cost_is_worst():
    # every cost is NaN except the very last initial source
    solver = make(pop_size=4, max_evals=4)
    for i in range(4):
        x = solver.ask()
        solver.tell(x, float("nan") if i < 3 else 2.0)
    assert solver.done and solver.best_cost == 2.0


def test_done_and_ask_after_finish():
    solver = make(max_evals=50)
    assert not solver.done
    run_ask_tell(solver, sphere)
    assert solver.done and solver.n_evals == 50
    with pytest.raises(RuntimeError, match="finished"):
        solver.ask()
    with pytest.raises(RuntimeError, match="ask"):
        solver.tell(np.zeros(3), 0.0)


def test_reset_starts_an_identical_run():
    solver = make(max_evals=200)
    first = run_ask_tell(solver, sphere)
    solver.reset()
    assert not solver.done
    second = run_ask_tell(solver, sphere)
    np.testing.assert_array_equal(np.array(first), np.array(second))


def test_reset_discards_a_run_in_progress():
    solver = make(max_evals=200)
    for _ in range(30):
        x = solver.ask()
        solver.tell(x, sphere(x))
    solver.reset()
    run_ask_tell(solver, sphere)
    _, c0, h0 = make(sphere, max_evals=200).solve()
    assert solver.best_cost == c0 and solver.history == h0


def test_solve_discards_a_run_in_progress():
    solver = make(sphere, max_evals=200)
    x = solver.ask()
    solver.tell(x, 99.0)  # a wrong cost that must not leak into solve()
    _, c, h = solver.solve()
    _, c0, h0 = make(sphere, max_evals=200).solve()
    assert c == c0 and h == h0 and solver.done


def test_a_failed_evaluation_can_be_retried():
    solver = make(max_evals=100)
    x = solver.ask()
    # the evaluation of x crashes; nothing is reported and the caller asks again
    retry = solver.ask()
    np.testing.assert_array_equal(x, retry)
    solver.tell(retry, sphere(retry))
    run_ask_tell(solver, sphere)
    _, c0, _ = make(sphere, max_evals=100).solve()
    assert solver.best_cost == c0


def test_solve_without_objective_raises():
    with pytest.raises(TypeError, match="ask"):
        make().solve()


def test_objective_must_be_callable_or_none():
    with pytest.raises(TypeError):
        make(func=42)


def test_objective_may_modify_its_input():
    def destructive(x):
        cost = sphere(x)
        x[:] = 0.0
        return cost

    p0, c0, h0 = make(sphere).solve()
    p1, c1, h1 = make(destructive).solve()
    np.testing.assert_array_equal(p0, p1)
    assert c0 == c1 and h0 == h1


def test_exception_in_objective_propagates_and_solve_recovers():
    calls = {"n": 0}

    def flaky(x):
        calls["n"] += 1
        if calls["n"] == 10:
            raise RuntimeError("boom")
        return sphere(x)

    solver = make(flaky, max_evals=200)
    with pytest.raises(RuntimeError, match="boom"):
        solver.solve()
    _, c, h = solver.solve()  # a fresh run
    _, c0, h0 = make(sphere, max_evals=200).solve()
    assert c == c0 and h == h0


# ---------------------------------------------------------------------- #
# Callbacks in ask/tell mode
# ---------------------------------------------------------------------- #
def test_callbacks_run_in_ask_tell_mode():
    states = []
    solver = make(max_evals=300, callbacks=states.append)
    run_ask_tell(solver, sphere)
    assert [(s.n_evals, s.best_cost) for s in states] == solver.history[1:]


def test_callback_stop_ends_ask_tell_run():
    solver = make(max_evals=5000, callbacks=lambda s: s.cycle == 3)
    points = run_ask_tell(solver, sphere)
    assert solver.done and solver.stopped_early and solver.n_cycles == 3
    assert len(points) == solver.n_evals < 5000


def test_early_stopping_in_ask_tell_mode():
    solver = make(max_evals=5000, callbacks=EarlyStopping(patience=4))
    run_ask_tell(solver, lambda x: 1.0)
    assert solver.stopped_early and solver.n_cycles == 5
