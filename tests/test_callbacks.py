import numpy as np
import pytest
from sklearn.base import clone
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

from abcopt import ABCSearchCV, CanonicalABCSolver, CycleState, EarlyStopping

X, y = load_iris(return_X_y=True)


def sphere(x):
    return float(np.sum(x ** 2))


def make(callbacks=None, **kw):
    kw.setdefault("max_evals", 2000)
    kw.setdefault("random_state", 0)
    return CanonicalABCSolver(sphere, 3, [-5.0] * 3, [5.0] * 3, callbacks=callbacks, **kw)


def search(**kw):
    kw.setdefault("estimator", DecisionTreeClassifier(random_state=0))
    kw.setdefault("param_space", {"max_depth": {"type": "int", "range": (1, 8)}})
    kw.setdefault("max_evals", 60)
    kw.setdefault("pop_size", 6)
    kw.setdefault("random_state", 0)
    return ABCSearchCV(**kw)


# ---------------------------------------------------------------------- #
# CanonicalABCSolver
# ---------------------------------------------------------------------- #
def test_passive_callbacks_do_not_change_the_run():
    states = []
    p0, c0, h0 = make().solve()
    solver = make(callbacks=[states.append, lambda s: None, lambda s: False])
    p1, c1, h1 = solver.solve()
    np.testing.assert_array_equal(p0, p1)
    assert c0 == c1 and h0 == h1
    assert not solver.stopped_early
    assert solver.n_cycles == len(states) == len(h1) - 1


def test_callback_receives_cycle_state():
    states = []
    solver = make(callbacks=states.append)  # a single callable is accepted
    _, cost, history = solver.solve()
    assert all(isinstance(s, CycleState) for s in states)
    assert [s.cycle for s in states] == list(range(1, len(states) + 1))
    assert [(s.n_evals, s.best_cost) for s in states] == history[1:]
    assert states[-1].max_evals == 2000
    assert states[-1].best_score is None
    assert sphere(states[-1].best_params) == cost


def test_state_params_are_a_copy():
    def tamper(state):
        state.best_params[:] = 99.0

    p0, c0, _ = make().solve()
    p1, c1, _ = make(callbacks=tamper).solve()
    np.testing.assert_array_equal(p0, p1)
    assert c0 == c1


def test_returning_true_stops_after_that_cycle():
    solver = make(callbacks=lambda s: s.cycle == 5)
    _, cost, history = solver.solve()
    assert solver.n_cycles == 5 and solver.stopped_early
    assert len(history) == 6
    assert history[-1][0] < 2000
    # the stopped run is a prefix of the full run
    _, _, full = make().solve()
    assert history == full[:6]
    assert cost == history[-1][1]


def test_every_callback_is_called_even_after_a_stop_request():
    seen = []
    solver = make(callbacks=[lambda s: True, lambda s: seen.append(s.cycle)])
    solver.solve()
    assert seen == [1] and solver.n_cycles == 1


def test_stop_on_the_last_cycle_is_not_early():
    solver = make(max_evals=200, callbacks=lambda s: s.n_evals >= s.max_evals)
    solver.solve()
    assert solver._evals == 200 and not solver.stopped_early


def test_solve_twice_resets_cycle_counters():
    solver = make(callbacks=lambda s: s.cycle == 3)
    solver.solve()
    solver.solve()
    assert solver.n_cycles == 3 and len(solver.history) == 4


@pytest.mark.parametrize("bad", [1, "x", [print, 3]])
def test_invalid_callbacks_raise(bad):
    with pytest.raises(TypeError):
        make(callbacks=bad)


# ---------------------------------------------------------------------- #
# EarlyStopping
# ---------------------------------------------------------------------- #
def flat(x):
    return 1.0  # never improves after the first cycle


def test_early_stopping_stops_after_patience_cycles():
    es = EarlyStopping(patience=4)
    solver = CanonicalABCSolver(flat, 2, [-1, -1], [1, 1], max_evals=5000,
                                random_state=0, callbacks=es)
    solver.solve()
    # cycle 1 sets the reference, cycles 2..5 do not improve
    assert solver.n_cycles == 5 and es.stopped_cycle == 5
    assert solver.stopped_early


def test_early_stopping_min_delta():
    def states(costs):
        return [CycleState(cycle=i + 1, n_evals=0, max_evals=0, best_cost=c, best_params=None)
                for i, c in enumerate(costs)]

    es = EarlyStopping(patience=2, min_delta=0.1)
    # improvements of 0.05 do not count, so it stops at cycle 3
    assert [es(s) for s in states([1.0, 0.95, 0.9])] == [False, False, True]
    es = EarlyStopping(patience=2, min_delta=0.1)
    assert [es(s) for s in states([1.0, 0.8, 0.6, 0.6, 0.6])] == [False, False, False, False, True]


def test_early_stopping_is_reusable_across_runs():
    es = EarlyStopping(patience=3)
    for _ in range(2):
        solver = CanonicalABCSolver(flat, 2, [-1, -1], [1, 1], max_evals=5000,
                                    random_state=0, callbacks=es)
        solver.solve()
        assert solver.n_cycles == 4 and es.stopped_cycle == 4


def test_early_stopping_never_triggers_while_improving():
    es = EarlyStopping(patience=1000)
    solver = make(callbacks=es)
    solver.solve()
    assert es.stopped_cycle is None and not solver.stopped_early


@pytest.mark.parametrize("kw", [{"patience": 0}, {"min_delta": -1}])
def test_early_stopping_validates(kw):
    with pytest.raises(ValueError):
        EarlyStopping(**kw)


# ---------------------------------------------------------------------- #
# ABCSearchCV
# ---------------------------------------------------------------------- #
def test_search_without_stop_is_unchanged():
    states = []
    a = search().fit(X, y)
    b = search(callbacks=states.append).fit(X, y)
    assert a.best_params_ == b.best_params_ and a.best_score_ == b.best_score_
    assert a.history_ == b.history_ and a.results_ == b.results_
    assert not b.stopped_early_ and b.n_cycles_ == len(states)


def test_search_state_is_in_score_terms():
    states = []
    s = search(callbacks=states.append).fit(X, y)
    last = states[-1]
    assert isinstance(last.best_params, dict) and set(last.best_params) == {"max_depth"}
    assert last.best_score == -last.best_cost
    assert last.best_params == s.best_params_ and last.best_score == s.best_score_
    assert [(st.n_evals, st.best_score) for st in states] == s.history_[1:]


def test_search_early_stopping_refits_best_so_far():
    s = search(max_evals=500, callbacks=EarlyStopping(patience=3)).fit(X, y)
    assert s.stopped_early_
    assert s.n_evals_ < 500 and len(s.results_) == s.n_evals_
    assert s.best_score_ == max(r["score"] for r in s.results_)
    assert s.best_params_ == s.cv_results_["params"][s.best_index_]
    assert s.predict(X).shape == y.shape


def test_search_clone_keeps_callbacks():
    es = EarlyStopping(patience=5)
    s = clone(search(callbacks=[es]))
    assert s.get_params()["callbacks"][0].patience == 5
