import warnings

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.datasets import load_iris, make_regression
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from abcopt import ABCSearchCV

X, y = load_iris(return_X_y=True)


def search(**kw):
    kw.setdefault("estimator", DecisionTreeClassifier(random_state=0))
    kw.setdefault("param_space", {"max_depth": {"type": "int", "range": (1, 8)}})
    kw.setdefault("max_evals", 30)
    kw.setdefault("pop_size", 6)
    kw.setdefault("random_state", 0)
    return ABCSearchCV(**kw)


def test_fit_sets_attributes_and_predicts():
    s = search().fit(X, y)
    assert set(s.best_params_) == {"max_depth"}
    assert 1 <= s.best_params_["max_depth"] <= 8
    assert 0.8 < s.best_score_ <= 1.0
    assert s.n_evals_ == 30 and len(s.results_) == 30
    assert s.predict(X).shape == y.shape
    assert s.predict_proba(X).shape == (len(X), 3)
    assert 0 <= s.score(X, y) <= 1


def test_int_params_are_ints_and_cover_both_ends():
    s = search(max_evals=60).fit(X, y)
    values = {r["params"]["max_depth"] for r in s.results_}
    assert all(isinstance(v, int) for v in values)
    assert min(values) >= 1 and max(values) <= 8


def test_reproducible_with_random_state():
    a = search(random_state=7).fit(X, y)
    b = search(random_state=7).fit(X, y)
    assert a.best_params_ == b.best_params_
    assert [r["params"] for r in a.results_] == [r["params"] for r in b.results_]


def test_categorical_and_log_float_params():
    space = {
        "C": {"type": "float", "range": (1e-3, 1e3), "log": True},
        "kernel": {"type": "categorical", "choices": ["linear", "rbf"]},
    }
    s = search(estimator=SVC(), param_space=space).fit(X, y)
    assert s.best_params_["kernel"] in ("linear", "rbf")
    assert 1e-3 <= s.best_params_["C"] <= 1e3
    kernels = {r["params"]["kernel"] for r in s.results_}
    assert kernels <= {"linear", "rbf"}


def test_neg_scorer_for_regression():
    Xr, yr = make_regression(n_samples=120, n_features=5, noise=5.0, random_state=0)
    s = search(
        estimator=Ridge(),
        param_space={"alpha": {"type": "float", "range": (1e-3, 100), "log": True}},
        scoring="neg_mean_squared_error",
    ).fit(Xr, yr)
    assert s.best_score_ < 0  # neg_* scores are negative, like in GridSearchCV
    assert s.best_score_ == max(r["score"] for r in s.results_)


def test_history_is_monotonic_non_decreasing():
    s = search().fit(X, y)
    scores = [v for _, v in s.history_]
    assert all(b >= a for a, b in zip(scores, scores[1:]))


def test_failed_candidates_warn_instead_of_silent():
    space = {"n_neighbors": {"type": "int", "range": (-5, 5)}}  # <=0 is invalid
    with pytest.warns(RuntimeWarning, match="failed"):
        s = search(estimator=KNeighborsClassifier(), param_space=space).fit(X, y)
    assert s.best_params_["n_neighbors"] >= 1


def test_error_score_raise_propagates():
    space = {"n_neighbors": {"type": "int", "range": (-5, -1)}}
    with pytest.raises(Exception):
        search(estimator=KNeighborsClassifier(), param_space=space,
               error_score="raise").fit(X, y)


def test_all_candidates_failing_raises_clear_error():
    space = {"n_neighbors": {"type": "int", "range": (-5, -1)}}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(RuntimeError, match="All candidates failed"):
            search(estimator=KNeighborsClassifier(), param_space=space).fit(X, y)


def test_refit_false_has_no_best_estimator():
    s = search(refit=False).fit(X, y)
    assert not hasattr(s, "best_estimator_")
    with pytest.raises(AttributeError):
        s.predict(X)


@pytest.mark.parametrize(
    "space",
    [
        {},
        {"max_depth": {"type": "bool", "range": (0, 1)}},
        {"max_depth": {"type": "int", "range": (5, 1)}},
        {"max_depth": {"type": "categorical", "choices": []}},
        {"C": {"type": "float", "range": (0, 1), "log": True}},
    ],
)
def test_invalid_param_space_raises(space):
    with pytest.raises(ValueError):
        search(param_space=space).fit(X, y)


def test_is_cloneable():
    s = search()
    c = clone(s)
    assert c.get_params()["max_evals"] == s.get_params()["max_evals"]


# --------------------------------------------------------------------------- #
# 0.2.2: cache and scikit-learn compatibility
# --------------------------------------------------------------------------- #
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, GroupKFold

DISCRETE = {"max_depth": {"type": "int", "range": (1, 6)},
            "criterion": {"type": "categorical", "choices": ["gini", "entropy"]}}


def test_cache_reuses_scores_without_changing_the_search():
    a = search(param_space=DISCRETE, max_evals=60, cache=True).fit(X, y)
    b = search(param_space=DISCRETE, max_evals=60, cache=False).fit(X, y)
    # identical trajectory and result: caching only saves time
    assert [r["params"] for r in a.results_] == [r["params"] for r in b.results_]
    assert a.best_params_ == b.best_params_ and a.best_score_ == b.best_score_
    assert a.n_evals_ == b.n_evals_ == 60
    assert a.n_unique_evals_ <= 12 and a.n_unique_evals_ < a.n_evals_
    assert sum(r["cached"] for r in a.results_) == 60 - a.n_unique_evals_
    assert not any(r["cached"] for r in b.results_)


def test_cache_avoids_repeated_cross_validation(monkeypatch):
    import abcopt.sklearn_wrapper as w
    calls = []
    real = w.cross_validate
    monkeypatch.setattr(w, "cross_validate", lambda *a, **k: calls.append(1) or real(*a, **k))
    s = search(param_space=DISCRETE, max_evals=60).fit(X, y)
    assert len(calls) == s.n_unique_evals_


def test_cv_results_matches_gridsearchcv_format():
    s = search(max_evals=30).fit(X, y)
    g = GridSearchCV(DecisionTreeClassifier(random_state=0), {"max_depth": [1, 2]}, cv=3).fit(X, y)
    for key in ["params", "mean_test_score", "std_test_score", "rank_test_score",
                "split0_test_score", "split2_test_score", "mean_fit_time", "std_fit_time",
                "mean_score_time", "std_score_time", "param_max_depth"]:
        assert key in s.cv_results_ and key in g.cv_results_, key
    df = pd.DataFrame(s.cv_results_)
    assert len(df) == s.n_unique_evals_
    assert df.loc[s.best_index_, "rank_test_score"] == 1
    assert df.loc[s.best_index_, "params"] == s.best_params_
    assert np.isclose(df.mean_test_score.max(), s.best_score_)
    assert s.n_splits_ == 3 and s.refit_time_ >= 0


def test_classes_and_estimator_methods_are_exposed():
    s = search(estimator=LogisticRegression(max_iter=500),
               param_space={"C": {"type": "float", "range": (0.01, 10), "log": True}},
               max_evals=12, pop_size=4).fit(X, y)
    np.testing.assert_array_equal(s.classes_, [0, 1, 2])
    assert s.decision_function(X).shape == (len(X), 3)
    assert s.predict_log_proba(X).shape == (len(X), 3)
    tree = search().fit(X, y)
    assert not hasattr(tree, "decision_function")  # DecisionTree has none


def test_fit_params_and_groups_are_forwarded():
    w = np.ones(len(y)); w[y == 0] = 5.0
    s = search(max_evals=12, pop_size=4).fit(X, y, sample_weight=w)
    assert s.best_estimator_.tree_ is not None
    groups = np.arange(len(y)) % 5
    g = search(cv=GroupKFold(n_splits=5), max_evals=12, pop_size=4).fit(X, y, groups=groups)
    assert g.n_splits_ == 5 and "split4_test_score" in g.cv_results_


def test_score_uses_the_scoring_parameter():
    Xr, yr = make_regression(n_samples=120, n_features=5, noise=5.0, random_state=0)
    s = search(estimator=Ridge(), param_space={"alpha": {"type": "float", "range": (1e-3, 10), "log": True}},
               scoring="neg_mean_squared_error", max_evals=12, pop_size=4).fit(Xr, yr)
    assert s.score(Xr, yr) < 0  # neg MSE, like GridSearchCV.score
