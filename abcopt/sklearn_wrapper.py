import inspect
import json
import math
import time
import warnings

import numpy as np
from sklearn.base import BaseEstimator, MetaEstimatorMixin, clone, is_classifier
from sklearn.metrics import check_scoring
from sklearn.model_selection import check_cv, cross_validate
from sklearn.utils.metaestimators import available_if

from .core import CanonicalABCSolver

_VALID_TYPES = ("float", "int", "categorical")
_CV_PARAMS_ARG = "params" if "params" in inspect.signature(cross_validate).parameters else "fit_params"


def _estimator_has(attr):
    """Allow the method only if the refitted best estimator (or the base estimator) has it."""
    def check(self):
        if hasattr(self, "best_estimator_"):
            return hasattr(self.best_estimator_, attr)
        return hasattr(self.estimator, attr)
    return check


class ABCSearchCV(MetaEstimatorMixin, BaseEstimator):
    """
    Hyperparameter search with the canonical Artificial Bee Colony algorithm.

    Works like ``GridSearchCV`` / ``RandomizedSearchCV``: every candidate is
    scored with cross-validation and the best one is refit on the full data.

    Parameters
    ----------
    estimator : estimator object
        Any scikit-learn compatible estimator.
    param_space : dict
        Maps each hyperparameter name to a specification:

        - ``{"type": "float", "range": (low, high)}``
        - ``{"type": "float", "range": (low, high), "log": True}``
        - ``{"type": "int", "range": (low, high)}`` (both ends inclusive)
        - ``{"type": "categorical", "choices": [a, b, c]}``

        Categorical choices are mapped to equal-width intervals of one continuous
        dimension, so ABC treats neighbouring choices as "close". This works well
        for a handful of choices.
    cv : int, cross-validation generator or iterable, default=3
    scoring : str or callable, default="accuracy"
        Any scikit-learn scorer. Higher is better (use ``neg_*`` scorers
        for errors), exactly as in ``GridSearchCV``.
    pop_size : int, default=20
        Number of food sources.
    max_evals : int, default=200
        Total number of candidates the algorithm evaluates.
    limit : int or None, default=None
        Abandonment limit for scout bees.
    random_state : int or None, default=None
        Seed for reproducible searches.
    n_jobs : int or None, default=None
        Parallel jobs used inside each cross-validation.
    error_score : "raise" or float, default=np.nan
        What to do when a candidate fails to fit. ``"raise"`` re-raises the
        error; otherwise the candidate gets this score (NaN is treated as the
        worst possible score) and a warning is issued.
    refit : bool, default=True
        Refit the best estimator on the whole dataset.
    cache : bool, default=True
        Integer and categorical parameters are rounded, so ABC often proposes a
        configuration it has already evaluated. With ``cache=True`` its stored
        score is reused instead of running cross-validation again. The candidate
        still counts towards ``max_evals``, so the search itself is unchanged; only
        time is saved.
    verbose : bool, default=False

    Attributes
    ----------
    best_params_ : dict
    best_score_ : float
        Mean cross-validated score of the best candidate.
    best_index_ : int
        Index of the best candidate in ``cv_results_``.
    best_estimator_ : estimator
        Only available when ``refit=True``.
    cv_results_ : dict of numpy arrays
        One entry per unique evaluated configuration, in the same format as
        ``GridSearchCV.cv_results_`` (``params``, ``param_<name>``,
        ``mean_test_score``, ``std_test_score``, ``rank_test_score``,
        ``split<k>_test_score``, ``mean_fit_time``, ...). Convert with
        ``pandas.DataFrame(search.cv_results_)``.
    results_ : list of dict
        Every candidate in the order ABC proposed it, with its ``params``,
        ``score`` and whether it was served from the ``cache``.
    history_ : list of (int, float)
        ``(evaluations_used, best_score_so_far)`` after each ABC cycle.
    n_evals_ : int
        Number of candidates evaluated by the algorithm (``== max_evals``).
    n_unique_evals_ : int
        Number of distinct configurations evaluated (with ``cache=True``, the
        number of cross-validation runs actually performed).
    n_splits_ : int
    refit_time_ : float
        Seconds spent refitting the best estimator.
    classes_ : ndarray
        Class labels (classifiers only, after refit).
    """

    def __init__(self, estimator, param_space, cv=3, scoring="accuracy",
                 pop_size=20, max_evals=200, limit=None, random_state=None,
                 n_jobs=None, error_score=np.nan, refit=True, cache=True, verbose=False):
        self.estimator = estimator
        self.param_space = param_space
        self.cv = cv
        self.scoring = scoring
        self.pop_size = pop_size
        self.max_evals = max_evals
        self.limit = limit
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.error_score = error_score
        self.refit = refit
        self.cache = cache
        self.verbose = verbose

    # ------------------------------------------------------------------ #
    # Search-space encoding: every parameter becomes one continuous
    # dimension that ABC can search; _decode maps it back.
    # ------------------------------------------------------------------ #
    def _build_space(self):
        if not isinstance(self.param_space, dict) or not self.param_space:
            raise ValueError("param_space must be a non-empty dict.")

        keys, lb, ub = [], [], []
        for name, spec in self.param_space.items():
            p_type = spec.get("type")
            if p_type not in _VALID_TYPES:
                raise ValueError(
                    f"Parameter '{name}': type must be one of {_VALID_TYPES}, got {p_type!r}."
                )
            if p_type == "categorical":
                choices = spec.get("choices")
                if not choices:
                    raise ValueError(f"Parameter '{name}': 'choices' must be a non-empty list.")
                lo, hi = 0.0, float(len(choices))
            else:
                low, high = spec.get("range", (None, None))
                if low is None or high is None or low > high:
                    raise ValueError(f"Parameter '{name}': 'range' must be (low, high) with low <= high.")
                if spec.get("log"):
                    if low <= 0:
                        raise ValueError(f"Parameter '{name}': log scale needs a positive range.")
                    lo, hi = math.log10(low), math.log10(high)
                elif p_type == "int":
                    # widen by 0.5 so both ends are reachable after rounding
                    lo, hi = low - 0.5, high + 0.5
                else:
                    lo, hi = float(low), float(high)
            keys.append(name)
            lb.append(lo)
            ub.append(hi)
        return keys, lb, ub

    def _decode(self, vector):
        params = {}
        for name, val in zip(self._keys, vector):
            spec = self.param_space[name]
            p_type = spec["type"]
            if p_type == "categorical":
                choices = spec["choices"]
                params[name] = choices[min(int(val), len(choices) - 1)]
                continue
            if spec.get("log"):
                val = 10 ** val
            if p_type == "int":
                low, high = spec["range"]
                params[name] = int(min(max(round(val), low), high))
            else:
                params[name] = float(val)
        return params

    @staticmethod
    def _cache_key(params):
        return json.dumps(params, sort_keys=True, default=repr)

    # ------------------------------------------------------------------ #
    def _evaluate(self, params, X, y, groups, fit_params):
        model = clone(self.estimator).set_params(**params)
        try:
            out = cross_validate(model, X, y, groups=groups, cv=self._cv, scoring=self._scorer,
                                 n_jobs=self.n_jobs, error_score="raise",
                                 **{_CV_PARAMS_ARG: fit_params or None})
            return {"split_scores": np.asarray(out["test_score"], float),
                    "fit_time": np.asarray(out["fit_time"], float),
                    "score_time": np.asarray(out["score_time"], float)}
        except Exception as exc:
            if self.error_score == "raise":
                raise
            warnings.warn(
                f"Candidate {params} failed and was given score "
                f"{self.error_score}: {type(exc).__name__}: {exc}",
                RuntimeWarning,
                stacklevel=3,
            )
            n = self.n_splits_
            return {"split_scores": np.full(n, float(self.error_score)),
                    "fit_time": np.zeros(n), "score_time": np.zeros(n)}

    def fit(self, X, y=None, groups=None, **fit_params):
        """
        Run the search.

        Parameters
        ----------
        X, y : array-like
        groups : array-like, optional
            Group labels for group-aware cross-validation splitters.
        **fit_params : dict
            Passed to the estimator's ``fit`` (e.g. ``sample_weight``).
        """
        self._keys, lb, ub = self._build_space()
        self._cv = check_cv(self.cv, y, classifier=is_classifier(self.estimator))
        self.n_splits_ = self._cv.get_n_splits(X, y, groups)
        self._scorer = check_scoring(self.estimator, scoring=self.scoring)

        self.results_ = []
        unique = {}   # cache key -> evaluation record
        order = []    # cache keys in order of first evaluation

        def objective(vector):
            params = self._decode(vector)
            key = self._cache_key(params)
            cached = self.cache and key in unique
            if not cached:
                rec = self._evaluate(params, X, y, groups, fit_params)
                rec["params"] = params
                if key not in unique:  # with cache=False, cv_results_ keeps the first run
                    unique[key] = rec
                    order.append(key)
            else:
                rec = unique[key]
            score = float(np.mean(rec["split_scores"]))
            self.results_.append({"params": params, "score": score, "cached": bool(cached)})
            if self.verbose:
                tag = " (cached)" if cached else ""
                print(f"[{len(self.results_)}/{self.max_evals}] score={score:.5f}{tag} {params}")
            # ABC minimizes; NaN scores become +inf cost inside the solver.
            return -score

        solver = CanonicalABCSolver(
            objective, len(self._keys), lb, ub,
            pop_size=self.pop_size, max_evals=self.max_evals,
            limit=self.limit, random_state=self.random_state,
        )
        best_vec, best_cost, history = solver.solve()

        if not np.isfinite(best_cost):
            raise RuntimeError(
                "All candidates failed. Check param_space and the estimator; "
                "set error_score='raise' to see the underlying error."
            )

        self.best_params_ = self._decode(best_vec)
        self.best_score_ = -best_cost
        self.history_ = [(n, -c) for n, c in history]
        self.n_evals_ = len(self.results_)
        self.n_unique_evals_ = len(order)
        self.cv_results_ = self._format_cv_results([unique[k] for k in order])
        self.best_index_ = order.index(self._cache_key(self.best_params_))

        if self.refit:
            self.best_estimator_ = clone(self.estimator).set_params(**self.best_params_)
            t0 = time.perf_counter()
            if y is None:
                self.best_estimator_.fit(X, **fit_params)
            else:
                self.best_estimator_.fit(X, y, **fit_params)
            self.refit_time_ = time.perf_counter() - t0
        return self

    def _format_cv_results(self, records):
        n = len(records)
        splits = np.array([r["split_scores"] for r in records]).reshape(n, -1)
        fit_t = np.array([r["fit_time"] for r in records]).reshape(n, -1)
        score_t = np.array([r["score_time"] for r in records]).reshape(n, -1)
        mean = splits.mean(axis=1)  # NaN if any split failed, as in GridSearchCV
        ranks_src = np.where(np.isnan(mean), -np.inf, mean)
        order = np.argsort(-ranks_src, kind="stable")
        rank = np.empty(n, dtype=np.int32)
        rank[order] = np.arange(1, n + 1)

        res = {
            "mean_fit_time": fit_t.mean(axis=1), "std_fit_time": fit_t.std(axis=1),
            "mean_score_time": score_t.mean(axis=1), "std_score_time": score_t.std(axis=1),
        }
        for name in self._keys:
            res[f"param_{name}"] = np.ma.MaskedArray(
                np.array([r["params"][name] for r in records], dtype=object), mask=False)
        res["params"] = [r["params"] for r in records]
        for k in range(splits.shape[1]):
            res[f"split{k}_test_score"] = splits[:, k]
        res["mean_test_score"] = mean
        res["std_test_score"] = splits.std(axis=1)
        res["rank_test_score"] = rank
        return res

    # ------------------------------------------------------------------ #
    def _check_refit(self, method):
        if not hasattr(self, "best_estimator_"):
            raise AttributeError(
                f"{method} is only available after fit() with refit=True."
            )
        return self.best_estimator_

    @property
    def classes_(self):
        return self._check_refit("classes_").classes_

    def predict(self, X):
        return self._check_refit("predict").predict(X)

    @available_if(_estimator_has("predict_proba"))
    def predict_proba(self, X):
        return self._check_refit("predict_proba").predict_proba(X)

    @available_if(_estimator_has("predict_log_proba"))
    def predict_log_proba(self, X):
        return self._check_refit("predict_log_proba").predict_log_proba(X)

    @available_if(_estimator_has("decision_function"))
    def decision_function(self, X):
        return self._check_refit("decision_function").decision_function(X)

    @available_if(_estimator_has("transform"))
    def transform(self, X):
        return self._check_refit("transform").transform(X)

    def score(self, X, y):
        """Score of the refitted best estimator using ``scoring`` (as in GridSearchCV)."""
        est = self._check_refit("score")
        return check_scoring(self.estimator, scoring=self.scoring)(est, X, y)
