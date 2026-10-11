import math
import warnings

import numpy as np
from sklearn.base import BaseEstimator, MetaEstimatorMixin, clone
from sklearn.model_selection import cross_val_score

from .core import CanonicalABCSolver

_VALID_TYPES = ("float", "int", "categorical")


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
    cv : int or cross-validation generator, default=3
    scoring : str or callable, default="accuracy"
        Any scikit-learn scorer. Higher is better (use ``neg_*`` scorers
        for errors), exactly as in ``GridSearchCV``.
    pop_size : int, default=20
        Number of food sources.
    max_evals : int, default=200
        Total number of candidates evaluated (each one is a full CV run).
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
    verbose : bool, default=False

    Attributes
    ----------
    best_params_ : dict
    best_score_ : float
        Mean cross-validated score of the best candidate.
    best_estimator_ : estimator
        Only available when ``refit=True``.
    history_ : list of (int, float)
        ``(evaluations_used, best_score_so_far)`` after each ABC cycle.
    results_ : list of dict
        Every evaluated candidate with its ``params`` and ``score``.
    n_evals_ : int
        Number of candidates evaluated.
    """

    def __init__(self, estimator, param_space, cv=3, scoring="accuracy",
                 pop_size=20, max_evals=200, limit=None, random_state=None,
                 n_jobs=None, error_score=np.nan, refit=True, verbose=False):
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

    # ------------------------------------------------------------------ #
    def fit(self, X, y=None, groups=None):
        self._keys, lb, ub = self._build_space()
        self.results_ = []

        def objective(vector):
            params = self._decode(vector)
            model = clone(self.estimator).set_params(**params)
            try:
                scores = cross_val_score(
                    model, X, y, groups=groups, cv=self.cv, scoring=self.scoring,
                    n_jobs=self.n_jobs, error_score="raise",
                )
                score = float(np.mean(scores))
            except Exception as exc:
                if self.error_score == "raise":
                    raise
                warnings.warn(
                    f"Candidate {params} failed and was given score "
                    f"{self.error_score}: {type(exc).__name__}: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                score = float(self.error_score)
            self.results_.append({"params": params, "score": score})
            if self.verbose:
                print(f"[{len(self.results_)}/{self.max_evals}] score={score:.5f} {params}")
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

        if self.refit:
            self.best_estimator_ = clone(self.estimator).set_params(**self.best_params_)
            if y is None:
                self.best_estimator_.fit(X)
            else:
                self.best_estimator_.fit(X, y)
        return self

    # ------------------------------------------------------------------ #
    def _check_refit(self, method):
        if not hasattr(self, "best_estimator_"):
            raise AttributeError(
                f"{method} is only available after fit() with refit=True."
            )
        return self.best_estimator_

    def predict(self, X):
        return self._check_refit("predict").predict(X)

    def predict_proba(self, X):
        return self._check_refit("predict_proba").predict_proba(X)

    def score(self, X, y):
        return self._check_refit("score").score(X, y)
