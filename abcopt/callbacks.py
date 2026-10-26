"""Callbacks for monitoring and stopping an ABC run.

A callback is any callable that takes a :class:`CycleState`. It is called once
after every ABC cycle (employed, onlooker and scout phases). If it returns
``True`` (any truthy value) the run stops after that cycle; ``None`` or
``False`` lets it continue.

Example::

    def log(state):
        print(state.cycle, state.n_evals, state.best_cost)

    solver = CanonicalABCSolver(f, 2, lb, ub, callbacks=[log, EarlyStopping(patience=20)])
"""

from dataclasses import dataclass
from typing import Any, Optional

__all__ = ["CycleState", "EarlyStopping"]


@dataclass(frozen=True)
class CycleState:
    """Snapshot passed to callbacks after each cycle.

    Attributes
    ----------
    cycle : int
        Number of completed cycles, starting at 1.
    n_evals : int
        Objective evaluations used so far.
    max_evals : int
        Evaluation budget of the run.
    best_cost : float
        Lowest cost found so far (``CanonicalABCSolver`` minimizes).
    best_params : object
        Best solution so far: a copy of the ``np.ndarray`` for
        ``CanonicalABCSolver``, the decoded hyperparameter ``dict`` for
        ``ABCSearchCV``.
    best_score : float or None
        Best mean CV score so far (``ABCSearchCV`` only, ``-best_cost``);
        ``None`` for ``CanonicalABCSolver``.
    """

    cycle: int
    n_evals: int
    max_evals: int
    best_cost: float
    best_params: Any
    best_score: Optional[float] = None


class EarlyStopping:
    """Stop when the best cost has not improved for ``patience`` cycles.

    A cycle counts as an improvement when the best cost drops by more than
    ``min_delta``. Works unchanged with ``ABCSearchCV``, where a lower cost
    means a higher score.

    Parameters
    ----------
    patience : int, default=10
        Number of cycles without improvement before stopping.
    min_delta : float, default=0.0
        Minimum decrease of the best cost that counts as an improvement.

    Attributes
    ----------
    stopped_cycle : int or None
        Cycle at which this callback stopped the last run, ``None`` if it did not.

    Notes
    -----
    The same instance can be reused across runs: its state is reset whenever a
    run starts (``state.cycle == 1``).
    """

    def __init__(self, patience=10, min_delta=0.0):
        if int(patience) < 1:
            raise ValueError("patience must be at least 1.")
        if min_delta < 0:
            raise ValueError("min_delta must be >= 0.")
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self._reset()

    def _reset(self):
        self._best = float("inf")
        self._wait = 0
        self.stopped_cycle = None

    def __call__(self, state):
        if state.cycle == 1:
            self._reset()
        if state.best_cost < self._best - self.min_delta:
            self._best = state.best_cost
            self._wait = 0
            return False
        self._wait += 1
        if self._wait >= self.patience:
            self.stopped_cycle = state.cycle
            return True
        return False

    def __repr__(self):
        return f"EarlyStopping(patience={self.patience}, min_delta={self.min_delta})"
