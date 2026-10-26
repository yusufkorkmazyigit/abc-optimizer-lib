import numpy as np

from .callbacks import CycleState


class CanonicalABCSolver:
    """
    Canonical Artificial Bee Colony (ABC) algorithm.

    Follows the basic ABC of D. Karaboğa (2005) step by step as implemented in the
    reference code published by the ABC research group at Erciyes University
    (https://github.com/artificialbeecolony/Python_ABC):

    1. Initialization: every food source is drawn uniformly within the bounds.
    2. Employed bees: each source i changes one random dimension j using a random
       neighbour k != i:  v_ij = x_ij + phi * (x_ij - x_kj),  phi ~ U(-1, 1),
       clipped to the bounds; greedy selection on fitness.
    3. Probabilities: p_i = 0.9 * fit_i / max(fit) + 0.1.
    4. Onlooker bees: sources are visited cyclically; source i is improved
       (same move and greedy selection) when U(0, 1) < p_i, until SN onlookers
       have been placed.
    5. The best solution so far is memorized.
    6. Scout bee: the source with the most failed trials is replaced by a random
       solution when its trial counter reaches ``limit``.

    The loop stops when ``max_evals`` objective evaluations have been used, or
    earlier when a callback asks it to.
    Random numbers are drawn in the same order as in the reference code, so for an
    identical stream of U(0, 1) numbers both produce identical trajectories
    (see ``tests/test_reference_equivalence.py``).

    The solver can be driven in two ways:

    - ``solve()`` calls ``objective_func`` itself and returns the result.
    - ``ask()`` / ``tell()`` hand each candidate to the caller, who evaluates it
      anywhere (another process, a cluster job, a lab experiment) and reports
      the cost back. ABC is sequential, so one candidate is pending at a time.
      For the same seed both ways visit exactly the same points.

    Parameters
    ----------
    objective_func : callable or None
        Function taking a 1-D ``np.ndarray`` of length ``n_params`` and
        returning a scalar cost to **minimize**. May be ``None`` when the
        solver is only driven through ``ask()`` / ``tell()``.
    n_params : int
        Number of dimensions (decision variables).
    lb, ub : array-like of shape (n_params,)
        Lower and upper bounds for each dimension.
    pop_size : int, default=20
        Number of food sources (SN), equal to the number of employed bees and to
        the number of onlooker bees. The colony size in the reference code
        (``NumberOfPopulation``) is ``2 * pop_size``.
    max_evals : int, default=1000
        Total budget of objective-function evaluations.
    limit : int or None, default=None
        Number of failed trials after which a food source is abandoned.
        Defaults to ``pop_size * n_params`` (Karaboğa & Baştürk, 2008).
    random_state : int, np.random.Generator or None, default=None
        Seed or generator for reproducible runs.
    callbacks : callable, list of callables or None, default=None
        Called after every cycle with a :class:`~abcopt.CycleState`. If any
        callback returns ``True`` the run stops after that cycle (all callbacks
        of that cycle are still called). See :class:`~abcopt.EarlyStopping`.
        Callbacks never consume random numbers, so without a stop request the
        trajectory is identical to a run without callbacks.

    Attributes
    ----------
    best_params : np.ndarray
        Best solution found.
    best_cost : float
        Objective value of ``best_params``.
    history : list of (int, float)
        ``(evaluations_used, best_cost_so_far)`` after initialization and after each cycle.
    n_cycles : int
        Number of completed cycles.
    stopped_early : bool
        ``True`` if a callback stopped the run before ``max_evals`` was used.
    n_evals : int
        Objective evaluations used in the current or last run.
    done : bool
        ``True`` once the current run has finished (budget used or stopped by a
        callback). ``False`` before the first ``ask()`` and while a run is going on.

    Examples
    --------
    >>> import numpy as np
    >>> solver = CanonicalABCSolver(None, n_params=2, lb=[-5, -5], ub=[5, 5],
    ...                             max_evals=500, random_state=0)
    >>> while not solver.done:
    ...     x = solver.ask()
    ...     solver.tell(x, float(np.sum(x ** 2)))
    >>> solver.best_cost < 1e-3
    True
    """

    def __init__(self, objective_func, n_params, lb, ub, pop_size=20,
                 max_evals=1000, limit=None, random_state=None, callbacks=None):
        self.objective_func = objective_func
        self.D = int(n_params)
        self.lb = np.asarray(lb, dtype=float).ravel()
        self.ub = np.asarray(ub, dtype=float).ravel()
        self.SN = int(pop_size)
        self.total_evaluation = int(max_evals)
        self.limit = limit if limit is not None else self.SN * self.D
        self.random_state = random_state
        self.callbacks = self._as_callback_list(callbacks)

        self._validate()

        self.best_params = None
        self.best_cost = float("inf")
        self.history = []
        self.n_cycles = 0
        self.stopped_early = False
        self._evals = 0
        self._gen = None       # the running ABC loop (a generator), None when idle
        self._pending = None   # candidate handed out by ask() and awaiting tell()
        self._done = False

    @staticmethod
    def _as_callback_list(callbacks):
        if callbacks is None:
            return []
        if callable(callbacks):
            return [callbacks]
        try:
            callbacks = list(callbacks)
        except TypeError:
            raise TypeError("callbacks must be a callable or a list of callables.") from None
        for cb in callbacks:
            if not callable(cb):
                raise TypeError(f"Every callback must be callable, got {cb!r}.")
        return callbacks

    def _validate(self):
        if self.objective_func is not None and not callable(self.objective_func):
            raise TypeError("objective_func must be callable or None.")
        if self.D < 1:
            raise ValueError("n_params must be at least 1.")
        if self.lb.shape != (self.D,) or self.ub.shape != (self.D,):
            raise ValueError(
                f"lb and ub must both have length n_params={self.D}, "
                f"got {self.lb.size} and {self.ub.size}."
            )
        if np.any(self.lb > self.ub):
            raise ValueError("Every lower bound must be <= its upper bound.")
        if self.SN < 2:
            raise ValueError("pop_size must be at least 2 (neighbours are needed).")
        if self.total_evaluation < self.SN:
            raise ValueError(
                f"max_evals ({self.total_evaluation}) must be >= pop_size ({self.SN}) "
                "so that the initial population can be evaluated."
            )
        if self.limit < 1:
            raise ValueError("limit must be at least 1.")

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _u(self):
        """One U(0, 1) draw. All randomness goes through this method."""
        return float(self._rng.random())

    @staticmethod
    def _calculate_fitness(cost):
        """Convert a cost (to be minimized) into a fitness (to be maximized)."""
        return 1.0 / (1.0 + cost) if cost >= 0 else 1.0 + abs(cost)

    def _stop(self):
        return self._evals >= self.total_evaluation

    def _evaluate(self, x):
        """Hand ``x`` out for evaluation and receive its cost (the only yield point)."""
        cost = yield np.copy(x)
        self._evals += 1
        return cost

    def _init_source(self, i):
        """Place food source i at a uniformly random position (initialization and scouts)."""
        for j in range(self.D):
            self.foods[i, j] = self._u() * (self.ub[j] - self.lb[j]) + self.lb[j]
        self.costs[i] = yield from self._evaluate(self.foods[i])
        self.fitness[i] = self._calculate_fitness(self.costs[i])
        self.trial[i] = 0

    def _try_improve(self, i):
        """Neighbourhood move on food source i followed by greedy selection."""
        j = int(self._u() * self.D)
        k = int(self._u() * self.SN)
        while k == i:
            k = int(self._u() * self.SN)
        new_sol = np.copy(self.foods[i])
        phi = (self._u() - 0.5) * 2.0
        new_sol[j] = self.foods[i, j] + (self.foods[i, j] - self.foods[k, j]) * phi
        new_sol[j] = min(max(new_sol[j], self.lb[j]), self.ub[j])

        new_cost = yield from self._evaluate(new_sol)
        new_fit = self._calculate_fitness(new_cost)
        if new_fit > self.fitness[i]:
            self.foods[i] = new_sol
            self.costs[i] = new_cost
            self.fitness[i] = new_fit
            self.trial[i] = 0
        else:
            self.trial[i] += 1

    def _memorize_best(self):
        for i in range(self.SN):
            if self.costs[i] < self.best_cost:
                self.best_cost = float(self.costs[i])
                self.best_params = np.copy(self.foods[i])

    # ------------------------------------------------------------------ #
    # Phases
    # ------------------------------------------------------------------ #
    def _employed_bees(self):
        i = 0
        while i < self.SN and not self._stop():
            yield from self._try_improve(i)
            i += 1

    def _calculate_probabilities(self):
        maxfit = np.max(self.fitness)
        if maxfit > 0:
            self.prob = 0.9 * (self.fitness / maxfit) + 0.1
        else:
            # Every source has infinite cost (fitness 0). The reference code would divide
            # by zero and loop forever here; selecting every source is the natural limit.
            self.prob = np.ones(self.SN)

    def _onlooker_bees(self):
        i, t = 0, 0
        while t < self.SN and not self._stop():
            if self._u() < self.prob[i]:
                t += 1
                yield from self._try_improve(i)
            i = (i + 1) % self.SN

    def _scout_bee(self):
        worst = int(np.argmax(self.trial))
        if self.trial[worst] >= self.limit and not self._stop():
            yield from self._init_source(worst)

    def _run_callbacks(self):
        """Call every callback; return True if any of them asks to stop."""
        if not self.callbacks:
            return False
        state = CycleState(
            cycle=self.n_cycles,
            n_evals=self._evals,
            max_evals=self.total_evaluation,
            best_cost=self.best_cost,
            best_params=np.copy(self.best_params),
        )
        stop = False
        for cb in self.callbacks:
            if cb(state):
                stop = True
        return stop

    # ------------------------------------------------------------------ #
    # The ABC loop. It yields every candidate and receives its cost, so the
    # same code serves solve() and ask()/tell().
    # ------------------------------------------------------------------ #
    def _reset_state(self):
        if getattr(self, "_rng_override", None) is not None:  # testing hook
            self._rng = self._rng_override
        elif isinstance(self.random_state, np.random.Generator):
            self._rng = self.random_state
        else:
            self._rng = np.random.default_rng(self.random_state)
        self._evals = 0
        self.best_params = None
        self.best_cost = float("inf")
        self.history = []
        self.n_cycles = 0
        self.stopped_early = False

        self.foods = np.zeros((self.SN, self.D))
        self.costs = np.full(self.SN, np.inf)
        self.fitness = np.zeros(self.SN)
        self.trial = np.zeros(self.SN, dtype=int)
        self.prob = np.zeros(self.SN)

    def _loop(self):
        for i in range(self.SN):
            yield from self._init_source(i)
        self.best_cost, self.best_params = float(self.costs[0]), np.copy(self.foods[0])
        self._memorize_best()
        self.history.append((self._evals, self.best_cost))

        while not self._stop():
            yield from self._employed_bees()
            self._calculate_probabilities()
            yield from self._onlooker_bees()
            self._memorize_best()
            yield from self._scout_bee()
            self.history.append((self._evals, self.best_cost))
            self.n_cycles += 1
            if self._run_callbacks():
                self.stopped_early = not self._stop()
                break

    def _advance(self, cost=None):
        """Run the loop up to the next candidate; mark the run done when it ends."""
        try:
            self._pending = self._gen.send(cost)
        except StopIteration:
            self._pending = None
            self._gen = None
            self._done = True

    @staticmethod
    def _as_cost(cost):
        try:
            cost = float(cost)
        except (TypeError, ValueError):
            raise TypeError(f"cost must be a real scalar, got {cost!r}.") from None
        return float("inf") if np.isnan(cost) else cost

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    @property
    def done(self):
        return self._done

    @property
    def n_evals(self):
        return self._evals

    def reset(self):
        """Discard any run in progress. The next ``ask()`` starts a new run."""
        if self._gen is not None:
            self._gen.close()
        self._gen = None
        self._pending = None
        self._done = False

    def ask(self):
        """
        Return the next candidate to evaluate.

        The first call starts a run. Until its cost is reported with ``tell()``
        the same candidate is returned again, so an interrupted evaluation can
        simply be retried.

        Returns
        -------
        x : np.ndarray of shape (n_params,)
            A copy; modifying it does not affect the solver.

        Raises
        ------
        RuntimeError
            If the run has finished (``done`` is ``True``). Call ``reset()``
            to start a new one.
        """
        if self._done:
            raise RuntimeError(
                "The run has finished (budget used or stopped by a callback). "
                "Read best_params / best_cost, or call reset() to start a new run."
            )
        if self._gen is None:
            self._reset_state()
            self._gen = self._loop()
            self._advance()
        return np.copy(self._pending)

    def tell(self, x, cost):
        """
        Report the cost of the candidate returned by the last ``ask()``.

        Parameters
        ----------
        x : array-like of shape (n_params,)
            The candidate, exactly as returned by ``ask()``.
        cost : float
            Its objective value (to be minimized). ``NaN`` is treated as the
            worst possible cost.

        Raises
        ------
        RuntimeError
            If no candidate is pending.
        ValueError
            If ``x`` is not the pending candidate.
        """
        if self._pending is None:
            raise RuntimeError("No candidate is pending; call ask() first.")
        x = np.asarray(x, dtype=float)
        if x.shape != self._pending.shape or not np.array_equal(x, self._pending):
            raise ValueError(
                "x is not the pending candidate. ABC is sequential: tell() must "
                "report the point returned by the last ask()."
            )
        self._advance(self._as_cost(cost))

    def solve(self):
        """
        Run the optimization, calling ``objective_func`` for every candidate.

        Starts a new run; a run in progress through ``ask()`` / ``tell()`` is
        discarded.

        Returns
        -------
        best_params : np.ndarray
        best_cost : float
        history : list of (int, float)
        """
        if self.objective_func is None:
            raise TypeError(
                "solve() needs an objective_func; without one, drive the solver "
                "with ask() and tell()."
            )
        self.reset()
        self.ask()  # starts the run
        while not self._done:
            # _pending is a private copy, so the objective may modify it freely
            self._advance(self._as_cost(self.objective_func(self._pending)))
        return self.best_params, self.best_cost, self.history
