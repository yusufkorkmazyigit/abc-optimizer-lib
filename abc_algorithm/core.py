import numpy as np


class CanonicalABCSolver:
    """
    Canonical Artificial Bee Colony (ABC) algorithm.

    A faithful implementation of the algorithm proposed by D. Karaboğa (2005):
    employed bees, onlooker bees (roulette-wheel selection) and scout bees,
    with a single dimension perturbed per move.

    Parameters
    ----------
    objective_func : callable
        Function taking a 1-D ``np.ndarray`` of length ``n_params`` and
        returning a scalar cost to **minimize**.
    n_params : int
        Number of dimensions (decision variables).
    lb, ub : array-like of shape (n_params,)
        Lower and upper bounds for each dimension.
    pop_size : int, default=20
        Number of food sources (SN). Must be at least 2.
    max_evals : int, default=1000
        Total budget of objective-function evaluations.
    limit : int or None, default=None
        Number of unsuccessful trials before a food source is abandoned.
        Defaults to ``pop_size * n_params``.
    random_state : int, np.random.Generator or None, default=None
        Seed or generator for reproducible runs.

    Attributes
    ----------
    best_params : np.ndarray
        Best solution found.
    best_cost : float
        Objective value of ``best_params``.
    history : list of (int, float)
        ``(evaluations_used, best_cost_so_far)`` after each cycle.
    """

    def __init__(self, objective_func, n_params, lb, ub, pop_size=20,
                 max_evals=1000, limit=None, random_state=None):
        self.objective_func = objective_func
        self.D = int(n_params)
        self.lb = np.asarray(lb, dtype=float).ravel()
        self.ub = np.asarray(ub, dtype=float).ravel()
        self.SN = int(pop_size)
        self.total_evaluation = int(max_evals)
        self.limit = limit if limit is not None else self.SN * self.D
        self.random_state = random_state

        self._validate()

        self.best_params = None
        self.best_cost = float("inf")
        self.history = []

    def _validate(self):
        if not callable(self.objective_func):
            raise TypeError("objective_func must be callable.")
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

    @staticmethod
    def _calculate_fitness(cost):
        """Convert a cost (to be minimized) into a fitness (to be maximized)."""
        return 1.0 / (1.0 + cost) if cost >= 0 else 1.0 + abs(cost)

    def _random_solution(self):
        return self.lb + self._rng.random(self.D) * (self.ub - self.lb)

    def _neighbour(self, i):
        k = self._rng.integers(0, self.SN - 1)
        return k if k < i else k + 1  # uniform over all indices except i

    def _perturb(self, current, neighbour):
        """Canonical move: change exactly one randomly chosen dimension."""
        j = self._rng.integers(0, self.D)
        phi = self._rng.uniform(-1.0, 1.0)
        new_sol = np.copy(current)
        new_sol[j] = current[j] + phi * (current[j] - neighbour[j])
        new_sol[j] = np.clip(new_sol[j], self.lb[j], self.ub[j])
        return new_sol

    def _evaluate(self, x):
        cost = float(self.objective_func(x))
        if np.isnan(cost):
            cost = float("inf")
        self._evals += 1
        return cost

    def _try_improve(self, i):
        """Greedy selection between food source i and a mutated copy."""
        k = self._neighbour(i)
        new_sol = self._perturb(self.foods[i], self.foods[k])
        new_cost = self._evaluate(new_sol)
        new_fit = self._calculate_fitness(new_cost)
        if new_fit > self.fitness[i]:
            self.foods[i] = new_sol
            self.costs[i] = new_cost
            self.fitness[i] = new_fit
            self.trial[i] = 0
        else:
            self.trial[i] += 1

    def _memorize_best(self):
        idx = int(np.argmin(self.costs))
        if self.costs[idx] < self.best_cost:
            self.best_cost = float(self.costs[idx])
            self.best_params = np.copy(self.foods[idx])

    def solve(self):
        """
        Run the optimization.

        Returns
        -------
        best_params : np.ndarray
        best_cost : float
        history : list of (int, float)
        """
        self._rng = np.random.default_rng(self.random_state)
        self._evals = 0
        self.best_params = None
        self.best_cost = float("inf")
        self.history = []

        # Initialization
        self.foods = self.lb + self._rng.random((self.SN, self.D)) * (self.ub - self.lb)
        self.costs = np.array([self._evaluate(x) for x in self.foods])
        self.fitness = np.array([self._calculate_fitness(c) for c in self.costs])
        self.trial = np.zeros(self.SN, dtype=int)
        self._memorize_best()
        self.history.append((self._evals, self.best_cost))

        while self._evals < self.total_evaluation:
            # 1. Employed bees: every food source is visited once.
            for i in range(self.SN):
                if self._evals >= self.total_evaluation:
                    break
                self._try_improve(i)

            # 2. Onlooker bees: roulette-wheel selection proportional to fitness.
            total_fit = np.sum(self.fitness)
            prob = (self.fitness / total_fit) if total_fit > 0 else np.full(self.SN, 1.0 / self.SN)
            t, i = 0, 0
            while t < self.SN and self._evals < self.total_evaluation:
                if self._rng.random() < prob[i]:
                    t += 1
                    self._try_improve(i)
                i = (i + 1) % self.SN

            self._memorize_best()

            # 3. Scout bee: abandon the most exhausted source if over the limit.
            worst = int(np.argmax(self.trial))
            if self.trial[worst] > self.limit and self._evals < self.total_evaluation:
                self.foods[worst] = self._random_solution()
                self.costs[worst] = self._evaluate(self.foods[worst])
                self.fitness[worst] = self._calculate_fitness(self.costs[worst])
                self.trial[worst] = 0
                self._memorize_best()

            self.history.append((self._evals, self.best_cost))

        return self.best_params, self.best_cost, self.history
