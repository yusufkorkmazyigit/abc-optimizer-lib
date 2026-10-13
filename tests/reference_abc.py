"""
Reference ABC used only for testing.

A line-by-line port of ``ABC.py`` from the official implementation published by the
ABC research group at Erciyes University:
https://github.com/artificialbeecolony/Python_ABC (commit 35432a6, MIT License,
Copyright (c) 2020 ABCPython, author Omur Sahin).

Changes from the original, none of which affect the algorithm:
- configuration is passed as arguments instead of a Config object;
- ``random.random()`` is replaced by an injected ``uniform()`` callable so that a test
  can feed the same random stream to this port and to abcopt;
- the objective returns a float instead of a tuple; progress bars and reporting removed;
- only the minimization branch is kept.
"""
import numpy as np


class ReferenceABC:
    def __init__(self, objective, dimension, lower, upper, food_number, max_evals, limit, uniform):
        self.OBJECTIVE_FUNCTION = objective
        self.DIMENSION = dimension
        self.LOWER_BOUND = lower
        self.UPPER_BOUND = upper
        self.FOOD_NUMBER = food_number
        self.MAXIMUM_EVALUATION = max_evals
        self.LIMIT = limit
        self.random = uniform

        self.foods = np.zeros((self.FOOD_NUMBER, self.DIMENSION))
        self.f = np.ones((self.FOOD_NUMBER))
        self.fitness = np.ones((self.FOOD_NUMBER)) * np.iinfo(int).max
        self.trial = np.zeros((self.FOOD_NUMBER))
        self.prob = [0 for x in range(self.FOOD_NUMBER)]
        self.solution = np.zeros((self.DIMENSION))
        self.globalParams = [0 for x in range(self.DIMENSION)]
        self.evalCount = 0

    def calculate_function(self, sol):
        return self.OBJECTIVE_FUNCTION(sol)

    def calculate_fitness(self, fun):
        self.evalCount += 1
        if fun >= 0:
            result = 1 / (fun + 1)
        else:
            result = 1 + abs(fun)
        return result

    def stopping_condition(self):
        return bool(self.evalCount >= self.MAXIMUM_EVALUATION)

    def memorize_best_source(self):
        for i in range(self.FOOD_NUMBER):
            if self.f[i] < self.globalOpt:
                self.globalOpt = np.copy(self.f[i])
                self.globalParams = np.copy(self.foods[i][:])

    def init(self, index):
        if not self.stopping_condition():
            for i in range(self.DIMENSION):
                self.foods[index][i] = self.random() * (self.UPPER_BOUND - self.LOWER_BOUND) + self.LOWER_BOUND
            self.solution = np.copy(self.foods[index][:])
            self.f[index] = self.calculate_function(self.solution)
            self.fitness[index] = self.calculate_fitness(self.f[index])
            self.trial[index] = 0

    def initial(self):
        for i in range(self.FOOD_NUMBER):
            self.init(i)
        self.globalOpt = np.copy(self.f[0])
        self.globalParams = np.copy(self.foods[0][:])

    def _move(self, i):
        r = self.random()
        self.param2change = int(r * self.DIMENSION)
        r = self.random()
        self.neighbour = int(r * self.FOOD_NUMBER)
        while self.neighbour == i:
            r = self.random()
            self.neighbour = int(r * self.FOOD_NUMBER)
        self.solution = np.copy(self.foods[i][:])
        r = self.random()
        self.solution[self.param2change] = self.foods[i][self.param2change] + (
            self.foods[i][self.param2change] - self.foods[self.neighbour][self.param2change]) * (r - 0.5) * 2
        if self.solution[self.param2change] < self.LOWER_BOUND:
            self.solution[self.param2change] = self.LOWER_BOUND
        if self.solution[self.param2change] > self.UPPER_BOUND:
            self.solution[self.param2change] = self.UPPER_BOUND
        self.ObjValSol = self.calculate_function(self.solution)
        self.FitnessSol = self.calculate_fitness(self.ObjValSol)
        if self.FitnessSol > self.fitness[i]:
            self.trial[i] = 0
            self.foods[i][:] = np.copy(self.solution)
            self.f[i] = self.ObjValSol
            self.fitness[i] = self.FitnessSol
        else:
            self.trial[i] = self.trial[i] + 1

    def send_employed_bees(self):
        i = 0
        while (i < self.FOOD_NUMBER) and (not self.stopping_condition()):
            self._move(i)
            i += 1

    def calculate_probabilities(self):
        maxfit = np.copy(max(self.fitness))
        for i in range(self.FOOD_NUMBER):
            self.prob[i] = (0.9 * (self.fitness[i] / maxfit)) + 0.1

    def send_onlooker_bees(self):
        i = 0
        t = 0
        while (t < self.FOOD_NUMBER) and (not self.stopping_condition()):
            r = self.random()
            if r < self.prob[i]:
                t += 1
                self._move(i)
            i += 1
            i = i % self.FOOD_NUMBER

    def send_scout_bees(self):
        if np.amax(self.trial) >= self.LIMIT:
            self.init(self.trial.argmax(axis=0))

    def run(self):
        """Main loop of ABCAlgorithm.py."""
        self.initial()
        self.memorize_best_source()
        while not self.stopping_condition():
            self.send_employed_bees()
            self.calculate_probabilities()
            self.send_onlooker_bees()
            self.memorize_best_source()
            self.send_scout_bees()
        return self.globalParams, float(self.globalOpt)
