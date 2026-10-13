# 🐝 abcopt

**Canonical Artificial Bee Colony (ABC) optimization in pure Python, with a scikit-learn compatible hyperparameter search.**

[![PyPI](https://img.shields.io/pypi/v/abcopt)](https://pypi.org/project/abcopt/)
[![tests](https://github.com/yusufkorkmazyigit/abc-optimizer-lib/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/yusufkorkmazyigit/abc-optimizer-lib/actions/workflows/tests.yml)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![scikit-learn](https://img.shields.io/badge/sklearn-compatible-orange)
![Dependencies](https://img.shields.io/badge/deps-numpy%20%7C%20scikit--learn-lightgrey)

**abcopt** implements the **Artificial Bee Colony** algorithm proposed by **Derviş Karaboğa (2005)** at Erciyes University, step for step as in the [reference implementation](https://github.com/artificialbeecolony/Python_ABC) published by the ABC research group. It is useful for two things:

1. **Black-box optimization:** minimize any objective function within box constraints, with no gradients required.
2. **Hyperparameter tuning:** a drop-in alternative to `GridSearchCV` / `RandomizedSearchCV` for any scikit-learn compatible estimator (scikit-learn, LightGBM, XGBoost, …).

---

## ✨ Features

- **Canonical algorithm, verified:** employed, onlooker and scout bee phases exactly as in the Erciyes University reference code. A test runs both implementations on the same random-number stream and checks that they evaluate identical points and return identical results.
- **scikit-learn API:** `ABCSearchCV` exposes `fit`, `predict`, `score`, `best_params_`, `best_score_` and `best_estimator_`.
- **Mixed search spaces:** `float` (optionally log-scaled), `int` and `categorical` hyperparameters.
- **Reproducible:** pass `random_state` and every run is identical.
- **Evaluation budget:** you set `max_evals`, so the run cost is predictable.
- **Robust:** failing candidates raise a warning instead of being hidden (or stop the search with `error_score="raise"`), and invalid inputs fail fast with clear messages.
- **Convergence history:** every run returns its best-so-far curve for plotting and comparison.
- **Tested:** 48 tests, including the reference-equivalence test, run on every push via GitHub Actions.
- **Lightweight:** depends only on `numpy` and `scikit-learn`.

---

## 📦 Installation

```bash
pip install abcopt
```

```python
from abcopt import ABCSearchCV, CanonicalABCSolver
```

For development:

```bash
git clone https://github.com/yusufkorkmazyigit/abc-optimizer-lib.git
cd abc-optimizer-lib
pip install -e ".[test]"
```

---

## 🚀 Quick Start

### 1. Hyperparameter tuning with `ABCSearchCV`

```python
import lightgbm as lgb
from sklearn.datasets import load_breast_cancer
from abcopt import ABCSearchCV

X, y = load_breast_cancer(return_X_y=True)

param_space = {
    "learning_rate": {"type": "float", "range": (0.01, 0.3), "log": True},
    "n_estimators":  {"type": "int",   "range": (50, 500)},
    "num_leaves":    {"type": "int",   "range": (20, 100)},
    "boosting_type": {"type": "categorical", "choices": ["gbdt", "dart"]},
}

search = ABCSearchCV(
    estimator=lgb.LGBMClassifier(verbosity=-1),
    param_space=param_space,
    cv=3,
    scoring="accuracy",
    pop_size=20,     # number of food sources
    max_evals=100,   # total objective evaluations (each one = one CV run)
    random_state=42, # reproducible search
)
search.fit(X, y)

print("Best CV score:", search.best_score_)
print("Best params:  ", search.best_params_)
y_pred = search.predict(X)   # uses best_estimator_, refit on the full data
```

### 2. Function minimization with `CanonicalABCSolver`

```python
import numpy as np
from abcopt import CanonicalABCSolver

def sphere(x):
    return np.sum(x ** 2)    # global minimum: 0 at x = 0

solver = CanonicalABCSolver(
    objective_func=sphere,
    n_params=3,
    lb=[-10, -10, -10],
    ub=[10, 10, 10],
    max_evals=2000,
    random_state=0,
)
best_x, best_cost, history = solver.solve()
print(f"x* = {best_x}, f(x*) = {best_cost:.5f}")
```

### 3. Plotting convergence

```python
import matplotlib.pyplot as plt

evals, costs = zip(*history)          # solver: best cost so far
# evals, scores = zip(*search.history_) # ABCSearchCV: best CV score so far
plt.plot(evals, costs)
plt.xlabel("Function evaluations")
plt.ylabel("Best cost")
plt.title("ABC convergence")
plt.show()
```

---

## 📖 API Reference

### `CanonicalABCSolver`

```python
CanonicalABCSolver(objective_func, n_params, lb, ub,
                   pop_size=20, max_evals=1000, limit=None, random_state=None)
```

| Parameter | Description |
|---|---|
| `objective_func` | Callable taking a 1-D `np.ndarray` and returning a scalar cost to **minimize** |
| `n_params` | Number of dimensions |
| `lb`, `ub` | Lower / upper bounds per dimension |
| `pop_size` | Number of food sources (SN) = number of employed bees = number of onlooker bees. The colony size (`NumberOfPopulation` in the reference code) is `2 × pop_size`. |
| `max_evals` | Total objective-function evaluation budget |
| `limit` | A source is abandoned once its failed-trial counter reaches `limit` (default: `pop_size × n_params`) |
| `random_state` | Seed (or `np.random.Generator`) for reproducible runs |

`NaN` objective values are treated as the worst possible cost.

`solve()` returns `(best_params, best_cost, history)`, where `history` is a list of `(eval_count, best_cost)` tuples.

### `ABCSearchCV`

```python
ABCSearchCV(estimator, param_space, cv=3, scoring="accuracy",
            pop_size=20, max_evals=200, limit=None, random_state=None,
            n_jobs=None, error_score=np.nan, refit=True, verbose=False)
```

**`param_space`**: one entry per hyperparameter:

| Spec | Example |
|---|---|
| Continuous | `{"type": "float", "range": (0.01, 0.3)}` |
| Continuous, log scale | `{"type": "float", "range": (1e-4, 1e-1), "log": True}` |
| Integer (both ends inclusive) | `{"type": "int", "range": (2, 12)}` |
| Categorical | `{"type": "categorical", "choices": ["gini", "entropy"]}` |

- **`scoring`**: any scikit-learn scorer; higher is better, exactly like `GridSearchCV` (use `neg_*` scorers for errors).
- **`error_score`**: score given to candidates that fail to fit (a `RuntimeWarning` is issued); `"raise"` stops the search instead.
- **`n_jobs`**: parallel jobs inside each cross-validation.
- **After `fit`:** `best_params_`, `best_score_`, `best_estimator_` (when `refit=True`), `history_`, `results_` (every candidate with its score), `n_evals_`.
- **Methods:** `predict`, `predict_proba`, `score`.

---

## 🧠 How ABC Works

ABC models the foraging behaviour of a honey bee colony. Each **food source** is a candidate solution and its **nectar amount** is the solution's fitness. Three kinds of bees share the work:

![One cycle of the canonical ABC](https://raw.githubusercontent.com/yusufkorkmazyigit/abc-optimizer-lib/main/docs/abc_cycle.svg)

| Phase | What happens | Why |
|---|---|---|
| **Employed bees** | Each source changes one random dimension relative to a random other source, `v_ij = x_ij + φ·(x_ij − x_kj)`, `φ ∈ [−1, 1]`, and keeps the better of old and new. | Local search around every source |
| **Onlooker bees** | Sources are visited in turn; source *i* gets an extra try with probability `p_i = 0.9 · fit_i / max(fit) + 0.1`, until `pop_size` tries have been made. | More effort on promising regions |
| **Scout bee** | Once per cycle, the source with the most failed tries is replaced by a random one if its counter has reached `limit`. | Escapes local minima |

Fitness is derived from cost as `1 / (1 + f)` for `f ≥ 0` and `1 + |f|` otherwise.

### Watch it run

A real `CanonicalABCSolver` run (10 food sources, `limit = 12`) on the 2-D Rastrigin function, which has a minimum at every grid point but only one global minimum (green star). Each frame is one phase; numbers next to the sources are their failed-trial counters.

![ABC on the Rastrigin function](https://raw.githubusercontent.com/yusufkorkmazyigit/abc-optimizer-lib/main/docs/abc_animation.gif)

Notice how sources slide along one axis at a time (one dimension per move), how onlookers concentrate on the best sources, and how a scout throws an exhausted source to a random place, even one sitting next to the optimum, since the best solution is always kept in memory. The animation is generated by [`docs/make_animation.py`](https://github.com/yusufkorkmazyigit/abc-optimizer-lib/blob/main/docs/make_animation.py).

---|---|
| **Employed bees** | Each one improves its own food source by perturbing one dimension towards or away from a random neighbour: `v_ij = x_ij + φ·(x_ij − x_kj)`, `φ ∈ [−1, 1]`. Greedy selection keeps the better solution. |
| **Onlooker bees** | Visit the sources in turn and improve source *i* with probability `p_i = 0.9 · fit_i / max(fit) + 0.1`, until `pop_size` onlookers have been placed, so promising regions are exploited more. |
| **Scout bee** | Once per cycle, the source with the most failed trials is replaced by a random one if its counter has reached `limit`, which keeps exploration alive. |

Fitness is derived from cost as `1 / (1 + f)` for `f ≥ 0` and `1 + |f|` otherwise.

---

## 🔬 Case Study: Federated Learning on MedMNIST

The library was used to tune a **federated learning** setup on **PathMNIST** (colon pathology images, 9 classes). The data is split across **5 simulated hospitals** with a **Dirichlet distribution (α = 0.5)**, a non-IID setting where FedAvg with default hyperparameters struggles.

`CanonicalABCSolver` searched the clients' **learning rate** (0.001–0.1) and **SGD momentum** (0.0–0.9), scoring each candidate with a short 3-round FL simulation as a proxy task. The best configuration was then trained for 15 rounds:

| Method | LR | Momentum | Test accuracy (round 15) |
|---|---|---|---|
| Standard FedAvg | 0.010 | 0.50 | 78.05% |
| **ABC-tuned FedAvg** | **0.069** | **0.57** | **85.28%** (+7.2 pts) |

For reference, centralized training with the default hyperparameters reached 80.29%. *(This case study was run with abcopt 0.1, before the reference-alignment changes of 0.2.1.)*

![ABC vs. standard FedAvg](https://raw.githubusercontent.com/yusufkorkmazyigit/abc-optimizer-lib/main/examples/abc_fedavg_final_result.png)

📓 [Full notebook and analysis](https://github.com/yusufkorkmazyigit/abc-optimizer-lib/blob/main/examples/Federated_Learning_MedMNIST_Optimization.ipynb) · Related repo: [fedLearningwithABC](https://github.com/yusufkorkmazyigit/fedLearningwithABC)

---

## 📊 Benchmark

abcopt was compared with **Random Search** and **Optuna TPE** under identical evaluation budgets, using abcopt 0.2.1 ([full notebook](https://github.com/yusufkorkmazyigit/abc-optimizer-lib/blob/main/examples/benchmark/benchmark.ipynb)).

**Black-box functions (D = 10, 10 seeds), best value found (lower is better):**

| Function | ABC @ 1k | TPE @ 1k | ABC @ 10k | TPE @ 2k | Random @ 10k |
|---|---|---|---|---|---|
| Sphere | 1.5 | **0.27** | **4e-11** | 0.22 | 12.3 |
| Rosenbrock | 1.3e3 | **88.5** | **0.51** | 114 | 1.1e4 |
| Rastrigin | **29.9** | 59.7 | **0.09** | 47.3 | 71.2 |
| Ackley | 12.8 | **5.58** | **0.0003** | 5.8 | 16.7 |

![Long-budget convergence](https://raw.githubusercontent.com/yusufkorkmazyigit/abc-optimizer-lib/main/examples/benchmark/benchmark_functions_long.png)

**Takeaways**

- With a **short budget**, TPE is usually ahead; ABC beats Random Search everywhere and wins on the highly multimodal Rastrigin.
- With a **larger budget**, ABC converges to near the global optimum on all four functions while TPE stalls.
- ABC's own overhead is tiny: about **0.02 s vs 24 s** for TPE over 2,000 evaluations.
- For **hyperparameter tuning with only 50 candidates**, the three methods are within noise on test accuracy, and TPE finds the best CV score. For very small budgets of expensive evaluations, TPE is the better choice.

---

## 🗺️ Roadmap

- [x] Categorical and log-scaled hyperparameters
- [x] `random_state` for reproducibility
- [x] Unit tests and continuous integration
- [x] Parallel cross-validation (`n_jobs`)
- [x] Benchmark against Random Search and Optuna TPE
- [ ] Parallel evaluation of candidates within a colony
- [x] PyPI release

Contributions and issues are welcome.

---

## 🧪 Development

```bash
pip install -e ".[test]"
pytest
```

See [CHANGELOG.md](https://github.com/yusufkorkmazyigit/abc-optimizer-lib/blob/main/CHANGELOG.md) for release notes.

---

## 📚 Reference

Karaboğa, D. (2005). *An Idea Based on Honey Bee Swarm for Numerical Optimization*. Technical Report TR06, Erciyes University, Engineering Faculty, Computer Engineering Department.

If you use this library in your research, please cite the original paper above and link to this repository.

## 📝 License

Released under the [MIT License](https://github.com/yusufkorkmazyigit/abc-optimizer-lib/blob/main/LICENSE).

---

<sub>Developed by [Yusuf Korkmazyiğit](https://github.com/yusufkorkmazyigit), Computer Engineering, Erciyes University.</sub>
