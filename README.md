# 🐝 abcopt

**Canonical Artificial Bee Colony (ABC) optimization in pure Python, with a scikit-learn compatible hyperparameter search.**

[![PyPI](https://img.shields.io/pypi/v/abcopt)](https://pypi.org/project/abcopt/)
[![tests](https://github.com/yusufkorkmazyigit/abc-optimizer-lib/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/yusufkorkmazyigit/abc-optimizer-lib/actions/workflows/tests.yml)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![scikit-learn](https://img.shields.io/badge/sklearn-compatible-orange)
![Dependencies](https://img.shields.io/badge/deps-numpy%20%7C%20scikit--learn-lightgrey)

**abcopt** is a faithful implementation of the **Artificial Bee Colony** algorithm proposed by **Derviş Karaboğa (2005)** at Erciyes University. It is useful for two things:

1. **Black-box optimization:** minimize any objective function within box constraints, with no gradients required.
2. **Hyperparameter tuning:** a drop-in alternative to `GridSearchCV` / `RandomizedSearchCV` for any scikit-learn compatible estimator (scikit-learn, LightGBM, XGBoost, …).

---

## ✨ Features

- **Canonical algorithm:** employed, onlooker (roulette-wheel selection) and scout bee phases, following the original paper; one dimension is perturbed per move.
- **scikit-learn API:** `ABCSearchCV` exposes `fit`, `predict`, `score`, `best_params_`, `best_score_` and `best_estimator_`.
- **Mixed search spaces:** `float` (optionally log-scaled), `int` and `categorical` hyperparameters.
- **Reproducible:** pass `random_state` and every run is identical.
- **Evaluation budget:** you set `max_evals`, so the run cost is predictable.
- **Robust:** failing candidates raise a warning instead of being hidden (or stop the search with `error_score="raise"`), and invalid inputs fail fast with clear messages.
- **Convergence history:** every run returns its best-so-far curve for plotting and comparison.
- **Tested:** 30 unit tests run on every push via GitHub Actions.
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
| `pop_size` | Number of food sources (SN) |
| `max_evals` | Total objective-function evaluation budget |
| `limit` | Trials without improvement before a source is abandoned (default: `pop_size × n_params`) |
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

ABC models the foraging behaviour of a honey bee colony. Each **food source** is a candidate solution, and its **nectar amount** is the solution's fitness.

| Phase | Role |
|---|---|
| **Employed bees** | Each one improves its own food source by perturbing one dimension towards or away from a random neighbour: `v_ij = x_ij + φ·(x_ij − x_kj)`, `φ ∈ [−1, 1]`. Greedy selection keeps the better solution. |
| **Onlooker bees** | Pick sources with probability proportional to fitness (roulette wheel), so promising regions are exploited more. |
| **Scout bees** | A source that has not improved for `limit` trials is abandoned and replaced by a random one, which keeps exploration alive. |

Fitness is derived from cost as `1 / (1 + f)` for `f ≥ 0` and `1 + |f|` otherwise.

---

## 🔬 Case Study: Federated Learning on MedMNIST

The library was used to tune a **federated learning** setup on **PathMNIST** (colon pathology images, 9 classes). The data is split across **5 simulated hospitals** with a **Dirichlet distribution (α = 0.5)**, a non-IID setting where FedAvg with default hyperparameters struggles.

`CanonicalABCSolver` searched the clients' **learning rate** (0.001–0.1) and **SGD momentum** (0.0–0.9), scoring each candidate with a short 3-round FL simulation as a proxy task. The best configuration was then trained for 15 rounds:

| Method | LR | Momentum | Test accuracy (round 15) |
|---|---|---|---|
| Standard FedAvg | 0.010 | 0.50 | 78.05% |
| **ABC-tuned FedAvg** | **0.069** | **0.57** | **85.28%** (+7.2 pts) |

For reference, centralized training with the default hyperparameters reached 80.29%.

![ABC vs. standard FedAvg](https://raw.githubusercontent.com/yusufkorkmazyigit/abc-optimizer-lib/main/examples/abc_fedavg_final_result.png)

📓 [Full notebook and analysis](https://github.com/yusufkorkmazyigit/abc-optimizer-lib/blob/main/examples/Federated_Learning_MedMNIST_Optimization.ipynb) · Related repo: [fedLearningwithABC](https://github.com/yusufkorkmazyigit/fedLearningwithABC)

---

## 🗺️ Roadmap

- [x] Categorical and log-scaled hyperparameters
- [x] `random_state` for reproducibility
- [x] Unit tests and continuous integration
- [x] Parallel cross-validation (`n_jobs`)
- [ ] Benchmarks against `GridSearchCV`, `RandomizedSearchCV` and Optuna
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
