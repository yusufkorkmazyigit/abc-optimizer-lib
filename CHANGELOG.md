# Changelog

## 0.2.1

### Changed
- `CanonicalABCSolver` now follows the Erciyes University reference implementation (https://github.com/artificialbeecolony/Python_ABC) step for step:
  - onlooker selection probability `p_i = 0.9 * fit_i / max(fit) + 0.1` (previously `fit_i / sum(fit)`);
  - a food source is abandoned when its trial counter **reaches** `limit` (previously when it exceeded it);
  - random numbers are drawn in the same order as the reference code.
- If every food source has infinite cost, all sources are selected by onlookers instead of dividing by zero.

### Added
- `tests/test_reference_equivalence.py`: runs abcopt and a line-by-line port of the reference code on the same random stream and checks for identical trajectories (18 scenarios).
- Benchmark notebook comparing ABC with Random Search and Optuna TPE (`examples/benchmark/`).

## 0.2.0

### Added
- `random_state` in `CanonicalABCSolver` and `ABCSearchCV` for reproducible runs.
- Categorical hyperparameters: `{"type": "categorical", "choices": [...]}`.
- Log-scaled `float`/`int` hyperparameters: `"log": True`.
- `ABCSearchCV`: `n_jobs`, `error_score`, `refit`, `predict_proba`, `results_` and `n_evals_`.
- Input validation with clear error messages for bounds, budget and `param_space`.
- 30 unit tests (`pytest`) and a GitHub Actions workflow on Python 3.9, 3.11 and 3.12.
- First release on PyPI: `pip install abcopt`.
- `pyproject.toml` packaging and a GitHub Actions workflow that publishes releases to PyPI.

### Changed
- **The package is renamed from `abc_algorithm` to `abcopt`.** Update imports to `from abcopt import ABCSearchCV, CanonicalABCSolver`.
- Failing candidates now emit a `RuntimeWarning` instead of being silently scored as infinitely bad. Use `error_score="raise"` to stop on the first error.
- Integer ranges are now inclusive at both ends with equal probability for every value.
- `ABCSearchCV.history_` stores the best **score** so far (previously the internal cost).
- `ABCSearchCV(verbose=...)` now defaults to `False`.
- Calling `solve()` twice no longer appends to the previous run's history.
- `NaN` objective values are treated as the worst possible cost.
- Neighbour selection in the bee phases now uses a single draw instead of a rejection loop (same distribution).

### Removed
- `setup.py` (replaced by `pyproject.toml`).
