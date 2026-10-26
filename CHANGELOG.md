# Changelog

## Unreleased

### Added
- `callbacks` argument on `CanonicalABCSolver` and `ABCSearchCV` (#1). Each callback is called after every ABC cycle with a `CycleState` (cycle, evaluations used, budget, best cost, best params and, for `ABCSearchCV`, best score and decoded params). Returning `True` stops the run after that cycle; the best configuration so far is refit as usual.
- `EarlyStopping(patience, min_delta)`: stops when the best cost has not improved for `patience` cycles.
- `CanonicalABCSolver.n_cycles` / `stopped_early` and `ABCSearchCV.n_cycles_` / `stopped_early_`.

Callbacks never draw random numbers, so a run without a stop request is identical to one without callbacks; the reference-equivalence tests are unchanged.

## 0.2.2

### Added
- `ABCSearchCV(cache=True)`: a configuration that ABC proposes again (common with integer and categorical parameters) reuses its stored score instead of being cross-validated again. The search and its result are unchanged; in our tests 46–75% of candidates on discrete spaces were repeats, and tuning ran about twice as fast.
- scikit-learn compatibility: `cv_results_` in `GridSearchCV` format, `best_index_`, `n_splits_`, `refit_time_`, `n_unique_evals_`, `classes_`, `predict_log_proba`, `decision_function`, `transform`.
- `fit(X, y, groups=None, **fit_params)` forwards `fit_params` (for example `sample_weight`) to the estimator.
- `results_` entries record whether a candidate was served from the cache.
- `CITATION.cff`, Python version classifiers.

### Changed
- `ABCSearchCV.score` now uses the `scoring` parameter, as `GridSearchCV.score` does (previously the estimator's default `score`).
- README restructured: animation, quick start, algorithm overview, benchmark, API reference, notes and limitations, citation.

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
