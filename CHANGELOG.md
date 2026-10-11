# Changelog

## 0.2.0

### Added
- `random_state` in `CanonicalABCSolver` and `ABCSearchCV` for reproducible runs.
- Categorical hyperparameters: `{"type": "categorical", "choices": [...]}`.
- Log-scaled `float`/`int` hyperparameters: `"log": True`.
- `ABCSearchCV`: `n_jobs`, `error_score`, `refit`, `predict_proba`, `results_` and `n_evals_`.
- Input validation with clear error messages for bounds, budget and `param_space`.
- 30 unit tests (`pytest`) and a GitHub Actions workflow on Python 3.9, 3.11 and 3.12.
- `pyproject.toml` packaging (distribution name `abc-optimizer-lib`; the import name is still `abc_algorithm`).

### Changed
- Failing candidates now emit a `RuntimeWarning` instead of being silently scored as infinitely bad. Use `error_score="raise"` to stop on the first error.
- Integer ranges are now inclusive at both ends with equal probability for every value.
- `ABCSearchCV.history_` stores the best **score** so far (previously the internal cost).
- `ABCSearchCV(verbose=...)` now defaults to `False`.
- Calling `solve()` twice no longer appends to the previous run's history.
- `NaN` objective values are treated as the worst possible cost.
- Neighbour selection in the bee phases now uses a single draw instead of a rejection loop (same distribution).

### Removed
- `setup.py` (replaced by `pyproject.toml`).
