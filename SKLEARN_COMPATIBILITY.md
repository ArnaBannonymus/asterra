# scikit-learn compatibility

Asterra aims to be scikit-learn-compatible where appropriate. It is **not** part of scikit-learn core.

## Estimator API choices

- Transformers follow `BaseEstimator` + `TransformerMixin` and implement:
  - `fit`, `transform`, and `fit_transform`
  - `get_params` / `set_params` via `BaseEstimator`
- Cross-validation splitters implement the `BaseCrossValidator` interface.
- Metrics are provided as callables/classes suitable for `sklearn.metrics.make_scorer`.

## Pipeline and model_selection integration

- `MixedResolutionTransformer` can be used in a `sklearn.pipeline.Pipeline`.
- Splitters can be used in `cross_val_score` / `GridSearchCV` as long as the passed `X` provides the needed
  support metadata (e.g., `EOData` or arrays plus `groups` as documented).

## Compatibility checks attempted

The test suite includes smoke tests that:
- import and instantiate estimators
- run `sklearn.utils.estimator_checks.check_estimator` on selected transformers where feasible

Full compliance with every estimator check is not guaranteed for EO-first objects that accept `EOData` as `X`.

## Deviations and rationale

- Some components accept rich Python objects (`EOData`) instead of strict 2D numeric arrays. This is compatible
  with scikit-learn pipelines, but not every generic estimator check assumes such inputs.
- Support-aware metrics may require a `SupportMatrix` or group structure; users typically wrap them with
  `functools.partial` or create a scorer closure.

## Statement on scikit-learn inclusion

Asterra is an external library. Any future scikit-learn inclusion would require separate design discussion and
community review, and is not completed by this repository.

