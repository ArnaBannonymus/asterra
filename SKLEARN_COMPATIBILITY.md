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
- run clone/pipeline smoke checks on selected transformers

### Notes (2026-04-17)

- Clone smoke checks (`sklearn.base.clone`) passed for `MaskInvalid` and `FlattenGrid`.
- A `Pipeline` smoke test passed for `MixedResolutionTransformer` + `Ridge`.
- An attempted `check_estimator(...)` run in this environment aborted with `SIGABRT` due to an OpenMP runtime error
  (`OMP: Error #179: Function Can't open SHM2 failed`). The attempt and stderr are captured in
  `build_artifacts/compatibility/sklearn_compat.txt`.

Full compliance with every estimator check is not guaranteed for EO-first objects that accept `EOData` as `X`, and
some environments may require OpenMP configuration to run the full check suite.

## Deviations and rationale

- Some components accept rich Python objects (`EOData`) instead of strict 2D numeric arrays. This is compatible
  with scikit-learn pipelines, but not every generic estimator check assumes such inputs.
- Support-aware metrics may require a `SupportMatrix` or group structure; users typically wrap them with
  `functools.partial` or create a scorer closure.

## Statement on scikit-learn inclusion

Asterra is an external library. Any future scikit-learn inclusion would require separate design discussion and
community review, and is not completed by this repository.
