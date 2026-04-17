# Release summary

## Package

- Name: `asterra`
- Version: `0.1.2`
- Tagline: Support-aware machine learning for Earth observation
- License: BSD-3-Clause (`LICENSE`, `license = "BSD-3-Clause"` in `pyproject.toml`)

## Implemented modules

- `asterra.data`: `BandSchema`, `SupportSpec`, `EOData`
- `asterra.support`: `SupportMatrix`, `SupportProjector`, `OverlapWeighter`, `MixedResolutionTransformer`
- `asterra.io`: `.npy` loader + sensor presets (Sentinel-1/2, PlanetScope, NISAR-style SAR)
- `asterra.model_selection`: `BufferedSpatialKFold`, `TileTimeGroupSplit`, leakage utilities
- `asterra.metrics`: `SupportAwareRMSE`, `SupportAwareMAE`, `SupportAwareF1`
- `asterra.preprocessing`: `FlattenGrid`, `MaskInvalid`, `BandSelector`

## Tests

- Command: `pytest -ra`
- Result: PASS (see `build_artifacts/test_logs/pytest_0.1.1.txt`)

## Examples

Example outputs were generated with `PYTHONPATH=src` and saved to:
- `build_artifacts/example_logs/01_mixed_resolution_s2_planetscope.txt`
- `build_artifacts/example_logs/02_sentinel1_sar_vvvh.txt`
- `build_artifacts/example_logs/03_generic_npy_custom_bands.txt`
- `build_artifacts/example_logs/04_spatial_cv_splitters.txt`
- `build_artifacts/example_logs/05_sklearn_pipeline_minimal.txt`
- `build_artifacts/example_logs/06_real_mixed_resolution_planet_to_s2_ndvi.txt`
- `build_artifacts/example_logs/07_real_sentinel1_vvvh_labelmap_spatial_cv.txt`
- `build_artifacts/example_logs/08_real_s1slc_cvdl_complex_patches.txt`

## Build artifacts

- Build command: `python -m build --outdir build_artifacts/dist`
- Outputs:
  - `build_artifacts/dist/asterra-0.1.2-py3-none-any.whl`
  - `build_artifacts/dist/asterra-0.1.2.tar.gz`
- Build log: `build_artifacts/build_logs/build_0.1.2.txt`

## scikit-learn compatibility checks

- Clone + pipeline smoke checks: OK
- `check_estimator(...)` attempt: aborted with `SIGABRT` due to OpenMP runtime error; captured in
  `build_artifacts/compatibility/sklearn_compat.txt`

## GitHub / PyPI status

- GitHub remote: `https://github.com/ArnaBannonymus/asterra.git`
- GitHub push: succeeded (`main`, `codex/v0.1.0`, and tag `v0.1.0`)
- PyPI publish: via GitHub Actions trusted publishing on tag push (`.github/workflows/release.yml`).
  Status must be verified in GitHub Actions / PyPI for this environment.

### Next commands (when configured)

- (Optional) Open a PR from `codex/v0.1.0` to `main`.
- Configure PyPI trusted publishing for this GitHub repo, then push a release tag (workflow: `.github/workflows/release.yml`):
  - `git tag -a v0.1.2 -m "asterra 0.1.2"`
  - `git push origin v0.1.2`

## Prepared for potential future upstreaming

- `UPSTREAMING.md`, `DESIGN_BOUNDARIES.md`, and `upstream_candidate/` drafts document which components are EO-specific
  vs potentially generic, and include a draft scikit-learn discussion text.

## Recommended next improvements

- Expand `SupportSpec` / overlap builders to support affine transforms and optional geospatial dependencies.
- Add PEP 660 editable install support (or switch to a standard build backend once SPDX license-expression is widely
  supported by backends in your target environment).
- Add coverage reporting and a small smoke test for the on-disk examples (kept optional).
