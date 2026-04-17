# TODO / Implementation plan (0.1.0)

This file tracks the concrete work plan and rationale for the `0.1.0` release.

## Package foundations
- [x] Add `pyproject.toml` with BSD-3-Clause metadata and src-layout packaging
- [x] Add core documentation files (README, design boundaries, sklearn compatibility)
- [x] Add BSD-3-Clause `LICENSE`

## Core data model
- [x] Implement `BandSchema` (band name validation, selection helpers)
- [x] Implement `SupportSpec` for grid / samples / groups supports (pixel-space and optional georeferencing)
- [x] Implement `EOData` (array + `BandSchema` + `SupportSpec`) with `.from_npy` and `.from_array`

## Sparse support operators
- [x] Implement `SupportMatrix` (SciPy sparse CSR + metadata)
- [x] Implement `SupportMatrix.from_grid_to_grid` using rectangle overlap in a common coordinate space
- [x] Implement `SupportMatrix.from_samples_to_groups` for group aggregation
- [x] Implement `SupportProjector` and basic overlap weighting/normalization utilities

## scikit-learn compatibility layer
- [x] Implement `MixedResolutionTransformer` (`fit/transform`, band-aware)
- [x] Implement leakage-aware splitters (`BufferedSpatialKFold`, `TileTimeGroupSplit`)
- [x] Implement basic leakage checks (buffer violations, group overlap checks)

## Metrics
- [x] Implement `SupportAwareRMSE`, `SupportAwareMAE`, and `SupportAwareF1`
- [x] Add sklearn scorer examples and smoke tests

## Examples
- [x] Mixed-resolution Sentinel-2 + PlanetScope synthetic projection
- [x] Sentinel-1/generic SAR VV/VH example
- [x] Generic `.npy` custom-band example
- [x] Spatial CV example with leakage-safe splitters
- [x] Minimal sklearn pipeline example

## Tests and release artifacts
- [x] Add pytest suite covering core objects and projections
- [x] Run tests and capture logs in `build_artifacts/`
- [x] Build wheel and sdist; store outputs in `build_artifacts/`
- [x] Generate `upstream_candidate/` materials for potential future generic extraction
- [x] Fill `RELEASE_SUMMARY.md` with executed commands and results
