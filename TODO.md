# TODO / Implementation plan (0.1.0)

This file tracks the concrete work plan and rationale for the `0.1.0` release.

## Package foundations
- [ ] Add `pyproject.toml` with BSD-3-Clause metadata and src-layout packaging
- [ ] Add core documentation files (README, design boundaries, sklearn compatibility)
- [ ] Add BSD-3-Clause `LICENSE`

## Core data model
- [ ] Implement `BandSchema` (band name validation, selection helpers)
- [ ] Implement `SupportSpec` for grid / samples / groups supports (pixel-space and optional georeferencing)
- [ ] Implement `EOData` (array + `BandSchema` + `SupportSpec`) with `.from_npy` and `.from_array`

## Sparse support operators
- [ ] Implement `SupportMatrix` (SciPy sparse CSR + metadata)
- [ ] Implement `SupportMatrix.from_grid_to_grid` using rectangle overlap in a common coordinate space
- [ ] Implement `SupportMatrix.from_samples_to_groups` for group aggregation
- [ ] Implement `SupportProjector` and basic overlap weighting/normalization utilities

## scikit-learn compatibility layer
- [ ] Implement `MixedResolutionTransformer` (`fit/transform`, band-aware)
- [ ] Implement leakage-aware splitters (`BufferedSpatialKFold`, `TileTimeGroupSplit`)
- [ ] Implement basic leakage checks (buffer violations, group overlap checks)

## Metrics
- [ ] Implement `SupportAwareRMSE`, `SupportAwareMAE`, and `SupportAwareF1`
- [ ] Add sklearn scorer examples and smoke tests

## Examples
- [ ] Mixed-resolution Sentinel-2 + PlanetScope synthetic projection
- [ ] Sentinel-1/generic SAR VV/VH example
- [ ] Generic `.npy` custom-band example
- [ ] Spatial CV example with leakage-safe splitters
- [ ] Minimal sklearn pipeline example

## Tests and release artifacts
- [ ] Add pytest suite covering core objects and projections
- [ ] Run tests and capture logs in `build_artifacts/`
- [ ] Build wheel and sdist; store outputs in `build_artifacts/`
- [ ] Generate `upstream_candidate/` materials for potential future generic extraction
- [ ] Fill `RELEASE_SUMMARY.md` with executed commands and results

