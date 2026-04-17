# Design

## Problem: mixed spatial supports

EO datasets are frequently combined across:
- sensors with different spatial resolutions
- labels defined on parcels/tiles/scenes/time windows
- patch-based workflows with overlap and neighborhood dependence

In such settings, *“just resample”* is often insufficient:
- label support and feature support may not match
- patch overlap creates leakage across train/test
- metrics may be computed on a support different from the label definition

## Core idea: `SupportMatrix`

`SupportMatrix` is a sparse operator that maps values from one support to another.

Examples:
- **grid → grid**: overlap-area weights between two grids (e.g., 5m pixels aggregated to 10m pixels)
- **samples → groups**: membership weights mapping samples to parcel/tile groups

Projections are sparse matrix multiplications:

```
X_target = M @ X_source
```

where `M` is `(n_target, n_source)`.

## Why sparse

Support mismatch naturally produces sparse relationships:
- a coarse pixel overlaps only a small neighborhood of fine pixels
- a parcel contains only a subset of samples

Sparse matrices keep mappings explicit, testable, and efficient enough for common workflows.

## scikit-learn compatibility

Where appropriate, Asterra components implement the estimator API:
- transformers: `fit/transform`, work in `Pipeline`
- splitters: `BaseCrossValidator` interface
- metrics: callables compatible with `make_scorer` (often wrapped to include a support matrix)

See `SKLEARN_COMPATIBILITY.md` for notes and constraints.

