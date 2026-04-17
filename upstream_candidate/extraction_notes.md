# Extraction notes

## What is currently EO-specific

In Asterra, grid overlap construction depends on:
- `SupportSpec` fields like `origin` and `resolution`
- grid semantics (2D shapes, flattening conventions)

Those pieces are EO-adjacent but could be generalized (regular grids exist outside EO).

## What is plausibly generic

- Sparse operator container + projection helpers:
  - `SupportMatrix` (as a CSR matrix wrapper)
  - row-normalization utilities (`row_normalize_csr`)
- Group aggregation constructor:
  - `from_samples_to_groups(group_ids, normalize=True)`

## Minimal extraction strategy

1. Move the sparse wrapper and group constructor into a small module:
   - no EO metadata
   - no sensor presets
   - no CRS/transform logic
2. Keep grid-overlap builders external until:
   - grid support metadata is proven broadly useful beyond EO
   - performance and correctness are benchmarked on larger grids
3. Provide strong tests:
   - shape validation
   - deterministic projection behavior
   - empty-row behavior and normalization semantics

## Estimator check notes

The operator itself is not an estimator. The transformer wrapper can be made compatible with `Pipeline`.
Full `check_estimator` compatibility should be demonstrated in a clean environment (OpenMP runtime conflicts can
abort the process on some platforms).

