# Upstreaming notes (scikit-learn)

Asterra is an external package first. It is **not** part of scikit-learn, and no component here should be
presented as accepted into scikit-learn core without a separate community process.

## EO-specific components (remain external)

- Sensor band presets (`asterra.io.sensors`)
- EO-centric data model glue (`EOData`, `BandSchema`, EO-focused support specs)
- EO leakage semantics beyond generic grouping/buffering

## Potentially generic components (candidates for future discussion)

These components are designed to be *potentially generic* (but are shipped in Asterra first):

- `SupportMatrix` as a sparse operator mapping one sample support to another
  - grid-to-grid overlap and samples-to-groups aggregation
- `SupportProjector` as a thin wrapper around sparse projections with normalization controls
- splitters that combine grouping and spatial buffering (if expressed in generic terms)
- support-aware scorers that project predictions onto label supports before computing a metric

## Why external first

- The support-mismatch problem is common in EO, but generality and API stability need real-world validation.
- scikit-learn requires strong evidence of broad usefulness and maintenance commitment.
- EO-specific details should not shape scikit-learn core APIs.

## Evidence needed before scikit-learn inclusion is realistic

- multiple independent, non-EO use cases demonstrating generic value
- stable API with user adoption and maintenance history
- clear performance characteristics and sparse-operator correctness tests
- agreement with scikit-learn maintainers on scope and API design

## Candidate shortlist for discussion

See `upstream_candidate/` for:
- a concise summary of candidate generic components
- API sketches and extraction notes
- draft discussion/issue text

## Draft plan for a future discussion

0. (Optional) Submit a small docs-only PR to list Asterra under scikit-learn “Related Projects” (no core changes).
1. Validate the sparse support operator APIs in Asterra across EO workflows.
2. Identify a minimal generic subset (no sensor/geospatial assumptions).
3. Prepare a standalone repository/module with tests and docs focused on generic use.
4. Open a scikit-learn discussion proposing only the generic subset, with:
   - motivating examples beyond EO (group aggregation, coarse labeling, multi-resolution sampling)
   - estimator/check compatibility notes
   - performance considerations
