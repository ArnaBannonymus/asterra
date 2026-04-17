# Upstream candidate (draft)

This folder contains materials for a *possible future* discussion with scikit-learn maintainers about extracting a
**generic** subset of Asterra.

Nothing here is accepted into scikit-learn core. This is a preparatory draft only.

## Candidate concept: sparse support operators

The most plausibly generic idea in Asterra is a sparse operator that maps values defined on one “support” to values
defined on another “support”, using overlap or group structure.

In Asterra this is implemented as:
- `asterra.support.SupportMatrix` (SciPy CSR matrix + metadata)
- `asterra.support.SupportProjector` (thin transformer wrapper)

Potential non-EO use cases:
- aggregating sample-level predictions to group/region-level labels
- reconciling measurements on different discretizations of the same domain
- “coarse label” supervision where labels are defined on larger regions than inputs
- leakage-safe evaluation where overlaps create dependence

## Contents

- `api_sketch.md`: an EO-free API sketch for a minimal generic subset
- `extraction_notes.md`: what would need to change to make a standalone generic module
- `draft_sklearn_discussion.md`: a draft discussion/issue text for scikit-learn
- `draft_related_projects_entry.md`: a draft snippet for a scikit-learn “Related Projects” PR
