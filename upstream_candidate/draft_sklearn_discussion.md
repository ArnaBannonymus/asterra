# Draft scikit-learn discussion text (generic subset only)

Title: Proposal (discussion): sparse “support” projection operator for group/overlap aggregation

Hi scikit-learn maintainers,

I’m exploring a small, generic utility concept that has appeared across multiple applied workflows:
representing the mapping from one set of sample units to another as a sparse matrix, then using it to project
features/predictions/labels in a consistent way.

The core object would be a thin wrapper around a SciPy CSR matrix `M` shaped `(n_target, n_source)` with helpers
for:
- projecting `X_source` (2D) and `y_source` (1D) to the target support (`M @ X`, `M @ y`)
- constructing a “samples → groups” mapping from group IDs (with optional row-normalization for mean aggregation)

Potential use cases (non-domain-specific):
- aggregating instance-level predictions to group-level metrics/labels
- coarse supervision (labels at a higher support than inputs)
- evaluation protocols where overlap/group structure is required to interpret scores

I have implemented and tested a version of this concept in an external domain package (Earth observation), keeping
EO-specific pieces separate. Before investing further, I’d like feedback on:

1. whether scikit-learn would consider a small sparse projection/aggregation helper in core (or prefers a separate
   utility package)
2. what minimal API would be acceptable and what scope constraints are required
3. whether the overlap-based “grid → grid” constructors are out of scope for core

If there’s interest, I can provide:
- a minimal, EO-free implementation
- unit tests and performance notes
- examples demonstrating utility outside EO

Thanks for your time.

