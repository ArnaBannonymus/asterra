# API sketch (generic subset)

This sketch intentionally avoids EO terms (sensors, CRS, rasters). The goal is to describe a **generic** sparse
projection/aggregation operator that could apply to many domains.

## `SupportMatrix` (generic)

Proposed responsibilities:
- carry a sparse matrix `M` shaped `(n_target, n_source)`
- provide lightweight validation and safe projection helpers

Minimal API sketch:

```python
from dataclasses import dataclass
import numpy as np
import scipy.sparse as sp

@dataclass(frozen=True)
class SupportMatrix:
    matrix: sp.csr_matrix
    normalized: bool = True

    def project(self, X_source: np.ndarray) -> np.ndarray:
        ...

    def project_1d(self, y_source: np.ndarray) -> np.ndarray:
        ...

    def transpose(self) -> "SupportMatrix":
        ...

    @classmethod
    def from_samples_to_groups(cls, group_ids, *, normalize: bool = True) -> "SupportMatrix":
        ...
```

## `SupportProjector` (generic transformer)

```python
from sklearn.base import BaseEstimator, TransformerMixin

class SupportProjector(BaseEstimator, TransformerMixin):
    def __init__(self, support_matrix: SupportMatrix): ...
    def fit(self, X, y=None): return self
    def transform(self, X): ...
```

## Open questions for a future discussion

- Should a generic `SupportMatrix` live in scikit-learn core, or in a separate utility package?
- What guarantees are needed around normalization (mean vs sum) and empty rows?
- What minimal metadata (if any) belongs in core vs in domain packages?

