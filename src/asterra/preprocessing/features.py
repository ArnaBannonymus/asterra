from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from asterra.data.bands import BandSchema
from asterra.data.eodata import EOData


class BandSelector(BaseEstimator, TransformerMixin):
    """Select bands by name.

    This transformer is designed for pipelines that operate on :class:`EOData`.
    If ``band_schema`` is provided, it can also operate on raw NumPy arrays by
    resolving band names to indices.
    """

    def __init__(self, bands: Sequence[str], *, band_schema: BandSchema | Sequence[str] | None = None):
        self.bands = tuple(bands)
        self.band_schema = band_schema
        self._indices: list[int] | None = None

    def fit(self, X: Any, y: Any = None) -> "BandSelector":
        schema = self._resolve_schema(X)
        self._indices = schema.indices(self.bands)
        return self

    def transform(self, X: Any) -> np.ndarray:
        if self._indices is None:
            raise ValueError("BandSelector is not fitted yet. Call fit() first.")
        if isinstance(X, EOData):
            return X.array[..., self._indices].reshape(-1, len(self._indices))
        arr = np.asarray(X)
        if arr.ndim == 2:
            return arr[:, self._indices]
        if arr.ndim == 3:
            h, w, _ = arr.shape
            return arr[..., self._indices].reshape(h * w, len(self._indices))
        raise ValueError("BandSelector.transform expects EOData, (N, B), or (H, W, B) arrays.")

    def _resolve_schema(self, X: Any) -> BandSchema:
        if isinstance(self.band_schema, BandSchema):
            return self.band_schema
        if self.band_schema is not None:
            return BandSchema.from_names(self.band_schema)
        if isinstance(X, EOData):
            return X.band_schema
        raise ValueError("BandSelector requires EOData input or an explicit band_schema.")

