from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from asterra.data.eodata import EOData
from asterra.data.support import SupportSpec

from .matrix import SupportMatrix


class SupportProjector(BaseEstimator, TransformerMixin):
    """scikit-learn-style transformer projecting features via a :class:`SupportMatrix`."""

    def __init__(self, support_matrix: SupportMatrix):
        self.support_matrix = support_matrix

    def fit(self, X: Any, y: Any = None) -> "SupportProjector":  # noqa: D401
        return self

    def transform(self, X: Any) -> np.ndarray:
        if isinstance(X, EOData):
            X_src = X.as_samples()
        else:
            X_arr = np.asarray(X)
            if X_arr.ndim == 3:
                h, w, b = X_arr.shape
                X_src = X_arr.reshape(h * w, b)
            elif X_arr.ndim == 2:
                X_src = X_arr
            else:
                raise ValueError("SupportProjector.transform expects EOData, (N, B), or (H, W, B) arrays.")
        return self.support_matrix.project_features(X_src)


@dataclass
class _FittedMixedResolution:
    target: SupportSpec
    matrices: tuple[SupportMatrix, ...]
    band_names: tuple[str, ...]


class MixedResolutionTransformer(BaseEstimator, TransformerMixin):
    """Project one or more EO grids onto a common target support and concatenate bands.

    This transformer is designed for *support-aware* feature construction in
    scikit-learn pipelines.

    Parameters
    ----------
    target_support:
        The target support to which all inputs are projected.
    normalize:
        Whether to row-normalize grid overlap matrices.
    """

    def __init__(self, target_support: SupportSpec, *, normalize: bool = True):
        self.target_support = target_support
        self.normalize = normalize
        self._fitted: _FittedMixedResolution | None = None

    def fit(self, X: Any, y: Any = None) -> "MixedResolutionTransformer":
        eodata_list = _coerce_eodata_list(X)
        mats: list[SupportMatrix] = []
        band_names: list[str] = []
        for eo in eodata_list:
            if eo.support.kind != "grid" or self.target_support.kind != "grid":
                raise ValueError("MixedResolutionTransformer currently supports only grid→grid projections.")
            mats.append(
                SupportMatrix.from_grid_to_grid(source=eo.support, target=self.target_support, normalize=self.normalize)
            )
            band_names.extend(eo.band_schema.band_names)
        self._fitted = _FittedMixedResolution(
            target=self.target_support, matrices=tuple(mats), band_names=tuple(band_names)
        )
        return self

    def transform(self, X: Any) -> np.ndarray:
        if self._fitted is None:
            raise ValueError("MixedResolutionTransformer is not fitted yet. Call fit() first.")
        eodata_list = _coerce_eodata_list(X)
        if len(eodata_list) != len(self._fitted.matrices):
            raise ValueError(
                "Number of inputs at transform time must match fit time. "
                f"fit inputs={len(self._fitted.matrices)} transform inputs={len(eodata_list)}."
            )
        projected: list[np.ndarray] = []
        for eo, mat in zip(eodata_list, self._fitted.matrices, strict=True):
            projected.append(mat.project_features(eo.as_samples()))
        return np.concatenate(projected, axis=1)

    @property
    def output_band_names_(self) -> tuple[str, ...]:
        if self._fitted is None:
            raise AttributeError("output_band_names_ is only available after fit().")
        return self._fitted.band_names


def _coerce_eodata_list(X: Any) -> list[EOData]:
    if isinstance(X, EOData):
        return [X]
    if isinstance(X, Sequence) and not isinstance(X, (str, bytes)):
        if len(X) == 0:
            raise ValueError("Expected at least one EOData input.")
        out: list[EOData] = []
        for item in X:
            if not isinstance(item, EOData):
                raise ValueError("MixedResolutionTransformer expects EOData or a sequence of EOData.")
            out.append(item)
        return out
    raise ValueError("MixedResolutionTransformer expects EOData or a sequence of EOData.")

