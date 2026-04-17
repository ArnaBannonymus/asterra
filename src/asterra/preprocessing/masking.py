from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted


class MaskInvalid(BaseEstimator, TransformerMixin):
    """Replace non-finite values (NaN/inf) with a fill value."""

    def __init__(self, *, fill_value: float = np.nan):
        self.fill_value = float(fill_value)

    def fit(self, X: Any, y: Any = None) -> "MaskInvalid":  # noqa: D401
        X_arr = check_array(np.asarray(X), ensure_all_finite=False)
        self.n_features_in_ = int(X_arr.shape[1])
        return self

    def transform(self, X: Any) -> np.ndarray:
        check_is_fitted(self, attributes=["n_features_in_"])
        arr = check_array(np.asarray(X), ensure_all_finite=False)
        if int(arr.shape[1]) != int(self.n_features_in_):
            raise ValueError(
                "MaskInvalid.transform received a different number of features than during fit. "
                f"fit n_features_in_={self.n_features_in_} transform n_features={arr.shape[1]}."
            )
        out = arr.copy()
        bad = ~np.isfinite(out)
        if np.any(bad):
            out[bad] = self.fill_value
        return out
