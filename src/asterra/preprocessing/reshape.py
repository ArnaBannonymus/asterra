from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted

from asterra.data.eodata import EOData


class FlattenGrid(BaseEstimator, TransformerMixin):
    """Flatten grid-shaped EO arrays to a 2D sample matrix.

    Accepts:
    - :class:`asterra.data.EOData`
    - NumPy arrays with shape (H, W, B) or (T, H, W, B)
    - 2D arrays (N, B), passed through
    """

    def fit(self, X: Any, y: Any = None) -> "FlattenGrid":  # noqa: D401
        if isinstance(X, EOData):
            self.n_features_in_ = int(X.array.shape[-1])
            return self
        arr = np.asarray(X)
        if arr.ndim == 2:
            arr2 = check_array(arr, ensure_all_finite=False)
            self.n_features_in_ = int(arr2.shape[1])
            return self
        if arr.ndim in (3, 4):
            self.n_features_in_ = int(arr.shape[-1])
            return self
        raise ValueError(f"FlattenGrid.fit expects 2D/3D/4D arrays or EOData. Got shape={arr.shape}.")

    def transform(self, X: Any) -> np.ndarray:
        check_is_fitted(self, attributes=["n_features_in_"])
        if isinstance(X, EOData):
            if int(X.array.shape[-1]) != int(self.n_features_in_):
                raise ValueError(
                    "FlattenGrid.transform received a different number of features than during fit. "
                    f"fit n_features_in_={self.n_features_in_} transform n_features={X.array.shape[-1]}."
                )
            return X.as_samples()
        arr = np.asarray(X)
        if arr.ndim == 2:
            arr2 = check_array(arr, ensure_all_finite=False)
            if int(arr2.shape[1]) != int(self.n_features_in_):
                raise ValueError(
                    "FlattenGrid.transform received a different number of features than during fit. "
                    f"fit n_features_in_={self.n_features_in_} transform n_features={arr2.shape[1]}."
                )
            return arr2
        if arr.ndim == 3:
            h, w, b = arr.shape
            if int(b) != int(self.n_features_in_):
                raise ValueError(
                    "FlattenGrid.transform received a different number of features than during fit. "
                    f"fit n_features_in_={self.n_features_in_} transform n_features={b}."
                )
            return arr.reshape(h * w, b)
        if arr.ndim == 4:
            t, h, w, b = arr.shape
            if int(b) != int(self.n_features_in_):
                raise ValueError(
                    "FlattenGrid.transform received a different number of features than during fit. "
                    f"fit n_features_in_={self.n_features_in_} transform n_features={b}."
                )
            return arr.reshape(t * h * w, b)
        raise ValueError(f"FlattenGrid.transform expects 2D/3D/4D arrays or EOData. Got shape={arr.shape}.")
