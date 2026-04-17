from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score


def f1_binary(y_true: np.ndarray, y_pred: np.ndarray, *, threshold: float = 0.5) -> float:
    y_t = np.asarray(y_true).ravel()
    y_p = np.asarray(y_pred).ravel()
    if y_t.shape != y_p.shape:
        raise ValueError(f"y_true and y_pred must have the same shape. Got {y_t.shape} vs {y_p.shape}.")
    if y_p.dtype.kind in {"f"}:
        y_p = (y_p >= float(threshold)).astype(int)
    return float(f1_score(y_t, y_p))

