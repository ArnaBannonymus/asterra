from __future__ import annotations

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_t = np.asarray(y_true, dtype=float).ravel()
    y_p = np.asarray(y_pred, dtype=float).ravel()
    if y_t.shape != y_p.shape:
        raise ValueError(f"y_true and y_pred must have the same shape. Got {y_t.shape} vs {y_p.shape}.")
    return float(np.sqrt(np.mean((y_t - y_p) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_t = np.asarray(y_true, dtype=float).ravel()
    y_p = np.asarray(y_pred, dtype=float).ravel()
    if y_t.shape != y_p.shape:
        raise ValueError(f"y_true and y_pred must have the same shape. Got {y_t.shape} vs {y_p.shape}.")
    return float(np.mean(np.abs(y_t - y_p)))

