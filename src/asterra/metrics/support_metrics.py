from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import f1_score

from asterra.support.matrix import SupportMatrix


@dataclass(frozen=True, slots=True)
class SupportAwareRMSE:
    """Support-aware RMSE with optional projection."""

    support_matrix: SupportMatrix | None = None

    def __call__(self, y_true: Any, y_pred: Any) -> float:
        y_t = np.asarray(y_true, dtype=float).ravel()
        y_p = np.asarray(y_pred, dtype=float).ravel()
        if self.support_matrix is not None:
            if y_p.shape[0] != self.support_matrix.source.n_items:
                raise ValueError("y_pred length must match support_matrix.source.n_items.")
            if y_t.shape[0] != self.support_matrix.target.n_items:
                raise ValueError("y_true length must match support_matrix.target.n_items.")
            y_p = self.support_matrix.project_labels(y_p)
        if y_t.shape != y_p.shape:
            raise ValueError(f"y_true and y_pred must have the same shape. Got {y_t.shape} vs {y_p.shape}.")
        return float(np.sqrt(np.mean((y_t - y_p) ** 2)))


@dataclass(frozen=True, slots=True)
class SupportAwareMAE:
    """Support-aware MAE with optional projection."""

    support_matrix: SupportMatrix | None = None

    def __call__(self, y_true: Any, y_pred: Any) -> float:
        y_t = np.asarray(y_true, dtype=float).ravel()
        y_p = np.asarray(y_pred, dtype=float).ravel()
        if self.support_matrix is not None:
            if y_p.shape[0] != self.support_matrix.source.n_items:
                raise ValueError("y_pred length must match support_matrix.source.n_items.")
            if y_t.shape[0] != self.support_matrix.target.n_items:
                raise ValueError("y_true length must match support_matrix.target.n_items.")
            y_p = self.support_matrix.project_labels(y_p)
        if y_t.shape != y_p.shape:
            raise ValueError(f"y_true and y_pred must have the same shape. Got {y_t.shape} vs {y_p.shape}.")
        return float(np.mean(np.abs(y_t - y_p)))


@dataclass(frozen=True, slots=True)
class SupportAwareF1:
    """Support-aware binary F1 with optional projection.

    Notes
    -----
    If a support matrix is provided, predictions are projected onto the label
    support (target). Projected predictions are thresholded to obtain binary
    labels.
    """

    support_matrix: SupportMatrix | None = None
    threshold: float = 0.5

    def __call__(self, y_true: Any, y_pred: Any) -> float:
        y_t = np.asarray(y_true).ravel()
        y_p = np.asarray(y_pred).ravel()
        if self.support_matrix is not None:
            if y_p.shape[0] != self.support_matrix.source.n_items:
                raise ValueError("y_pred length must match support_matrix.source.n_items.")
            if y_t.shape[0] != self.support_matrix.target.n_items:
                raise ValueError("y_true length must match support_matrix.target.n_items.")
            y_p = self.support_matrix.project_labels(y_p.astype(float, copy=False))
        if y_t.shape != y_p.shape:
            raise ValueError(f"y_true and y_pred must have the same shape. Got {y_t.shape} vs {y_p.shape}.")
        if y_p.dtype.kind in {"f"}:
            y_p = (y_p >= float(self.threshold)).astype(int)
        return float(f1_score(y_t, y_p))

