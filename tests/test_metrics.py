from __future__ import annotations

import numpy as np

from asterra.metrics import SupportAwareF1, SupportAwareMAE, SupportAwareRMSE
from asterra.support import SupportMatrix


def test_support_aware_rmse_no_projection() -> None:
    m = SupportAwareRMSE()
    assert m([0.0, 1.0], [0.0, 3.0]) == np.sqrt(2.0)


def test_support_aware_rmse_with_projection() -> None:
    # 3 samples -> 2 groups: group a has samples 0,1; group b has sample 2
    M = SupportMatrix.from_samples_to_groups(group_ids=["a", "a", "b"], normalize=True)
    metric = SupportAwareRMSE(support_matrix=M)

    y_true_groups = np.array([2.0, 10.0])
    y_pred_samples = np.array([1.0, 3.0, 10.0])
    assert metric(y_true_groups, y_pred_samples) == 0.0


def test_support_aware_mae_with_projection() -> None:
    M = SupportMatrix.from_samples_to_groups(group_ids=["a", "a", "b"], normalize=True)
    metric = SupportAwareMAE(support_matrix=M)
    y_true_groups = np.array([2.0, 10.0])
    y_pred_samples = np.array([0.0, 4.0, 8.0])
    # projected preds are [2.0, 8.0] => MAE = (0 + 2)/2
    assert metric(y_true_groups, y_pred_samples) == 1.0


def test_support_aware_f1_with_projection() -> None:
    M = SupportMatrix.from_samples_to_groups(group_ids=["a", "a", "b"], normalize=True)
    metric = SupportAwareF1(support_matrix=M, threshold=0.5)
    y_true_groups = np.array([1, 0])
    y_pred_samples = np.array([1.0, 1.0, 0.0])
    # projected preds: [1.0, 0.0] => perfect
    assert metric(y_true_groups, y_pred_samples) == 1.0

