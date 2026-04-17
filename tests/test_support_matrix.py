from __future__ import annotations

import numpy as np

from asterra.data import SupportSpec
from asterra.support import SupportMatrix


def test_support_matrix_grid_to_grid_mean_aggregation() -> None:
    # Source grid: 2x2 with unit pixels
    src = SupportSpec.grid(shape=(2, 2), resolution=(1.0, 1.0), origin=(0.0, 0.0))
    # Target grid: 1x1 with a pixel covering the full 2x2 area
    tgt = SupportSpec.grid(shape=(1, 1), resolution=(2.0, 2.0), origin=(0.0, 0.0))
    M = SupportMatrix.from_grid_to_grid(source=src, target=tgt, normalize=True)

    X_src = np.array([[1.0], [2.0], [3.0], [4.0]], dtype=float)
    X_tgt = M.project_features(X_src)
    assert X_tgt.shape == (1, 1)
    assert float(X_tgt[0, 0]) == 2.5


def test_support_matrix_grid_to_grid_upsampling() -> None:
    src = SupportSpec.grid(shape=(1, 1), resolution=(2.0, 2.0), origin=(0.0, 0.0))
    tgt = SupportSpec.grid(shape=(2, 2), resolution=(1.0, 1.0), origin=(0.0, 0.0))
    M = SupportMatrix.from_grid_to_grid(source=src, target=tgt, normalize=True)

    X_src = np.array([[5.0]], dtype=float)
    X_tgt = M.project_features(X_src)
    assert X_tgt.shape == (4, 1)
    assert np.allclose(X_tgt.ravel(), 5.0)


def test_support_matrix_samples_to_groups_mean() -> None:
    M = SupportMatrix.from_samples_to_groups(group_ids=["a", "a", "b"], normalize=True)
    X = np.array([[1.0], [3.0], [10.0]])
    Xg = M.project_features(X)
    assert Xg.shape == (2, 1)
    assert np.allclose(Xg.ravel(), [2.0, 10.0])

