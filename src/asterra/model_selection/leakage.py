from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Iterable

import numpy as np
from scipy.spatial import cKDTree

from asterra.utils.checks import check_2d


def buffered_train_mask(
    *,
    train_coords: np.ndarray,
    test_coords: np.ndarray,
    buffer: float,
) -> np.ndarray:
    """Return a boolean mask keeping only train samples beyond `buffer` from test samples."""

    if buffer < 0:
        raise ValueError("buffer must be >= 0.")
    train = check_2d(np.asarray(train_coords), name="train_coords")
    test = check_2d(np.asarray(test_coords), name="test_coords")
    if train.shape[1] != 2 or test.shape[1] != 2:
        raise ValueError("train_coords and test_coords must have shape (n_samples, 2).")
    if buffer == 0 or train.shape[0] == 0 or test.shape[0] == 0:
        return np.ones(train.shape[0], dtype=bool)
    tree = cKDTree(test.astype(float, copy=False))
    neighbors = tree.query_ball_point(train.astype(float, copy=False), r=float(buffer))
    return np.fromiter((len(nb) == 0 for nb in neighbors), dtype=bool, count=train.shape[0])


def group_overlap(train_groups: Iterable[Hashable], test_groups: Iterable[Hashable]) -> set[Hashable]:
    """Return the set of group IDs appearing in both train and test."""

    tr = _as_group_set(train_groups, name="train_groups")
    te = _as_group_set(test_groups, name="test_groups")
    return tr.intersection(te)


def _as_group_set(groups: Iterable[Hashable], *, name: str) -> set[Hashable]:
    arr = np.asarray(list(groups), dtype=object)
    if arr.ndim == 1:
        return set(arr.tolist())
    if arr.ndim == 2 and arr.shape[1] == 2:
        out: set[Hashable] = set()
        for i in range(arr.shape[0]):
            out.add((arr[i, 0], arr[i, 1]))
        return out
    raise ValueError(f"{name} must be 1D (labels) or 2D with shape (n_samples, 2). Got shape={arr.shape}.")


@dataclass(frozen=True, slots=True)
class LeakageReport:
    """Lightweight leakage report object."""

    overlapping_groups: tuple[Hashable, ...] = ()

    @property
    def has_leakage(self) -> bool:
        return len(self.overlapping_groups) > 0
