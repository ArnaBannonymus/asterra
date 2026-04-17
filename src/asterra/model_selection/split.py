from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np
from scipy.spatial import cKDTree
from sklearn.model_selection import BaseCrossValidator, GroupKFold

from asterra.data.eodata import EOData


def _extract_coords(X: Any, groups: Any) -> np.ndarray:
    if isinstance(X, EOData) and X.support.kind == "samples" and X.support.coords is not None:
        return np.asarray(X.support.coords, dtype=float)
    if groups is None:
        raise ValueError(
            "BufferedSpatialKFold requires coordinates. Provide EOData with support.coords, "
            "or pass coords via the `groups` argument (shape (n_samples, 2))."
        )
    coords = np.asarray(groups)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coords must have shape (n_samples, 2) when passed via groups.")
    if not np.issubdtype(coords.dtype, np.number):
        raise ValueError("coords must be numeric.")
    return coords.astype(float, copy=False)


@dataclass
class BufferedSpatialKFold(BaseCrossValidator):
    """Spatial K-fold with an exclusion buffer around the test fold.

    This splitter first creates folds using a simple spatial blocking strategy:
    samples are sorted by x-coordinate and split into contiguous blocks.

    A buffer distance (in the same units as ``coords``) is then applied:
    any training sample within ``buffer`` of any test sample is removed from
    that training split.
    """

    n_splits: int = 5
    buffer: float = 0.0

    def split(self, X: Any, y: Any = None, groups: Any = None) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        coords = _extract_coords(X, groups)
        n = coords.shape[0]
        if self.n_splits < 2:
            raise ValueError("n_splits must be >= 2.")
        if self.n_splits > n:
            raise ValueError("n_splits cannot be greater than n_samples.")
        if self.buffer < 0:
            raise ValueError("buffer must be >= 0.")

        order = np.argsort(coords[:, 0], kind="mergesort")
        fold_sizes = np.full(self.n_splits, n // self.n_splits, dtype=int)
        fold_sizes[: n % self.n_splits] += 1
        current = 0

        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            test_idx = order[start:stop]
            train_idx = np.concatenate([order[:start], order[stop:]])
            current = stop

            if self.buffer > 0 and train_idx.size > 0 and test_idx.size > 0:
                tree = cKDTree(coords[test_idx])
                neighbors = tree.query_ball_point(coords[train_idx], r=self.buffer)
                keep = np.fromiter((len(nb) == 0 for nb in neighbors), dtype=bool, count=train_idx.size)
                train_idx = train_idx[keep]

            yield train_idx, test_idx

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:  # noqa: D401
        return self.n_splits


@dataclass
class TileTimeGroupSplit(BaseCrossValidator):
    """Group-aware splitting for (tile, time) identifiers.

    The `groups` argument must be provided and should identify each sample's
    (tile, time) membership. Each unique (tile, time) pair is kept within a
    single fold.
    """

    n_splits: int = 5

    def split(self, X: Any, y: Any = None, groups: Any = None) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        if groups is None:
            raise ValueError("TileTimeGroupSplit requires `groups` with (tile, time) identifiers.")
        g = np.asarray(groups, dtype=object)
        if g.ndim == 2 and g.shape[1] == 2:
            n_samples = int(g.shape[0])
            combined = np.empty(n_samples, dtype=object)
            for i in range(n_samples):
                combined[i] = (g[i, 0], g[i, 1])
        elif g.ndim == 1:
            combined = g
        else:
            raise ValueError("groups must be 1D of tuples or 2D with shape (n_samples, 2).")
        gkf = GroupKFold(n_splits=self.n_splits)
        n_samples = combined.shape[0]
        dummy_X = np.zeros((n_samples, 1), dtype=float)
        yield from gkf.split(dummy_X, y=y, groups=combined)

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:  # noqa: D401
        return self.n_splits
