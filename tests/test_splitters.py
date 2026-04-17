from __future__ import annotations

import numpy as np

from asterra.model_selection import BufferedSpatialKFold, TileTimeGroupSplit
from asterra.model_selection.leakage import group_overlap


def test_buffered_spatial_kfold_basic() -> None:
    n = 20
    coords = np.stack([np.linspace(0.0, 10.0, n), np.zeros(n)], axis=1)
    cv = BufferedSpatialKFold(n_splits=4, buffer=0.0)
    splits = list(cv.split(X=np.zeros((n, 1)), groups=coords))
    assert len(splits) == 4
    for tr, te in splits:
        assert len(set(tr).intersection(set(te))) == 0


def test_buffered_spatial_kfold_buffer_excludes() -> None:
    n = 30
    coords = np.stack([np.linspace(0.0, 100.0, n), np.zeros(n)], axis=1)
    cv0 = BufferedSpatialKFold(n_splits=3, buffer=0.0)
    cvb = BufferedSpatialKFold(n_splits=3, buffer=10.0)
    tr0, te0 = next(cv0.split(X=np.zeros((n, 1)), groups=coords))
    trb, teb = next(cvb.split(X=np.zeros((n, 1)), groups=coords))
    assert len(teb) == len(te0)  # same fold sizes before buffering
    assert len(trb) <= len(tr0)


def test_tile_time_group_split_no_group_overlap() -> None:
    n = 12
    tiles = np.array(["T0"] * 6 + ["T1"] * 6, dtype=object)
    times = np.array([0, 1, 0, 1, 0, 1] * 2, dtype=object)
    groups = np.stack([tiles, times], axis=1)

    cv = TileTimeGroupSplit(n_splits=3)
    for tr, te in cv.split(X=np.zeros((n, 1)), groups=groups):
        overlap = group_overlap([tuple(x) for x in groups[tr]], [tuple(x) for x in groups[te]])
        assert len(overlap) == 0
