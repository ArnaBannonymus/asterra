from __future__ import annotations

import numpy as np

from asterra.model_selection import BufferedSpatialKFold, TileTimeGroupSplit
from asterra.model_selection.leakage import group_overlap


def main() -> None:
    # Synthetic sample coordinates in a 2D plane
    n = 30
    x = np.linspace(0.0, 100.0, n)
    coords = np.stack([x, np.zeros_like(x)], axis=1)

    cv = BufferedSpatialKFold(n_splits=3, buffer=10.0)
    for i, (tr, te) in enumerate(cv.split(X=np.zeros((n, 1)), groups=coords)):
        print(f"BufferedSpatialKFold fold={i} train={len(tr)} test={len(te)}")

    # Tile/time grouping example: each sample belongs to a (tile, time) group
    tiles = np.array(["T0"] * 10 + ["T1"] * 10 + ["T2"] * 10, dtype=object)
    times = np.array([0] * 5 + [1] * 5 + [0] * 5 + [1] * 5 + [0] * 5 + [1] * 5, dtype=object)
    groups = np.stack([tiles, times], axis=1)

    gcv = TileTimeGroupSplit(n_splits=3)
    for i, (tr, te) in enumerate(gcv.split(X=np.zeros((n, 1)), groups=groups)):
        overlap = group_overlap([tuple(x) for x in groups[tr]], [tuple(x) for x in groups[te]])
        print(f"TileTimeGroupSplit fold={i} group_overlap={len(overlap)} train={len(tr)} test={len(te)}")


if __name__ == "__main__":
    main()

