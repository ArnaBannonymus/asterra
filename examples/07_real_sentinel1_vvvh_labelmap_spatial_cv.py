from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

try:
    import rasterio
except ModuleNotFoundError as e:  # pragma: no cover
    raise SystemExit(
        "This example requires rasterio. Install it (e.g. `conda install rasterio`) and rerun."
    ) from e

from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import f1_score

from asterra.model_selection import BufferedSpatialKFold


def _pixel_center_coords(ds: Any) -> np.ndarray:
    """Return pixel-center coordinates as (n_pixels, 2) for a north-up raster."""

    t = ds.transform
    dx, dy = ds.res
    # Use positive dy with a "pixel-down" convention consistent across rasters for distance computations.
    dy = float(abs(dy))
    xs = t.c + (np.arange(ds.width, dtype=float) + 0.5) * float(dx)
    ys = t.f + (np.arange(ds.height, dtype=float) + 0.5) * dy
    X, Y = np.meshgrid(xs, ys)
    return np.stack([X.ravel(), Y.ravel()], axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Sentinel-1 VV/VH + label_map example with leakage-aware spatial CV.\n"
            "Reads GeoTIFFs, flattens to samples, then runs BufferedSpatialKFold."
        )
    )
    parser.add_argument("--vv", type=Path, required=True, help="Path to Sentinel-1 VV GeoTIFF.")
    parser.add_argument("--vh", type=Path, required=True, help="Path to Sentinel-1 VH GeoTIFF.")
    parser.add_argument(
        "--label-map",
        type=Path,
        required=True,
        help="Path to label_map GeoTIFF (integer class labels).",
    )
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--buffer", type=float, default=30.0, help="Buffer distance in dataset units (meters).")
    parser.add_argument("--max-samples", type=int, default=50000, help="Max pixels sampled for the demo.")
    args = parser.parse_args()

    with rasterio.open(args.vv) as ds_vv, rasterio.open(args.vh) as ds_vh, rasterio.open(
        args.label_map
    ) as ds_y:
        if (ds_vv.height, ds_vv.width) != (ds_vh.height, ds_vh.width) or (ds_vv.height, ds_vv.width) != (
            ds_y.height,
            ds_y.width,
        ):
            raise SystemExit("Input rasters must have identical shapes.")
        if ds_vv.transform != ds_vh.transform or ds_vv.transform != ds_y.transform:
            raise SystemExit("Input rasters must be on the same grid (transform mismatch).")

        vv = ds_vv.read(1).astype("float32", copy=False)
        vh = ds_vh.read(1).astype("float32", copy=False)
        y = ds_y.read(1).astype("int64", copy=False)

        coords = _pixel_center_coords(ds_vv)

    X = np.stack([vv.ravel(), vh.ravel()], axis=1)
    y1 = y.ravel()

    # Filter invalid labels (convention: negative => ignore). Adjust if your label_map uses a different nodata.
    keep = y1 >= 0
    X = X[keep]
    y1 = y1[keep]
    coords = coords[keep]

    # Downsample for speed
    if args.max_samples > 0 and X.shape[0] > args.max_samples:
        rng = np.random.RandomState(0)
        idx = rng.choice(X.shape[0], size=int(args.max_samples), replace=False)
        X = X[idx]
        y1 = y1[idx]
        coords = coords[idx]

    # Binarize for a simple demo metric if labels are multi-class
    y_bin = (y1 > 0).astype(int)

    cv = BufferedSpatialKFold(n_splits=int(args.n_splits), buffer=float(args.buffer))
    f1s: list[float] = []
    for fold, (tr, te) in enumerate(cv.split(X=np.zeros((X.shape[0], 1)), groups=coords)):
        # RidgeClassifier is used here because some OpenMP-linked estimators can abort in constrained
        # environments (see `build_artifacts/compatibility/sklearn_compat.txt` for an example).
        clf = RidgeClassifier()
        clf.fit(X[tr], y_bin[tr])
        pred = clf.predict(X[te])
        f1 = float(f1_score(y_bin[te], pred))
        f1s.append(f1)
        print(f"fold={fold} train={len(tr)} test={len(te)} f1={f1:.4f}")

    print("mean_f1:", float(np.mean(f1s)))


if __name__ == "__main__":
    main()
