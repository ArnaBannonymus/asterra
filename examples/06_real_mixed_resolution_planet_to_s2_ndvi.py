from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import rasterio
    from rasterio.windows import Window
except ModuleNotFoundError as e:  # pragma: no cover
    raise SystemExit(
        "This example requires rasterio. Install it (e.g. `conda install rasterio`) and rerun."
    ) from e

from asterra.data import EOData, SupportSpec
from asterra.support import SupportMatrix


@dataclass(frozen=True, slots=True)
class WindowSpec:
    row_off: int
    col_off: int
    height: int
    width: int

    def to_window(self) -> Window:
        return Window(self.col_off, self.row_off, self.width, self.height)


def support_from_rasterio(ds: Any, *, window: Window) -> SupportSpec:
    win_transform = rasterio.windows.transform(window, ds.transform)
    dx, dy = ds.res
    origin = (float(win_transform.c), float(win_transform.f))
    return SupportSpec.grid(
        shape=(int(window.height), int(window.width)),
        resolution=(float(dx), float(dy)),
        origin=origin,
        crs=str(ds.crs) if ds.crs is not None else None,
    )


def read_array(ds: Any, *, window: Window) -> np.ndarray:
    """Read a rasterio dataset window as (H, W, B)."""

    arr = ds.read(window=window)  # (B, H, W)
    if arr.ndim != 3:
        raise ValueError(f"Expected raster read to return (B, H, W). Got shape={arr.shape}.")
    return np.moveaxis(arr, 0, -1)


def ndvi_from_planet_pf_sr(arr: np.ndarray) -> np.ndarray:
    """Compute NDVI from Planet PF-SR 4-band imagery (assumed order: B, G, R, NIR)."""

    if arr.ndim != 3 or arr.shape[2] != 4:
        raise ValueError(f"Expected Planet array shape (H, W, 4). Got shape={arr.shape}.")
    red = arr[..., 2].astype("float32", copy=False)
    nir = arr[..., 3].astype("float32", copy=False)
    denom = nir + red
    out = np.zeros_like(red, dtype="float32")
    mask = denom != 0.0
    out[mask] = (nir[mask] - red[mask]) / denom[mask]
    return out[..., None]  # (H, W, 1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Mixed-resolution example using real GeoTIFFs:\n"
            "- Planet PF-SR (3m, 4-band)\n"
            "- Sentinel-2 NDVI (10m, 1-band)\n\n"
            "The example reads a small spatial window, computes NDVI at 3m, then projects it onto the 10m grid "
            "using asterra.support.SupportMatrix."
        )
    )
    parser.add_argument(
        "--planet-pf-sr",
        type=Path,
        required=True,
        help="Path to Planet PF-SR 4-band GeoTIFF (e.g., .../PF-SR/2019-03-04.tif).",
    )
    parser.add_argument(
        "--s2-lr-ndvi",
        type=Path,
        required=True,
        help="Path to Sentinel-2 low-res NDVI GeoTIFF (10m) (e.g., .../LR_NDVI_2019-03-04.tif).",
    )
    parser.add_argument(
        "--lr-window-size",
        type=int,
        default=64,
        help="Window size (in Sentinel-2 10m pixels) to read and evaluate.",
    )
    args = parser.parse_args()

    with rasterio.open(args.s2_lr_ndvi) as ds_lr, rasterio.open(args.planet_pf_sr) as ds_ps:
        if ds_lr.crs is not None and ds_ps.crs is not None and ds_lr.crs != ds_ps.crs:
            raise SystemExit(f"CRS mismatch: s2={ds_lr.crs} planet={ds_ps.crs}")

        # Choose a centered LR window
        size = int(args.lr_window_size)
        if size <= 0:
            raise SystemExit("--lr-window-size must be positive.")
        row_off = max(0, (ds_lr.height - size) // 2)
        col_off = max(0, (ds_lr.width - size) // 2)
        win_lr = WindowSpec(row_off=row_off, col_off=col_off, height=size, width=size).to_window()

        # Convert LR bounds to the corresponding Planet window
        b = rasterio.windows.bounds(win_lr, transform=ds_lr.transform)
        win_ps = rasterio.windows.from_bounds(*b, transform=ds_ps.transform).round_offsets().round_lengths()

        # Read arrays as (H, W, B)
        lr = read_array(ds_lr, window=win_lr).astype("float32", copy=False)  # (H, W, 1)
        planet = read_array(ds_ps, window=win_ps).astype("float32", copy=False)  # (H, W, 4)

        planet_ndvi = ndvi_from_planet_pf_sr(planet)  # (H, W, 1)

        s_lr = support_from_rasterio(ds_lr, window=win_lr)
        s_ps = support_from_rasterio(ds_ps, window=win_ps)

    eo_lr = EOData.from_array(lr, band_schema=["NDVI"], support=s_lr)
    eo_ps_ndvi = EOData.from_array(planet_ndvi, band_schema=["NDVI"], support=s_ps)

    M = SupportMatrix.from_grid_to_grid(source=eo_ps_ndvi.support, target=eo_lr.support, normalize=True)
    ndvi_ps_on_lr = M.project_features(eo_ps_ndvi.as_samples()).reshape(size, size)
    ndvi_lr = eo_lr.array[..., 0]

    mae = float(np.mean(np.abs(ndvi_ps_on_lr - ndvi_lr)))
    rmse = float(np.sqrt(np.mean((ndvi_ps_on_lr - ndvi_lr) ** 2)))

    print("Inputs:")
    print("  planet_pf_sr:", args.planet_pf_sr)
    print("  s2_lr_ndvi:", args.s2_lr_ndvi)
    print("Windows:")
    print("  lr window:", win_lr)
    print("  planet window:", win_ps)
    print("Supports:")
    print("  planet (source):", eo_ps_ndvi.support.grid_shape, eo_ps_ndvi.support.resolution, eo_ps_ndvi.support.origin)
    print("  s2 lr (target):", eo_lr.support.grid_shape, eo_lr.support.resolution, eo_lr.support.origin)
    print("SupportMatrix:", M.matrix.shape, "nnz=", int(M.matrix.nnz), "normalized=", M.normalized)
    print("Projection quality (Planet NDVI -> S2 10m NDVI window):")
    print("  MAE:", mae)
    print("  RMSE:", rmse)


if __name__ == "__main__":
    main()

