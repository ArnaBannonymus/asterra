from __future__ import annotations

import argparse
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "xdg-cache"))

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError as e:  # pragma: no cover
    raise SystemExit("This script requires matplotlib to generate PNGs.") from e

try:
    import rasterio
    from rasterio.windows import Window
except ModuleNotFoundError as e:  # pragma: no cover
    raise SystemExit("This script requires rasterio to read GeoTIFF inputs.") from e


def _import_asterra() -> None:
    """Import asterra, assuming execution from repo root."""

    import sys

    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    import asterra as _  # noqa: F401


@dataclass(frozen=True, slots=True)
class WindowSpec:
    row_off: int
    col_off: int
    height: int
    width: int

    def to_window(self) -> Window:
        return Window(self.col_off, self.row_off, self.width, self.height)


def _support_from_rasterio(ds: Any, *, window: Window) -> "SupportSpec":
    from asterra.data import SupportSpec

    win_transform = rasterio.windows.transform(window, ds.transform)
    dx, dy = ds.res
    origin = (float(win_transform.c), float(win_transform.f))
    return SupportSpec.grid(
        shape=(int(window.height), int(window.width)),
        resolution=(float(dx), float(dy)),
        origin=origin,
        crs=str(ds.crs) if ds.crs is not None else None,
    )


def _read_array(ds: Any, *, window: Window) -> np.ndarray:
    """Read a rasterio dataset window as (H, W, B)."""

    arr = ds.read(window=window)  # (B, H, W)
    if arr.ndim != 3:
        raise ValueError(f"Expected raster read to return (B, H, W). Got shape={arr.shape}.")
    return np.moveaxis(arr, 0, -1)


def _robust_vmin_vmax(x: np.ndarray, p_lo: float = 2.0, p_hi: float = 98.0) -> tuple[float, float]:
    arr = np.asarray(x, dtype=float)
    if arr.size == 0:
        return 0.0, 1.0
    lo = float(np.nanpercentile(arr, p_lo))
    hi = float(np.nanpercentile(arr, p_hi))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo = float(np.nanmin(arr))
        hi = float(np.nanmax(arr))
    if lo == hi:
        hi = lo + 1.0
    return lo, hi


def _to_db(x: np.ndarray, *, eps: float = 1e-8) -> np.ndarray:
    """Convert a nonnegative backscatter-like array to decibels.

    Assumes `x` is in linear scale (e.g., sigma0). Values <= 0 are clipped to `eps`.
    """

    arr = np.asarray(x, dtype="float32", order="C")
    return (10.0 * np.log10(np.clip(arr, float(eps), None))).astype("float32", copy=False)


def _ndvi_from_planet_pf_sr(arr: np.ndarray) -> np.ndarray:
    """Compute NDVI from Planet PF-SR 4-band imagery (assumed order: B, G, R, NIR)."""

    if arr.ndim != 3 or arr.shape[2] != 4:
        raise ValueError(f"Expected Planet array shape (H, W, 4). Got shape={arr.shape}.")
    red = arr[..., 2].astype("float32", copy=False)
    nir = arr[..., 3].astype("float32", copy=False)
    denom = nir + red
    out = np.zeros_like(red, dtype="float32")
    mask = denom != 0.0
    out[mask] = (nir[mask] - red[mask]) / denom[mask]
    return out


def plot_planet_to_s2_ndvi(
    *,
    planet_pf_sr: Path,
    s2_lr_ndvi: Path,
    out_path: Path,
    lr_window_size: int = 64,
) -> None:
    _import_asterra()
    from asterra.data import EOData
    from asterra.support import SupportMatrix

    with rasterio.open(s2_lr_ndvi) as ds_lr, rasterio.open(planet_pf_sr) as ds_ps:
        if ds_lr.crs is not None and ds_ps.crs is not None and ds_lr.crs != ds_ps.crs:
            raise ValueError(f"CRS mismatch: s2={ds_lr.crs} planet={ds_ps.crs}")

        size = int(lr_window_size)
        row_off = max(0, (ds_lr.height - size) // 2)
        col_off = max(0, (ds_lr.width - size) // 2)
        win_lr = WindowSpec(row_off=row_off, col_off=col_off, height=size, width=size).to_window()

        bounds = rasterio.windows.bounds(win_lr, transform=ds_lr.transform)
        win_ps = rasterio.windows.from_bounds(*bounds, transform=ds_ps.transform).round_offsets().round_lengths()

        lr = _read_array(ds_lr, window=win_lr).astype("float32", copy=False)[..., 0]  # (H, W)
        planet = _read_array(ds_ps, window=win_ps).astype("float32", copy=False)  # (H, W, 4)
        planet_ndvi = _ndvi_from_planet_pf_sr(planet)  # (H, W)

        s_lr = _support_from_rasterio(ds_lr, window=win_lr)
        s_ps = _support_from_rasterio(ds_ps, window=win_ps)

    eo_lr = EOData.from_array(lr[..., None], band_schema=["NDVI"], support=s_lr)
    eo_ps = EOData.from_array(planet_ndvi[..., None], band_schema=["NDVI"], support=s_ps)

    M = SupportMatrix.from_grid_to_grid(source=eo_ps.support, target=eo_lr.support, normalize=True)
    proj = M.project_features(eo_ps.as_samples()).reshape(size, size)

    diff = proj - lr
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff**2)))
    nnz_per_row = np.diff(M.matrix.indptr).reshape(size, size)
    row_sum = np.asarray(M.matrix.sum(axis=1)).ravel().reshape(size, size)

    vmin, vmax = _robust_vmin_vmax(np.stack([lr, proj], axis=0))
    dmax = float(np.nanpercentile(np.abs(diff), 98))
    if dmax == 0.0:
        dmax = float(np.max(np.abs(diff)) + 1e-6)

    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    ax = axes[0, 0]
    im = ax.imshow(lr, cmap="RdYlGn", vmin=vmin, vmax=vmax)
    ax.set_title("Sentinel-2 NDVI (10m window)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.axis("off")

    ax = axes[0, 1]
    im = ax.imshow(proj, cmap="RdYlGn", vmin=vmin, vmax=vmax)
    ax.set_title("Planet NDVI → 10m (SupportMatrix)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.axis("off")

    ax = axes[0, 2]
    im = ax.imshow(diff, cmap="coolwarm", vmin=-dmax, vmax=dmax)
    ax.set_title(f"Difference (MAE={mae:.4f}, RMSE={rmse:.4f})")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.axis("off")

    ax = axes[1, 0]
    im = ax.imshow(nnz_per_row, cmap="viridis")
    ax.set_title("SupportMatrix nnz per target pixel")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.axis("off")

    ax = axes[1, 1]
    im = ax.imshow(row_sum, cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_title("SupportMatrix row sum (normalize=True)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.axis("off")

    ax = axes[1, 2]
    r = min(512, M.matrix.shape[0])
    c = min(8000, M.matrix.shape[1])
    ax.spy(M.matrix[:r, :c], markersize=0.5)
    ax.set_title(f"Sparse pattern (top-left {r}×{c})")
    ax.set_xlabel("source index")
    ax.set_ylabel("target index")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.suptitle("Asterra local-dataset projection sanity check", y=1.02, fontsize=14)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_sentinel1_vv_vh_labels(
    *,
    vv_tif: Path,
    vh_tif: Path,
    label_map_tif: Path,
    out_path: Path,
    window_size: int = 256,
) -> None:
    with rasterio.open(vv_tif) as ds_vv, rasterio.open(vh_tif) as ds_vh, rasterio.open(
        label_map_tif
    ) as ds_y:
        if (ds_vv.height, ds_vv.width) != (ds_vh.height, ds_vh.width) or (ds_vv.height, ds_vv.width) != (
            ds_y.height,
            ds_y.width,
        ):
            raise ValueError("Input rasters must have identical shapes.")
        if ds_vv.transform != ds_vh.transform or ds_vv.transform != ds_y.transform:
            raise ValueError("Input rasters must be on the same grid (transform mismatch).")

        size = int(window_size)
        size = int(min(size, ds_vv.height, ds_vv.width))
        if size <= 0:
            raise ValueError(f"window_size must be positive. Got {window_size}.")

        # Pick a window with strong texture/contrast so the SAR layers look like SAR.
        max_row_off = max(0, int(ds_vv.height - size))
        max_col_off = max(0, int(ds_vv.width - size))
        row_offsets = np.unique(np.linspace(0, max_row_off, num=min(5, max_row_off + 1)).round().astype(int))
        col_offsets = np.unique(np.linspace(0, max_col_off, num=min(5, max_col_off + 1)).round().astype(int))
        best_score = -np.inf
        best_row_off, best_col_off = 0, 0
        for row_off in row_offsets:
            for col_off in col_offsets:
                win0 = WindowSpec(
                    row_off=int(row_off),
                    col_off=int(col_off),
                    height=size,
                    width=size,
                ).to_window()
                vv0 = ds_vv.read(1, window=win0).astype("float32", copy=False)
                score = float(np.nanstd(_to_db(vv0)))
                if np.isfinite(score) and score > best_score:
                    best_score = score
                    best_row_off, best_col_off = int(row_off), int(col_off)

        win = WindowSpec(row_off=best_row_off, col_off=best_col_off, height=size, width=size).to_window()

        vv = ds_vv.read(1, window=win).astype("float32", copy=False)
        vh = ds_vh.read(1, window=win).astype("float32", copy=False)
        y = ds_y.read(1, window=win).astype("int32", copy=False)

    vv_db = _to_db(vv)
    vh_db = _to_db(vh)

    vv_lo, vv_hi = _robust_vmin_vmax(vv_db)
    vh_lo, vh_hi = _robust_vmin_vmax(vh_db)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    ax = axes[0]
    im = ax.imshow(vv_db, cmap="gray", vmin=vv_lo, vmax=vv_hi)
    ax.set_title("Sentinel-1 VV (dB)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.axis("off")

    ax = axes[1]
    im = ax.imshow(vh_db, cmap="gray", vmin=vh_lo, vmax=vh_hi)
    ax.set_title("Sentinel-1 VH (dB)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.axis("off")

    ax = axes[2]
    im = ax.imshow(y, cmap="tab20")
    ax.set_title("Label map (window)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.axis("off")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_cvdl_patch(
    *,
    city_dir: Path,
    out_path: Path,
    seed: int = 0,
) -> None:
    hh_path = city_dir / "HH_Complex_Patches.npy"
    hv_path = city_dir / "HV_Complex_Patches.npy"
    y_path = city_dir / "Labels.npy"

    hh = np.load(hh_path, mmap_mode="r")
    hv = np.load(hv_path, mmap_mode="r")
    y = np.load(y_path, mmap_mode="r").astype("int64", copy=False).ravel()

    rng = np.random.RandomState(int(seed))
    idx = int(rng.randint(0, hh.shape[0]))

    hh0 = np.log1p(np.abs(np.asarray(hh[idx]))).astype("float32", copy=False)
    hv0 = np.log1p(np.abs(np.asarray(hv[idx]))).astype("float32", copy=False)
    label = int(y[idx])

    vmin, vmax = _robust_vmin_vmax(np.concatenate([hh0.ravel(), hv0.ravel()]))

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
    ax = axes[0]
    im = ax.imshow(hh0, cmap="magma", vmin=vmin, vmax=vmax)
    ax.set_title(f"HH log1p(|.|) (label={label})")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.axis("off")

    ax = axes[1]
    im = ax.imshow(hv0, cmap="magma", vmin=vmin, vmax=vmax)
    ax.set_title("HV log1p(|.|)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.axis("off")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate README visuals (local-dataset sanity checks).")
    parser.add_argument("--out-dir", type=Path, default=Path("docs/assets"))

    parser.add_argument("--planet-pf-sr", type=Path, required=True)
    parser.add_argument("--s2-lr-ndvi", type=Path, required=True)
    parser.add_argument("--lr-window-size", type=int, default=64)

    parser.add_argument("--s1-vv", type=Path, required=True)
    parser.add_argument("--s1-vh", type=Path, required=True)
    parser.add_argument("--label-map", type=Path, required=True)
    parser.add_argument("--s1-window-size", type=int, default=256)

    parser.add_argument("--cvdl-city-dir", type=Path, required=True)
    parser.add_argument("--cvdl-seed", type=int, default=0)

    args = parser.parse_args()
    out_dir = args.out_dir

    plot_planet_to_s2_ndvi(
        planet_pf_sr=args.planet_pf_sr,
        s2_lr_ndvi=args.s2_lr_ndvi,
        out_path=out_dir / "planet_to_s2_ndvi_window.png",
        lr_window_size=args.lr_window_size,
    )
    plot_sentinel1_vv_vh_labels(
        vv_tif=args.s1_vv,
        vh_tif=args.s1_vh,
        label_map_tif=args.label_map,
        out_path=out_dir / "sentinel1_vv_vh_labelmap.png",
        window_size=args.s1_window_size,
    )
    plot_cvdl_patch(
        city_dir=args.cvdl_city_dir,
        out_path=out_dir / "cvdl_hh_hv_patch.png",
        seed=args.cvdl_seed,
    )

    print("Wrote:")
    print(" -", out_dir / "planet_to_s2_ndvi_window.png")
    print(" -", out_dir / "sentinel1_vv_vh_labelmap.png")
    print(" -", out_dir / "cvdl_hh_hv_patch.png")


if __name__ == "__main__":
    main()
