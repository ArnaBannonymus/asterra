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
    raise SystemExit(
        "This example requires matplotlib for visualization. Install it and rerun (e.g. `conda install matplotlib`)."
    ) from e

try:
    import rasterio
    from rasterio.windows import Window
except ModuleNotFoundError as e:  # pragma: no cover
    raise SystemExit("This example requires rasterio. Install it and rerun (e.g. `conda install rasterio`).") from e

def _ensure_asterra_importable() -> None:
    """Import asterra, assuming either an installed package or execution from repo root."""

    try:
        import asterra as _  # noqa: F401

        return
    except ModuleNotFoundError:
        import sys

        root = Path(__file__).resolve().parents[1]
        src = root / "src"
        if src.is_dir() and str(src) not in sys.path:
            sys.path.insert(0, str(src))
        try:
            import asterra as _  # noqa: F401

            return
        except ModuleNotFoundError as e:
            raise SystemExit(
                "asterra is not importable. Install it (e.g. `python -m pip install asterra`) or run this example "
                "from the source tree with `PYTHONPATH=src`."
            ) from e


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
        resolution=(float(dx), float(abs(dy))),
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


def main() -> None:
    _ensure_asterra_importable()
    from asterra.data import EOData
    from asterra.support import SupportMatrix

    parser = argparse.ArgumentParser(
        description=(
            "Sentinel-2 (NDVI GeoTIFF) demo with SupportMatrix visualization.\n\n"
            "This script reads a Sentinel-2 NDVI raster as the target grid, computes NDVI from a Planet PF-SR "
            "4-band raster on a higher-resolution grid, then projects Planet NDVI onto the Sentinel-2 grid using "
            "asterra.support.SupportMatrix.\n\n"
            "Outputs: a PNG figure with the NDVI windows, projection error, SupportMatrix nnz-per-pixel, row-sum "
            "sanity check, and a sparse-pattern 'spy' view of the matrix."
        )
    )
    parser.add_argument("--s2-lr-ndvi", type=Path, required=True, help="Path to Sentinel-2 NDVI GeoTIFF (10m).")
    parser.add_argument("--planet-pf-sr", type=Path, required=True, help="Path to Planet PF-SR 4-band GeoTIFF (3m).")
    parser.add_argument(
        "--window-size",
        type=int,
        default=64,
        help="Window size (in Sentinel-2 pixels) for the demo figure.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("asterra_sentinel2_supportmatrix_demo.png"),
        help="Output PNG path.",
    )
    parser.add_argument(
        "--spy-rows",
        type=int,
        default=512,
        help="Rows to include in the sparse-pattern matrix view (top-left subset).",
    )
    parser.add_argument(
        "--spy-cols",
        type=int,
        default=8000,
        help="Columns to include in the sparse-pattern matrix view (top-left subset).",
    )
    parser.add_argument(
        "--matrix-npz",
        type=Path,
        default=None,
        help="Optional path to save the SupportMatrix as a SciPy .npz (sparse).",
    )
    args = parser.parse_args()

    size = int(args.window_size)
    if size <= 0:
        raise SystemExit("--window-size must be positive.")

    with rasterio.open(args.s2_lr_ndvi) as ds_lr, rasterio.open(args.planet_pf_sr) as ds_ps:
        if ds_lr.crs is not None and ds_ps.crs is not None and ds_lr.crs != ds_ps.crs:
            raise SystemExit(f"CRS mismatch: s2={ds_lr.crs} planet={ds_ps.crs}")

        row_off = max(0, (ds_lr.height - size) // 2)
        col_off = max(0, (ds_lr.width - size) // 2)
        win_lr = WindowSpec(row_off=row_off, col_off=col_off, height=size, width=size).to_window()

        bounds = rasterio.windows.bounds(win_lr, transform=ds_lr.transform)
        win_ps = rasterio.windows.from_bounds(*bounds, transform=ds_ps.transform).round_offsets().round_lengths()

        lr = _read_array(ds_lr, window=win_lr).astype("float32", copy=False)[..., 0]  # (H, W)
        planet = _read_array(ds_ps, window=win_ps).astype("float32", copy=False)  # (H, W, 4)
        planet_ndvi = _ndvi_from_planet_pf_sr(planet)

        s_lr = _support_from_rasterio(ds_lr, window=win_lr)
        s_ps = _support_from_rasterio(ds_ps, window=win_ps)

    eo_lr = EOData.from_array(lr[..., None], band_schema=["NDVI"], support=s_lr)
    eo_ps = EOData.from_array(planet_ndvi[..., None], band_schema=["NDVI"], support=s_ps)

    M = SupportMatrix.from_grid_to_grid(source=eo_ps.support, target=eo_lr.support, normalize=True)
    ndvi_ps_on_lr = M.project_features(eo_ps.as_samples()).reshape(size, size)

    mae = float(np.mean(np.abs(ndvi_ps_on_lr - lr)))
    rmse = float(np.sqrt(np.mean((ndvi_ps_on_lr - lr) ** 2)))

    nnz_per_row = np.diff(M.matrix.indptr).astype(int, copy=False)
    nnz_map = nnz_per_row.reshape(size, size)
    nnz_min = int(nnz_per_row.min())
    nnz_median = int(np.median(nnz_per_row))
    nnz_max = int(nnz_per_row.max())
    off = int(np.sum(nnz_per_row != nnz_max))

    row_sum = np.asarray(M.matrix.sum(axis=1)).ravel()
    rs_min = float(row_sum.min())
    rs_mean = float(row_sum.mean())
    rs_max = float(row_sum.max())

    print("Inputs:")
    print("  s2_lr_ndvi:", args.s2_lr_ndvi)
    print("  planet_pf_sr:", args.planet_pf_sr)
    print("Windows:")
    print("  s2 window:", win_lr)
    print("  planet window:", win_ps)
    print("Supports:")
    print("  s2 (target):", eo_lr.support.grid_shape, eo_lr.support.resolution, eo_lr.support.origin)
    print("  planet (source):", eo_ps.support.grid_shape, eo_ps.support.resolution, eo_ps.support.origin)
    print("SupportMatrix:")
    print("  shape:", tuple(M.matrix.shape))
    print("  nnz:", int(M.matrix.nnz))
    print(
        "  nnz/row: min=",
        nnz_min,
        "median=",
        nnz_median,
        "max=",
        nnz_max,
    )
    print("  row_sum: min=", rs_min, "max=", rs_max, "mean=", rs_mean)
    print("Projection quality (Planet NDVI -> S2 NDVI window):")
    print("  MAE:", mae)
    print("  RMSE:", rmse)

    vmin, vmax = _robust_vmin_vmax(np.stack([lr, ndvi_ps_on_lr], axis=0))
    dmax = float(_robust_vmin_vmax(ndvi_ps_on_lr - lr)[1])

    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)

    ax = axes[0, 0]
    im = ax.imshow(lr, cmap="RdYlGn", vmin=vmin, vmax=vmax)
    ax.set_title("Sentinel-2 NDVI (target)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.axis("off")

    ax = axes[0, 1]
    im = ax.imshow(ndvi_ps_on_lr, cmap="RdYlGn", vmin=vmin, vmax=vmax)
    ax.set_title("Planet NDVI → S2 (SupportMatrix)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.axis("off")

    ax = axes[0, 2]
    im = ax.imshow(ndvi_ps_on_lr - lr, cmap="coolwarm", vmin=-dmax, vmax=dmax)
    ax.set_title(f"Difference (MAE={mae:.4f}, RMSE={rmse:.4f})")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.axis("off")

    ax = axes[1, 0]
    im = ax.imshow(
        nnz_map,
        cmap="viridis",
        vmin=float(nnz_min) - 0.5,
        vmax=float(nnz_max) + 0.5,
        interpolation="nearest",
    )
    ax.set_title(f"nnz/target (off={off})\nmin={nnz_min} med={nnz_median} max={nnz_max}")
    fig.colorbar(
        im,
        ax=ax,
        fraction=0.046,
        pad=0.04,
        ticks=list(range(nnz_min, nnz_max + 1)),
    )
    rr, cc = np.where(nnz_map != nnz_max)
    if rr.size:
        ax.scatter(cc, rr, s=6, c="black", marker=".", alpha=0.7)
    ax.axis("off")

    ax = axes[1, 1]
    if (rs_max - rs_min) <= 1e-12:
        pad = 1e-3
        ax.bar(
            [rs_mean],
            [row_sum.size],
            width=2.0 * pad,
            color="tab:blue",
            alpha=0.85,
            edgecolor="white",
        )
        ax.set_xlim(rs_mean - 2.5 * pad, rs_mean + 2.5 * pad)
    else:
        pad = max(1e-6, 0.05 * (rs_max - rs_min))
        ax.hist(
            row_sum,
            bins=25,
            range=(rs_min - pad, rs_max + pad),
            color="tab:blue",
            alpha=0.85,
            edgecolor="white",
        )
        ax.set_xlim(rs_min - pad, rs_max + pad)
    ax.axvline(1.0, color="black", linewidth=1.0, linestyle="--", alpha=0.8)
    ax.set_title(f"row sum (normalize=True)\nmin={rs_min:.6f} mean={rs_mean:.6f} max={rs_max:.6f}")
    ax.set_xlabel("row sum")
    ax.set_ylabel("count")

    ax = axes[1, 2]
    rr_nz, cc_nz = M.matrix.nonzero()
    bins_target = 128
    bins_source = 256
    H, _, _ = np.histogram2d(
        rr_nz,
        cc_nz,
        bins=(bins_target, bins_source),
        range=((0, M.matrix.shape[0]), (0, M.matrix.shape[1])),
    )
    dens = np.log10(1.0 + H)
    im = ax.imshow(dens, cmap="magma", aspect="auto", origin="lower", interpolation="nearest")
    ax.set_title(f"SupportMatrix nnz density (log10(1+count))\nshape={M.matrix.shape} nnz={int(M.matrix.nnz)}")
    ax.set_xlabel("source bin")
    ax.set_ylabel("target bin")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.suptitle("Asterra local-dataset Sentinel-2 demo", y=1.02, fontsize=14)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

    print("Wrote:", out_path)

    if args.matrix_npz is not None:
        try:
            from scipy.sparse import save_npz
        except ModuleNotFoundError as e:  # pragma: no cover
            raise SystemExit("Saving a sparse matrix requires SciPy.") from e
        path = Path(args.matrix_npz)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_npz(path, M.matrix)
        print("Wrote:", path)


if __name__ == "__main__":
    main()
