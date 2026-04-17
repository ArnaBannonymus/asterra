from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from asterra.data.support import SupportSpec


def _cell_bounds(
    *,
    row: int,
    col: int,
    origin: tuple[float, float],
    resolution: tuple[float, float],
) -> tuple[float, float, float, float]:
    x0, y0 = origin
    dx, dy = resolution
    xmin = x0 + col * dx
    xmax = xmin + dx
    ymin = y0 + row * dy
    ymax = ymin + dy
    return xmin, xmax, ymin, ymax


def _overlap_1d(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def grid_to_grid_overlap(
    *,
    source: SupportSpec,
    target: SupportSpec,
    min_overlap: float = 0.0,
) -> sp.csr_matrix:
    """Compute an overlap-area matrix from a source grid to a target grid.

    The returned matrix has shape (n_target_pixels, n_source_pixels).
    Entry (i, j) is the area of overlap between target cell i and source cell j.

    Notes
    -----
    This implementation assumes simple axis-aligned grids described by (origin, resolution).
    If affine transforms are provided, this function raises.
    """

    if source.kind != "grid" or target.kind != "grid":
        raise ValueError("grid_to_grid_overlap requires both source and target to be grid supports.")
    s_shape, s_res, s_origin = source.grid_params()
    t_shape, t_res, t_origin = target.grid_params()

    sh, sw = s_shape
    th, tw = t_shape
    sdx, sdy = s_res
    tdx, tdy = t_res
    sx0, sy0 = s_origin
    tx0, ty0 = t_origin

    # Precompute extents for quick rejection
    s_xmin, s_xmax = sx0, sx0 + sw * sdx
    s_ymin, s_ymax = sy0, sy0 + sh * sdy

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    for tr in range(th):
        for tc in range(tw):
            t_xmin, t_xmax, t_ymin, t_ymax = _cell_bounds(
                row=tr, col=tc, origin=(tx0, ty0), resolution=(tdx, tdy)
            )
            if t_xmax <= s_xmin or t_xmin >= s_xmax or t_ymax <= s_ymin or t_ymin >= s_ymax:
                continue

            sc0 = int(np.floor((t_xmin - sx0) / sdx))
            sc1 = int(np.ceil((t_xmax - sx0) / sdx)) - 1
            sr0 = int(np.floor((t_ymin - sy0) / sdy))
            sr1 = int(np.ceil((t_ymax - sy0) / sdy)) - 1

            sc0 = max(0, min(sw - 1, sc0))
            sc1 = max(0, min(sw - 1, sc1))
            sr0 = max(0, min(sh - 1, sr0))
            sr1 = max(0, min(sh - 1, sr1))

            t_idx = tr * tw + tc
            for sr in range(sr0, sr1 + 1):
                s_ymin_c = sy0 + sr * sdy
                s_ymax_c = s_ymin_c + sdy
                oy = _overlap_1d(t_ymin, t_ymax, s_ymin_c, s_ymax_c)
                if oy <= 0.0:
                    continue
                for sc in range(sc0, sc1 + 1):
                    s_xmin_c = sx0 + sc * sdx
                    s_xmax_c = s_xmin_c + sdx
                    ox = _overlap_1d(t_xmin, t_xmax, s_xmin_c, s_xmax_c)
                    if ox <= 0.0:
                        continue
                    area = ox * oy
                    if area <= min_overlap:
                        continue
                    s_idx = sr * sw + sc
                    rows.append(t_idx)
                    cols.append(s_idx)
                    data.append(area)

    mat = sp.coo_matrix((data, (rows, cols)), shape=(th * tw, sh * sw), dtype=float).tocsr()
    mat.eliminate_zeros()
    return mat


@dataclass(frozen=True, slots=True)
class OverlapWeighter:
    """Factory for overlap-based weights between supports.

    For v0.1.0, the primary weight is axis-aligned grid overlap area.
    """

    min_overlap: float = 0.0

    def grid_to_grid(self, *, source: SupportSpec, target: SupportSpec) -> sp.csr_matrix:
        return grid_to_grid_overlap(source=source, target=target, min_overlap=self.min_overlap)

