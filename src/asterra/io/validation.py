from __future__ import annotations

from typing import Sequence

import numpy as np


def validate_supported_shape(arr: np.ndarray) -> None:
    a = np.asarray(arr)
    if a.ndim not in (2, 3, 4):
        raise ValueError(f"Unsupported array ndim={a.ndim}. Expected 2, 3, or 4.")
    if a.shape[-1] <= 0:
        raise ValueError("Band dimension must be positive.")


def validate_band_names(band_names: Sequence[str], *, n_bands: int) -> None:
    if len(band_names) != n_bands:
        raise ValueError(f"band_names length must match number of bands ({n_bands}). Got {len(band_names)}.")
    if any((not isinstance(b, str) or b.strip() == "") for b in band_names):
        raise ValueError("band_names must be non-empty strings.")

