from __future__ import annotations

from collections import Counter
from typing import Iterable, TypeVar

import numpy as np

T = TypeVar("T")


def check_band_names_unique(band_names: Iterable[str]) -> None:
    names = list(band_names)
    counts = Counter(names)
    dup = [k for (k, v) in counts.items() if v > 1]
    if dup:
        raise ValueError(f"Band names must be unique. Duplicates: {sorted(dup)!r}")


def check_1d(x: np.ndarray, *, name: str = "x") -> np.ndarray:
    arr = np.asarray(x)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D. Got shape={arr.shape}.")
    return arr


def check_2d(x: np.ndarray, *, name: str = "x") -> np.ndarray:
    arr = np.asarray(x)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D. Got shape={arr.shape}.")
    return arr


def check_non_empty(seq: Iterable[T], *, name: str) -> list[T]:
    items = list(seq)
    if len(items) == 0:
        raise ValueError(f"{name} must be non-empty.")
    return items

