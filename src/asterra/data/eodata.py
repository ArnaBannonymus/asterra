from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .bands import BandSchema
from .support import SupportSpec


def _as_path(path: str | Path) -> Path:
    return path if isinstance(path, Path) else Path(path)


def _validate_array_shape(arr: np.ndarray) -> None:
    if arr.ndim not in (2, 3, 4):
        raise ValueError(
            "EOData arrays must have shape (N, B), (H, W, B), or (T, H, W, B). "
            f"Got array with shape={arr.shape}."
        )
    if arr.ndim == 2:
        if arr.shape[0] <= 0 or arr.shape[1] <= 0:
            raise ValueError("EOData array shape (N, B) must have positive N and B.")
    if arr.ndim == 3:
        if arr.shape[0] <= 0 or arr.shape[1] <= 0 or arr.shape[2] <= 0:
            raise ValueError("EOData array shape (H, W, B) must have positive H, W, and B.")
    if arr.ndim == 4:
        if arr.shape[0] <= 0 or arr.shape[1] <= 0 or arr.shape[2] <= 0 or arr.shape[3] <= 0:
            raise ValueError("EOData array shape (T, H, W, B) must have positive dimensions.")


@dataclass(frozen=True, slots=True)
class EOData:
    """An EO dataset represented as array + band schema + support metadata.

    Parameters
    ----------
    array:
        The underlying data array.
        Supported shapes: (N, B), (H, W, B), (T, H, W, B).
    band_schema:
        Description of the band axis.
    support:
        Support metadata describing the spatial (and/or sample) support.
    attrs:
        Optional free-form metadata.
    time:
        Optional time coordinate for arrays with a leading T axis.
    """

    array: np.ndarray
    band_schema: BandSchema
    support: SupportSpec
    attrs: Mapping[str, Any] = field(default_factory=dict)
    time: np.ndarray | None = None

    def __post_init__(self) -> None:
        arr = np.asarray(self.array)
        _validate_array_shape(arr)
        b = arr.shape[-1]
        if len(self.band_schema) != b:
            raise ValueError(
                "BandSchema length must match the array's band dimension. "
                f"len(band_schema)={len(self.band_schema)} but array.shape[-1]={b}."
            )
        if arr.ndim == 2 and self.support.kind != "samples":
            raise ValueError("Array with shape (N, B) requires support.kind='samples'.")
        if arr.ndim in (3, 4) and self.support.kind != "grid":
            raise ValueError("Array with shape (H, W, B) or (T, H, W, B) requires support.kind='grid'.")
        if arr.ndim == 3:
            h, w, _ = arr.shape
            if self.support.grid_shape != (h, w):
                raise ValueError(
                    "Grid support shape must match array spatial shape. "
                    f"support.grid_shape={self.support.grid_shape} but array has (H, W)=({h}, {w})."
                )
        if arr.ndim == 4:
            t, h, w, _ = arr.shape
            if self.time is None:
                object.__setattr__(self, "time", np.arange(t))
            else:
                time = np.asarray(self.time)
                if time.shape != (t,):
                    raise ValueError(f"time must have shape ({t},) for array with T={t}.")
                object.__setattr__(self, "time", time)
            if self.support.grid_shape != (h, w):
                raise ValueError(
                    "Grid support shape must match array spatial shape. "
                    f"support.grid_shape={self.support.grid_shape} but array has (H, W)=({h}, {w})."
                )
        object.__setattr__(self, "array", arr)

    @classmethod
    def from_array(
        cls,
        array: np.ndarray,
        *,
        band_schema: BandSchema | list[str] | tuple[str, ...],
        support: SupportSpec | Mapping[str, Any] | None = None,
        attrs: Mapping[str, Any] | None = None,
        time: np.ndarray | None = None,
    ) -> "EOData":
        arr = np.asarray(array)
        _validate_array_shape(arr)
        if isinstance(band_schema, BandSchema):
            schema = band_schema
        else:
            schema = BandSchema.from_names(band_schema)

        if support is None:
            if arr.ndim == 2:
                coords = np.zeros((arr.shape[0], 2), dtype=float)
                support_obj = SupportSpec.samples(coords)
            else:
                h, w = arr.shape[-3], arr.shape[-2]
                support_obj = SupportSpec.grid(shape=(h, w), resolution=(1.0, 1.0), origin=(0.0, 0.0))
        elif isinstance(support, SupportSpec):
            support_obj = support
        else:
            if support.get("kind") == "grid" and not (support.get("shape") or support.get("grid_shape")):
                h, w = arr.shape[-3], arr.shape[-2]
                support = dict(support)
                support["shape"] = (int(h), int(w))
            support_obj = SupportSpec.from_dict(support)
        return cls(array=arr, band_schema=schema, support=support_obj, attrs=attrs or {}, time=time)

    @classmethod
    def from_npy(
        cls,
        path: str | Path,
        *,
        band_schema: BandSchema | list[str] | tuple[str, ...],
        support: SupportSpec | Mapping[str, Any] | None = None,
        mmap_mode: str | None = None,
        allow_pickle: bool = False,
        attrs: Mapping[str, Any] | None = None,
    ) -> "EOData":
        p = _as_path(path)
        arr = np.load(p, mmap_mode=mmap_mode, allow_pickle=allow_pickle)
        return cls.from_array(arr, band_schema=band_schema, support=support, attrs=attrs)

    def select_bands(self, names: list[str] | tuple[str, ...]) -> "EOData":
        idx = self.band_schema.indices(names)
        schema = self.band_schema.select(names)
        new_arr = self.array[..., idx]
        return EOData(array=new_arr, band_schema=schema, support=self.support, attrs=self.attrs, time=self.time)

    def as_samples(self) -> np.ndarray:
        """Return a 2D (n_samples, n_bands) view suitable for scikit-learn."""

        if self.array.ndim == 2:
            return np.asarray(self.array)
        if self.array.ndim == 3:
            h, w, b = self.array.shape
            return self.array.reshape(h * w, b)
        t, h, w, b = self.array.shape
        return self.array.reshape(t * h * w, b)

    def __repr__(self) -> str:  # pragma: no cover (debug helper)
        return (
            f"EOData(shape={self.array.shape}, bands={len(self.band_schema)}, "
            f"support={self.support.kind})"
        )
