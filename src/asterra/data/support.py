from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Hashable, Iterable, Literal, Mapping, Sequence

import numpy as np

SupportKind = Literal["grid", "samples", "groups"]


def _as_float_pair(x: Any, *, name: str) -> tuple[float, float]:
    if not isinstance(x, Sequence) or len(x) != 2:
        raise ValueError(f"{name} must be a pair of floats.")
    return (float(x[0]), float(x[1]))


@dataclass(frozen=True, slots=True)
class SupportSpec:
    """A description of the support on which values are defined.

    A support may represent:
    - a 2D grid ("grid"): pixels/cells in a raster-like array
    - independent samples ("samples"): a collection of points/samples
    - groups ("groups"): higher-level regions/units with identifiers

    The purpose of this class is to carry *just enough* metadata for constructing
    overlap/group mappings (via :class:`asterra.support.SupportMatrix`) without
    hardcoding EO sensors or file formats.
    """

    kind: SupportKind

    # Grid support
    grid_shape: tuple[int, int] | None = None
    resolution: tuple[float, float] | None = None
    origin: tuple[float, float] | None = None
    transform: tuple[float, float, float, float, float, float] | None = None

    # Sample support
    coords: np.ndarray | None = None  # (n, d) float
    n_samples: int | None = None

    # Group support
    labels: tuple[Hashable, ...] | None = None  # unique labels for group supports

    # Common
    crs: str | None = None

    def __post_init__(self) -> None:
        if self.kind == "grid":
            if self.grid_shape is None:
                raise ValueError("grid support requires grid_shape=(H, W).")
            if len(self.grid_shape) != 2:
                raise ValueError("grid_shape must be a (H, W) pair.")
            h, w = self.grid_shape
            if int(h) <= 0 or int(w) <= 0:
                raise ValueError("grid_shape must be positive.")
            if self.transform is not None and len(self.transform) != 6:
                raise ValueError("transform must be a 6-tuple (affine).")
            if self.transform is None:
                if self.resolution is None:
                    raise ValueError("grid support requires resolution when transform is not provided.")
                if self.origin is None:
                    raise ValueError("grid support requires origin when transform is not provided.")
                dx, dy = _as_float_pair(self.resolution, name="resolution")
                if dx <= 0 or dy <= 0:
                    raise ValueError("resolution values must be positive.")
                _as_float_pair(self.origin, name="origin")
        elif self.kind == "samples":
            if self.coords is None:
                if self.n_samples is None:
                    raise ValueError("samples support requires coords or n_samples.")
                if int(self.n_samples) <= 0:
                    raise ValueError("n_samples must be a positive integer.")
            else:
                coords = np.asarray(self.coords)
                if coords.ndim != 2 or coords.shape[0] <= 0:
                    raise ValueError("coords must have shape (n_samples, d) with n_samples > 0.")
                if not np.issubdtype(coords.dtype, np.number):
                    raise ValueError("coords must be numeric.")
                object.__setattr__(self, "coords", coords.astype(float, copy=False))
                if self.n_samples is None:
                    object.__setattr__(self, "n_samples", int(coords.shape[0]))
                elif int(self.n_samples) != int(coords.shape[0]):
                    raise ValueError(
                        "n_samples must match coords.shape[0] when coords are provided. "
                        f"n_samples={self.n_samples} coords.shape[0]={coords.shape[0]}."
                    )
        elif self.kind == "groups":
            if self.labels is None:
                raise ValueError("groups support requires labels (unique group identifiers).")
            if len(self.labels) == 0:
                raise ValueError("groups labels must be non-empty.")
        else:  # pragma: no cover (SupportKind typing)
            raise ValueError(f"Unknown support kind: {self.kind!r}.")

    @classmethod
    def grid(
        cls,
        *,
        shape: tuple[int, int],
        resolution: tuple[float, float] = (1.0, 1.0),
        origin: tuple[float, float] = (0.0, 0.0),
        crs: str | None = None,
        transform: tuple[float, float, float, float, float, float] | None = None,
    ) -> "SupportSpec":
        return cls(
            kind="grid",
            grid_shape=shape,
            resolution=resolution if transform is None else None,
            origin=origin if transform is None else None,
            transform=transform,
            crs=crs,
        )

    @classmethod
    def samples(cls, coords: np.ndarray, *, crs: str | None = None) -> "SupportSpec":
        coords_arr = np.asarray(coords)
        return cls(kind="samples", coords=coords_arr, n_samples=int(coords_arr.shape[0]), crs=crs)

    @classmethod
    def samples_n(cls, n_samples: int, *, crs: str | None = None) -> "SupportSpec":
        return cls(kind="samples", coords=None, n_samples=int(n_samples), crs=crs)

    @classmethod
    def groups(cls, labels: Iterable[Hashable], *, crs: str | None = None) -> "SupportSpec":
        return cls(kind="groups", labels=tuple(labels), crs=crs)

    @classmethod
    def from_dict(cls, spec: Mapping[str, Any]) -> "SupportSpec":
        kind = spec.get("kind")
        if kind == "grid":
            shape = spec.get("shape") or spec.get("grid_shape")
            if shape is None:
                raise ValueError("grid support dict requires 'shape' or 'grid_shape'.")
            transform = spec.get("transform")
            return cls.grid(
                shape=tuple(shape),
                resolution=_as_float_pair(spec.get("resolution", (1.0, 1.0)), name="resolution"),
                origin=_as_float_pair(spec.get("origin", (0.0, 0.0)), name="origin"),
                crs=spec.get("crs"),
                transform=tuple(transform) if transform is not None else None,
            )
        if kind == "samples":
            coords = spec.get("coords")
            if coords is not None:
                return cls.samples(np.asarray(coords), crs=spec.get("crs"))
            n_samples = spec.get("n_samples")
            if n_samples is None:
                raise ValueError("samples support dict requires 'coords' or 'n_samples'.")
            return cls.samples_n(int(n_samples), crs=spec.get("crs"))
        if kind == "groups":
            labels = spec.get("labels")
            if labels is None:
                raise ValueError("groups support dict requires 'labels'.")
            return cls.groups(labels, crs=spec.get("crs"))
        raise ValueError(f"Unknown or missing support kind in dict: {kind!r}.")

    @property
    def n_items(self) -> int:
        """Number of units in this support (pixels, samples, or groups)."""

        if self.kind == "grid":
            assert self.grid_shape is not None
            h, w = self.grid_shape
            return int(h) * int(w)
        if self.kind == "samples":
            if self.coords is not None:
                return int(self.coords.shape[0])
            assert self.n_samples is not None
            return int(self.n_samples)
        if self.kind == "groups":
            assert self.labels is not None
            return int(len(self.labels))
        raise RuntimeError("Unexpected SupportSpec.kind")  # pragma: no cover

    def grid_params(self) -> tuple[tuple[int, int], tuple[float, float], tuple[float, float]]:
        """Return (shape, resolution, origin) for simple north-up grids.

        Raises
        ------
        ValueError
            If this support is not a grid or uses an affine transform.
        """

        if self.kind != "grid":
            raise ValueError("grid_params is only available for grid supports.")
        if self.transform is not None:
            raise ValueError("grid_params is not available when transform is provided.")
        assert self.grid_shape is not None and self.resolution is not None and self.origin is not None
        return self.grid_shape, self.resolution, self.origin
