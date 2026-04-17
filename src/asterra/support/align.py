from __future__ import annotations

from asterra.data.support import SupportSpec


def assert_compatible_grids(source: SupportSpec, target: SupportSpec) -> None:
    """Best-effort compatibility checks for grid-to-grid overlap computations."""

    if source.kind != "grid" or target.kind != "grid":
        raise ValueError("Both supports must be grid supports.")
    if source.transform is not None or target.transform is not None:
        raise ValueError("Affine transforms are not supported for overlap in v0.1.0 (use origin/resolution).")
    if source.crs is not None and target.crs is not None and source.crs != target.crs:
        raise ValueError(f"CRS mismatch: source.crs={source.crs!r} target.crs={target.crs!r}.")

