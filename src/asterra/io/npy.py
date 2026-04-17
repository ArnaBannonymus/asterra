from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from asterra.data.bands import BandSchema
from asterra.data.eodata import EOData
from asterra.data.support import SupportSpec


def load_npy(
    path: str | Path,
    *,
    band_schema: BandSchema | list[str] | tuple[str, ...],
    support: SupportSpec | Mapping[str, Any] | None = None,
    mmap_mode: str | None = None,
    allow_pickle: bool = False,
    attrs: Mapping[str, Any] | None = None,
) -> EOData:
    """Load a `.npy` array into :class:`asterra.data.EOData`."""

    return EOData.from_npy(
        path,
        band_schema=band_schema,
        support=support,
        mmap_mode=mmap_mode,
        allow_pickle=allow_pickle,
        attrs=attrs,
    )

