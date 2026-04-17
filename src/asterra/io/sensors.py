from __future__ import annotations

from collections.abc import Iterable
from typing import Mapping

from asterra.data.bands import BandSchema


def sentinel2_rgbn() -> BandSchema:
    """Sentinel-2 RGB + NIR convenience schema (B02, B03, B04, B08)."""

    return BandSchema.from_names(
        ["B02", "B03", "B04", "B08"],
        aliases={"blue": "B02", "green": "B03", "red": "B04", "nir": "B08"},
        metadata={
            "B02": {"center_nm": 492},
            "B03": {"center_nm": 560},
            "B04": {"center_nm": 665},
            "B08": {"center_nm": 842},
        },
    )


def sentinel1_vvvh() -> BandSchema:
    """Sentinel-1 VV/VH-style SAR schema."""

    return BandSchema.from_names(["VV", "VH"], aliases={"vv": "VV", "vh": "VH"})


def planetscope_4band() -> BandSchema:
    """PlanetScope 4-band (Blue, Green, Red, NIR) convenience schema."""

    return BandSchema.from_names(
        ["B", "G", "R", "NIR"],
        aliases={"blue": "B", "green": "G", "red": "R", "nir": "NIR"},
    )


def nisar_sar_schema(
    pols: Iterable[str] = ("HH", "HV", "VH", "VV"),
    *,
    extras: Mapping[str, Mapping[str, object]] | None = None,
) -> BandSchema:
    """Configurable SAR schema in a NISAR-style polarization vocabulary."""

    pols_t = tuple(pols)
    aliases = {p.lower(): p for p in pols_t}
    return BandSchema.from_names(pols_t, aliases=aliases, metadata=extras or {})

