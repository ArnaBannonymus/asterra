from __future__ import annotations

import pytest

from asterra.data import BandSchema


def test_band_schema_unique_names() -> None:
    with pytest.raises(ValueError, match="unique"):
        BandSchema.from_names(["B1", "B1"])


def test_band_schema_alias_resolution() -> None:
    schema = BandSchema.from_names(["B02", "B03"], aliases={"blue": "B02"})
    assert schema.resolve("blue") == "B02"
    assert schema.index("blue") == 0


def test_band_schema_select() -> None:
    schema = BandSchema.from_names(["B02", "B03", "B04"])
    sub = schema.select(["B04", "B02"])
    assert sub.band_names == ("B04", "B02")

