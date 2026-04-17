from __future__ import annotations

import numpy as np
import pytest

from asterra.data import BandSchema, EOData, SupportSpec


def test_eodata_grid_shape_handling() -> None:
    schema = BandSchema.from_names(["b0", "b1"])
    support = SupportSpec.grid(shape=(3, 4), resolution=(1.0, 1.0), origin=(0.0, 0.0))
    arr = np.zeros((3, 4, 2), dtype="float32")
    eo = EOData(array=arr, band_schema=schema, support=support)
    assert eo.as_samples().shape == (12, 2)


def test_eodata_time_series_handling() -> None:
    schema = BandSchema.from_names(["b0"])
    support = SupportSpec.grid(shape=(2, 2), resolution=(1.0, 1.0), origin=(0.0, 0.0))
    arr = np.zeros((3, 2, 2, 1), dtype="float32")
    eo = EOData(array=arr, band_schema=schema, support=support)
    assert eo.time is not None and eo.time.shape == (3,)
    assert eo.as_samples().shape == (12, 1)


def test_eodata_select_bands() -> None:
    schema = BandSchema.from_names(["a", "b", "c"])
    support = SupportSpec.grid(shape=(1, 1), resolution=(1.0, 1.0), origin=(0.0, 0.0))
    arr = np.array([[[1.0, 2.0, 3.0]]], dtype="float32")
    eo = EOData(array=arr, band_schema=schema, support=support)
    sub = eo.select_bands(["c", "a"])
    assert sub.band_schema.band_names == ("c", "a")
    assert sub.array.shape == (1, 1, 2)
    assert float(sub.array[0, 0, 0]) == 3.0


def test_eodata_from_npy(tmp_path) -> None:
    arr = np.arange(12, dtype="float32").reshape(3, 4)
    path = tmp_path / "x.npy"
    np.save(path, arr)
    eo = EOData.from_npy(path, band_schema=["b0", "b1", "b2", "b3"], support={"kind": "samples", "n_samples": 3})
    assert eo.array.shape == (3, 4)
    assert eo.support.kind == "samples"


def test_eodata_support_shape_inferred_from_dict() -> None:
    arr = np.zeros((2, 3, 1), dtype="float32")
    eo = EOData.from_array(
        arr,
        band_schema=["b0"],
        support={"kind": "grid", "resolution": (2.0, 2.0), "origin": (0.0, 0.0)},
    )
    assert eo.support.grid_shape == (2, 3)


def test_eodata_band_count_mismatch_raises() -> None:
    support = SupportSpec.grid(shape=(1, 1), resolution=(1.0, 1.0), origin=(0.0, 0.0))
    arr = np.zeros((1, 1, 2), dtype="float32")
    with pytest.raises(ValueError, match="BandSchema length"):
        EOData(array=arr, band_schema=BandSchema.from_names(["b0"]), support=support)

