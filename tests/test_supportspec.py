from __future__ import annotations

import numpy as np
import pytest

from asterra.data import SupportSpec
from asterra.support import SupportMatrix


def test_supportspec_grid_validation() -> None:
    s = SupportSpec.grid(shape=(2, 3), resolution=(1.0, 1.0), origin=(0.0, 0.0))
    assert s.n_items == 6


def test_supportspec_samples_coords() -> None:
    coords = np.array([[0.0, 0.0], [1.0, 2.0]])
    s = SupportSpec.samples(coords)
    assert s.n_items == 2


def test_supportspec_samples_n() -> None:
    s = SupportSpec.samples_n(5)
    assert s.n_items == 5


def test_supportspec_groups() -> None:
    s = SupportSpec.groups(["a", "b"])
    assert s.n_items == 2


def test_grid_crs_mismatch_raises() -> None:
    src = SupportSpec.grid(shape=(2, 2), resolution=(1.0, 1.0), origin=(0.0, 0.0), crs="EPSG:4326")
    tgt = SupportSpec.grid(shape=(2, 2), resolution=(1.0, 1.0), origin=(0.0, 0.0), crs="EPSG:3857")
    with pytest.raises(ValueError, match="CRS mismatch"):
        SupportMatrix.from_grid_to_grid(source=src, target=tgt)

