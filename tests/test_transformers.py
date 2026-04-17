from __future__ import annotations

import numpy as np

from asterra.data import EOData, SupportSpec
from asterra.io import sensors
from asterra.preprocessing import FlattenGrid
from asterra.support import MixedResolutionTransformer


def test_flatten_grid_transformer() -> None:
    t = FlattenGrid()
    X = np.zeros((2, 3, 4), dtype=float)
    out = t.fit_transform(X)
    assert out.shape == (6, 4)


def test_mixed_resolution_transformer_shapes() -> None:
    rng = np.random.RandomState(0)
    s2 = EOData.from_array(
        rng.randn(4, 4, 4).astype("float32"),
        band_schema=sensors.sentinel2_rgbn(),
        support={"kind": "grid", "resolution": (10.0, 10.0), "origin": (0.0, 0.0)},
    )
    ps = EOData.from_array(
        rng.randn(8, 8, 4).astype("float32"),
        band_schema=sensors.planetscope_4band(),
        support={"kind": "grid", "resolution": (5.0, 5.0), "origin": (0.0, 0.0)},
    )
    target = SupportSpec.grid(shape=(4, 4), resolution=(10.0, 10.0), origin=(0.0, 0.0))
    mix = MixedResolutionTransformer(target_support=target)
    out = mix.fit_transform([s2, ps])
    assert out.shape == (16, 8)
    assert len(mix.output_band_names_) == 8

