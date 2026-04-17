from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from asterra.data import EOData, SupportSpec
from asterra.io import sensors
from asterra.preprocessing.masking import MaskInvalid
from asterra.preprocessing.reshape import FlattenGrid
from asterra.support import MixedResolutionTransformer


def test_sklearn_clone_smoke_transformers() -> None:
    # `check_estimator` is intentionally not used here: some environments abort (SIGABRT)
    # due to OpenMP runtime conflicts when running the full estimator check suite.
    clone(MaskInvalid())
    clone(FlattenGrid())


def test_pipeline_with_mixed_resolution_transformer() -> None:
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
    pipe = Pipeline([("mix", MixedResolutionTransformer(target_support=target)), ("model", Ridge(alpha=1.0))])

    X = [s2, ps]
    X_feat = pipe.named_steps["mix"].fit_transform(X)
    y = (X_feat[:, 0] + rng.randn(X_feat.shape[0]) * 0.01).astype("float64")
    pipe.fit(X, y)
    pred = pipe.predict(X)
    assert pred.shape == y.shape
