from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from asterra.data import EOData, SupportSpec
from asterra.io import sensors
from asterra.support import MixedResolutionTransformer


def main() -> None:
    rng = np.random.RandomState(3)

    # Two sources at different resolutions covering the same area
    s2 = EOData.from_array(
        rng.randn(32, 32, 4).astype("float32"),
        band_schema=sensors.sentinel2_rgbn(),
        support={"kind": "grid", "resolution": (10.0, 10.0), "origin": (0.0, 0.0)},
    )
    ps = EOData.from_array(
        rng.randn(64, 64, 4).astype("float32"),
        band_schema=sensors.planetscope_4band(),
        support={"kind": "grid", "resolution": (5.0, 5.0), "origin": (0.0, 0.0)},
    )

    target = SupportSpec.grid(shape=(32, 32), resolution=(10.0, 10.0), origin=(0.0, 0.0))
    mix = MixedResolutionTransformer(target_support=target)

    pipe = Pipeline([("mix", mix), ("model", Ridge(alpha=1.0))])

    X = [s2, ps]
    X_feat = mix.fit_transform(X)
    y = (X_feat[:, 0] * 0.5 + X_feat[:, 1] * -0.25 + rng.randn(X_feat.shape[0]) * 0.05).astype(
        "float64"
    )

    pipe.fit(X, y)
    pred = pipe.predict(X)
    print("Pipeline fit OK.")
    print("X features:", X_feat.shape, "y:", y.shape, "pred:", pred.shape)


if __name__ == "__main__":
    main()

