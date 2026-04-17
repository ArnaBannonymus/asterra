from __future__ import annotations

import numpy as np

from asterra.data import EOData
from asterra.io import sensors
from asterra.support import SupportMatrix


def main() -> None:
    rng = np.random.RandomState(0)

    # Sentinel-2-like grid (10m)
    arr_s2 = rng.randn(32, 32, 4).astype("float32")
    s2 = EOData.from_array(
        arr_s2,
        band_schema=sensors.sentinel2_rgbn(),
        support={"kind": "grid", "resolution": (10.0, 10.0), "origin": (0.0, 0.0)},
    )

    # PlanetScope-like grid (5m) covering the same area
    arr_ps = rng.randn(64, 64, 4).astype("float32")
    ps = EOData.from_array(
        arr_ps,
        band_schema=sensors.planetscope_4band(),
        support={"kind": "grid", "resolution": (5.0, 5.0), "origin": (0.0, 0.0)},
    )

    # Project PlanetScope pixels to Sentinel-2 pixels
    M = SupportMatrix.from_grid_to_grid(source=ps.support, target=s2.support)
    X_ps_on_s2 = M.project_features(ps.as_samples())

    print("SupportMatrix:", M.matrix.shape, "normalized=", M.normalized)
    print("Projected PlanetScope -> Sentinel-2:", X_ps_on_s2.shape)
    print("Example mean (first band):", float(X_ps_on_s2[:, 0].mean()))


if __name__ == "__main__":
    main()

