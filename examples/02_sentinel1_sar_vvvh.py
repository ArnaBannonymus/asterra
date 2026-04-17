from __future__ import annotations

import numpy as np

from asterra.data import EOData
from asterra.io import sensors


def main() -> None:
    rng = np.random.RandomState(1)

    # Synthetic SAR backscatter with VV/VH polarizations
    arr = rng.randn(16, 16, 2).astype("float32")
    eo = EOData.from_array(
        arr,
        band_schema=sensors.sentinel1_vvvh(),
        support={"kind": "grid", "resolution": (10.0, 10.0), "origin": (0.0, 0.0)},
    )

    vv = eo.array[..., eo.band_schema.index("VV")]
    vh = eo.array[..., eo.band_schema.index("VH")]

    # Simple derived feature: VV - VH (often used as a crude polarization contrast proxy)
    contrast = vv - vh
    print("EOData:", eo)
    print("Contrast feature stats:", float(contrast.mean()), float(contrast.std()))


if __name__ == "__main__":
    main()

