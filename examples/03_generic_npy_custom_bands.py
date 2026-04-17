from __future__ import annotations

from pathlib import Path

import numpy as np

from asterra.io import load_npy


def main() -> None:
    rng = np.random.RandomState(2)
    arr = rng.randn(10, 3).astype("float32")  # (N, B)

    out_dir = Path("build_artifacts") / "example_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    npy_path = out_dir / "custom_samples.npy"
    np.save(npy_path, arr)

    eo = load_npy(
        npy_path,
        band_schema=["custom_0", "custom_1", "custom_2"],
        support={"kind": "samples", "n_samples": 10},
    )
    print("Loaded:", eo)
    print("Bands:", eo.band_schema.band_names)
    print("As samples:", eo.as_samples().shape)


if __name__ == "__main__":
    main()

