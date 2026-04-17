from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import train_test_split

from asterra.data import EOData
from asterra.io import sensors


def _extract_patch_features(hh: np.ndarray, hv: np.ndarray, *, batch_size: int = 64) -> np.ndarray:
    """Extract simple per-patch summary features from complex SAR patches.

    Features (per patch):
    - mean(log1p(|HH|)), std(log1p(|HH|))
    - mean(log1p(|HV|)), std(log1p(|HV|))
    - mean(HH) - mean(HV), std(HH) - std(HV)
    """

    if hh.shape != hv.shape:
        raise ValueError(f"HH and HV arrays must have the same shape. Got {hh.shape} vs {hv.shape}.")
    if hh.ndim != 3:
        raise ValueError(f"Expected patch stacks with shape (N, H, W). Got {hh.shape}.")

    n = int(hh.shape[0])
    feats = np.empty((n, 6), dtype="float32")

    for start in range(0, n, int(batch_size)):
        stop = min(n, start + int(batch_size))
        hh_b = np.asarray(hh[start:stop])
        hv_b = np.asarray(hv[start:stop])

        mag_hh = np.log1p(np.abs(hh_b)).astype("float32", copy=False)
        mag_hv = np.log1p(np.abs(hv_b)).astype("float32", copy=False)

        m_hh = mag_hh.mean(axis=(1, 2))
        s_hh = mag_hh.std(axis=(1, 2))
        m_hv = mag_hv.mean(axis=(1, 2))
        s_hv = mag_hv.std(axis=(1, 2))

        feats[start:stop, 0] = m_hh
        feats[start:stop, 1] = s_hh
        feats[start:stop, 2] = m_hv
        feats[start:stop, 3] = s_hv
        feats[start:stop, 4] = m_hh - m_hv
        feats[start:stop, 5] = s_hh - s_hv

    return feats


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Example using an on-disk S1SLC_CVDL complex patch dataset:\n"
            "- loads HH/HV complex patch stacks (memmap)\n"
            "- extracts simple per-patch features\n"
            "- trains a lightweight classifier\n\n"
            "This demonstrates Asterra's 'array + band schema + support metadata' style for non-image EO tensors."
        )
    )
    parser.add_argument(
        "--city-dir",
        type=Path,
        required=True,
        help="Path to a CVDL city folder containing HH_Complex_Patches.npy, HV_Complex_Patches.npy, Labels.npy.",
    )
    parser.add_argument("--n-samples", type=int, default=1024, help="Number of patches to sample for the demo.")
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    hh_path = args.city_dir / "HH_Complex_Patches.npy"
    hv_path = args.city_dir / "HV_Complex_Patches.npy"
    y_path = args.city_dir / "Labels.npy"

    hh = np.load(hh_path, mmap_mode="r")
    hv = np.load(hv_path, mmap_mode="r")
    y = np.load(y_path, mmap_mode="r").astype("int64", copy=False).ravel()

    n_total = int(hh.shape[0])
    n = int(min(max(1, args.n_samples), n_total))

    rng = np.random.RandomState(0)
    idx = rng.choice(n_total, size=n, replace=False)

    X = _extract_patch_features(hh[idx], hv[idx])

    # Asterra wrapper: samples + schema
    eo = EOData.from_array(
        X,
        band_schema=["mean_hh", "std_hh", "mean_hv", "std_hv", "mean_diff", "std_diff"],
        support={"kind": "samples", "n_samples": X.shape[0]},
        attrs={"source": "S1SLC_CVDL", "city": args.city_dir.name, "pol_schema": sensors.nisar_sar_schema(("HH", "HV")).band_names},
    )

    y0 = y[idx] - 1  # make 0-based classes (CVDL labels are 1..K)

    X_train, X_test, y_train, y_test = train_test_split(
        eo.as_samples(), y0, test_size=float(args.test_size), random_state=0, stratify=y0
    )

    clf = RidgeClassifier()
    clf.fit(X_train, y_train)
    acc = float(clf.score(X_test, y_test))

    print("CVDL city:", args.city_dir)
    print("Samples used:", n, "/", n_total)
    print("Features:", eo.band_schema.band_names)
    print("Accuracy (demo, RidgeClassifier):", acc)


if __name__ == "__main__":
    main()
