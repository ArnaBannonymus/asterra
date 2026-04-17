# asterra

Support-aware machine learning for Earth observation.

## Motivation

Earth observation (EO) machine learning workflows routinely mix data sources and label types with **mismatched
spatial supports**:

- sensors with different pixel sizes (e.g., Sentinel-2 vs PlanetScope)
- labels defined on parcels/fields, tiles, scenes, or time windows (not per-pixel)
- coarse-to-fine (and fine-to-coarse) supervision
- patch overlap leakage and neighborhood dependence in evaluation

Ignoring these mismatches often leads to:
- biased features/labels due to incorrect aggregation
- silent leakage (overlapping patches, neighboring pixels, same-tile same-date)
- metrics that do not correspond to the true label support

## What Asterra does

Asterra is a **NumPy-first**, **scikit-learn-compatible** package for building *support-aware* pipelines.

The core abstraction is a sparse, overlap-based **`SupportMatrix`** that maps one support to another. It powers:

- mixed-resolution aggregation and projection (grid ↔ grid, samples → groups)
- support-aware feature and label projection
- leakage-safe splitting utilities (buffers, tile/time grouping)
- support-aware metrics

## Installation

```bash
python -m pip install asterra
```

Optional geospatial extras (not required for pixel-space workflows):

```bash
python -m pip install "asterra[geo]"
```

## Quickstart

```python
import numpy as np
from asterra.data import EOData
from asterra.io import sensors
from asterra.support import SupportMatrix

# Synthetic Sentinel-2-like grid (H, W, B)
arr_s2 = np.random.RandomState(0).randn(32, 32, 4).astype("float32")
e_s2 = EOData.from_array(
    arr_s2,
    band_schema=sensors.sentinel2_rgbn(),
    support={"kind": "grid", "resolution": (10.0, 10.0), "origin": (0.0, 0.0)},
)

# Synthetic PlanetScope-like grid on a different resolution
arr_ps = np.random.RandomState(1).randn(64, 64, 4).astype("float32")
e_ps = EOData.from_array(
    arr_ps,
    band_schema=sensors.planetscope_4band(),
    support={"kind": "grid", "resolution": (5.0, 5.0), "origin": (0.0, 0.0)},
)

# Map PlanetScope pixels (source) onto Sentinel-2 pixels (target)
M = SupportMatrix.from_grid_to_grid(source=e_ps.support, target=e_s2.support)
X_ps_on_s2 = M.project_features(e_ps.as_samples())
print(X_ps_on_s2.shape)  # (32*32, 4)
```

## Supported inputs

- `.npy` arrays with shapes `(H, W, B)`, `(T, H, W, B)`, `(N, B)`
- generic EO arrays with user-provided metadata:
  - `band_names`
  - georeferencing (`resolution`/`origin` or affine `transform`/`crs`) when available
  - pixel-space coordinates when georeferencing is not available
  - explicit group identifiers for parcel/tile/time supports

## Built-in sensor presets

Sensor helpers are convenience presets; the core library is sensor-agnostic.

- Sentinel-2 (common optical bands)
- Sentinel-1 (VV/VH-style SAR schema)
- PlanetScope (4-band and 8-band styles)
- NISAR-style configurable SAR schemas

## Architecture

The project is organized to keep EO-specific functionality separate from potentially generic sparse support logic:

- `asterra.data`: EO data model (array + band schema + support metadata)
- `asterra.support`: sparse support operators (SupportMatrix, projection)
- `asterra.preprocessing`: reshape/masking and band-aware transformers
- `asterra.model_selection`: leakage-aware splitters/utilities
- `asterra.metrics`: support-aware metrics
- `asterra.io`: `.npy` loader + sensor presets

See `DESIGN_BOUNDARIES.md` and `UPSTREAMING.md` for boundary notes and candidate generic components.

## Release status

`0.1.0` is the first public, focused release. The API is intentionally narrow and may evolve based on user
feedback and scientific validation.

## Roadmap (high level)

- richer support specifications (polygons/parcels via optional geo extras)
- additional support-aware scorers and splitters
- integration examples with real EO stacks (while keeping the core sensor-agnostic)

