# Quickstart

This quickstart uses synthetic arrays so it can be run anywhere (no external EO downloads).

## 1) Create `EOData`

```python
import numpy as np
from asterra.data import EOData
from asterra.io import sensors

arr = np.random.RandomState(0).randn(32, 32, 4).astype("float32")

eo = EOData.from_array(
    arr,
    band_schema=sensors.sentinel2_rgbn(),
    support={"kind": "grid", "resolution": (10.0, 10.0), "origin": (0.0, 0.0)},
)

print(eo.array.shape)
print(eo.band_schema.band_names)
print(eo.support.kind)
```

## 2) Build a `SupportMatrix` (grid → grid)

```python
import numpy as np
from asterra.data import EOData, SupportSpec
from asterra.io import sensors
from asterra.support import SupportMatrix

src = EOData.from_array(
    np.random.RandomState(1).randn(64, 64, 4).astype("float32"),
    band_schema=sensors.planetscope_4band(),
    support={"kind": "grid", "resolution": (5.0, 5.0), "origin": (0.0, 0.0)},
)
target = SupportSpec.grid(shape=(32, 32), resolution=(10.0, 10.0), origin=(0.0, 0.0))

M = SupportMatrix.from_grid_to_grid(source=src.support, target=target)
X_on_target = M.project_features(src.as_samples())
print(X_on_target.shape)  # (32*32, 4)
```

## 3) Use a scikit-learn pipeline (mixed-resolution)

```python
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from asterra.data import SupportSpec
from asterra.support import MixedResolutionTransformer

target = SupportSpec.grid(shape=(32, 32), resolution=(10.0, 10.0), origin=(0.0, 0.0))

pipe = Pipeline(
    [
        ("mix", MixedResolutionTransformer(target_support=target)),
        ("model", Ridge()),
    ]
)

X = [src]  # can be a list of EOData at different resolutions
y = np.random.RandomState(2).randn(target.n_items)
pipe.fit(X, y)
pred = pipe.predict(X)
print(pred.shape)
```

