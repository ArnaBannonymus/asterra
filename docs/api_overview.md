# API overview

## Data model

- `asterra.data.BandSchema`
  - band names, aliases, and per-band metadata
- `asterra.data.SupportSpec`
  - grid / samples / groups support descriptions
- `asterra.data.EOData`
  - array + `BandSchema` + `SupportSpec`

## Sparse support operators

- `asterra.support.SupportMatrix`
  - sparse overlap/group mappings (grid↔grid, samples→groups)
- `asterra.support.SupportProjector`
  - scikit-learn-style transformer for sparse projections
- `asterra.support.MixedResolutionTransformer`
  - project multiple EO grids to a common target and concatenate bands

## Model selection

- `asterra.model_selection.BufferedSpatialKFold`
  - spatial blocking with optional buffer exclusion
- `asterra.model_selection.TileTimeGroupSplit`
  - group-aware splitting for (tile, time) identifiers

## Metrics

- `asterra.metrics.SupportAwareRMSE`
- `asterra.metrics.SupportAwareMAE`
- `asterra.metrics.SupportAwareF1`

## I/O helpers

- `asterra.io.load_npy`
- `asterra.io.sensors.*` (Sentinel-1/2, PlanetScope, NISAR-style SAR schemas)

