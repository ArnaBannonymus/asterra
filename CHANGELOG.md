# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] - 2026-04-17

### Added
- NumPy-first EO data model: `BandSchema`, `SupportSpec`, `EOData`.
- Sparse overlap-based mapping via `SupportMatrix` (grid↔grid, samples→groups).
- Support-aware projection utilities and a mixed-resolution scikit-learn transformer.
- Leakage-aware splitters (`BufferedSpatialKFold`, `TileTimeGroupSplit`).
- Support-aware metrics (RMSE/MAE/F1) with optional projection.
- Examples, documentation, and a pytest suite.

## [0.1.1] - 2026-04-17

### Changed
- README and examples now refer to **local/on-disk datasets** (instead of “real data” phrasing).
- Sentinel-1 README visual is rendered in **dB** and uses an auto-selected textured window for a more typical SAR look.
