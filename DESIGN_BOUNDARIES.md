# Design boundaries

## What belongs in Asterra

Asterra is a domain package for **support-aware learning** in Earth observation workflows. It includes:

- EO data model helpers (band schemas, sensor presets, lightweight metadata handling)
- sparse support mappings that make mixed-resolution aggregation correct and reusable
- leakage-aware evaluation patterns common in EO (tile/time grouping, spatial buffers, patch overlap concerns)
- support-aware metrics that can reconcile predictions and labels defined on different supports

## What does not belong in scikit-learn core

scikit-learn is a general-purpose ML library. The following are intentionally external:

- sensor-specific presets (Sentinel-1/2, PlanetScope, NISAR-style band conventions)
- geospatial file I/O (GeoTIFF, SAFE products, STAC, etc.)
- domain-specific coordinate systems and EO metadata conventions
- EO-specific leakage rules (same-tile same-date, neighboring-patch dependence) beyond generic grouping/buffering

## Boundary between domain package and generic ML utility

The boundary Asterra enforces is:

**Core generic concept (potentially upstreamable):** sparse operators that map samples between supports using
overlap/group structure, plus generic splitters/scorers built on public scikit-learn APIs.

**EO-specific surface area (stays external):** anything that assumes EO sensors, georeferencing conventions, or
EO workflow semantics.

Candidate generic components and an upstream discussion plan are documented in `UPSTREAMING.md` and
`upstream_candidate/`.

