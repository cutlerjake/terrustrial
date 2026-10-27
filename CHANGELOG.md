# Changelog

## 0.2.0 (unreleased)

Breaking release. The repository becomes a workspace with two crates:

- `terrustrial-core` 0.2.0 is a new joint geostatistical engine written from
  scratch. Nothing in it depends on the 0.1 code, and its API is new: validated
  `data` types, a linear model of coregionalization in `model`, joint Gaussian
  prediction in `gaussian`, ordered parallel estimation and sequential Gaussian
  simulation drivers, stable neighborhood selection, persistent observation
  adapters (binary point files, SQLite stores, packed spatial pages, streaming
  grid files), variography and LMC fitting, normal-score transforms,
  nonstationary covariance, implicit-field domain classification with
  drillhole evidence scoring and validation, high-order kernel embeddings and
  multiple-point pattern sampling. Every driver takes an explicit memory
  budget and cancellation token, random streams are keyed by seed,
  realization, item and purpose, and numerical failures are returned as
  errors. Optional features: `sqlite`, `fft`, `gpu`, `mi-allocator`,
  `profiling` and `experimental-two-phase`. The crate forbids `unsafe` code.
- `terrustrial` 0.2.0, the root crate, keeps the 0.1 modules (`geometry`,
  `group_operators`, `spatial_database`, `systems`, `variography`) with small
  internal changes, depends on `terrustrial-core`, and adds the `qs` module:
  Quick Sampling multiple-point simulation (Gravey and Mariethoz 2020) for
  continuous, categorical and multivariate attributes on regular 3D grids.
  The `prelude::re_exports` module now re-exports
  `terrustrial_core::spatial::hierarchy` instead of `rstar`, and the spatial
  database no longer depends on `rstar`. Dependencies were updated (`faer`
  0.24, `rand` 0.10, `rand_distr` 0.6, `itertools` 0.15, `ordered-float` 5,
  `indicatif` 0.18) and `rustfft`, `realfft` and `num-complex` were added for
  Quick Sampling.
- Comparison examples and tests for the legacy and core drivers
  (`examples/parallel_estimation_comparison.rs`,
  `examples/parallel_simulation_comparison.rs`, `examples/rewrite_baseline.rs`,
  `tests/kriging_reference.rs`, `tests/spatial_hierarchy_migration.rs`) and a
  Windows scaling benchmark runner (`scripts/benchmark_windows.ps1`).
- License files renamed to `LICENSE-MIT` and `LICENSE-APACHE`; the license
  itself is unchanged (MIT OR Apache-2.0).

## 0.1.1 (2025-12-08)

Last release of the original design: generalized sequential kriging and
simulation on an R-tree spatial database with composite variograms and LU
system solvers. Tagged `v0.1.1`.

## 0.1.0 (2025-12-08)

First crates.io release.
