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
- `terrustrial` 0.2.0, the root crate, is a facade: `pub use
  terrustrial_core::*`, features `sqlite`, `fft`, `gpu`, `mi-allocator`,
  `profiling` and `experimental-two-phase` forwarded to the core, plus the new
  `qs` module: Quick Sampling multiple-point simulation (Gravey and Mariethoz
  2020) for continuous, categorical and multivariate attributes on regular 3D
  grids, with the `qs_demo` and `strebelle_bench` examples.
- Removed: the 0.1 modules `geometry`, `group_operators`,
  `spatial_database`, `systems` and `variography`, the `prelude`, the
  `FORWARD`, `RIGHT` and `UP` constants, and the dependencies they carried
  (`faer`, `dyn-stack`, `rstar`, `rand_distr`, `num-traits`, `ultraviolet`,
  `serde`, `ordered-float`, `itertools`, `permutation`, `indicatif`, `csv`,
  `wide`). The root crate now depends on `terrustrial-core`, `rand`, `rayon`,
  `rustfft`, `realfft` and `num-complex`. The README maps the 0.1 types to
  their 0.2 replacements; indicator kriging and inverse distance have none.
- `scripts/benchmark_windows.ps1`, the Windows scaling benchmark runner for
  the core's `storage_comparison` example.
- License files renamed to `LICENSE-MIT` and `LICENSE-APACHE`; the license
  itself is unchanged (MIT OR Apache-2.0).

## 0.1.1 (2025-12-08)

Last release of the original design: generalized sequential kriging and
simulation on an R-tree spatial database with composite variograms and LU
system solvers. Tagged `v0.1.1`.

## 0.1.0 (2025-12-08)

First crates.io release.
