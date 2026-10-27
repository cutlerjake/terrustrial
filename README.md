![terrustrial logo](https://raw.githubusercontent.com/cutlerjake/terrustrial/main/assets/logo.svg)

Geostatistics in Rust: joint kriging and sequential Gaussian simulation with
explicit resource and numerical contracts.

Version 0.2 is a rewrite. The engine lives in `terrustrial-core`, whose API
has nothing in common with 0.1. The `terrustrial` crate is a facade over it:
it re-exports every module of the core, forwards its feature flags, and adds
the `qs` Quick Sampling module. The 0.1 API is gone; the migration note at
the end maps its types to their replacements.

## Workspace layout

| Path | Crate | Contents |
| --- | --- | --- |
| `crates/terrustrial-core/` | `terrustrial-core` 0.2.0 | The joint estimation and simulation engine: validated data types, linear model of coregionalization, joint Gaussian prediction, ordered parallel estimation and pipelined simulation drivers, neighborhood selection, persistent stores, variography and fitting, transforms, nonstationary covariance, domain evidence, high-order statistics and multiple-point pattern sampling. Forbids `unsafe` code. |
| `src/` | `terrustrial` 0.2.0 | `pub use terrustrial_core::*` plus `qs`: Quick Sampling multiple-point simulation on regular 3D grids (Gravey and Mariethoz 2020) for continuous, categorical and multivariate attributes. |
| `examples/` | `terrustrial` | `qs_demo` and `strebelle_bench`, the Quick Sampling demos. |
| `crates/terrustrial-core/examples/`, `crates/terrustrial-core/tests/` | `terrustrial-core` | Benchmarks and diagnostics for every subsystem, and the integration tests that check each driver against an independent oracle. |
| `scripts/benchmark_windows.ps1` | | Windows scaling runner: focused correctness tests, then estimation and simulation sweeps over worker counts for resident and packed storage, written to `benchmark-results/<timestamp>/`. |

## Installing

```toml
[dependencies]
terrustrial = { version = "0.2", features = ["sqlite"] }
```

Depending on `terrustrial-core` directly gives the same modules under the
`terrustrial_core::` path without Quick Sampling. The default feature set is
empty; see the feature table below.

## Example

Ordinary kriging of one variable at two targets from four point observations.
The prepared `GaussianPlan` can be reused for repeated prediction, and the
joint prediction-error covariance is retained because it was requested. This
program is the root crate's doc test; the same program under
`terrustrial_core::` paths is the doc test of `terrustrial-core`.

```rust
use terrustrial::{
    data::{Observation, Point3, Site},
    execution::{Cancellation, MemoryBudget},
    gaussian::{GaussianPlan, NumericalTolerance, Ordinary, PrepareOptions, Uncertainty},
    model::{Anisotropy, Coregionalization, Correlation, Lmc, Structure},
};

fn site(x: f64, y: f64) -> terrustrial::Result<Site> {
    Ok(Site::point(Point3::new([x, y, 0.0])?, 0))
}

fn main() -> terrustrial::Result<()> {
    // One exponential structure with a 25 m scale and unit sill for one variable.
    let model = Lmc::new(vec![Structure {
        correlation: Correlation::Exponential,
        anisotropy: Anisotropy::isotropic(25.0)?,
        coregionalization: Coregionalization::scalar(1.0)?,
    }])?;

    // Observations carry a stable id, a site (support and variable index),
    // a value and a measurement noise variance.
    let observations = vec![
        Observation::new(1, site(0.0, 0.0)?, 1.2, 0.0)?,
        Observation::new(2, site(10.0, 0.0)?, 0.8, 0.0)?,
        Observation::new(3, site(0.0, 10.0)?, 1.9, 0.0)?,
        Observation::new(4, site(10.0, 10.0)?, 1.4, 0.0)?,
    ];
    let targets = [site(5.0, 5.0)?, site(2.0, 8.0)?];

    // Workspace memory is charged against an explicit budget, and any
    // operation can be cancelled from another thread.
    let budget = MemoryBudget::new(64 << 20);
    let cancellation = Cancellation::default();

    let plan = GaussianPlan::prepare(
        &model,
        &observations,
        &targets,
        &Ordinary { variables: 1 },
        PrepareOptions {
            uncertainty: Uncertainty::Joint,
            tolerance: NumericalTolerance::default(),
            budget: &budget,
            cancellation: &cancellation,
        },
    )?;

    let values: Vec<f64> = observations.iter().map(Observation::value).collect();
    let estimates = plan.estimate(&values)?;
    let covariance = plan.covariance().expect("joint uncertainty was requested");
    assert_eq!(estimates.len(), targets.len());
    println!("estimates: {estimates:?}");
    println!("error variance at the first target: {}", covariance[(0, 0)]);
    Ok(())
}
```

For many targets, `estimation::estimate_groups` streams groups of targets
through scoped workers with bounded batch admission, and
`simulation::simulate_pipelined` runs sequential Gaussian simulation through a
rolling lookahead window. The integration tests under
`crates/terrustrial-core/tests/` are the most complete usage reference. The
module table in [the crate README](crates/terrustrial-core/README.md) lists
what each module contains.

## Contracts

Memory admission. `execution::MemoryBudget` is shared logical accounting for
explicitly reserved payload and workspace bytes. Drivers reserve their batch,
worker and cache allowances before dispatching work, and a request that does
not fit fails with `Error::MemoryBudget { requested, available }` rather than
allocating. It is not an operating-system RSS limit and does not replace
fallible allocation; allocator overhead, the SQLite page cache and the
operating-system file cache are outside it.

Cancellation. `execution::Cancellation` is a shared flag. Drivers check it
between units of work and return `Error::Cancelled`; a packed builder that is
cancelled before publication removes its temporary database.

Deterministic random streams. `execution::random_stream` keys a ChaCha8 stream
with the seed, realization, item and purpose identifiers. No worker identity is
involved, so simulation results do not depend on the worker count or on which
storage adapter served the conditioning data.

Numerical policy. `gaussian::NumericalTolerance` is an explicit roundoff
tolerance in covariance units; no nugget is introduced. Singular or
non-positive-definite systems, numerically rank-deficient drift and
insufficient neighborhoods are reported as errors. Memory exhaustion never
reduces a neighborhood.

Out-of-core data. Observation sources stream without materializing the
dataset; `grid::Source` reads windows of a grid whose global indices are
64-bit; `storage::grid` ingests coordinate-keyed grids by bounded external
merge sort. The SQLite adapters bound their page cache and serve neighborhood
queries with O(k) working storage.

Checkpoints. With the `sqlite` feature, `simulation::simulate_checkpointed`
commits each simulated group and its `storage::Checkpoint` in one transaction,
so an interrupted run leaves either the previous or the complete group, never
partial output. `simulate_checkpointed_identified` derives the checkpoint
identity from the model, data, path, seed and options through the `identity`
module before starting or resuming.

## Feature flags

Each feature of `terrustrial` forwards to the feature of the same name in
`terrustrial-core`.

| Feature | Pulls in | Enables |
| --- | --- | --- |
| `sqlite` | `rusqlite` with bundled SQLite | `storage::SqliteStore`, `storage::SqliteReadPool`, `storage::packed`, checkpointed simulation, `variography::disk`, deferred page reads in the execution driver. |
| `fft` | `rustfft`, `concrete-ntt` | `variography::grid::{counts, spectral}`, `mps::fft` and `simulation::mps::simulate_fft_approximate`. |
| `gpu` | `wgpu` (Vulkan, Metal and DX12 backends), `pollster`, `bytemuck` | `compute::gpu::Context`, `mps::gpu` portable WGSL pattern scoring and `simulation::mps::simulate_gpu_approximate`. |
| `mi-allocator` | `mimalloc` | Used by the `storage_comparison` example only; the library installs no global allocator. |
| `profiling` | nothing | Aggregate timing diagnostics in `execution::profiling`. |
| `experimental-two-phase` | nothing | An experimental all-plan simulation benchmark path with no streaming or plan persistence. |

## Quick Sampling

`terrustrial::qs` is an FFT-accelerated Direct Sampling implementation for
regular 3D grids: `simulate_continuous`, `simulate_categorical` and
`simulate_multivariate` take a `Grid3` training image, `HardSample`
conditioning data, a `QsTemplate` and `QsParams`. The simulation grid need not
match the training image's shape, only its cell topology. The examples read
their inputs from paths given by `QS_DATA` (`qs_demo`, a drill-hole style
dataset) and `STREB_TI` (`strebelle_bench`, the Strebelle channel training
image); neither dataset is in the repository.

## Building

- The workspace is developed with Rust 1.98.0 and commits its `Cargo.lock`.
- `sqlite` and `mi-allocator` compile C sources through the `cc` crate, so a C
  compiler is required (MSVC Build Tools on Windows).
- `gpu` compiles without system libraries. A native compute adapter is needed
  at run time; `compute::gpu::Context::new` returns
  `Error::BackendUnavailable` when none is found, and the GPU test is marked
  `#[ignore]` so it must be run explicitly:
  `cargo test -p terrustrial-core --features gpu --test mps_gpu -- --ignored`.

## Benchmarks

`storage_comparison` times estimation or simulation on synthetic inputs with
resident or packed storage and prints a JSON record whose output hash must
match across backends and worker counts:

```bash
cargo run --release -p terrustrial-core --features sqlite --example storage_comparison -- packed simulate 10000 2048 8 131072 64
```

The positional arguments are the backend, the mode, the number of
conditioning points, the number of operations, the worker count, the cache
page count and the admission window; commit and queued group counts are
optional. On Windows, `--placement=physical` or `--placement=physical-writer`
pins workers to distinct physical cores, and `--publication-decode` moves
packed payload decoding to a separate helper core. `scripts/benchmark_windows.ps1`
wraps this into a full scaling sweep.

## Migration from 0.1

The 0.1 modules (`geometry`, `group_operators`, `spatial_database`, `systems`,
`variography`) are not in 0.2, and nothing in 0.2 is source-compatible with
them. The last release of the original design is tagged `v0.1.1` and stays on
crates.io. Where the 0.1 concepts went:

| 0.1 | 0.2 |
| --- | --- |
| `SpatialAcceleratedDB` with `Support` values | `data::Observation` (id, `data::Site`, value, noise variance) in any `spatial::ObservationSource`: a slice, `spatial::SpatialIndex`, or a `storage` adapter |
| `GroupProvider::optimized_groups` | `data::Group` of `data::Target`, built by the caller |
| `CompositeVariogram` of `SphericalVariogram`, `ExponentialVariogram`, ... | `model::Lmc` of `model::Structure` (`Correlation`, `Anisotropy`, `Coregionalization`) |
| `Ellipsoid` search with `ConditioningParams` | `spatial::Neighborhood` (maximum count, radius) and the per-variable quotas in `spatial::search` |
| `generalized_sequential_kriging::estimate` with `SolvedLUOKSystemBuilder` or `SolvedLUSKSystemBuilder` | `estimation::estimate_groups` with `gaussian::Ordinary` or `gaussian::Simple`, or `gaussian::GaussianPlan` for a single system |
| `generalized_sequential_kriging::simulate` | `simulation::simulate_groups` or `simulation::simulate_pipelined` |
| `generalized_sequential_indicator_kriging`, `inverse_distance` | no replacement |

## License

Licensed under either of the MIT license (`LICENSE-MIT`) or the Apache License,
Version 2.0 (`LICENSE-APACHE`), at your option.
