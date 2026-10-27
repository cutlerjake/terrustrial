# terrustrial-core

Joint geostatistical modeling with explicit resource and numerical contracts.

`terrustrial-core` is the estimation and simulation engine of the
[terrustrial](https://github.com/cutlerjake/terrustrial) workspace. It replaces
the original `terrustrial` 0.1 design rather than extending it: the legacy
algorithms are not dependencies, and the API is new. The crate forbids
`unsafe` code. The `terrustrial` crate re-exports every module of this one
under `terrustrial::`, forwards the same feature flags, and adds Quick
Sampling multiple-point simulation.

Every driver takes validated inputs, a joint covariance model, a memory budget
and a cancellation token, and reports numerical trouble as an error value
instead of returning a silently adjusted result.

## Example

Ordinary kriging of one variable at two targets from four point observations.
The prepared `GaussianPlan` can be reused for repeated prediction, and the
joint prediction-error covariance is retained because it was requested.

```rust
use terrustrial_core::{
    data::{Observation, Point3, Site},
    execution::{Cancellation, MemoryBudget},
    gaussian::{GaussianPlan, NumericalTolerance, Ordinary, PrepareOptions, Uncertainty},
    model::{Anisotropy, Coregionalization, Correlation, Lmc, Structure},
};

fn site(x: f64, y: f64) -> terrustrial_core::Result<Site> {
    Ok(Site::point(Point3::new([x, y, 0.0])?, 0))
}

fn main() -> terrustrial_core::Result<()> {
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
rolling lookahead window. The integration tests under `tests/` check every
driver against independent oracles and are the most complete usage reference.

## Modules

| Module | Contents |
| --- | --- |
| `data` | Validated points, point and block supports, sites, observations, targets and groups. |
| `model` | Stationary linear model of coregionalization: nugget, spherical, exponential, Gaussian and Matern correlations, anisotropy frames, factor-form coregionalization and block quadrature. |
| `gaussian` | Generalized least squares and joint conditional Gaussian prediction: simple, ordinary, linear and external drift, reusable plans and workspaces, conditioning diagnostics. |
| `estimation` | Ordered parallel estimation over groups, with reusable storage, residual and factorial variants. |
| `simulation` | Grouped joint sequential Gaussian simulation with explicit path and conditioning dependencies, pipelined and checkpointed drivers, and MPS drivers. |
| `spatial` | Stable neighborhood selection and the resident spatial index. |
| `storage` | Persistent observation adapters: binary point files, SQLite stores and pools, packed spatial pages and streaming grid files. |
| `grid` | Bounded reads of immutable regular-grid columns with 64-bit global indices. |
| `execution` | Memory budgets and reservations, cancellation, deterministic random streams, execution options and thread startup hooks. |
| `identity` | Versioned, domain-separated BLAKE3 fingerprints of structured inputs for checkpoint identity. |
| `variography` | Experimental variograms with explicit pair and boundary conventions, constrained single-structure fitting, and sampled, indexed, disk and grid variants. |
| `fitting` | Joint least-squares LMC fitting with positive-semidefinite sill constraints. |
| `transforms` | Marginal normal-score transforms with explicit tie and tail semantics. |
| `anisotropy` | Nonstationary covariance with local tensor fitting (Paciorek and Schervish 2006). |
| `domains` | Implicit scalar-field interfaces, domain classification, drillhole evidence scoring and fail-closed validation. |
| `high_order` | Statistical learning of moments: conditional kernel embeddings fitted to positive joint mixtures (Yao et al. 2020). |
| `mps` | Joint mixed-variable k-best pattern sampling; the exhaustive scalar engine is the correctness oracle for the accelerated matchers. |
| `compute` | Shared numerical utilities and the optional GPU context. |

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

The default feature set is empty.

| Feature | Pulls in | Enables |
| --- | --- | --- |
| `sqlite` | `rusqlite` with bundled SQLite | `storage::SqliteStore`, `storage::SqliteReadPool`, `storage::packed`, checkpointed simulation, `variography::disk`, deferred page reads in the execution driver. |
| `fft` | `rustfft`, `concrete-ntt` | `variography::grid::{counts, spectral}`, `mps::fft` and `simulation::mps::simulate_fft_approximate`. |
| `gpu` | `wgpu` (Vulkan, Metal and DX12 backends), `pollster`, `bytemuck` | `compute::gpu::Context`, `mps::gpu` portable WGSL pattern scoring and `simulation::mps::simulate_gpu_approximate`. |
| `mi-allocator` | `mimalloc` | Used by the `storage_comparison` example only; the library installs no global allocator. |
| `profiling` | nothing | Aggregate timing diagnostics in `execution::profiling`. |
| `experimental-two-phase` | nothing | An experimental all-plan simulation benchmark path with no streaming or plan persistence. |

## Build requirements

- `sqlite` and `mi-allocator` compile C sources through the `cc` crate, so a C
  compiler is required (MSVC Build Tools on Windows).
- `gpu` compiles without system libraries. A native compute adapter is needed
  at run time; `compute::gpu::Context::new` returns
  `Error::BackendUnavailable` when none is found, and the GPU test is marked
  `#[ignore]` so it must be run explicitly.

## License

Licensed under either of the MIT license or the Apache License, Version 2.0,
at your option.
