//! Quick Sampling (QS) - FFT-accelerated Direct Sampling for regular 3D
//! grids.
//!
//! References
//! - Gravey & Mariethoz (2020). "QuickSampling v1.0: a robust and simplified
//!   pixel-based multiple-point simulation approach." *Geoscientific Model
//!   Development*.
//!
//! Three drivers:
//! - [`simulate::simulate_continuous`] for continuous attributes.
//! - [`simulate::simulate_categorical`] for categorical attributes.
//! - [`simulate::simulate_multivariate`] for co-located mixed-type variables.
//!
//! All drivers operate on a regular 3D grid for both the training image and
//! the simulation grid. The simulation grid does not need to be the same
//! shape as the TI; only the cell topology matches.

pub mod cluster;
pub mod grid;
pub mod mismatch;
pub mod simulate;
pub mod template;

pub use grid::Grid3;
pub use mismatch::{FftCtx, TiFreq};
pub use simulate::{
    simulate_categorical, simulate_continuous, simulate_multivariate, HardSample,
    MultivariateOutput, MultivariateVariable, QsParams,
};
pub use template::{LagWeight, QsTemplate};
