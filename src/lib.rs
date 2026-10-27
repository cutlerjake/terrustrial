//! # Terrustrial
//!
//! **Terrustrial** is an experimental Rust library for geostatistics, variograms, and kriging, designed for high performance and flexibility in geospatial analysis.
//!
//! This crate keeps the original 0.1 estimation API and adds the [`qs`] Quick
//! Sampling module. The joint estimation and simulation engine introduced in 0.2
//! lives in the separate [`terrustrial_core`] crate, which has a new API; see
//! the repository README for the workspace layout and a migration note.
//!
//! ## Features
//!
//! - **Geometry primitives**: Axis-aligned bounding boxes, ellipsoids, and support for spatial queries.
//! - **Spatial databases**: Efficient spatial indexing and coordinate system utilities.
//! - **Group operators**: Kriging, indicator kriging, and inverse distance estimation methods.
//! - **Variography**: Experimental and model variograms, including spherical, exponential, and composite models.
//! - **System solvers**: Linear algebra utilities for solving kriging systems.
//!
//! ## Example
//!
//! ```rust
//! use terrustrial::{
//!     geometry::{ellipsoid::Ellipsoid, support::Support},
//!     group_operators::{generalized_sequential_kriging::estimate, ConditioningParams},
//!     spatial_database::{
//!         coordinate_system::CoordinateSystem, group_provider::GroupProvider,
//!         SpatialAcceleratedDB,
//!     },
//!     systems::solved_systems::ok_system::SolvedLUOKSystemBuilder,
//!     variography::model_variograms::{
//!         composite::{CompositeVariogram, VariogramType}, spherical::SphericalVariogram,
//!     },
//! };
//! use ultraviolet::{DRotor3, DVec3};
//!
//! let samples = vec![
//!     Support::Point(DVec3::new(0.0, 0.0, 0.0)),
//!     Support::Point(DVec3::new(1.0, 0.0, 0.0)),
//!     Support::Point(DVec3::new(0.0, 1.0, 0.0)),
//!     Support::Point(DVec3::new(0.0, 0.0, 1.0)),
//! ];
//! let cond = SpatialAcceleratedDB::new(samples, vec![1.0, 2.0, 3.0, 4.0]);
//! let targets = vec![Support::Point(DVec3::new(0.5, 0.5, 0.5))];
//! let groups = GroupProvider::optimized_groups(&targets, 1.0, 1.0, 1.0, 1, 1, 1);
//! // Variogram setup
//! let vgram_rot = DRotor3::from_euler_angles(0.0, 0.0, 0.0);
//! let range = DVec3::new(8.0, 8.0, 8.0);
//! let sill = 1.0;
//! let spherical_vgram = CompositeVariogram::new(vec![VariogramType::Spherical(
//!     SphericalVariogram::new(range, sill, vgram_rot),
//! )]);
//! // Search ellipsoid
//! let search_ellipsoid = Ellipsoid::new(
//!     2.0, 2.0, 2.0,
//!     CoordinateSystem::new(DVec3::zero(), vgram_rot),
//! );
//! // Conditioning params
//! let params = ConditioningParams::default();
//! // Estimate grades using ordinary kriging
//! let values = estimate(
//!     &cond,
//!     &params,
//!     &spherical_vgram,
//!     search_ellipsoid,
//!     &groups,
//!     SolvedLUOKSystemBuilder,
//! );
//! assert_eq!(values.len(), 1);
//! assert!(values[0].is_finite());
//! ```
//!
//! ## Modules
//!
//! - [`geometry`]: Geometric primitives and spatial support
//! - [`group_operators`]: Kriging and estimation algorithms
//! - [`qs`]: Quick Sampling multiple-point simulation on regular grids
//! - [`spatial_database`]: Spatial data structures and indexing
//! - [`systems`]: Linear system solvers and modifiers
//! - [`variography`]: Variogram models and experimental tools
//!
//! ## License
//!
//! Licensed under MIT or Apache-2.0.

use ultraviolet::DVec3;

pub mod geometry;
pub mod group_operators;
pub mod qs;
pub mod spatial_database;
pub mod systems;
pub mod variography;

pub mod prelude {

    pub mod re_exports {
        pub use terrustrial_core::spatial::hierarchy;
        pub use ultraviolet;
    }

    pub use crate::group_operators::{
        generalized_sequential_indicator_kriging::estimate as gsik_estimate,
        generalized_sequential_kriging::{estimate as gsk_estimate, simulate as gsk_simulate},
        inverse_distance::estimate as id_estimate,
    };
}

pub const FORWARD: DVec3 = DVec3 {
    x: 0.0,
    y: 1.0,
    z: 0.0,
};
pub const RIGHT: DVec3 = DVec3 {
    x: 1.0,
    y: 0.0,
    z: 0.0,
};
pub const UP: DVec3 = DVec3 {
    x: 0.0,
    y: 0.0,
    z: 1.0,
};
