//! # Terrustrial
//!
//! **Terrustrial** is an experimental Rust library for geostatistics, variograms, and kriging, designed for high performance and flexibility in geospatial analysis.
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
//! use terrustrial::prelude::*;
//! // Read point cloud from file
//! let cond = SpatialAcceleratedDB::from_csv_index(FILE_PATH, "X", "Y", "Z", "CU")
//!     .expect("Failed to create gdb");
//! // Create blocks and group provider
//! let blocks = get_blocks();
//! let groups = GroupProvider::optimized_groups(&blocks, 5.0, 5.0, 10.0, 2, 2, 2);
//! // Variogram setup
//! let vgram_rot = DRotor3::from_euler_angles(0.0, 0.0, 0.0);
//! let range = DVec3::new(100.0, 200.0, 100.0);
//! let sill = 1.0;
//! let spherical_vgram = CompositeVariogram::new(vec![VariogramType::Spherical(
//!     SphericalVariogram::new(range, sill, vgram_rot),
//! )]);
//! // Search ellipsoid
//! let search_ellipsoid = Ellipsoid::new(
//!     200.0, 50.0, 50.0,
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
//! ```
//!
//! ## Modules
//!
//! - [`geometry`]: Geometric primitives and spatial support
//! - [`group_operators`]: Kriging and estimation algorithms
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
pub mod spatial_database;
pub mod systems;
pub mod variography;

pub mod prelude {

    pub mod re_exports {
        pub use rstar;
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
