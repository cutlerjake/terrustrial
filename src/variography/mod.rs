//! # Variography
//!
//! This module provides tools for experimental and model variogram analysis in geostatistics.
//!
//! ## Features
//!
//! - Submodules for experimental variogram calculation and model variogram definitions.
//! - Support for composite, spherical, exponential, Gaussian, and nugget variogram models.
//!
//! ## Usage
//!
//! Use experimental variogram routines to analyze spatial correlation in data, and model variograms for kriging and simulation.
//!
//! ## Submodules
//!
//! - [`experimental`]: Experimental variogram calculation utilities.
//! - [`model_variograms`]: Model variogram definitions and combinators.
//!
//! ## Example
//!
//! See crate-level documentation for usage examples.
//!

pub mod experimental;
pub mod model_variograms;
