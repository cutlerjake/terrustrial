//! Replacement geostatistical engine. Legacy algorithms are not dependencies.
//!
//! Validated inputs, joint covariance, deterministic random streams, and explicit
//! numerical failures are the foundation. The crate README below describes the
//! modules, the resource and numerical contracts, and the feature flags.
//!
#![doc = include_str!("../README.md")]
#![forbid(unsafe_code)]

pub mod anisotropy;
pub mod compute;
pub mod data;
pub mod domains;
pub mod error;
pub mod estimation;
pub mod execution;
pub mod fitting;
pub mod gaussian;
pub mod grid;
pub mod high_order;
pub mod identity;
pub mod model;
pub mod mps;
pub mod simulation;
pub mod spatial;
pub mod storage;
pub mod transforms;
pub mod variography;

pub use error::{Error, Result};
