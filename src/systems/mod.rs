//! # Systems
//!
//! This module provides linear system solvers and utilities for geostatistical estimation.
//!
//! ## Features
//!
//! - Submodules for LU decomposition, system building, and solution modifiers.
//! - Solvers for ordinary kriging and related spatial estimation systems.
//! - Modifiers for mean transformation and negative weight filtering.
//!
//! ## Usage
//!
//! Use the provided solvers and builders to construct and solve kriging systems for spatial prediction.
//!
//! ## Submodules
//!
//! - [`lu`]: LU decomposition and system solver.
//! - [`system_builder`]: Utilities for building kriging systems.
//! - [`modifiers`]: System solution modifiers (e.g., mean transform).
//! - [`solved_systems`]: Implementations of solved system types.
//!
//! ## Example
//!
//! See crate-level documentation for usage examples.
//!

pub mod lu;
pub mod modifiers;
pub mod solved_systems;
pub mod system_builder;
