//! Geostatistics in Rust. This crate re-exports every module of
//! [`terrustrial_core`], the joint estimation and simulation engine, forwards
//! its feature flags, and adds the [`qs`] Quick Sampling module. The
//! repository README follows.
//!
#![doc = include_str!("../README.md")]

pub use terrustrial_core::*;

pub mod qs;
