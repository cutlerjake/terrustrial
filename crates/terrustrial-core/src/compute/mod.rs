//! Shared numerical utilities and optional accelerator context.
#[cfg(feature = "gpu")]
pub mod gpu;
pub(crate) mod linalg;
pub(crate) mod scaling;
