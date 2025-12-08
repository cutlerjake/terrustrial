//! # Group Operators
//!
//! This module provides core geostatistical estimation algorithms for spatial data analysis.
//!
//! ## Features
//!
//! - **Generalized Sequential Kriging (GSK):** Flexible kriging estimator and simulation routines.
//! - **Generalized Sequential Indicator Kriging (GSIK):** Indicator-based estimation for categorical or thresholded data.
//! - **Inverse Distance Estimation (ID):** Fast, simple spatial interpolation using inverse distance weighting.
//!
//! ## Types
//!
//! - [`ConditioningFilterMap`]: Utility for filtering and clipping conditioning data based on value and distance.
//! - [`ConditioningParams`]: Parameters for controlling the number and selection of conditioning points.
//!
//! ## Usage
//!
//! Use the estimation functions from submodules for spatial prediction and simulation. See crate-level docs for examples.
//!
//! ## Submodules
//!
//! - [`generalized_sequential_kriging`]
//! - [`generalized_sequential_indicator_kriging`]
//! - [`inverse_distance`]
//!

pub mod generalized_sequential_indicator_kriging;
pub mod generalized_sequential_kriging;
pub mod inverse_distance;

/// An entry in the conditioning filter map.
#[derive(Debug, Clone)]
pub struct ClipEntry {
    /// Distance threshold for clipping.
    ///
    /// If the distance h to the conditioning point exceeds this value,
    /// the value will be clipped to the specified range.
    pub h: f64,

    /// Value range for clipping.
    ///
    /// Values outside this range will be clipped to the nearest bound
    /// if the distance exceeds h.
    pub range: [f64; 2],
}

/// Parameters for conditioning data selection and filtering.
#[derive(Debug, Clone)]
pub struct ConditioningFilterMap {
    /// Clips applied to conditioning data based on distance.
    pub clips: Vec<ClipEntry>,

    /// Filter values outside of this value range.
    ///
    /// All values outside this range will be excluded from conditioning data.
    pub valid_value_range: [f64; 2],
}

impl ConditioningFilterMap {
    pub fn filter_map(&self, value: f64, h: f64) -> Option<f64> {
        if value < self.valid_value_range[0] || value > self.valid_value_range[1] {
            return None;
        }
        for clip in self.clips.iter().rev() {
            if h > clip.h {
                if value < clip.range[0] {
                    return Some(clip.range[0]);
                } else if value > clip.range[1] {
                    return Some(clip.range[1]);
                }
            }
        }

        Some(value)
    }
}

/// Parameters for conditioning data selection and filtering.
///
/// These parameters control how many conditioning points are used,
/// how they are selected, and how extreme values are handled.
#[derive(Debug, Clone)]
pub struct ConditioningParams {
    /// Number of conditioning points.
    pub max_n_cond: usize,
    pub min_n_cond: usize,

    /// Limit on the number of conditioning points per octant.
    pub max_octant: usize,
    pub min_conditioned_octants: usize,

    /// Clips extreme values if h > clip_h.
    pub clip_h: Vec<f64>,
    pub clip_range: Vec<[f64; 2]>,

    /// Filter values outside of this value range.
    pub valid_value_range: [f64; 2],

    /// Limit number of points from same source group.
    pub same_source_group_limit: usize,

    /// Dynamic orientation for search ellipsoid.
    pub orient_search: bool,
    /// Dynamic orientation for variogram.
    pub orient_variogram: bool,
}

impl ConditioningParams {
    /// Creates a new [`ConditioningParams`] instance with the specified parameters.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        max_n_cond: usize,
        min_n_cond: usize,
        max_octant: usize,
        min_conditioned_octants: usize,
        clip_h: Vec<f64>,
        clip_range: Vec<[f64; 2]>,
        valid_value_range: [f64; 2],
        same_source_group_limit: usize,
        orient_search: bool,
        orient_variogram: bool,
    ) -> Self {
        Self {
            max_n_cond,
            min_n_cond,
            max_octant,
            min_conditioned_octants,

            clip_h,
            clip_range,

            valid_value_range,

            same_source_group_limit,

            orient_search,
            orient_variogram,
        }
    }
}

impl Default for ConditioningParams {
    fn default() -> Self {
        Self {
            max_n_cond: 32,
            min_n_cond: 4,
            max_octant: 8,
            min_conditioned_octants: 1,
            clip_h: vec![],
            clip_range: vec![],
            valid_value_range: [f64::NEG_INFINITY, f64::INFINITY],
            same_source_group_limit: usize::MAX,
            orient_search: false,
            orient_variogram: false,
        }
    }
}
