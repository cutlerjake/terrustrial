//! # Group Provider
//!
//! This module provides utilities for grouping spatial supports into regular 3D blocks for efficient spatial analysis.
//!
//! ## Features
//!
//! - [`GroupProvider`]: Organizes supports into spatial groups using a grid-based partitioning scheme.
//!     - Optimizes grouping for parallel kriging and simulation.
//!     - Tracks original and group indices for mapping results.
//! - Efficient spatial queries using packed hierarchy acceleration.
//!
//! ## Usage
//!
//! Use `GroupProvider::optimized_groups` to partition supports into groups based on grid dimensions and spacing. This enables batch processing and spatial locality for geostatistical algorithms.
//!
//! ## Example
//!
//! ```rust
//! use terrustrial::{geometry::support::Support, spatial_database::group_provider::GroupProvider};
//! use ultraviolet::DVec3;
//! let supports = vec![
//!     Support::Point(DVec3::zero()),
//!     Support::Point(DVec3::new(1.0, 0.0, 0.0)),
//! ];
//! let groups = GroupProvider::optimized_groups(&supports, 1.0, 1.0, 1.0, 2, 1, 1);
//! assert_eq!(groups.n_nodes(), 2);
//! assert_eq!(groups.n_groups(), 1);
//! ```
//!
//! ## Dependencies
//!
//! - [`terrustrial_core::spatial::hierarchy`]: For spatial indexing and queries.
//! - [`Support`]: Geometric primitives representing spatial locations.
//!

use std::ops::Range;

use terrustrial_core::spatial::hierarchy::{Bounded, Bounds, Hierarchy};

use crate::geometry::support::Support;

use super::SpatialData;

/// Provides grouped spatial supports for batch processing.
///
/// Each group contains a set of supports organized in a regular 3D grid.
#[derive(Debug, Clone)]
pub struct GroupProvider {
    supports: Vec<Support>,
    original_idxs: Vec<usize>,
    group_idxs: Vec<usize>,
}

impl GroupProvider {
    /// Creates optimized groups of supports based on specified grid dimensions and spacing.
    ///
    /// # Arguments
    /// - `supports`: The list of spatial supports to be grouped.
    /// - `dx`: Grid spacing in the x-direction.
    /// - `dy`: Grid spacing in the y-direction.  
    /// - `dz`: Grid spacing in the z-direction.
    /// - `gx`: Number of supports per group in the x-direction.
    /// - `gy`: Number of supports per group in the y-direction.
    /// - `gz`: Number of supports per group in the z-direction.
    pub fn optimized_groups(
        supports: &[Support],
        dx: f64,
        dy: f64,
        dz: f64,
        gx: usize,
        gy: usize,
        gz: usize,
    ) -> Self {
        let target_point_tree = Hierarchy::new(
            supports
                .iter()
                .copied()
                .enumerate()
                .map(|(i, support)| SpatialData {
                    support,
                    data_idx: i as u32,
                })
                .collect(),
        )
        .expect("valid target bounds and available memory");
        let Some(bounds) = target_point_tree.bounds() else {
            return Self {
                supports: vec![],
                original_idxs: vec![],
                group_idxs: vec![0],
            };
        };
        let mut claimed = vec![false; supports.len()];

        let group_size = [gx, gy, gz];
        let env_size = [
            group_size[0] as f64 * dx,
            group_size[1] as f64 * dy,
            group_size[2] as f64 * dz,
        ];
        let mut groups = Vec::new();
        let mut original_idxs = Vec::new();
        let mut x = bounds.lower[0];
        while x <= bounds.upper[0] {
            let mut y = bounds.lower[1];
            while y <= bounds.upper[1] {
                let mut z = bounds.lower[2];
                while z <= bounds.upper[2] {
                    let envelope = Bounds::from_corners(
                        [x, y, z],
                        [x + env_size[0], y + env_size[1], z + env_size[2]],
                    );

                    let mut points = target_point_tree
                        .query(envelope)
                        .filter(|entry| {
                            let i = entry.data_idx as usize;
                            if claimed[i] {
                                false
                            } else {
                                claimed[i] = true;
                                true
                            }
                        })
                        .copied()
                        .collect::<Vec<_>>();

                    //sort points by x, y, z
                    points.sort_by(|a, b| a.bounds().lower.partial_cmp(&b.bounds().lower).unwrap());

                    points
                        .chunks(group_size.iter().product())
                        .for_each(|chunk| {
                            groups.push(chunk.iter().map(|geom| geom.support).collect::<Vec<_>>());
                            original_idxs.push(
                                chunk
                                    .iter()
                                    .map(|geo| geo.data_idx as usize)
                                    .collect::<Vec<_>>(),
                            );
                        });

                    z += env_size[2];
                }
                y += env_size[1]
            }
            x += env_size[0]
        }

        let mut idxs = vec![0];
        let mut cnt = 0;
        groups.iter().for_each(|group| {
            cnt += group.len();
            idxs.push(cnt);
        });
        let flat_groups = groups.into_iter().flatten().collect::<Vec<_>>();
        let flat_idxs = original_idxs.into_iter().flatten().collect::<Vec<_>>();

        Self {
            supports: flat_groups,
            original_idxs: flat_idxs,
            group_idxs: idxs,
        }
    }

    /// Returns a slice of all supports in the group provider.
    #[inline(always)]
    pub fn get_supports(&self) -> &[Support] {
        &self.supports
    }

    /// Returns the total number of supports (nodes) in the group provider.
    #[inline(always)]
    pub fn n_nodes(&self) -> usize {
        self.supports.len()
    }

    /// Returns the total number of groups in the group provider.
    #[inline(always)]
    pub fn n_groups(&self) -> usize {
        self.group_idxs.len().saturating_sub(1)
    }

    /// Returns the range of indices for the supports in the specified group.
    #[inline(always)]
    #[track_caller]
    pub fn get_group_range(&self, group: usize) -> Range<usize> {
        let low = self.group_idxs[group];
        let high = self.group_idxs[group + 1];
        low..high
    }

    /// Returns a slice of supports in the specified group.
    #[inline(always)]
    #[track_caller]
    pub fn get_group(&self, group: usize) -> &[Support] {
        &self.supports[self.get_group_range(group)]
    }

    /// Returns a slice of original indices for the supports in the specified group.
    #[inline(always)]
    #[track_caller]
    pub fn get_original_idxs(&self, group: usize) -> &[usize] {
        &self.original_idxs[self.get_group_range(group)]
    }

    /// Returns the maximum size (number of supports) among all groups.
    #[inline(always)]
    pub fn max_group_size(&self) -> usize {
        self.group_idxs
            .windows(2)
            .map(|window| window[1] - window[0])
            .max()
            .unwrap_or(0)
    }
}
