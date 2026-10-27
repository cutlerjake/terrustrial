//! # Coordinate System Utilities
//!
//! Provides a 3D coordinate system abstraction for geospatial transformations and spatial queries.
//!
//! ## Features
//!
//! - [`CoordinateSystem`]: Represents a local/global coordinate system using translation and rotation (via `ultraviolet`).
//!     - Easily convert between local and global coordinates.
//!     - Set origin and rotation dynamically.
//! - `octant`: Utility function to determine the octant of a 3D point.
//!
//! ## Usage
//!
//! Use `CoordinateSystem::new(origin, rotation)` to create a new coordinate system, and access transformations via `into_local` and `into_global`.
//!
//! ## Dependencies
//!
//! - [`ultraviolet`]: For vector, rotation, and isometry math.
//!
//! ## Example
//!
//! ```rust
//! use terrustrial::spatial_database::coordinate_system::*;
//! use ultraviolet::{DRotor3, DVec3};
//! let origin = DVec3::zero();
//! let rotation = DRotor3::from_euler_angles(0.0, 0.0, 0.0);
//! let cs = CoordinateSystem::new(origin, rotation);
//! assert_eq!(cs.into_local().transform_vec(origin), DVec3::zero());
//! ```

use ultraviolet::{DIsometry3, DRotor3, DVec3};

/// 3D Coordinate System
///
/// Represents a coordinate system defined by an origin and rotation.
/// Provides transformations between local and global coordinates.
#[derive(Clone, Copy, Debug)]
pub struct CoordinateSystem {
    into_local: DIsometry3,
    into_global: DIsometry3,
}

impl CoordinateSystem {
    /// Creates a new CoordinateSystem with the specified origin and rotation.
    ///
    /// # Arguments
    /// - `origin`: The origin of the coordinate system in global coordinates.
    /// - `rotation`: The rotation of the coordinate system as a DRotor3.
    #[inline(always)]
    pub fn new(origin: DVec3, rotation: DRotor3) -> Self {
        let into_global = DIsometry3::new(origin, rotation);
        let into_local = into_global.inversed();

        Self {
            into_local,
            into_global,
        }
    }

    /// Returns a reference to the isometry that transforms global coordinates into local coordinates.
    #[inline(always)]
    pub fn into_local(&self) -> &DIsometry3 {
        &self.into_local
    }

    /// Returns a reference to the isometry that transforms local coordinates into global coordinates.
    #[inline(always)]
    pub fn into_global(&self) -> &DIsometry3 {
        &self.into_global
    }

    /// Sets the origin of the coordinate system.
    ///
    /// # Arguments
    /// - `origin`: The new origin in global coordinates.
    #[inline(always)]
    pub fn set_origin(&mut self, origin: DVec3) {
        self.into_global.translation = origin;
        self.into_local = self.into_global.inversed();
    }

    /// Sets the rotation of the coordinate system.
    ///
    /// # Arguments
    /// - `rotation`: The new rotation as a DRotor3.
    #[inline(always)]
    pub fn set_rotation(&mut self, rotation: DRotor3) {
        self.into_global.rotation = rotation;
        self.into_local = self.into_global.inversed();
    }
}

/// Octant of a point
pub fn octant(point: &DVec3) -> u8 {
    match (point.x >= 0.0, point.y >= 0.0, point.z >= 0.0) {
        (true, true, true) => 0,
        (false, true, true) => 1,
        (false, false, true) => 2,
        (true, false, true) => 3,
        (true, true, false) => 4,
        (false, true, false) => 5,
        (false, false, false) => 6,
        (true, false, false) => 7,
    }
}
