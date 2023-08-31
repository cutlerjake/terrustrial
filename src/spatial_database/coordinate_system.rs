use nalgebra::{Isometry, Isometry3, Point3, Translation3, UnitQuaternion};

use simba::simd::f32x16;

//use std::simd::f32x16;

#[derive(Debug, Copy, Clone, PartialEq, PartialOrd)]
pub struct GridSpacing {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl GridSpacing {
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Default)]
pub struct CoordinateSystem {
    pub translation: Translation3<f32>,
    pub rotation: UnitQuaternion<f32>,
    pub inverse_rotation: UnitQuaternion<f32>,
    pub world_to_local: Isometry3<f32>,
    pub local_to_world: Isometry3<f32>,
}

impl CoordinateSystem {
    /// Creates a new coordinate system from a translation and a rotation quaternion
    /// # Arguments
    /// * `translation` - translation component of the coordinate system (origin of coordinate system)
    /// * `quat` - rotation component of the coordinate system
    pub fn new(translation: Translation3<f32>, quat: UnitQuaternion<f32>) -> Self {
        let local_to_world = Isometry::from_parts(translation, quat);
        let world_to_local = local_to_world.inverse();
        Self {
            translation,
            rotation: quat,
            inverse_rotation: quat.inverse(),
            world_to_local,
            local_to_world,
        }
    }

    /// Set the origin of the coordinate system
    pub fn set_origin(&mut self, origin: Point3<f32>) {
        self.translation = Translation3::new(origin.x, origin.y, origin.z);
        self.local_to_world = Isometry::from_parts(self.translation, self.rotation);
        self.world_to_local = self.local_to_world.inverse();
    }

    /// Create a new coordinate system from an origin and euler angles
    pub fn from_origin_and_euler_angles(
        origin: Point3<f32>,
        roll: f32,
        pitch: f32,
        yaw: f32,
    ) -> Self {
        //create rotation quaternion from euler angles
        let rot_quaternion = UnitQuaternion::from_euler_angles(roll, pitch, yaw);
        //create translation quaternion from origin
        let trans = Translation3::new(origin.x, origin.y, origin.z);
        //return coordinate system
        Self::new(trans, rot_quaternion)
    }

    /// Origin of the coordinate system
    pub fn origin(&self) -> Point3<f32> {
        Point3::new(self.translation.x, self.translation.y, self.translation.z)
    }

    /// Convert a point from global to local coordinates
    pub fn global_to_local(&self, point: &Point3<f32>) -> Point3<f32> {
        self.world_to_local.transform_point(point)
    }

    /// Vectorized version of global to local transformation
    pub fn vectorized_global_to_local_isomety(&self) -> Isometry3<f32x16> {
        let rot = *self.world_to_local.rotation.quaternion();
        let trans = self.world_to_local.translation;

        let simd_rot = UnitQuaternion::from_quaternion(rot.coords.cast::<f32x16>().into());
        let simd_trans = trans.vector.cast::<f32x16>().into();

        Isometry3::from_parts(simd_trans, simd_rot)
    }

    /// Convert a point from local to global coordinates
    pub fn local_to_global(&self, point: &Point3<f32>) -> Point3<f32> {
        self.local_to_world.transform_point(point)
    }

    /// Vectorized version of local to global transformation
    pub fn vectorized_local_to_global(&self) -> Isometry3<f32x16> {
        let rot = *self.local_to_world.rotation.quaternion();
        let trans = self.local_to_world.translation;

        let simd_rot = UnitQuaternion::from_quaternion(rot.coords.cast::<f32x16>().into());
        let simd_trans = trans.vector.cast::<f32x16>().into();

        Isometry3::from_parts(simd_trans, simd_rot)
    }
}

/// Octant of a point
pub fn octant(point: &Point3<f32>) -> u8 {
    match (point.x >= 0.0, point.y >= 0.0, point.z >= 0.0) {
        (true, true, true) => 1,
        (false, true, true) => 2,
        (false, false, true) => 3,
        (true, false, true) => 4,
        (true, true, false) => 5,
        (false, true, false) => 6,
        (false, false, false) => 7,
        (true, false, false) => 8,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f32::consts::PI;
    #[test]
    fn no_offset_no_rotation() {
        let origin = Point3::new(0.0, 0.0, 0.0);
        let roll = 0.0;
        let pitch = 0.0;
        let yaw = 0.0;
        let coordinate_system =
            CoordinateSystem::from_origin_and_euler_angles(origin, roll, pitch, yaw);
        let point = Point3::new(1.0, 1.0, 1.0);
        let transformed_point = coordinate_system.global_to_local(&point);
        assert_relative_eq!(transformed_point, point, epsilon = 1e-6);
    }

    #[test]
    fn offset_no_rotation() {
        let origin = Point3::new(1.0, 1.0, 1.0);
        let roll = 0.0;
        let pitch = 0.0;
        let yaw = 0.0;
        let coordinate_system =
            CoordinateSystem::from_origin_and_euler_angles(origin, roll, pitch, yaw);
        let point = Point3::new(2.0, 2.0, 2.0);
        let transformed_point = coordinate_system.global_to_local(&point);
        assert_relative_eq!(
            transformed_point,
            Point3::new(1.0, 1.0, 1.0),
            epsilon = 1e-6
        );
    }

    #[test]
    fn no_offset_rotation() {
        let origin = Point3::new(0.0, 0.0, 0.0);
        let roll = 0.0;
        let pitch = 0.0;
        let yaw = PI / 2.0;
        let coordinate_system =
            CoordinateSystem::from_origin_and_euler_angles(origin, roll, pitch, yaw);
        let point = Point3::new(1.0, 1.0, 0.0);
        let transformed_point = coordinate_system.global_to_local(&point);
        assert_relative_eq!(
            transformed_point,
            Point3::new(1.0, -1.0, 0.0),
            epsilon = 1e-6
        );
    }

    #[test]
    fn offset_1d_rotation() {
        let origin = Point3::new(1.0, 1.0, 1.0);
        let roll = 0.0;
        let pitch = 0.0;
        let yaw = PI / 2.0;
        let coordinate_system =
            CoordinateSystem::from_origin_and_euler_angles(origin, roll, pitch, yaw);
        let point = Point3::new(2.0, 1.0, 1.0);
        let transformed_point = coordinate_system.global_to_local(&point);
        assert_relative_eq!(
            transformed_point,
            Point3::new(0f32, -1f32, 0f32),
            epsilon = 1e-6
        );
    }

    //test 3d roation where point is not at origin
    #[test]
    fn offset_3d_rotation() {
        let origin = Point3::new(1.0, 2.0, 3.0);
        let roll = PI / 3.0;
        let pitch = PI / 4.0;
        let yaw = PI / 2.0;
        let coordinate_system =
            CoordinateSystem::from_origin_and_euler_angles(origin, roll, pitch, yaw);
        let point = Point3::new(2.0, 1.0, 1.0);
        let transformed_point = coordinate_system.global_to_local(&point);
        assert_relative_eq!(
            transformed_point,
            Point3::new(0.7071068, -2.3371172, -0.19463491),
            epsilon = 1e-6
        );
    }

    #[test]
    fn back_offset_rotation() {
        let origin = Point3::new(1.0, 2.0, 3.0);
        let roll = PI / 3.0;
        let pitch = PI / 4.0;
        let yaw = PI / 2.0;
        let coordinate_system =
            CoordinateSystem::from_origin_and_euler_angles(origin, roll, pitch, yaw);
        let point = Point3::new(1.3660254, 4.38013939, 2.55171226);
        let transformed_point = coordinate_system.global_to_local(&point);
        let back_transformed_point = coordinate_system.local_to_global(&transformed_point);
        assert_relative_eq!(point, back_transformed_point, epsilon = 1e-6);
    }
}
