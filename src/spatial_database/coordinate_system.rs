use nalgebra::{
    Isometry, Isometry3, Point3, SimdRealField, SimdValue, Translation3, UnitQuaternion, Vector3,
};
use num_traits::Float;

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

#[derive(Copy, Clone, Debug)]
pub struct CoordinateSystem<T>
where
    T: Float,
{
    pub translation: Translation3<T>,
    pub rotation: UnitQuaternion<T>,
    pub inverse_rotation: UnitQuaternion<T>,
    pub world_to_local: Isometry3<T>,
    pub local_to_world: Isometry3<T>,
}

impl<T> CoordinateSystem<T>
where
    T: Float + SimdValue + SimdRealField,
    <T as SimdValue>::Element: nalgebra::RealField,
{
    /// Creates a new coordinate system from a translation and a rotation quaternion
    /// # Arguments
    /// * `translation` - translation component of the coordinate system (origin of coordinate system)
    /// * `quat` - rotation component of the coordinate system
    pub fn new(translation: Translation3<T>, quat: UnitQuaternion<T>) -> Self {
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

    /// Set the origin of the coordinate system.
    pub fn set_origin(&mut self, origin: Point3<T>) {
        self.translation = Translation3::new(origin.x, origin.y, origin.z);
        self.local_to_world = Isometry::from_parts(self.translation, self.rotation);
        self.world_to_local = self.local_to_world.inverse();
    }

    /// Set the rotation of the coordinate system.
    pub fn set_rotation(&mut self, quat: UnitQuaternion<T>) {
        self.rotation = quat;
        self.inverse_rotation = quat.inverse();
        self.local_to_world = Isometry::from_parts(self.translation, self.rotation);
        self.world_to_local = self.local_to_world.inverse();
    }

    /// Create a new coordinate system from an origin and euler angles.
    pub fn from_origin_and_euler_angles(origin: Point3<T>, roll: T, pitch: T, yaw: T) -> Self {
        //create rotation quaternion from euler angles
        let rot_quaternion = UnitQuaternion::from_euler_angles(roll, pitch, yaw);
        //create translation quaternion from origin
        let trans = Translation3::new(origin.x, origin.y, origin.z);
        //return coordinate system
        Self::new(trans, rot_quaternion)
    }

    /// Origin of the coordinate system
    pub fn origin(&self) -> Point3<T> {
        Point3::new(self.translation.x, self.translation.y, self.translation.z)
    }

    /// Convert a point from global to local coordinates
    pub fn world_to_local_point(&self, point: &Point3<T>) -> Point3<T> {
        self.world_to_local.transform_point(point)
    }

    /// Convert a vector from global to local coordinates
    pub fn world_to_local_vector(&self, vector: &Vector3<T>) -> Vector3<T> {
        self.world_to_local.transform_vector(vector)
    }

    /// Convert a point from local to global coordinates
    pub fn local_world_point(&self, point: &Point3<T>) -> Point3<T> {
        self.local_to_world.transform_point(point)
    }

    /// Convert a vector from local to global coordinates
    pub fn local_world_vector(&self, vector: &Vector3<T>) -> Vector3<T> {
        self.local_to_world.transform_vector(vector)
    }
}

/// Octant of a point
pub fn octant(point: &Point3<f32>) -> u8 {
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
        let transformed_point = coordinate_system.world_to_local_point(&point);
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
        let transformed_point = coordinate_system.world_to_local_point(&point);
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
        let transformed_point = coordinate_system.world_to_local_point(&point);
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
        let transformed_point = coordinate_system.world_to_local_point(&point);
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
        let transformed_point = coordinate_system.world_to_local_point(&point);
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
        let transformed_point = coordinate_system.world_to_local_point(&point);
        let back_transformed_point = coordinate_system.local_world_point(&transformed_point);
        assert_relative_eq!(point, back_transformed_point, epsilon = 1e-6);
    }
}
