use nalgebra::{Isometry3, UnitQuaternion, Vector3};
use nalgebra::{Point3, SimdRealField};
use parry3d::bounding_volume::Aabb;
use rand::Rng;
use simba::simd::SimdValue;

use crate::spatial_database::coordinate_system::CoordinateSystem;

use super::Geometry;

#[derive(Clone, Debug)]
pub struct Ellipsoid {
    pub a: f32,
    pub b: f32,
    pub c: f32,
    pub coordinate_system: CoordinateSystem<f32>,
}

impl Ellipsoid {
    /// Create a new Ellipsoid with given major (a), semi-major (b), and minor axis (c)
    ///  coordinate system defines location and orientation of ellipsoid
    ///      - location is defined by the translation component of the coordinate system
    ///      - orientation is defined by the rotation component of the coordinate system
    pub fn new(a: f32, b: f32, c: f32, coordinate_system: CoordinateSystem<f32>) -> Self {
        Self {
            a,
            b,
            c,
            coordinate_system,
        }
    }

    /// Computes the bounding box of the ellipsoid in world coordinates
    pub fn bounding_box(&self) -> Aabb {
        // let mins = Point3::new(-self.a, -self.b, -self.c);
        // let maxs = Point3::new(self.a, self.b, self.c);

        let mins = Point3::new(-self.b, -self.a, -self.c);
        let maxs = Point3::new(self.b, self.a, self.c);

        let bbox = Aabb::new(mins, maxs);
        bbox.transform_by(&self.coordinate_system.local_to_world)
    }

    pub fn normalized_local_distance_sq(&self, point: &Point3<f32>) -> f32 {
        let u = point.y / self.a;
        let v = point.x / self.b;
        let w = point.z / self.c;

        u * u + v * v + w * w
    }

    pub fn normalized_local_distance(&self, point: &Point3<f32>) -> f32 {
        self.normalized_local_distance_sq(point).sqrt()
    }

    /// Checks if ellipsoid contains a point (world coordinates)
    pub fn contains(&self, point: &Point3<f32>) -> bool {
        let point = self.coordinate_system.world_to_local.transform_point(point);

        self.normalized_local_distance_sq(&point) <= 1.0
    }

    pub fn contains_local_point(&self, point: &Point3<f32>) -> bool {
        self.normalized_local_distance_sq(point) <= 1.0
    }

    //randomly rotate search ellipsoid
    pub fn random_rotation(&self) -> Self {
        let mut rng = rand::thread_rng();

        let x = rng.gen_range(0.0..2.0 * std::f32::consts::PI);
        let y = rng.gen_range(0.0..2.0 * std::f32::consts::PI);
        let z = rng.gen_range(0.0..2.0 * std::f32::consts::PI);

        let rot = UnitQuaternion::from_euler_angles(x, y, z);

        let cs = CoordinateSystem::new(self.coordinate_system.origin().coords.into(), rot);

        Self::new(self.a, self.b, self.c, cs)
    }
}

impl Geometry for Ellipsoid {
    fn bounding_box(&self) -> Aabb {
        self.bounding_box()
    }

    fn contains(&self, point: &Point3<f32>) -> bool {
        self.contains(point)
    }

    fn translate_to(&mut self, translation: &Point3<f32>) {
        self.coordinate_system.set_origin(*translation);
    }

    fn vectorized_contains<T>(&self, points: &Point3<T>) -> <T as SimdValue>::SimdBool
    where
        T: SimdValue<Element = f32> + Clone + SimdRealField,
    {
        let rot = *self.coordinate_system.world_to_local.rotation.quaternion();
        let trans = self.coordinate_system.world_to_local.translation;

        let simd_rot = UnitQuaternion::from_quaternion(rot.coords.map(|x| T::splat(x)).into());
        let simd_trans = trans.vector.map(|x| T::splat(x)).into();

        let simd_world_to_local = Isometry3::from_parts(simd_trans, simd_rot);

        let mut points = simd_world_to_local.transform_point(points);

        let normalizer = Vector3::new(T::splat(self.a), T::splat(self.b), T::splat(self.c));

        points.coords.component_div_assign(&normalizer);

        let iso_h = points.coords.norm();

        iso_h.simd_le(T::splat(1.0))
    }

    #[inline(always)]
    fn vectorized_iso_distance<T>(&self, point: &Point3<T>) -> T
    where
        T: SimdValue<Element = f32> + Clone + SimdRealField,
    {
        let rot = *self.coordinate_system.world_to_local.rotation.quaternion();
        let trans = self.coordinate_system.world_to_local.translation;

        let simd_rot = UnitQuaternion::from_quaternion(rot.coords.map(|x| T::splat(x)).into());
        let simd_trans = trans.vector.map(|x| T::splat(x)).into();

        let simd_world_to_local = Isometry3::from_parts(simd_trans, simd_rot);

        let point = simd_world_to_local.transform_point(point);

        point.coords.norm()
    }

    fn coordinate_system<V>(&self) -> &CoordinateSystem<f32> {
        &self.coordinate_system
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::{Translation3, UnitQuaternion};

    use approx::assert_relative_eq;
    use simba::simd::WideF32x4;

    use super::*;
    use crate::geometry::ellipsoid::Ellipsoid;
    use crate::spatial_database::coordinate_system::CoordinateSystem;

    #[test]
    fn test_ellipse_bounding_box() {
        let coordinate_system = CoordinateSystem::new(
            Translation3::new(0f32, 0f32, 0f32),
            UnitQuaternion::from_euler_angles(0.0, 0.0, 0.0),
        );

        let ellipse = Ellipsoid::new(1f32, 2f32, 3f32, coordinate_system);

        let bbox = ellipse.bounding_box();

        assert_relative_eq!(bbox.mins.x, -1f32, epsilon = 0.0001);
        assert_relative_eq!(bbox.mins.y, -2f32, epsilon = 0.0001);
        assert_relative_eq!(bbox.mins.z, -3f32, epsilon = 0.0001);

        assert_relative_eq!(bbox.maxs.x, 1f32, epsilon = 0.0001);
        assert_relative_eq!(bbox.maxs.y, 2f32, epsilon = 0.0001);
        assert_relative_eq!(bbox.maxs.z, 3f32, epsilon = 0.0001);
    }

    #[test]
    fn test_translated_ellipse_bounding_box() {
        let coordinate_system = CoordinateSystem::new(
            Translation3::new(1f32, 2f32, 3f32),
            UnitQuaternion::from_euler_angles(0.0, 0.0, 0.0),
        );

        let ellipse = Ellipsoid::new(1f32, 2f32, 3f32, coordinate_system);

        let bbox = ellipse.bounding_box();
        println!("{:?}", bbox);

        assert_relative_eq!(bbox.mins.x, 0f32, epsilon = 0.0001);
        assert_relative_eq!(bbox.mins.y, 0f32, epsilon = 0.0001);
        assert_relative_eq!(bbox.mins.z, 0f32, epsilon = 0.0001);

        assert_relative_eq!(bbox.maxs.x, 2f32, epsilon = 0.0001);
        assert_relative_eq!(bbox.maxs.y, 4f32, epsilon = 0.0001);
        assert_relative_eq!(bbox.maxs.z, 6f32, epsilon = 0.0001);
    }

    #[test]
    fn test_rotated_ellipse_bounding_box() {
        let coordinate_system = CoordinateSystem::new(
            Translation3::new(0f32, 0f32, 0f32),
            UnitQuaternion::from_euler_angles(0.0, 0.0, std::f32::consts::PI / 2f32),
        );

        let ellipse = Ellipsoid::new(1f32, 2f32, 3f32, coordinate_system);

        let bbox = ellipse.bounding_box();

        assert_relative_eq!(bbox.mins.x, -2f32, epsilon = 0.0001);
        assert_relative_eq!(bbox.mins.y, -1f32, epsilon = 0.0001);
        assert_relative_eq!(bbox.mins.z, -3f32, epsilon = 0.0001);

        assert_relative_eq!(bbox.maxs.x, 2f32, epsilon = 0.0001);
        assert_relative_eq!(bbox.maxs.y, 1f32, epsilon = 0.0001);
        assert_relative_eq!(bbox.maxs.z, 3f32, epsilon = 0.0001);
    }

    #[test]
    fn test_transformed_ellipse_bounding_box() {
        let coordinate_system = CoordinateSystem::new(
            Translation3::new(1f32, 2f32, 3f32),
            UnitQuaternion::from_euler_angles(0.0, 0.0, std::f32::consts::PI / 2f32),
        );

        let ellipse = Ellipsoid::new(1f32, 2f32, 3f32, coordinate_system);

        let bbox = ellipse.bounding_box();

        assert_relative_eq!(bbox.mins.x, -1f32, epsilon = 0.0001);
        assert_relative_eq!(bbox.mins.y, 1f32, epsilon = 0.0001);
        assert_relative_eq!(bbox.mins.z, 0f32, epsilon = 0.0001);

        assert_relative_eq!(bbox.maxs.x, 3f32, epsilon = 0.0001);
        assert_relative_eq!(bbox.maxs.y, 3f32, epsilon = 0.0001);
        assert_relative_eq!(bbox.maxs.z, 6f32, epsilon = 0.0001);
    }

    #[test]
    fn test_ellipse_contains_point() {
        let coordinate_system = CoordinateSystem::new(
            Translation3::new(0f32, 0f32, 0f32),
            UnitQuaternion::from_euler_angles(0.0, 0.0, 0.0),
        );

        let ellipse = Ellipsoid::new(1f32, 2f32, 3f32, coordinate_system);

        let point = Point3::new(0f32, 0f32, 0f32);

        assert!(ellipse.contains(&point));
    }

    #[test]
    fn test_ellipse_contains_vec_point() {
        let coordinate_system = CoordinateSystem::new(
            Translation3::new(0f32, 0f32, 0f32),
            UnitQuaternion::from_euler_angles(0.0, 0.0, 0.0),
        );

        let ellipse = Ellipsoid::new(1f32, 2f32, 3f32, coordinate_system);

        let point = Point3::new(
            WideF32x4::splat(0f32),
            WideF32x4::splat(0f32),
            WideF32x4::splat(0f32),
        );

        assert!(ellipse.vectorized_contains(&point).0.all());

        let point = Point3::new(
            WideF32x4::splat(1f32),
            WideF32x4::splat(0f32),
            WideF32x4::splat(0f32),
        );

        assert!(ellipse.vectorized_contains(&point).0.all());

        let point = Point3::new(
            WideF32x4::splat(2f32),
            WideF32x4::splat(0f32),
            WideF32x4::splat(0f32),
        );

        assert!(!ellipse.vectorized_contains(&point).0.all());
    }
}
