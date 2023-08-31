use nalgebra::Point3;
use nalgebra::{Isometry3, UnitQuaternion, Vector3};
use parry3d::bounding_volume::Aabb;
use simba::simd::f32x16;
use simba::simd::SimdPartialOrd;
use simba::simd::SimdValue;

use crate::{
    kriging::KrigingParameters, spatial_database::coordinate_system::octant,
    spatial_database::coordinate_system::CoordinateSystem,
};

use super::Geometry;

#[derive(Clone, Debug)]
pub struct Ellipsoid {
    pub a: f32,
    pub b: f32,
    pub c: f32,
    pub coordinate_system: CoordinateSystem,
}

impl Ellipsoid {
    /// Create a new Ellipsoid with given major (a), semi-major (b), and minor axis (c)
    ///  coordinate system defines location and orientation of ellipsoid
    ///      - location is defined by the translation component of the coordinate system
    ///      - orientation is defined by the rotation component of the coordinate system
    pub fn new(a: f32, b: f32, c: f32, coordinate_system: CoordinateSystem) -> Self {
        Self {
            a,
            b,
            c,
            coordinate_system,
        }
    }

    /// Computes the bounding box of the ellipsoid in world coordinates
    pub fn bounding_box(&self) -> Aabb {
        let mins = Point3::new(-self.a, -self.b, -self.c);
        let maxs = Point3::new(self.a, self.b, self.c);

        let bbox = Aabb::new(mins, maxs);
        bbox.transform_by(&self.coordinate_system.local_to_world)
    }

    /// Checks if ellipsoid contains a point (world coordinates)
    pub fn contains(&self, point: &Point3<f32>) -> bool {
        let point = self.coordinate_system.world_to_local.transform_point(point);

        let x = point.x / self.a;
        let y = point.y / self.b;
        let z = point.z / self.c;

        x * x + y * y + z * z <= 1f32
    }

    /// Vectorized version of contains
    #[inline(always)]
    pub fn vectorized_contains(&self, points: &Point3<f32x16>) -> <f32x16 as SimdValue>::SimdBool {
        let rot = *self.coordinate_system.world_to_local.rotation.quaternion();
        let trans = self.coordinate_system.world_to_local.translation;

        let simd_rot = UnitQuaternion::from_quaternion(rot.coords.cast::<f32x16>().into());
        let simd_trans = trans.vector.cast::<f32x16>().into();

        let simd_world_to_local = Isometry3::from_parts(simd_trans, simd_rot);

        let mut points = simd_world_to_local.transform_point(points);

        let normalizer = Vector3::new(
            f32x16::splat(self.a),
            f32x16::splat(self.b),
            f32x16::splat(self.c),
        );
        points.coords.component_div_assign(&normalizer);

        let iso_h = points.coords.norm();

        iso_h.simd_le(f32x16::splat(1.0))
    }

    /// Copmute the isomitrized distance of a point in world coordinate, to the center of the ellipsoid
    pub fn iso_distance(&self, point: &Point3<f32>) -> f32 {
        let point = self.coordinate_system.world_to_local.transform_point(point);

        point.coords.norm()
    }

    // Compute the indices of the points to include in each octant for kriging
    //  NOT USED
    pub fn octant_points(
        &self,
        points: &[Point3<f32>],
        parameters: &KrigingParameters,
    ) -> Vec<usize> {
        let init_size = parameters.min_octant_data;
        let mut octant_points = vec![Vec::with_capacity(init_size); 8];
        let mut octant_flag = vec![Vec::with_capacity(points.len()); 8];

        //insert all points into respective octant
        points.iter().enumerate().for_each(|(i, p)| {
            let point = self.coordinate_system.world_to_local.transform_point(p);
            let octant = octant(&point);
            octant_points[octant as usize - 1].push(*p);
            octant_flag[octant as usize - 1].push(i);
        });

        //sort each octant by distance to origin
        octant_flag.iter_mut().for_each(|octant| {
            octant.sort_by(|a, b| {
                let a = self
                    .coordinate_system
                    .world_to_local
                    .transform_point(&points[*a])
                    .coords;
                let b = self
                    .coordinate_system
                    .world_to_local
                    .transform_point(&points[*b])
                    .coords;

                a.norm().partial_cmp(&b.norm()).unwrap()
            });
        });

        octant_flag
            .iter_mut()
            .for_each(|octant| octant.truncate(parameters.max_octant_data));
        octant_flag.into_iter().flatten().collect()
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

    fn vectorized_contains(&self, points: &Point3<f32x16>) -> <f32x16 as SimdValue>::SimdBool {
        self.vectorized_contains(points)
    }

    fn iso_distance(&self, point: &Point3<f32>) -> f32 {
        self.iso_distance(point)
    }

    fn coordinate_system(&self) -> &CoordinateSystem {
        &self.coordinate_system
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::{Translation3, UnitQuaternion};

    use approx::assert_relative_eq;

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
}
