use ultraviolet::DVec3;

use crate::spatial_database::coordinate_system::CoordinateSystem;

use super::aabb::Aabb;

#[derive(Clone, Debug)]
pub struct Ellipsoid {
    pub a: f64,
    pub b: f64,
    pub c: f64,
    pub coordinate_system: CoordinateSystem,
}

impl Ellipsoid {
    /// Create a new Ellipsoid with given major (a), semi-major (b), and minor axis (c)
    ///  coordinate system defines location and orientation of ellipsoid
    ///      - location is defined by the translation component of the coordinate system
    ///      - orientation is defined by the rotation component of the coordinate system
    pub fn new(a: f64, b: f64, c: f64, coordinate_system: CoordinateSystem) -> Self {
        Self {
            a,
            b,
            c,
            coordinate_system,
        }
    }

    /// Computes the bounding box of the ellipsoid in world coordinates
    pub fn bounding_box(&self) -> Aabb {
        let mins = DVec3::new(-self.b, -self.a, -self.c);
        let maxs = DVec3::new(self.b, self.a, self.c);

        let bbox = Aabb::from_min_max(mins, maxs);
        bbox.transformed_by(*self.coordinate_system.into_global())
    }

    pub fn normalized_local_distance_sq(&self, point: &DVec3) -> f64 {
        let u = point.y / self.a;
        let v = point.x / self.b;
        let w = point.z / self.c;

        u * u + v * v + w * w
    }

    pub fn normalized_local_distance(&self, point: &DVec3) -> f64 {
        self.normalized_local_distance_sq(point).sqrt()
    }

    /// Checks if ellipsoid contains a point (world coordinates)
    pub fn contains(&self, point: &DVec3) -> bool {
        let point = self.coordinate_system.into_local().transform_vec(*point);

        self.normalized_local_distance_sq(&point) <= 1.0
    }

    pub fn contains_local_point(&self, point: &DVec3) -> bool {
        self.normalized_local_distance_sq(point) <= 1.0
    }

    pub fn translate_to(&mut self, origin: DVec3) {
        self.coordinate_system.set_origin(origin);
    }

    #[inline(always)]
    pub fn may_contain_local_point_at_sq_dist(&self, dist: f64) -> bool {
        self.a * self.a > dist || self.b * self.b > dist
    }
}
