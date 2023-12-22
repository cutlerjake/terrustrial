use nalgebra::{Point3, UnitQuaternion};

pub struct VariogramTolerance {
    pub p1: Point3<f32>,
    pub length: f32,
    pub a: f32,
    pub a_tol: f32,
    pub a_dist_threshold: f32,
    pub b: f32,
    pub b_tol: f32,
    pub b_dist_threshold: f32,
    pub orientation: UnitQuaternion<f32>,
}

impl VariogramTolerance {
    pub fn new(
        p1: Point3<f32>,
        length: f32,
        a: f32,
        a_tol: f32,
        b: f32,
        b_tol: f32,
        orientation: UnitQuaternion<f32>,
    ) -> Self {
        let a_dist_threshold = a / a_tol.tan();
        let b_dist_threshold = b / b_tol.tan();

        Self {
            p1,
            length,
            a,
            a_tol,
            a_dist_threshold,
            b,
            b_tol,
            b_dist_threshold,
            orientation,
        }
    }

    pub fn offset_along_axis(&mut self, offset: f32) {
        self.p1 += self.orientation * nalgebra::Vector3::x() * offset;
    }

    pub fn set_base(&mut self, base: Point3<f32>) {
        self.p1 = base;
    }

    pub fn loose_aabb(&self) -> parry3d::bounding_volume::Aabb {
        let offset = f32::max(self.a, self.b);

        let min = Point3::new(self.p1.x - offset, self.p1.y - offset, self.p1.z - offset);

        let cylinder_axis = self.orientation * nalgebra::Vector3::x() * self.length;

        let max = Point3::new(
            self.p1.x + cylinder_axis.x + offset,
            self.p1.y + cylinder_axis.y + offset,
            self.p1.z + cylinder_axis.z + offset,
        );

        parry3d::bounding_volume::Aabb::new(min, max)
    }

    pub fn contains_point(&self, point: Point3<f32>) -> bool {
        let cylinder_axis = self.orientation * nalgebra::Vector3::x() * self.length;
        let length_sq = cylinder_axis.dot(&cylinder_axis);

        let p1 = self.p1;

        let v = point - p1;
        let dot = v.dot(&cylinder_axis);

        if dot <= 0.0 || dot >= length_sq {
            return false;
        }

        let cylinder_aligned_point = self.orientation.inverse_transform_point(&v.into());

        let axial_dist = dot.sqrt();

        let a = if axial_dist >= self.a_dist_threshold {
            self.a
        } else {
            axial_dist * self.a_tol.tan()
        };

        let b = if axial_dist >= self.b_dist_threshold {
            self.b
        } else {
            axial_dist * self.b_tol.tan()
        };

        if cylinder_aligned_point.y * cylinder_aligned_point.y / (a * a)
            + cylinder_aligned_point.z * cylinder_aligned_point.z / (b * b)
            > 1f32
        {
            return false;
        }

        true
    }
}
