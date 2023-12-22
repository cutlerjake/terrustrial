use nalgebra::{Point3, UnitQuaternion};

pub struct EllipticalCylindar {
    pub p1: Point3<f32>,
    pub length: f32,
    pub a: f32,
    pub b: f32,
    pub orientation: UnitQuaternion<f32>,
}

impl EllipticalCylindar {
    pub fn new(
        p1: Point3<f32>,
        length: f32,
        a: f32,
        b: f32,
        orientation: UnitQuaternion<f32>,
    ) -> Self {
        Self {
            p1,
            length,
            a,
            b,
            orientation,
        }
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

        if cylinder_aligned_point.y * cylinder_aligned_point.y / (self.a * self.a)
            + cylinder_aligned_point.z * cylinder_aligned_point.z / (self.b + self.b)
            > 1f32
        {
            return false;
        }

        true
    }
}
