use nalgebra::{Point3, UnitQuaternion};
use parry3d::bounding_volume::{Aabb, BoundingVolume};

use crate::{FORWARD, RIGHT, UP};

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
        let cylinder_axis = self.orientation * FORWARD.into_inner() * self.length;
        let aabb1 = bounding_box_of_oriented_ellipse(self.p1, self.a, self.b, self.orientation);
        let aabb2 = bounding_box_of_oriented_ellipse(
            self.p1 + cylinder_axis,
            self.a,
            self.b,
            self.orientation,
        );

        aabb1.merged(&aabb2)

        // let offset = f32::max(self.a, self.b);

        // let min = Point3::new(self.p1.x - offset, self.p1.y - offset, self.p1.z - offset);

        // let max = Point3::new(
        //     self.p1.x + cylinder_axis.x + offset,
        //     self.p1.y + cylinder_axis.y + offset,
        //     self.p1.z + cylinder_axis.z + offset,
        // );

        // parry3d::bounding_volume::Aabb::new(min, max)
    }

    pub fn contains_point(&self, point: Point3<f32>) -> bool {
        let cylinder_axis = self.orientation * FORWARD.into_inner() * self.length;
        let length_sq = cylinder_axis.dot(&cylinder_axis);

        let p1 = self.p1;

        let v = point - p1;
        let dot = v.dot(&cylinder_axis);

        if dot <= 0.0 || dot >= length_sq {
            return false;
        }

        let cylinder_aligned_point = self.orientation.inverse_transform_point(&v.into());

        if cylinder_aligned_point.x * cylinder_aligned_point.x / (self.a * self.a)
            + cylinder_aligned_point.z * cylinder_aligned_point.z / (self.b + self.b)
            > 1f32
        {
            return false;
        }

        true
    }
}

// Function to compute the bounding box of an oriented ellipse in 3D space
pub fn bounding_box_of_oriented_ellipse(
    center: Point3<f32>,
    major: f32,
    semi_major: f32,
    rotation: UnitQuaternion<f32>,
) -> Aabb {
    let u = rotation * RIGHT.scale(major);
    let v = rotation * UP.scale(semi_major);

    let half_extents = (u.component_mul(&u) + v.component_mul(&v)).map(|x| x.sqrt());

    Aabb::from_half_extents(center, half_extents)
}
