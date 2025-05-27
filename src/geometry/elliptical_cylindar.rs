use ultraviolet::{DRotor3, DVec3};

use crate::{FORWARD, RIGHT, UP};

use super::aabb::Aabb;

pub struct EllipticalCylindar {
    pub p1: DVec3,
    pub length: f64,
    pub a: f64,
    pub b: f64,
    pub orientation: DRotor3,
}

impl EllipticalCylindar {
    pub fn new(p1: DVec3, length: f64, a: f64, b: f64, orientation: DRotor3) -> Self {
        Self {
            p1,
            length,
            a,
            b,
            orientation,
        }
    }

    pub fn loose_aabb(&self) -> Aabb {
        let cylinder_axis = self.orientation * FORWARD * self.length;
        let aabb1 = bounding_box_of_oriented_ellipse(self.p1, self.a, self.b, self.orientation);
        let aabb2 = bounding_box_of_oriented_ellipse(
            self.p1 + cylinder_axis,
            self.a,
            self.b,
            self.orientation,
        );

        aabb1.merged(&aabb2)
    }

    pub fn contains_point(&self, point: DVec3) -> bool {
        let cylinder_axis = self.orientation * FORWARD * self.length;
        let length_sq = cylinder_axis.dot(cylinder_axis);

        let p1 = self.p1;

        let v = point - p1;
        let dot = v.dot(cylinder_axis);

        if dot <= 0.0 || dot >= length_sq {
            return false;
        }

        let mut cylinder_aligned_point = v;
        self.orientation
            .reversed()
            .rotate_vec(&mut cylinder_aligned_point);

        if cylinder_aligned_point.x * cylinder_aligned_point.x / (self.a * self.a)
            + cylinder_aligned_point.z * cylinder_aligned_point.z / (self.b + self.b)
            > 1f64
        {
            return false;
        }

        true
    }
}

// Function to compute the bounding box of an oriented ellipse in 3D space
pub fn bounding_box_of_oriented_ellipse(
    center: DVec3,
    major: f64,
    semi_major: f64,
    rotation: DRotor3,
) -> Aabb {
    let u = rotation * RIGHT * major;
    let v = rotation * UP * semi_major;

    let half_extents = (u * u + v * v).map(|x| x.sqrt());

    Aabb::new(center, half_extents)
}
