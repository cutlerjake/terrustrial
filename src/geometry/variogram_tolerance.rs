use nalgebra::{Isometry3, Point3, Translation, UnitQuaternion};
use parry3d::bounding_volume::Aabb;

use crate::{FORWARD, RIGHT, UP};

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
    pub base: Point3<f32>,
    pub offset: f32,
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
            base: p1,
            offset: 0.0,
        }
    }

    pub fn offset_along_axis(&mut self, offset: f32) {
        // self.p1 += self.axis.scale(offset);
        self.p1 += self.orientation * FORWARD.into_inner() * offset;
        self.offset += offset;
    }

    pub fn set_base(&mut self, base: Point3<f32>) {
        self.p1 = base;
        self.base = base;
        self.offset = 0.0;
    }

    pub fn loose_aabb(&self) -> parry3d::bounding_volume::Aabb {
        let delta_a = (self.offset * self.offset - self.a * self.a).max(0.0);
        let delta_b = (self.offset * self.offset - self.b * self.b).max(0.0);

        let delta = if delta_a < delta_b {
            delta_a.sqrt()
        } else {
            delta_b.sqrt()
        };

        // Create min and max points for the AABB
        let min = FORWARD.scale(delta) + UP.scale(-self.b) + RIGHT.scale(-self.a);
        let max = FORWARD.scale(self.length + self.offset) + UP.scale(self.b) + RIGHT.scale(self.a);

        let mut aabb = Aabb::new(min.into(), max.into());

        let isometry = Isometry3::from_parts(Translation::from(self.base), self.orientation);
        aabb = aabb.transform_by(&isometry);

        aabb
    }

    pub fn contains_point(&self, point: Point3<f32>) -> bool {
        let dist = (point - self.base).norm();

        if dist < self.offset || dist > self.offset + self.length {
            return false;
        }

        let cylinder_axis = self.orientation * FORWARD.into_inner() * (self.length + self.offset);

        let v = point - self.base;
        let dot = v.dot(&cylinder_axis);

        if dot <= 0.0 {
            return false;
        }

        let cylinder_aligned_point = self.orientation.inverse_transform_point(&v.into());

        let axial_dist = ((point - self.base).dot(&cylinder_axis)).sqrt();

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

        if cylinder_aligned_point.x * cylinder_aligned_point.x / (a * a)
            + cylinder_aligned_point.z * cylinder_aligned_point.z / (b * b)
            > 1f32
        {
            return false;
        }

        true
    }
}
