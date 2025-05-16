use nalgebra::{Point3, SimdComplexField, SimdPartialOrd, SimdValue, UnitQuaternion, Vector3};
use parry3d::bounding_volume::BoundingVolume;
use simba::simd::{WideBoolF32x8, WideF32x8};

use crate::FORWARD;

use super::elliptical_cylindar::bounding_box_of_oriented_ellipse;

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
        let cylinder_axis = self.orientation * FORWARD.into_inner();

        let far_dist = self.offset + self.length;
        let far_a = if far_dist >= self.a_dist_threshold {
            self.a
        } else {
            far_dist * self.a_tol.tan()
        };

        let far_b = if far_dist >= self.b_dist_threshold {
            self.b
        } else {
            far_dist * self.b_tol.tan()
        };

        // AABB of the ellipse at the far end of the cylinder
        let far_aabb = bounding_box_of_oriented_ellipse(
            self.base + far_dist * cylinder_axis,
            far_a,
            far_b,
            self.orientation,
        );

        let near_a = if self.offset >= self.a_dist_threshold {
            self.a
        } else {
            self.offset * self.a_tol.tan()
        };

        let near_b = if self.offset >= self.b_dist_threshold {
            self.b
        } else {
            self.offset * self.b_tol.tan()
        };

        let max_ab = near_a.max(near_b);
        let d = self.offset;

        let shift = d - f32::sqrt(d * d - max_ab * max_ab);
        let pt = self.base + (self.offset - shift) * cylinder_axis;
        let near_aabb = bounding_box_of_oriented_ellipse(pt, far_a, far_b, self.orientation);

        far_aabb.merged(&near_aabb)
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

    pub fn contains_point_vectorized(&self, points: &Point3<WideF32x8>) -> WideBoolF32x8 {
        let base = Point3::splat(self.base);
        let dist = (points - base).norm();

        let dist_mask = dist.simd_lt(WideF32x8::splat(self.offset))
            | dist.simd_gt(WideF32x8::splat(self.offset + self.length));

        let cylinder_axis =
            Vector3::splat(self.orientation * FORWARD.into_inner() * (self.length + self.offset));

        let v = points - base;
        let dot = v.dot(&cylinder_axis);

        let dot_mask = dot.simd_le(WideF32x8::splat(0.0));

        let wide_orientation = UnitQuaternion::splat(self.orientation);

        let cylinder_aligned_point = wide_orientation.inverse_transform_point(&v.into());

        let axial_dist = ((points - base).dot(&cylinder_axis)).simd_sqrt();

        let a_mask = axial_dist.simd_ge(WideF32x8::splat(self.a_dist_threshold));
        let a = WideF32x8::splat(self.a)
            .select(a_mask, axial_dist * WideF32x8::splat(self.a_tol.tan()));

        let b_mask = axial_dist.simd_ge(WideF32x8::splat(self.b_dist_threshold));
        let b = WideF32x8::splat(self.b)
            .select(b_mask, axial_dist * WideF32x8::splat(self.b_tol.tan()));

        let x = cylinder_aligned_point.x / a;
        let z = cylinder_aligned_point.z / b;

        (x * x + z * z).simd_le(WideF32x8::splat(1.0)) & !dist_mask & !dot_mask
    }
}

pub fn collect_points(points: &[Point3<f32>]) -> Vec<Point3<WideF32x8>> {
    let mut result = Vec::new();
    let chunk_size = 8;

    for chunk in points.chunks(chunk_size) {
        let mut x = [f32::MAX; 8];
        let mut y = [f32::MAX; 8];
        let mut z = [f32::MAX; 8];

        for (i, point) in chunk.iter().enumerate() {
            if i < chunk_size {
                x[i] = point.x;
                y[i] = point.y;
                z[i] = point.z;
            }
        }

        // Create WideF32x8 from the collected data
        let wide_x = WideF32x8::from(x);
        let wide_y = WideF32x8::from(y);
        let wide_z = WideF32x8::from(z);

        result.push(Point3::new(wide_x, wide_y, wide_z));
    }

    result
}
