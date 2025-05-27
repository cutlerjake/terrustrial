use ultraviolet::{DMat3x4, DVec3, DVec3x4};
use wide::{f64x4, CmpGe, CmpGt, CmpLe, CmpLt};

use crate::{spatial_database::coordinate_system::NewCoordinateSystem, FORWARD};

use super::{aabb::Aabb, elliptical_cylindar::bounding_box_of_oriented_ellipse};

pub struct VariogramTolerance {
    pub p1: DVec3,
    pub length: f64,
    pub a: f64,
    pub a_tol: f64,
    pub a_dist_threshold: f64,
    pub b: f64,
    pub b_tol: f64,
    pub b_dist_threshold: f64,
    pub cs: NewCoordinateSystem,
    pub offset: f64,
}

impl VariogramTolerance {
    pub fn new(
        p1: DVec3,
        length: f64,
        a: f64,
        a_tol: f64,
        b: f64,
        b_tol: f64,
        cs: NewCoordinateSystem,
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
            cs,
            offset: 0.0,
        }
    }

    pub fn offset_along_axis(&mut self, offset: f64) {
        // self.p1 += self.axis.scale(offset);
        self.p1 += self.cs.into_local().rotation * FORWARD * offset;
        self.offset += offset;
    }

    pub fn set_base(&mut self, base: DVec3) {
        self.p1 = base;
        self.cs.set_origin(base);
        self.offset = 0.0;
    }

    pub fn loose_aabb(&self) -> Aabb {
        let cylinder_axis = self.cs.into_local().rotation * FORWARD;

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
            self.cs.into_global().translation + far_dist * cylinder_axis,
            far_a,
            far_b,
            self.cs.into_global().rotation,
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

        let shift = d - f64::sqrt(d * d - max_ab * max_ab);
        let pt = self.cs.into_global().translation + (self.offset - shift) * cylinder_axis;
        let near_aabb =
            bounding_box_of_oriented_ellipse(pt, far_a, far_b, self.cs.into_global().rotation);

        far_aabb.merged(&near_aabb)
    }

    pub fn contains_point(&self, point: DVec3) -> bool {
        let dist = (point - self.cs.into_global().translation).mag();

        if dist < self.offset || dist > self.offset + self.length {
            return false;
        }

        let cylinder_axis = self.cs.into_global().rotation * FORWARD * (self.length + self.offset);

        let v = point - self.cs.into_global().translation;
        let dot = v.dot(cylinder_axis);

        if dot <= 0.0 {
            return false;
        }

        let mut cylinder_aligned_point = v;
        self.cs
            .into_local()
            .rotation
            .rotate_vec(&mut cylinder_aligned_point);

        let axial_dist = ((point - self.cs.into_global().translation).dot(cylinder_axis)).sqrt();

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
            > 1f64
        {
            return false;
        }

        true
    }

    pub fn contains_point_vectorized(&self, points: DVec3x4) -> [bool; 4] {
        // let base = Point3::splat(self.base);
        let base = DVec3x4::splat(self.cs.into_global().translation);
        let dist = (points - base).mag();

        let dist_mask = dist.cmp_lt(self.offset) | dist.cmp_gt(self.offset + self.length);

        let cylinder_axis =
            DVec3x4::splat(self.cs.into_global().rotation * FORWARD * (self.length + self.offset));

        let v = points - base;
        let dot = v.dot(cylinder_axis);

        let dot_mask = dot.cmp_le(0.0);

        let rot_mat = self.cs.into_local().rotation.into_matrix();
        let wide_rot_mat = DMat3x4::new(
            DVec3x4::splat(rot_mat.cols[0]),
            DVec3x4::splat(rot_mat.cols[1]),
            DVec3x4::splat(rot_mat.cols[2]),
        );

        let cylinder_aligned_point = wide_rot_mat * v;
        let axial_dist = ((points - base).dot(cylinder_axis)).sqrt();

        let a_mask = axial_dist.cmp_ge(self.a_dist_threshold);
        let a = (f64x4::splat(self.a) & a_mask)
            | (axial_dist * f64x4::splat(self.a_tol.tan()) & !a_mask);

        let b_mask = axial_dist.cmp_ge(self.b_dist_threshold);
        let b = (f64x4::splat(self.b) & b_mask)
            | (axial_dist * f64x4::splat(self.b_tol.tan()) & !b_mask);

        let x = cylinder_aligned_point.x / a;
        let z = cylinder_aligned_point.z / b;

        ((x * x + z * z).cmp_le(1.0) & !dist_mask & !dot_mask)
            .as_array_ref()
            .map(|x| x != 0.0)
    }
}

pub fn collect_points(points: &[DVec3]) -> Vec<DVec3x4> {
    let mut result = Vec::new();
    const CHUNK_SIZE: usize = 4;

    for chunk in points.chunks(CHUNK_SIZE) {
        let mut x = [f64::MAX; CHUNK_SIZE];
        let mut y = [f64::MAX; CHUNK_SIZE];
        let mut z = [f64::MAX; CHUNK_SIZE];

        for (i, point) in chunk.iter().enumerate() {
            if i < CHUNK_SIZE {
                x[i] = point.x;
                y[i] = point.y;
                z[i] = point.z;
            }
        }

        // Create WideF64x4 from the collected data
        let wide_x = f64x4::from(x);
        let wide_y = f64x4::from(y);
        let wide_z = f64x4::from(z);

        result.push(DVec3x4::new(wide_x, wide_y, wide_z));
    }

    result
}
