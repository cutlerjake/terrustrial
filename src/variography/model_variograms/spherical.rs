use nalgebra::UnitQuaternion;
use nalgebra::Vector3;

use crate::spatial_database::coordinate_system::CoordinateSystem;

use super::VariogramModel;
use simba::simd::f32x16;
use simba::simd::SimdPartialOrd;
use simba::simd::SimdValue;

pub struct SphericalVariogram {
    range: Vector3<f32>,
    sill: f32,
    nugget: f32,
    rotation: UnitQuaternion<f32>,
    vec_rotation: UnitQuaternion<f32x16>,
}

impl SphericalVariogram {
    pub fn new(
        range: Vector3<f32>,
        sill: f32,
        nugget: f32,
        coordinate_system: CoordinateSystem,
    ) -> Self {
        let vec_cs = coordinate_system.vectorized_global_to_local_isomety();
        Self {
            range,
            sill,
            nugget,
            rotation: coordinate_system.rotation,
            vec_rotation: vec_cs.rotation,
        }
    }

    #[inline(always)]
    pub fn variogram(&self, h: Vector3<f32>) -> f32 {
        let mut h = self.rotation.transform_vector(&h);
        // let h = self.coordinate_system.global_to_local(&h.into());
        // let iso_h = f32::sqrt(
        //     (h.x / self.range.x).powi(2)
        //         + (h.y / self.range.y).powi(2)
        //         + (h.z / self.range.z).powi(2),
        // );

        h.component_div_assign(&self.range);
        let iso_h = h.norm();

        if iso_h == 0f32 {
            //0f32
            self.nugget
        } else if iso_h <= 1f32 {
            self.nugget + (self.sill - self.nugget) * (1.5 * iso_h - 0.5 * iso_h.powi(3))
        } else {
            self.sill
        }
    }

    #[inline(always)]
    pub fn covariogram(&self, h: Vector3<f32>) -> f32 {
        self.sill - self.variogram(h)
    }

    #[inline(always)]
    pub fn vectorized_variogram(&self, h: Vector3<f32x16>) -> f32x16 {
        let mut h = self.vec_rotation.transform_vector(&h);

        let rx = f32x16::splat(self.range.x);
        let ry = f32x16::splat(self.range.y);
        let rz = f32x16::splat(self.range.z);
        let simd_range = Vector3::new(rx, ry, rz);
        h.component_div_assign(&simd_range);
        let iso_h = h.norm();

        let mask = !iso_h.simd_eq(f32x16::splat(0.0));

        let simd_nugget = f32x16::splat(self.nugget);
        let simd_sill = f32x16::splat(self.sill);
        let simd_1_5 = f32x16::splat(1.5);
        let simd_0_5 = f32x16::splat(0.5);

        //create simd variance
        let mut simd_v = simd_nugget
            + (simd_sill - simd_nugget) * (simd_1_5 * iso_h - simd_0_5 * iso_h * iso_h * iso_h);

        //set lanes of simd variance to nugget where lanes of iso_h == 0.0
        simd_v = simd_v.select(mask, f32x16::splat(self.nugget));

        let mask = iso_h.simd_le(f32x16::splat(1.0));

        simd_v.select(mask, simd_sill)

        // if iso_h == 0f32 {
        //     0f32
        // } else if iso_h <= 1f32 {
        //     self.nugget + (self.sill - self.nugget) * (1.5 * iso_h - 0.5 * iso_h.powi(3))
        // } else {
        //     self.sill
        // }
    }

    #[inline(always)]
    pub fn vectorized_covariogram(&self, h: Vector3<f32x16>) -> f32x16 {
        let simd_sill = f32x16::splat(self.sill);
        simd_sill - self.vectorized_variogram(h)
    }
}

impl VariogramModel for SphericalVariogram {
    fn variogram(&self, h: Vector3<f32>) -> f32 {
        self.variogram(h)
    }

    fn covariogram(&self, h: Vector3<f32>) -> f32 {
        self.covariogram(h)
    }

    fn c_0(&self) -> f32 {
        self.sill
    }

    fn vectorized_variogram(&self, h: Vector3<f32x16>) -> f32x16 {
        self.vectorized_variogram(h)
    }
    fn vectorized_covariogram(&self, h: Vector3<f32x16>) -> f32x16 {
        self.vectorized_covariogram(h)
    }
}
