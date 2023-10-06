use nalgebra::SimdRealField;
use nalgebra::UnitQuaternion;
use nalgebra::Vector3;
use num_traits::Float;
use num_traits::NumCast;

use crate::spatial_database::coordinate_system::CoordinateSystem;

use super::VariogramModel;
use simba::simd::SimdPartialOrd;
use simba::simd::SimdValue;

pub struct SphericalVariogram<T>
where
    T: SimdRealField + Copy,
{
    range: Vector3<T>,
    sill: T,
    nugget: T,
    rotation: UnitQuaternion<T>,
}

impl<T> SphericalVariogram<T>
where
    T: SimdRealField + Copy,
{
    pub fn new(range: Vector3<T>, sill: T, nugget: T, rotation: UnitQuaternion<T>) -> Self {
        Self {
            range,
            sill,
            nugget,
            rotation: rotation,
        }
    }

    // #[inline(always)]
    // pub fn variogram(&self, h: Vector3<f32>) -> f32 {
    //     let mut h = self.rotation.transform_vector(&h);

    //     h.component_div_assign(&self.range);
    //     let iso_h = h.norm();

    //     if iso_h == 0f32 {
    //         0f32
    //         //self.nugget
    //     } else if iso_h <= 1f32 {
    //         self.nugget + (self.sill - self.nugget) * (1.5 * iso_h - 0.5 * iso_h.powi(3))
    //     } else {
    //         self.sill
    //     }
    // }

    // #[inline(always)]
    // pub fn covariogram(&self, h: Vector3<f32>) -> f32 {
    //     self.sill - self.variogram(h)
    // }

    #[inline(always)]
    pub fn vectorized_variogram(&self, h: Vector3<T>) -> T
    where
        T: SimdPartialOrd + SimdRealField,
        <T as SimdValue>::Element: SimdRealField + Float,
    {
        let mut h = self.rotation.transform_vector(&h);

        // let rx = T::splat(self.range.x);
        // let ry = T::splat(self.range.y);
        // let rz = T::splat(self.range.z);
        // let simd_range = Vector3::new(rx, ry, rz);
        h.component_div_assign(&self.range);
        let iso_h = h.norm();

        let mask = !iso_h.simd_eq(T::splat(
            <<T as SimdValue>::Element as NumCast>::from(0.0).unwrap(),
        ));

        // let simd_nugget = T::splat(self.nugget);
        // let simd_sill = T::splat(self.sill);
        let simd_1_5 = T::splat(<<T as SimdValue>::Element as NumCast>::from(1.5).unwrap());
        let simd_0_5 = T::splat(<<T as SimdValue>::Element as NumCast>::from(0.5).unwrap());

        //create simd variance
        let mut simd_v = self.nugget
            + (self.sill - self.nugget) * (simd_1_5 * iso_h - simd_0_5 * iso_h * iso_h * iso_h);

        //set lanes of simd variance to nugget where lanes of iso_h == 0.0
        simd_v = simd_v.select(
            mask,
            T::splat(<<T as SimdValue>::Element as NumCast>::from(0.0).unwrap()),
        );

        let mask = iso_h.simd_le(T::splat(
            <<T as SimdValue>::Element as NumCast>::from(1.0).unwrap(),
        ));

        simd_v.select(mask, self.sill)
    }

    #[inline(always)]
    pub fn vectorized_covariogram(&self, h: Vector3<T>) -> T
    where
        T: SimdPartialOrd + SimdRealField,
        <T as SimdValue>::Element: SimdRealField + Float,
    {
        let simd_sill = self.sill;
        simd_sill - self.vectorized_variogram(h)
    }
}

impl<T> VariogramModel<T> for SphericalVariogram<T>
where
    T: SimdPartialOrd + SimdRealField + Copy,
    <T as SimdValue>::Element: SimdRealField + Float,
{
    // fn variogram(&self, h: Vector3<f32>) -> f32 {
    //     self.variogram(h)
    // }

    // fn covariogram(&self, h: Vector3<f32>) -> f32 {
    //     self.covariogram(h)
    // }

    // #[inline(always)]
    // fn c_0(&self) -> f32 {
    //     self.sill
    // }

    #[inline(always)]
    fn c_0(&self) -> <T as SimdValue>::Element {
        self.sill.extract(0)
    }

    #[inline(always)]
    fn variogram(&self, h: Vector3<T>) -> T {
        self.vectorized_variogram(h)
    }

    #[inline(always)]
    fn covariogram(&self, h: Vector3<T>) -> T {
        self.vectorized_covariogram(h)
    }
}

#[cfg(test)]
mod test {
    use nalgebra::Translation3;

    use super::*;
    #[test]
    fn spherical_vgram_var() {
        let sill = 1.0;
        let nugget = 0.1;
        let range = 300.0;
        let cs =
            CoordinateSystem::new(Translation3::new(0.0, 0.0, 0.0), UnitQuaternion::identity());

        let vgram = SphericalVariogram::new(
            Vector3::new(range, range, range),
            sill,
            nugget,
            UnitQuaternion::identity(),
        );

        let dists = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(46.1, 0.0, 0.0),
            Vector3::new(72.8, 0.0, 0.0),
            Vector3::new(68.01, 0.0, 0.0),
        ];

        println!("Variance");
        for d in dists.iter() {
            println!("dist: {} v: {}", d.norm(), vgram.variogram(*d));
        }

        println!("Covariance");
        for d in dists.iter() {
            println!("dist: {} v: {}", d.norm(), vgram.covariogram(*d));
        }
    }
}
