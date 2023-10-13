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
    T: SimdValue<Element = f32> + Copy,
{
    range: Vector3<T>,
    sill: T,
    nugget: T,
    rotation: UnitQuaternion<T>,
}

impl<T> SphericalVariogram<T>
where
    T: SimdValue<Element = f32> + Copy,
{
    pub fn new(range: Vector3<T>, sill: T, nugget: T, rotation: UnitQuaternion<T>) -> Self {
        Self {
            range,
            sill,
            nugget,
            rotation: rotation,
        }
    }
}

impl<T> VariogramModel<T> for SphericalVariogram<T>
where
    T: SimdValue<Element = f32> + SimdRealField + Copy,
{
    #[inline(always)]
    fn c_0(&self) -> <T as SimdValue>::Element {
        self.sill.extract(0)
    }

    #[inline(always)]
    fn variogram(&self, h: Vector3<T>) -> T {
        let mut h = self.rotation.transform_vector(&h);

        h.component_div_assign(&self.range);
        let iso_h = h.norm();

        let mask = !iso_h.simd_eq(T::splat(0.0));

        let simd_1_5 = T::splat(1.5);
        let simd_0_5 = T::splat(0.5);

        //create simd variance
        let mut simd_v = self.nugget
            + (self.sill - self.nugget) * (simd_1_5 * iso_h - simd_0_5 * iso_h * iso_h * iso_h);

        //set lanes of simd variance to 0 where lanes of iso_h == 0.0
        simd_v = simd_v.select(mask, T::splat(0.0));

        let mask = iso_h.simd_le(T::splat(1.0));

        simd_v.select(mask, self.sill)
    }

    #[inline(always)]
    fn covariogram(&self, h: Vector3<T>) -> T {
        self.sill - self.variogram(h)
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
        let cs = CoordinateSystem::new(
            Translation3::new(0.0f32, 0.0, 0.0),
            UnitQuaternion::identity(),
        );

        let vgram = SphericalVariogram::<f32>::new(
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
