use nalgebra::Quaternion;
use nalgebra::SimdRealField;
use nalgebra::UnitQuaternion;
use nalgebra::Vector3;
use simba::simd::WideF32x8;

use super::VariogramModel;
use simba::simd::SimdValue;

#[derive(Clone, Copy, Debug)]
pub struct SphericalVariogram<T>
where
    T: SimdValue<Element = f32> + Copy,
{
    pub range: Vector3<T>,
    pub sill: T,
    pub rotation: UnitQuaternion<T>,
}

impl<T> Default for SphericalVariogram<T>
where
    T: SimdValue<Element = f32> + Copy + SimdRealField,
{
    fn default() -> Self {
        let quat = UnitQuaternion::from_quaternion(Quaternion::from_real(T::splat(1.0)));
        Self {
            range: Vector3::new(T::splat(1.0), T::splat(1.0), T::splat(1.0)),
            sill: T::splat(1.0),
            rotation: quat,
        }
    }
}

impl<T> SphericalVariogram<T>
where
    T: SimdValue<Element = f32> + Copy,
    T: SimdRealField,
    T::Element: SimdRealField,
{
    pub fn new(range: Vector3<T>, sill: T, rotation: UnitQuaternion<T>) -> Self {
        Self {
            range,
            sill,
            rotation: rotation.inverse(),
        }
    }
}

impl SphericalVariogram<f32> {
    pub fn to_f32x8(&self) -> SphericalVariogram<WideF32x8> {
        SphericalVariogram {
            range: Vector3::splat(self.range),
            sill: WideF32x8::splat(self.sill),
            rotation: UnitQuaternion::splat(self.rotation),
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
        let mut simd_v = self.sill * (simd_1_5 * iso_h - simd_0_5 * iso_h * iso_h * iso_h);

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

    use super::*;
    #[test]
    fn spherical_vgram_var() {
        let sill = 1.0;
        let range = 300.0;

        let vgram = SphericalVariogram::<f32>::new(
            Vector3::new(range, range, range),
            sill,
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
