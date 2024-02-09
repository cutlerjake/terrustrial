use nalgebra::{SimdRealField, SimdValue, Vector3};
use simba::simd::WideF32x8;

use super::VariogramModel;

#[derive(Clone, Copy, Debug, Default)]
pub struct Nugget<T>
where
    T: SimdValue<Element = f32> + Copy,
{
    pub nugget: T,
}

impl<T> Nugget<T>
where
    T: SimdValue<Element = f32> + Copy,
{
    pub fn new(nugget: T) -> Self {
        Self { nugget }
    }
}

impl Nugget<f32> {
    pub fn to_f32x8(&self) -> Nugget<WideF32x8> {
        Nugget {
            nugget: WideF32x8::splat(self.nugget),
        }
    }
}

impl<T> VariogramModel<T> for Nugget<T>
where
    T: SimdValue<Element = f32> + SimdRealField + Copy,
{
    #[inline(always)]
    fn c_0(&self) -> <T as SimdValue>::Element {
        self.nugget.extract(0)
    }

    #[inline(always)]
    fn variogram(&self, h: Vector3<T>) -> T {
        let iso_h = h.norm_squared();

        let mask = !iso_h.simd_eq(T::splat(0.0));

        //create simd variance
        let simd_v = self.nugget;

        //set lanes of simd variance to 0 where lanes of iso_h == 0.0
        simd_v.select(mask, T::splat(0.0))
    }

    #[inline(always)]
    fn covariogram(&self, h: Vector3<T>) -> T {
        self.nugget - self.variogram(h)
    }
}
