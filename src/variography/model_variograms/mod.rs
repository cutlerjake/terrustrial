use nalgebra::{SimdRealField, SimdValue, Vector3};
use num_traits::Float;
use simba::simd::SimdPartialOrd;

pub mod spherical;

pub trait VariogramModel<T>
where
    T: SimdPartialOrd + SimdRealField,
    <T as SimdValue>::Element: SimdRealField + Float,
{
    // fn variogram(&self, h: Vector3<f32>) -> f32;
    // fn covariogram(&self, h: Vector3<f32>) -> f32;

    fn c_0(&self) -> <T as SimdValue>::Element;
    fn variogram(&self, h: Vector3<T>) -> T;
    fn covariogram(&self, h: Vector3<T>) -> T;
}
