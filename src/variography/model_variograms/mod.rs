use nalgebra::{SimdValue, Vector3};

pub mod composite;
pub mod iso_exponential;
pub mod iso_fitter;
pub mod iso_gaussian;
pub mod iso_nugget;
pub mod iso_spherical;
pub mod nugget;
pub mod spherical;

pub trait VariogramModel<T>
where
    T: SimdValue<Element = f32> + Copy,
{
    // fn variogram(&self, h: Vector3<f32>) -> f32;
    // fn covariogram(&self, h: Vector3<f32>) -> f32;

    fn c_0(&self) -> <T as SimdValue>::Element;
    fn variogram(&self, h: Vector3<T>) -> T;
    fn covariogram(&self, h: Vector3<T>) -> T;
}
