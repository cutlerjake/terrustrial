use nalgebra::SimdValue;
use ultraviolet::{f64x4, DRotor3, DVec3, DVec3x4};

pub mod composite;
pub mod composite_iso;
pub mod iso_exponential;
pub mod iso_gaussian;
pub mod iso_nugget;
pub mod iso_spherical;
pub mod nugget;
pub mod spherical;

pub trait IsoVariogramModel<T>
where
    T: SimdValue + Copy,
{
    fn c_0(&self) -> <T as SimdValue>::Element;
    fn variogram(&self, h: T) -> T;
    fn covariogram(&self, h: T) -> T;
}

pub trait VariogramModel: Clone + Send {
    // fn variogram(&self, h: Vector3<f32>) -> f32;
    // fn covariogram(&self, h: Vector3<f32>) -> f32;

    fn c_0(&self) -> f64;
    fn variogram(&self, h: DVec3) -> f64;
    fn covariogram(&self, h: DVec3) -> f64;

    fn variogram_simd(&self, h: DVec3x4) -> f64x4;
    fn covariogram_simd(&self, h: DVec3x4) -> f64x4;

    fn set_orientation(&mut self, orientation: DRotor3);
}
