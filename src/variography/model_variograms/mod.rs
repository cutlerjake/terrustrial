use nalgebra::Vector3;

use simba::simd::f32x16;

pub mod spherical;

pub trait VariogramModel {
    fn variogram(&self, h: Vector3<f32>) -> f32;
    fn covariogram(&self, h: Vector3<f32>) -> f32;
    fn c_0(&self) -> f32;

    fn vectorized_variogram(&self, h: Vector3<f32x16>) -> f32x16;
    fn vectorized_covariogram(&self, h: Vector3<f32x16>) -> f32x16;
}
