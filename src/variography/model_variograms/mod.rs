use nalgebra::{SimdValue, UnitQuaternion, Vector3};

pub mod composite;
pub mod iso_exponential;
pub mod iso_gaussian;
pub mod iso_nugget;
pub mod iso_spherical;
pub mod nugget;
// pub mod optimizer;
pub mod composite_iso;
pub mod spherical;

pub trait IsoVariogramModel<T>
where
    T: SimdValue + Copy,
{
    fn c_0(&self) -> <T as SimdValue>::Element;
    fn variogram(&self, h: T) -> T;
    fn covariogram(&self, h: T) -> T;
}

pub trait VariogramModel<T>: Clone + Send
where
    T: SimdValue<Element = f32> + Copy,
{
    // fn variogram(&self, h: Vector3<f32>) -> f32;
    // fn covariogram(&self, h: Vector3<f32>) -> f32;

    fn c_0(&self) -> <T as SimdValue>::Element;
    fn variogram(&self, h: Vector3<T>) -> T;
    fn covariogram(&self, h: Vector3<T>) -> T;
    fn set_orientation(&mut self, orientation: UnitQuaternion<T>);
}
