use nalgebra::{Isometry3, Point3, RealField, SimdRealField};
use num_traits::Float;
use parry3d::bounding_volume::Aabb;
use simba::{
    scalar::{SubsetOf, SupersetOf},
    simd::SimdValue,
};

use crate::spatial_database::coordinate_system::CoordinateSystem;

pub mod ellipsoid;
pub mod template;
pub mod tolerance;

pub trait Geometry {
    //translate geometry to new origin
    fn translate_to(&mut self, translation: &Point3<f32>);
    fn bounding_box(&self) -> Aabb;
    fn contains(&self, point: &Point3<f32>) -> bool;

    fn vectorized_contains<T>(&self, points: &Point3<T>) -> <T as SimdValue>::SimdBool
    where
        T: SimdValue<Element = f32> + Clone + SimdRealField;

    fn vectorized_iso_distance<T>(&self, point: &Point3<T>) -> T
    where
        T: SimdValue<Element = f32> + Clone + SimdRealField;
    fn coordinate_system<T>(&self) -> &CoordinateSystem<f32>;
}
