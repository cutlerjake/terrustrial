use nalgebra::Point3;
use parry3d::bounding_volume::Aabb;
use simba::simd::f32x16;
use simba::simd::SimdValue;

use crate::spatial_database::coordinate_system::CoordinateSystem;

pub mod ellipsoid;
pub mod template;
pub mod tolerance;

pub trait Geometry {
    fn bounding_box(&self) -> Aabb;
    fn contains(&self, point: &Point3<f32>) -> bool;
    fn translate_to(&mut self, translation: &Point3<f32>);

    fn vectorized_contains(&self, points: &Point3<f32x16>) -> <f32x16 as SimdValue>::SimdBool;
    fn iso_distance(&self, point: &Point3<f32>) -> f32;
    fn coordinate_system(&self) -> &CoordinateSystem;
}
