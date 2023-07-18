use std::fmt::Debug;

use nalgebra::Point3;
use parry3d::bounding_volume::Aabb;

use crate::geometry::Geometry;

use self::{coordinate_system::CoordinateSystem, gridded_databases::GriddedDataBaseInterface};

pub mod coordinate_system;
pub mod gridded_databases;

pub trait SpatialDataBase<T> {
    type INDEX: Debug;
    fn inds_in_bounding_box(&self, bounding_box: &Aabb) -> Vec<Self::INDEX>;
    fn point_at_ind(&self, inds: &Self::INDEX) -> Point3<f32>;
    fn data_at_ind(&self, ind: &Self::INDEX) -> Option<T>;
    fn data_and_points(&self) -> (Vec<T>, Vec<Point3<f32>>);
}

impl<T, G> SpatialDataBase<T> for G
where
    G: GriddedDataBaseInterface<T>,
{
    type INDEX = [usize; 3];

    fn inds_in_bounding_box(&self, bounding_box: &Aabb) -> Vec<Self::INDEX> {
        <Self as GriddedDataBaseInterface<T>>::inds_in_bounding_box(self, bounding_box)
    }

    fn point_at_ind(&self, inds: &Self::INDEX) -> Point3<f32> {
        <Self as GriddedDataBaseInterface<T>>::ind_to_point(self, &inds.map(|i| i as isize))
    }

    fn data_at_ind(&self, ind: &Self::INDEX) -> Option<T> {
        <Self as GriddedDataBaseInterface<T>>::data_at_ind(self, ind)
    }

    fn data_and_points(&self) -> (Vec<T>, Vec<Point3<f32>>) {
        <Self as GriddedDataBaseInterface<T>>::data_and_points(self)
    }
}

pub trait QueryEngineInterface {
    fn nearest_points_and_values<T>(
        &self,
        point: &Point3<f32>,
        num_points: usize,
    ) -> (Vec<Point3<f32>>, Vec<&T>);
}
