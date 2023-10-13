use std::fmt::Debug;

use nalgebra::Point3;
use parry3d::bounding_volume::Aabb;

use crate::geometry::ellipsoid::Ellipsoid;

use self::gridded_databases::GriddedDataBaseInterface;

pub mod coordinate_system;
pub mod gridded_databases;
pub mod normalized;
pub mod qbvh;
pub mod rtree_point_set;
pub mod zero_mean;

pub trait PointProvider {
    fn points(&self) -> &[Point3<f32>];
}

pub trait SpatialQueryable<T, G> {
    fn query(&self, point: &Point3<f32>) -> (Vec<T>, Vec<Point3<f32>>);
    fn geometry(&self) -> &G;
}

pub trait ConditioningProvider<G, T, P> {
    type Shape;
    fn query(
        &self,
        point: &Point3<f32>,
        ellipsoid: &G,
        params: &P,
    ) -> (Vec<usize>, Vec<T>, Vec<Self::Shape>);

    fn points(&self) -> &[Point3<f32>];
    fn data(&self) -> &[T];
    fn data_mut(&mut self) -> &mut [T];
}

pub trait SpatialDataBase<T> {
    type INDEX: Debug;
    fn inds_in_bounding_box(&self, bounding_box: &Aabb) -> Vec<Self::INDEX>;
    fn point_at_ind(&self, inds: &Self::INDEX) -> Point3<f32>;
    fn data_at_ind(&self, ind: &Self::INDEX) -> Option<T>;
    fn data_and_points(&self) -> (Vec<T>, Vec<Point3<f32>>);
    fn data_and_inds(&self) -> (Vec<T>, Vec<Self::INDEX>);
    fn set_data_at_ind(&mut self, ind: &Self::INDEX, data: T);
}

macro_rules! impl_spatial_database_for_grid {
    ($( ($impl_type:ty, $data_type:ty) ),*) => {
        $(
            impl SpatialDataBase<$data_type> for $impl_type {
                type INDEX = [usize; 3];

                fn inds_in_bounding_box(&self, bounding_box: &Aabb) -> Vec<Self::INDEX> {
                    <Self as GriddedDataBaseInterface<$data_type>>::inds_in_bounding_box(self, bounding_box)
                }

                fn point_at_ind(&self, inds: &Self::INDEX) -> Point3<f32> {
                    <Self as GriddedDataBaseInterface<$data_type>>::ind_to_point(self, &inds.map(|i| i as isize))
                }

                fn data_at_ind(&self, ind: &Self::INDEX) -> Option<$data_type> {
                    <Self as GriddedDataBaseInterface<$data_type>>::data_at_ind(self, ind)
                }

                fn data_and_points(&self) -> (Vec<$data_type>, Vec<Point3<f32>>) {
                    <Self as GriddedDataBaseInterface<$data_type>>::data_and_points(self)
                }

                fn data_and_inds(&self) -> (Vec<$data_type>, Vec<Self::INDEX>) {
                    <Self as GriddedDataBaseInterface<$data_type>>::data_and_inds(self)
                }

                fn set_data_at_ind(&mut self, ind: &Self::INDEX, data: $data_type) {
                    <Self as GriddedDataBaseInterface<$data_type>>::set_data_at_ind(self, ind, data)
                }
            }
        )*
    };
}

// impl_spatial_database_for_grid!(
//     (
//         gridded_databases::incomplete_grid::InCompleteGriddedDataBase<f32, f32>,
//         f32
//     ),
//     (
//         gridded_databases::incomplete_grid::InCompleteGriddedDataBase<f64, f32>,
//         f64
//     ),
//     (
//         gridded_databases::complete_grid::CompleteGriddedDataBase<f32, f32>,
//         f32
//     ),
//     (
//         gridded_databases::complete_grid::CompleteGriddedDataBase<f64, f32>,
//         f64
//     )
// );
