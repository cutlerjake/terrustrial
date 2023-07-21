use nalgebra::Point3;
use parry3d::bounding_volume::Aabb;

use crate::geometry::Geometry;

use self::gridded_data_base_query_engine::GriddedDataBaseOctantQueryEngine;

use super::coordinate_system::{CoordinateSystem, GridSpacing};

pub mod complete_grid;
pub mod gridded_data_base_query_engine;
pub mod gridded_db;
pub mod incomplete_grid;

/// Gridded database interface.
pub trait GriddedDataBaseInterface<T> {
    fn coord_to_high_ind(&self, point: &Point3<f32>) -> [isize; 3];
    fn offset_ind(&self, ind: [usize; 3], offset: [isize; 3]) -> Option<[usize; 3]>;
    fn data_at_ind(&self, ind: &[usize; 3]) -> Option<T>;
    fn ind_to_point(&self, ind: &[isize; 3]) -> Point3<f32>;
    fn offsets_from_ind_in_geometry<G>(&self, ind: &[usize; 3], geometry: &G) -> Vec<[isize; 3]>
    where
        G: Geometry;

    //fn signed_ind_to_point(&self, ind: &[isize; 3]) -> Point3<f32>;

    // fn init_query_engine_for_geometry<G: Geometry>(
    //     &self,
    //     geometry: G,
    // ) -> GriddedDataBaseOctantQueryEngine<G>;

    fn inds_in_bounding_box(&self, bounding_box: &Aabb) -> Vec<[usize; 3]>;

    fn data_and_points(&self) -> (Vec<T>, Vec<Point3<f32>>);
    fn data_and_inds(&self) -> (Vec<T>, Vec<[usize; 3]>);
    fn set_data_at_ind(&mut self, ind: &[usize; 3], data: T);
    fn shape(&self) -> [usize; 3];
    fn grid_spacing(&self) -> GridSpacing;
    fn coordinate_system(&self) -> CoordinateSystem;
}
