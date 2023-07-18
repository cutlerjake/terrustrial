use nalgebra::Point3;
use ndarray::Array3;
use ordered_float::OrderedFloat;
use parry3d::bounding_volume::Aabb;

use crate::{
    geometry::Geometry,
    spatial_database::coordinate_system::{octant, CoordinateSystem, GridSpacing},
};

use super::{gridded_db::RawGriddedDataBase, GriddedDataBaseInterface, GriddedDataBaseQueryEngine};

/// Grid implementation for hgandling complete grids.
pub struct CompleteGriddedDataBase<T> {
    pub(crate) grid: RawGriddedDataBase<T>,
}

impl<T> CompleteGriddedDataBase<T> {
    pub fn new(
        grid: Array3<T>,
        block_size: GridSpacing,
        coordinate_system: CoordinateSystem,
    ) -> Self {
        let grid = RawGriddedDataBase::new(grid, block_size, coordinate_system);
        Self { grid }
    }
}

impl<T> GriddedDataBaseInterface<T> for CompleteGriddedDataBase<T>
where
    T: Copy,
{
    fn coord_to_high_ind(&self, point: &Point3<f32>) -> [isize; 3] {
        self.grid.coord_to_high_ind_with_negative(point)
    }

    fn offset_ind(&self, ind: [usize; 3], offset: [isize; 3]) -> Option<[usize; 3]> {
        self.grid.offset_ind(ind, offset)
    }
    fn data_at_ind(&self, ind: &[usize; 3]) -> Option<T> {
        self.grid.data_at_ind(ind)
    }

    fn ind_to_point(&self, ind: &[isize; 3]) -> Point3<f32> {
        self.grid.ind_to_point_with_negative(*ind)
    }

    fn offsets_from_ind_in_geometry<G>(&self, ind: &[usize; 3], geometry: &G) -> Vec<[isize; 3]>
    where
        G: Geometry,
    {
        self.grid.offsets_from_ind_in_geometry(ind, geometry)
    }

    fn init_query_engine_for_geometry<G: Geometry>(
        &self,
        geometry: G,
    ) -> GriddedDataBaseQueryEngine<G> {
        GriddedDataBaseQueryEngine::new(geometry, self)
    }

    fn inds_in_bounding_box(&self, bounding_box: &Aabb) -> Vec<[usize; 3]> {
        self.grid.inds_in_bounding_box(bounding_box)
    }

    fn data_and_points(&self) -> (Vec<T>, Vec<Point3<f32>>) {
        self.grid.data_and_points()
    }
}
