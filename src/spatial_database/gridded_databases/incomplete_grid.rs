use nalgebra::Point3;
use ndarray::Array3;
use ordered_float::OrderedFloat;
use parry3d::bounding_volume::Aabb;

use crate::{
    geometry::Geometry,
    spatial_database::coordinate_system::{octant, CoordinateSystem, GridSpacing},
};

use super::{
    gridded_db::RawGriddedDataBase, GriddedDataBaseInterface, GriddedDataBaseOctantQueryEngine,
};

/// Grid implementation for handling incomplete grids.
pub struct InCompleteGriddedDataBase<T> {
    pub(crate) grid: RawGriddedDataBase<Option<T>>,
}

impl<T> InCompleteGriddedDataBase<T> {
    pub fn new(
        grid: Array3<Option<T>>,
        block_size: GridSpacing,
        coordinate_system: CoordinateSystem,
    ) -> Self {
        let grid = RawGriddedDataBase::new(grid, block_size, coordinate_system);
        Self { grid }
    }
}

impl<T> GriddedDataBaseInterface<T> for InCompleteGriddedDataBase<T>
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
        self.grid.data_at_ind(ind).unwrap_or(None)
    }

    fn ind_to_point(&self, ind: &[isize; 3]) -> Point3<f32> {
        self.grid.ind_to_point_with_negative(*ind)
    }

    fn offsets_from_ind_in_geometry<G>(&self, ind: &[usize; 3], geometry: &G) -> Vec<[isize; 3]>
    where
        G: Geometry,
    {
        self.grid._offsets_from_ind_in_geometry(*ind, geometry)
    }

    // fn init_query_engine_for_geometry<G: Geometry>(
    //     &self,
    //     geometry: G,
    // ) -> GriddedDataBaseOctantQueryEngine<G> {
    //     GriddedDataBaseOctantQueryEngine::new(geometry, self)
    // }

    fn inds_in_bounding_box(&self, aabb: &Aabb) -> Vec<[usize; 3]> {
        self.grid.inds_in_bounding_box(aabb)
    }

    fn data_and_points(&self) -> (Vec<T>, Vec<Point3<f32>>) {
        let (data, points) = self.grid.data_and_points();

        data.iter()
            .zip(points)
            .filter(|(val, point)| val.is_some())
            .map(|(val, point)| (val.unwrap(), point))
            .unzip()
    }
}
