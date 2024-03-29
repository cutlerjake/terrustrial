use std::{collections::HashMap, error, str::FromStr};

use itertools::izip;
use nalgebra::Point3;
use ndarray::Array3;
use parry3d::bounding_volume::Aabb;

use crate::{
    geometry::Geometry,
    spatial_database::coordinate_system::{CoordinateSystem, GridSpacing},
};

use super::{gridded_db::RawGriddedDataBase, GriddedDataBaseInterface};

/// Grid implementation for handling incomplete grids.
pub struct InCompleteGriddedDataBase<T> {
    pub raw_grid: RawGriddedDataBase<Option<T>>,
}

impl<T> InCompleteGriddedDataBase<T> {
    pub fn new(
        grid: Array3<Option<T>>,
        block_size: GridSpacing,
        coordinate_system: CoordinateSystem,
    ) -> Self {
        let grid = RawGriddedDataBase::new(grid, block_size, coordinate_system);
        Self { raw_grid: grid }
    }
}

impl<T> InCompleteGriddedDataBase<T>
where
    T: Copy + FromStr,
    <T as FromStr>::Err: std::error::Error + 'static,
{
    pub fn from_csv_index(
        csv_path: &str,
        i_col: &str,
        j_col: &str,
        k_col: &str,
        value_col: &str,
        grid_spacing: GridSpacing,
        coordinate_system: CoordinateSystem,
    ) -> Result<Self, Box<dyn error::Error>> {
        //storage for data
        let mut i_vec = Vec::new();
        let mut j_vec = Vec::new();
        let mut k_vec = Vec::new();
        let mut value_vec = Vec::new();

        //read data from csv
        let mut rdr = csv::Reader::from_path(csv_path)?;
        for result in rdr.deserialize() {
            let record: HashMap<String, String> = result?;

            let i = record[i_col].parse::<usize>()?;
            let j = record[j_col].parse::<usize>()?;
            let k = record[k_col].parse::<usize>()?;
            let value = record[value_col].parse::<T>()?;

            i_vec.push(i);
            j_vec.push(j);
            k_vec.push(k);
            value_vec.push(value);
        }

        //compute grid size
        let i_max = i_vec.iter().max().unwrap();
        let j_max = j_vec.iter().max().unwrap();
        let k_max = k_vec.iter().max().unwrap();

        let i_min = i_vec.iter().min().unwrap();
        let j_min = j_vec.iter().min().unwrap();
        let k_min = k_vec.iter().min().unwrap();

        let i_size = i_max - i_min + 1;
        let j_size = j_max - j_min + 1;
        let k_size = k_max - k_min + 1;

        //create grid
        let mut grid = Array3::from_shape_simple_fn((i_size, j_size, k_size), || None);

        //fill grid
        for (i, j, k, value) in izip!(i_vec.iter(), j_vec.iter(), k_vec.iter(), value_vec.iter()) {
            grid[[i - i_min, j - j_min, k - k_min]] = Some(*value);
        }

        Ok(Self::new(grid, grid_spacing, coordinate_system))
    }
}

impl<T> GriddedDataBaseInterface<T> for InCompleteGriddedDataBase<T>
where
    T: Copy,
{
    fn coord_to_high_ind(&self, point: &Point3<f32>) -> [isize; 3] {
        self.raw_grid.coord_to_high_ind_with_negative(point)
    }

    fn offset_ind(&self, ind: [usize; 3], offset: [isize; 3]) -> Option<[usize; 3]> {
        self.raw_grid.offset_ind(ind, offset)
    }
    fn data_at_ind(&self, ind: &[usize; 3]) -> Option<T> {
        self.raw_grid.data_at_ind(ind).unwrap_or(None)
    }

    fn ind_to_point(&self, ind: &[isize; 3]) -> Point3<f32> {
        self.raw_grid.ind_to_point_with_negative(*ind)
    }

    fn offsets_from_ind_in_geometry<G>(&self, ind: &[usize; 3], geometry: &G) -> Vec<[isize; 3]>
    where
        G: Geometry,
    {
        self.raw_grid._offsets_from_ind_in_geometry(*ind, geometry)
    }

    // fn init_query_engine_for_geometry<G: Geometry>(
    //     &self,
    //     geometry: G,
    // ) -> GriddedDataBaseOctantQueryEngine<G> {
    //     GriddedDataBaseOctantQueryEngine::new(geometry, self)
    // }

    fn inds_in_bounding_box(&self, aabb: &Aabb) -> Vec<[usize; 3]> {
        self.raw_grid.inds_in_bounding_box(aabb)
    }

    fn data_and_points(&self) -> (Vec<T>, Vec<Point3<f32>>) {
        let (data, points) = self.raw_grid.data_and_points();

        data.iter()
            .zip(points)
            .filter(|(val, _)| val.is_some())
            .map(|(val, point)| (val.unwrap(), point))
            .unzip()
    }

    fn data_and_inds(&self) -> (Vec<T>, Vec<[usize; 3]>) {
        self.raw_grid
            .grid
            .indexed_iter()
            .filter(|(_, val)| val.is_some())
            .map(|(ind, val)| (val.unwrap(), [ind.0, ind.1, ind.2]))
            .unzip()
    }

    fn set_data_at_ind(&mut self, ind: &[usize; 3], data: T) {
        self.raw_grid.set_data_at_ind(ind, Some(data));
    }

    fn shape(&self) -> [usize; 3] {
        let shape = self.raw_grid.grid.shape();
        [shape[0], shape[1], shape[2]]
    }

    fn grid_spacing(&self) -> GridSpacing {
        self.raw_grid.grid_spacing
    }

    fn coordinate_system(&self) -> CoordinateSystem {
        self.raw_grid.coordinate_system
    }
}
