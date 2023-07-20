use std::{collections::HashMap, error, mem::MaybeUninit, str::FromStr};

use itertools::izip;
use nalgebra::Point3;
use ndarray::Array3;
use parry3d::bounding_volume::Aabb;

use crate::{
    geometry::Geometry,
    spatial_database::coordinate_system::{CoordinateSystem, GridSpacing},
};

use super::GriddedDataBaseInterface;

/// A raw gridded database
/// # Members
/// * `grid` - Array storing data
/// * `block_size` - Size of each block
/// * `coordinate_system` - Coordinate system of the grid (Location and orientation of the grid)
///
pub struct RawGriddedDataBase<T> {
    pub grid: Array3<T>,
    pub grid_spacing: GridSpacing,
    pub coordinate_system: CoordinateSystem,
}

impl<T> RawGriddedDataBase<T> {
    /// Create a new raw gridded database
    /// # Arguments
    /// * `grid` - Array storing data
    /// * `block_size` - Size of each block
    /// * `coordinate_system` - Coordinate system of the grid (Location and orientation of the grid)
    pub fn new(
        grid: Array3<T>,
        grid_spacing: GridSpacing,
        coordinate_system: CoordinateSystem,
    ) -> Self {
        Self {
            grid,
            grid_spacing,
            coordinate_system,
        }
    }

    /// Coordinates of point in local coordinate system
    fn transform_point_to_grid(&self, point: &Point3<f32>) -> Point3<f32> {
        self.coordinate_system.global_to_local(point)
    }

    /// divide point by block size
    fn normalize_point_to_grid_spacing(&self, point: &Point3<f32>) -> Point3<f32> {
        let block_size = self.grid_spacing;
        Point3::new(
            point.x / block_size.x,
            point.y / block_size.y,
            point.z / block_size.z,
        )
    }

    /// Convert point to grid indices (Does not work for points outside of grid)
    /// try to avoid using this function
    pub fn coord_to_ind(&self, point: &Point3<f32>) -> Option<[usize; 3]> {
        let point = self.transform_point_to_grid(point);
        let point = self.normalize_point_to_grid_spacing(&point);
        //check coords for positiveness
        if point.x < 0.0 || point.y < 0.0 || point.z < 0.0 {
            return None;
        }

        //check coords within alpha of integer inds
        let alpha = 0.01;
        if (point.x - point.x.round()).abs() > alpha
            || (point.y - point.y.round()).abs() > alpha
            || (point.z - point.z.round()).abs() > alpha
        {
            return None;
        }

        //convert coords to usize
        let x = point.x.round() as usize;
        let y = point.y.round() as usize;
        let z = point.z.round() as usize;

        Some([x, y, z])
    }

    /// Convert a point to a grid index defined by the ceiling of the normalized local coordinates
    pub fn grid_aligned_coord_to_high_ind(&self, point: &Point3<f32>) -> Option<[usize; 3]> {
        //normalize coords to block size
        let point = self.normalize_point_to_grid_spacing(&point);

        //set all point coordinates below 0 to 0
        let point = Point3::new(point.x.max(0.0), point.y.max(0.0), point.z.max(0.0));

        Some([
            point.x.ceil() as usize,
            point.y.ceil() as usize,
            point.z.ceil() as usize,
        ])
    }

    /// Convert a point to a grid index defined by the ceiling of the normalized local coordinates (may be negative)
    pub fn grid_aligned_coord_to_high_ind_with_negative(&self, point: &Point3<f32>) -> [isize; 3] {
        //normalize coords to block size
        let point = self.normalize_point_to_grid_spacing(&point);

        [
            point.x.ceil() as isize,
            point.y.ceil() as isize,
            point.z.ceil() as isize,
        ]
    }

    //convert a point to a grid index defined by the ceiling of the normalized local coordinates
    pub fn coord_to_high_ind(&self, point: &Point3<f32>) -> Option<[usize; 3]> {
        let point = self.transform_point_to_grid(point);

        self.grid_aligned_coord_to_high_ind(&point)
    }

    /// Convert a point to a grid index defined by the ceiling of the normalized local coordinates (may be negative)
    pub fn coord_to_high_ind_with_negative(&self, point: &Point3<f32>) -> [isize; 3] {
        let point = self.transform_point_to_grid(point);

        self.grid_aligned_coord_to_high_ind_with_negative(&point)
    }

    /// Convert a point to a grid index defined by the floor of the normalized local coordinates
    pub fn grid_aligned_coord_to_low_ind(&self, point: &Point3<f32>) -> Option<[usize; 3]> {
        //normalize coords to block size
        let point = self.normalize_point_to_grid_spacing(&point);

        //set all point coordinates larger then grid dimension to grid dimension
        let point = Point3::new(
            point.x.min(self.grid.shape()[0] as f32),
            point.y.min(self.grid.shape()[1] as f32),
            point.z.min(self.grid.shape()[2] as f32),
        );

        Some([
            point.x.floor() as usize,
            point.y.floor() as usize,
            point.z.floor() as usize,
        ])
    }

    /// Convert a point to a grid index defined by the floor of the normalized local coordinates (may be negative)
    pub fn grid_aligned_coord_to_low_ind_with_negative(&self, point: &Point3<f32>) -> [isize; 3] {
        //normalize coords to block size
        let point = self.normalize_point_to_grid_spacing(&point);

        [
            point.x.floor() as isize,
            point.y.floor() as isize,
            point.z.floor() as isize,
        ]
    }

    /// Convert a point to a grid index defined by the floor of the normalized local coordinates
    pub fn coord_to_low_ind(&self, point: &Point3<f32>) -> Option<[usize; 3]> {
        let point = self.transform_point_to_grid(point);

        self.grid_aligned_coord_to_low_ind(&point)
    }

    /// Point at index grid in world coordinates
    pub fn ind_to_point(&self, ind: [usize; 3]) -> Point3<f32> {
        let mut x = ind[0] as f32;
        let mut y = ind[1] as f32;
        let mut z = ind[2] as f32;

        //scale coords to grid
        x = x * self.grid_spacing.x;
        y = y * self.grid_spacing.y;
        z = z * self.grid_spacing.z;

        //create point
        let point = Point3::new(x, y, z);

        //tranform point
        self.coordinate_system.local_to_global(&point)
    }

    /// Point at index grid in world coordinates (may be negative)
    pub fn ind_to_point_with_negative(&self, ind: [isize; 3]) -> Point3<f32> {
        let mut x = ind[0] as f32;
        let mut y = ind[1] as f32;
        let mut z = ind[2] as f32;

        //scale coords to grid
        x = x * self.grid_spacing.x;
        y = y * self.grid_spacing.y;
        z = z * self.grid_spacing.z;

        //create point
        let point = Point3::new(x, y, z);

        //tranform point
        self.coordinate_system.local_to_global(&point)
    }

    /// Grid data at point
    /// try not to use
    pub fn get_at_point(&self, point: &Point3<f32>) -> Option<&T> {
        //rotate coords to grid
        let ind = self.coord_to_ind(point)?;

        self.grid.get(ind)
    }

    /// Coordinates of all points in grid
    pub fn get_coords(&self) -> Array3<[f32; 3]> {
        let mut coords = Array3::from_shape_fn(self.grid.dim(), |_| [0.0; 3]);

        for (ind, _) in self.grid.indexed_iter() {
            let point = self.ind_to_point([ind.0, ind.1, ind.2]);

            //set coords
            coords[ind] = [point.x, point.y, point.z];
        }
        coords
    }

    /// Grid indices of points within bounding box
    /// Bounding box considered grid aligned in world coordinates
    fn _inds_in_bounding_box(&self, bounding_box: &Aabb) -> Vec<[usize; 3]> {
        //align bounding box to grid
        //let aligned_bounding_box = bounding_box.global_to_local(&self.coordinate_system);
        let aligned_bounding_box =
            bounding_box.transform_by(&self.coordinate_system.world_to_local);
        //compute center of bounding box

        //get inds of bounding box
        let Some(min_ind) = self.grid_aligned_coord_to_low_ind(&aligned_bounding_box.mins) else {
            return vec![];
        };

        let Some(mut max_ind) = self.grid_aligned_coord_to_high_ind(&aligned_bounding_box.maxs)
        else {
            return vec![];
        };
        //ensure min_ind and max_ind are valid
        if min_ind
            .iter()
            .zip(self.grid.shape().iter())
            .any(|(a, b)| a >= b)
        {
            return vec![];
        }

        max_ind
            .iter_mut()
            .zip(self.grid.shape().iter())
            .for_each(|(a, b)| {
                if *a >= *b {
                    *a = *b - 1;
                }
            });

        // println!("min_ind: {:?}", min_ind);
        // println!("max_ind: {:?}", max_ind);
        let size = max_ind
            .iter()
            .zip(min_ind.iter())
            .map(|(max, min)| max - min + 1)
            .product();
        let mut inds = Vec::with_capacity(size);
        for x in min_ind[0]..=max_ind[0] {
            for y in min_ind[1]..=max_ind[1] {
                for z in min_ind[2]..=max_ind[2] {
                    inds.push([x, y, z]);
                }
            }
        }

        inds
    }

    /// Offsets from ind within geometry
    pub fn _offsets_from_ind_in_geometry<G>(&self, ind: [usize; 3], geometry: &G) -> Vec<[isize; 3]>
    where
        G: Geometry,
    {
        //align bounding box to grid
        //let aligned_bounding_box = bounding_box.global_to_local(&self.coordinate_system);
        let bounding_box = geometry.bounding_box();
        let aligned_bounding_box =
            bounding_box.transform_by(&self.coordinate_system.world_to_local);
        //compute center of bounding box

        //get inds of bounding box
        let min_ind = self.grid_aligned_coord_to_low_ind_with_negative(&aligned_bounding_box.mins);
        let max_ind = self.grid_aligned_coord_to_high_ind_with_negative(&aligned_bounding_box.maxs);

        //convert reference ind
        let ref_ind = ind.map(|x| x as isize);

        //compute size of vec required
        let size = max_ind
            .iter()
            .zip(min_ind.iter())
            .map(|(max, min)| max - min + 1)
            .product::<isize>() as usize;

        //create vec
        let mut inds = Vec::with_capacity(size);

        //compute offset from reference ind and add to vec
        for x in min_ind[0]..=max_ind[0] {
            for y in min_ind[1]..=max_ind[1] {
                for z in min_ind[2]..=max_ind[2] {
                    if geometry.contains(&self.ind_to_point_with_negative([x, y, z])) {
                        inds.push([x - ref_ind[0], y - ref_ind[1], z - ref_ind[2]]);
                    }
                }
            }
        }

        inds
    }
}

impl<T> GriddedDataBaseInterface<T> for RawGriddedDataBase<T>
where
    T: Copy,
{
    fn coord_to_high_ind(&self, point: &Point3<f32>) -> [isize; 3] {
        self.coord_to_high_ind_with_negative(point)
    }

    fn offset_ind(&self, ind: [usize; 3], offset: [isize; 3]) -> Option<[usize; 3]> {
        let mut new_ind = ind;

        for i in 0..3 {
            if offset[i] < 0 {
                if ind[i] < (-offset[i]) as usize {
                    return None;
                }
                new_ind[i] = ind[i] - (-offset[i]) as usize;
            } else {
                if ind[i] + offset[i] as usize >= self.grid.shape()[i] {
                    return None;
                }
                new_ind[i] = ind[i] + offset[i] as usize;
            }
        }

        Some(new_ind)
    }
    fn data_at_ind(&self, ind: &[usize; 3]) -> Option<T> {
        self.grid.get(*ind).copied()
    }

    fn ind_to_point(&self, ind: &[isize; 3]) -> Point3<f32> {
        self.ind_to_point_with_negative(*ind)
    }

    fn offsets_from_ind_in_geometry<G>(&self, ind: &[usize; 3], geometry: &G) -> Vec<[isize; 3]>
    where
        G: Geometry,
    {
        self._offsets_from_ind_in_geometry(*ind, geometry)
    }

    // fn init_query_engine_for_geometry<G: Geometry>(
    //     &self,
    //     geometry: G,
    // ) -> GriddedDataBaseOctantQueryEngine<G> {
    //     GriddedDataBaseOctantQueryEngine::new(geometry, self)
    // }

    fn inds_in_bounding_box(&self, bounding_box: &Aabb) -> Vec<[usize; 3]> {
        self._inds_in_bounding_box(bounding_box)
    }

    fn data_and_points(&self) -> (Vec<T>, Vec<Point3<f32>>) {
        self.grid.indexed_iter().fold(
            (Vec::new(), Vec::new()),
            |(mut data, mut points), (ind, val)| {
                points.push(self.ind_to_point([ind.0, ind.1, ind.2]));
                data.push(*val);
                (data, points)
            },
        )
    }

    fn data_and_inds(&self) -> (Vec<T>, Vec<[usize; 3]>) {
        self.grid
            .indexed_iter()
            .map(|(ind, val)| (val, [ind.0, ind.1, ind.2]))
            .unzip()
    }

    fn set_data_at_ind(&mut self, ind: &[usize; 3], data: T) {
        self.grid[[ind[0], ind[1], ind[2]]] = data;
    }

    fn shape(&self) -> [usize; 3] {
        let shape = self.grid.shape();
        [shape[0], shape[1], shape[2]]
    }

    fn grid_spacing(&self) -> GridSpacing {
        self.grid_spacing
    }

    fn coordinate_system(&self) -> CoordinateSystem {
        self.coordinate_system
    }
}
