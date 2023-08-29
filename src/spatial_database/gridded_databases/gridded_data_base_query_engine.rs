use nalgebra::Point3;
use ordered_float::OrderedFloat;

use crate::{
    geometry::Geometry,
    spatial_database::{coordinate_system::octant, SpatialQueryable},
};

use super::GriddedDataBaseInterface;

/// Stores offsets for each octant of a geometry, allowing for fast queries of points in geometry
pub struct GriddedDataBaseOctantQueryEngine<'a, G, GDB, T>
where
    G: Geometry,
    GDB: GriddedDataBaseInterface<T>,
{
    pub(crate) octant_offsets: Vec<Vec<[isize; 3]>>,
    pub(crate) geometry: G,
    pub(crate) max_octant_size: usize,
    pub(crate) db: &'a GDB,
    pub(crate) phantom: std::marker::PhantomData<T>,
}

impl<'a, G, GDB, T> GriddedDataBaseOctantQueryEngine<'a, G, GDB, T>
where
    G: Geometry,
    GDB: GriddedDataBaseInterface<T>,
{
    /// Create a new GriddedDataBaseQueryEngine for a geometry
    /// # Arguments
    /// * `geometry` - The geometry to create the query engine for
    /// * `gdb` - The gridded database to use for the query engine
    pub fn new(geometry: G, gdb: &'a GDB, max_octant_size: usize) -> Self {
        let mut geometry = geometry;
        let ref_point = gdb.ind_to_point(&[0, 0, 0]);
        geometry.translate_to(&ref_point);

        //get all offsets for geometry
        let offsets = gdb.offsets_from_ind_in_geometry(&[0, 0, 0], &geometry);

        //splits offsets into octant groups
        let mut octants = vec![Vec::new(); 8];
        for offset in offsets {
            //get point in world space
            let mut p = gdb.ind_to_point(&offset);
            //transform to local space of geometry
            p = geometry
                .coordinate_system()
                .world_to_local
                .transform_point(&p);
            let octant = octant(&p);
            octants[octant as usize - 1].push(offset);
        }

        //sort offsets by iso_distance to ref point
        octants.iter_mut().for_each(|offset| {
            offset.sort_by_key(|offset| {
                //can simply use offset since reference point is origin
                let p = gdb.ind_to_point(offset);
                let d = geometry.iso_distance(&p);
                OrderedFloat(d)
            });
        });

        //create query engine
        GriddedDataBaseOctantQueryEngine {
            octant_offsets: octants,
            geometry,
            max_octant_size,
            db: gdb,
            phantom: std::marker::PhantomData,
        }
    }

    /// Get the nearest points and values to a point in the geometry
    /// # Arguments
    /// * `point` - The point to get the nearest points and values for
    /// * `octant_size` - The number of points to get from each octant
    /// * `gdb` - The gridded database to use for the query engine
    ///     * must have same grid size and orientation as gdb used for construction of query engine
    pub fn nearest_points_and_values(&self, point: &Point3<f32>) -> (Vec<Point3<f32>>, Vec<T>) {
        let _ = self.geometry;
        //this only works if the grid ang geometry have similar orientation
        //TODO: convert to high ind relative to geometry rotation
        let point_ind = self.db.coord_to_high_ind(point).map(|x| x as usize);
        let mut points = Vec::<Point3<f32>>::new();
        let mut values = Vec::with_capacity(self.max_octant_size * 8);
        for offsets in self.octant_offsets.iter() {
            let mut oct_cnt = 0;
            for offset in offsets {
                let Some(ind) = self.db.offset_ind(point_ind, *offset) else {
                    continue;
                };
                if let Some(v) = self.db.data_at_ind(&ind) {
                    let p = self.db.ind_to_point(&ind.map(|x| x as isize));
                    points.push(p);
                    values.push(v);
                    oct_cnt += 1;
                    if oct_cnt == self.max_octant_size {
                        break;
                    }
                }
            }
        }

        (points, values)
    }

    /// Get the nearest points and values to a point in the geometry
    /// # Arguments
    /// * `point` - The point to get the nearest points and values for
    /// * `octant_size` - The number of points to get from each octant
    /// * `gdb` - The gridded database to use for the query engine
    ///     * must have same grid size and orientation as gdb used for construction of query engine
    pub fn nearest_points_and_values_masked<F>(
        &self,
        point: &Point3<f32>,
        mask: F,
    ) -> (Vec<Point3<f32>>, Vec<T>)
    where
        F: Fn([usize; 3]) -> bool,
    {
        let _ = self.geometry;
        //this only works if the grid ang geometry have similar orientation
        //TODO: convert to high ind relative to geometry rotation
        let point_ind = self.db.coord_to_high_ind(point).map(|x| x as usize);
        let mut points = Vec::<Point3<f32>>::new();
        let mut values = Vec::with_capacity(self.max_octant_size * 8);
        for offsets in self.octant_offsets.iter() {
            let mut oct_cnt = 0;
            for offset in offsets {
                let Some(ind) = self.db.offset_ind(point_ind, *offset) else {
                    continue;
                };

                if !mask(ind) {
                    continue;
                }
                if let Some(v) = self.db.data_at_ind(&ind) {
                    let p = self.db.ind_to_point(&ind.map(|x| x as isize));
                    points.push(p);
                    values.push(v);
                    oct_cnt += 1;
                    if oct_cnt == self.max_octant_size {
                        break;
                    }
                }
            }
        }

        (points, values)
    }

    /// Get the nearest points and values to a point in the geometry
    /// # Arguments
    /// * `point` - The point to get the nearest points and values for
    /// * `octant_size` - The number of points to get from each octant
    /// * `gdb` - The gridded database to use for the query engine
    ///     * must have same grid size and orientation as gdb used for construction of query engine
    pub fn nearest_inds<F>(&self, point: &Point3<f32>) -> Vec<[usize; 3]> {
        let _ = self.geometry;
        //this only works if the grid ang geometry have similar orientation
        //TODO: convert to high ind relative to geometry rotation
        let point_ind = self.db.coord_to_high_ind(point).map(|x| x as usize);
        let mut inds = Vec::new();
        for offsets in self.octant_offsets.iter() {
            let mut oct_cnt = 0;
            for offset in offsets {
                let Some(ind) = self.db.offset_ind(point_ind, *offset) else {
                    continue;
                };

                if let Some(_) = self.db.data_at_ind(&ind) {
                    inds.push(ind);
                    oct_cnt += 1;
                    if oct_cnt == self.max_octant_size {
                        break;
                    }
                }
            }
        }

        inds
    }

    /// Get the nearest points and values to a point in the geometry
    /// # Arguments
    /// * `point` - The point to get the nearest points and values for
    /// * `octant_size` - The number of points to get from each octant
    /// * `gdb` - The gridded database to use for the query engine
    ///     * must have same grid size and orientation as gdb used for construction of query engine
    pub fn nearest_inds_and_points_masked<F>(
        &self,
        point: &Point3<f32>,
        mask: F,
    ) -> (Vec<[usize; 3]>, Vec<Point3<f32>>)
    where
        F: Fn([usize; 3]) -> bool,
    {
        let _ = self.geometry;
        //this only works if the grid ang geometry have similar orientation
        //TODO: convert to high ind relative to geometry rotation
        let point_ind = self.db.coord_to_high_ind(point).map(|x| x as usize);
        let mut inds = Vec::new();
        let mut points = Vec::<Point3<f32>>::new();
        for offsets in self.octant_offsets.iter() {
            let mut oct_cnt = 0;
            for offset in offsets {
                let Some(ind) = self.db.offset_ind(point_ind, *offset) else {
                    continue;
                };
                //print!("ind: {:?}, ", ind);

                if !mask(ind) {
                    continue;
                }
                let p = self.db.ind_to_point(&ind.map(|x| x as isize));
                inds.push(ind);
                points.push(p);
                oct_cnt += 1;
                if oct_cnt == self.max_octant_size {
                    break;
                }
            }
        }

        (inds, points)
    }

    pub fn retain_offsets<F: Fn(&[isize; 3]) -> bool>(&mut self, filter: F) {
        for offsets in self.octant_offsets.iter_mut() {
            offsets.retain(|offset| filter(offset));
        }
    }
}

impl<'a, G, GDB, T> SpatialQueryable<T, G> for GriddedDataBaseOctantQueryEngine<'a, G, GDB, T>
where
    G: Geometry,
    GDB: GriddedDataBaseInterface<T>,
{
    fn query(&self, point: &Point3<f32>) -> (Vec<T>, Vec<Point3<f32>>) {
        let (points, values) = self.nearest_points_and_values(point);
        (values, points)
    }

    fn geometry(&self) -> &G {
        &self.geometry
    }
}

// impl<'a, GDB, T, G> SpatialQuery<GDB, T, G> for GriddedDataBaseOctantQueryEngine<'a, G, GDB, T>
// where
//     GDB: GriddedDataBaseInterface<T>,
//     G: Geometry,
//     T: Copy,
// {
//     type PARAMETERS = usize;

//     fn new(db: &GDB, geometry: G, parameters: Self::PARAMETERS) -> Self {
//         Self::new(geometry, db, parameters)
//     }

//     fn query(&self, point: &Point3<f32>) -> (Vec<T>, Vec<Point3<f32>>) {
//         let (points, values) = self.nearest_points_and_values(point);
//         (values, points)
//     }
// }
