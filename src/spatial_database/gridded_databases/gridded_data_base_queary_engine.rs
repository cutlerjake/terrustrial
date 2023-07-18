use nalgebra::Point3;
use ordered_float::OrderedFloat;

use crate::{
    geometry::{self, Geometry},
    spatial_database::coordinate_system::octant,
};

use super::GriddedDataBaseInterface;

/// Stores offsets for each octant of a geometry, allowing for fast queries of points in geometry
pub struct GriddedDataBaseQueryEngine<G>
where
    G: Geometry,
{
    octant_offsets: Vec<Vec<[isize; 3]>>,
    geometry: G,
}

impl<G> GriddedDataBaseQueryEngine<G>
where
    G: Geometry,
{
    /// Create a new GriddedDataBaseQueryEngine for a geometry
    /// # Arguments
    /// * `geometry` - The geometry to create the query engine for
    /// * `gdb` - The gridded database to use for the query engine
    pub fn new<T, GDB: GriddedDataBaseInterface<T>>(geometry: G, gdb: &GDB) -> Self {
        let mut geometry = geometry;
        let ref_point = gdb.ind_to_point(&[0, 0, 0]);
        geometry.translate_to(&ref_point);

        //get all offsets for geometry
        let mut offsets = gdb.offsets_from_ind_in_geometry(&[0, 0, 0], &geometry);

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
            let octant = octant(&p.into());
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
        GriddedDataBaseQueryEngine {
            octant_offsets: octants,
            geometry: geometry,
        }
    }

    /// Get the nearest points and values to a point in the geometry
    /// # Arguments
    /// * `point` - The point to get the nearest points and values for
    /// * `octant_size` - The number of points to get from each octant
    /// * `gdb` - The gridded database to use for the query engine
    ///     * must have same grid size and orientation as gdb used for construction of query engine
    pub fn nearest_points_and_values<T, GDB: GriddedDataBaseInterface<T>>(
        &self,
        point: &Point3<f32>,
        octant_size: usize,
        gdb: &GDB,
    ) -> (Vec<Point3<f32>>, Vec<T>) {
        //this only works if the grid ang geometry have similar orientation
        //TODO: convert to high ind relative to geometry rotation
        let point_ind = gdb.coord_to_high_ind(point).map(|x| x as usize);
        let mut points = Vec::<Point3<f32>>::new();
        let mut values = Vec::with_capacity(octant_size * 8);
        for offsets in self.octant_offsets.iter() {
            let mut oct_cnt = 0;
            for offset in offsets {
                let Some(ind) = gdb.offset_ind(point_ind, *offset) else {
                    continue;
                };
                if let Some(v) = gdb.data_at_ind(&ind) {
                    let p = gdb.ind_to_point(&ind.map(|x| x as isize));
                    points.push(p);
                    values.push(v);
                    oct_cnt += 1;
                    if oct_cnt == octant_size {
                        break;
                    }
                }
            }
        }

        (points, values)
    }
}
