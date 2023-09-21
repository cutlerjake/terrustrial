use core::panic;

use indicatif::ParallelProgressIterator;
use itertools::{iproduct, Itertools};
use nalgebra::Point3;
use ndarray::Array3;
use rand::{rngs::StdRng, seq::SliceRandom, Rng, SeedableRng};
use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};

use crate::{
    geometry::Geometry,
    spatial_database::{
        gridded_databases::{
            gridded_data_base_query_engine::GriddedDataBaseOctantQueryEngine,
            GriddedDataBaseInterface,
        },
        SpatialQueryable,
    },
    variography::model_variograms::VariogramModel,
};

use super::lu::LUSystem;

pub struct GSGSParameters {
    pub max_octant_cond_data: usize,
    pub max_octant_sim_data: usize,
    pub group_size: [usize; 3],
}

pub struct GSGS<S, V, G>
where
    S: SpatialQueryable<f32, G>,
{
    conditioning_data: S,
    variogram_model: V,
    gsgs_parameters: GSGSParameters,
    phantom: std::marker::PhantomData<G>,
}

impl<S, V, G> GSGS<S, V, G>
where
    S: SpatialQueryable<f32, G> + Sync,
    V: VariogramModel + Sync,
    G: Geometry + Sync + Clone,
{
    /// Create a new simple kriging estimator with the given parameters
    /// # Arguments
    /// * `conditioning_data` - The data to condition the kriging system on (Must be normalized)
    /// * `variogram_model` - The variogram model to use
    /// * `search_ellipsoid` - The search ellipsoid to use
    /// * `gsgs_parameters` - The gsgs parameters to use
    /// # Returns
    /// A new simple kriging estimator
    pub fn new(conditioning_data: S, variogram_model: V, gsgs_parameters: GSGSParameters) -> Self {
        Self {
            conditioning_data,
            variogram_model,
            gsgs_parameters,
            phantom: std::marker::PhantomData,
        }
    }

    #[inline(always)]
    fn create_path<GDB>(
        &self,
        grid: &GDB,
        step: [usize; 3],
        rng: &mut StdRng,
    ) -> (Vec<Vec<[usize; 3]>>, Array3<usize>)
    where
        GDB: GriddedDataBaseInterface<f32> + std::marker::Sync,
    {
        // Array to store simulation order
        let mut simulation_order = Array3::from_elem(grid.shape(), 0);

        //create path over groups
        let i_steps = (grid.shape()[0] as f32 / step[0] as f32).ceil() as usize;
        let j_steps = (grid.shape()[1] as f32 / step[1] as f32).ceil() as usize;
        let k_steps = (grid.shape()[2] as f32 / step[2] as f32).ceil() as usize;
        let grid_shape = grid.shape();

        let mut path = Vec::new();
        for (group_i, group_j, group_k) in iproduct!(0..i_steps, 0..j_steps, 0..k_steps) {
            //get group bounds
            let i_min = group_i * step[0];
            let i_max = ((group_i + 1) * step[0]).min(grid_shape[0]);
            let j_min = group_j * step[1];
            let j_max = ((group_j + 1) * step[1]).min(grid_shape[1]);
            let k_min = group_k * step[2];
            let k_max = ((group_k + 1) * step[2]).min(grid_shape[2]);

            //get all points in group
            let mut group_points = Vec::new();
            for i in i_min..i_max {
                for j in j_min..j_max {
                    for k in k_min..k_max {
                        group_points.push([i, j, k]);
                    }
                }
            }

            //shuffle group points
            group_points.shuffle(rng);
            path.push(group_points);
        }

        //shuffle path order
        path.shuffle(rng);

        //set simulation order
        for (ind, val) in path.iter().enumerate() {
            for val in val.iter() {
                simulation_order[*val] = ind;
            }
        }

        (path, simulation_order)
    }

    /// Perform simple kriging at all kriging points
    pub fn simulate_grid<GDB>(&self, grid: &mut GDB, rng: &mut StdRng)
    where
        GDB: GriddedDataBaseInterface<f32> + std::marker::Sync,
    {
        // create query engine for simulation grid
        let geometry = self.conditioning_data.geometry().clone();
        let mut sim_qe = GriddedDataBaseOctantQueryEngine::new(
            geometry.clone(),
            grid,
            self.gsgs_parameters.max_octant_sim_data,
        );

        //filter offsets to avoid unnecessary searching
        let step = self.gsgs_parameters.group_size;
        let internal_offsets = iproduct!(0..step[0], 0..step[1], 0..step[2])
            .map(|(i, j, k)| [i, j, k])
            .collect_vec();
        let retainer = |offset: &[isize; 3]| {
            if offset.iter().any(|v| *v < 0) {
                return true;
            }
            let offset = offset.map(|v| v as usize);
            !internal_offsets.iter().any(|v| v == &offset)
        };
        sim_qe.retain_offsets(retainer);

        //create path over groups
        let (path, simulation_order) = self.create_path(grid, self.gsgs_parameters.group_size, rng);

        // Note: We do not know the values so we can't populate the grid
        // But at each location in the grid we now all the points that will be previously simulated and the locations of the conditioning data
        // thus, we can solve for the weights in parrallel, then populate the grid sequentially
        let sequential_data = path
            .par_iter()
            .progress()
            .map_with(
                (
                    LUSystem::new(
                        self.gsgs_parameters.group_size.iter().product(),
                        (self.gsgs_parameters.max_octant_cond_data
                            + self.gsgs_parameters.max_octant_sim_data)
                            * 8,
                    ),
                    StdRng::from_entropy(),
                ),
                |(local_system, local_rng), inds| {
                    //get point at center of group
                    let sim_points = inds
                        .iter()
                        .map(|ind| grid.ind_to_point(&ind.map(|i| i as isize)))
                        .collect::<Vec<_>>();
                    let point = sim_points.iter().fold(Point3::origin(), |mut acc, p| {
                        acc.coords += p.coords;
                        acc
                    }) / inds.len() as f32;

                    //get nearest conditioning  points and values
                    let (cond_values, mut cond_points) = self.conditioning_data.query(&point);

                    // get nearest simulation points
                    let (sim_cond_inds, sim_cond_points) =
                        sim_qe.nearest_inds_and_points_masked(&point, |neighbor_ind| {
                            //true
                            simulation_order[neighbor_ind] < simulation_order[inds[0]]
                        });

                    //println!("sim_cond_inds: {}", sim_cond_inds.len());

                    //append simulation points to conditioning points
                    cond_points.extend(sim_cond_points.iter());

                    // Cholesky error when simulating a point present in conditioning data
                    // this is a quick but not great fix (randomly shift point by very small value)
                    // TODO: remove duplicate point(s) from sim_points and populate with conditioning value
                    // let cond_points = cond_points
                    //     .iter_mut()
                    //     .map(|point| {
                    //         point.coords.x += local_rng.gen::<f32>() * 0.0001;
                    //         point.coords.y += local_rng.gen::<f32>() * 0.0001;
                    //         point.coords.z += local_rng.gen::<f32>() * 0.0001;
                    //         *point
                    //     })
                    //     .collect_vec();

                    let mini_system = local_system.create_mini_system(
                        &cond_points,
                        &sim_points,
                        &self.variogram_model,
                    );

                    (inds, cond_values, sim_cond_inds, mini_system)
                },
            )
            .collect::<Vec<_>>();

        sequential_data.into_iter().for_each(
            |(inds, cond_values, sim_cond_inds, mut mini_system)| {
                //get simulation values
                let sim_values = sim_cond_inds
                    .iter()
                    .map(|ind| grid.data_at_ind(ind).expect("No value at ind"))
                    .collect::<Vec<_>>();

                let values = cond_values
                    .into_iter()
                    .chain(sim_values)
                    .collect::<Vec<_>>();

                mini_system.populate_w_vec(values.as_slice(), rng);

                let vals = mini_system.simulate();

                //set values
                for (ind, val) in inds.iter().zip(vals.iter()) {
                    grid.set_data_at_ind(ind, *val);
                }
            },
        );
    }
}

#[cfg(test)]
mod test {

    use super::*;
    use std::{fs::File, io::Write};

    use nalgebra::{UnitQuaternion, Vector3};
    use num_traits::Float;
    use rand::SeedableRng;

    use crate::{
        geometry::ellipsoid::Ellipsoid,
        spatial_database::{
            coordinate_system::{CoordinateSystem, GridSpacing},
            gridded_databases::incomplete_grid::InCompleteGriddedDataBase,
            normalized::Normalize,
        },
        variography::model_variograms::spherical::SphericalVariogram,
    };

    #[test]
    fn gsgs_test() {
        // Define the coordinate system for the grid
        // origing at x = 0, y = 0, z = 0
        // azimuth = 0, dip = 0, plunge = 0
        let coordinate_system = CoordinateSystem::new(
            Point3::new(0.5, 0.5, 0.5).into(),
            UnitQuaternion::from_euler_angles(0.0.to_radians(), 0.0.to_radians(), 0.0.to_radians()),
        );

        // create a gridded database from a csv file (walker lake)
        let mut gdb = InCompleteGriddedDataBase::from_csv_index(
            "./data/walker.csv",
            "X",
            "Y",
            "Z",
            "V",
            GridSpacing {
                x: 1.0,
                y: 1.0,
                z: 1.0,
            },
            coordinate_system,
        )
        .expect("Failed to create gdb");

        // normalize the data
        gdb.normalize();

        // create a grid to store the simulation values
        let sim_grid_arr = Array3::<Option<f32>>::from_elem(gdb.shape(), None);
        let mut sim_db = InCompleteGriddedDataBase::new(
            sim_grid_arr,
            gdb.grid_spacing().clone(),
            CoordinateSystem::new(
                Point3::new(0.0, 0.0, 0.0).into(),
                UnitQuaternion::from_euler_angles(
                    0.0.to_radians(),
                    0.0.to_radians(),
                    0.0.to_radians(),
                ),
            ),
        );

        // create a spherical variogram
        // azimuth = 0, dip = 0, plunge = 0
        // range_x = 150, range_y = 50, range_z = 1
        // sill = 1, nugget = 0
        let vgram_rot =
            UnitQuaternion::from_euler_angles(0.0.to_radians(), 0.0.to_radians(), 0.0.to_radians());
        let vgram_origin = Point3::new(0.0, 0.0, 0.0);
        let vgram_coordinate_system = CoordinateSystem::new(vgram_origin.into(), vgram_rot);
        let range = Vector3::new(150.0, 50.0, 10.0);
        let sill = 1.0;
        let nugget = 0.2;

        let spherical_vgram =
            SphericalVariogram::new(range, sill, nugget, vgram_coordinate_system.clone());

        // create search ellipsoid
        let search_ellipsoid = Ellipsoid::new(450.0, 150.0, 10.0, vgram_coordinate_system.clone());
        //let search_ellipsoid = Ellipsoid::new(500.0, 500.0, 1.0, vgram_coordinate_system.clone());

        // create a query engine for the conditioning data
        let query_engine = GriddedDataBaseOctantQueryEngine::new(search_ellipsoid, &gdb, 40);

        // create a gsgs system
        let gsgs = GSGS::new(
            query_engine,
            spherical_vgram,
            GSGSParameters {
                max_octant_cond_data: 40,
                max_octant_sim_data: 40,
                group_size: [4, 4, 1],
            },
        );

        let mut rng = StdRng::from_entropy();

        //simulate values on grid
        gsgs.simulate_grid(&mut sim_db, &mut rng);

        //save values to file for visualization
        let (values, points) = sim_db.data_and_points();

        let mut out = File::create("./test_results/gsgs.txt").unwrap();
        let _ = out.write_all(b"surfs\n");
        let _ = out.write_all(b"4\n");
        let _ = out.write_all(b"x\n");
        let _ = out.write_all(b"y\n");
        let _ = out.write_all(b"z\n");
        let _ = out.write_all(b"value\n");

        for (point, value) in points.iter().zip(values.iter()) {
            //println!("point: {:?}, value: {}", point, value);
            let _ = out
                .write_all(format!("{} {} {} {}\n", point.x, point.y, point.z, value).as_bytes());
        }
    }
}
