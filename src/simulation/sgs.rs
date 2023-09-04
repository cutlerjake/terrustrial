use ndarray::Array3;
use rand::{prelude::SliceRandom, rngs::StdRng};
use rand_distr::Distribution;
use rand_distr::Normal;
use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};

use crate::{
    geometry::Geometry,
    kriging::simple_kriging::SimpleKrigingSystem,
    spatial_database::{
        gridded_databases::{
            gridded_data_base_query_engine::GriddedDataBaseOctantQueryEngine,
            GriddedDataBaseInterface,
        },
        SpatialQueryable,
    },
    variography::model_variograms::VariogramModel,
};

pub struct SGSParameters {
    pub max_octant_cond_data: usize,
    pub max_octant_sim_data: usize,
}

pub struct SGS<S, V, G>
where
    S: SpatialQueryable<f32, G>,
{
    conditioning_data: S,
    variogram_model: V,
    sgs_parameters: SGSParameters,
    phantom: std::marker::PhantomData<G>,
}

impl<S, V, G> SGS<S, V, G>
where
    S: SpatialQueryable<f32, G> + Sync,
    V: VariogramModel + Sync,
    G: Geometry + Sync + Clone,
{
    /// Create a new simple kriging estimator with the given parameters
    /// # Arguments
    /// * `conditioning_data` - The data to condition the kriging system on (mut be normalized)
    /// * `variogram_model` - The variogram model to use
    /// * `search_ellipsoid` - The search ellipsoid to use
    /// * `sgs_parameters` - The SGS parameters to use
    /// # Returns
    /// A new simple kriging estimator
    pub fn new(conditioning_data: S, variogram_model: V, sgs_parameters: SGSParameters) -> Self {
        Self {
            conditioning_data,
            variogram_model,
            sgs_parameters,
            phantom: std::marker::PhantomData,
        }
    }

    /// Perform simple kriging at all kriging points
    pub fn simulate_grid<GDB>(&self, grid: &mut GDB, rng: &mut StdRng)
    where
        GDB: GriddedDataBaseInterface<f32> + std::marker::Sync,
    {
        //construct kriging system
        let mut kriging_system = SimpleKrigingSystem::new(
            (self.sgs_parameters.max_octant_cond_data + self.sgs_parameters.max_octant_sim_data)
                * 8,
        );

        // create query engine for simulation grid
        let geometry = self.conditioning_data.geometry().clone();
        let sim_qe = GriddedDataBaseOctantQueryEngine::new(
            geometry.clone(),
            grid,
            self.sgs_parameters.max_octant_sim_data,
        );

        // Array to store simulation order
        let mut simulation_order = Array3::from_elem(grid.shape(), 0);
        let mut path = simulation_order
            .indexed_iter()
            .map(|(ind, _)| ind)
            .collect::<Vec<_>>();

        //shuffle path order
        path.shuffle(&mut rand::thread_rng());

        //set iteration order
        for (ind, val) in path.iter().enumerate() {
            simulation_order[*val] = ind;
        }

        // Note: We do not know the values so we can't populate the grid
        // But at each location in the grid we now all the points that will be previously simulated and the locations of the conditioning data
        // thus, we can solve for the weights in parrallel, then populate the grid sequentially
        let sequential_data = path
            .par_iter()
            .map_with(kriging_system.clone(), |local_system, ind| {
                let ind = [ind.0 as isize, ind.1 as isize, ind.2 as isize];
                //get kriging point
                let point = grid.ind_to_point(&ind);

                //get nearest conditioning  points and values
                let (cond_values, mut cond_points) = self.conditioning_data.query(&point);

                // get nearest simulation points
                let (sim_inds, sim_points) =
                    sim_qe.nearest_inds_and_points_masked(&point, |neighbor_ind| {
                        simulation_order[neighbor_ind]
                            < simulation_order[ind.map(|ind| ind as usize)]
                    });

                //append simulation points to conditioning points
                cond_points.extend(sim_points.iter());

                let mini_system = local_system.build_mini_system(
                    cond_points.as_slice(),
                    &point,
                    &self.variogram_model,
                );

                (ind, cond_values, sim_inds, mini_system)
            })
            .collect::<Vec<_>>();

        let mut values_mat = kriging_system.values.clone();

        sequential_data
            .into_iter()
            .for_each(|(ind, cond_values, sim_inds, mini_system)| {
                //get simulation values
                let sim_values = sim_inds
                    .iter()
                    .map(|ind| grid.data_at_ind(ind).expect("No value at ind"))
                    .collect::<Vec<_>>();

                let values = cond_values
                    .iter()
                    .chain(sim_values.iter())
                    .collect::<Vec<_>>();

                //store values and weights in kriging system
                unsafe {
                    values_mat.set_dims(values.len(), 1);
                };
                for i in 0..values.len() {
                    unsafe { values_mat.write_unchecked(i, 0, *values[i]) };
                }

                //compute mean
                let mean = mini_system.estimate(values_mat.as_ref());
                //assert!(mean == mini_mean);

                //compute variance
                let variance = mini_system.variance();

                let value = mean + Normal::new(0.0, variance).unwrap().sample(rng);

                //set value
                grid.set_data_at_ind(&ind.map(|v| v as usize), value);
            });
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use std::{fs::File, io::Write};

    use nalgebra::{Point3, UnitQuaternion, Vector3};
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
    fn sgs_test() {
        // Define the coordinate system for the grid
        // origing at x = 0, y = 0, z = 0
        // azimuth = 0, dip = 0, plunge = 0
        let coordinate_system = CoordinateSystem::new(
            Point3::new(0.01, 0.01, 0.01).into(),
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
        let range = Vector3::new(150.0, 50.0, 1.0);
        let sill = 1.0;
        let nugget = 0.0;

        let spherical_vgram =
            SphericalVariogram::new(range, sill, nugget, vgram_coordinate_system.clone());

        // create search ellipsoid
        let search_ellipsoid = Ellipsoid::new(450.0, 150.0, 1.0, vgram_coordinate_system.clone());

        // create a query engine for the conditioning data
        let query_engine = GriddedDataBaseOctantQueryEngine::new(search_ellipsoid, &gdb, 16);

        // create a gsgs system
        let gsgs = SGS::new(
            query_engine,
            spherical_vgram,
            SGSParameters {
                max_octant_cond_data: 16,
                max_octant_sim_data: 16,
            },
        );

        let mut rng = StdRng::from_entropy();

        //simulate values on grid
        gsgs.simulate_grid(&mut sim_db, &mut rng);

        //save values to file for visualization
        let (values, points) = sim_db.data_and_points();

        let mut out = File::create("./test_results/sgs.txt").unwrap();
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
