use std::os::windows::thread;

use nalgebra::Point3;
use ndarray::Array3;
use rand::{prelude::SliceRandom, Rng};
use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};

use crate::{
    geometry::{self, Geometry},
    kriging::simple_kriging::SimpleKrigingSystem,
    spatial_database::{
        gridded_databases::{
            gridded_data_base_queary_engine::GriddedDataBaseOctantQueryEngine,
            gridded_db::RawGriddedDataBase, GriddedDataBaseInterface,
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
    /// * `conditioning_data` - The data to condition the kriging system on
    /// * `variogram_model` - The variogram model to use
    /// * `search_ellipsoid` - The search ellipsoid to use
    /// * `kriging_parameters` - The kriging parameters to use
    /// # Returns
    /// A new simple kriging estimator
    pub fn new(
        conditioning_data: S,
        variogram_model: V,
        kriging_parameters: SGSParameters,
    ) -> Self {
        Self {
            conditioning_data,
            variogram_model,
            sgs_parameters: kriging_parameters,
            phantom: std::marker::PhantomData,
        }
    }

    /// Perform simple kriging at all kriging points
    pub fn simulate_grid<GDB>(&self, grid: &mut GDB)
    where
        GDB: GriddedDataBaseInterface<f32> + std::marker::Sync,
    {
        //construct kriging system
        let mut kriging_system = SimpleKrigingSystem::new(
            (self.sgs_parameters.max_octant_cond_data + self.sgs_parameters.max_octant_sim_data)
                * 8,
        );

        // create queary engine for simulation grid
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

        //create grid octant queary engine for simulation order
        // let so_db = RawGriddedDataBase::new(
        //     simulation_order.clone(),
        //     grid.grid_spacing(),
        //     grid.coordinate_system(),
        // );

        // let so_qe = GriddedDataBaseOctantQueryEngine::new(
        //     geometry.clone(),
        //     &so_db,
        //     self.sgs_parameters.max_octant_cond_data,
        // );

        // Equipped with the simulation order and the original conditioning data we can solve for the SK weights in parralel
        // Note: We do not know the values so we can't populate the grid
        // But at each location in the grid we now all the points that will be previously simulated and the locations of the conditioning data
        // thus, we can solve for the weights in parrallel, then populate the grid in serial
        let sequential_data = path
            .par_iter()
            .map_with(kriging_system.clone(), |local_system, ind| {
                let ind = [ind.0 as isize, ind.1 as isize, ind.2 as isize];
                //get kriging point
                let kriging_point = grid.ind_to_point(&ind);

                //get nearest conditioning  points and values
                let (cond_values, mut cond_points) = self.conditioning_data.query(&kriging_point);

                // get nearest simulation points
                let (sim_inds, sim_points) =
                    sim_qe.nearest_inds_and_points_masked(&kriging_point, |neighbor_ind| {
                        simulation_order[neighbor_ind]
                            < simulation_order[ind.map(|ind| ind as usize)]
                    });

                //append simulation points to conditioning points
                cond_points.extend(sim_points.iter());

                //set dimension of kriging system
                local_system.set_dim(cond_points.len());

                //build kriging system for point
                local_system.vectorized_build_covariance_matrix_and_vector(
                    cond_points.as_slice(),
                    &kriging_point,
                    &self.variogram_model,
                );

                //compute weights of kriging system
                local_system.compute_weights();
                let weights = local_system.weights.clone();

                (ind, kriging_point, cond_values, sim_inds, weights)
            })
            .collect::<Vec<_>>();

        let mut rng = rand::thread_rng();
        sequential_data.into_iter().for_each(
            |(ind, kriging_point, cond_values, sim_inds, weights)| {
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
                unsafe { kriging_system.values.set_dims(values.len(), 1) };
                for i in 0..values.len() {
                    unsafe { kriging_system.values.write_unchecked(i, 0, *values[i]) };
                }
                kriging_system.weights = weights;

                //compute mean
                let mean = kriging_system.estimate();

                //compute variance
                let variance = kriging_system.variance();

                let value = mean + rng.gen::<f32>() * variance;

                //set value
                grid.set_data_at_ind(&ind.map(|v| v as usize), value);
            },
        );
    }
}
