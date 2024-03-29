use std::marker::PhantomData;

use indicatif::ParallelProgressIterator;
use nalgebra::Point3;
use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};

use crate::{spatial_database::SpatialQueryable, variography::model_variograms::VariogramModel};

pub mod simple_kriging;

pub trait KrigingSystem: Clone {
    fn new(n_elems: usize) -> Self;
    fn build_system<V>(
        &mut self,
        conditioning_points: &[Point3<f32>],
        conditioning_values: &[f32],
        kriging_point: &Point3<f32>,
        variogram_model: &V,
    ) where
        V: VariogramModel;

    fn estimate(&self) -> f32;
    fn variance(&self) -> f32;
}

pub struct KrigingParameters {
    pub max_cond_data: usize,
    pub min_cond_data: usize,
    pub min_octant_data: usize,
    pub max_octant_data: usize,
}

pub struct Kriging<S, V, G, KS>
where
    S: SpatialQueryable<f32, G>,
    KS: KrigingSystem + Send,
{
    conditioning_data: S,
    variogram_model: V,
    kriging_parameters: KrigingParameters,
    phantom: PhantomData<fn() -> G>, //ugly hack to get around the fact that G does not implement Sync or Send
    phantom2: PhantomData<KS>, //ugly hack to get around the fact that KS does not implement Sync or Send
}

impl<S, V, G, KS> Kriging<S, V, G, KS>
where
    S: SpatialQueryable<f32, G> + Sync,
    V: VariogramModel + Sync,
    KS: KrigingSystem + Send + Sync,
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
        kriging_parameters: KrigingParameters,
    ) -> Self {
        Self {
            conditioning_data,
            variogram_model,
            kriging_parameters,
            phantom: PhantomData,
            phantom2: PhantomData,
        }
    }

    /// Perform simple kriging at all kriging points
    pub fn krig(&self, kriging_points: &[Point3<f32>]) -> Vec<f32> {
        //construct kriging system
        //let kriging_system = SimpleKrigingSystem::new(self.kriging_parameters.max_octant_data * 8);

        let kriging_system = KS::new(self.kriging_parameters.max_octant_data * 8);

        kriging_points
            .par_iter()
            .progress()
            .map_with(kriging_system.clone(), |local_system, kriging_point| {
                //let mut local_system = kriging_system.clone();
                //get nearest points and values
                let (cond_values, cond_points) = self.conditioning_data.query(kriging_point);

                //build kriging system for point
                local_system.build_system(
                    &cond_points,
                    cond_values.as_slice(),
                    kriging_point,
                    &self.variogram_model,
                );

                local_system.estimate()
            })
            .collect::<Vec<f32>>()
    }
}
