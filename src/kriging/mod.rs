use nalgebra::Point3;
use num_traits::Float;
use simba::simd::{SimdPartialOrd, SimdRealField, SimdValue};

use crate::variography::model_variograms::VariogramModel;

use self::simple_kriging::SKBuilder;

pub mod generalized_sequential_indicator_kriging;
pub mod generalized_sequential_kriging;
//pub mod greedy_generalized_sequential_kriging;
pub mod simple_kriging;

#[derive(Debug, Clone)]
pub struct ConditioningParams {
    //number of conditioning points
    pub max_n_cond: usize,
    pub min_n_cond: usize,

    //limit on the number of conditioning points per octant
    pub max_octant: usize,
    pub min_conditioned_octants: usize,

    //clips extreme values if h > clip_h
    pub clip_h: Vec<f32>,
    pub clip_range: Vec<[f32; 2]>,

    //filter values outside of this value range
    pub valid_value_range: [f32; 2],

    //limit number of points from same source group
    pub same_source_group_limit: usize,

    //dynamic orientation
    pub orient_search: bool,
    pub orient_variogram: bool,
}

impl ConditioningParams {
    pub fn new(
        max_n_cond: usize,
        min_n_cond: usize,
        max_octant: usize,
        min_conditioned_octants: usize,
        clip_h: Vec<f32>,
        clip_range: Vec<[f32; 2]>,
        valid_value_range: [f32; 2],
        same_source_group_limit: usize,
        orient_search: bool,
        orient_variogram: bool,
    ) -> Self {
        Self {
            max_n_cond,
            min_n_cond,
            max_octant,
            min_conditioned_octants,

            clip_h,
            clip_range,

            valid_value_range,

            same_source_group_limit,

            orient_search,
            orient_variogram,
        }
    }

    pub fn clipped_value(&self, value: f32, h: f32) -> f32 {
        for (i, clip_h) in self.clip_h.iter().enumerate().rev() {
            if h > *clip_h {
                let clip_range = self.clip_range[i];
                if value < clip_range[0] {
                    return clip_range[0];
                } else if value > clip_range[1] {
                    return clip_range[1];
                }
            }
        }

        value
    }
}

impl Default for ConditioningParams {
    fn default() -> Self {
        Self {
            max_n_cond: 32,
            min_n_cond: 4,
            max_octant: 8,
            min_conditioned_octants: 1,
            clip_h: vec![],
            clip_range: vec![],
            valid_value_range: [f32::MIN, f32::MAX],
            same_source_group_limit: usize::MAX,
            orient_search: false,
            orient_variogram: false,
        }
    }
}

pub trait KrigingSystem<V, T>: Clone
where
    V: VariogramModel<T>,
    T: SimdPartialOrd + SimdRealField + SimdValue<Element = f32> + Copy,
    <T as SimdValue>::Element: SimdRealField + Float,
{
    fn new(n_elems: usize) -> Self;
    fn build_system<SKB>(
        &mut self,
        conditioning_points: &[SKB::Support],
        conditioning_values: &[f32],
        kriging_point: &Point3<f32>,
        variogram_model: &V,
    ) where
        SKB: SKBuilder;

    fn estimate(&self) -> f32;
    fn variance(&self) -> f32;
}

pub struct KrigingParameters {
    pub max_cond_data: usize,
    pub min_cond_data: usize,
    pub min_octant_data: usize,
    pub max_octant_data: usize,
    pub orient_search: bool,
    pub orient_variogram: bool,
}

// pub struct Kriging<S, V, G, KS, T>
// where
//     S: SpatialQueryable<f32, G>,
//     KS: KrigingSystem<V, T> + Send,
//     T: SimdPartialOrd + SimdRealField + SimdValue<Element = f32> + Copy,
//     V: VariogramModel<T>,
// {
//     conditioning_data: S,
//     variogram_model: V,
//     kriging_parameters: KrigingParameters,
//     phantom: PhantomData<fn() -> G>, //ugly hack to get around the fact that G does not implement Sync or Send
//     phantom2: PhantomData<KS>, //ugly hack to get around the fact that KS does not implement Sync or Send
//     phantom3: PhantomData<T>,
// }

// impl<S, V, G, KS, T> Kriging<S, V, G, KS, T>
// where
//     S: SpatialQueryable<f32, G> + Sync,
//     V: VariogramModel<T> + Sync,
//     KS: KrigingSystem<V, T> + Send + Sync,
//     T: SimdPartialOrd + SimdRealField + SimdValue<Element = f32> + Copy,
// {
//     /// Create a new simple kriging estimator with the given parameters
//     /// # Arguments
//     /// * `conditioning_data` - The data to condition the kriging system on
//     /// * `variogram_model` - The variogram model to use
//     /// * `search_ellipsoid` - The search ellipsoid to use
//     /// * `kriging_parameters` - The kriging parameters to use
//     /// # Returns
//     /// A new simple kriging estimator
//     pub fn new(
//         conditioning_data: S,
//         variogram_model: V,
//         kriging_parameters: KrigingParameters,
//     ) -> Self {
//         Self {
//             conditioning_data,
//             variogram_model,
//             kriging_parameters,
//             phantom: PhantomData,
//             phantom2: PhantomData,
//             phantom3: PhantomData,
//         }
//     }

//     /// Perform simple kriging at all kriging points
//     pub fn krig(&self, kriging_points: &[Point3<f32>]) -> Vec<f32> {
//         //construct kriging system
//         //let kriging_system = SimpleKrigingSystem::new(self.kriging_parameters.max_octant_data * 8);

//         let kriging_system = KS::new(self.kriging_parameters.max_octant_data * 8);

//         kriging_points
//             .par_iter()
//             .progress()
//             .map_with(kriging_system.clone(), |local_system, kriging_point| {
//                 //let mut local_system = kriging_system.clone();
//                 //get nearest points and values
//                 let (cond_values, cond_points) = self.conditioning_data.query(kriging_point);

//                 //build kriging system for point
//                 local_system.build_system::(
//                     &cond_points,
//                     cond_values.as_slice(),
//                     kriging_point,
//                     &self.variogram_model,
//                 );

//                 local_system.estimate()
//             })
//             .collect::<Vec<f32>>()
//     }
// }
