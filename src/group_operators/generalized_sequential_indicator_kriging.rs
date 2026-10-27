use crate::group_operators::generalized_sequential_kriging as gsk;
use crate::spatial_database::group_provider::GroupProvider;
use crate::spatial_database::MappedIterNearest;
use crate::systems::solved_systems::SolvedSystemBuilder;
use crate::variography::model_variograms::composite::CompositeVariogram;
use crate::{geometry::ellipsoid::Ellipsoid, spatial_database::ConditioningProvider};

use itertools::izip;

use crate::group_operators::ConditioningParams;

/// The cpdf associated with the indicator kriging estimates.
///
/// `p[i]` is the probability that the value is less than `x[i]`.
/// The raw pdf may not be valid, (negative kriging weights, poorly modeled variograms, lack of data, etc..).
/// A correction method is provided, which uses a two-pass averaging method to ensure valid order relations.
#[derive(Clone, Debug, Default)]
pub struct IKCPDF {
    pub p: Vec<f64>,
    pub x: Vec<f64>,
}

impl IKCPDF {
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            p: Vec::with_capacity(capacity),
            x: Vec::with_capacity(capacity),
        }
    }

    pub fn correct(&mut self) {
        //clamp p within 0 and 1
        self.p.iter_mut().for_each(|x| {
            *x = (*x).clamp(0.0, 1.0);
        });

        let mut curr_max = f64::MIN;
        let forward_running_max = self.p.iter().map(|v| {
            if v > &curr_max {
                curr_max = *v;
            }
            curr_max
        });

        let mut curr_min = f64::MAX;
        let backward_min = self.p.iter().rev().map(|v| {
            if v < &curr_min {
                curr_min = *v;
            }
            curr_min
        });

        self.p = izip!(forward_running_max, backward_min.rev())
            .map(|(f, b)| (f + b) / 2.0)
            .collect();

        //set p to 1.0 for last theshold
        *self.p.last_mut().unwrap() = 1.0;
    }
}

#[allow(clippy::too_many_arguments)]
pub fn estimate(
    thresholds: &[f64],
    conditioning_data: &impl ConditioningProvider<Data = f64>,
    conditioning_params: &ConditioningParams,
    vgram_models: &[CompositeVariogram],
    ellipsoid: &Ellipsoid,
    groups: &GroupProvider,
    kriging_type: impl SolvedSystemBuilder,
) -> Vec<IKCPDF> {
    let mut cpdfs: Vec<IKCPDF> = Vec::new();

    for (i, theshold) in thresholds.iter().enumerate() {
        //create indicator conditioning provider

        let cond = MappedIterNearest::new(conditioning_data, |mut x| {
            if x.data <= *theshold {
                x.data = 1.0;
            } else {
                x.data = 0.0;
            }

            x
        });

        let estimates = gsk::estimate(
            &cond,
            conditioning_params,
            &vgram_models[i],
            ellipsoid.clone(),
            groups,
            kriging_type.clone(),
        );

        //update cpdfs
        estimates.iter().enumerate().for_each(|(i, e)| {
            if let Some(cdpf) = cpdfs.get_mut(i) {
                cdpf.p.push(*e);
                cdpf.x.push(*theshold);
            } else {
                let mut cpdf = IKCPDF::with_capacity(thresholds.len());
                cpdf.p.push(*e);
                cpdf.x.push(*theshold);
                cpdfs.push(cpdf);
            }
        });
    }
    cpdfs
}
