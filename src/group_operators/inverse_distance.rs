use itertools::Itertools;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use ultraviolet::DVec3;

use crate::spatial_database::coordinate_system::CoordinateSystem;
use crate::spatial_database::group_provider::GroupProvider;
use crate::{geometry::ellipsoid::Ellipsoid, spatial_database::ConditioningProvider};

use super::ConditioningParams;

#[allow(clippy::too_many_arguments)]
pub fn estimate(
    conditioning_data: &impl ConditioningProvider<Data = f64>,
    conditioning_params: &ConditioningParams,
    coordinate_system: &CoordinateSystem,
    scaling: DVec3,
    search_ellipsoid: &Ellipsoid,
    groups: &GroupProvider,
    power: f64,
) -> Vec<f64> {
    (0..groups.n_groups())
        .into_par_iter()
        .map_with(
            (coordinate_system, search_ellipsoid.clone()),
            |(cs, ellipsoid), group_idx| {
                let group = groups.get_group(group_idx);
                // Determine the center of the group.
                let center = group.iter().fold(DVec3::zero(), |mut acc, x| {
                    acc += x.center();
                    acc
                }) / (group.len() as f64);

                // Get the conditioning data for the group.
                ellipsoid.translate_to(center);

                // Get nearest points and values.
                let (_, cond_values, cond_points, sufficiently_conditioned) =
                    conditioning_data.query(&center, ellipsoid, conditioning_params);

                if sufficiently_conditioned {
                    // Compute scaled distances to each node in group
                    let weights = group
                        .iter()
                        .cartesian_product(cond_points.iter())
                        .map(|(node, cond_point)| {
                            // Distance between node to be estimated and conditioning point in world space.
                            let distance = node.center() - cond_point.center();

                            // Convert distance to local space.
                            let distance = cs.into_local().rotation * distance;

                            // Scale distance.
                            let distance = distance / scaling;

                            // Convert distance back to world space.
                            let distance = distance.mag();

                            // Compute the inverse distance.
                            distance.powf(power).recip()
                        })
                        .collect::<Vec<_>>();

                    // Compute and return the estimate for the group.
                    weights
                        .chunks(cond_values.len())
                        .map(|chunk| {
                            chunk
                                .iter()
                                .zip(cond_values.iter())
                                .map(|(w, v)| w * v)
                                .sum::<f64>()
                                / chunk.iter().sum::<f64>()
                        })
                        .collect()
                } else {
                    vec![f64::NAN; group.len()]
                }
            },
        )
        .flatten()
        .collect::<Vec<_>>()
}
