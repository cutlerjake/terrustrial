use std::ops::Range;

use crate::geometry::ellipsoid::Ellipsoid;
use crate::spatial_database::group_provider::GroupProvider;
use crate::spatial_database::ConditioningProvider;
use crate::spatial_database::FilterMapResult;
use crate::spatial_database::FilterMappedIterNearest;
use crate::spatial_database::SpatialAcceleratedDB;
use crate::systems::lu::LUSystem;
use crate::systems::solved_systems::SolvedLUSystem;
use crate::systems::solved_systems::SolvedSystemBuilder;
use crate::variography::model_variograms::composite::CompositeVariogram;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use ultraviolet::DVec3;

use crate::group_operators::ConditioningParams;
use rayon::prelude::*;

pub fn estimate(
    conditioning_data: &impl ConditioningProvider<Data = f64>,
    conditioning_params: &ConditioningParams,
    vgram: &CompositeVariogram,
    ellipsoid: Ellipsoid,
    groups: &GroupProvider,
    kriging_type: impl SolvedSystemBuilder,
) -> Vec<f64> {
    let local_system = LUSystem::new(groups.max_group_size(), conditioning_params.max_n_cond);

    (0..groups.n_groups())
        .into_par_iter()
        .map_with(
            (
                ellipsoid.clone(),
                local_system.clone(),
                vec![],
                vec![],
                vec![],
                vec![],
            ),
            |(ellipsoid, local_system, h_buffer, pt_buffer, var_buffer, ind_buffer), group_idx| {
                let group = groups.get_group(group_idx);
                // 1. Get center of group.
                let center = group.iter().fold(DVec3::zero(), |mut acc, x| {
                    acc += x.center();
                    acc
                }) / (group.len() as f64);

                //translate search ellipsoid to group center
                ellipsoid.translate_to(center);

                //get nearest points and values
                let (_, cond_values, mut supports, sufficiently_conditioned) =
                    conditioning_data.query(&center, ellipsoid, conditioning_params);
                let n_cond = supports.len();

                if sufficiently_conditioned {
                    //build kriging system for point
                    supports.extend_from_slice(group);
                    local_system.build_cov_matrix(
                        n_cond,
                        group.len(),
                        &supports,
                        vgram,
                        h_buffer,
                        pt_buffer,
                        var_buffer,
                        ind_buffer,
                    );

                    let Ok(mut solved_system) = kriging_type.build(local_system) else {
                        return vec![f64::NAN; group.len()];
                    };

                    solved_system.populate_cond_values_est(cond_values.as_slice());

                    return solved_system.estimate();
                }
                vec![f64::NAN; group.len()]
            },
        )
        .flatten()
        .collect::<Vec<f64>>()
}

pub fn simulate(
    conditioning_data: &impl ConditioningProvider<Data = f64>,
    data_conditioning_params: &ConditioningParams,
    sim_conditioning_params: &ConditioningParams,
    vgram: &CompositeVariogram,
    ellipsoid: Ellipsoid,
    groups: &GroupProvider,
    kriging_type: impl SolvedSystemBuilder,
) -> Vec<f64> {
    // 1. Create a randomized simulation path.
    let mut rng = StdRng::from_entropy();
    let mut simulation_path = (0..groups.n_groups()).collect::<Vec<_>>();
    simulation_path.shuffle(&mut rng);

    // 2. Get the simulation index of each node
    let simulation_order = simulation_path.iter().enumerate().fold(
        vec![0; groups.n_nodes()],
        |mut simulation_order, (i, group_idx)| {
            let group_range = groups.get_group_range(*group_idx);
            simulation_order[group_range].fill(i);
            simulation_order
        },
    );

    // 3. Build a spatially accelerated db over the nodes to be simulated.
    let sim_location_db =
        SpatialAcceleratedDB::new(groups.get_supports().to_vec(), simulation_order);

    // 4. Identify the simulation groups that are ill-conditioned
    let sim_conditioned = (0..groups.n_groups())
        .into_par_iter()
        .map_with(ellipsoid.clone(), |ellipsoid, group_idx| {
            let group = groups.get_group(group_idx);

            // 1. Get center of group.
            let center = group.iter().fold(DVec3::zero(), |mut acc, x| {
                acc += x.center();
                acc
            }) / (group.len() as f64);

            //translate search ellipsoid to group center
            ellipsoid.translate_to(center);

            //get nearest points and values
            let (_, _cond_values, _supports, sufficiently_conditioned) =
                conditioning_data.query(&center, ellipsoid, data_conditioning_params);

            sufficiently_conditioned
        })
        .collect::<Vec<_>>();

    // 5. Create a local system large enough to handle the largest simulation group.
    let local_system = LUSystem::new(
        groups.max_group_size(),
        data_conditioning_params.max_n_cond + sim_conditioning_params.max_n_cond,
    );

    // Helper enum to determine what values to insert into the output vec fo simulated values.
    pub enum Action<T> {
        Insert(Range<usize>, T),
        Nullify(Range<usize>),
    }

    // 6. Solve all valid systems
    let actions = simulation_path
        .par_iter()
        .copied()
        .enumerate()
        .map_with(
            (
                ellipsoid.clone(),
                local_system.clone(),
                vec![],
                vec![],
                vec![],
                vec![],
            ),
            |(ellipsoid, local_system, h_buffer, pt_buffer, var_buffer, ind_buffer),
             (sim_step, group_idx)| {
                let group = groups.get_group(group_idx);
                let group_range = groups.get_group_range(group_idx);

                if !sim_conditioned[group_idx] {
                    return Action::Nullify(group_range);
                }
                // Get center of group.
                let center = group.iter().fold(DVec3::zero(), |mut acc, x| {
                    acc += x.center();
                    acc
                }) / (group.len() as f64);

                //translate search ellipsoid to group center
                ellipsoid.translate_to(center);

                // Get the hard-data conditioning.
                let (_, cond_values, mut supports, sufficiently_conditioned) =
                    conditioning_data.query(&center, ellipsoid, data_conditioning_params);

                // Get the conditioning of previously simulated supports.
                let sim_query = FilterMappedIterNearest::new(&sim_location_db, |elem| {
                    if !ellipsoid.may_contain_local_point_at_sq_dist(elem.sq_dist) {
                        return FilterMapResult::ExitEarly;
                    }
                    let group_ind = simulation_path[elem.data];
                    if elem.data < sim_step && sim_conditioned[group_ind] {
                        FilterMapResult::Mapped(elem)
                    } else {
                        FilterMapResult::Ignore
                    }
                });

                let (sim_cond_inds, _, sim_supports, _) =
                    sim_query.query(&center, ellipsoid, sim_conditioning_params);

                supports.extend_from_slice(&sim_supports);
                let n_cond = supports.len();

                if sufficiently_conditioned {
                    //build kriging system for group
                    supports.extend_from_slice(group);
                    local_system.build_cov_matrix(
                        n_cond,
                        group.len(),
                        &supports,
                        vgram,
                        h_buffer,
                        pt_buffer,
                        var_buffer,
                        ind_buffer,
                    );

                    let Ok(solved_system) = kriging_type.build(local_system) else {
                        return Action::Nullify(group_range);
                    };

                    return Action::Insert(
                        group_range,
                        (solved_system, sim_cond_inds, cond_values),
                    );
                }
                Action::Nullify(group_range)
            },
        )
        .collect::<Vec<_>>();

    // 7. Actually compute the simulated values.
    let mut out = vec![0.0; groups.n_nodes()];
    for action in actions {
        match action {
            Action::Insert(range, (mut solved_system, sim_cond_inds, mut cond_values)) => {
                cond_values.extend(sim_cond_inds.into_iter().map(|i| out[i]));
                solved_system.populate_cond_values_sim(&cond_values, &mut rng);
                let sim_values = solved_system.simulate();
                out[range].copy_from_slice(&sim_values);
            }
            Action::Nullify(range) => {
                out[range].fill(f64::NAN);
            }
        }
    }
    out
}
