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
    println!("Finding ill conditioned groups");
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
    println!("Solving systems.");
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
    println!("Computing simulated values");
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

#[cfg(test)]
mod test {
    use std::{fs::File, io::Write};

    use itertools::Itertools;
    use num_traits::Float;
    use ultraviolet::DRotor3;

    use crate::{
        geometry::{aabb::Aabb, support::Support},
        spatial_database::{coordinate_system::CoordinateSystem, DiscretiveVolume},
        systems::solved_systems::{
            ok_system::SolvedLUOKSystemBuilder, sk_system::SolvedLUSKSystemBuilder,
        },
        variography::model_variograms::{
            composite::{CompositeVariogram, VariogramType},
            spherical::SphericalVariogram,
        },
    };

    use super::*;

    #[test]
    fn gsk_ok_test() {
        // create a gridded database from a csv file (walker lake)
        println!("Reading Cond Data");
        let cond = SpatialAcceleratedDB::from_csv_index("C:/Users/2jake/OneDrive - McGill University/Fall2022/MIME525/Project4/mineralized_domain_composites.csv", "X", "Y", "Z", "CU")
            .expect("Failed to create gdb");

        //

        let vgram_rot = DRotor3::from_euler_angles(0.00.to_radians(), 0.0, 0.0);
        let range = DVec3 {
            x: 200.0,
            y: 50.0,
            z: 50.0,
        };
        let sill = 1.0;

        let spherical_vgram = CompositeVariogram::new(vec![VariogramType::Spherical(
            SphericalVariogram::new(range, sill, vgram_rot),
        )]);

        // create search ellipsoid
        let search_ellipsoid = Ellipsoid::new(
            200f64,
            50f64,
            50f64,
            CoordinateSystem::new(DVec3::zero(), vgram_rot),
        );

        println!("Reading Target Data");
        let targ = SpatialAcceleratedDB::<f64>::from_csv_index(
            "C:/Users/2jake/OneDrive - McGill University/Fall2022/MIME525/Project4/target.csv",
            "X",
            "Y",
            "Z",
            "V",
        )
        .unwrap();

        let points = targ.supports.iter().map(|s| s.center()).collect_vec();

        //map points in vec of group of points (64)
        //map points in vec of group of points (64)
        let mut block_inds = Vec::new();
        let all_points = points
            .iter()
            .enumerate()
            .map(|(i, point)| {
                let aabb = Aabb::from_min_max(
                    DVec3::new(point.x, point.y, point.z),
                    DVec3::new(point.x + 5.0, point.y + 5.0, point.z + 10.0),
                );

                let disc_points = aabb.discretize(5.0, 5.0, 10.0);
                block_inds.append(vec![i; disc_points.len()].as_mut());
                disc_points
            })
            .flatten()
            .map(|p| {
                Support::Point(DVec3 {
                    x: p.x,
                    y: p.y,
                    z: p.z,
                })
            })
            .collect::<Vec<_>>();

        let groups = GroupProvider::optimized_groups(&all_points, 5f64, 5f64, 10f64, 2, 2, 2);

        // let mut node_provider = PointGroupProvider::from_groups(groups.clone(), orientations);
        let mut params = ConditioningParams::default();
        params.orient_search = true;
        params.orient_variogram = true;
        params.max_n_cond = 36;
        params.valid_value_range = [0.0, f64::INFINITY];

        println!("params: {:#?}", params);
        let time1 = std::time::Instant::now();
        // let builder = SolvedNegativeWeightFilteredSystemBuilder::new(SolvedLUOKSystemBuilder);
        let values = simulate(
            &cond,
            &params,
            &params,
            &spherical_vgram,
            search_ellipsoid,
            &groups,
            SolvedLUSKSystemBuilder,
        );
        let time2 = std::time::Instant::now();
        println!("Time: {:?}", (time2 - time1).as_secs());
        println!(
            "Points per minute: {}",
            values.len() as f64 / (time2 - time1).as_secs_f64() * 60.0
        );

        // let block_values = values.iter().zip(inds.iter().flatten()).fold(
        //     vec![vec![]; points.len()],
        //     |mut acc, (value, ind)| {
        //         acc[block_inds[*ind]].push(*value);
        //         acc
        //     },
        // );

        let block_values = (0..groups.n_groups()).fold(vec![vec![]; points.len()], |mut acc, i| {
            let group_inds = groups.get_original_idxs(i);
            let group_range = groups.get_group_range(i);

            for (og_i, val) in group_inds.iter().zip(values[group_range].iter()) {
                acc[block_inds[*og_i]].push(*val)
            }

            acc
        });

        let avg_block_values = block_values
            .iter()
            .map(|x| x.iter().sum::<f64>() / x.len() as f64)
            .collect::<Vec<_>>();

        //save values to file for visualization

        let mut out = File::create("./test_results/lu_ok.txt").unwrap();
        let _ = out.write_all(b"surfs\n");
        let _ = out.write_all(b"4\n");
        let _ = out.write_all(b"x\n");
        let _ = out.write_all(b"y\n");
        let _ = out.write_all(b"z\n");
        let _ = out.write_all(b"value\n");

        for (point, value) in points.iter().zip(avg_block_values.iter()) {
            //println!("point: {:?}, value: {}", point, value);
            let _ = out
                .write_all(format!("{} {} {} {}\n", point.x, point.y, point.z, value).as_bytes());
        }

        let mut out = File::create("./test_results/lu_ok_cond_data.csv").unwrap();

        let _ = out.write(b"x,y,z,v\n");

        for (point, value) in cond.supports.iter().zip(cond.data.iter()) {
            //println!("point: {:?}, value: {}", point, value);
            let point = point.center();
            let _ = out
                .write_all(format!("{},{},{},{}\n", point.x, point.y, point.z, value).as_bytes());
        }

        let mut out = File::create("./test_results/lu_ok.csv").unwrap();
        //write header
        let _ = out.write_all("X,Y,Z,DX,DY,DZ,V\n".as_bytes());

        //write each row

        for (point, value) in points.iter().zip(avg_block_values.iter()) {
            //println!("point: {:?}, value: {}", point, value);
            let _ = out.write_all(
                format!(
                    "{},{},{},{},{},{},{}\n",
                    point.x, point.y, point.z, 5, 5, 10, value
                )
                .as_bytes(),
            );
        }
    }

    #[test]
    fn gsk_ok_block_test() {
        // create a gridded database from a csv file (walker lake)
        println!("Reading Cond Data");
        let cond = SpatialAcceleratedDB::from_csv_index("C:/Users/2jake/OneDrive - McGill University/Fall2022/MIME525/Project4/mineralized_domain_composites.csv", "X", "Y", "Z", "CU")
            .expect("Failed to create gdb");

        //

        let vgram_rot = DRotor3::from_euler_angles(0.00.to_radians(), 0.0, 0.0);
        let range = DVec3 {
            x: 200.0,
            y: 50.0,
            z: 50.0,
        };
        let sill = 1.0;

        let spherical_vgram = CompositeVariogram::new(vec![VariogramType::Spherical(
            SphericalVariogram::new(range, sill, vgram_rot),
        )]);

        // create search ellipsoid
        let search_ellipsoid = Ellipsoid::new(
            200f64,
            50f64,
            50f64,
            CoordinateSystem::new(DVec3::zero(), vgram_rot),
        );

        println!("Reading Target Data");
        let targ = SpatialAcceleratedDB::<f64>::from_csv_index(
            "C:/Users/2jake/OneDrive - McGill University/Fall2022/MIME525/Project4/target.csv",
            "X",
            "Y",
            "Z",
            "V",
        )
        .unwrap();

        let points = targ.supports.iter().map(|s| s.center()).collect_vec();

        //map points in vec of group of points (64)
        //map points in vec of group of points (64)
        let all_blocks = points
            .iter()
            .enumerate()
            .map(|(_i, point)| {
                let aabb = Aabb::from_min_max(
                    DVec3::new(point.x, point.y, point.z),
                    DVec3::new(point.x + 5.0, point.y + 5.0, point.z + 10.0),
                );

                let disc = DVec3::new(5.0, 5.0, 10.0);

                Support::Aabb { aabb, disc }
            })
            .collect::<Vec<_>>();

        let groups = GroupProvider::optimized_groups(&all_blocks, 5f64, 5f64, 10f64, 2, 2, 2);

        // let mut node_provider = PointGroupProvider::from_groups(groups.clone(), orientations);
        let mut params = ConditioningParams::default();
        params.orient_search = true;
        params.orient_variogram = true;
        params.max_n_cond = 36;
        params.valid_value_range = [0.0, f64::INFINITY];

        println!("params: {:#?}", params);
        let time1 = std::time::Instant::now();
        // let builder = SolvedNegativeWeightFilteredSystemBuilder::new(SolvedLUOKSystemBuilder);
        let values = estimate(
            &cond,
            &params,
            &spherical_vgram,
            search_ellipsoid,
            &groups,
            SolvedLUOKSystemBuilder,
        );
        let time2 = std::time::Instant::now();
        println!("Time: {:?}", (time2 - time1).as_secs());
        println!(
            "Points per minute: {}",
            values.len() as f64 / (time2 - time1).as_secs_f64() * 60.0
        );

        // let block_values = values.iter().zip(inds.iter().flatten()).fold(
        //     vec![vec![]; points.len()],
        //     |mut acc, (value, ind)| {
        //         acc[block_inds[*ind]].push(*value);
        //         acc
        //     },
        // );

        // let block_values = (0..groups.n_groups()).fold(vec![vec![]; points.len()], |mut acc, i| {
        //     let group_inds = groups.get_original_idxs(i);
        //     let group_range = groups.get_group_range(i);

        //     for (og_i, val) in group_inds.iter().zip(values[group_range].iter()) {
        //         acc[block_inds[*og_i]].push(*val)
        //     }

        //     acc
        // });

        // let avg_block_values = block_values
        //     .iter()
        //     .map(|x| x.iter().sum::<f64>() / x.len() as f64)
        //     .collect::<Vec<_>>();

        //save values to file for visualization

        let mut out = File::create("./test_results/lu_ok.txt").unwrap();
        let _ = out.write_all(b"surfs\n");
        let _ = out.write_all(b"4\n");
        let _ = out.write_all(b"x\n");
        let _ = out.write_all(b"y\n");
        let _ = out.write_all(b"z\n");
        let _ = out.write_all(b"value\n");

        for (point, value) in groups.get_supports().iter().zip(values.iter()) {
            let point = point.center();
            //println!("point: {:?}, value: {}", point, value);
            let _ = out
                .write_all(format!("{} {} {} {}\n", point.x, point.y, point.z, value).as_bytes());
        }

        let mut out = File::create("./test_results/lu_ok_cond_data.csv").unwrap();

        let _ = out.write(b"x,y,z,v\n");

        for (point, value) in cond.supports.iter().zip(cond.data.iter()) {
            //println!("point: {:?}, value: {}", point, value);
            let point = point.center();
            let _ = out
                .write_all(format!("{},{},{},{}\n", point.x, point.y, point.z, value).as_bytes());
        }

        let mut out = File::create("./test_results/lu_ok.csv").unwrap();
        //write header
        let _ = out.write_all("X,Y,Z,DX,DY,DZ,V\n".as_bytes());

        //write each row

        for (point, value) in groups.get_supports().iter().zip(values.iter()) {
            let point = point.center();
            //println!("point: {:?}, value: {}", point, value);
            let _ = out.write_all(
                format!(
                    "{},{},{},{},{},{},{}\n",
                    point.x, point.y, point.z, 5, 5, 10, value
                )
                .as_bytes(),
            );
        }
    }

    // #[test]
    // fn gsk_ok_point_test() {
    //     // create a gridded database from a csv file (walker lake)
    //     println!("Reading Cond Data");
    //     let cond = PointSet::from_csv_index("C:/Users/2jake/OneDrive - McGill University/Fall2022/MIME525/Project4/mineralized_domain_composites.csv", "X", "Y", "Z", "CU")
    //         .expect("Failed to create gdb");

    //     //

    //     let vgram_rot = UnitQuaternion::from_euler_angles(
    //         WideF64x4::splat(0.0),
    //         WideF64x4::splat(0.0),
    //         WideF64x4::splat(0f64.to_radians()),
    //     );
    //     let range = Vector3::new(
    //         WideF64x4::splat(200.0),
    //         WideF64x4::splat(200.0),
    //         WideF64x4::splat(200.0),
    //     );
    //     let sill = WideF64x4::splat(1.0f64);

    //     let spherical_vgram = CompositeVariogram::new(vec![VariogramType::Spherical(
    //         SphericalVariogram::new(range, sill, vgram_rot),
    //     )]);

    //     // create search ellipsoid
    //     let search_ellipsoid = Ellipsoid::new(
    //         100f64,
    //         100f64,
    //         100f64,
    //         CoordinateSystem::new(
    //             Translation3::new(0.0, 0.0, 0.0),
    //             UnitQuaternion::from_euler_angles(0.0, 0.0, 0f64.to_radians()),
    //         ),
    //     );

    //     // create a gsk system
    //     let parameters = GSKSystemParameters {
    //         max_group_size: 125,
    //     };

    //     let gsk = GSK::new(parameters);
    //     // let gsk = GSK::new(cond.clone(), spherical_vgram, search_ellipsoid, parameters);

    //     println!("Reading Target Data");
    //     let targ = PointSet::<f64>::from_csv_index(
    //         "C:/Users/2jake/OneDrive - McGill University/Fall2022/MIME525/Project4/target.csv",
    //         "X",
    //         "Y",
    //         "Z",
    //         "V",
    //     )
    //     .unwrap();

    //     let points = targ.points.clone();

    //     //map points in vec of group of points (64)
    //     //map points in vec of group of points (64)
    //     let mut block_inds = Vec::new();
    //     let all_points = points
    //         .iter()
    //         .enumerate()
    //         .map(|(i, point)| {
    //             let aabb = Aabb::new(
    //                 Point3::new(point.x, point.y, point.z),
    //                 Point3::new(point.x + 5.0, point.y + 5.0, point.z + 10.0),
    //             );

    //             let disc_points = aabb.discretize(5.0, 5.0, 10.0);
    //             block_inds.append(vec![i; disc_points.len()].as_mut());
    //             disc_points
    //         })
    //         .flatten()
    //         .map(|p| p.cast())
    //         .collect::<Vec<_>>();

    //     let (groups, _inds) = optimize_groups(all_points.as_slice(), 1f64, 1f64, 1f64, 5, 5, 5);

    //     let orientations = vec![UnitQuaternion::identity(); groups.len()];
    //     let node_provider = PointGroupProvider::from_groups(groups.clone(), orientations);
    //     let mut params = ConditioningParams::default();
    //     params.max_n_cond = 24;
    //     params.max_octant = 2;
    //     let builder = SolvedLUOKSystemBuilder;
    //     let time1 = std::time::Instant::now();
    //     let values = gsk.estimate::<SKPointSupportBuilder, _, _, _>(
    //         &cond,
    //         &params,
    //         spherical_vgram,
    //         search_ellipsoid,
    //         &node_provider,
    //         builder,
    //     );
    //     let time2 = std::time::Instant::now();
    //     println!("Time: {:?}", (time2 - time1).as_secs());
    //     println!(
    //         "Points per minute: {}",
    //         values.len() as f64 / (time2 - time1).as_secs_f64() * 60.0
    //     );

    //     let mut out = File::create("./test_results/lu_ok_point.txt").unwrap();
    //     let _ = out.write_all(b"x,y,z,v\n");
    //     let _ = out.write_all(b"4\n");
    //     let _ = out.write_all(b"x\n");
    //     let _ = out.write_all(b"y\n");
    //     let _ = out.write_all(b"z\n");
    //     let _ = out.write_all(b"value\n");

    //     for (point, value) in groups.iter().flatten().zip(values.iter()) {
    //         //println!("point: {:?}, value: {}", point, value);
    //         let _ = out
    //             .write_all(format!("{} {} {} {}\n", point.x, point.y, point.z, value).as_bytes());
    //     }

    //     let mut out = File::create("./test_results/lu_ok_cond_data.txt").unwrap();
    //     let _ = out.write_all(b"x,y,z,v\n");

    //     for (point, value) in cond.points.iter().zip(cond.data.iter()) {
    //         //println!("point: {:?}, value: {}", point, value);
    //         let _ = out
    //             .write_all(format!("{} {} {} {}\n", point.x, point.y, point.z, value).as_bytes());
    //     }

    //     let mut out = File::create("./test_results/lu_ok_point.csv").unwrap();
    //     //write header
    //     let _ = out.write_all("X,Y,Z,DX,DY,DZ,V\n".as_bytes());

    //     //write each row

    //     for (point, value) in groups.iter().flatten().zip(values.iter()) {
    //         //println!("point: {:?}, value: {}", point, value);
    //         let _ = out.write_all(
    //             format!(
    //                 "{},{},{},{},{},{},{}\n",
    //                 point.x, point.y, point.z, 5, 5, 10, value
    //             )
    //             .as_bytes(),
    //         );
    //     }
    // }

    // #[test]
    // fn gsk_sk_test() {
    //     // create a gridded database from a csv file (walker lake)
    //     println!("Reading Cond Data");
    //     let mut cond = PointSet::from_csv_index("C:/Users/2jake/OneDrive - McGill University/Fall2022/MIME525/Project4/mineralized_domain_composites.csv", "X", "Y", "Z", "CU")
    //         .expect("Failed to create gdb");

    //     //

    //     let (mean, std_dev) = cond.normalize();
    //     let mean = cond.data.iter().sum::<f64>() / cond.data.len() as f64;

    //     let vgram_rot = UnitQuaternion::from_euler_angles(
    //         WideF64x4::splat(0.0),
    //         WideF64x4::splat(0.0),
    //         WideF64x4::splat(0f64.to_radians()),
    //     );
    //     let range = Vector3::new(
    //         WideF64x4::splat(50.0),
    //         WideF64x4::splat(200.0),
    //         WideF64x4::splat(50.0),
    //     );
    //     let sill = WideF64x4::splat(1.0f64);

    //     let spherical_vgram = CompositeVariogram::new(vec![VariogramType::Spherical(
    //         SphericalVariogram::new(range, sill, vgram_rot),
    //     )]);

    //     // create search ellipsoid
    //     let search_ellipsoid = Ellipsoid::new(
    //         50f64,
    //         200f64,
    //         50f64,
    //         CoordinateSystem::new(
    //             Translation3::new(0.0, 0.0, 0.0),
    //             UnitQuaternion::from_euler_angles(0.0, 0.0, 0f64.to_radians()),
    //         ),
    //     );

    //     // create a gsk system
    //     let parameters = GSKSystemParameters { max_group_size: 8 };

    //     let gsk = GSK::new(parameters);
    //     // let gsk = GSK::new(cond.clone(), spherical_vgram, search_ellipsoid, parameters);

    //     println!("Reading Target Data");
    //     let targ = PointSet::<f64>::from_csv_index(
    //         "C:/Users/2jake/OneDrive - McGill University/Fall2022/MIME525/Project4/target.csv",
    //         "X",
    //         "Y",
    //         "Z",
    //         "V",
    //     )
    //     .unwrap();

    //     let points = targ.points.clone();

    //     //map points in vec of group of points (64)
    //     //map points in vec of group of points (64)
    //     let mut block_inds = Vec::new();
    //     let all_points = points
    //         .iter()
    //         .enumerate()
    //         .map(|(i, point)| {
    //             let aabb = Aabb::new(
    //                 Point3::new(point.x, point.y, point.z),
    //                 Point3::new(point.x + 5.0, point.y + 5.0, point.z + 10.0),
    //             );

    //             let disc_points = aabb.discretize(5.0, 5.0, 10.0);
    //             block_inds.append(vec![i; disc_points.len()].as_mut());
    //             disc_points
    //         })
    //         .flatten()
    //         .map(|p| p.cast())
    //         .collect::<Vec<_>>();

    //     let (groups, inds) = optimize_groups(all_points.as_slice(), 5f64, 5f64, 10f64, 2, 2, 2);

    //     let orientations = groups
    //         .iter()
    //         .map(|group| {
    //             UnitQuaternion::from_euler_angles(0.0, 0.0, (group.center().x * 0.1).to_radians())
    //         })
    //         .collect::<Vec<_>>();

    //     //write orientations to fileq
    //     let mut csv_str = "x,y,z,ang1,ang2,ang3".to_string();
    //     for (point, orientation) in groups.iter().zip(orientations.iter()) {
    //         csv_str.push_str(&format!(
    //             "{},{},{},{},{},{}\n",
    //             point.center().x,
    //             point.center().y,
    //             point.center().z,
    //             orientation.euler_angles().0,
    //             orientation.euler_angles().1,
    //             orientation.euler_angles().2
    //         ));
    //     }

    //     let _ = std::fs::write("./test_results/lu_sk_orientations.csv", csv_str);

    //     let mut node_provider = PointGroupProvider::from_groups(groups.clone(), orientations);
    //     let mut params = ConditioningParams::default();
    //     params.orient_search = true;
    //     params.orient_variogram = true;
    //     params.max_n_cond = 12;
    //     // params.min_n_cond = 0;
    //     params.valid_value_range = [0.0, f64::INFINITY];

    //     println!("params: {:#?}", params);
    //     let time1 = std::time::Instant::now();
    //     // let builder = SolvedNegativeWeightFilteredSystemBuilder::new(SolvedLUOKSystemBuilder);
    //     //let value_transform = MeanTransfrom::new(mean);
    //     // let builder = ModifiedSolvedLUSystemBuilder::new(SolvedLUSKSystemBuilder, value_transform);
    //     // let builder = SolvedLUSKSystemBuilder;
    //     let mut values = gsk.simulate::<SKPointSupportBuilder, _, _, _>(
    //         &cond,
    //         &params,
    //         &params,
    //         spherical_vgram,
    //         search_ellipsoid,
    //         &mut node_provider,
    //         SolvedLUSKSystemBuilder,
    //     );

    //     values.iter_mut().for_each(|v| *v = *v * std_dev + mean);
    //     let time2 = std::time::Instant::now();
    //     println!("Time: {:?}", (time2 - time1).as_secs());
    //     println!(
    //         "Points per minute: {}",
    //         values.len() as f64 / (time2 - time1).as_secs_f64() * 60.0
    //     );

    //     let block_values = values.iter().zip(inds.iter().flatten()).fold(
    //         vec![vec![]; points.len()],
    //         |mut acc, (value, ind)| {
    //             acc[block_inds[*ind]].push(*value);
    //             acc
    //         },
    //     );

    //     let avg_block_values = block_values
    //         .iter()
    //         .map(|x| x.iter().sum::<f64>() / x.len() as f64)
    //         .collect::<Vec<_>>();
    //     //save values to file for visualization

    //     let mut out = File::create("./test_results/lu_sk.txt").unwrap();
    //     let _ = out.write_all(b"surfs\n");
    //     let _ = out.write_all(b"4\n");
    //     let _ = out.write_all(b"x\n");
    //     let _ = out.write_all(b"y\n");
    //     let _ = out.write_all(b"z\n");
    //     let _ = out.write_all(b"value\n");

    //     for (point, value) in points.iter().zip(avg_block_values.iter()) {
    //         //println!("point: {:?}, value: {}", point, value);
    //         let _ = out
    //             .write_all(format!("{} {} {} {}\n", point.x, point.y, point.z, value).as_bytes());
    //     }

    //     let mut out = File::create("./test_results/lu_sk_cond_data.csv").unwrap();

    //     let _ = out.write(b"x,y,z,v\n");

    //     for (point, value) in cond.points.iter().zip(cond.data.iter()) {
    //         //println!("point: {:?}, value: {}", point, value);
    //         let _ = out
    //             .write_all(format!("{},{},{},{}\n", point.x, point.y, point.z, value).as_bytes());
    //     }

    //     let mut out = File::create("./test_results/lu_sk.csv").unwrap();
    //     //write header
    //     let _ = out.write_all("X,Y,Z,DX,DY,DZ,V\n".as_bytes());

    //     //write each row

    //     for (point, value) in points.iter().zip(avg_block_values.iter()) {
    //         //println!("point: {:?}, value: {}", point, value);
    //         let _ = out.write_all(
    //             format!(
    //                 "{},{},{},{},{},{},{}\n",
    //                 point.x, point.y, point.z, 5, 5, 10, value
    //             )
    //             .as_bytes(),
    //         );
    //     }
    // }

    // #[test]
    // fn gsk_ok_db_test() {
    //     // create a gridded database from a csv file (walker lake)
    //     println!("Reading Cond Data");
    //     let cond = PointSet::from_csv_index("C:/Users/2jake/OneDrive - McGill University/Fall2022/MIME525/Project4/mineralized_domain_composites.csv", "X", "Y", "Z", "CU")
    //         .expect("Failed to create gdb");

    //     //

    //     let vgram_rot = UnitQuaternion::identity();
    //     let range = Vector3::new(
    //         WideF32x8::splat(200.0),
    //         WideF32x8::splat(200.0),
    //         WideF32x8::splat(200.0),
    //     );
    //     let sill = WideF32x8::splat(1.0f32);

    //     let spherical_vgram = SphericalVariogram::new(range, sill, vgram_rot);

    //     // create search ellipsoid
    //     let search_ellipsoid = Ellipsoid::new(
    //         200f32,
    //         200f32,
    //         200f32,
    //         CoordinateSystem::new(Translation3::new(0.0, 0.0, 0.0), UnitQuaternion::identity()),
    //     );

    //     // create a gsk system
    //     let group_size = 10;
    //     let parameters = GSKSystemParameters {
    //         max_group_size: group_size,
    //     };

    //     let gsk = GSK::new(parameters);
    //     // let gsk = GSK::new(cond.clone(), spherical_vgram, search_ellipsoid, parameters);

    //     println!("Reading Target Data");
    //     let targ = PointSet::<f32>::from_csv_index(
    //         "C:/Users/2jake/OneDrive - McGill University/Fall2022/MIME525/Project4/target.csv",
    //         "X",
    //         "Y",
    //         "Z",
    //         "V",
    //     )
    //     .unwrap();

    //     let points = targ.points.clone();

    //     //map points in vec of group of points (64)
    //     let mut groups = Vec::new();
    //     let mut group = Vec::new();
    //     for (i, point) in points.iter().enumerate() {
    //         //iterate over 5x5x10 grid originating at point
    //         let mut block = Vec::new();
    //         for x in 0..5 {
    //             for y in 0..5 {
    //                 for z in 0..10 {
    //                     block.push(Point3::new(
    //                         point.x + x as f32,
    //                         point.y + y as f32,
    //                         point.z + z as f32,
    //                     ));
    //                 }
    //             }
    //         }
    //         group.push(block);

    //         if (i % group_size - 1 == 0 && i != 0) || i == points.len() - 1 {
    //             groups.push(group.clone());
    //             group.clear();
    //         }
    //     }

    //     let node_provider = VolumeGroupProvider::from_groups(
    //         groups.clone(),
    //         vec![UnitQuaternion::identity(); groups.len()],
    //     );
    //     let time1 = std::time::Instant::now();
    //     let values = gsk.estimate::<SKVolumeSupportBuilder, SolvedLUOKSystem, _, _, _>(
    //         &cond,
    //         &Default::default(),
    //         spherical_vgram,
    //         search_ellipsoid,
    //         &node_provider,
    //     );
    //     let time2 = std::time::Instant::now();
    //     println!("Time: {:?}", (time2 - time1).as_secs());
    //     println!(
    //         "Points per minute: {}",
    //         values.len() as f32 / (time2 - time1).as_secs_f32() * 60.0
    //     );

    //     //save values to file for visualization

    //     let mut out = File::create("./test_results/lu_ok_db.txt").unwrap();
    //     let _ = out.write_all(b"surfs\n");
    //     let _ = out.write_all(b"4\n");
    //     let _ = out.write_all(b"x\n");
    //     let _ = out.write_all(b"y\n");
    //     let _ = out.write_all(b"z\n");
    //     let _ = out.write_all(b"value\n");

    //     for (point, value) in points.iter().zip(values.iter()) {
    //         //println!("point: {:?}, value: {}", point, value);
    //         let _ = out
    //             .write_all(format!("{} {} {} {}\n", point.x, point.y, point.z, value).as_bytes());
    //     }

    //     let mut out = File::create("./test_results/lu_ok_db_cond_data.txt").unwrap();
    //     let _ = out.write_all(b"surfs\n");
    //     let _ = out.write_all(b"4\n");
    //     let _ = out.write_all(b"x\n");
    //     let _ = out.write_all(b"y\n");
    //     let _ = out.write_all(b"z\n");
    //     let _ = out.write_all(b"value\n");

    //     for (point, value) in cond.points.iter().zip(cond.data.iter()) {
    //         //println!("point: {:?}, value: {}", point, value);
    //         let _ = out
    //             .write_all(format!("{} {} {} {}\n", point.x, point.y, point.z, value).as_bytes());
    //     }

    //     let mut out = File::create("./test_results/lu_ok_db.csv").unwrap();
    //     //write header
    //     let _ = out.write_all("X,Y,Z,XS,YS,ZS,V\n".as_bytes());

    //     //write each row

    //     for (point, value) in points.iter().zip(values.iter()) {
    //         //println!("point: {:?}, value: {}", point, value);
    //         let _ = out.write_all(
    //             format!(
    //                 "{},{},{},{},{},{},{}\n",
    //                 point.x, point.y, point.z, 5, 5, 10, value
    //             )
    //             .as_bytes(),
    //         );
    //     }
    // }
}
