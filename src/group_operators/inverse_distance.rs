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

#[cfg(test)]
mod test {
    use std::{fs::File, io::Write};

    use itertools::Itertools;
    use num_traits::Float;
    use ultraviolet::DRotor3;

    use crate::{
        geometry::{aabb::Aabb, support::Support},
        spatial_database::{
            coordinate_system::CoordinateSystem, DiscretiveVolume, SpatialAcceleratedDB,
        },
    };

    use super::*;

    #[test]
    fn inv_dist() {
        let cond = SpatialAcceleratedDB::from_csv_index("C:/Users/2jake/OneDrive - McGill University/Fall2022/MIME525/Project4/mineralized_domain_composites.csv", "X", "Y", "Z", "CU")
            .expect("Failed to create gdb");

        //

        let rot = DRotor3::from_euler_angles(0.00.to_radians(), 0.0, 0.0);

        // create search ellipsoid
        let search_ellipsoid = Ellipsoid::new(
            200f64,
            50f64,
            50f64,
            CoordinateSystem::new(DVec3::zero(), rot),
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
        let values = estimate(
            &cond,
            &params,
            &CoordinateSystem::new(DVec3::zero(), DRotor3::identity()),
            DVec3::new(1.0, 1.0, 1.0),
            &search_ellipsoid,
            &groups,
            2.0,
        );
        let time2 = std::time::Instant::now();
        println!("Time: {:?}", (time2 - time1).as_secs());
        println!(
            "Points per minute: {}",
            values.len() as f32 / (time2 - time1).as_secs_f32() * 60.0
        );

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

        let mut out = File::create("./test_results/inv_dist.csv").unwrap();
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
}
