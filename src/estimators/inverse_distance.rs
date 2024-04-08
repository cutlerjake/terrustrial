use itertools::Itertools;
use nalgebra::{Point3, Vector3};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::geometry::Geometry;
use crate::spatial_database::SupportInterface;
use crate::spatial_database::SupportTransform;
use crate::{
    geometry::ellipsoid::Ellipsoid,
    node_providers::NodeProvider,
    spatial_database::{coordinate_system::CoordinateSystem, ConditioningProvider},
};

use super::ConditioningParams;

pub struct InverseDistance;

impl InverseDistance {
    #[allow(clippy::too_many_arguments)]
    pub fn estimate<S, NP>(
        &self,
        conditioning_data: &S,
        conditioning_params: &ConditioningParams,
        coordinate_system: &CoordinateSystem<f32>,
        scaling: Vector3<f32>,
        search_ellipsoid: &Ellipsoid,
        groups: &NP,
        power: f32,
    ) -> Vec<f32>
    where
        S: ConditioningProvider<Ellipsoid, f32, ConditioningParams> + Sync + std::marker::Send,
        S::Shape: SupportTransform<NP::Support>,
        NP: NodeProvider + Sync,
        <NP as NodeProvider>::Support: SupportInterface,
    {
        (0..groups.n_groups())
            .into_par_iter()
            .map(|group| (groups.get_group(group), groups.get_orientation(group)))
            .map_with(
                (
                    conditioning_params,
                    coordinate_system.clone(),
                    search_ellipsoid.clone(),
                ),
                |(cond_params, cs, ellipsoid), (group, orientation)| {
                    // Determine the center of the group.
                    let center = group.iter().fold(Point3::<f32>::origin(), |mut acc, x| {
                        acc.coords += x.center().coords;
                        acc
                    }) / (group.len() as f32);

                    // Get the conditioning data for the group.
                    ellipsoid.translate_to(&center);

                    if cond_params.orient_search {
                        ellipsoid.coordinate_system.set_rotation(*orientation);
                    }

                    // Get nearest points and values.
                    let (_, cond_values, cond_points, sufficiently_conditioned) =
                        conditioning_data.query(&center, ellipsoid, conditioning_params);

                    if sufficiently_conditioned {
                        // Convert points to correct support
                        let cond_points = cond_points
                            .into_iter()
                            .map(|x| x.transform())
                            .collect::<Vec<_>>();

                        // Compute scaled distances to each node in group
                        let weights = group
                            .iter()
                            .cartesian_product(cond_points.iter())
                            .map(|(node, cond_point)| {
                                // Distance between node to be estimated and conditioning point in world space.
                                let distance = node.center().coords - cond_point.center().coords;

                                // Convert distance to local space.
                                let distance = cs.world_to_local_vector(&distance);

                                // Scale distance.
                                let distance = distance.component_div(&scaling);

                                // Convert distance back to world space.
                                let distance = distance.norm();

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
                                    .sum::<f32>()
                                    / chunk.iter().sum::<f32>()
                            })
                            .collect()
                    } else {
                        vec![f32::NAN; group.len()]
                    }
                },
            )
            .flatten()
            .collect::<Vec<_>>()
    }
}

#[cfg(test)]
mod test {
    use std::{fs::File, io::Write};

    use nalgebra::{Translation3, UnitQuaternion};
    use parry3d::bounding_volume::Aabb;

    use crate::{
        estimators::generalized_sequential_kriging::optimize_groups,
        node_providers::point_group::PointGroupProvider,
        spatial_database::{rtree_point_set::point_set::PointSet, DiscretiveVolume},
    };

    use super::*;

    #[test]
    fn inv_dist() {
        // create a gridded database from a csv file (walker lake)
        println!("Reading Cond Data");
        let cond: PointSet<f32> = PointSet::from_csv_index("C:/Users/2jake/OneDrive - McGill University/Fall2022/MIME525/Project4/mineralized_domain_composites.csv", "X", "Y", "Z", "CU")
            .expect("Failed to create gdb");

        // create search ellipsoid
        let search_ellipsoid = Ellipsoid::new(
            50f32,
            200f32,
            50f32,
            CoordinateSystem::new(
                Translation3::new(0.0, 0.0, 0.0),
                UnitQuaternion::from_euler_angles(0.0, 0.0, 0f32.to_radians()),
            ),
        );

        println!("Reading Target Data");
        let targ = PointSet::<f32>::from_csv_index(
            "C:/Users/2jake/OneDrive - McGill University/Fall2022/MIME525/Project4/target.csv",
            "X",
            "Y",
            "Z",
            "V",
        )
        .unwrap();

        let points = targ.points.clone();

        //map points in vec of group of points (64)
        //map points in vec of group of points (64)
        let mut block_inds = Vec::new();
        let all_points = points
            .iter()
            .enumerate()
            .map(|(i, point)| {
                let aabb = Aabb::new(
                    Point3::new(point.x, point.y, point.z),
                    Point3::new(point.x + 5.0, point.y + 5.0, point.z + 10.0),
                );

                let disc_points = aabb.discretize(5f32, 5f32, 10f32);
                block_inds.append(vec![i; disc_points.len()].as_mut());
                disc_points
            })
            .flatten()
            .collect::<Vec<_>>();

        let (groups, inds) = optimize_groups(all_points.as_slice(), 5f32, 5f32, 10f32, 2, 2, 2);

        let orientations = groups
            .iter()
            .map(|group| {
                UnitQuaternion::from_euler_angles(0.0, 0.0, (group.center().x * 0.1).to_radians())
            })
            .collect::<Vec<_>>();

        //write orientations to fileq
        let mut csv_str = "x,y,z,ang1,ang2,ang3".to_string();
        for (point, orientation) in groups.iter().zip(orientations.iter()) {
            csv_str.push_str(&format!(
                "{},{},{},{},{},{}\n",
                point.center().x,
                point.center().y,
                point.center().z,
                orientation.euler_angles().0,
                orientation.euler_angles().1,
                orientation.euler_angles().2
            ));
        }

        let _ = std::fs::write("./test_results/inv_orientations.csv", csv_str);

        let node_provider = PointGroupProvider::from_groups(groups.clone(), orientations);
        let mut params = ConditioningParams::default();
        params.orient_search = true;
        params.orient_variogram = true;
        params.max_n_cond = 12;
        params.valid_value_range = [0.0, f32::INFINITY];

        println!("params: {:#?}", params);
        let est = InverseDistance;
        let time1 = std::time::Instant::now();
        // let builder = SolvedNegativeWeightFilteredSystemBuilder::new(SolvedLUOKSystemBuilder);
        let values = est.estimate::<_, _>(
            &cond,
            &params,
            &CoordinateSystem::new(
                Translation3::new(0.0, 0.0, 0.0),
                UnitQuaternion::from_euler_angles(0.0, 0.0, 0f32.to_radians()),
            ),
            Vector3::new(1.0, 1.0, 1.0),
            &search_ellipsoid,
            &node_provider,
            2.0,
        );
        let time2 = std::time::Instant::now();
        println!("Time: {:?}", (time2 - time1).as_secs());
        println!(
            "Points per minute: {}",
            values.len() as f32 / (time2 - time1).as_secs_f32() * 60.0
        );

        let block_values = values.iter().zip(inds.iter().flatten()).fold(
            vec![vec![]; points.len()],
            |mut acc, (value, ind)| {
                acc[block_inds[*ind]].push(*value);
                acc
            },
        );

        let avg_block_values = block_values
            .iter()
            .map(|x| x.iter().sum::<f32>() / x.len() as f32)
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
