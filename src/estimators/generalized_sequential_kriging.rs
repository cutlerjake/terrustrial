use crate::geometry::ellipsoid::Ellipsoid;
use crate::geometry::Geometry;
use crate::node_providers::NodeProvider;
use crate::spatial_database::rtree_point_set::point_set::PointSet;
use crate::spatial_database::rtree_point_set::support_set::SupportSet;
use crate::spatial_database::ConditioningProvider;
use crate::spatial_database::FilteredIterNearest;
use crate::spatial_database::IterNearest;
use crate::spatial_database::SupportInterface;
use crate::spatial_database::SupportTransform;
use crate::systems::lu::LUSystem;
use crate::systems::solved_systems::SolvedLUSystem;
use crate::systems::solved_systems::SolvedSystemBuilder;
use crate::variography::model_variograms::VariogramModel;
use nalgebra::Point3;
use nalgebra::UnitQuaternion;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rstar::primitives::GeomWithData;
use rstar::RTree;
use rstar::RTreeObject;
use rstar::AABB;
use simba::simd::SimdPartialOrd;
use simba::simd::SimdRealField;
use simba::simd::SimdValue;

use super::simple_kriging::SKBuilder;
use crate::estimators::ConditioningParams;
use rayon::prelude::*;

#[derive(Clone, Copy, Debug)]
pub struct GSKSystemParameters {
    pub max_group_size: usize,
}

pub struct GSK {
    system_params: GSKSystemParameters,
}

impl GSK {
    pub fn new(system_params: GSKSystemParameters) -> Self {
        Self { system_params }
    }

    pub fn estimate<SKB, S, V, VT>(
        &self,
        conditioning_data: &S,
        conditioning_params: &ConditioningParams,
        variogram_model: V,
        search_ellipsoid: Ellipsoid,
        groups: &(impl NodeProvider<Support = SKB::Support> + Sync),
        solved_system: impl SolvedSystemBuilder,
    ) -> Vec<f32>
    where
        SKB: SKBuilder,
        V: VariogramModel<VT> + std::marker::Sync,
        VT: SimdPartialOrd + SimdRealField + SimdValue<Element = f32> + Copy,
        S: ConditioningProvider<Ellipsoid, f32, ConditioningParams> + Send + Sync,
        <S as IterNearest>::Shape: SupportTransform<SKB::Support>,
    {
        let system = LUSystem::new(
            self.system_params.max_group_size,
            conditioning_params.max_n_cond,
        );

        groups
            .groups_and_orientations()
            .map_with(
                (
                    system.clone(),
                    solved_system.clone(),
                    search_ellipsoid.clone(),
                    variogram_model.clone(),
                ),
                |(local_system, local_solved_system, ellipsoid, vgram), (group, orientation)| {
                    //get center of group
                    let center = group.iter().fold(Point3::<f32>::origin(), |mut acc, x| {
                        acc.coords += x.center().coords;
                        acc
                    }) / (group.len() as f32);

                    //translate search ellipsoid to group center
                    ellipsoid.translate_to(&center);

                    //orient ellipsoid
                    if conditioning_params.orient_search {
                        ellipsoid.coordinate_system.set_rotation(orientation);
                    }

                    //orient variogram
                    if conditioning_params.orient_variogram {
                        vgram.set_orientation(UnitQuaternion::splat(orientation));
                    }

                    //get nearest points and values
                    let (_, cond_values, cond_points, sufficiently_conditioned) =
                        conditioning_data.query(&center, ellipsoid, conditioning_params);

                    if sufficiently_conditioned {
                        //convert points to support
                        let cond_points = cond_points
                            .into_iter()
                            .map(|x| x.transform())
                            .collect::<Vec<_>>();

                        //build kriging system for point
                        local_system.build_cov_matrix::<_, _, SKB>(&cond_points, group, vgram);
                        if let Ok(mut solved_system) = local_solved_system.build(local_system) {
                            solved_system.populate_cond_values_est(cond_values.as_slice());
                            return solved_system.estimate();
                        }
                    }
                    vec![f32::NAN; group.len()]
                },
            )
            .flatten()
            .collect::<Vec<f32>>()
    }

    pub fn simulate<SKB, S, V, VT>(
        &self,
        conditioning_data: &S,
        cond_conditioning_params: &ConditioningParams,
        sim_conditioning_params: &ConditioningParams,
        variogram_model: V,
        search_ellipsoid: Ellipsoid,
        groups: &mut (impl NodeProvider<Support = SKB::Support> + Sync),
        solved_system: impl SolvedSystemBuilder,
    ) -> Vec<f32>
    where
        SKB: SKBuilder,
        V: VariogramModel<VT> + std::marker::Sync,
        VT: SimdPartialOrd + SimdRealField + SimdValue<Element = f32> + Copy,
        S: ConditioningProvider<Ellipsoid, f32, ConditioningParams> + Send + Sync,
        <S as IterNearest>::Shape: SupportTransform<SKB::Support>,
        <SKB as SKBuilder>::Support: Clone,
    {
        pub enum Action<T> {
            Simulate(T),
            Nullify(u32),
        }

        let system = LUSystem::new(
            self.system_params.max_group_size,
            cond_conditioning_params.max_n_cond + sim_conditioning_params.max_n_cond,
        );

        let mut rng = StdRng::from_entropy();
        // 1. Randomize the order of the groups
        groups.randomize(&mut rng);

        // 2. Build a conditioning provider from the simulation locations.
        let _data = vec![0.0f32; groups.n_nodes()];
        let mut _tags = vec![];
        let mut sim_step = 0;
        let _points = (0..groups.n_groups())
            .map(|i| {
                let group = groups.get_group(i);
                _tags.extend(std::iter::repeat(sim_step).take(group.len()));
                sim_step += 1;
                group
            })
            .flatten()
            .cloned()
            .collect::<Vec<_>>();

        let sim_conditioning_data = SupportSet::new(_points, _data, _tags);

        let actions = groups
            .indexed_groups_and_orientations()
            .map_with(
                (
                    system.clone(),
                    solved_system.clone(),
                    search_ellipsoid.clone(),
                    variogram_model.clone(),
                ),
                |(local_system, local_solved_system, ellipsoid, vgram),
                 (sim_idx, group, orientation)| {
                    //get center of group
                    let center = group.iter().fold(Point3::<f32>::origin(), |mut acc, x| {
                        acc.coords += x.center().coords;
                        acc
                    }) / (group.len() as f32);

                    //translate search ellipsoid to group center
                    ellipsoid.translate_to(&center);

                    //orient ellipsoid
                    if cond_conditioning_params.orient_search {
                        ellipsoid.coordinate_system.set_rotation(orientation);
                    }

                    //orient variogram
                    if cond_conditioning_params.orient_variogram {
                        vgram.set_orientation(UnitQuaternion::splat(orientation));
                    }

                    //get nearest points and values
                    let (_, cond_values, cond_points, sufficiently_conditioned) =
                        conditioning_data.query(&center, ellipsoid, cond_conditioning_params);

                    // Create a query that only fetches previously simulated nodes.
                    let filtered_sim_conditioning_data =
                        FilteredIterNearest::new(&sim_conditioning_data, |x| {
                            // Only fetch nodes that are not already simulated.
                            x.tag < sim_idx as u32
                        });

                    let (sim_cond_idx, _, sim_cond_points, _sufficiently_conditioned) =
                        filtered_sim_conditioning_data.query(
                            &center,
                            ellipsoid,
                            cond_conditioning_params,
                        );

                    if sufficiently_conditioned {
                        //convert points to support
                        let mut cond_points = cond_points
                            .into_iter()
                            .map(|x| x.transform())
                            .collect::<Vec<_>>();

                        cond_points.extend(sim_cond_points.into_iter().map(|x| x.transform()));

                        //build kriging system for point
                        local_system.build_cov_matrix::<_, _, SKB>(&cond_points, group, vgram);
                        if let Ok(solved_system) = local_solved_system.build(local_system) {
                            return Action::Simulate((solved_system, sim_cond_idx, cond_values));
                        }
                    }
                    Action::Nullify(group.len() as u32)
                },
            )
            .collect::<Vec<_>>();

        let mut simulated_values = vec![f32::NAN; groups.n_nodes()];

        let mut cnt = 0;

        for action in actions {
            match action {
                Action::Simulate((mut solved_system, sim_cond_idx, mut cond_values)) => {
                    // Populate the conditioning values for the simulation.

                    cond_values.extend(sim_cond_idx.iter().map(|x| simulated_values[*x as usize]));
                    solved_system.populate_cond_values_sim(cond_values.as_slice(), &mut rng);
                    let simulated = solved_system.simulate();
                    let n_simulated = simulated.len();
                    // Populate the simulated values in the group.
                    for (i, value) in simulated.into_iter().enumerate() {
                        simulated_values[i + cnt] = value;
                    }
                    cnt += n_simulated;
                }
                Action::Nullify(size) => {
                    cnt += size as usize;
                }
            }
        }

        simulated_values
    }
}

pub fn optimize_groups(
    points: &[Point3<f32>],
    dx: f32,
    dy: f32,
    dz: f32,
    gx: usize,
    gy: usize,
    gz: usize,
) -> (Vec<Vec<Point3<f32>>>, Vec<Vec<usize>>) {
    let mut target_point_tree = RTree::bulk_load(
        points
            .iter()
            .enumerate()
            .map(|(i, point)| GeomWithData::<[f32; 3], usize>::new([point.x, point.y, point.z], i))
            .collect(),
    );
    let bounds = target_point_tree.root().envelope();

    let group_size = [gx, gy, gz];
    let env_size = [
        group_size[0] as f32 * dx,
        group_size[1] as f32 * dy,
        group_size[2] as f32 * dz,
    ];
    let mut groups = Vec::new();
    let mut inds = Vec::new();
    let mut x = bounds.lower()[0];
    while x <= bounds.upper()[0] {
        let mut y = bounds.lower()[1];
        while y <= bounds.upper()[1] {
            let mut z = bounds.lower()[2];
            while z <= bounds.upper()[2] {
                let envelope = AABB::from_corners(
                    [x, y, z],
                    [x + env_size[0], y + env_size[1], z + env_size[2]],
                );

                let mut points = target_point_tree
                    .drain_in_envelope_intersecting(envelope)
                    .collect::<Vec<_>>();

                //sort points by x, y, z
                points.sort_by(|a, b| {
                    a.envelope()
                        .lower()
                        .partial_cmp(&b.envelope().lower())
                        .unwrap()
                });

                points
                    .chunks(group_size.iter().product())
                    .for_each(|chunk| {
                        groups.push(
                            chunk
                                .iter()
                                .map(|geom| {
                                    Point3::new(
                                        geom.envelope().lower()[0],
                                        geom.envelope().lower()[1],
                                        geom.envelope().lower()[2],
                                    )
                                })
                                .collect::<Vec<_>>(),
                        );

                        inds.push(chunk.iter().map(|geom| geom.data).collect::<Vec<_>>());
                    });

                z += env_size[2];
            }
            y += env_size[1]
        }
        x += env_size[0]
    }
    (groups, inds)
}

#[cfg(test)]
mod test {
    use std::{fs::File, io::Write};

    use nalgebra::{Translation3, UnitQuaternion, Vector3};
    use parry3d::bounding_volume::Aabb;
    use simba::simd::WideF32x8;

    use crate::{
        estimators::simple_kriging::SKPointSupportBuilder,
        node_providers::point_group::PointGroupProvider,
        spatial_database::{
            coordinate_system::CoordinateSystem, rtree_point_set::point_set::PointSet,
            DiscretiveVolume,
        },
        systems::{
            modifiers::{mean_transform::MeanTransfrom, ModifiedSolvedLUSystemBuilder},
            solved_systems::{
                ok_system::SolvedLUOKSystemBuilder, sk_system::SolvedLUSKSystemBuilder,
            },
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
        let cond = PointSet::from_csv_index("C:/Users/2jake/OneDrive - McGill University/Fall2022/MIME525/Project4/mineralized_domain_composites.csv", "X", "Y", "Z", "CU")
            .expect("Failed to create gdb");

        //

        let vgram_rot = UnitQuaternion::from_euler_angles(
            WideF32x8::splat(0.0),
            WideF32x8::splat(0.0),
            WideF32x8::splat(0f32.to_radians()),
        );
        let range = Vector3::new(
            WideF32x8::splat(50.0),
            WideF32x8::splat(200.0),
            WideF32x8::splat(50.0),
        );
        let sill = WideF32x8::splat(1.0f32);

        let spherical_vgram = CompositeVariogram::new(vec![VariogramType::Spherical(
            SphericalVariogram::new(range, sill, vgram_rot),
        )]);

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

        // create a gsk system
        let parameters = GSKSystemParameters { max_group_size: 8 };

        let gsk = GSK::new(parameters);
        // let gsk = GSK::new(cond.clone(), spherical_vgram, search_ellipsoid, parameters);

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

        let _ = std::fs::write("./test_results/lu_ok_orientations.csv", csv_str);

        let node_provider = PointGroupProvider::from_groups(groups.clone(), orientations);
        let mut params = ConditioningParams::default();
        params.orient_search = true;
        params.orient_variogram = true;
        params.max_n_cond = 12;
        params.valid_value_range = [0.0, f32::INFINITY];

        println!("params: {:#?}", params);
        let time1 = std::time::Instant::now();
        // let builder = SolvedNegativeWeightFilteredSystemBuilder::new(SolvedLUOKSystemBuilder);
        let builder = SolvedLUOKSystemBuilder;
        let values = gsk.estimate::<SKPointSupportBuilder, _, _, _>(
            &cond,
            &params,
            spherical_vgram,
            search_ellipsoid,
            &node_provider,
            builder,
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

        for (point, value) in cond.points.iter().zip(cond.data.iter()) {
            //println!("point: {:?}, value: {}", point, value);
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
    fn gsk_ok_point_test() {
        // create a gridded database from a csv file (walker lake)
        println!("Reading Cond Data");
        let cond = PointSet::from_csv_index("C:/Users/2jake/OneDrive - McGill University/Fall2022/MIME525/Project4/mineralized_domain_composites.csv", "X", "Y", "Z", "CU")
            .expect("Failed to create gdb");

        //

        let vgram_rot = UnitQuaternion::from_euler_angles(
            WideF32x8::splat(0.0),
            WideF32x8::splat(0.0),
            WideF32x8::splat(0f32.to_radians()),
        );
        let range = Vector3::new(
            WideF32x8::splat(200.0),
            WideF32x8::splat(200.0),
            WideF32x8::splat(200.0),
        );
        let sill = WideF32x8::splat(1.0f32);

        let spherical_vgram = CompositeVariogram::new(vec![VariogramType::Spherical(
            SphericalVariogram::new(range, sill, vgram_rot),
        )]);

        // create search ellipsoid
        let search_ellipsoid = Ellipsoid::new(
            100f32,
            100f32,
            100f32,
            CoordinateSystem::new(
                Translation3::new(0.0, 0.0, 0.0),
                UnitQuaternion::from_euler_angles(0.0, 0.0, 0f32.to_radians()),
            ),
        );

        // create a gsk system
        let parameters = GSKSystemParameters {
            max_group_size: 125,
        };

        let gsk = GSK::new(parameters);
        // let gsk = GSK::new(cond.clone(), spherical_vgram, search_ellipsoid, parameters);

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

        let (groups, _inds) = optimize_groups(all_points.as_slice(), 1f32, 1f32, 1f32, 5, 5, 5);

        let orientations = vec![UnitQuaternion::identity(); groups.len()];
        let node_provider = PointGroupProvider::from_groups(groups.clone(), orientations);
        let mut params = ConditioningParams::default();
        params.max_n_cond = 24;
        params.max_octant = 2;
        let builder = SolvedLUOKSystemBuilder;
        let time1 = std::time::Instant::now();
        let values = gsk.estimate::<SKPointSupportBuilder, _, _, _>(
            &cond,
            &params,
            spherical_vgram,
            search_ellipsoid,
            &node_provider,
            builder,
        );
        let time2 = std::time::Instant::now();
        println!("Time: {:?}", (time2 - time1).as_secs());
        println!(
            "Points per minute: {}",
            values.len() as f32 / (time2 - time1).as_secs_f32() * 60.0
        );

        let mut out = File::create("./test_results/lu_ok_point.txt").unwrap();
        let _ = out.write_all(b"x,y,z,v\n");
        let _ = out.write_all(b"4\n");
        let _ = out.write_all(b"x\n");
        let _ = out.write_all(b"y\n");
        let _ = out.write_all(b"z\n");
        let _ = out.write_all(b"value\n");

        for (point, value) in groups.iter().flatten().zip(values.iter()) {
            //println!("point: {:?}, value: {}", point, value);
            let _ = out
                .write_all(format!("{} {} {} {}\n", point.x, point.y, point.z, value).as_bytes());
        }

        let mut out = File::create("./test_results/lu_ok_cond_data.txt").unwrap();
        let _ = out.write_all(b"x,y,z,v\n");

        for (point, value) in cond.points.iter().zip(cond.data.iter()) {
            //println!("point: {:?}, value: {}", point, value);
            let _ = out
                .write_all(format!("{} {} {} {}\n", point.x, point.y, point.z, value).as_bytes());
        }

        let mut out = File::create("./test_results/lu_ok_point.csv").unwrap();
        //write header
        let _ = out.write_all("X,Y,Z,DX,DY,DZ,V\n".as_bytes());

        //write each row

        for (point, value) in groups.iter().flatten().zip(values.iter()) {
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
    fn gsk_sk_test() {
        // create a gridded database from a csv file (walker lake)
        println!("Reading Cond Data");
        let cond = PointSet::from_csv_index("C:/Users/2jake/OneDrive - McGill University/Fall2022/MIME525/Project4/mineralized_domain_composites.csv", "X", "Y", "Z", "CU")
            .expect("Failed to create gdb");

        //
        let mean = cond.data.iter().sum::<f32>() / cond.data.len() as f32;

        let vgram_rot = UnitQuaternion::from_euler_angles(
            WideF32x8::splat(0.0),
            WideF32x8::splat(0.0),
            WideF32x8::splat(0f32.to_radians()),
        );
        let range = Vector3::new(
            WideF32x8::splat(50.0),
            WideF32x8::splat(200.0),
            WideF32x8::splat(50.0),
        );
        let sill = WideF32x8::splat(1.0f32);

        let spherical_vgram = CompositeVariogram::new(vec![VariogramType::Spherical(
            SphericalVariogram::new(range, sill, vgram_rot),
        )]);

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

        // create a gsk system
        let parameters = GSKSystemParameters { max_group_size: 8 };

        let gsk = GSK::new(parameters);
        // let gsk = GSK::new(cond.clone(), spherical_vgram, search_ellipsoid, parameters);

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

        let _ = std::fs::write("./test_results/lu_sk_orientations.csv", csv_str);

        let node_provider = PointGroupProvider::from_groups(groups.clone(), orientations);
        let mut params = ConditioningParams::default();
        params.orient_search = true;
        params.orient_variogram = true;
        params.max_n_cond = 12;
        params.valid_value_range = [0.0, f32::INFINITY];

        println!("params: {:#?}", params);
        let time1 = std::time::Instant::now();
        // let builder = SolvedNegativeWeightFilteredSystemBuilder::new(SolvedLUOKSystemBuilder);
        let value_transform = MeanTransfrom::new(mean);
        let builder = ModifiedSolvedLUSystemBuilder::new(SolvedLUSKSystemBuilder, value_transform);
        // let builder = SolvedLUSKSystemBuilder;
        let values = gsk.estimate::<SKPointSupportBuilder, _, _, _>(
            &cond,
            &params,
            spherical_vgram,
            search_ellipsoid,
            &node_provider,
            builder,
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

        let mut out = File::create("./test_results/lu_sk.txt").unwrap();
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

        let mut out = File::create("./test_results/lu_sk_cond_data.csv").unwrap();

        let _ = out.write(b"x,y,z,v\n");

        for (point, value) in cond.points.iter().zip(cond.data.iter()) {
            //println!("point: {:?}, value: {}", point, value);
            let _ = out
                .write_all(format!("{},{},{},{}\n", point.x, point.y, point.z, value).as_bytes());
        }

        let mut out = File::create("./test_results/lu_sk.csv").unwrap();
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
