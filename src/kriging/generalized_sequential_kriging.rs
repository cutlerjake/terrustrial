use crate::decomposition::lu::LUSystem;
use crate::decomposition::lu::MiniLUSystem;
use crate::geometry::ellipsoid::Ellipsoid;
use crate::geometry::Geometry;
use crate::spatial_database::ConditioningProvider;
use crate::variography::model_variograms::VariogramModel;
use indicatif::ParallelProgressIterator;
use nalgebra::Point3;
use rstar::primitives::GeomWithData;
use rstar::RTree;
use rstar::RTreeObject;
use rstar::AABB;
use simba::simd::SimdPartialOrd;
use simba::simd::SimdRealField;
use simba::simd::SimdValue;

use super::simple_kriging::ConditioningParams;
use super::simple_kriging::SKBuilder;
use super::simple_kriging::SupportInterface;
use super::simple_kriging::SupportTransform;
use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};

#[derive(Clone, Copy, Debug)]
pub struct GSKParameters {
    pub max_group_size: usize,
    pub max_cond_data: usize,
}

pub struct GSK<S, V, VT>
where
    S: ConditioningProvider<Ellipsoid, f32, ConditioningParams> + Sync + std::marker::Send,
    V: VariogramModel<VT>,
    VT: SimdPartialOrd + SimdRealField + SimdValue<Element = f32> + Copy,
{
    conditioning_data: S,
    variogram_model: V,
    search_ellipsoid: Ellipsoid,
    parameters: GSKParameters,
    phantom_v_type: std::marker::PhantomData<VT>,
}

impl<S, V, VT> GSK<S, V, VT>
where
    S: ConditioningProvider<Ellipsoid, f32, ConditioningParams> + Sync + std::marker::Send,
    V: VariogramModel<VT> + std::marker::Sync,
    VT: SimdPartialOrd + SimdRealField + SimdValue<Element = f32> + Copy,
{
    pub fn new(
        conditioning_data: S,
        variogram_model: V,
        search_ellipsoid: Ellipsoid,
        parameters: GSKParameters,
    ) -> Self {
        Self {
            conditioning_data,
            variogram_model,
            search_ellipsoid,
            parameters,
            phantom_v_type: std::marker::PhantomData,
        }
    }
    pub fn estimate<SKB, MS>(&self, groups: &Vec<Vec<SKB::Support>>) -> Vec<f32>
    where
        SKB: SKBuilder,
        S::Shape: SupportTransform<SKB::Support>,
        //SKB::Support: SupportTransform<S::Shape>,
        //S::Shape: SupportTransform<NewSupport = SKB::Support>,
        <SKB as SKBuilder>::Support: SupportInterface, // why do I need this the trait already requires this?!?!?
        SKB::Support: Sync,
        MS: MiniLUSystem,
    {
        let system = LUSystem::new(
            self.parameters.max_group_size,
            (self.parameters.max_cond_data) * 8,
        );

        let cond_params = ConditioningParams::new(self.parameters.max_cond_data);

        groups
            .par_iter()
            .progress()
            .map_with(
                (system.clone(), self.search_ellipsoid.clone()),
                |(local_system, ellipsoid), group| {
                    //get center of group
                    let center = group.iter().fold(Point3::<f32>::origin(), |mut acc, x| {
                        acc.coords += x.center().coords;
                        acc
                    }) / (group.len() as f32);

                    //translate search ellipsoid to group center
                    ellipsoid.translate_to(&center);
                    //get nearest points and values
                    let (_, cond_values, cond_points) =
                        self.conditioning_data
                            .query(&center, &ellipsoid, &cond_params);

                    //convert points to support
                    let cond_points = cond_points
                        .into_iter()
                        .map(|x| x.transform())
                        .collect::<Vec<_>>();

                    //build kriging system for point
                    let mut mini_system = local_system.create_mini_system::<V, VT, SKB, MS>(
                        cond_points.as_slice(),
                        group.as_slice(),
                        &self.variogram_model,
                    );

                    mini_system.populate_cond_values_est(cond_values.as_slice());

                    mini_system.estimate()
                },
            )
            .flatten()
            .collect::<Vec<f32>>()
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
                    [x, y, z].into(),
                    [x + env_size[0], y + env_size[1], z + env_size[2]].into(),
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
    use rstar::{primitives::GeomWithData, RTree, RTreeObject, AABB};
    use simba::simd::WideF32x8;

    use crate::{
        decomposition::lu::{
            AverageTransfrom, MiniLUOKSystem, MiniLUSKSystem, ModifiedMiniLUSystem,
        },
        kriging::simple_kriging::{SKPointSupportBuilder, SKVolumeSupportBuilder},
        spatial_database::{
            coordinate_system::CoordinateSystem, rtree_point_set::point_set::PointSet,
            zero_mean::ZeroMeanTransform, DiscretiveVolume,
        },
        variography::model_variograms::spherical::SphericalVariogram,
    };

    use super::*;

    #[test]
    fn gsk_ok_test() {
        // create a gridded database from a csv file (walker lake)
        println!("Reading Cond Data");
        let cond = PointSet::from_csv_index("C:/Users/2jake/OneDrive - McGill University/Fall2022/MIME525/Project4/mineralized_domain_composites.csv", "X", "Y", "Z", "CU")
            .expect("Failed to create gdb");

        //

        let vgram_rot = UnitQuaternion::identity();
        let range = Vector3::new(
            WideF32x8::splat(200.0),
            WideF32x8::splat(200.0),
            WideF32x8::splat(200.0),
        );
        let sill = WideF32x8::splat(1.0f32);
        let nugget = WideF32x8::splat(0.2);

        let spherical_vgram = SphericalVariogram::new(range, sill, nugget, vgram_rot);

        // create search ellipsoid
        let search_ellipsoid = Ellipsoid::new(
            200f32,
            200f32,
            200f32,
            CoordinateSystem::new(Translation3::new(0.0, 0.0, 0.0), UnitQuaternion::identity()),
        );

        // create a gsk system
        let parameters = GSKParameters {
            max_group_size: 250,
            max_cond_data: 100,
        };
        let gsk = GSK::new(cond.clone(), spherical_vgram, search_ellipsoid, parameters);

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
        let mut groups = Vec::new();
        //let mut group = Vec::new();
        for (i, point) in points.iter().enumerate() {
            let aabb = Aabb::new(
                Point3::new(point.x, point.y, point.z),
                Point3::new(point.x + 5.0, point.y + 5.0, point.z + 10.0),
            );

            groups.push(aabb.discretize(2f32, 2f32, 2f32));
        }

        let time1 = std::time::Instant::now();
        let values =
            gsk.estimate::<SKPointSupportBuilder, ModifiedMiniLUSystem<MiniLUOKSystem, AverageTransfrom>>(&groups);
        let time2 = std::time::Instant::now();
        println!("Time: {:?}", (time2 - time1).as_secs());
        println!(
            "Points per minute: {}",
            values.len() as f32 / (time2 - time1).as_secs_f32() * 60.0
        );

        //save values to file for visualization

        let mut out = File::create("./test_results/lu_ok.txt").unwrap();
        let _ = out.write_all(b"surfs\n");
        let _ = out.write_all(b"4\n");
        let _ = out.write_all(b"x\n");
        let _ = out.write_all(b"y\n");
        let _ = out.write_all(b"z\n");
        let _ = out.write_all(b"value\n");

        for (point, value) in points.iter().zip(values.iter()) {
            //println!("point: {:?}, value: {}", point, value);
            let _ = out
                .write_all(format!("{} {} {} {}\n", point.x, point.y, point.z, value).as_bytes());
        }

        let mut out = File::create("./test_results/lu_ok_cond_data.txt").unwrap();
        let _ = out.write_all(b"surfs\n");
        let _ = out.write_all(b"4\n");
        let _ = out.write_all(b"x\n");
        let _ = out.write_all(b"y\n");
        let _ = out.write_all(b"z\n");
        let _ = out.write_all(b"value\n");

        for (point, value) in cond.points.iter().zip(cond.data.iter()) {
            //println!("point: {:?}, value: {}", point, value);
            let _ = out
                .write_all(format!("{} {} {} {}\n", point.x, point.y, point.z, value).as_bytes());
        }

        let mut out = File::create("./test_results/lu_ok.csv").unwrap();
        //write header
        let _ = out.write_all("X,Y,Z,XS,YS,ZS,V\n".as_bytes());

        //write each row

        for (point, value) in points.iter().zip(values.iter()) {
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
        let mut cond = PointSet::from_csv_index("C:/Users/2jake/OneDrive - McGill University/Fall2022/MIME525/Project4/mineralized_domain_composites.csv", "X", "Y", "Z", "CU")
            .expect("Failed to create gdb");

        let mt = ZeroMeanTransform::from(cond.data());
        cond.data.iter_mut().for_each(|x| *x = mt.transform(*x));

        let vgram_rot = UnitQuaternion::identity();
        let range = Vector3::new(
            WideF32x8::splat(200.0),
            WideF32x8::splat(200.0),
            WideF32x8::splat(200.0),
        );
        let sill = WideF32x8::splat(1.0f32);
        let nugget = WideF32x8::splat(0.2);

        let spherical_vgram = SphericalVariogram::new(range, sill, nugget, vgram_rot);

        // create search ellipsoid
        let search_ellipsoid = Ellipsoid::new(
            200f32,
            200f32,
            200f32,
            CoordinateSystem::new(Translation3::new(0.0, 0.0, 0.0), UnitQuaternion::identity()),
        );

        // create a gsk system
        let parameters = GSKParameters {
            max_group_size: 250,
            max_cond_data: 100,
        };
        let gsk = GSK::new(cond.clone(), spherical_vgram, search_ellipsoid, parameters);

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
        let mut groups = Vec::new();
        let mut group = Vec::new();
        for (i, point) in points.iter().enumerate() {
            for x in 0..5 {
                for y in 0..5 {
                    for z in 0..10 {
                        group.push(Point3::new(
                            point.x + x as f32,
                            point.y + y as f32,
                            point.z + z as f32,
                        ));
                    }
                }
            }

            groups.push(group.clone());
            group.clear();
        }

        let time1 = std::time::Instant::now();
        let values =
            gsk.estimate::<SKPointSupportBuilder, ModifiedMiniLUSystem<MiniLUSKSystem, AverageTransfrom>>(&groups);
        let time2 = std::time::Instant::now();
        println!("Time: {:?}", (time2 - time1).as_secs());
        println!(
            "Points per minute: {}",
            values.len() as f32 / (time2 - time1).as_secs_f32() * 60.0
        );

        //save values to file for visualization

        let mut out = File::create("./test_results/lu_sk_block_mean.txt").unwrap();
        let _ = out.write_all(b"surfs\n");
        let _ = out.write_all(b"4\n");
        let _ = out.write_all(b"x\n");
        let _ = out.write_all(b"y\n");
        let _ = out.write_all(b"z\n");
        let _ = out.write_all(b"value\n");

        for (point, value) in points.iter().zip(values.iter()) {
            //println!("point: {:?}, value: {}", point, value);
            let _ = out
                .write_all(format!("{} {} {} {}\n", point.x, point.y, point.z, value).as_bytes());
        }

        let mut out = File::create("./test_results/lu_ok_block_mean_cond_data.txt").unwrap();
        let _ = out.write_all(b"surfs\n");
        let _ = out.write_all(b"4\n");
        let _ = out.write_all(b"x\n");
        let _ = out.write_all(b"y\n");
        let _ = out.write_all(b"z\n");
        let _ = out.write_all(b"value\n");

        for (point, value) in cond.points.iter().zip(cond.data.iter()) {
            //println!("point: {:?}, value: {}", point, value);
            let _ = out
                .write_all(format!("{} {} {} {}\n", point.x, point.y, point.z, value).as_bytes());
        }

        let mut out = File::create("./test_results/lu_sk_block_mean..csv").unwrap();
        //write header
        let _ = out.write_all("X,Y,Z,XS,YS,ZS,V\n".as_bytes());

        //write each row

        for (point, value) in points.iter().zip(values.iter()) {
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
    fn gsk_ok_db_test() {
        // create a gridded database from a csv file (walker lake)
        println!("Reading Cond Data");
        let cond = PointSet::from_csv_index("C:/Users/2jake/OneDrive - McGill University/Fall2022/MIME525/Project4/mineralized_domain_composites.csv", "X", "Y", "Z", "CU")
            .expect("Failed to create gdb");

        //

        let vgram_rot = UnitQuaternion::identity();
        let range = Vector3::new(
            WideF32x8::splat(200.0),
            WideF32x8::splat(200.0),
            WideF32x8::splat(200.0),
        );
        let sill = WideF32x8::splat(1.0f32);
        let nugget = WideF32x8::splat(0.2);

        let spherical_vgram = SphericalVariogram::new(range, sill, nugget, vgram_rot);

        // create search ellipsoid
        let search_ellipsoid = Ellipsoid::new(
            200f32,
            200f32,
            200f32,
            CoordinateSystem::new(Translation3::new(0.0, 0.0, 0.0), UnitQuaternion::identity()),
        );

        // create a gsk system
        let group_size = 10;
        let parameters = GSKParameters {
            max_group_size: group_size,
            max_cond_data: 20,
        };
        let gsk = GSK::new(cond.clone(), spherical_vgram, search_ellipsoid, parameters);

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
        let mut groups = Vec::new();
        let mut group = Vec::new();
        for (i, point) in points.iter().enumerate() {
            //iterate over 5x5x10 grid originating at point
            let mut block = Vec::new();
            for x in 0..5 {
                for y in 0..5 {
                    for z in 0..10 {
                        block.push(Point3::new(
                            point.x + x as f32,
                            point.y + y as f32,
                            point.z + z as f32,
                        ));
                    }
                }
            }
            group.push(block);

            if (i % group_size - 1 == 0 && i != 0) || i == points.len() - 1 {
                groups.push(group.clone());
                group.clear();
            }
        }

        let time1 = std::time::Instant::now();
        let values = gsk.estimate::<SKVolumeSupportBuilder, MiniLUOKSystem>(&groups);
        let time2 = std::time::Instant::now();
        println!("Time: {:?}", (time2 - time1).as_secs());
        println!(
            "Points per minute: {}",
            values.len() as f32 / (time2 - time1).as_secs_f32() * 60.0
        );

        //save values to file for visualization

        let mut out = File::create("./test_results/lu_ok_db.txt").unwrap();
        let _ = out.write_all(b"surfs\n");
        let _ = out.write_all(b"4\n");
        let _ = out.write_all(b"x\n");
        let _ = out.write_all(b"y\n");
        let _ = out.write_all(b"z\n");
        let _ = out.write_all(b"value\n");

        for (point, value) in points.iter().zip(values.iter()) {
            //println!("point: {:?}, value: {}", point, value);
            let _ = out
                .write_all(format!("{} {} {} {}\n", point.x, point.y, point.z, value).as_bytes());
        }

        let mut out = File::create("./test_results/lu_ok_db_cond_data.txt").unwrap();
        let _ = out.write_all(b"surfs\n");
        let _ = out.write_all(b"4\n");
        let _ = out.write_all(b"x\n");
        let _ = out.write_all(b"y\n");
        let _ = out.write_all(b"z\n");
        let _ = out.write_all(b"value\n");

        for (point, value) in cond.points.iter().zip(cond.data.iter()) {
            //println!("point: {:?}, value: {}", point, value);
            let _ = out
                .write_all(format!("{} {} {} {}\n", point.x, point.y, point.z, value).as_bytes());
        }

        let mut out = File::create("./test_results/lu_ok_db.csv").unwrap();
        //write header
        let _ = out.write_all("X,Y,Z,XS,YS,ZS,V\n".as_bytes());

        //write each row

        for (point, value) in points.iter().zip(values.iter()) {
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
    fn gsk_large_model() {
        // create a gridded database from a csv file (walker lake)
        println!("Reading Cond Data");
        let cond = PointSet::from_csv_index(
            "C:\\GitRepos\\terrustrial\\data\\point_set.csv",
            "X",
            "Y",
            "Z",
            "Value",
        )
        .expect("Failed to create gdb");

        //

        let vgram_rot = UnitQuaternion::identity();
        let range = Vector3::new(
            WideF32x8::splat(400.0),
            WideF32x8::splat(400.0),
            WideF32x8::splat(400.0),
        );
        let sill = WideF32x8::splat(1.0f32);
        let nugget = WideF32x8::splat(0.2);

        let spherical_vgram = SphericalVariogram::new(range, sill, nugget, vgram_rot);

        // create search ellipsoid
        let search_ellipsoid = Ellipsoid::new(
            400f32,
            400f32,
            400f32,
            CoordinateSystem::new(Translation3::new(0.0, 0.0, 0.0), UnitQuaternion::identity()),
        );

        // create a gsk system
        let group_size = 125;
        let parameters = GSKParameters {
            max_group_size: group_size,
            max_cond_data: 25,
        };
        let gsk = GSK::new(cond.clone(), spherical_vgram, search_ellipsoid, parameters);

        println!("Reading Target Data");
        let mut reader =
            csv::Reader::from_path("C:\\GitRepos\\terrustrial\\data\\new_model.csv").unwrap();

        let mut aabbs = Vec::new();
        for record in reader.deserialize() {
            let record: (f32, f32, f32, f32, f32, f32, f32) = record.unwrap();
            aabbs.push(Aabb::new(
                Point3::new(record.0, record.1, record.2),
                Point3::new(
                    record.0 + record.3,
                    record.1 + record.4,
                    record.2 + record.5,
                ),
            ));
        }

        println!("Discretizing");
        //discretize each block
        let dx = 2f32;
        let dy = 2f32;
        let dz = 2f32;

        let points = aabbs
            .iter()
            .map(|x| x.discretize(dx, dy, dz))
            .flatten()
            .collect::<Vec<_>>();

        println!("Optimizing groups");

        let (groups, point_inds) = optimize_groups(points.as_slice(), dx, dy, dz, 5, 5, 5);

        let time1 = std::time::Instant::now();
        let values = gsk.estimate::<SKPointSupportBuilder, MiniLUOKSystem>(&groups);
        let time2 = std::time::Instant::now();
        println!("Time: {:?}", (time2 - time1).as_secs());

        //save values to file for visualization
        let mut out = File::create("./test_results/gsk_large_model.txt").unwrap();
        let _ = out.write_all(b"surfs\n");
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

        let mut out = File::create("./test_results/gsk_large_model_cond_data.txt").unwrap();
        let _ = out.write_all(b"surfs\n");
        let _ = out.write_all(b"4\n");
        let _ = out.write_all(b"x\n");
        let _ = out.write_all(b"y\n");
        let _ = out.write_all(b"z\n");
        let _ = out.write_all(b"value\n");

        for (point, value) in cond.points.iter().zip(cond.data.iter()) {
            //println!("point: {:?}, value: {}", point, value);
            let _ = out
                .write_all(format!("{} {} {} {}\n", point.x, point.y, point.z, value).as_bytes());
        }

        // let mut out = File::create("./test_results/gsk_large_model.csv").unwrap();
        // //write header
        // let _ = out.write_all("X,Y,Z,XS,YS,ZS,V\n".as_bytes());

        // //write each row

        // for (point, value) in points.iter().zip(values.iter()) {
        //     //println!("point: {:?}, value: {}", point, value);
        //     let _ = out.write_all(
        //         format!(
        //             "{},{},{},{},{},{},{}\n",
        //             point.x, point.y, point.z, 5, 5, 10, value
        //         )
        //         .as_bytes(),
        //     );
        // }
    }
}
