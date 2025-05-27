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
/// p[i] is the probability that the value is less than x[i].
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

#[cfg(test)]
mod test {
    use std::{fs::File, io::Write};

    use itertools::Itertools;
    use num_traits::Float;
    use ultraviolet::{DRotor3, DVec3};

    use crate::{
        geometry::{aabb::Aabb, support::Support},
        spatial_database::{
            coordinate_system::NewCoordinateSystem, DiscretiveVolume, SpatialAcceleratedDB,
        },
        systems::solved_systems::ok_system::SolvedLUOKSystemBuilder,
        variography::model_variograms::{composite::VariogramType, spherical::SphericalVariogram},
    };

    use super::*;

    #[test]
    fn gsik_ok_test() {
        // create a gridded database from a csv file (walker lake)
        println!("Reading Cond Data");
        let cond = SpatialAcceleratedDB::from_csv_index("C:/Users/2jake/OneDrive - McGill University/Fall2022/MIME525/Project4/mineralized_domain_composites.csv", "X", "Y", "Z", "CU")
            .expect("Failed to create gdb");

        let thresholds = vec![
            0., 0.33333333, 0.66666667, 1., 1.33333333, 1.66666667, 2., 2.33333333, 2.66666667, 3.,
        ];

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
            NewCoordinateSystem::new(DVec3::zero(), vgram_rot),
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

        let builder = SolvedLUOKSystemBuilder;
        let time1 = std::time::Instant::now();
        let values = estimate(
            thresholds.as_slice(),
            &mut cond.clone(),
            &Default::default(),
            &vec![spherical_vgram; thresholds.len()],
            &search_ellipsoid,
            &groups,
            builder,
        );
        let time2 = std::time::Instant::now();
        println!("Time: {:?}", (time2 - time1).as_secs());
        println!(
            "Points per minute: {}",
            values.len() as f32 / (time2 - time1).as_secs_f32() * 60.0
        );

        //save values to file for visualization

        let mut out = File::create("./test_results/lu_ik_ok.txt").unwrap();
        let _ = out.write_all(b"surfs\n");
        let _ = out.write_all(b"4\n");
        let _ = out.write_all(b"x\n");
        let _ = out.write_all(b"y\n");
        let _ = out.write_all(b"z\n");
        for theshold in thresholds.iter() {
            let _ = out.write_all(format!("leq_{}\n", theshold).as_bytes());
        }

        for (support, value) in groups.get_supports().iter().zip(values.iter()) {
            let point = support.center();
            //println!("point: {:?}, value: {}", point, value);
            let _ = out.write_all(format!("{} {} {}", point.x, point.y, point.z).as_bytes());
            for v in value.p.iter() {
                let _ = out.write_all(format!(" {}", v).as_bytes());
            }
            let _ = out.write_all(b"\n");
        }

        let mut out = File::create("./test_results/lu_ik_ok.csv").unwrap();
        //write header
        let _ = out.write_all("X,Y,Z,DX,DY,DZ".as_bytes());
        for theshold in thresholds.iter() {
            let _ = out.write_all(format!(",leq_{}", theshold).as_bytes());
        }
        let _ = out.write_all(b"\n");

        //write each row

        for (support, value) in groups.get_supports().iter().zip(values.iter()) {
            let point = support.center();
            //println!("point: {:?}, value: {}", point, value);
            let _ = out.write_all(
                format!("{},{},{},{},{},{}", point.x, point.y, point.z, 5, 5, 10).as_bytes(),
            );

            for v in value.p.iter() {
                let _ = out.write_all(format!(",{}", v).as_bytes());
            }
            let _ = out.write_all(b"\n");
        }
    }
}
