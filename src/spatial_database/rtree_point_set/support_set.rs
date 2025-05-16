use std::collections::HashMap;
use std::error;
use std::str::FromStr;

use nalgebra::Point3;
use parry3d::bounding_volume::Aabb;
use rstar::primitives::GeomWithData;
use rstar::{RTree, AABB};
use serde::{Deserialize, Serialize};

use crate::spatial_database::{IterNearest, SpatialDataBase, SupportInterface};

type Point = GeomWithData<[f32; 3], u32>;

#[derive(Clone, Serialize, Deserialize)]
pub struct SupportSet<S, T> {
    pub tree: RTree<Point>,
    pub points: Vec<S>,
    pub data: Vec<T>,
    pub source_tag: Vec<usize>,
}

impl<S: SupportInterface, T> SupportSet<S, T> {
    pub fn new(points: Vec<S>, data: Vec<T>, source_tag: Vec<usize>) -> Self {
        let tree_points = points
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let p = p.center();
                Point::new([p.x, p.y, p.z], i as u32)
            })
            .collect();
        let tree = RTree::bulk_load(tree_points);

        Self {
            tree,
            points,
            data,
            source_tag,
        }
    }
}

impl<T: Copy, S: SupportInterface + Clone> IterNearest for SupportSet<S, T> {
    type Shape = S;

    type Data = T;

    fn iter_nearest(
        &self,
        location: &Point3<f32>,
    ) -> impl Iterator<Item = crate::spatial_database::IterNearestElem<Self::Shape, Self::Data>> + '_
    {
        self.tree
            .nearest_neighbor_iter_with_distance_2(&[location.x, location.y, location.z])
            .map(|(point, dist)| crate::spatial_database::IterNearestElem {
                shape: self.points[point.data as usize].clone(),
                dist,
                data: self.data[point.data as usize],
                tag: self.source_tag[point.data as usize] as u32,
                idx: point.data,
            })
    }
}

#[cfg(test)]
mod tests {
    use std::hint::black_box;

    use crate::estimators::ConditioningParams;
    use crate::geometry::ellipsoid::Ellipsoid;
    use crate::spatial_database::coordinate_system::CoordinateSystem;
    use crate::spatial_database::ConditioningDataCollector;
    use crate::spatial_database::ConditioningProvider;

    use super::*;
    use nalgebra::distance;
    use parry3d::bounding_volume::Aabb;
    use rand;
    use rand::rngs::ThreadRng;
    use rand::Rng;

    #[inline(always)]
    fn gen_random_point(aabb: Aabb, rng: &mut ThreadRng) -> Point3<f32> {
        let min = aabb.mins;
        let max = aabb.maxs;
        let x = rng.gen_range(min.coords.x..max.coords.x);
        let y = rng.gen_range(min.coords.y..max.coords.y);
        let z = rng.gen_range(min.coords.z..max.coords.z);
        Point3::new(x, y, z)
    }

    fn gen_random_points(n: usize, aabb: Aabb) -> Vec<Point3<f32>> {
        let mut rng = rand::thread_rng();
        let mut points = Vec::with_capacity(n);
        for _ in 0..n {
            points.push(gen_random_point(aabb, &mut rng));
        }

        points
    }

    #[test]
    fn rtree_conditioning_insertion() {
        let n_points = 1000;

        let points = gen_random_points(
            n_points,
            Aabb::new(
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(1000.0, 1000.0, 1000.0),
            ),
        );
        let query_point = Point3::new(500.0, 500.0, 500.0);
        let n_cond = 20;

        let quat = nalgebra::UnitQuaternion::identity();
        let cs = CoordinateSystem::new(query_point.coords.into(), quat);

        let ellipsoid = Ellipsoid::new(200f32, 200f32, 200f32, cs);

        let cond_params = ConditioningParams::default();

        let mut cond_points = ConditioningDataCollector::new(&ellipsoid, &cond_params);

        for point in points.iter() {
            let dist = distance(&point, &query_point);
            cond_points.try_insert_shape(*point, 0.0, dist, 0, 0);
        }
        let mut true_points = points.clone();
        true_points.sort_by(|a, b| {
            let dist_a = distance(a, &query_point);
            let dist_b = distance(b, &query_point);
            dist_a.partial_cmp(&dist_b).unwrap()
        });

        cond_points.octant_shapes.iter_mut().for_each(|points| {
            points.sort_by(|a, b| {
                let dist_a = distance(a, &query_point);
                let dist_b = distance(b, &query_point);
                dist_a.partial_cmp(&dist_b).unwrap()
            })
        });

        println!("qbvh octant search");
        let mut all_cond_points = cond_points
            .octant_shapes
            .iter()
            .flatten()
            .collect::<Vec<_>>();

        all_cond_points.sort_by(|a, b| {
            let dist_a = distance(a, &query_point);
            let dist_b = distance(b, &query_point);
            dist_a.partial_cmp(&dist_b).unwrap()
        });

        let all_dists = all_cond_points
            .iter()
            .map(|point| distance(point, &query_point))
            .collect::<Vec<_>>();

        all_cond_points
            .iter()
            .zip(all_dists.iter())
            .for_each(|(point, dist)| {
                println!("point: {:?}, dist: {:?}", point, dist);
            });

        println!("true points");
        for point in true_points.iter().take(n_cond) {
            println!(
                "point: {:?}, dist: {:?}",
                point,
                distance(point, &query_point)
            );
        }
    }

    #[test]
    fn rtree_collector() {
        let n_points = 1000;

        let points = gen_random_points(
            n_points,
            Aabb::new(
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(1000.0, 1000.0, 1000.0),
            ),
        );

        let data = vec![0.0; n_points];

        let tags = (0..n_points).collect();

        let point_set = SupportSet::new(points.clone(), data, tags);
        let query_point = Point3::new(500.0, 500.0, 500.0);
        let n_cond = 20;

        let quat = nalgebra::UnitQuaternion::identity();
        let cs = CoordinateSystem::new(query_point.coords.into(), quat);

        let ellipsoid = Ellipsoid::new(200f32, 200f32, 200f32, cs);

        let (_, _, mut c_points, _) =
            point_set.query(&query_point, &ellipsoid, &ConditioningParams::default());

        let mut true_points = points.clone();
        true_points.sort_by(|a, b| {
            let dist_a = distance(a, &query_point);
            let dist_b = distance(b, &query_point);
            dist_a.partial_cmp(&dist_b).unwrap()
        });

        // println!("cond points: {:?}", cond_points.octant_points);

        // let mut c_points = cond_points
        //     .octant_points
        //     .iter()
        //     .flatten()
        //     .collect::<Vec<_>>();
        c_points.sort_by(|a, b| {
            let dist_a = distance(a, &query_point);
            let dist_b = distance(b, &query_point);
            dist_a.partial_cmp(&dist_b).unwrap()
        });

        for i in 0..n_cond {
            println!("index: {}", i);
            println!("true points: {:?}", true_points[i]);
            println!("walker points: {:?}", c_points[i]);
            println!("true dist: {}", distance(&true_points[i], &query_point));
            println!("walker dist: {}", distance(&c_points[i], &query_point));
            if true_points[i] != c_points[i] {
                panic!();
            }
        }

        // println!("true points: {:?}", true_points[0..n_cond].to_vec());
        // println!("walker points: {:?}", cond_points.closest_points);
    }

    #[test]
    fn valid_value_range() {
        let points = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 1.0, 1.0),
            Point3::new(2.0, 2.0, 2.0),
            Point3::new(3.0, 3.0, 3.0),
            Point3::new(4.0, 4.0, 4.0),
            Point3::new(5.0, 5.0, 5.0),
            Point3::new(6.0, 6.0, 6.0),
            Point3::new(7.0, 7.0, 7.0),
        ];

        let data = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 10.0];
        let tags = (0..8).collect();
        let point_set = SupportSet::new(points, data, tags);
        let query_point = Point3::new(7.0, 7.0, 7.0);
        let quat = nalgebra::UnitQuaternion::identity();
        let cs = CoordinateSystem::new(query_point.coords.into(), quat);
        let ellipsoid = Ellipsoid::new(100f32, 100f32, 100f32, cs);

        let (_, vals, _, _res) = point_set.query(
            &query_point,
            &ellipsoid,
            &ConditioningParams {
                valid_value_range: [0.0, 2.0],
                ..ConditioningParams::default()
            },
        );

        println!("vals: {:?}", vals);
    }

    #[test]
    fn clip_range() {
        let points = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 1.0, 1.0),
            Point3::new(2.0, 2.0, 2.0),
            Point3::new(3.0, 3.0, 3.0),
            Point3::new(4.0, 4.0, 4.0),
            Point3::new(5.0, 5.0, 5.0),
            Point3::new(6.0, 6.0, 6.0),
            Point3::new(7.0, 7.0, 7.0),
        ];

        let data = vec![1.0, 10.0, 3.0, 100.0, 10.0, 1.0, 1.0, 10.0];

        let tags = (0..8).collect();

        let point_set = SupportSet::new(points, data, tags);

        let query_point = Point3::new(0.0, 0.0, 0.0);

        let quat = nalgebra::UnitQuaternion::identity();

        let cs = CoordinateSystem::new(query_point.coords.into(), quat);

        let ellipsoid = Ellipsoid::new(100f32, 100f32, 100f32, cs);

        let (_, vals, _, _res) = point_set.query(
            &query_point,
            &ellipsoid,
            &ConditioningParams {
                clip_h: vec![0.1],
                clip_range: vec![[0.0, 2.0]],
                ..ConditioningParams::default()
            },
        );

        println!("vals: {:?}", vals);
    }

    #[test]
    fn rtree_speed_test() {
        let n_points = 1_000_000;

        println!("Generating {} points", n_points);
        let points = gen_random_points(
            n_points,
            Aabb::new(
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(1000.0, 1000.0, 1000.0),
            ),
        );

        let data = vec![0.0; n_points];

        let tags = (0..n_points).collect();

        println!("Building point set");
        let point_set = SupportSet::new(points.clone(), data, tags);
        let mut rng = rand::thread_rng();
        println!("Starting speed test");
        let time = std::time::Instant::now();
        for _ in 0..10_000 {
            let query_point = gen_random_point(
                Aabb::new(
                    Point3::new(0.0, 0.0, 0.0),
                    Point3::new(1000.0, 1000.0, 1000.0),
                ),
                &mut rng,
            );
            let quat = nalgebra::UnitQuaternion::identity();
            let cs = CoordinateSystem::new(query_point.coords.into(), quat);

            let ellipsoid = Ellipsoid::new(100f32, 100f32, 100f32, cs);

            black_box(point_set.query(
                black_box(&query_point),
                black_box(&ellipsoid),
                black_box(&ConditioningParams::default()),
            ));
        }
        println!(
            "Speed test completed: queries: {:?}, time: {:?}",
            10000,
            time.elapsed()
        );
    }
}
