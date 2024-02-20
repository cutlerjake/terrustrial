use std::collections::HashMap;
use std::error;
use std::str::FromStr;

use nalgebra::Point3;
use parry3d::bounding_volume::Aabb;
use rstar::primitives::GeomWithData;
use rstar::{RTree, AABB};
use serde::{Deserialize, Serialize};

use crate::geometry::ellipsoid::Ellipsoid;
use crate::kriging::simple_kriging::ConditioningParams;
use crate::spatial_database::coordinate_system::octant;
use crate::spatial_database::{ConditioningProvider, SpatialDataBase};

type Point = GeomWithData<[f32; 3], u32>;

#[derive(Clone, Serialize, Deserialize)]
pub struct PointSet<T> {
    pub tree: RTree<Point>,
    pub points: Vec<Point3<f32>>,
    pub data: Vec<T>,
}

impl<T> PointSet<T> {
    pub fn new(points: Vec<Point3<f32>>, data: Vec<T>) -> Self {
        let tree_points = points
            .iter()
            .enumerate()
            .map(|(i, p)| Point::new([p.x, p.y, p.z], i as u32))
            .collect();
        let tree = RTree::bulk_load(tree_points);

        Self { tree, points, data }
    }
}

impl<T> PointSet<T>
where
    T: FromStr,
    <T as FromStr>::Err: std::error::Error + 'static,
{
    pub fn from_csv_index(
        csv_path: &str,
        x_col: &str,
        y_col: &str,
        z_col: &str,
        value_col: &str,
    ) -> Result<Self, Box<dyn error::Error>> {
        //storage for data
        let mut point_vec = Vec::new();
        let mut value_vec = Vec::new();

        //read data from csv
        let mut rdr = csv::Reader::from_path(csv_path)?;
        for result in rdr.deserialize() {
            let record: HashMap<String, String> = result?;

            let x = record[x_col].parse::<f32>()?;
            let y = record[y_col].parse::<f32>()?;
            let z = record[z_col].parse::<f32>()?;
            let value = record[value_col].parse::<T>()?;

            point_vec.push(Point3::new(x, y, z));

            value_vec.push(value);
        }

        Ok(Self::new(point_vec, value_vec))
    }
}

impl<T> SpatialDataBase<T> for PointSet<T>
where
    T: Clone,
{
    type INDEX = usize;

    fn inds_in_bounding_box(&self, bounding_box: &Aabb) -> Vec<Self::INDEX> {
        let envelope = AABB::from_corners(
            [
                bounding_box.mins.x,
                bounding_box.mins.y,
                bounding_box.mins.z,
            ],
            [
                bounding_box.maxs.x,
                bounding_box.maxs.y,
                bounding_box.maxs.z,
            ],
        );
        self.tree
            .locate_in_envelope(&envelope)
            .map(|geom| geom.data as usize)
            .collect()
    }

    fn point_at_ind(&self, inds: &Self::INDEX) -> Point3<f32> {
        self.points[*inds]
    }

    fn data_at_ind(&self, ind: &Self::INDEX) -> Option<T> {
        self.data.get(*ind).cloned()
    }

    fn data_and_points(&self) -> (Vec<T>, Vec<Point3<f32>>) {
        (self.data.clone(), self.points.clone())
    }

    fn data_and_inds(&self) -> (Vec<T>, Vec<Self::INDEX>) {
        (self.data.clone(), (0..self.data.len()).collect())
    }

    fn set_data_at_ind(&mut self, ind: &Self::INDEX, data: T) {
        self.data[*ind] = data;
    }
}

pub struct ConditioningDataCollector<'b> {
    pub n_cond: usize,
    pub max_accepted_dist: f32,
    pub ellipsoid: &'b Ellipsoid,
    pub octant_points: Vec<Vec<Point3<f32>>>,
    pub octant_inds: Vec<Vec<u32>>,
    pub full_octants: u8,
    pub conditioned_octants: u8,
    pub stop: bool,
}

impl<'b> ConditioningDataCollector<'b> {
    pub fn new(ellipsoid: &'b Ellipsoid, n_cond: usize) -> Self {
        Self {
            n_cond,
            max_accepted_dist: f32::MAX,
            ellipsoid,
            octant_points: (0..8)
                .map(|_| Vec::with_capacity(n_cond))
                .collect::<Vec<_>>(),
            octant_inds: (0..8)
                .map(|_| Vec::with_capacity(n_cond))
                .collect::<Vec<_>>(),
            full_octants: 0,
            conditioned_octants: 0,
            stop: false,
        }
    }

    #[inline(always)]
    pub fn all_octants_full(&self) -> bool {
        self.full_octants == 8
    }

    #[inline(always)]
    pub fn insert_octant_point(&mut self, point: Point3<f32>, dist: f32, ind: u32) {
        //println!("point: {:?}, dist: {:?}", point, dist);
        // if point is further away the primary ellipsoid axis then it cannot be in the ellipsoid
        // and no further points can be in the ellipsoid
        if self.ellipsoid.a * self.ellipsoid.a < dist {
            self.stop = true;
            return;
        }
        //check if point in ellipsoid
        if !self.ellipsoid.contains(&point) {
            return;
        }

        //determine octant of point in ellispoid coordinate system
        let local_point = self.ellipsoid.coordinate_system.global_to_local(&point);
        let octant = octant(&local_point);

        //get octant points and distances
        let points = &mut self.octant_points[octant as usize];
        let inds = &mut self.octant_inds[octant as usize];
        if points.len() < self.n_cond {
            points.push(point);
            inds.push(ind);

            if points.len() == 1 {
                self.conditioned_octants += 1;
            }
            if points.len() == self.n_cond {
                self.full_octants += 1;
                if self.all_octants_full() {
                    self.stop = true;
                }
            }
        }
    }
}

impl<T> ConditioningProvider<Ellipsoid, T, ConditioningParams> for PointSet<T>
where
    T: Clone,
{
    type Shape = Point3<f32>;
    fn query(
        &self,
        point: &Point3<f32>,
        ellipsoid: &Ellipsoid,
        params: &ConditioningParams,
    ) -> (Vec<usize>, Vec<T>, Vec<Self::Shape>, bool) {
        let mut cond_points = ConditioningDataCollector::new(ellipsoid, params.max_n_cond);

        for (point, dist) in self
            .tree
            .nearest_neighbor_iter_with_distance_2(&[point.x, point.y, point.z])
        {
            let env = point.geom();
            cond_points.insert_octant_point(Point3::new(env[0], env[1], env[2]), dist, point.data);
            if cond_points.stop {
                break;
            }
        }

        let inds: Vec<usize> = cond_points
            .octant_inds
            .into_iter()
            .flatten()
            .map(|i| i as usize)
            .collect();
        let points = cond_points.octant_points.into_iter().flatten().collect();
        let data = inds.iter().map(|ind| self.data[*ind].clone()).collect();

        let res = cond_points.conditioned_octants >= params.min_conditioned_octants as u8;

        (inds, data, points, res)
    }

    fn points(&self) -> &[Point3<f32>] {
        self.points.as_slice()
    }

    fn data(&self) -> &[T] {
        self.data.as_slice()
    }

    fn data_mut(&mut self) -> &mut [T] {
        self.data.as_mut_slice()
    }
}

#[cfg(test)]
mod tests {
    use std::hint::black_box;

    use crate::spatial_database::coordinate_system::CoordinateSystem;

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

        let mut cond_points = ConditioningDataCollector::new(&ellipsoid, n_cond);

        for point in points.iter() {
            let dist = distance(&point, &query_point);
            cond_points.insert_octant_point(*point, dist, 0);
        }
        let mut true_points = points.clone();
        true_points.sort_by(|a, b| {
            let dist_a = distance(a, &query_point);
            let dist_b = distance(b, &query_point);
            dist_a.partial_cmp(&dist_b).unwrap()
        });

        cond_points.octant_points.iter_mut().for_each(|points| {
            points.sort_by(|a, b| {
                let dist_a = distance(a, &query_point);
                let dist_b = distance(b, &query_point);
                dist_a.partial_cmp(&dist_b).unwrap()
            })
        });

        println!("qbvh octant search");
        let mut all_cond_points = cond_points
            .octant_points
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

        // for (i, (octant_points, octant_dists)) in cond_points
        //     .octant_points
        //     .iter()
        //     .zip(cond_points.octant_distances.iter())
        //     .enumerate()
        // {
        //     for (point, dist) in octant_points.iter().zip(octant_dists.iter()) {
        //         println!("point: {:?}, dist: {:?}", point, dist);
        //     }
        // }

        println!("true points");
        for point in true_points.iter().take(n_cond) {
            println!(
                "point: {:?}, dist: {:?}",
                point,
                distance(point, &query_point)
            );
        }
        // for i in 0..n_cond {
        //     if true_points[i] != cond_points.closest_points[i] {
        //         println!("index: {}", i);
        //         println!("true points: {:?}", true_points[i]);
        //         println!("walker points: {:?}", cond_points.closest_points[i]);
        //         println!("true dist: {}", distance(&true_points[i], &query_point));
        //         println!(
        //             "walker dist: {}",
        //             distance(&cond_points.closest_points[i], &query_point)
        //         );
        //         panic!();
        //     }
        // }
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

        let point_set = PointSet::new(points.clone(), data);
        let query_point = Point3::new(500.0, 500.0, 500.0);
        let n_cond = 20;

        let quat = nalgebra::UnitQuaternion::identity();
        let cs = CoordinateSystem::new(query_point.coords.into(), quat);

        let ellipsoid = Ellipsoid::new(200f32, 200f32, 200f32, cs);

        let (_, _, mut c_points, _) = point_set.query(
            &query_point,
            &ellipsoid,
            &ConditioningParams {
                max_n_cond: 20,
                min_conditioned_octants: 0,
            },
        );

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

        println!("Building point set");
        let point_set = PointSet::new(points.clone(), data);
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
                black_box(&ConditioningParams {
                    max_n_cond: 20,
                    min_conditioned_octants: 0,
                }),
            ));
        }
        println!(
            "Speed test completed: queries: {:?}, time: {:?}",
            10000,
            time.elapsed()
        );
    }
}
