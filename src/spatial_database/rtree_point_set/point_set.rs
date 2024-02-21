use std::collections::HashMap;
use std::error;
use std::str::FromStr;

use nalgebra::Point3;
use ordered_float::OrderedFloat;
use parry3d::bounding_volume::Aabb;
use rstar::primitives::GeomWithData;
use rstar::{RTree, AABB};
use serde::{Deserialize, Serialize};

use crate::geometry::ellipsoid::Ellipsoid;
use crate::kriging::ConditioningParams;
use crate::spatial_database::coordinate_system::octant;
use crate::spatial_database::{ConditioningProvider, SpatialDataBase};
use permutation::Permutation;

type Point = GeomWithData<[f32; 3], u32>;

#[derive(Clone, Serialize, Deserialize)]
pub struct PointSet<T> {
    pub tree: RTree<Point>,
    pub points: Vec<Point3<f32>>,
    pub data: Vec<T>,
    pub source_tag: Vec<usize>,
}

impl<T> PointSet<T> {
    pub fn new(points: Vec<Point3<f32>>, data: Vec<T>, source_tag: Vec<usize>) -> Self {
        let tree_points = points
            .iter()
            .enumerate()
            .map(|(i, p)| Point::new([p.x, p.y, p.z], i as u32))
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

        //no source tag -> give all points a unique tag
        let source_tag = (0..point_vec.len()).collect();

        Ok(Self::new(point_vec, value_vec, source_tag))
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
    pub cond_params: &'b ConditioningParams,
    pub max_accepted_dist: f32,
    pub ellipsoid: &'b Ellipsoid,
    pub octant_points: Vec<Vec<Point3<f32>>>,
    pub octant_norm_dists: Vec<Vec<f32>>,
    pub octant_values: Vec<Vec<f32>>,
    pub octant_inds: Vec<Vec<u32>>,
    pub octant_counts: Vec<u32>,
    pub full_octants: u8,
    pub conditioned_octants: u8,
    pub source_tag: Vec<u32>,
    pub source_count: Vec<u32>,
    pub stop: bool,
}

impl<'b> ConditioningDataCollector<'b> {
    pub fn new(ellipsoid: &'b Ellipsoid, cond_params: &'b ConditioningParams) -> Self {
        let octant_max = cond_params.max_octant;
        Self {
            cond_params,
            max_accepted_dist: f32::MAX,
            ellipsoid,
            octant_points: (0..8)
                .map(|_| Vec::with_capacity(octant_max))
                .collect::<Vec<_>>(),
            octant_norm_dists: (0..8)
                .map(|_| Vec::with_capacity(octant_max))
                .collect::<Vec<_>>(),
            octant_values: (0..8)
                .map(|_| Vec::with_capacity(octant_max))
                .collect::<Vec<_>>(),
            octant_inds: (0..8)
                .map(|_| Vec::with_capacity(octant_max))
                .collect::<Vec<_>>(),
            octant_counts: vec![0; 8],
            full_octants: 0,
            conditioned_octants: 0,
            source_tag: Vec::new(),
            source_count: Vec::new(),
            stop: false,
        }
    }

    #[inline(always)]
    pub fn all_octants_full(&self) -> bool {
        self.full_octants == 8
    }

    #[inline(always)]
    pub fn increment_or_insert_tag(&mut self, tag: u32) -> bool {
        if let Some(ind) = self.source_tag.iter().position(|&x| x == tag) {
            if self.source_count[ind] < self.cond_params.same_source_group_limit as u32 {
                self.source_count[ind] += 1;
                return true;
            } else {
                return false;
            }
        } else {
            self.source_tag.push(tag);
            self.source_count.push(1);
            return true;
        }
    }

    #[inline(always)]
    pub fn decrement_tag(&mut self, tag: u32) {
        if let Some(ind) = self.source_tag.iter().position(|&x| x == tag) {
            self.source_count[ind] -= 1;
        }
    }

    #[inline(always)]
    pub fn max_octant_dist(&self, octant: usize) -> Option<(usize, f32)> {
        self.octant_norm_dists[octant]
            .iter()
            .copied()
            .enumerate()
            .max_by_key(|(_, dist)| OrderedFloat(*dist))
    }

    #[inline(always)]
    pub fn insert_point(
        &mut self,
        octant: usize,
        point: Point3<f32>,
        value: f32,
        dist: f32,
        ind: u32,
        tag: u32,
    ) {
        // println!("tag: {:?}", tag);
        if !self.increment_or_insert_tag(tag) {
            return;
        }

        let clipped_value = self.cond_params.clipped_value(value, dist);
        self.octant_points[octant].push(point);
        self.octant_inds[octant].push(ind);
        self.octant_norm_dists[octant].push(dist);
        self.octant_values[octant].push(clipped_value);
        self.octant_counts[octant] += 1;

        if self.octant_points[octant].len() == 1 {
            self.conditioned_octants += 1;
        }
    }

    #[inline(always)]
    pub fn remove_point(&mut self, octant: usize, ind: usize) {
        let tag = self.source_tag[ind];
        self.octant_points[octant].swap_remove(ind);
        self.octant_inds[octant].swap_remove(ind);
        self.octant_norm_dists[octant].swap_remove(ind);
        self.octant_values[octant].swap_remove(ind);
        self.octant_counts[octant] -= 1;
        self.decrement_tag(tag);
    }

    #[inline(always)]
    pub fn try_insert_point(
        &mut self,
        point: Point3<f32>,
        value: f32,
        dist: f32,
        ind: u32,
        tag: u32,
    ) {
        // if point is further away the primary ellipsoid axis then it cannot be in the ellipsoid
        // and no further points can be in the ellipsoid
        if self.ellipsoid.a * self.ellipsoid.a < dist {
            self.stop = true;
            return;
        }

        //point is not in valid value range -> ignore
        if value < self.cond_params.valid_value_range[0]
            || value > self.cond_params.valid_value_range[1]
        {
            return;
        }

        let local_point = self.ellipsoid.coordinate_system.global_to_local(&point);

        //check if point in ellipsoid
        let h = self.ellipsoid.normalized_local_distance_sq(&local_point);

        if h > 1.0 {
            return;
        }

        //determine octant of point in ellispoid coordinate system
        let octant = octant(&local_point);

        //if octant is not full we can insert point
        if self.octant_points[octant as usize].len() < self.cond_params.max_octant {
            self.insert_point(octant as usize, point, value, h, ind, tag);
            return;
        }

        if let Some((ind, max_dist)) = self.max_octant_dist(octant as usize) {
            if h < max_dist {
                self.remove_point(octant as usize, ind);
                self.insert_point(octant as usize, point, value, h, ind as u32, tag);
                return;
            }

            let h_major = local_point.coords.norm();

            if h_major > max_dist {
                self.full_octants += 1;
                if self.all_octants_full() {
                    self.stop = true;
                }
            }
        }
    }
}

impl ConditioningProvider<Ellipsoid, f32, ConditioningParams> for PointSet<f32> {
    type Shape = Point3<f32>;
    fn query(
        &self,
        point: &Point3<f32>,
        ellipsoid: &Ellipsoid,
        params: &ConditioningParams,
    ) -> (Vec<usize>, Vec<f32>, Vec<Self::Shape>, bool) {
        let mut cond_points = ConditioningDataCollector::new(ellipsoid, params);

        for (point, dist) in self
            .tree
            .nearest_neighbor_iter_with_distance_2(&[point.x, point.y, point.z])
        {
            let env = point.geom();

            let value = self.data[point.data as usize];
            let tag = self.source_tag[point.data as usize];
            cond_points.try_insert_point(
                Point3::new(env[0], env[1], env[2]),
                value,
                dist,
                point.data,
                tag as u32,
            );
            if cond_points.stop {
                break;
            }
        }

        let mut inds: Vec<usize> = cond_points
            .octant_inds
            .into_iter()
            .flatten()
            .map(|i| i as usize)
            .collect();
        let mut points: Vec<_> = cond_points.octant_points.into_iter().flatten().collect();
        // let mut data: Vec<f32> = inds.iter().map(|ind| self.data[*ind].clone()).collect();
        let mut data = cond_points
            .octant_values
            .into_iter()
            .flatten()
            .collect::<Vec<f32>>();

        if data.len() > params.max_n_cond {
            let mut octant_counts = cond_points.octant_counts;
            let mut can_remove_flag =
                if cond_points.conditioned_octants > params.min_conditioned_octants as u8 {
                    vec![true; 8]
                } else {
                    octant_counts.iter().map(|&count| count > 1).collect()
                };

            let mut octant_inds = cond_points
                .octant_norm_dists
                .iter()
                .enumerate()
                .flat_map(|(i, d)| vec![i; d.len()])
                .collect::<Vec<_>>();

            // println!("octant_inds: {:?}", octant_inds);

            let mut dists: Vec<f32> = cond_points
                .octant_norm_dists
                .into_iter()
                .flatten()
                .collect();

            //sort data, inds, points and dists by distance
            let mut sorted_inds = (0..inds.len()).collect::<Vec<_>>();
            sorted_inds.sort_by_key(|i| OrderedFloat(dists[*i]));

            let mut permutation = Permutation::oneline(sorted_inds).inverse();

            permutation.apply_slice_in_place(&mut inds);
            permutation.apply_slice_in_place(&mut points);
            permutation.apply_slice_in_place(&mut dists);
            permutation.apply_slice_in_place(&mut data);
            permutation.apply_slice_in_place(&mut octant_inds);

            let mut end = octant_inds.len();

            while data.len() > params.max_n_cond {
                let Some(r_ind) = octant_inds[0..end]
                    .iter()
                    .rev()
                    .position(|oct| can_remove_flag[*oct])
                else {
                    break;
                };

                let ind = end - r_ind - 1;

                end = ind;

                let octant = octant_inds[ind];

                //remove value
                inds.swap_remove(ind);
                points.swap_remove(ind);
                dists.swap_remove(ind);
                data.swap_remove(ind);
                octant_inds.swap_remove(ind);

                //update octant counts
                octant_counts[octant] -= 1;

                //update conditioned octants as needed
                if octant_counts[octant] == 0 {
                    cond_points.conditioned_octants -= 1;
                }

                //update can remove flag
                if cond_points.conditioned_octants < params.min_conditioned_octants as u8 {
                    can_remove_flag = octant_counts.iter().map(|&count| count > 1).collect();
                }
            }
        }

        let res = cond_points.conditioned_octants >= params.min_conditioned_octants as u8
            && data.len() >= params.min_n_cond;

        (inds, data, points, res)
    }

    fn points(&self) -> &[Point3<f32>] {
        self.points.as_slice()
    }

    fn data(&self) -> &[f32] {
        self.data.as_slice()
    }

    fn data_mut(&mut self) -> &mut [f32] {
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

        let cond_params = ConditioningParams::default();

        let mut cond_points = ConditioningDataCollector::new(&ellipsoid, &cond_params);

        for point in points.iter() {
            let dist = distance(&point, &query_point);
            cond_points.try_insert_point(*point, 0.0, dist, 0, 0);
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

        let point_set = PointSet::new(points.clone(), data, tags);
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
        let point_set = PointSet::new(points, data, tags);
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

        let point_set = PointSet::new(points, data, tags);

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
        let point_set = PointSet::new(points.clone(), data, tags);
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
