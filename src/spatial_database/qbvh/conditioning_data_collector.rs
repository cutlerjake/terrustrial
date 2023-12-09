use crate::{
    geometry::ellipsoid::Ellipsoid,
    spatial_database::{coordinate_system::octant, qbvh::point_set::PointSet},
};

use nalgebra::{distance, Point3, SimdBool as _, SimdPartialOrd, SimdValue};
use parry3d::{
    bounding_volume::SimdAabb,
    math::{SimdBool, SimdReal, SIMD_WIDTH},
    partitioning::SimdBestFirstVisitStatus,
    simba::simd::AutoSimd,
};

use super::simd_n_best_first_visitor::SimdNBestFirstVisitor;

pub enum InsertionResult {
    InsertedNotFull,
    InsertedFull,
    NotInserted,
    NotInsertedOutOfRange,
}

/// A visitor the computes the conditioning data for a simulation point
/// the closest n_cond points are retained in each octant
pub struct ConditioningDataCollector<'a, 'b, T> {
    pub n_cond: usize,
    pub point: Point3<f32>,
    pub simd_point: Point3<AutoSimd<[f32; 4]>>,
    pub point_set: &'a PointSet<T>,
    pub max_accepted_dist: f32,
    pub ellipsoid: &'b Ellipsoid,
    pub octant_points: Vec<Vec<Point3<f32>>>,
    pub octant_distances: Vec<Vec<f32>>,
    pub octant_inds: Vec<Vec<u32>>,
    pub octant_max_inds: Vec<usize>,
    pub octant_max_distances: Vec<f32>,
    pub full_octants: u8,
}

impl<'a, 'b, T> ConditioningDataCollector<'a, 'b, T> {
    pub fn new(
        point: Point3<f32>,
        ellipsoid: &'b Ellipsoid,
        n_cond: usize,
        point_set: &'a PointSet<T>,
    ) -> Self {
        let simd_point: Point3<AutoSimd<[f32; 4]>> = Point3::new(
            AutoSimd::splat(point.coords.x),
            AutoSimd::splat(point.coords.y),
            AutoSimd::splat(point.coords.z),
        );
        ConditioningDataCollector {
            point,
            simd_point,
            n_cond,
            point_set,
            max_accepted_dist: f32::MAX,
            ellipsoid,
            octant_points: vec![Vec::new(); 8],
            octant_distances: vec![Vec::new(); 8],
            octant_inds: vec![Vec::new(); 8],
            octant_max_inds: vec![0; 8],
            octant_max_distances: vec![f32::MAX; 8],
            full_octants: 0,
        }
    }

    #[inline(always)]
    fn update_max_octant_ind(&mut self, octant: usize) {
        self.octant_max_inds[octant] = self.octant_distances[octant]
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0;
    }

    #[inline(always)]
    fn update_max_accepted_octant_dist(&mut self, octant: usize) {
        self.octant_max_distances[octant] =
            self.octant_distances[octant][self.octant_max_inds[octant]];
    }

    #[inline(always)]
    pub fn all_octants_full(&self) -> bool {
        self.full_octants == 8
    }

    #[inline(always)]
    pub fn update_max_dist(&mut self) {
        if self.all_octants_full() {
            self.max_accepted_dist = *self
                .octant_distances
                .iter()
                .flatten()
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap();
        }
    }

    #[inline(always)]
    pub fn insert_octant_point(
        &mut self,
        point: Point3<f32>,
        dist: f32,
        ind: u32,
    ) -> InsertionResult {
        // if point is further away the primary ellipsoid axis then it cannot be in the ellipsoid
        // and no further points can be in the ellipsoid
        if self.ellipsoid.a < dist {
            return InsertionResult::NotInserted;
        }
        //check if point in ellipsoid
        if !self.ellipsoid.contains(&point) {
            return InsertionResult::NotInserted;
        }

        //determine octant of point in ellispoid coordinate system
        let local_point = self.ellipsoid.coordinate_system.global_to_local(&point);
        let octant = octant(&local_point);

        //get octant points and distances
        let points = &mut self.octant_points[octant as usize];
        let distances = &mut self.octant_distances[octant as usize];
        let inds = &mut self.octant_inds[octant as usize];

        //if octant is not full, insert point and return
        if points.len() < self.n_cond {
            //insert
            points.push(point);
            distances.push(dist);
            inds.push(ind);

            //if octant is now full update max octant dists and max dist needs updating

            if points.len() == self.n_cond {
                //update full octant count
                self.full_octants += 1;
                //update max ind
                self.update_max_octant_ind(octant as usize);
                //update max accepted dist
                self.update_max_accepted_octant_dist(octant as usize);

                //update max accepted dist
                self.update_max_dist();

                //octant now full
                return InsertionResult::InsertedFull;
            }
            //octant not full
            return InsertionResult::InsertedNotFull;
        }

        //else if point is closer than furthest point, insert point and remove furthest point
        if dist < self.octant_max_distances[octant as usize] {
            //insert point
            self.octant_points[octant as usize][self.octant_max_inds[octant as usize]] = point;
            self.octant_distances[octant as usize][self.octant_max_inds[octant as usize]] = dist;
            self.octant_inds[octant as usize][self.octant_max_inds[octant as usize]] = ind;

            //update max ind
            self.update_max_octant_ind(octant as usize);
            //update max accepted dist
            self.update_max_accepted_octant_dist(octant as usize);

            //update max accepted dist
            self.update_max_dist();

            return InsertionResult::InsertedFull;
        }

        InsertionResult::NotInserted
    }
}

impl<'a, 'b, LeafData, T> SimdNBestFirstVisitor<LeafData, SimdAabb>
    for ConditioningDataCollector<'a, 'b, T>
{
    type Result = ();

    fn visit(
        &mut self,
        threshold: &mut f32,
        bv: &SimdAabb,
        data: Option<[Option<&u32>; SIMD_WIDTH]>,
    ) -> SimdBestFirstVisitStatus<Self::Result> {
        //mask to select only element with dist less than current furthest point
        let dists = bv.distance_to_local_point(&Point3::splat(self.point));
        let mask = dists.simd_lt(SimdReal::splat(*threshold));

        if let Some(data) = data {
            let bitmask = mask.bitmask();
            let mut weights = [0.0; SIMD_WIDTH];
            let mut mask = [false; SIMD_WIDTH];
            //let mut results = [None; SIMD_WIDTH];

            for ii in 0..SIMD_WIDTH {
                if (bitmask & (1 << ii)) != 0 && data[ii].is_some() {
                    // get point index
                    let part_id = *data[ii].unwrap();
                    // get point
                    let point = self.point_set.points[part_id as usize];
                    // get distance from query point to point
                    let dist = distance(&point, &self.point);

                    //insert point if distance is less than current furthest point
                    match self.insert_octant_point(point, dist, part_id) {
                        InsertionResult::InsertedNotFull => {
                            //If inserted dist <= threshold
                            mask[ii] = true;
                        }
                        InsertionResult::InsertedFull => {
                            //Conditioning set full -> must update threshold
                            *threshold = self.max_accepted_dist;
                            //If inserted: dist <= threshold
                            mask[ii] = true;
                        }
                        InsertionResult::NotInserted => {}
                        InsertionResult::NotInsertedOutOfRange => {
                            return SimdBestFirstVisitStatus::ExitEarly(None);
                        }
                    }

                    weights[ii] = dist;
                }
            }

            SimdBestFirstVisitStatus::MaybeContinue {
                weights: SimdReal::from(weights),
                mask: SimdBool::from(mask),
                results: [None; SIMD_WIDTH],
            }
        } else {
            SimdBestFirstVisitStatus::MaybeContinue {
                weights: dists,
                mask,
                results: [None; SIMD_WIDTH],
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::hint::black_box;

    use crate::spatial_database::coordinate_system::CoordinateSystem;
    use crate::spatial_database::qbvh::n_best_first::NBestFirst;

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
    fn conditioning_insertion() {
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

        let mut cond_points =
            ConditioningDataCollector::new(query_point, &ellipsoid, n_cond, &point_set);

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
    fn collector() {
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

        let mut cond_points =
            ConditioningDataCollector::new(query_point, &ellipsoid, n_cond, &point_set);
        point_set.tree.traverse_n_best_first(&mut cond_points);

        let mut true_points = points.clone();
        true_points.sort_by(|a, b| {
            let dist_a = distance(a, &query_point);
            let dist_b = distance(b, &query_point);
            dist_a.partial_cmp(&dist_b).unwrap()
        });

        println!("cond points: {:?}", cond_points.octant_points);

        let mut c_points = cond_points
            .octant_points
            .iter()
            .flatten()
            .collect::<Vec<_>>();
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
            if true_points[i] != *c_points[i] {
                panic!();
            }
        }

        // println!("true points: {:?}", true_points[0..n_cond].to_vec());
        // println!("walker points: {:?}", cond_points.closest_points);
    }

    #[test]
    fn speed_test() {
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
        let n_cond = 8;
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

            let mut cond_points =
                ConditioningDataCollector::new(query_point, &ellipsoid, n_cond, &point_set);
            black_box(
                point_set
                    .tree
                    .traverse_n_best_first(black_box(&mut cond_points)),
            );
        }
        println!(
            "Speed test completed: queries: {:?}, time: {:?}",
            10000,
            time.elapsed()
        );
    }
}
