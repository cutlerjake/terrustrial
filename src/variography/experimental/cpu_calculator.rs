use crate::geometry::variogram_tolerance::VariogramTolerance;
use crate::spatial_database::rtree_point_set::point_set::PointSet;

use super::{ExperimentalVarigoramCalculator, LagBounds};

use itertools::{izip, Itertools};
use nalgebra::{Point3, UnitQuaternion};
use rayon::iter::{ParallelBridge, ParallelIterator};
use rstar::AABB;

#[derive(Clone)]
pub struct CPUCalculator {
    //data tree
    data: PointSet<f32>,

    //lag bounds
    lags: Vec<LagBounds>,

    //tolerance
    a: f32,
    a_tol: f32,
    b: f32,
    b_tol: f32,
}

impl CPUCalculator {
    pub fn new(
        data: PointSet<f32>,
        lags: Vec<LagBounds>,
        a: f32,
        a_tol: f32,
        b: f32,
        b_tol: f32,
    ) -> Self {
        Self {
            data,
            lags,
            a,
            a_tol,
            b,
            b_tol,
        }
    }
}

impl ExperimentalVarigoramCalculator for CPUCalculator {
    fn calculate_for_orientations(
        &self,
        orientations: &[UnitQuaternion<f32>],
    ) -> Vec<super::ExpirmentalVariogram> {
        let runs = orientations
            .iter()
            .cartesian_product(self.lags.iter())
            .enumerate();

        let mut counts = vec![0; orientations.len() * self.lags.len()];
        let mut semivar = vec![0f32; orientations.len() * self.lags.len()];

        let exp_data = runs
            .par_bridge()
            .map(|(index, (orientation, lag_bound))| {
                let mut tolerance = VariogramTolerance::new(
                    Point3::origin(),
                    lag_bound.ub - lag_bound.lb,
                    self.a,
                    self.a_tol,
                    self.b,
                    self.b_tol,
                    *orientation,
                );

                let mut count = 0;
                let mut semivariance = 0f32;

                for (i, point) in self.data.points.iter().enumerate() {
                    //value of current point
                    let value = self.data.data[i];

                    //set current point as base
                    tolerance.set_base(*point);

                    //offset by lag
                    tolerance.offset_along_axis(lag_bound.lb);

                    //query for points within tolerance
                    let bounding_box = tolerance.loose_aabb();
                    let rtree_box = AABB::from_corners(
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

                    for pair_data in self.data.tree.locate_in_envelope_intersecting(&rtree_box) {
                        let pair_point = Point3::from(*pair_data.geom());
                        let pair_ind = pair_data.data;
                        let pair_value = self.data.data[pair_ind as usize];

                        //skip if same point
                        if pair_ind as usize == i {
                            continue;
                        }

                        if !tolerance.contains_point(pair_point) {
                            continue;
                        }

                        count += 1;
                        semivariance += (value - pair_value) * (value - pair_value);
                    }
                }

                (index, count, semivariance)
            })
            .collect::<Vec<_>>();

        for (ind, count, semivariance) in exp_data.into_iter() {
            counts[ind] = count;
            semivar[ind] = semivariance;
        }

        let mut variograms = Vec::with_capacity(orientations.len());

        for (semivar_chunk, count_chunk, orientation) in izip!(
            semivar.chunks(self.lags.len()),
            counts.chunks(self.lags.len()),
            orientations
        ) {
            let mut variogram = super::ExpirmentalVariogram {
                orientation: *orientation,
                lags: self.lags.clone(),
                semivariance: semivar_chunk.to_vec(),
                counts: count_chunk.to_vec(),
            };

            variogram
                .semivariance
                .iter_mut()
                .zip(variogram.counts.iter())
                .for_each(|(v, &c)| {
                    *v /= 2f32 * c as f32;
                });

            variograms.push(variogram);
        }
        variograms
    }
}

#[cfg(test)]
mod test {
    use nalgebra::UnitQuaternion;

    use super::*;
    use rand::seq::SliceRandom;
    use rand::thread_rng;

    #[test]
    fn cpu_vgram() {
        let path = r"C:\Users\2jake\OneDrive\Desktop\foresight\testing_geostats_data\point_cloud\point_set_300_reduced.csv";
        let mut reader = csv::Reader::from_path(path).expect("Unable to open file.");

        let mut coords = Vec::new();
        let mut values = Vec::new();

        for record in reader.deserialize() {
            let (x, y, z, _dx, _dy, _dz, v): (f32, f32, f32, f32, f32, f32, f32) = record.unwrap();
            coords.push(Point3::new(x, y, z));
            values.push(v);
        }
        println!("Values: {:?}", values);
        let mut ind = (0..coords.len()).collect::<Vec<_>>();
        let mut rng = thread_rng();
        let tags = (0..coords.len()).collect::<Vec<_>>();
        for _ in 0..100 {
            ind.shuffle(&mut rng);

            let shuffled_coords = ind.iter().map(|&i| coords[i]).collect::<Vec<_>>();
            let shuffled_values = ind.iter().map(|&i| values[i]).collect::<Vec<_>>();
            let point_set = PointSet::new(shuffled_coords, shuffled_values, tags.clone());
            let quat = UnitQuaternion::identity();

            let lag_lb = (0..15).map(|i| i as f32 * 10f32).collect::<Vec<_>>();
            let lag_ub = (0..15).map(|i| (i + 1) as f32 * 10f32).collect::<Vec<_>>();
            let lag_bounds = lag_lb
                .iter()
                .zip(lag_ub.iter())
                .map(|(lb, ub)| LagBounds::new(*lb, *ub))
                .collect::<Vec<_>>();

            let cpu_calc = CPUCalculator::new(point_set, lag_bounds, 10f32, 0.1f32, 10f32, 0.1f32);

            let vgrams = cpu_calc.calculate_for_orientations(&[quat]);

            for vgram in vgrams.iter() {
                println!("{:?}", vgram.semivariance);
            }
            println!("");
        }
    }
}
