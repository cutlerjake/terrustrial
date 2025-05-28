use crate::{
    geometry::variogram_tolerance::{collect_points, VariogramTolerance},
    spatial_database::{coordinate_system::CoordinateSystem, SpatialAcceleratedDB},
};

use super::LagBounds;

use itertools::{izip, Itertools};
use rayon::iter::{ParallelBridge, ParallelIterator};
use rstar::AABB;
use ultraviolet::{DRotor3, DVec3};

#[derive(Clone)]
pub struct CPUCalculator {
    //data tree
    data: SpatialAcceleratedDB<f64>,

    //lag bounds
    lags: Vec<LagBounds>,

    //tolerance
    a: f64,
    a_tol: f64,
    b: f64,
    b_tol: f64,
}

impl CPUCalculator {
    pub fn new(
        data: SpatialAcceleratedDB<f64>,
        lags: Vec<LagBounds>,
        a: f64,
        a_tol: f64,
        b: f64,
        b_tol: f64,
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

    pub fn calculate_for_orientations_vectorized(
        &self,
        orientations: &[DRotor3],
    ) -> Vec<super::ExpirmentalVariogram> {
        let runs = orientations
            .iter()
            .cartesian_product(self.lags.iter())
            .enumerate();

        let mut counts = vec![0; orientations.len() * self.lags.len()];
        let mut semivar = vec![0.0; orientations.len() * self.lags.len()];

        let exp_data = runs
            .par_bridge()
            .map(|(index, (orientation, lag_bound))| {
                let cs = CoordinateSystem::new(DVec3::zero(), *orientation);
                let mut tolerance = VariogramTolerance::new(
                    DVec3::zero(),
                    lag_bound.ub - lag_bound.lb,
                    self.a,
                    self.a_tol,
                    self.b,
                    self.b_tol,
                    cs,
                );

                let mut count = 0;
                let mut semivariance = 0.0;

                for (i, point) in self.data.supports.iter().map(|s| s.center()).enumerate() {
                    //value of current point
                    let value = self.data.data[i];

                    //set current point as base
                    tolerance.set_base(point);

                    //offset by lag
                    tolerance.offset_along_axis(lag_bound.lb);

                    //query for points within tolerance
                    let bounding_box = tolerance.loose_aabb();
                    let rtree_box = AABB::from_corners(
                        [
                            bounding_box.mins().x,
                            bounding_box.mins().y,
                            bounding_box.mins().z,
                        ],
                        [
                            bounding_box.maxs().x,
                            bounding_box.maxs().y,
                            bounding_box.maxs().z,
                        ],
                    );

                    let (points, vals) = self
                        .data
                        .tree
                        .locate_in_envelope_intersecting(&rtree_box)
                        .filter_map(|pair_data: &crate::spatial_database::SpatialData| {
                            let pair_point = pair_data.support().center();
                            let pair_ind = pair_data.data_idx();
                            let pair_value = self.data.data[pair_ind as usize];

                            //skip if same point
                            if pair_ind as usize == i {
                                return None;
                            }

                            Some((pair_point, pair_value))
                        })
                        .unzip::<_, _, Vec<_>, Vec<_>>();

                    // Collect points into vec of [`DVec3x4`]
                    let point = collect_points(points.as_slice());

                    for (val_ind, (point, _val)) in point.iter().zip(vals.iter()).enumerate() {
                        let mask = tolerance.contains_point_vectorized(*point);

                        for (i, m) in mask.into_iter().enumerate() {
                            if m {
                                count += 1;
                                semivariance += (value - vals[4 * val_ind + i])
                                    * (value - vals[4 * val_ind + i]);
                            }
                        }
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
                    *v /= 2.0 * c as f64;
                });

            variograms.push(variogram);
        }
        variograms
    }
}

#[cfg(test)]
mod test {

    use crate::geometry::support::Support;

    use super::*;
    use rand::seq::SliceRandom;
    use rand::thread_rng;

    #[test]
    fn cpu_vgram() {
        let path = r"F:\git repos\implicit\point_set.csv";
        let mut reader = csv::Reader::from_path(path).expect("Unable to open file.");

        let mut coords = Vec::new();
        let mut values = Vec::new();

        for record in reader.deserialize() {
            let (x, y, u, _v): (f64, f64, f64, f64) = record.unwrap();
            coords.push(Support::Point(DVec3::new(x, y, 0.0)));
            values.push(u);
        }

        println!("Values: {:?}", values.len());
        let mut ind = (0..coords.len()).collect::<Vec<_>>();
        let mut rng = thread_rng();
        for i in 0..1 {
            println!("Iteration: {}", i);
            ind.shuffle(&mut rng);

            let shuffled_coords = ind.iter().map(|&i| coords[i]).collect::<Vec<_>>();
            let shuffled_values = ind.iter().map(|&i| values[i]).collect::<Vec<_>>();
            let point_set = SpatialAcceleratedDB::new(shuffled_coords, shuffled_values);
            let rot = DRotor3::identity();

            let lag_lb = (0..15).map(|i| i as f64 * 10f64).collect::<Vec<_>>();
            let lag_ub = (0..15).map(|i| (i + 1) as f64 * 10f64).collect::<Vec<_>>();
            let lag_bounds = lag_lb
                .iter()
                .zip(lag_ub.iter())
                .map(|(lb, ub)| LagBounds::new(*lb, *ub))
                .collect::<Vec<_>>();

            let cpu_calc = CPUCalculator::new(
                point_set,
                lag_bounds,
                5f64,
                45.0f64.to_radians(),
                5f64,
                45.0f64.to_radians(),
            );

            let time = std::time::Instant::now();
            let vgrams = cpu_calc.calculate_for_orientations_vectorized(&[rot]);
            println!("Vectorized Time: {:?}", time.elapsed());

            for vgram in vgrams.iter() {
                println!("{:?}", vgram.semivariance);
            }
            for vgram in vgrams.iter() {
                println!("{:?}", vgram.counts);
            }
            println!("");
        }
    }
}
