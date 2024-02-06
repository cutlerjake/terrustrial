use crate::geometry::variogram_tolerance::VariogramTolerance;
use crate::spatial_database::rtree_point_set::point_set::PointSet;

use super::{ExperimentalVarigoramCalculator, LagBounds};

use itertools::{izip, Itertools};
use nalgebra::{Point3, UnitVector3};
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
        orientations: &[(UnitVector3<f32>, f32)],
    ) -> Vec<super::ExpirmentalVariogram> {
        let runs = orientations
            .iter()
            .cartesian_product(self.lags.iter())
            .enumerate();

        let mut counts = vec![0; orientations.len() * self.lags.len()];
        let mut semivar = vec![0f32; orientations.len() * self.lags.len()];

        let exp_data = runs
            .par_bridge()
            .map(|(index, ((axis, rot), lag_bound))| {
                let mut tolerance = VariogramTolerance::new(
                    Point3::origin(),
                    lag_bound.ub - lag_bound.lb,
                    self.a,
                    self.a_tol,
                    self.b,
                    self.b_tol,
                    *axis,
                    *rot,
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

        for (semivar_chunk, count_chunk, (axis, rot)) in izip!(
            semivar.chunks(self.lags.len()),
            counts.chunks(self.lags.len()),
            orientations
        ) {
            let mut variogram = super::ExpirmentalVariogram {
                axis: *axis,
                rot: *rot,
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
    use nalgebra::{UnitQuaternion, Vector3};

    use super::*;

    #[test]
    fn cpu_vgram() {
        let path = r"C:\Users\2jake\OneDrive - McGill University\Fall2022\MIME525\Project4\drillholes_jake.csv";
        let mut reader = csv::Reader::from_path(path).expect("Unable to open file.");

        let mut coords = Vec::new();
        let mut values = Vec::new();

        for record in reader.deserialize() {
            let (x, y, z, v): (f32, f32, f32, f32) = record.unwrap();
            coords.push(Point3::new(x, y, z));
            values.push(v);
        }

        let point_set = PointSet::new(coords, values);

        let lag_lb = (0..15).map(|i| i as f32 * 10f32).collect::<Vec<_>>();
        let lag_ub = (0..15).map(|i| (i + 1) as f32 * 10f32).collect::<Vec<_>>();
        let lag_bounds = lag_lb
            .iter()
            .zip(lag_ub.iter())
            .map(|(lb, ub)| LagBounds::new(*lb, *ub))
            .collect::<Vec<_>>();

        // create quaternions
        let mut axis_angles = vec![];
        for i in 1..5 {
            for j in 0..5 {
                for k in 0..5 {
                    let axis =
                        UnitVector3::new_normalize(Vector3::from([i as f32, j as f32, k as f32]));
                    let rot = 0.0;
                    axis_angles.push((axis, rot));
                }
            }
        }

        let cpu_calc = CPUCalculator::new(point_set, lag_bounds, 10f32, 0.1f32, 10f32, 0.1f32);

        let vgrams = cpu_calc.calculate_for_orientations(&axis_angles);

        for vgram in vgrams.iter() {
            println!("{:?}", vgram);
            break;
        }
    }
}
