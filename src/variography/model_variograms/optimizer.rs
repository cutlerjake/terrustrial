use nalgebra::{Quaternion, Unit, UnitVector3};
use nalgebra::{UnitQuaternion, Vector3};
use ordered_float::OrderedFloat;

use crate::variography::experimental::cuda_calculator::CudaCalculator;
use crate::variography::experimental::ExperimentalVarigoramCalculator;

use super::aniso_fitter::AnisoFitter;
use super::iso_fitter::{CompositeVariogramFitter, VariogramType};
use argmin::core::Error;
use argmin::core::{CostFunction, Executor};
use argmin::solver::particleswarm::ParticleSwarm;

pub struct VariogramOptimizer {
    params: CudaCalculator,
    structures: Vec<VariogramType>,
}

impl CostFunction for &&mut VariogramOptimizer {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, params: &Self::Param) -> Result<Self::Output, Error> {
        //cost for all paericles to be computed in bulk
        let _ = params;
        panic!()
    }

    fn bulk_cost<'a, P>(&self, params: &'a [P]) -> Result<Vec<Self::Output>, Error>
    where
        P: std::borrow::Borrow<Self::Param> + argmin::core::SyncAlias,
        Self::Output: argmin::core::SendAlias,
        Self: argmin::core::SyncAlias,
    {
        //map params to rotations
        let rotations = params
            .iter()
            .map(|p| {
                let ang_1 = p.borrow()[0];
                let ang_2 = p.borrow()[1];
                let ang_3 = p.borrow()[2];

                UnitQuaternion::from_euler_angles(ang_1 as f32, ang_2 as f32, ang_3 as f32)
            })
            .collect::<Vec<_>>();

        //compute experimental variograms in all directions
        let exp_variograms = self.params.calculate_for_orientations(rotations.as_slice());

        //fit model for each experimental variogram
        let ranges = exp_variograms.iter().map(|v| {
            let mut fitter = CompositeVariogramFitter::new(
                v.lags.iter().map(|l| l.mid_point()).collect(),
                v.semivariance.clone(),
                self.structures.clone(),
            );
            let _ = fitter.fit();
            -fitter.range()
        });

        Ok(ranges.collect())
    }
}
impl VariogramOptimizer {
    pub fn new(params: CudaCalculator, structures: Vec<VariogramType>) -> Self {
        Self { params, structures }
    }

    pub fn fit(&mut self) {
        //identify direction with greatest range
        let high = vec![2f64; 3];
        let low = vec![0f64; 3];

        let solver = ParticleSwarm::new((low, high), 40);

        let particle;
        {
            let res = Executor::new(&self, solver)
                .configure(|state| state.max_iters(10))
                .run()
                .unwrap();
            particle = res.state.best_individual.unwrap();
        }

        let quat = UnitQuaternion::from_euler_angles(
            particle.position[0],
            particle.position[1],
            particle.position[2],
        );
        // let axis = quat * Vector3::x_axis();
        let axis = quat * FORWARD;

        //find intermiediate direction
        let perps = gen_perp_fan(axis, 100);

        //compute quaternion from perp to x-axis
        let perp_quats = perps
            .iter()
            .map(|p| {
                UnitQuaternion::rotation_between(&p.try_cast::<f32>().unwrap(), &FORWARD).unwrap()
            })
            .collect::<Vec<_>>();

        //compute experimental variograms in all directions
        let exp_variograms = self
            .params
            .calculate_for_orientations(perp_quats.as_slice());

        //get perp with greatest range
        let ranges = exp_variograms.iter().map(|v| {
            let mut fitter = CompositeVariogramFitter::new(
                v.lags.iter().map(|l| l.mid_point()).collect(),
                v.semivariance.clone(),
                self.structures.clone(),
            );
            let _ = fitter.fit();
            fitter.range()
        });

        //get perp with greatest range
        let max_range = ranges
            .enumerate()
            .max_by_key(|(_, r)| OrderedFloat(*r))
            .unwrap()
            .0;

        let perp = perps[max_range];

        // cross product to construct minor axis
        let minor_axis = Unit::new_normalize(axis.cross(&perp));

        //fit major, intermediate, and minor variogram models
        let rotations = vec![
            Unit::new_unchecked(Quaternion::from_vector(
                quat.coords.try_cast::<f32>().unwrap(),
            )),
            Unit::new_unchecked(Quaternion::from_vector(
                perp_quats[max_range].coords.try_cast::<f32>().unwrap(),
            )),
            UnitQuaternion::rotation_between(&minor_axis.try_cast::<f32>().unwrap(), &FORAWRD)
                .unwrap(),
        ];

        let exp_vgram = self.params.calculate_for_orientations(rotations.as_slice());

        //fit each experimental variogram
        let mut aniso_fitter = AnisoFitter::new(
            exp_vgram[0]
                .lags
                .iter()
                .map(|l| l.mid_point() as f64)
                .collect(),
            exp_vgram[0]
                .semivariance
                .iter()
                .cloned()
                .map(|v| v as f64)
                .collect(),
            exp_vgram[1]
                .semivariance
                .iter()
                .cloned()
                .map(|v| v as f64)
                .collect(),
            exp_vgram[2]
                .semivariance
                .iter()
                .cloned()
                .map(|v| v as f64)
                .collect(),
            self.structures.clone(),
        );

        let (lower, upper) = aniso_fitter.get_bounds();

        let solver = ParticleSwarm::new((lower, upper), 40);

        let res = Executor::new(&mut aniso_fitter, solver)
            .configure(|state| state.max_iters(100))
            .run()
            .unwrap();

        let params = res.state.best_individual.unwrap().position;

        let mut aniso_vgram = aniso_fitter.aniso_variogram_from_slice(&params);

        //create quaternion from major, intermediate, and minor axes
        let quat = UnitQuaternion::from_basis_unchecked(&[
            axis.into_inner().cast::<f32>(),
            perp.into_inner().cast::<f32>(),
            minor_axis.into_inner().cast::<f32>(),
        ]);

        aniso_vgram.set_orientation(quat);
    }
}

pub fn gen_perp_fan(axis: UnitVector3<f64>, n_perp: u32) -> Vec<UnitVector3<f64>> {
    //generate fan of Unituaternions perpendicular to original_quat
    let mut perps = Vec::with_capacity(n_perp as usize);

    //get axis of rotation
    let x = axis[0];
    let y = axis[1];
    let z = axis[2];

    //initial perpindicualr vector
    let perp = Unit::new_normalize(Vector3::new(
        f64::copysign(z, x),
        f64::copysign(z, y),
        -f64::copysign(x, z) - f64::copysign(y, z),
    ));

    //generate fan
    for i in 0..n_perp {
        let angle = 2.0 * std::f64::consts::PI * (i as f64 / n_perp as f64);
        let rotation = UnitQuaternion::from_axis_angle(&axis, angle);
        perps.push(rotation * perp);
    }

    perps
}

#[cfg(test)]
mod test {
    use cudarc::nvrtc::Ptx;
    use nalgebra::Point3;

    use crate::variography::experimental::cuda_calculator::CudaFlatBVH;
    use crate::variography::experimental::LagBounds;

    use super::*;

    #[test]
    fn perp_fan() {
        let axis = Unit::new_normalize(Vector3::new(1.0, 1.0, 1.0));
        let perps = gen_perp_fan(axis, 10);

        for p in perps.iter() {
            assert!((p.dot(&axis)).abs() < 1e-6);
        }
    }

    #[test]
    fn orientation() {
        //read points from csv
        //let path = r"C:\GitRepos\terrustrial\data\walker.csv";
        let path = r"C:\Users\2jake\OneDrive - McGill University\Fall2022\MIME525\Project4\drillholes_jake.csv";
        let mut reader = csv::Reader::from_path(path).expect("Unable to open file.");

        let mut coords = Vec::new();
        let mut values = Vec::new();

        for record in reader.deserialize() {
            let (x, y, z, v): (f32, f32, f32, f32) = record.unwrap();
            coords.push(Point3::new(x, y, z));
            values.push(v);
        }

        let bvh = CudaFlatBVH::new(coords, values);

        //create 10 lag bounds
        let lag_lb = (0..15).map(|i| i as f32 * 10f32).collect::<Vec<_>>();
        let lag_ub = (0..15).map(|i| (i + 1) as f32 * 10f32).collect::<Vec<_>>();
        let lag_bounds = lag_lb
            .iter()
            .zip(lag_ub.iter())
            .map(|(lb, ub)| LagBounds::new(*lb, *ub))
            .collect::<Vec<_>>();

        //get device
        let device = cudarc::driver::CudaDevice::new(0).unwrap();
        device
            .load_ptx(
                Ptx::from_file(".\\src\\variography\\experimental\\kernel_bvh.ptx"),
                "vgram",
                &["vgram_kernel"],
            )
            .expect("unable to load kernel");
        let vgram_kernel = device.get_func("vgram", "vgram_kernel").unwrap();

        let gpu_vgram = CudaCalculator::new(
            device,
            vgram_kernel,
            bvh,
            lag_bounds,
            10f32,
            0.1f32,
            10f32,
            0.1f32,
        );

        let structures = vec![
            VariogramType::IsoSphericalNoNugget(Default::default()),
            VariogramType::Nugget(Default::default()),
        ];

        let mut opt = VariogramOptimizer::new(gpu_vgram, structures);

        opt.fit()
    }
}
