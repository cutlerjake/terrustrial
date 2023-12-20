use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaFunction, CudaSlice, LaunchAsync, LaunchConfig};
use itertools::izip;
use nalgebra::UnitQuaternion;

use super::{
    ExpirmentalVariogram, Float3, GPUBVHFlatNode, GPUFlatBVH, GPUQuaternion, GPUVariogramParams,
    LagBounds,
};

pub struct GPUVGRAM {
    //cuda device
    device: Arc<CudaDevice>,

    //cuda kernel
    kernel: CudaFunction,

    //bvh nodes
    num_nodes: u32,
    nodes: CudaSlice<GPUBVHFlatNode>,

    //data locations and values
    points: CudaSlice<Float3>,
    values: CudaSlice<f32>,
    num_data: u32,

    //lag bounds
    lags: CudaSlice<LagBounds>,
    num_lags: u32,
    h_lags: Vec<LagBounds>,

    //tolerance
    a: f32,
    a_tol: f32,
    a_dist_threshold: f32,
    b: f32,
    b_tol: f32,
    b_dist_threshold: f32,
}

impl GPUVGRAM {
    pub fn new(
        device: Arc<CudaDevice>,
        kernel: CudaFunction,
        bvh: GPUFlatBVH,
        lag_bounds: Vec<LagBounds>,
        a: f32,
        a_tol: f32,
        b: f32,
        b_tol: f32,
    ) -> Self {
        //distance at which max cylindar dimension reached along major and minor axis
        let a_dist_threshold = a / a_tol.tan();
        let b_dist_threshold = b / b_tol.tan();

        //copy points and values to device
        let (points, values): (Vec<_>, Vec<_>) = bvh
            .bvh_points()
            .iter()
            .map(|bvh_point| (bvh_point.coords, bvh_point.data))
            .unzip();
        let n_data = points.len() as u32;
        let dev_points = device
            .htod_copy(points)
            .expect("Unable to copy points to device");
        let dev_values = device
            .htod_copy(values)
            .expect("Unable to copy values to device");

        //copy bvh nodes to device
        let n_nodes = bvh.nodes.len() as u32;
        let dev_nodes = device
            .htod_copy(bvh.nodes)
            .expect("Unable to copy bvh nodes to device");

        //copy lag bounds to device
        let n_lags = lag_bounds.len() as u32;
        let dev_lag_bounds = device
            .htod_copy(lag_bounds.clone())
            .expect("Unable to copy lag bounds to device");

        Self {
            device,
            kernel,
            num_nodes: n_nodes,
            nodes: dev_nodes,
            points: dev_points,
            values: dev_values,
            num_data: n_data,
            lags: dev_lag_bounds,
            h_lags: lag_bounds,
            num_lags: n_lags,
            a: a,
            a_dist_threshold: a_dist_threshold,
            a_tol,
            b: b,
            b_dist_threshold: b_dist_threshold,
            b_tol,
        }
    }

    pub fn compute_for_orientations(
        &self,
        rotations: Vec<UnitQuaternion<f32>>,
    ) -> Vec<ExpirmentalVariogram> {
        //create inverse of rotations
        let inv_rotations: Vec<_> = rotations.iter().map(|r| r.inverse()).collect();

        //map quaternions to GPUQuaternion
        let n_rotations = rotations.len() as u32;
        let gpu_rotations = rotations
            .iter()
            .map(|r| GPUQuaternion::from(*r))
            .collect::<Vec<_>>();

        let gpu_inv_rotations = inv_rotations
            .iter()
            .map(|r| GPUQuaternion::from(*r))
            .collect::<Vec<_>>();

        //create gou parameters
        let gpu_params = self.create_gpu_params(n_rotations);

        //copy rotations and inverse rotations to device
        let dev_rotations = self
            .device
            .htod_copy(gpu_rotations)
            .expect("Unable to copy rotations to device");
        let dev_inv_rotations = self
            .device
            .htod_copy(gpu_inv_rotations)
            .expect("Unable to copy inverse rotations to device");

        //allocate output buffers
        let mut dev_semi_var = self
            .device
            .alloc_zeros::<f32>(250 * n_rotations as usize)
            .unwrap();
        let mut dev_counts = self
            .device
            .alloc_zeros::<u32>(250 * n_rotations as usize)
            .unwrap();

        //kernel laucnh configuration
        let kernel_cfg = LaunchConfig::for_num_elems(self.num_data * n_rotations);

        // run kernel
        let _ = unsafe {
            self.kernel.clone().launch(
                kernel_cfg,
                (
                    &self.points,
                    &self.nodes,
                    &self.values,
                    &self.lags,
                    &dev_rotations,
                    &dev_inv_rotations,
                    gpu_params,
                    &mut dev_semi_var,
                    &mut dev_counts,
                ),
            )
        };

        //copy results back to host
        let semi_var = self.device.dtoh_sync_copy(&dev_semi_var).unwrap();
        let counts = self.device.dtoh_sync_copy(&dev_counts).unwrap();

        //create experimental variograms
        let mut variograms = Vec::with_capacity(n_rotations as usize);
        for (semi_var_chunk, counts_chunk, rotation) in
            izip!(semi_var.chunks(250), counts.chunks(250), rotations.iter())
        {
            variograms.push(ExpirmentalVariogram {
                orientation: rotation.clone(),
                semivariance: semi_var_chunk[0..self.num_lags as usize]
                    .into_iter()
                    .zip(counts_chunk[0..self.num_lags as usize].iter())
                    .map(|(s, c)| s / 2f32 * *c as f32)
                    .collect(),
                counts: counts_chunk[0..self.num_lags as usize].to_vec(),
                lags: self.h_lags.clone(),
            });
        }

        variograms
    }

    pub fn create_gpu_params(&self, n_rotations: u32) -> GPUVariogramParams {
        GPUVariogramParams {
            num_data: self.num_data,
            num_bvh_nodes: self.num_nodes,
            num_lags: self.num_lags,
            num_rotations: n_rotations,
            a: self.a,
            a_tol: self.a_tol,
            b: self.b,
            b_tol: self.b_tol,
            a_dist_threshold: self.a_dist_threshold,
            b_dist_threshold: self.b_dist_threshold,
        }
    }
}

#[cfg(test)]
mod test {
    use cudarc::nvrtc::Ptx;
    use nalgebra::Point3;

    use super::*;
    #[test]
    fn gpu_vgram() {
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

        let bvh = GPUFlatBVH::new(coords, values);

        //create 10 lag bounds
        let lag_lb = (0..15).map(|i| i as f32 * 10f32).collect::<Vec<_>>();
        let lag_ub = (0..15).map(|i| (i + 1) as f32 * 10f32).collect::<Vec<_>>();
        let lag_bounds = lag_lb
            .iter()
            .zip(lag_ub.iter())
            .map(|(lb, ub)| LagBounds::new(*lb, *ub))
            .collect::<Vec<_>>();

        // create quaternions
        let mut quats = vec![UnitQuaternion::identity()];
        for ang1 in 0..1 {
            for ang2 in 0..1 {
                for ang3 in 0..1 {
                    quats.push(UnitQuaternion::from_euler_angles(
                        (ang1 as f32 * 10f32).to_radians(),
                        (ang2 as f32 * 10f32).to_radians(),
                        (ang3 as f32 * 10f32).to_radians(),
                    ));
                }
            }
        }

        let n_rotations = quats.len();
        println!("n_rotations: {}", n_rotations);

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

        let gpu_vgram = GPUVGRAM::new(
            device,
            vgram_kernel,
            bvh,
            lag_bounds,
            10f32,
            0.1f32,
            10f32,
            0.1f32,
        );

        for _ in 0..3 {
            let exp_vgrams = gpu_vgram.compute_for_orientations(quats.clone());

            println!("exp_vgrams: {:?}", exp_vgrams);
        }
    }
}
