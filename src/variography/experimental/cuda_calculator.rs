use std::sync::Arc;

use bvh::{aabb::Bounded, bounding_hierarchy::BHShape};
use cudarc::driver::{CudaDevice, CudaFunction, CudaSlice, DeviceRepr, LaunchAsync, LaunchConfig};
use itertools::izip;
use nalgebra::{Point3, UnitQuaternion};

use super::{ExperimentalVarigoramCalculator, ExpirmentalVariogram, LagBounds};

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CudaVariogramParams {
    //data
    pub num_data: u32,

    //bvh
    pub num_bvh_nodes: u32,

    //lags
    pub num_lags: u32,

    //rotations
    pub num_rotations: u32,

    //ellipsoid dimensions
    pub a: f32,
    pub a_tol: f32,
    pub a_dist_threshold: f32,
    pub b: f32,
    pub b_tol: f32,
    pub b_dist_threshold: f32,
}

unsafe impl DeviceRepr for CudaVariogramParams {}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CudaFloat3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl CudaFloat3 {
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }
}

unsafe impl DeviceRepr for CudaFloat3 {}

impl From<CudaFloat3> for Point3<f32> {
    fn from(value: CudaFloat3) -> Self {
        Point3::new(value.x, value.y, value.z)
    }
}

impl From<Point3<f32>> for CudaFloat3 {
    fn from(value: Point3<f32>) -> Self {
        CudaFloat3 {
            x: value.x,
            y: value.y,
            z: value.z,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CudaGPUAABB {
    pub min: CudaFloat3,
    pub max: CudaFloat3,
}

unsafe impl DeviceRepr for CudaGPUAABB {}

#[derive(Copy, Clone, Debug)]
pub struct CudaBVHPoint {
    coords: CudaFloat3,
    data: f32,
    node_index: usize,
}

impl Bounded<f32, 3> for CudaBVHPoint {
    fn aabb(&self) -> bvh::aabb::Aabb<f32, 3> {
        bvh::aabb::Aabb::with_bounds(self.coords.into(), self.coords.into())
    }
}

impl BHShape<f32, 3> for CudaBVHPoint {
    fn set_bh_node_index(&mut self, index: usize) {
        self.node_index = index;
    }

    fn bh_node_index(&self) -> usize {
        self.node_index
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CudaBVHFlatNode {
    pub aabb: CudaGPUAABB,
    pub entry_index: u32,
    pub exit_index: u32,
    pub shape_index: u32,
}

unsafe impl DeviceRepr for CudaBVHFlatNode {}

#[derive(Debug, Clone)]
pub struct CudaFlatBVH {
    nodes: Vec<CudaBVHFlatNode>,
    bvh_points: Vec<CudaBVHPoint>,
}

impl CudaFlatBVH {
    pub fn new(points: Vec<Point3<f32>>, data: Vec<f32>) -> Self {
        let mut bvh_points = points
            .into_iter()
            .zip(data.into_iter())
            .map(|(point, data)| CudaBVHPoint {
                coords: point.into(),
                data,
                node_index: 0,
            })
            .collect::<Vec<_>>();
        let flat_bvh = bvh::bvh::Bvh::build(&mut bvh_points.as_mut_slice())
            .flatten()
            .iter()
            .map(|node| CudaBVHFlatNode {
                aabb: CudaGPUAABB {
                    min: CudaFloat3 {
                        x: node.aabb.min.x,
                        y: node.aabb.min.y,
                        z: node.aabb.min.z,
                    },
                    max: CudaFloat3 {
                        x: node.aabb.max.x,
                        y: node.aabb.max.y,
                        z: node.aabb.max.z,
                    },
                },
                entry_index: node.entry_index,
                exit_index: node.exit_index,
                shape_index: node.shape_index,
            })
            .collect::<Vec<_>>();
        Self {
            nodes: flat_bvh,
            bvh_points,
        }
    }

    pub fn bvh_points(&self) -> &Vec<CudaBVHPoint> {
        &self.bvh_points
    }

    pub fn nodes(&self) -> &Vec<CudaBVHFlatNode> {
        &self.nodes
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CudaQuaternion {
    pub w: f32,
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl CudaQuaternion {
    pub fn new(w: f32, x: f32, y: f32, z: f32) -> Self {
        Self { w, x, y, z }
    }
}

unsafe impl DeviceRepr for CudaQuaternion {}

impl From<UnitQuaternion<f32>> for CudaQuaternion {
    fn from(value: UnitQuaternion<f32>) -> Self {
        CudaQuaternion {
            w: value.w,
            x: value.i,
            y: value.j,
            z: value.k,
        }
    }
}

pub struct CudaCalculator {
    //cuda device
    device: Arc<CudaDevice>,

    //cuda kernel
    kernel: CudaFunction,

    //bvh nodes
    num_nodes: u32,
    nodes: CudaSlice<CudaBVHFlatNode>,

    //data locations and values
    points: CudaSlice<CudaFloat3>,
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

impl CudaCalculator {
    pub fn new(
        device: Arc<CudaDevice>,
        kernel: CudaFunction,
        bvh: CudaFlatBVH,
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

    pub fn create_cuda_params(&self, n_rotations: u32) -> CudaVariogramParams {
        CudaVariogramParams {
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

impl ExperimentalVarigoramCalculator for CudaCalculator {
    fn calculate_for_orientations(
        &self,
        rotations: &[UnitQuaternion<f32>],
    ) -> Vec<ExpirmentalVariogram> {
        //create inverse of rotations
        let inv_rotations: Vec<_> = rotations.iter().map(|r| r.inverse()).collect();

        //map quaternions to GPUQuaternion
        let n_rotations = rotations.len() as u32;
        let gpu_rotations = rotations
            .iter()
            .map(|r| CudaQuaternion::from(*r))
            .collect::<Vec<_>>();

        let gpu_inv_rotations = inv_rotations
            .iter()
            .map(|r| CudaQuaternion::from(*r))
            .collect::<Vec<_>>();

        //create gou parameters
        let gpu_params = self.create_cuda_params(n_rotations);

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

        let bvh = CudaFlatBVH::new(coords, values);

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

        for _ in 0..3 {
            let exp_vgrams = gpu_vgram.calculate_for_orientations(quats.as_slice());

            println!("exp_vgrams: {:?}", exp_vgrams);
        }
    }
}
