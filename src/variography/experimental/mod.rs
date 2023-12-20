pub mod gpu_vgram;
use std::sync::Arc;

use bvh::{
    aabb::{Aabb, Bounded},
    bounding_hierarchy::BHShape,
};
use cudarc::{
    driver::{CudaDevice, DeviceRepr, LaunchAsync, LaunchConfig},
    nvrtc::Ptx,
};
use itertools::izip;
use nalgebra::{Point3, UnitQuaternion};
#[derive(Clone, Debug)]
pub struct VariogramParams {
    //data
    pub bvh: GPUFlatBVH,

    //lags
    pub lags: Vec<LagBounds>,

    //rotations
    pub rotations: Vec<UnitQuaternion<f32>>,

    //ellipsoid dimensions
    pub a: f32,
    pub a_tol: f32,
    pub a_dist_threshold: f32,
    pub b: f32,
    pub b_tol: f32,
    pub b_dist_threshold: f32,
}

impl VariogramParams {
    pub fn new(
        bvh: GPUFlatBVH,
        lags: Vec<LagBounds>,
        rotations: Vec<UnitQuaternion<f32>>,
        a: f32,
        a_tol: f32,
        b: f32,
        b_tol: f32,
    ) -> Self {
        //distance at which max cylindar dimension reached along major and minor axis
        let a_dist_threshold = a / a_tol.tan();
        let b_dist_threshold = b / b_tol.tan();
        Self {
            bvh,
            lags,
            rotations,
            a,
            a_tol,
            a_dist_threshold,
            b,
            b_tol,
            b_dist_threshold,
        }
    }

    fn to_gpu(&self) -> GPUVariogramParams {
        GPUVariogramParams {
            num_data: self.bvh.bvh_points.len() as u32,
            num_bvh_nodes: self.bvh.nodes().len() as u32,
            num_lags: self.lags.len() as u32,
            num_rotations: self.rotations.len() as u32,
            a: self.a,
            a_tol: self.a_tol,
            a_dist_threshold: self.a_dist_threshold,
            b: self.b,
            b_tol: self.b_tol,
            b_dist_threshold: self.b_dist_threshold,
        }
    }

    pub fn compute(&self, device: Arc<CudaDevice>) -> Vec<ExpirmentalVariogram> {
        device
            .load_ptx(
                Ptx::from_file(".\\src\\variography\\experimental\\kernel_bvh.ptx"),
                "vgram",
                &["vgram_kernel"],
            )
            .expect("unable to load kernel");

        let (points, values): (Vec<_>, Vec<_>) = self
            .bvh
            .bvh_points()
            .iter()
            .map(|bvh_point| (bvh_point.coords, bvh_point.data))
            .unzip();
        let bvh_nodes = self.bvh.nodes.clone();
        let dev_points = device.htod_copy(points).unwrap();
        let dev_values = device.htod_copy(values).unwrap();
        let dev_bvh_nodes = device.htod_copy(bvh_nodes).unwrap();

        let dev_lags = device.htod_copy(self.lags.clone()).unwrap();

        let (rotations, inv_rotations): (Vec<GPUQuaternion>, Vec<GPUQuaternion>) = self
            .rotations
            .clone()
            .into_iter()
            .map(|quat| {
                (
                    GPUQuaternion::from(quat),
                    GPUQuaternion::from(quat.inverse()),
                )
            })
            .unzip();
        let dev_rotations = device.htod_copy(rotations).unwrap();
        let dev_inv_rotations = device.htod_copy(inv_rotations).unwrap();

        let gpu_params = self.to_gpu();

        println!("gpu_params: {:?}", gpu_params);
        let vgram_kernel = device.get_func("vgram", "vgram_kernel").unwrap();

        let cfg = LaunchConfig::for_num_elems(gpu_params.num_data * gpu_params.num_rotations);

        let mut out_vgram = device
            .alloc_zeros::<f32>(250 * gpu_params.num_rotations as usize)
            .unwrap();
        let mut out_counts = device
            .alloc_zeros::<u32>(250 * gpu_params.num_rotations as usize)
            .unwrap();

        unsafe {
            vgram_kernel.launch(
                cfg,
                (
                    &dev_points,
                    &dev_bvh_nodes,
                    &dev_values,
                    &dev_lags,
                    &dev_rotations,
                    &dev_inv_rotations,
                    gpu_params,
                    &mut out_vgram,
                    &mut out_counts,
                ),
            )
        }
        .unwrap();

        let vgram = device.dtoh_sync_copy(&out_vgram).unwrap();
        let counts = device.dtoh_sync_copy(&out_counts).unwrap();

        let mut exp_vgrams = Vec::new();

        for (vgram_chunk, counts_chunk, rotation) in
            izip!(vgram.chunks(250), counts.chunks(250), self.rotations.iter())
        {
            let rot_semi_var = vgram_chunk[0..self.lags.len()]
                .iter()
                .zip(counts_chunk.iter())
                .map(|(v, c)| v / (2f32 * *c as f32))
                .collect::<Vec<_>>();
            exp_vgrams.push(ExpirmentalVariogram {
                orientation: rotation.clone(),
                lags: self.lags.clone(),
                semivariance: rot_semi_var,
                counts: counts[0..self.lags.len()].to_vec(),
            })
        }

        exp_vgrams
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GPUVariogramParams {
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

unsafe impl DeviceRepr for GPUVariogramParams {}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Float3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Float3 {
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }
}

unsafe impl DeviceRepr for Float3 {}

impl From<Float3> for Point3<f32> {
    fn from(value: Float3) -> Self {
        Point3::new(value.x, value.y, value.z)
    }
}

impl From<Point3<f32>> for Float3 {
    fn from(value: Point3<f32>) -> Self {
        Float3 {
            x: value.x,
            y: value.y,
            z: value.z,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GPUAABB {
    pub min: Float3,
    pub max: Float3,
}

unsafe impl DeviceRepr for GPUAABB {}

#[derive(Copy, Clone, Debug)]
pub struct BVHPoint {
    coords: Float3,
    data: f32,
    node_index: usize,
}

impl Bounded<f32, 3> for BVHPoint {
    fn aabb(&self) -> bvh::aabb::Aabb<f32, 3> {
        bvh::aabb::Aabb::with_bounds(self.coords.into(), self.coords.into())
    }
}

impl BHShape<f32, 3> for BVHPoint {
    fn set_bh_node_index(&mut self, index: usize) {
        self.node_index = index;
    }

    fn bh_node_index(&self) -> usize {
        self.node_index
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GPUBVHFlatNode {
    pub aabb: GPUAABB,
    pub entry_index: u32,
    pub exit_index: u32,
    pub shape_index: u32,
}

unsafe impl DeviceRepr for GPUBVHFlatNode {}

#[derive(Debug, Clone)]
pub struct GPUFlatBVH {
    nodes: Vec<GPUBVHFlatNode>,
    bvh_points: Vec<BVHPoint>,
}

impl GPUFlatBVH {
    pub fn new(points: Vec<Point3<f32>>, data: Vec<f32>) -> Self {
        let mut bvh_points = points
            .into_iter()
            .zip(data.into_iter())
            .map(|(point, data)| BVHPoint {
                coords: point.into(),
                data,
                node_index: 0,
            })
            .collect::<Vec<_>>();
        let flat_bvh = bvh::bvh::Bvh::build(&mut bvh_points.as_mut_slice())
            .flatten()
            .iter()
            .map(|node| GPUBVHFlatNode {
                aabb: GPUAABB {
                    min: Float3 {
                        x: node.aabb.min.x,
                        y: node.aabb.min.y,
                        z: node.aabb.min.z,
                    },
                    max: Float3 {
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

    pub fn bvh_points(&self) -> &Vec<BVHPoint> {
        &self.bvh_points
    }

    pub fn nodes(&self) -> &Vec<GPUBVHFlatNode> {
        &self.nodes
    }
}

pub trait IntersectsAABB {
    fn intersects_aabb(&self, aabb: &bvh::aabb::Aabb<f32, 3>) -> bool;
}

impl IntersectsAABB for Aabb<f32, 3> {
    fn intersects_aabb(&self, aabb: &bvh::aabb::Aabb<f32, 3>) -> bool {
        self.min.x <= aabb.max.x
            && self.max.x >= aabb.min.x
            && self.min.y <= aabb.max.y
            && self.max.y >= aabb.min.y
            && self.min.z <= aabb.max.z
            && self.max.z >= aabb.min.z
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GPUQuaternion {
    pub w: f32,
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl GPUQuaternion {
    pub fn new(w: f32, x: f32, y: f32, z: f32) -> Self {
        Self { w, x, y, z }
    }
}

unsafe impl DeviceRepr for GPUQuaternion {}

impl From<UnitQuaternion<f32>> for GPUQuaternion {
    fn from(value: UnitQuaternion<f32>) -> Self {
        GPUQuaternion {
            w: value.w,
            x: value.i,
            y: value.j,
            z: value.k,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LagBounds {
    pub lb: f32,
    pub ub: f32,
}

impl LagBounds {
    pub fn new(lb: f32, ub: f32) -> Self {
        Self { lb, ub }
    }

    pub fn mid_point(&self) -> f32 {
        (self.lb + self.ub) / 2f32
    }
}

unsafe impl DeviceRepr for LagBounds {}

#[derive(Debug)]
pub struct ExpirmentalVariogram {
    pub orientation: UnitQuaternion<f32>,
    pub lags: Vec<LagBounds>,
    pub semivariance: Vec<f32>,
    pub counts: Vec<u32>,
}

#[cfg(test)]
mod test {
    use crate::variography::model_variograms::{
        iso_fitter::{CompositeVariogramFitter, VariogramType},
        iso_nugget::Nugget,
        iso_spherical::IsoSpherical,
    };

    use super::*;

    #[test]
    fn cuda_vgram() {
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
        for ang1 in 0..36 {
            for ang2 in 0..36 {
                for ang3 in 0..36 {
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

        let vgram_params = VariogramParams::new(
            bvh,
            lag_bounds,
            quats,
            10f32,
            45f32.to_radians(),
            10f32,
            45f32.to_radians(),
        );

        //get device
        let dev = cudarc::driver::CudaDevice::new(0).unwrap();
        println!("Starting COmptue");
        let exp_vgrams = vgram_params.compute(dev);

        println!("Exp vgram: {:?}", exp_vgrams[0]);

        exp_vgrams[0]
            .lags
            .iter()
            .zip(exp_vgrams[0].semivariance.iter())
            .for_each(|(l, v)| {
                println!("{}, {},", l.mid_point(), v);
            });

        let x = exp_vgrams[0]
            .lags
            .iter()
            .map(|lag| lag.mid_point())
            .collect::<Vec<_>>();

        let y = exp_vgrams[0].semivariance.clone();

        let mut fit = CompositeVariogramFitter::new(
            x.clone(),
            y.clone(),
            vec![
                VariogramType::Nugget(Nugget { nugget: 1.0 }),
                VariogramType::IsoSphericalNoNugget(IsoSpherical::new(1.0, 1.0)),
            ],
        );

        //let mut solver = SolverDriver::builder(&fit)
        //.with_initial(vec![1.0, 1.0, 1.0])
        //.build();

        //let (x, norm) = solver
        //.find(|state| state.norm() <= 1e-6 || state.iter() >= 100)
        //.expect("solver error");
        //println!("x: {:?}", x);

        println!("Starting fit.");
        let _ = fit.fit();

        for (i, lag) in x.iter().enumerate() {
            println!("{},{},{},", lag, y[i], fit.variogram(*lag as f64));
        }
    }
}
