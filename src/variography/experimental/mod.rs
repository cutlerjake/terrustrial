pub mod cpu_calculator;
// pub mod cuda_calculator;

// use bvh::aabb::Aabb;
// use cudarc::driver::DeviceRepr;
use nalgebra::UnitQuaternion;

// pub trait IntersectsAABB {
//     fn intersects_aabb(&self, aabb: &bvh::aabb::Aabb<f32, 3>) -> bool;
// }

// impl IntersectsAABB for Aabb<f32, 3> {
//     fn intersects_aabb(&self, aabb: &bvh::aabb::Aabb<f32, 3>) -> bool {
//         self.min.x <= aabb.max.x
//             && self.max.x >= aabb.min.x
//             && self.min.y <= aabb.max.y
//             && self.max.y >= aabb.min.y
//             && self.min.z <= aabb.max.z
//             && self.max.z >= aabb.min.z
//     }
// }

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

// unsafe impl DeviceRepr for LagBounds {}

pub trait ExperimentalVarigoramCalculator {
    fn calculate_for_orientations(
        &self,
        orientations: &[UnitQuaternion<f32>],
    ) -> Vec<ExpirmentalVariogram>;
}
#[derive(Debug, Clone)]
pub struct ExpirmentalVariogram {
    pub orientation: UnitQuaternion<f32>,
    pub lags: Vec<LagBounds>,
    pub semivariance: Vec<f32>,
    pub counts: Vec<u32>,
}
