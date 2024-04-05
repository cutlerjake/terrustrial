pub mod cpu_calculator;

use nalgebra::UnitQuaternion;

#[repr(C)]
#[derive(Copy, Clone, Debug)]
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
