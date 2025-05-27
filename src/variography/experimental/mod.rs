pub mod cpu_calculator;

use ultraviolet::DRotor3;

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct LagBounds {
    pub lb: f64,
    pub ub: f64,
}

impl LagBounds {
    pub fn new(lb: f64, ub: f64) -> Self {
        Self { lb, ub }
    }

    pub fn mid_point(&self) -> f64 {
        (self.lb + self.ub) / 2f64
    }
}

#[derive(Debug, Clone)]
pub struct ExpirmentalVariogram {
    pub orientation: DRotor3,
    pub lags: Vec<LagBounds>,
    pub semivariance: Vec<f64>,
    pub counts: Vec<u32>,
}
