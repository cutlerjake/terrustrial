use super::IsoVariogramModel;

#[derive(Debug, Clone, Copy, Default)]
pub struct IsoExponential {
    pub range: f64,
    pub sill: f64,
}

impl IsoExponential {
    pub fn new(range: f64, sill: f64) -> Self {
        Self { range, sill }
    }

    pub fn variogram(self, h: f64) -> f64 {
        if h < self.range {
            return (self.sill) * (1.0 - (-h / self.range).exp());
        }
        self.sill
    }

    pub fn covariogram(self, h: f64) -> f64 {
        self.sill - self.variogram(h)
    }

    pub fn param_cnt() -> usize {
        2
    }

    pub fn update_from_slice(&mut self, params: &[f64]) {
        self.range = params[0];
        self.sill = params[1];
    }
}

impl IsoVariogramModel for IsoExponential {
    fn c_0(&self) -> f64 {
        self.sill
    }

    fn variogram(&self, h: f64) -> f64 {
        if h < self.range {
            return (self.sill) * (1.0 - (-h / self.range).exp());
        }
        self.sill
    }

    fn covariogram(&self, h: f64) -> f64 {
        self.sill - self.variogram(h)
    }
}
