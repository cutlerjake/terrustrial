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

    pub fn _variogram(self, h: f64) -> f64 {
        if h < self.range {
            return (self.sill) * (1.0 - (-h / self.range).exp());
        }
        self.sill
    }

    pub fn covariogram(self, h: f64) -> f64 {
        self.sill - self._variogram(h)
    }

    //derivative of variogram with respect to range
    // pub fn variogram_dr(self, h: f64) -> f64 {
    //     let r = self.range;

    //     (self.sill) * (h * (-h / r).exp()) / (r * r)
    // }

    // //derivative of variogram with respect to sill
    // pub fn variogram_ds(self, h: f32) -> f32 {
    //     let r = self.range;

    //     -(h * self.sill * (-h / r).exp()) / (r * r)
    // }

    // pub fn parameter_names() -> Vec<&'static str> {
    //     vec!["range", "sill"]
    // }
}

impl IsoVariogramModel<f64> for IsoExponential {
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
        self.sill - self._variogram(h)
    }
}
