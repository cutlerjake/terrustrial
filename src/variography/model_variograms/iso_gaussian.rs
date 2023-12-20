use super::IsoVariogramModel;

#[derive(Debug, Clone, Default, Copy)]
pub struct IsoGaussian {
    pub range: f64,
    pub sill: f64,
}

impl IsoGaussian {
    pub fn new(range: f64, sill: f64) -> Self {
        Self { range, sill }
    }

    pub fn variogram(&self, h: f64) -> f64 {
        if h < self.range {
            return self.sill * (1.0 - (-3f64 * h * h / (self.range * self.range)).exp());
        }
        return self.sill;
    }

    pub fn covariogram(&self, h: f64) -> f64 {
        self.sill - self.variogram(h)
    }

    //derivative of variogram with respect to range
    pub fn variogram_dr(&self, h: f64) -> f64 {
        let r = self.range;

        self.sill * (6f64 * h * h * (-3f64 * h * h / (r * r)).exp()) / (r * r * r)
    }

    //derivative of variogram with respect to sill
    //pub fn variogram_ds(&self, h: f32) -> f32 {
    //let r = self.range;

    //1f32 - (-3f32 * h * h / (r * r)).exp()
    //}
    pub fn parameter_names() -> Vec<&'static str> {
        vec!["range"]
    }
}

impl IsoVariogramModel<f64> for IsoGaussian {
    fn c_0(&self) -> f64 {
        self.sill as f64
    }

    fn variogram(&self, h: f64) -> f64 {
        if h < self.range {
            return self.sill * (1.0 - (-3f64 * h * h / (self.range * self.range)).exp());
        }
        return self.sill;
    }

    fn covariogram(&self, h: f64) -> f64 {
        self.sill - self.variogram(h)
    }
}
