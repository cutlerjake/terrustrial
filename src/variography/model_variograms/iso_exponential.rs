pub struct IsoExponential {
    pub range: f32,
    pub sill: f32,
}

impl IsoExponential {
    pub fn new(range: f32, sill: f32) -> Self {
        Self { range, sill }
    }

    pub fn variogram(self, h: f32) -> f32 {
        if h < self.range {
            return (self.sill) * (1.0 - (-h / self.range).exp());
        }
        return self.sill;
    }

    pub fn covariogram(self, h: f32) -> f32 {
        self.sill - self.variogram(h)
    }

    //derivative of variogram with respect to range
    pub fn variogram_dr(self, h: f32) -> f32 {
        let r = self.range;

        (self.sill) * (h * (-h / r).exp()) / (r * r)
    }

    //derivative of variogram with respect to sill
    pub fn variogram_ds(self, h: f32) -> f32 {
        let r = self.range;

        -(h * self.sill * (-h / r).exp()) / (r * r)
    }

    pub fn parameter_names() -> Vec<&'static str> {
        vec!["range", "sill"]
    }
}
