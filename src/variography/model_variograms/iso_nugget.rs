use super::IsoVariogramModel;

#[derive(Debug, Clone, Default, Copy)]
pub struct Nugget {
    pub nugget: f64,
}

impl Nugget {
    pub fn new(nugget: f64) -> Self {
        Self { nugget }
    }

    pub fn variogram(&self, h: f64) -> f64 {
        if h == 0f64 {
            return 0f64;
        }
        self.nugget
    }

    pub fn covariogram(&self, h: f64) -> f64 {
        self.nugget - self.variogram(h)
    }
    #[allow(unused_variables)]
    pub fn variogram_dn(&self, h: f64) -> f64 {
        1.0
    }

    pub fn parameter_names() -> Vec<&'static str> {
        vec!["nugget"]
    }
}

impl IsoVariogramModel<f64> for Nugget {
    fn c_0(&self) -> f64 {
        self.nugget
    }

    fn variogram(&self, h: f64) -> f64 {
        if h == 0f64 {
            return 0f64;
        }
        self.nugget
    }

    fn covariogram(&self, h: f64) -> f64 {
        self.nugget - self.variogram(h)
    }
}
