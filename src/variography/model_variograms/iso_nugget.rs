use super::IsoVariogramModel;

#[derive(Debug, Clone, Default, Copy)]
pub struct IsoNugget {
    pub nugget: f64,
}

impl IsoNugget {
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

    pub fn param_cnt() -> usize {
        1
    }

    pub fn update_from_slice(&mut self, params: &[f64]) {
        self.nugget = params[0];
    }
}

impl IsoVariogramModel<f64> for IsoNugget {
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
