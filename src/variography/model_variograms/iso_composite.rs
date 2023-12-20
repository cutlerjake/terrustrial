use super::{iso_fitter::VariogramType, IsoVariogramModel};

pub struct IsoComposite {
    pub structures: Vec<VariogramType>,
}

impl IsoVariogramModel<f64> for IsoComposite {
    fn c_0(&self) -> f64 {
        self.structures.iter().fold(0f64, |acc, v| acc + v.c_0())
    }

    fn variogram(&self, h: f64) -> f64 {
        self.structures
            .iter()
            .fold(0f64, |acc, v| acc + v.variogram(h))
    }

    fn covariogram(&self, h: f64) -> f64 {
        self.structures
            .iter()
            .fold(0f64, |acc, v| acc + v.covariogram(h))
    }
}

impl IsoComposite {
    pub fn new(structures: Vec<VariogramType>) -> Self {
        Self { structures }
    }
}
