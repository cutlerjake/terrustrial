use super::iso_exponential::IsoExponential;
use super::iso_gaussian::IsoGaussian;
use super::iso_nugget::IsoNugget;
use super::IsoVariogramModel;
use crate::variography::model_variograms::iso_spherical::IsoSpherical;
pub enum IsoVariogramType {
    Nugget(IsoNugget),
    Spherical(IsoSpherical),
    Exponential(IsoExponential),
    Gaussian(IsoGaussian),
}

impl IsoVariogramType {
    pub fn c_0(&self) -> f64 {
        match self {
            IsoVariogramType::Nugget(v) => v.c_0(),
            IsoVariogramType::Spherical(v) => v.c_0(),
            IsoVariogramType::Exponential(v) => v.c_0(),
            IsoVariogramType::Gaussian(v) => v.c_0(),
        }
    }

    pub fn variogram(&self, h: f64) -> f64 {
        match self {
            IsoVariogramType::Nugget(v) => v.variogram(h),
            IsoVariogramType::Spherical(v) => v.variogram(h),
            IsoVariogramType::Exponential(v) => v.variogram(h),
            IsoVariogramType::Gaussian(v) => v.variogram(h),
        }
    }

    pub fn covariogram(&self, h: f64) -> f64 {
        match self {
            IsoVariogramType::Nugget(v) => v.covariogram(h),
            IsoVariogramType::Spherical(v) => v.covariogram(h),
            IsoVariogramType::Exponential(v) => v.covariogram(h),
            IsoVariogramType::Gaussian(v) => v.covariogram(h),
        }
    }

    pub fn param_cnt(&self) -> usize {
        match self {
            IsoVariogramType::Nugget(_) => IsoNugget::param_cnt(),
            IsoVariogramType::Spherical(_) => IsoSpherical::param_cnt(),
            IsoVariogramType::Exponential(_) => IsoExponential::param_cnt(),
            IsoVariogramType::Gaussian(_) => IsoGaussian::param_cnt(),
        }
    }

    pub fn update_params(&mut self, params: &[f64]) {
        match self {
            IsoVariogramType::Nugget(v) => v.update_from_slice(params),
            IsoVariogramType::Spherical(v) => v.update_from_slice(params),
            IsoVariogramType::Exponential(v) => v.update_from_slice(params),
            IsoVariogramType::Gaussian(v) => v.update_from_slice(params),
        }
    }
}

pub struct CompositeIsoVariogram {
    pub variograms: Vec<IsoVariogramType>,
}

impl CompositeIsoVariogram {
    pub fn new(variograms: Vec<IsoVariogramType>) -> Self {
        Self { variograms }
    }

    pub fn c_0(&self) -> f64 {
        self.variograms.iter().map(|v| v.c_0()).sum()
    }

    pub fn variogram(&self, h: f64) -> f64 {
        self.variograms.iter().map(|v| v.variogram(h)).sum()
    }

    pub fn covariogram(&self, h: f64) -> f64 {
        self.variograms.iter().map(|v| v.covariogram(h)).sum()
    }

    pub fn param_cnt(&self) -> usize {
        self.variograms.iter().map(|v| v.param_cnt()).sum()
    }

    pub fn update_params(&mut self, params: &[f64]) {
        let mut idx = 0;

        for v in self.variograms.iter_mut() {
            let n = v.param_cnt();
            let p = &params[idx..idx + n];
            v.update_params(p);
            idx += n;
        }
    }
}
