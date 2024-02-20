use ordered_float::OrderedFloat;
use std::error::Error;
use std::fmt;
use std::fmt::Display;
use std::fmt::Formatter;

use itertools::izip;
use rand::Rng;
use rmpfit::{MPFitter, MPPar, MPResult};

use crate::variography::model_variograms::{iso_gaussian, iso_nugget, iso_spherical};

use super::IsoVariogramModel;

#[derive(Debug, Clone)]
pub enum FitError {
    NoSill,
    NoRange,
    InvalidSill(f64),
    InvalidRange(f64),
}

impl Display for FitError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            FitError::InvalidSill(sill) => write!(f, "Invalid sill: {}", sill),
            FitError::InvalidRange(range) => write!(f, "Invalid range: {}", range),
            FitError::NoRange => write!(f, "No range"),
            FitError::NoSill => write!(f, "No sill"),
        }
    }
}

impl Error for FitError {}

#[derive(Debug, Clone)]
pub enum VariogramType {
    IsoSphericalNoNugget(iso_spherical::IsoSpherical),
    IsoGaussian(iso_gaussian::IsoGaussian),
    Nugget(iso_nugget::Nugget),
}

impl IsoVariogramModel<f64> for VariogramType {
    fn c_0(&self) -> f64 {
        match self {
            VariogramType::IsoSphericalNoNugget(v) => v.c_0(),
            VariogramType::IsoGaussian(v) => v.c_0(),
            VariogramType::Nugget(v) => v.c_0(),
        }
    }

    fn variogram(&self, h: f64) -> f64 {
        match self {
            VariogramType::IsoSphericalNoNugget(v) => v.variogram(h),
            VariogramType::IsoGaussian(v) => v.variogram(h),
            VariogramType::Nugget(v) => v.variogram(h),
        }
    }

    fn covariogram(&self, h: f64) -> f64 {
        match self {
            VariogramType::IsoSphericalNoNugget(v) => v.covariogram(h),
            VariogramType::IsoGaussian(v) => v.covariogram(h),
            VariogramType::Nugget(v) => v.covariogram(h),
        }
    }
}

pub struct CompositeVariogramFitter {
    pub lags: Vec<f32>,
    pub exp_var: Vec<f32>,
    pub variograms: Vec<VariogramType>,
    pub n_params: usize,
    pub mppar_params: Vec<MPPar>,
}

impl CompositeVariogramFitter {
    pub fn new(lags: Vec<f32>, exp_var: Vec<f32>, variograms: Vec<VariogramType>) -> Self {
        let n_mppar_params = variograms.iter().fold(0usize, |acc, v| match v {
            VariogramType::IsoSphericalNoNugget(_) => acc + 2,
            VariogramType::IsoGaussian(_) => acc + 2,
            VariogramType::Nugget(_) => acc + 1,
        });
        let mppar_params = (0..n_mppar_params).map(|_| MPPar {
            limited_low: true,
            limit_low: 0.0,
            ..Default::default()
        });

        //println!("map: {:?}", derivative_map);
        Self {
            lags,
            exp_var,
            variograms,
            n_params: n_mppar_params,
            mppar_params: mppar_params.collect(),
        }
    }

    pub fn set_params_from_slice(&mut self, params: &[f64]) {
        let mut param_ind = 0;
        for vgram in self.variograms.iter_mut() {
            match vgram {
                VariogramType::IsoSphericalNoNugget(v) => {
                    v.range = params[param_ind];
                    v.sill = params[param_ind + 1];
                    param_ind += 2;
                }
                VariogramType::IsoGaussian(v) => {
                    v.range = params[param_ind];
                    v.sill = params[param_ind + 1];
                    param_ind += 2;
                }
                VariogramType::Nugget(v) => {
                    v.nugget = params[param_ind];
                    param_ind += 1;
                }
            }
        }
    }

    pub fn variogram(&self, h: f64) -> f64 {
        let mut variogram = 0f64;
        for v in &self.variograms {
            match v {
                VariogramType::IsoSphericalNoNugget(v) => variogram += v.variogram(h),
                VariogramType::IsoGaussian(v) => variogram += v.variogram(h),
                VariogramType::Nugget(v) => variogram += v.variogram(h),
            }
        }
        variogram
    }

    pub fn sill(&self) -> f64 {
        self.variograms.iter().fold(0f64, |acc, v| match v {
            VariogramType::IsoSphericalNoNugget(v) => acc + v.sill,
            VariogramType::IsoGaussian(v) => acc + v.sill,
            VariogramType::Nugget(v) => acc + v.nugget,
        })
    }
    pub fn covariogram(&self, h: f64) -> f64 {
        self.sill() - self.variogram(h)
    }

    pub fn range(&self) -> f64 {
        self.variograms
            .iter()
            .map(|v| match v {
                VariogramType::IsoSphericalNoNugget(v) => v.range,
                VariogramType::IsoGaussian(v) => v.range,
                VariogramType::Nugget(_) => 0f64,
            })
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
    }

    pub fn fit(&mut self) -> Result<(), FitError> {
        let mut rng = rand::thread_rng();
        let max_rng = self
            .lags
            .iter()
            .cloned()
            .max_by_key(|a| OrderedFloat(*a as f64))
            .map_or_else(
                || Err(FitError::NoRange),
                |x| {
                    if x.is_finite() {
                        Ok(x as f64)
                    } else {
                        Err(FitError::InvalidRange(x as f64))
                    }
                },
            )?;

        let range_rng = 0f64..max_rng;
        let max_sill = self
            .exp_var
            .iter()
            .cloned()
            .max_by_key(|v| OrderedFloat(*v as f64))
            .map_or_else(
                || Err(FitError::NoSill),
                |x| {
                    if x.is_finite() {
                        Ok(x as f64)
                    } else {
                        Err(FitError::InvalidSill(x as f64))
                    }
                },
            )?;

        let sill_rng = 0f64..max_sill;

        let best = (0..1000)
            .filter_map(|_| {
                let mut init_range = self.variograms.iter().fold(vec![], |mut acc, v| match v {
                    VariogramType::IsoSphericalNoNugget(_) => {
                        acc.extend(vec![
                            rng.gen_range(range_rng.clone()),
                            rng.gen_range(sill_rng.clone()),
                        ]);
                        acc
                    }
                    VariogramType::IsoGaussian(_) => {
                        acc.extend(vec![
                            rng.gen_range(range_rng.clone()),
                            rng.gen_range(sill_rng.clone()),
                        ]);
                        acc
                    }
                    VariogramType::Nugget(_) => {
                        acc.push(rng.gen_range(sill_rng.clone()));
                        acc
                    }
                });

                let Ok(res) = self.mpfit(init_range.as_mut_slice()) else {
                    return None;
                };

                let err = res.resid.iter().sum::<f64>();

                Some((err, init_range))
            })
            .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
            .unwrap();

        self.set_params_from_slice(best.1.as_slice());
        Ok(())
    }
}
impl MPFitter for CompositeVariogramFitter {
    fn eval(&mut self, params: &[f64], deviates: &mut [f64]) -> MPResult<()> {
        //update variogram parameters
        self.set_params_from_slice(params);

        //println!("params: {:?}", params);

        //compute deviates
        for (d, x, y) in izip!(deviates.iter_mut(), self.lags.iter(), self.exp_var.iter()) {
            *d = *y as f64 - self.variogram(*x as f64);
        }

        Ok(())
    }

    fn number_of_points(&self) -> usize {
        self.lags.len()
    }

    fn parameters(&self) -> Option<&[MPPar]> {
        Some(self.mppar_params.as_slice())
    }
}
