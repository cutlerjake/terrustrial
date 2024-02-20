use argmin::core::{CostFunction, Error};
use itertools::Itertools;

use super::{
    composite::{self, CompositeVariogram},
    iso_composite::IsoComposite,
    iso_fitter::VariogramType,
    spherical::SphericalVariogram,
};
use crate::variography::model_variograms::{nugget::Nugget, IsoVariogramModel};
pub struct AnisoFitter {
    pub lags: Vec<f64>,
    pub major_axis_semivar: Vec<f64>,
    pub intermediate_axis_semivar: Vec<f64>,
    pub minor_axis_semivar: Vec<f64>,
    pub structures: Vec<VariogramType>,
    pub n_shared_params: usize,
    pub n_unique_params: usize,
}

impl AnisoFitter {
    pub fn new(
        lags: Vec<f64>,
        major_axis_semivar: Vec<f64>,
        intermediate_axis_semivar: Vec<f64>,
        minor_axis_semivar: Vec<f64>,
        structures: Vec<VariogramType>,
    ) -> Self {
        let n_shared_params = structures.iter().fold(0usize, |acc, v| match v {
            VariogramType::IsoSphericalNoNugget(_) => acc + 1,
            VariogramType::IsoGaussian(_) => acc + 1,
            VariogramType::Nugget(_) => acc + 1,
        });
        let n_unique_params = structures.iter().fold(0usize, |acc, v| match v {
            VariogramType::IsoSphericalNoNugget(_) => acc + 3,
            VariogramType::IsoGaussian(_) => acc + 3,
            VariogramType::Nugget(_) => acc,
        });
        Self {
            lags,
            major_axis_semivar,
            intermediate_axis_semivar,
            minor_axis_semivar,
            structures,
            n_shared_params,
            n_unique_params,
        }
    }

    pub fn get_bounds(&self) -> (Vec<f64>, Vec<f64>) {
        let max_var = *self
            .major_axis_semivar
            .iter()
            .chain(self.intermediate_axis_semivar.iter())
            .chain(self.minor_axis_semivar.iter())
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();

        let max_range = *self
            .lags
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();

        let mut lower_bounds = vec![0.0; self.n_shared_params + self.n_unique_params];
        let mut upper_bounds = vec![0.0; self.n_shared_params + self.n_unique_params];

        let mut unique_ind = 0;
        let mut shared_ind = 0;

        for vgram in self.structures.iter() {
            match vgram {
                VariogramType::IsoSphericalNoNugget(_) => {
                    lower_bounds
                        [self.n_shared_params + unique_ind..self.n_shared_params + unique_ind + 3]
                        .iter_mut()
                        .for_each(|v| *v = 0.0);
                    upper_bounds
                        [self.n_shared_params + unique_ind..self.n_shared_params + unique_ind + 3]
                        .iter_mut()
                        .for_each(|v| *v = max_range);
                    unique_ind += 3;

                    lower_bounds[shared_ind] = 0.0;
                    upper_bounds[shared_ind] = max_var;
                    shared_ind += 1;
                }
                VariogramType::IsoGaussian(_) => {
                    lower_bounds
                        [self.n_shared_params + unique_ind..self.n_shared_params + unique_ind + 3]
                        .iter_mut()
                        .for_each(|v| *v = 0.0);
                    upper_bounds
                        [self.n_shared_params + unique_ind..self.n_shared_params + unique_ind + 3]
                        .iter_mut()
                        .for_each(|v| *v = max_range);
                    unique_ind += 3;

                    lower_bounds[shared_ind] = 0.0;
                    upper_bounds[shared_ind] = max_var;
                    shared_ind += 1;
                }
                VariogramType::Nugget(_) => {
                    lower_bounds[shared_ind] = 0.0;
                    upper_bounds[shared_ind] = max_var;
                    shared_ind += 1;
                }
            }
        }

        (lower_bounds, upper_bounds)
    }

    pub fn aniso_variogram_from_slice(&self, params: &[f64]) -> CompositeVariogram<f32> {
        let mut structures = self
            .structures
            .iter()
            .map(|v| match v {
                VariogramType::IsoSphericalNoNugget(_) => {
                    composite::VariogramType::<f32>::Spherical(SphericalVariogram::new(
                        Default::default(),
                        Default::default(),
                        Default::default(),
                    ))
                }
                VariogramType::IsoGaussian(_) => {
                    unimplemented!("Gaussian variogram not implemented");
                }

                VariogramType::Nugget(_) => {
                    composite::VariogramType::<f32>::Nugget(Nugget::new(Default::default()))
                }
            })
            .collect_vec();

        let unique = &params[0..self.n_unique_params];
        let shared = &params[self.n_unique_params..];

        let mut unique_ind = 0;
        let mut shared_ind = 0;

        for vgram in structures.iter_mut() {
            match vgram {
                composite::VariogramType::Spherical(v) => {
                    v.range[0] = unique[unique_ind] as f32;
                    v.range[1] = unique[unique_ind + 1] as f32;
                    v.range[2] = unique[unique_ind + 2] as f32;
                    v.sill = shared[shared_ind] as f32;
                    shared_ind += 1;
                    unique_ind += 3;
                }
                composite::VariogramType::Nugget(v) => {
                    v.nugget = shared[shared_ind] as f32;
                    shared_ind += 1;
                }
            }
        }

        CompositeVariogram::new(structures)
    }
}

impl CostFunction for &mut AnisoFitter {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, params: &Self::Param) -> Result<Self::Output, Error> {
        let shared = &params[0..self.n_shared_params];
        let unique = &params[self.n_shared_params..];

        let mut unique_ind = 0;

        let mut composite = IsoComposite::new(self.structures.clone());

        let mut residual = 0.0;
        for exp in [
            &self.major_axis_semivar,
            &self.intermediate_axis_semivar,
            &self.minor_axis_semivar,
        ] {
            let mut shared_ind = 0;
            for vgram in composite.structures.iter_mut() {
                match vgram {
                    VariogramType::IsoSphericalNoNugget(v) => {
                        v.range = unique[unique_ind];
                        v.sill = shared[shared_ind];
                        shared_ind += 1;
                        unique_ind += 1;
                    }
                    VariogramType::IsoGaussian(v) => {
                        v.range = unique[unique_ind];
                        v.sill = shared[shared_ind];
                        shared_ind += 1;
                        unique_ind += 1;
                    }
                    VariogramType::Nugget(v) => {
                        v.nugget = shared[shared_ind];
                        shared_ind += 1;
                    }
                }
            }

            for (exp_val, &lag) in exp.iter().zip(self.lags.iter()) {
                residual += (exp_val - composite.variogram(lag)).powi(2);
            }
        }

        Ok(residual)
    }
}
