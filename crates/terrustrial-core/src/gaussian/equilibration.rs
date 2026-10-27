//! Power-of-two congruence scaling for heterogeneous observation/target units.
//!
//! With C' = D^-1 C D^-1, K' = D^-1 K T^-1 and F' = D^-1 F Q^-1,
//! the existing GLS kernels operate unchanged. Physical weights are D^-1 W' T
//! and physical uncertainty is T V' T (also including any common covariance units).
use super::*;

pub(super) struct Equilibration {
    data_active: bool,
    target_active: bool,
    n: usize,
    m: usize,
    exponents: Vec<i16>,
    diagonal: Vec<f64>,
}

impl Equilibration {
    pub(super) fn new(n: usize, m: usize, p: usize) -> Result<Self> {
        let count = n
            .checked_add(m)
            .and_then(|v| v.checked_add(p))
            .ok_or(Error::SizeOverflow)?;
        let mut exponents = Vec::new();
        exponents
            .try_reserve_exact(count)
            .map_err(|_| invalid("Gaussian equilibration allocation"))?;
        exponents.resize(count, 0);
        let mut diagonal = Vec::new();
        diagonal
            .try_reserve_exact(m)
            .map_err(|_| invalid("Gaussian target diagonal allocation"))?;
        diagonal.resize(m, 0.);
        Ok(Self {
            data_active: false,
            target_active: false,
            n,
            m,
            exponents,
            diagonal,
        })
    }

    pub(super) fn prepare_data(
        &mut self,
        mut c: MatMut<'_, f64>,
        mut fd: MatMut<'_, f64>,
        observations: &[Observation],
    ) -> Result<f64> {
        self.data_active = false;
        self.target_active = false;
        let mut low = i32::MAX;
        let mut high = i32::MIN;
        for i in 0..c.nrows() {
            let sum = c[(i, i)] + observations[i].noise_variance();
            let variance = if sum.is_finite() {
                sum
            } else {
                c[(i, i)].max(observations[i].noise_variance())
            };
            if variance > 0. {
                let exponent = exponent(variance).div_euclid(2);
                self.exponents[i] = exponent as i16;
                low = low.min(exponent);
                high = high.max(exponent);
            } else {
                // Leave invalid or singular diagonals to the ordinary factorization path.
                return scaling::prepare_conditioning(c, observations);
            }
        }
        if c.nrows() == 0 || high - low < 4 {
            return scaling::prepare_conditioning(c, observations);
        }
        self.data_active = true;
        for j in 0..c.ncols() {
            for i in 0..c.nrows() {
                let shift = -self.data_exponent(i) - self.data_exponent(j);
                let value = c[(i, j)];
                c[(i, j)] = if i == j {
                    let noise = observations[i].noise_variance();
                    let total = value + noise;
                    if total.is_finite() {
                        normalize(total, shift)?
                    } else {
                        normalize(value, shift)? + normalize(noise, shift)?
                    }
                } else {
                    normalize(value, shift)?
                };
            }
        }
        for j in 0..fd.ncols() {
            let q = (0..fd.nrows())
                .filter(|&i| fd[(i, j)] != 0.)
                .map(|i| exponent(fd[(i, j)].abs()) - self.data_exponent(i))
                .max()
                .unwrap_or(0);
            self.exponents[self.n + self.m + j] = q as i16;
            for i in 0..fd.nrows() {
                fd[(i, j)] = normalize(fd[(i, j)], -self.data_exponent(i) - q)?;
            }
        }
        Ok(1.)
    }

    pub(super) fn prepare_targets(
        &mut self,
        model: &impl Covariance,
        observations: &[Observation],
        targets: &[Site],
        units: f64,
        cancellation: &Cancellation,
    ) -> Result<()> {
        self.target_active = false;
        let common = exponent(units);
        let mut low = i32::MAX;
        let mut high = i32::MIN;
        for (j, target) in targets.iter().enumerate() {
            cancellation.check()?;
            let variance = model.covariance(target, target)?;
            if !variance.is_finite() {
                return Err(Error::Numerical {
                    context: "target covariance diagonal",
                });
            }
            self.diagonal[j] = variance;
            let e = if variance != 0. {
                (exponent(variance.abs()) - common).div_euclid(2)
            } else if self.data_active {
                observations
                    .iter()
                    .enumerate()
                    .filter(|(_, o)| o.site().variable == target.variable)
                    .map(|(i, _)| self.data_exponent(i))
                    .max()
                    .unwrap_or(0)
            } else {
                0
            };
            self.exponents[self.n + j] = e as i16;
            low = low.min(e);
            high = high.max(e);
        }
        self.target_active = self.data_active || high - low >= 4 || low < -166 || high > 166;
        Ok(())
    }

    pub(super) fn data_scaling_active(&self) -> bool {
        self.data_active
    }
    // Optional residual checks use the prepared scaling without rebuilding or
    // changing it. Preserve the same rounded diagonal-plus-noise convention.
    pub(super) fn diagnostic_covariance(
        &self,
        value: f64,
        noise: f64,
        i: usize,
        j: usize,
        units: f64,
    ) -> Result<f64> {
        let scale = |v| {
            if self.data_active {
                normalize(v, -self.data_exponent(i) - self.data_exponent(j))
            } else {
                scaling::normalize(v, units)
            }
        };
        if i != j {
            return scale(value);
        }
        let total = value + noise;
        if total.is_finite() {
            scale(total)
        } else {
            Ok(scale(value)? + scale(noise)?)
        }
    }
    pub(super) fn diagnostic_cross(
        &self,
        value: f64,
        i: usize,
        j: usize,
        units: f64,
    ) -> Result<f64> {
        if self.target_active {
            normalize(
                value,
                -exponent(units) - self.data_exponent(i) - self.target_exponent(j),
            )
        } else {
            scaling::normalize(value, units)
        }
    }
    pub(super) fn diagnostic_target_covariance(
        &self,
        value: f64,
        i: usize,
        j: usize,
        units: f64,
    ) -> Result<f64> {
        normalize(value, -self.covariance_exponent(i, j, units))
    }
    pub(super) fn diagnostic_weight(&self, value: f64, i: usize, j: usize) -> Result<f64> {
        if self.target_active {
            normalize(value, self.data_exponent(i) - self.target_exponent(j))
        } else {
            Ok(value)
        }
    }
    fn data_exponent(&self, i: usize) -> i32 {
        if self.data_active {
            i32::from(self.exponents[i])
        } else {
            0
        }
    }
    fn target_exponent(&self, i: usize) -> i32 {
        if self.target_active {
            i32::from(self.exponents[self.n + i])
        } else {
            0
        }
    }
    fn covariance_exponent(&self, i: usize, j: usize, units: f64) -> i32 {
        exponent(units) + self.target_exponent(i) + self.target_exponent(j)
    }

    pub(super) fn normalize_cross(&self, mut k: MatMut<'_, f64>, units: f64) -> Result<()> {
        if !self.target_active {
            return scaling::normalize_matrix(k, units);
        }
        let common = exponent(units);
        for j in 0..k.ncols() {
            for i in 0..k.nrows() {
                k[(i, j)] = normalize(
                    k[(i, j)],
                    -common - self.data_exponent(i) - self.target_exponent(j),
                )?;
            }
        }
        Ok(())
    }
    pub(super) fn normalize_drift(&self, mut ft: MatMut<'_, f64>) -> Result<()> {
        if self.target_active {
            for j in 0..ft.ncols() {
                let q = if self.data_active {
                    i32::from(self.exponents[self.n + self.m + j])
                } else {
                    0
                };
                for i in 0..ft.nrows() {
                    ft[(i, j)] = normalize(ft[(i, j)], -self.target_exponent(i) - q)?;
                }
            }
        }
        Ok(())
    }
    // This wrapper runs for every target pair, including the ordinary fast path.
    #[inline(always)]
    pub(super) fn prior(&self, off_diagonal: f64, i: usize, j: usize, units: f64) -> Result<f64> {
        let value = if i == j {
            self.diagonal[i]
        } else {
            off_diagonal
        };
        if self.target_active {
            normalize(value, -self.covariance_exponent(i, j, units))
        } else {
            scaling::normalize(value, units)
        }
    }
    fn threshold(
        &self,
        tolerance: NumericalTolerance,
        scale: f64,
        units: f64,
        i: usize,
        j: usize,
    ) -> f64 {
        shift(tolerance.absolute, -self.covariance_exponent(i, j, units))
            + tolerance.relative * scale
    }
    pub(super) fn factor(
        &self,
        matrix: MatRef<'_, f64>,
        factor: MatMut<'_, f64>,
        tolerance: NumericalTolerance,
        prior_scale: f64,
        units: f64,
    ) -> Result<usize> {
        if !self.target_active {
            return psd_factor_into(matrix, factor, tolerance, prior_scale, units);
        }
        finite(matrix, "conditional covariance")?;
        let scale = (0..matrix.nrows())
            .map(|i| matrix[(i, i)].abs())
            .fold(prior_scale, f64::max);
        tolerance.threshold(0.)?;
        let max_exponent = (0..matrix.nrows())
            .map(|i| self.covariance_exponent(i, i, units))
            .max()
            .unwrap_or(0);
        if !(tolerance.absolute + shift(tolerance.relative * scale, max_exponent)).is_finite() {
            return Err(invalid("numerical tolerance overflow"));
        }
        psd_factor_with_threshold(matrix, factor, |i, j| {
            self.threshold(tolerance, scale, units, i, j)
        })
    }
    pub(super) fn restore_weights(&self, mut weights: MatMut<'_, f64>) -> Result<()> {
        if self.target_active {
            for j in 0..weights.ncols() {
                for i in 0..weights.nrows() {
                    weights[(i, j)] = normalize(
                        weights[(i, j)],
                        self.target_exponent(j) - self.data_exponent(i),
                    )?;
                }
            }
        }
        Ok(())
    }
    pub(super) fn restore(
        &self,
        mut covariance: MatMut<'_, f64>,
        mut factor: MatMut<'_, f64>,
        units: f64,
        tolerance: NumericalTolerance,
        prior_scale: f64,
    ) -> Result<usize> {
        if !self.target_active {
            return scaling::restore(covariance, factor, units, tolerance, prior_scale);
        }
        let scale = (0..covariance.nrows())
            .map(|i| covariance[(i, i)].abs())
            .fold(prior_scale, f64::max);
        let mut contraction = 1.0_f64;
        for j in 0..covariance.ncols() {
            for i in 0..covariance.nrows() {
                let value = covariance[(i, j)].abs();
                let limit = shift(f64::MAX, -self.covariance_exponent(i, j, units));
                if value > limit {
                    if value - limit > self.threshold(tolerance, scale, units, i, j) {
                        return Err(Error::Numerical {
                            context: "conditional variance exceeds representable physical units",
                        });
                    }
                    contraction = contraction.min((limit / value).next_down());
                }
            }
        }
        for j in 0..covariance.ncols() {
            for i in 0..covariance.nrows() {
                let value = covariance[(i, j)] * contraction;
                let restored = shift(value, self.covariance_exponent(i, j, units));
                if !restored.is_finite() || (i == j && value != 0. && restored == 0.) {
                    return Err(Error::Numerical {
                        context: "conditional variance exceeds representable physical units",
                    });
                }
                covariance[(i, j)] = restored;
            }
        }
        let common = exponent(units);
        let root = contraction.sqrt()
            * if common.rem_euclid(2) == 0 {
                1.
            } else {
                std::f64::consts::SQRT_2
            };
        for j in 0..factor.ncols() {
            for i in 0..factor.nrows() {
                factor[(i, j)] = shift(
                    factor[(i, j)] * root,
                    common.div_euclid(2) + self.target_exponent(i),
                );
            }
        }
        finite(factor.as_ref(), "conditional factor in physical units")?;
        Ok(usize::from(contraction != 1.))
    }
}

#[cfg(test)]
use crate::compute::scaling::power;
pub(super) use crate::compute::scaling::{exponent, shift};
fn normalize(value: f64, exponent: i32) -> Result<f64> {
    let normalized = shift(value, exponent);
    if !normalized.is_finite() || (value != 0. && normalized == 0.) {
        return Err(Error::Numerical {
            context: "Gaussian equilibration exceeds representable range",
        });
    }
    Ok(normalized)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binary_scaling_covers_subnormals_and_unrepresentable_scale_factors() {
        for e in -1074..=1023 {
            let value = power(e);
            assert_eq!(exponent(value), e);
            assert_eq!(normalize(value, -e).unwrap(), 1.);
            assert_eq!(normalize(-value, -e).unwrap(), -1.);
            assert_eq!(normalize(1., e).unwrap().to_bits(), value.to_bits());
        }
        assert_eq!(normalize(power(-1074), 2097).unwrap(), power(1023));
        assert_eq!(normalize(power(1023), -2097).unwrap(), power(-1074));
        assert!(normalize(1., 1024).is_err());
        assert!(normalize(1., -1075).is_err());
        assert_eq!(normalize(-0., 2097).unwrap().to_bits(), (-0_f64).to_bits());
        assert!(normalize(f64::NAN, 0).is_err());
        assert!(normalize(f64::INFINITY, 0).is_err());
    }

    #[test]
    fn heterogeneous_physical_range_repair_preserves_joint_factor() {
        let mut state = Equilibration::new(0, 2, 0).unwrap();
        state.target_active = true;
        state.exponents[..2].copy_from_slice(&[-500, 512]);
        let mut covariance = Mat::from_fn(2, 2, |i, j| if i == j { 1. } else { 0.5 });
        let mut factor = Mat::zeros(2, 2);
        factor[(0, 0)] = 1.;
        factor[(1, 0)] = 0.5;
        factor[(1, 1)] = 0.75_f64.sqrt();
        assert_eq!(
            state
                .restore(
                    covariance.as_mut(),
                    factor.as_mut(),
                    1.,
                    NumericalTolerance::default(),
                    1.
                )
                .unwrap(),
            1
        );
        assert!(covariance[(1, 1)].is_finite());
        for j in 0..2 {
            for i in 0..2 {
                let reconstructed: f64 = (0..2)
                    .map(|k| {
                        shift(factor[(i, k)], -state.target_exponent(i))
                            * shift(factor[(j, k)], -state.target_exponent(j))
                    })
                    .sum();
                let expected = shift(covariance[(i, j)], -state.covariance_exponent(i, j, 1.));
                assert!((reconstructed - expected).abs() < 1e-15);
            }
        }
        covariance.fill(0.);
        covariance[(1, 1)] = 1.1;
        assert!(state
            .restore(
                covariance.as_mut(),
                factor.as_mut(),
                1.,
                NumericalTolerance::default(),
                1.
            )
            .is_err());
    }
}
