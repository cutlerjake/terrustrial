//! Marginal normal-score transforms with explicit tie and tail semantics.
//! Marginal normality alone does not establish a jointly Gaussian copula.
mod moments;
pub use moments::MomentQuadrature;

use crate::{
    error::{checked_len, invalid},
    execution::{MemoryBudget, Reservation},
    Error, Result,
};
use statrs::distribution::{ContinuousCDF, Normal};

#[derive(Clone, Copy, Debug)]
pub enum TailPolicy {
    Clamp,
    Linear,
    Reject,
}
#[derive(Debug)]
pub struct NormalScore {
    values: Vec<f64>,
    scores: Vec<f64>,
    tail: TailPolicy,
    _reservation: Reservation,
}
impl NormalScore {
    /// Weighted midpoint empirical CDF. Equal values receive the midpoint of their
    /// combined probability mass. Missing/censored values are not inferred; omit
    /// missing values and specify a separate censoring model when needed.
    /// Weights may encode declustering. Zero/negative/nonfinite weights are errors.
    pub fn fit(samples: &[(f64, f64)], tail: TailPolicy, budget: &MemoryBudget) -> Result<Self> {
        if samples.is_empty()
            || samples
                .iter()
                .any(|(v, w)| !v.is_finite() || !w.is_finite() || *w <= 0.0)
        {
            return Err(invalid("finite values and positive weights required"));
        }
        let bytes = checked_len(&[samples.len(), 48])?;
        let _reservation =
            budget.reserve(u64::try_from(bytes).map_err(|_| Error::SizeOverflow)?)?;
        let maximum = samples.iter().map(|s| s.1).fold(0.0_f64, f64::max);
        let mut sorted: Vec<_> = samples.iter().map(|&(v, w)| (v, w / maximum)).collect();
        sorted.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut values = Vec::new();
        let mut weights = Vec::new();
        for (value, weight) in sorted {
            if values.last().is_some_and(|last| *last == value) {
                *weights.last_mut().unwrap() += weight;
            } else {
                values.push(value);
                weights.push(weight);
            }
        }
        let total = weights.iter().sum::<f64>();
        let mut cumulative = 0.0;
        let mut scores = Vec::with_capacity(values.len());
        let normal = Normal::new(0.0, 1.0).map_err(|e| invalid(e.to_string()))?;
        for weight in weights {
            let probability = (cumulative + 0.5 * weight) / total;
            cumulative += weight;
            if probability <= 0.0 || probability >= 1.0 {
                return Err(invalid("empirical probabilities exceed floating-point resolution; aggregate or regularize weights explicitly"));
            }
            let score = normal.inverse_cdf(probability);
            if !score.is_finite() || scores.last().is_some_and(|last| *last >= score) {
                return Err(invalid(
                    "normal-score knots are not numerically distinguishable",
                ));
            }
            scores.push(score);
        }
        Ok(Self {
            values,
            scores,
            tail,
            _reservation,
        })
    }
    pub fn knots(&self) -> impl ExactSizeIterator<Item = (f64, f64)> + '_ {
        self.values.iter().copied().zip(self.scores.iter().copied())
    }
    pub fn forward(&self, value: f64) -> Result<f64> {
        interpolate(&self.values, &self.scores, value, self.tail)
    }
    pub fn inverse(&self, score: f64) -> Result<f64> {
        interpolate(&self.scores, &self.values, score, self.tail)
    }
    /// Deterministic midpoint probability quadrature for E[g(Y)] and Var[g(Y)].
    /// This is an approximation whose convergence must be checked by increasing
    /// `points`. Inverse-transforming the kriging mean is generally a biased shortcut.
    pub fn conditional_moments(
        &self,
        mean: f64,
        variance: f64,
        points: usize,
    ) -> Result<(f64, f64)> {
        let normal = Normal::new(0.0, 1.0).map_err(|e| invalid(e.to_string()))?;
        self.moments_from_quantiles(
            mean,
            variance,
            points,
            (0..points).map(|i| moments::quantile(&normal, i, points)),
        )
    }
    /// Reuses standard-normal quadrature points across attributes and predictions.
    /// Identical arithmetic and tail policy to `conditional_moments`, without
    /// repeated inverse-CDF evaluation or allocation during prediction.
    pub fn conditional_moments_cached(
        &self,
        mean: f64,
        variance: f64,
        quadrature: &MomentQuadrature,
    ) -> Result<(f64, f64)> {
        self.moments_from_quantiles(
            mean,
            variance,
            quadrature.points(),
            quadrature.quantiles.iter().copied().map(Ok),
        )
    }
    fn moments_from_quantiles(
        &self,
        mean: f64,
        variance: f64,
        points: usize,
        quantiles: impl Iterator<Item = Result<f64>>,
    ) -> Result<(f64, f64)> {
        if !mean.is_finite() || !variance.is_finite() || variance < 0.0 || points == 0 {
            return Err(invalid("Gaussian moments or quadrature count"));
        }
        if variance == 0.0 {
            return Ok((self.inverse(mean)?, 0.0));
        }
        let deviation = variance.sqrt();
        let mut average = 0.0;
        let mut m2 = 0.0;
        for (i, quantile) in quantiles.enumerate() {
            let value = self.inverse(mean + deviation * quantile?)?;
            let delta = value - average;
            average += delta / (i + 1) as f64;
            m2 += delta * (value - average);
            if !average.is_finite() || !m2.is_finite() {
                return Err(Error::Numerical {
                    context: "back-transformed moments",
                });
            }
        }
        Ok((average, (m2 / points as f64).max(0.0)))
    }
}
fn interpolate(x: &[f64], y: &[f64], value: f64, tail: TailPolicy) -> Result<f64> {
    if !value.is_finite() {
        return Err(invalid("transform input must be finite"));
    }
    let n = x.len();
    let index = x.partition_point(|x| *x < value);
    if index < n && x[index] == value {
        return Ok(y[index]);
    }
    if value < x[0] || value > x[n - 1] {
        match tail {
            TailPolicy::Reject => return Err(invalid("transform input outside empirical knots")),
            TailPolicy::Clamp => return Ok(if value < x[0] { y[0] } else { y[n - 1] }),
            TailPolicy::Linear if n == 1 => {
                return Err(invalid(
                    "a constant sample has no linear extrapolation slope",
                ))
            }
            TailPolicy::Linear => {}
        }
    }
    if n == 1 {
        return Ok(y[0]);
    }
    let upper = index.clamp(1, n - 1);
    let lower = upper - 1;
    let width = x[upper] - x[lower];
    let fraction = if width.is_finite() {
        (value - x[lower]) / width
    } else {
        (value * 0.5 - x[lower] * 0.5) / (x[upper] * 0.5 - x[lower] * 0.5)
    };
    let out = if (0.0..=1.0).contains(&fraction) {
        (1.0 - fraction) * y[lower] + fraction * y[upper]
    } else {
        y[lower] + fraction * (y[upper] - y[lower])
    };
    if !out.is_finite() {
        return Err(Error::Numerical {
            context: "transform interpolation or tail extrapolation",
        });
    }
    Ok(out)
}
