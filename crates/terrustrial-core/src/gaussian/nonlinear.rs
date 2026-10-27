//! Probability corrections and Gaussian-anamorphosis disjunctive kriging.
use crate::{
    data::{Observation, Point3, Site},
    error::invalid,
    execution::{Cancellation, MemoryBudget},
    gaussian::{GaussianPlan, NumericalTolerance, PrepareOptions, Simple, Uncertainty},
    model::Covariance,
    Error, Result,
};

/// Euclidean projection of categorical estimates onto the probability simplex.
/// This changes the estimator; callers should expose raw values as diagnostics.
pub fn categorical_probabilities(raw: &[f64]) -> Result<Vec<f64>> {
    if raw.is_empty() || raw.iter().any(|x| !x.is_finite()) {
        return Err(invalid("finite nonempty probability vector required"));
    }
    // Subtract the maximum before the projection; this preserves the result and
    // avoids overflowing the sum for large positive estimates.
    let maximum = raw.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut sorted: Vec<_> = raw.iter().map(|x| (x - maximum).max(-2.0)).collect();
    sorted.sort_by(|a, b| b.total_cmp(a));
    let mut sum = 0.0;
    let mut threshold = 0.0;
    for (i, value) in sorted.iter().enumerate() {
        sum += value;
        let t = (sum - 1.0) / (i + 1) as f64;
        if *value > t {
            threshold = t;
        }
    }
    Ok(raw
        .iter()
        .map(|x| ((x - maximum).max(-2.0) - threshold).max(0.0))
        .collect())
}

/// Equal-weight isotonic least-squares correction for ordered threshold CDFs.
/// The last threshold need not cover the entire distribution and is not forced to 1.
pub fn threshold_cdf(raw: &[f64]) -> Result<Vec<f64>> {
    if raw.is_empty() || raw.iter().any(|x| !x.is_finite()) {
        return Err(invalid("finite nonempty CDF required"));
    }
    let mut blocks: Vec<(usize, f64)> = Vec::new();
    for value in raw {
        blocks.push((1, *value));
        while blocks.len() > 1 {
            let n = blocks.len();
            let a = blocks[n - 2];
            let b = blocks[n - 1];
            if a.1 <= b.1 {
                break;
            }
            let count = a.0.checked_add(b.0).ok_or(Error::SizeOverflow)?;
            // Weighted convex sum rather than potentially overflowing raw sums.
            let mean = a.1 * (a.0 as f64 / count as f64) + b.1 * (b.0 as f64 / count as f64);
            blocks.truncate(n - 2);
            blocks.push((count, mean));
        }
    }
    let mut out = Vec::with_capacity(raw.len());
    for (count, mean) in blocks {
        out.extend(std::iter::repeat_n(mean.clamp(0.0, 1.0), count));
    }
    Ok(out)
}

/// Normalized probabilists' Hermite polynomial, orthonormal for N(0,1).
pub fn hermite(order: usize, value: f64) -> Result<f64> {
    if !value.is_finite() {
        return Err(invalid("Hermite input must be finite"));
    }
    let (mut previous, mut current) = (1.0, value);
    if order == 0 {
        return Ok(previous);
    }
    for n in 1..order {
        let next = (value * current - (n as f64).sqrt() * previous) / ((n + 1) as f64).sqrt();
        previous = current;
        current = next;
        if !current.is_finite() {
            return Err(Error::Numerical {
                context: "Hermite recurrence overflow",
            });
        }
    }
    Ok(current)
}

struct HermiteCovariance<'a, C> {
    base: &'a C,
    order: i32,
}
impl<C: Covariance> Covariance for HermiteCovariance<'_, C> {
    fn variables(&self) -> usize {
        self.base.variables()
    }
    fn point_covariance(&self, a: Point3, va: usize, b: Point3, vb: usize) -> Result<f64> {
        let aa = self.base.point_covariance(a, va, a, va)?;
        let bb = self.base.point_covariance(b, vb, b, vb)?;
        if aa <= 0.0 || bb <= 0.0 {
            return Err(invalid(
                "positive marginal variance required for Gaussian correlation",
            ));
        }
        let correlation = self.base.point_covariance(a, va, b, vb)? / aa.sqrt() / bb.sqrt();
        if !correlation.is_finite() || correlation.abs() > 1.0 + 1e-10 {
            return Err(invalid("invalid correlation in disjunctive model"));
        }
        Ok(correlation.clamp(-1.0, 1.0).powi(self.order))
    }
}

/// Predict a function represented by coefficients in the normalized Hermite basis.
/// Observed values MUST already be marginal standard-normal scores from a jointly
/// Gaussian model. Each order uses rho(h)^order; this is not a generic non-Gaussian
/// conditional distribution. Only point support and zero measurement noise apply.
pub fn disjunctive<C: Covariance>(
    model: &C,
    observations: &[Observation],
    targets: &[Site],
    coefficients: &[f64],
    budget: &MemoryBudget,
    cancel: &Cancellation,
) -> Result<Vec<f64>> {
    if coefficients.is_empty()
        || coefficients.iter().any(|x| !x.is_finite())
        || coefficients.len() > i32::MAX as usize
    {
        return Err(invalid("disjunctive coefficients"));
    }
    if observations
        .iter()
        .any(|o| !o.site().support.is_point() || o.noise_variance() != 0.0)
        || targets.iter().any(|t| !t.support.is_point())
    {
        return Err(invalid(
            "disjunctive reference supports noise-free point observations only",
        ));
    }
    if targets.is_empty() {
        return Err(invalid("empty disjunctive targets"));
    }
    let mut out = vec![coefficients[0]; targets.len()];
    for (order, coefficient) in coefficients.iter().copied().enumerate().skip(1) {
        cancel.check()?;
        if coefficient == 0.0 {
            continue;
        }
        let covariance = HermiteCovariance {
            base: model,
            order: order as i32,
        };
        let plan = GaussianPlan::prepare(
            &covariance,
            observations,
            targets,
            &Simple,
            PrepareOptions {
                uncertainty: Uncertainty::None,
                tolerance: NumericalTolerance::default(),
                budget,
                cancellation: cancel,
            },
        )?;
        let transformed = observations
            .iter()
            .map(|o| hermite(order, o.value()))
            .collect::<Result<Vec<_>>>()?;
        for (value, estimate) in out.iter_mut().zip(plan.estimate(&transformed)?) {
            *value += coefficient * estimate;
        }
    }
    if out.iter().any(|x| !x.is_finite()) {
        return Err(Error::Numerical {
            context: "disjunctive prediction overflow",
        });
    }
    Ok(out)
}

#[derive(Clone, Debug)]
pub struct ProbabilityEstimate {
    pub raw: Vec<f64>,
    pub corrected: Vec<f64>,
    pub maximum_correction: f64,
}
fn correction(raw: Vec<f64>, categorical: bool) -> Result<ProbabilityEstimate> {
    let corrected = if categorical {
        categorical_probabilities(&raw)?
    } else {
        threshold_cdf(&raw)?
    };
    let maximum_correction = raw
        .iter()
        .zip(&corrected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    Ok(ProbabilityEstimate {
        raw,
        corrected,
        maximum_correction,
    })
}
/// Categorical indicator estimation with shared spatial weights. This assumes
/// proportional indicator covariance models, not a general joint categorical law.
/// SK uses the supplied class priors; unbiased OK weights cancel their effect.
/// Corrections are reported alongside the raw estimates rather than hidden.
pub fn categorical_indicators(
    plan: &GaussianPlan,
    labels: &[usize],
    priors: &[f64],
) -> Result<Vec<ProbabilityEstimate>> {
    if priors.is_empty()
        || priors
            .iter()
            .any(|p| !p.is_finite() || !(0.0..=1.0).contains(p))
        || (priors.iter().sum::<f64>() - 1.0).abs() > 1e-12
        || labels.iter().any(|v| *v >= priors.len())
    {
        return Err(invalid("categorical labels or prior probabilities"));
    }
    let targets = plan.weights().ncols();
    let mut raw = vec![vec![0.0; priors.len()]; targets];
    for (class, &prior) in priors.iter().enumerate() {
        let values: Vec<_> = labels
            .iter()
            .map(|value| if *value == class { 1.0 } else { 0.0 })
            .collect();
        let estimate =
            plan.estimate_with_mean(&values, &vec![prior; labels.len()], &vec![prior; targets])?;
        for (row, value) in raw.iter_mut().zip(estimate) {
            row[class] = value;
        }
    }
    raw.into_iter().map(|row| correction(row, true)).collect()
}
/// Threshold indicators with a shared covariance shape and increasing cutoffs.
/// Supplied priors are cumulative probabilities at those cutoffs, not class masses.
pub fn threshold_indicators(
    plan: &GaussianPlan,
    values: &[f64],
    cutoffs: &[f64],
    priors: &[f64],
) -> Result<Vec<ProbabilityEstimate>> {
    if cutoffs.is_empty()
        || cutoffs.len() != priors.len()
        || cutoffs.iter().chain(values).any(|v| !v.is_finite())
        || cutoffs.windows(2).any(|p| p[0] >= p[1])
        || priors
            .iter()
            .any(|p| !p.is_finite() || !(0.0..=1.0).contains(p))
        || priors.windows(2).any(|p| p[0] > p[1])
    {
        return Err(invalid("indicator cutoffs, values or cumulative priors"));
    }
    let targets = plan.weights().ncols();
    let mut raw = vec![vec![0.0; cutoffs.len()]; targets];
    for (index, (&cutoff, &prior)) in cutoffs.iter().zip(priors).enumerate() {
        let indicator: Vec<_> = values
            .iter()
            .map(|value| if *value <= cutoff { 1.0 } else { 0.0 })
            .collect();
        let estimate = plan.estimate_with_mean(
            &indicator,
            &vec![prior; values.len()],
            &vec![prior; targets],
        )?;
        for (row, value) in raw.iter_mut().zip(estimate) {
            row[index] = value;
        }
    }
    raw.into_iter().map(|row| correction(row, false)).collect()
}
