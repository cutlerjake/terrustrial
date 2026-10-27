use super::{Axis, Marginal};
use crate::{error::invalid, execution::Cancellation, Error, Result};
use rand::RngExt;
use rand_distr::{Distribution, StandardNormal};
use statrs::function::erf::erf;

pub(super) fn legendre(x: f64, out: &mut [f64]) {
    out[0] = 1.0;
    if out.len() > 1 {
        out[1] = x;
    }
    for n in 1..out.len().saturating_sub(1) {
        let n = n as f64;
        out[n as usize + 1] =
            ((2.0 * n + 1.0) * x * out[n as usize] - n * out[n as usize - 1]) / (n + 1.0);
    }
}

impl Marginal {
    pub(super) fn validate(&self, axis: Axis) -> Result<()> {
        match (self, axis) {
            (Self::TruncatedNormal { mean, deviation }, Axis::Continuous { .. })
                if mean.is_finite() && (-1.0..=1.0).contains(mean)
                    && deviation.is_finite() && *deviation >= f64::MIN_POSITIVE
                    && (*mean + *deviation != *mean || *mean - *deviation != *mean) => Ok(()),
            (Self::Categorical(probabilities), Axis::Categorical { classes } | Axis::SmoothedCategorical { classes, .. })
                if probabilities.len() == classes as usize
                    && probabilities.iter().all(|p| p.is_finite() && *p >= 0.0)
                    && (probabilities.iter().sum::<f64>() - 1.0).abs() <= 1e-12 => Ok(()),
            _ => Err(invalid("prototype marginal must match its axis; categorical probabilities must sum to one; normal location must be in [-1,1] with a resolvable positive deviation")),
        }
    }

    pub(super) fn normalize(&mut self) {
        if let Self::Categorical(p) = self {
            let sum = p.iter().sum::<f64>();
            p.iter_mut().for_each(|v| *v /= sum);
        }
    }

    // Integrate in standardized coordinates for narrow densities, in physical
    // coordinates for wide ones. The omitted normal tails beyond 12 sigma have
    // mass below 4e-33. Success requires successive rules to agree for every moment.
    pub(super) fn moments(&self, degree: usize, cancel: &Cancellation) -> Result<Vec<f64>> {
        let Self::TruncatedNormal { mean, deviation } = *self else {
            unreachable!()
        };
        let mut previous = vec![0.0; degree + 1];
        let mut basis = vec![0.0; degree + 1];
        let standardized = deviation < 0.25;
        let (lower, upper) = if standardized {
            (
                ((-1.0 - mean) / deviation).max(-12.0),
                ((1.0 - mean) / deviation).min(12.0),
            )
        } else {
            (-1.0, 1.0)
        };
        let mut count = 16;
        while count <= 2048 {
            cancel.check()?;
            let mut next = vec![0.0; degree + 1];
            let mut total = 0.0;
            for i in 0..count {
                let (node, weight) = gauss_node(count, i)?;
                let coordinate = (upper + lower) * 0.5 + (upper - lower) * 0.5 * node;
                let (x, t) = if standardized {
                    ((mean + deviation * coordinate).clamp(-1.0, 1.0), coordinate)
                } else {
                    (coordinate, (coordinate - mean) / deviation)
                };
                let weight = weight * (-0.5 * t * t).exp();
                legendre(x, &mut basis);
                for (moment, p) in next.iter_mut().zip(&basis) {
                    *moment += weight * p;
                }
                total += weight;
            }
            for moment in &mut next {
                *moment /= total;
            }
            if count > degree
                && count > 16
                && next
                    .iter()
                    .zip(&previous)
                    .all(|(a, b)| (a - b).abs() <= 2e-13)
            {
                next[0] = 1.0;
                return Ok(next);
            }
            previous = next;
            count *= 2;
        }
        Err(Error::Convergence {
            context: "truncated-normal Legendre quadrature",
            iterations: 2048,
            residual: f64::INFINITY,
        })
    }

    pub(super) fn log_density(&self, value: f64) -> f64 {
        match self {
            Self::Categorical(p) => p[value as usize].ln(),
            Self::TruncatedNormal { mean, deviation } => {
                if !(-1.0..=1.0).contains(&value) {
                    return f64::NEG_INFINITY;
                }
                let t = (value - mean) / deviation;
                let mass = 0.5
                    * (erf(((1.0 - mean) / deviation) / std::f64::consts::SQRT_2)
                        + erf(((1.0 + mean) / deviation) / std::f64::consts::SQRT_2));
                -0.5 * t * t - deviation.ln() - mass.ln() - 0.5 * (2.0 * std::f64::consts::PI).ln()
            }
        }
    }

    pub(super) fn sample(&self, rng: &mut impl rand::Rng, cancel: &Cancellation) -> Result<f64> {
        match self {
            Self::Categorical(p) => {
                let u: f64 = rng.random();
                let mut cdf = 0.0;
                let mut last = 0;
                for (i, probability) in p.iter().enumerate() {
                    if *probability > 0.0 {
                        last = i;
                    }
                    cdf += probability;
                    if u < cdf {
                        return Ok(i as f64);
                    }
                }
                Ok(last as f64)
            }
            Self::TruncatedNormal { mean, deviation } => {
                for _ in 0..1024 {
                    cancel.check()?;
                    let value = if *deviation <= 1.0 {
                        let z: f64 = StandardNormal.sample(rng);
                        mean + deviation * z
                    } else {
                        let x = rng.random_range(-1.0..1.0);
                        let t = (x - mean) / deviation;
                        if rng.random::<f64>() >= (-0.5 * t * t).exp() {
                            continue;
                        }
                        x
                    };
                    if (-1.0..=1.0).contains(&value) {
                        return Ok(value);
                    }
                }
                Err(Error::Numerical {
                    context: "truncated-normal rejection sampling exceeded 1024 attempts",
                })
            }
        }
    }
}

fn gauss_node(count: usize, index: usize) -> Result<(f64, f64)> {
    let n = count as f64;
    let mut x = (std::f64::consts::PI * (index as f64 + 0.75) / (n + 0.5)).cos();
    for _ in 0..32 {
        let (mut previous, mut current) = (1.0, x);
        for k in 1..count {
            let k = k as f64;
            (previous, current) = (
                current,
                ((2.0 * k + 1.0) * x * current - k * previous) / (k + 1.0),
            );
        }
        let derivative = n * (x * current - previous) / (x * x - 1.0);
        let step = current / derivative;
        if step.abs() <= 4.0 * f64::EPSILON {
            return Ok((x, 2.0 / ((1.0 - x * x) * derivative * derivative)));
        }
        x -= step;
    }
    Err(Error::Numerical {
        context: "Gauss-Legendre node convergence",
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn moments_agree_with_analytic_normal_half_normal_and_uniform_integrals() -> Result<()> {
        let cancel = Cancellation::default();
        let m = Marginal::TruncatedNormal {
            mean: 0.0,
            deviation: 0.1,
        }
        .moments(20, &cancel)?;
        assert!((m[1]).abs() < 1e-14);
        assert!((m[2] - (3.0 * 0.01 - 1.0) / 2.0).abs() < 1e-13);
        assert!((m[4] - (35.0 * 3.0 * 0.0001 - 30.0 * 0.01 + 3.0) / 8.0).abs() < 1e-13);
        let sigma = 0.03;
        let m = Marginal::TruncatedNormal {
            mean: 1.0,
            deviation: sigma,
        }
        .moments(20, &cancel)?;
        let mean = 1.0 - sigma * (2.0 / std::f64::consts::PI).sqrt();
        let second = 1.0 - 2.0 * sigma * (2.0 / std::f64::consts::PI).sqrt() + sigma * sigma;
        assert!((m[1] - mean).abs() < 1e-13);
        assert!((m[2] - (3.0 * second - 1.0) / 2.0).abs() < 1e-13);
        let m = Marginal::TruncatedNormal {
            mean: 0.0,
            deviation: f64::MAX,
        }
        .moments(32, &cancel)?;
        assert!(m.iter().skip(1).all(|v| v.abs() < 1e-13));
        // Simpson integration uses a separate polynomial recurrence and a
        // uniform physical grid, independent of the production Gaussian rule.
        let location = 0.27;
        let sigma = 0.31;
        let m = Marginal::TruncatedNormal {
            mean: location,
            deviation: sigma,
        }
        .moments(16, &cancel)?;
        let mut integrals = [0.0; 17];
        let n = 100_000;
        for i in 0..=n {
            let x = -1.0 + 2.0 * (i as f64) / (n as f64);
            let weight = if i == 0 || i == n {
                1.0
            } else if i % 2 == 0 {
                2.0
            } else {
                4.0
            };
            let density = (-0.5 * ((x - location) / sigma).powi(2)).exp();
            let mut p = [0.0; 17];
            p[0] = 1.0;
            p[1] = x;
            for k in 2..17 {
                p[k] = ((2 * k - 1) as f64 * x * p[k - 1] - (k - 1) as f64 * p[k - 2]) / k as f64;
            }
            for k in 0..17 {
                integrals[k] += weight * density * p[k];
            }
        }
        for (k, v) in m.iter().enumerate() {
            assert!(
                (v - integrals[k] / integrals[0]).abs() < 1e-11,
                "moment {k}"
            );
        }
        Ok(())
    }
}
