//! Families positive definite in every Euclidean dimension, as required by the
//! Paciorek-Schervish determinant construction. Spherical is intentionally absent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Family {
    Gaussian,
    Exponential,
    Matern32,
    Matern52,
}
impl Family {
    pub(super) fn log_correlation(self, distance_squared: f64) -> f64 {
        match self {
            Self::Gaussian => -distance_squared,
            Self::Exponential => -distance_squared.sqrt(),
            Self::Matern32 => -matern_decay(3.0_f64.sqrt() * distance_squared.sqrt(), false),
            Self::Matern52 => -matern_decay(5.0_f64.sqrt() * distance_squared.sqrt(), true),
        }
    }
    /// Invert gamma/sill in log space, retaining near-zero lag precision.
    pub(super) fn metric_target(self, fraction: f64) -> f64 {
        let target = -(-fraction).ln_1p();
        match self {
            Self::Gaussian => target,
            Self::Exponential => target * target,
            Self::Matern32 | Self::Matern52 => {
                if target == 0.0 {
                    return 0.0;
                }
                let five = self == Self::Matern52;
                let mut lower = 0.0;
                let mut upper = (target.sqrt() * 4.0).max(target * 2.0);
                while matern_decay(upper, five) < target {
                    upper *= 2.0;
                }
                for _ in 0..64 {
                    let middle = lower * 0.5 + upper * 0.5;
                    if matern_decay(middle, five) < target {
                        lower = middle;
                    } else {
                        upper = middle;
                    }
                }
                let x = lower * 0.5 + upper * 0.5;
                x * x / if five { 5.0 } else { 3.0 }
            }
        }
    }
}
fn matern_decay(x: f64, five: bool) -> f64 {
    if x.is_infinite() {
        return x;
    }
    if x < 0.05 {
        if !five {
            // x-ln(1+x) = x²/2-x³/3+..., avoiding cancellation.
            let mut power = x * x;
            let mut value = 0.0;
            for n in 2..=18 {
                value += power / n as f64;
                power *= -x;
            }
            value
        } else {
            // Integrate (x+x²)/(3+3x+x²), the derivative of
            // x-ln(1+x+x²/3). Recurrence for its Taylor coefficients.
            let mut previous = 0.0;
            let mut before_previous = 0.0;
            let mut power = x * x;
            let mut value = 0.0;
            for n in 1..=18 {
                let coefficient =
                    (if n <= 2 { 1.0 } else { 0.0 } - 3.0 * previous - before_previous) / 3.0;
                value += coefficient * power / (n + 1) as f64;
                before_previous = previous;
                previous = coefficient;
                power *= x;
            }
            value
        }
    } else if five {
        // Log-sum form avoids x*x overflow for very distant sites.
        x - (2.0 * x.ln() + (1.0 / 3.0 + 1.0 / x + (1.0 / x) / x).ln())
    } else {
        x - x.ln_1p()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn inverse_correlations_cover_small_lags_and_near_sill() {
        for family in [
            Family::Gaussian,
            Family::Exponential,
            Family::Matern32,
            Family::Matern52,
        ] {
            for fraction in [
                1e-200,
                1e-20,
                1e-12,
                0.001,
                0.1,
                0.5,
                0.99,
                1.0 - f64::EPSILON,
            ] {
                let q = family.metric_target(fraction);
                let recovered = -family.log_correlation(q).exp_m1();
                // Exponential squared distance can underflow at fractions <1e-162.
                if q == 0.0 {
                    assert_eq!(family, Family::Exponential);
                    continue;
                }
                assert!(
                    (recovered - fraction).abs() <= 2e-12 * fraction,
                    "{family:?} {fraction} {q} {recovered}"
                );
            }
            assert_eq!(family.log_correlation(f64::INFINITY), f64::NEG_INFINITY);
        }
    }
}
