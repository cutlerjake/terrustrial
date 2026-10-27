//! Extreme covariance units are factored in a representable power-of-two scale.
use super::*;

fn conditioning_units(covariance: MatRef<'_, f64>, observations: &[Observation]) -> f64 {
    let maximum = (0..covariance.nrows())
        .map(|i| {
            let variance = covariance[(i, i)];
            let noise = observations[i].noise_variance();
            let total = variance + noise;
            if total.is_finite() {
                total.abs()
            } else {
                // Finite variance and noise can have an unrepresentable sum.
                // Either operand supplies suitable units before they are added.
                variance.abs().max(noise)
            }
        })
        .fold(0.0_f64, f64::max);
    if maximum == 0.0 || (1e-100..=1e100).contains(&maximum) {
        1.0
    } else {
        // Subnormal variances use MIN_POSITIVE, avoiding an infinite reciprocal.
        // Other values use the power of two at the start of their exponent bin.
        f64::from_bits(maximum.to_bits() & (0x7ff_u64 << 52)).max(f64::MIN_POSITIVE)
    }
}

/// Add measurement noise in representable units. Preserve the original rounded
/// physical sum whenever it is finite, including its subnormal rounding.
pub(super) fn prepare_conditioning(
    mut covariance: MatMut<'_, f64>,
    observations: &[Observation],
) -> Result<f64> {
    let units = conditioning_units(covariance.as_ref(), observations);
    if units == 1.0 {
        for i in 0..covariance.nrows() {
            covariance[(i, i)] += observations[i].noise_variance();
        }
    } else {
        for (j, col) in covariance.col_iter_mut().enumerate() {
            for (i, value) in col.iter_mut().enumerate() {
                if i == j {
                    let noise = observations[i].noise_variance();
                    let total = *value + noise;
                    *value = if total.is_finite() {
                        normalize(total, units)?
                    } else {
                        normalize(*value, units)? + normalize(noise, units)?
                    };
                } else {
                    *value = normalize(*value, units)?;
                }
            }
        }
    }
    Ok(units)
}

pub(super) fn normalize(value: f64, units: f64) -> Result<f64> {
    if units == 1.0 {
        return Ok(value);
    }
    let normalized = value / units;
    if !normalized.is_finite() || (value != 0.0 && normalized == 0.0) {
        return Err(Error::Numerical {
            context: "covariance dynamic range exceeds representable normalization",
        });
    }
    Ok(normalized)
}

pub(super) fn normalize_matrix(matrix: MatMut<'_, f64>, units: f64) -> Result<()> {
    if units != 1.0 {
        for col in matrix.col_iter_mut() {
            for value in col.iter_mut() {
                *value = normalize(*value, units)?;
            }
        }
    }
    Ok(())
}

pub(super) fn restore(
    mut covariance: MatMut<'_, f64>,
    mut factor: MatMut<'_, f64>,
    units: f64,
    tolerance: NumericalTolerance,
    prior_scale: f64,
) -> Result<usize> {
    if units == 1.0 {
        return Ok(0);
    }
    let mut rounded = 0;
    if units > 1.0 {
        let maximum = covariance
            .as_ref()
            .col_iter()
            .flat_map(|column| column.iter().map(|v| v.abs()))
            .fold(0.0_f64, f64::max);
        let limit = f64::MAX / units;
        if maximum > limit {
            if maximum - limit > threshold(tolerance, maximum.max(prior_scale), units)? {
                return Err(Error::Numerical {
                    context: "conditional variance exceeds representable physical units",
                });
            }
            // A few ulps above MAX can arise from a representable posterior.
            // Repair only within the explicit tolerance. A common contraction
            // preserves positive semidefiniteness and scales draws consistently.
            let contraction = (limit / maximum).next_down();
            for column in covariance.as_mut().col_iter_mut() {
                for value in column.iter_mut() {
                    *value *= contraction;
                }
            }
            let root_contraction = contraction.sqrt();
            for column in factor.as_mut().col_iter_mut() {
                for value in column.iter_mut() {
                    *value *= root_contraction;
                }
            }
            rounded = 1;
        }
    }
    for j in 0..covariance.ncols() {
        for i in 0..covariance.nrows() {
            let value = covariance[(i, j)];
            let restored = value * units;
            if !restored.is_finite() || (i == j && value != 0.0 && restored == 0.0) {
                return Err(Error::Numerical {
                    context: "conditional variance exceeds representable physical units",
                });
            }
            covariance[(i, j)] = restored;
        }
    }
    let root = units.sqrt();
    for col in factor.col_iter_mut() {
        for value in col.iter_mut() {
            *value *= root;
        }
    }
    Ok(rounded)
}

pub(super) fn threshold(tolerance: NumericalTolerance, scale: f64, units: f64) -> Result<f64> {
    if units == 1.0 {
        return tolerance.threshold(scale);
    }
    let physical_scale = scale * units;
    if physical_scale.is_finite() {
        tolerance.threshold(physical_scale)?;
    } else {
        // The scale can round past MAX although its relative tolerance fits.
        // Validate parameters and multiply the relative term before restoring
        // units, leaving the covariance range check to restore().
        tolerance.threshold(0.)?;
        if !(tolerance.absolute + (tolerance.relative * scale) * units).is_finite() {
            return Err(invalid("numerical tolerance overflow"));
        }
    }
    Ok(tolerance.absolute / units + tolerance.relative * scale)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn physical_range_repair_respects_tolerance_and_scales_the_joint_factor() {
        let units = 2_f64.powi(1023);
        for (diagonal, tolerance, succeeds) in [
            (2_f64, NumericalTolerance::default(), true),
            (3., NumericalTolerance::default(), false),
            (
                2.,
                NumericalTolerance {
                    absolute: 0.,
                    relative: 0.,
                },
                false,
            ),
        ] {
            let mut covariance = Mat::from_fn(2, 2, |i, j| if i == j { diagonal } else { 1. });
            let mut factor = Mat::zeros(2, 2);
            factor[(0, 0)] = diagonal.sqrt();
            factor[(1, 0)] = 1. / diagonal.sqrt();
            factor[(1, 1)] = (diagonal - 1. / diagonal).sqrt();
            let result = restore(
                covariance.as_mut(),
                factor.as_mut(),
                units,
                tolerance,
                diagonal,
            );
            if !succeeds {
                assert!(result.is_err());
                continue;
            }
            assert_eq!(result.unwrap(), 1);
            for j in 0..2 {
                for i in 0..2 {
                    assert!(covariance[(i, j)].is_finite());
                    let reconstructed: f64 = (0..2)
                        .map(|k| (factor[(i, k)] / units.sqrt()) * (factor[(j, k)] / units.sqrt()))
                        .sum();
                    assert!((reconstructed - covariance[(i, j)] / units).abs() < 2e-15);
                }
            }
            assert_eq!(covariance[(0, 1)] / covariance[(0, 0)], 0.5);
            assert!((covariance[(0, 0)] / f64::MAX - 1.).abs() < 1e-15);
        }
    }
}
