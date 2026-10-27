//! Nested-prefix aggregation of high-order moment evidence.
//!
//! Each moment uses every replicate containing its required prefix, rather than
//! discarding all incomplete replicates. This implements the coefficient-pooling
//! principle in Yao et al. (2021, doi:10.1007/s11004-021-09923-3), with two EXPLICIT
//! changes relative to the printed telescoping expression: retain the target-only base
//! term, and express conditioning kernels relative to uniform probability measure.
//! Thus a continuous axis uses 2*K and a C-class axis uses C*delta. These changes
//! make projection dimensionally consistent, preserve unconditional moments, and
//! recover ordinary complete-replicate conditioning when all prefixes are complete.
//! This is not a claim to reproduce the paper's literal equation 10 or its sampler.
use super::{
    inner_product, legendre, Axis, Conditional, FitOptions, LogWeight, Plan, ReplicateVisitor, Sum,
};
use crate::{
    error::{checked_len, invalid},
    execution::Reservation,
    Error, Result,
};

/// Replayable, stable-order source of nested partial replicates. All target axes
/// must be present; conditioning contains the FIRST r axes of the supplied event.
/// A missing axis ends a prefix, even if farther axes are available. Two scans are
/// used so weights depend on exact support counts without storing replicate rows.
/// Do not mutate the source between scans. Changed prefix counts are rejected;
/// value-only changes cannot be detected by this interface.
pub trait PrefixSource {
    fn scan_prefixes(&self, visitor: &mut ReplicateVisitor<'_>) -> Result<()>;
}
impl PrefixSource for [(Vec<f64>, Vec<f64>)] {
    fn scan_prefixes(&self, visitor: &mut ReplicateVisitor<'_>) -> Result<()> {
        for (target, condition) in self {
            visitor(target, condition)?;
        }
        Ok(())
    }
}
/// Policy for interactions whose required prefix has insufficient evidence.
#[derive(Clone, Copy, Debug)]
pub enum Unsupported {
    Reject,
    /// Set those higher interaction coefficients to zero. This is an explicit
    /// modeling approximation: unsupported conditioning axes add no information.
    /// Their indices/counts remain visible in `used_prefix` and `support`.
    ZeroHigherInteractions,
}
#[derive(Clone, Copy)]
pub struct Options<'a> {
    /// Required support at every conditioning prefix, including the full event.
    /// This is an explicit evidence-count floor, not a statistical confidence level.
    pub minimum_replicates: u64,
    pub unsupported: Unsupported,
    pub fit: FitOptions<'a>,
}
#[derive(Debug)]
pub struct Aggregated<'a> {
    pub distribution: Conditional<'a>,
    pub evidence: Evidence,
}
#[derive(Debug)]
pub struct Evidence {
    /// Entry n counts rows having at least n conditioning attributes. Entry zero
    /// includes target-only rows. Retained for auditing which moments are informed.
    pub support: Vec<u64>,
    /// Highest prefix admitted by the evidence floor, possibly zero (target only).
    pub used_prefix: usize,
    /// Minimum |signed sum| / sum |terms| when forming one row's aggregate weight.
    /// A small value warns of cancellation within that row. Exact cancellations
    /// give zero. This is separate from the final distribution's normalizer check.
    pub minimum_row_fraction: f64,
    _reservation: Reservation,
}

impl Plan {
    /// Pool Legendre/indicator moments over nested prefixes and fit the SAME joint
    /// positive mixture as `condition`. Empty evidence fails; unsupported levels
    /// follow the explicitly chosen policy. Memory is independent
    /// of row count. Ordering is a modeling choice and must be recorded by callers.
    pub fn condition_prefixes<S: PrefixSource + ?Sized>(
        &self,
        source: &S,
        conditioning: &[(Axis, f64)],
        options: Options<'_>,
    ) -> Result<Aggregated<'_>> {
        options.fit.validate()?;
        if options.minimum_replicates == 0 {
            return Err(invalid("positive aggregation evidence floor required"));
        }
        let levels = conditioning
            .len()
            .checked_add(1)
            .ok_or(Error::SizeOverflow)?;
        let (mut width, mut maximum) = (0usize, 0usize);
        for &(axis, value) in conditioning {
            axis.validate_value(value)?;
            let n = axis.width()?;
            width = width.checked_add(n).ok_or(Error::SizeOverflow)?;
            maximum = maximum.max(n);
        }
        let bytes = checked_len(&[levels, 64])?
            .checked_add(checked_len(&[
                width.checked_add(maximum).ok_or(Error::SizeOverflow)?,
                8,
            ])?)
            .ok_or(Error::SizeOverflow)?;
        let _workspace = options
            .fit
            .budget
            .reserve(u64::try_from(bytes).map_err(|_| Error::SizeOverflow)?)?;
        let reservation = options
            .fit
            .budget
            .reserve(u64::try_from(checked_len(&[levels, 8])?).map_err(|_| Error::SizeOverflow)?)?;
        let mut histogram = vec![0u64; levels];
        source.scan_prefixes(&mut |targets, values| {
            options.fit.cancellation.check()?;
            self.validate_prefix(targets, values, conditioning)?;
            histogram[values.len()] = histogram[values.len()]
                .checked_add(1)
                .ok_or(Error::SizeOverflow)?;
            Ok(())
        })?;
        let mut support = histogram.clone();
        for n in (0..levels - 1).rev() {
            support[n] = support[n]
                .checked_add(support[n + 1])
                .ok_or(Error::SizeOverflow)?;
        }
        if support[0] < options.minimum_replicates {
            return Err(invalid("insufficient target evidence"));
        }
        let used_prefix = support
            .iter()
            .rposition(|count| *count >= options.minimum_replicates)
            .unwrap();
        if used_prefix < conditioning.len() && matches!(options.unsupported, Unsupported::Reject) {
            return Err(invalid("insufficient full-prefix evidence; reduce the event explicitly or supply more evidence"));
        }
        let mut offsets = Vec::with_capacity(conditioning.len());
        let mut event_basis = vec![0.0; width];
        let mut offset = 0;
        for &(axis, value) in conditioning {
            offsets.push(offset);
            let n = axis.width()?;
            if n > 0 {
                legendre(value, &mut event_basis[offset..offset + n]);
            }
            offset += n;
        }
        let mut row_basis = vec![0.0; maximum];
        // b_r = sum_{n<r} (1/C_n - 1/C_{n+1}) L_n + L_r/C_r.
        // Compute count differences as integers; converting counts to f64 before
        // subtracting would erase differences above 2^53. Equal-support terms are
        // exactly zero, so the complete-data case has no telescoping cancellation.
        let differences: Vec<_> = support[..=used_prefix]
            .windows(2)
            .map(|c| {
                if c[0] == c[1] {
                    None
                } else {
                    Some(((c[0] - c[1]) as f64).ln() - (c[0] as f64).ln() - (c[1] as f64).ln())
                }
            })
            .collect();
        let mut second_histogram = vec![0u64; levels];
        let mut minimum_row_fraction = 1.0_f64;
        let distribution = self.condition_weighted(options.fit, |visit| {
            source.scan_prefixes(&mut |targets, values| {
                options.fit.cancellation.check()?;
                self.validate_prefix(targets, values, conditioning)?;
                second_histogram[values.len()] = second_histogram[values.len()]
                    .checked_add(1)
                    .ok_or(Error::SizeOverflow)?;
                let mut sum = LogSum::new();
                let mut prefix = LogWeight {
                    sign: 1.0,
                    log: 0.0,
                };
                let prefix_len = values.len().min(used_prefix);
                for n in 0..=prefix_len {
                    let coefficient = if n == prefix_len {
                        Some(LogWeight {
                            sign: 1.0,
                            log: -(support[n] as f64).ln(),
                        })
                    } else {
                        differences[n].map(|log| LogWeight { sign: -1.0, log })
                    };
                    if let Some(coefficient) = coefficient {
                        sum.add(LogWeight {
                            sign: prefix.sign * coefficient.sign,
                            log: prefix.log + coefficient.log,
                        })?;
                    }
                    if n == prefix_len {
                        break;
                    }
                    let (axis, event) = conditioning[n];
                    let kernel = match axis {
                        Axis::Continuous { degree } => {
                            legendre(values[n], &mut row_basis[..degree + 1]);
                            2.0 * inner_product(
                                &row_basis[..degree + 1],
                                &event_basis[offsets[n]..][..degree + 1],
                            )
                        }
                        Axis::Categorical { classes }
                        | Axis::SmoothedCategorical { classes, .. } => {
                            classes as f64 * axis.categorical_inner(f64::from(values[n] == event))
                        }
                    };
                    if kernel == 0.0 {
                        break;
                    }
                    if !kernel.is_finite() {
                        return Err(Error::Numerical {
                            context: "aggregated conditioning kernel",
                        });
                    }
                    prefix.sign *= kernel.signum();
                    prefix.log += kernel.abs().ln();
                }
                let (weight, fraction) = sum.finish()?;
                minimum_row_fraction = minimum_row_fraction.min(fraction);
                visit(targets, weight)
            })?;
            if histogram != second_histogram {
                return Err(invalid(
                    "prefix source changed between counting and fitting scans",
                ));
            }
            Ok(())
        })?;
        Ok(Aggregated {
            distribution,
            evidence: Evidence {
                support,
                used_prefix,
                minimum_row_fraction,
                _reservation: reservation,
            },
        })
    }
    fn validate_prefix(
        &self,
        targets: &[f64],
        values: &[f64],
        conditioning: &[(Axis, f64)],
    ) -> Result<()> {
        if targets.len() != self.axes.len() || values.len() > conditioning.len() {
            return Err(invalid("prefix replicate dimensions"));
        }
        for (&axis, &value) in self.axes.iter().zip(targets) {
            axis.validate_value(value)?;
        }
        for (&(axis, _), &value) in conditioning.iter().zip(values) {
            axis.validate_value(value)?;
        }
        Ok(())
    }
}
struct LogSum {
    scale: f64,
    signed: Sum,
    absolute: Sum,
}
impl LogSum {
    fn new() -> Self {
        Self {
            scale: f64::NEG_INFINITY,
            signed: Sum::default(),
            absolute: Sum::default(),
        }
    }
    fn add(&mut self, term: LogWeight) -> Result<()> {
        if !term.log.is_finite() {
            return Err(Error::Numerical {
                context: "aggregated log weight",
            });
        }
        if term.log > self.scale {
            let factor = (self.scale - term.log).exp();
            self.signed.scale(factor);
            self.absolute.scale(factor);
            self.scale = term.log;
        }
        let value = (term.log - self.scale).exp();
        self.signed.add(term.sign * value);
        self.absolute.add(value);
        Ok(())
    }
    fn finish(self) -> Result<(Option<LogWeight>, f64)> {
        let value = self.signed.value();
        let absolute = self.absolute.value();
        if absolute == 0.0 {
            return Ok((None, 1.0));
        }
        if !value.is_finite() || !absolute.is_finite() {
            return Err(Error::Numerical {
                context: "aggregate signed row weight",
            });
        }
        let fraction = value.abs() / absolute;
        if value == 0.0 {
            Ok((None, 0.0))
        } else {
            Ok((
                Some(LogWeight {
                    sign: value.signum(),
                    log: self.scale + value.abs().ln(),
                }),
                fraction,
            ))
        }
    }
}
