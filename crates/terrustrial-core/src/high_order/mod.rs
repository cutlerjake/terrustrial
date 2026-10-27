//! Statistical-learning-of-moments (SLM) conditional kernel embeddings, fitted to
//! positive joint mixtures. The continuous formulation follows Yao et al. (2020),
//! equations 11 and 13 through 22. Categorical kernels and factorized joint
//! prototypes are explicit extensions. The `aggregate` module adds a separately
//! documented coefficient-pooling variant for nested partial replicates. This is
//! not a generic KDE or a learned neural simulator.
//!
//! Continuous data must already be transformed to [-1,1]. Replicates must have the
//! same spatial geometry as the conditioning event. Their selection, stationarity,
//! missing-data policy and sufficient coverage are the caller's responsibility.
//! The complete-row estimator does not drop conditioning coordinates. The prefix
//! estimator exposes evidence counts and an explicit unsupported-interaction policy.

pub mod aggregate;
pub mod grid;
mod prototype;
mod solver;

use crate::{
    error::{checked_len, invalid},
    execution::{random_stream, Cancellation, MemoryBudget, Reservation},
    Error, Result,
};
use prototype::legendre;
use rand::RngExt;
use std::mem::size_of;

/// Kernel for an attribute at one location. Repeated attributes at different
/// locations occupy separate axes; order means Legendre degree, not neighborhood size.
#[derive(Clone, Copy, Debug)]
pub enum Axis {
    Continuous {
        degree: usize,
    },
    Categorical {
        classes: u32,
    },
    /// K(a,b) = (1-uniform_mass) 1[a=b] + uniform_mass/classes.
    /// This positive-semidefinite kernel is strictly positive for nonzero mass,
    /// so unseen categorical events can use mismatching complete replicates.
    /// The mass is an explicit modeling parameter in [0,1), not an automatic
    /// fallback. Zero is identical to `Categorical`. If used in a plan, smoothing
    /// applies to the target objective as well as grid conditioning. Raw density
    /// evaluation/draws still use the fitted prototype distributions.
    SmoothedCategorical {
        classes: u32,
        uniform_mass: f64,
    },
}
impl Axis {
    fn width(self) -> Result<usize> {
        match self {
            Self::Continuous { degree } => degree.checked_add(1).ok_or(Error::SizeOverflow),
            _ => {
                let (classes, mass) = self.categorical().unwrap();
                if classes == 0
                    || !mass.is_finite()
                    || !(0.0..1.0).contains(&mass)
                    || (mass > 0.0 && mass / classes as f64 == 0.0)
                {
                    return Err(invalid(
                        "categorical axis needs classes and a representable uniform mass in [0,1)",
                    ));
                }
                Ok(0)
            }
        }
    }
    pub(crate) fn categorical(self) -> Option<(u32, f64)> {
        match self {
            Self::Continuous { .. } => None,
            Self::Categorical { classes } => Some((classes, 0.0)),
            Self::SmoothedCategorical {
                classes,
                uniform_mass,
            } => Some((classes, uniform_mass)),
        }
    }
    // Inner product of categorical probability vectors, or of a vector and a
    // point mass. Preserves the original arithmetic exactly when mass is zero.
    fn categorical_inner(self, overlap: f64) -> f64 {
        let (classes, mass) = self.categorical().unwrap();
        if mass == 0.0 {
            overlap
        } else {
            (1.0 - mass) * overlap + mass / classes as f64
        }
    }
    pub(crate) fn validate_value(self, value: f64) -> Result<()> {
        let valid = value.is_finite()
            && match self {
                Self::Continuous { .. } => (-1.0..=1.0).contains(&value),
                Self::Categorical { classes } | Self::SmoothedCategorical { classes, .. } => {
                    value >= 0.0 && value < classes as f64 && value.fract() == 0.0
                }
            };
        if valid {
            Ok(())
        } else {
            Err(invalid("high-order value outside its axis domain"))
        }
    }
}

/// A normalized univariate probability density or mass function.
#[derive(Clone, Debug)]
pub enum Marginal {
    /// A normal distribution truncated to [-1,1]. The mean is the untruncated
    /// location, also restricted to [-1,1]. No probability atoms occur at endpoints.
    TruncatedNormal { mean: f64, deviation: f64 },
    /// One probability per integer class label. Accepted roundoff in the sum
    /// (up to 1e-12) is explicitly normalized when the plan is built.
    Categorical(Vec<f64>),
}

/// A component is a product distribution. Drawing ONE shared component for the
/// entire target vector retains joint dependence between its attributes.
#[derive(Clone, Debug)]
pub struct Prototype(pub Vec<Marginal>);

/// The visitor receives complete target and conditioning vectors.
pub type ReplicateVisitor<'a> = dyn FnMut(&[f64], &[f64]) -> Result<()> + 'a;

/// Stream arbitrarily many complete rows without retaining the dataset in the
/// estimator. Visit order must be stable for run reproducibility. The visitor can
/// return cancellation, budget or numerical errors, which implementations must
/// propagate immediately. Every row has [target values] and [conditioning values]
/// in the exact axis order of the plan and query respectively.
pub trait ReplicateSource {
    fn scan(&self, visitor: &mut ReplicateVisitor<'_>) -> Result<()>;
}

/// Convenience source for caller-owned rows. A file/store implementation can
/// reuse its own row buffers instead; the estimator does not retain row borrows.
impl ReplicateSource for [(Vec<f64>, Vec<f64>)] {
    fn scan(&self, visitor: &mut ReplicateVisitor<'_>) -> Result<()> {
        for (target, conditioning) in self {
            visitor(target, conditioning)?;
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
pub struct FitOptions<'a> {
    pub maximum_iterations: usize,
    /// Relative simplex optimality gap, scaled by max(|Q|, |q|).
    pub tolerance: f64,
    /// Minimum positive signed sum / sum of absolute conditioning kernel weights.
    /// Controls cancellation of signed kernel weights, not spatial evidence coverage.
    /// Must lie in (0,1]. A value such as 1e-10 rejects near-zero normalizers.
    pub minimum_normalizer_fraction: f64,
    pub budget: &'a MemoryBudget,
    pub cancellation: &'a Cancellation,
}

impl FitOptions<'_> {
    pub(crate) fn validate(&self) -> Result<()> {
        self.cancellation.check()?;
        if self.maximum_iterations == 0
            || !self.tolerance.is_finite()
            || self.tolerance <= 0.0
            || !self.minimum_normalizer_fraction.is_finite()
            || self.minimum_normalizer_fraction <= 0.0
            || self.minimum_normalizer_fraction > 1.0
        {
            return Err(invalid("high-order convergence or normalizer options"));
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct Diagnostics {
    pub rows: u64,
    pub nonzero_kernel_rows: u64,
    pub normalizer_fraction: f64,
    pub negative_weight_fraction: f64,
    /// Log of the positive unnormalized conditioning mass. For complete rows,
    /// this is the SUM of raw kernels, not a log likelihood. For prefix aggregation
    /// it is an estimated density relative to the uniform conditioning measure.
    pub log_normalizer: f64,
    pub iterations: usize,
    pub relative_optimality_gap: f64,
}

/// Reusable prototype moments and positive-semidefinite product-kernel Gram
/// matrix. Storage O(P² + P sum(degree+1)), independent of the replicate count.
/// Dense prototype interaction is deliberate: do not use one prototype per row
/// for a large dataset. Prototype selection/bandwidth fitting are not automatic.
#[derive(Debug)]
pub struct Plan {
    axes: Vec<Axis>,
    prototypes: Vec<Prototype>,
    offsets: Vec<usize>,
    width: usize,
    moments: Vec<f64>,
    gram: Vec<f64>,
    _reservation: Reservation,
}
impl Plan {
    /// Minimize 0.5 alpha' (Q + regularization I) alpha - q' alpha on the simplex.
    /// Regularization is nonnegative and uses the raw, unscaled published kernel.
    /// Quadrature computes prototype moments to successive-rule tolerance 2e-13;
    /// inability to reach it is an error, never silently accepted.
    pub fn new(
        axes: &[Axis],
        mut prototypes: Vec<Prototype>,
        regularization: f64,
        budget: &MemoryBudget,
        cancellation: &Cancellation,
    ) -> Result<Self> {
        cancellation.check()?;
        if axes.is_empty()
            || prototypes.is_empty()
            || !regularization.is_finite()
            || regularization < 0.0
        {
            return Err(invalid("high-order plan dimensions or regularization"));
        }
        let mut width = 0usize;
        let mut max_width = 0;
        for &axis in axes {
            let n = axis.width()?;
            width = width.checked_add(n).ok_or(Error::SizeOverflow)?;
            max_width = max_width.max(n);
        }
        let p = prototypes.len();
        let mut bytes = checked_len(&[p, p, 8])?
            .checked_add(checked_len(&[p, width, 8])?)
            .and_then(|v| {
                v.checked_add(
                    axes.len()
                        .checked_mul(size_of::<Axis>() + size_of::<usize>())?,
                )
            })
            .and_then(|v| v.checked_add(p.checked_mul(size_of::<Prototype>())?))
            .ok_or(Error::SizeOverflow)?;
        for prototype in &prototypes {
            if prototype.0.len() != axes.len() {
                return Err(invalid("prototype dimension"));
            }
            bytes = bytes
                .checked_add(checked_len(&[prototype.0.len(), size_of::<Marginal>()])?)
                .ok_or(Error::SizeOverflow)?;
            for (marginal, &axis) in prototype.0.iter().zip(axes) {
                marginal.validate(axis)?;
                if let Marginal::Categorical(probabilities) = marginal {
                    bytes = bytes
                        .checked_add(checked_len(&[probabilities.len(), 8])?)
                        .ok_or(Error::SizeOverflow)?;
                }
            }
        }
        let reservation = budget.reserve(u64::try_from(bytes).map_err(|_| Error::SizeOverflow)?)?;
        let _workspace = budget.reserve(
            u64::try_from(checked_len(&[max_width, 8, 4])?).map_err(|_| Error::SizeOverflow)?,
        )?;
        let mut offsets = Vec::with_capacity(axes.len());
        let mut offset = 0;
        for &axis in axes {
            offsets.push(offset);
            offset += axis.width()?;
        }
        let mut moments = Vec::with_capacity(checked_len(&[p, width])?);
        for prototype in &mut prototypes {
            cancellation.check()?;
            for (marginal, axis) in prototype.0.iter_mut().zip(axes) {
                marginal.normalize();
                if let Axis::Continuous { degree } = *axis {
                    moments.extend(marginal.moments(degree, cancellation)?);
                }
            }
        }
        let mut gram = vec![0.0; checked_len(&[p, p])?];
        for i in 0..p {
            cancellation.check()?;
            for j in 0..=i {
                let mut product = 1.0;
                for (d, axis) in axes.iter().enumerate() {
                    let inner = match *axis {
                        Axis::Continuous { degree } => {
                            let a = &moments[i * width + offsets[d]..][..degree + 1];
                            let b = &moments[j * width + offsets[d]..][..degree + 1];
                            inner_product(a, b)
                        }
                        Axis::Categorical { .. } | Axis::SmoothedCategorical { .. } => {
                            let (Marginal::Categorical(a), Marginal::Categorical(b)) =
                                (&prototypes[i].0[d], &prototypes[j].0[d])
                            else {
                                unreachable!()
                            };
                            axis.categorical_inner(a.iter().zip(b).map(|(a, b)| a * b).sum())
                        }
                    };
                    product = checked_product(product, inner)?;
                }
                let entry = product + if i == j { regularization } else { 0.0 };
                if !entry.is_finite() {
                    return Err(Error::Numerical {
                        context: "high-order Gram matrix",
                    });
                }
                gram[i * p + j] = entry;
                gram[j * p + i] = entry;
            }
        }
        Ok(Self {
            axes: axes.to_vec(),
            prototypes,
            offsets,
            width,
            moments,
            gram,
            _reservation: reservation,
        })
    }

    pub fn axes(&self) -> &[Axis] {
        &self.axes
    }
    pub fn prototypes(&self) -> &[Prototype] {
        &self.prototypes
    }

    /// Fit a joint conditional probability from a single streaming pass through
    /// geometrically aligned replicates. Signed weights are accumulated with
    /// dynamic log scaling and compensated sums. Nonpositive or ill-conditioned
    /// normalizers fail explicitly. No nearest-pattern shortlist or clipping is used.
    pub fn condition<S: ReplicateSource + ?Sized>(
        &self,
        source: &S,
        conditioning: &[(Axis, f64)],
        options: FitOptions<'_>,
    ) -> Result<Conditional<'_>> {
        options.validate()?;
        let mut condition_width = 0usize;
        let mut maximum_width = 0;
        for &(axis, value) in conditioning {
            axis.validate_value(value)?;
            let n = axis.width()?;
            condition_width = condition_width.checked_add(n).ok_or(Error::SizeOverflow)?;
            maximum_width = maximum_width.max(n);
        }
        let bytes = checked_len(&[
            condition_width
                .checked_add(maximum_width)
                .ok_or(Error::SizeOverflow)?,
            8,
        ])?
        .checked_add(checked_len(&[
            conditioning.len(),
            size_of::<(usize, f64, f64)>(),
        ])?)
        .ok_or(Error::SizeOverflow)?;
        let _workspace = options
            .budget
            .reserve(u64::try_from(bytes).map_err(|_| Error::SizeOverflow)?)?;
        let mut event_basis = vec![0.0; condition_width];
        // A categorical axis has only two kernel values. Cache their logs once
        // per query rather than evaluating logarithms for every TI replicate.
        let mut kernels = Vec::with_capacity(conditioning.len());
        let mut index = 0;
        for &(axis, value) in conditioning {
            let n = axis.width()?;
            if n > 0 {
                legendre(value, &mut event_basis[index..index + n]);
                kernels.push((n, 0.0, 0.0));
            } else {
                kernels.push((
                    0,
                    axis.categorical_inner(1.0).ln(),
                    axis.categorical_inner(0.0).ln(),
                ));
            }
            index += n;
        }
        let mut row_basis = vec![0.0; maximum_width];
        self.condition_weighted(options, |visit| {
            source.scan(&mut |targets, values| {
                options.cancellation.check()?;
                if values.len() != conditioning.len() {
                    return Err(invalid("replicate dimension"));
                }
                for (&(axis, _), &value) in conditioning.iter().zip(values) {
                    axis.validate_value(value)?;
                }
                // A zero categorical factor annihilates the full conditioning
                // kernel. Reject it before computing any continuous polynomial.
                if conditioning
                    .iter()
                    .zip(values)
                    .any(|(&(axis, event), &value)| {
                        axis.categorical().is_some_and(|(_, mass)| mass == 0.0) && value != event
                    })
                {
                    return visit(targets, None);
                }
                let (mut log_weight, mut sign, mut offset) = (0.0, 1.0, 0);
                for ((&(_, event), &value), &(n, same_log, different_log)) in
                    conditioning.iter().zip(values).zip(&kernels)
                {
                    if n == 0 {
                        let log = if value == event {
                            same_log
                        } else {
                            different_log
                        };
                        if log == f64::NEG_INFINITY {
                            return visit(targets, None);
                        }
                        log_weight += log;
                        continue;
                    }
                    legendre(value, &mut row_basis[..n]);
                    let kernel = inner_product(&row_basis[..n], &event_basis[offset..offset + n]);
                    if kernel == 0.0 {
                        return visit(targets, None);
                    }
                    if !kernel.is_finite() {
                        return Err(Error::Numerical {
                            context: "conditioning kernel",
                        });
                    }
                    sign *= kernel.signum();
                    log_weight += kernel.abs().ln();
                    offset += n;
                }
                if !log_weight.is_finite() {
                    return Err(Error::Numerical {
                        context: "conditioning log kernel",
                    });
                }
                visit(
                    targets,
                    Some(LogWeight {
                        sign,
                        log: log_weight,
                    }),
                )
            })
        })
    }

    fn condition_weighted(
        &self,
        options: FitOptions<'_>,
        scan: impl FnOnce(&mut WeightedVisitor<'_>) -> Result<()>,
    ) -> Result<Conditional<'_>> {
        options.validate()?;
        let p = self.prototypes.len();
        let bytes = self
            .width
            .checked_add(p.checked_mul(12).ok_or(Error::SizeOverflow)?)
            .and_then(|v| v.checked_mul(8))
            .ok_or(Error::SizeOverflow)?;
        let _workspace = options
            .budget
            .reserve(u64::try_from(bytes).map_err(|_| Error::SizeOverflow)?)?;
        let reservation = options
            .budget
            .reserve(u64::try_from(checked_len(&[p, 8])?).map_err(|_| Error::SizeOverflow)?)?;
        let mut target_basis = vec![0.0; self.width];
        let mut normalizer = Sum::default();
        let mut absolute = Sum::default();
        let mut negative = Sum::default();
        let mut q = vec![Sum::default(); p];
        let mut log_scale = f64::NEG_INFINITY;
        let mut rows = 0u64;
        let mut contributing = 0u64;
        scan(&mut |targets, weight| {
            options.cancellation.check()?;
            if targets.len() != self.axes.len() {
                return Err(invalid("replicate target dimension"));
            }
            for (&axis, &value) in self.axes.iter().zip(targets) {
                axis.validate_value(value)?;
            }
            rows = rows.checked_add(1).ok_or(Error::SizeOverflow)?;
            let Some(LogWeight {
                sign,
                log: log_weight,
            }) = weight
            else {
                return Ok(());
            };
            if !log_weight.is_finite() || !matches!(sign, -1.0 | 1.0) {
                return Err(Error::Numerical {
                    context: "signed replicate log weight",
                });
            }
            contributing = contributing.checked_add(1).ok_or(Error::SizeOverflow)?;
            if log_weight > log_scale {
                let factor = (log_scale - log_weight).exp();
                normalizer.scale(factor);
                absolute.scale(factor);
                negative.scale(factor);
                q.iter_mut().for_each(|x| x.scale(factor));
                log_scale = log_weight;
            }
            let weight = sign * (log_weight - log_scale).exp();
            normalizer.add(weight);
            absolute.add(weight.abs());
            if weight < 0.0 {
                negative.add(-weight);
            }
            for (d, (&axis, &value)) in self.axes.iter().zip(targets).enumerate() {
                if let Axis::Continuous { degree } = axis {
                    legendre(value, &mut target_basis[self.offsets[d]..][..degree + 1]);
                }
            }
            for (i, sum) in q.iter_mut().enumerate() {
                // Structural zeros are common in mixed categorical prototypes.
                // They can be eliminated before continuous contraction without
                // changing floating-point summation order for nonzero terms.
                if self.prototypes[i].0.iter().zip(targets).zip(&self.axes).any(|((marginal, value), axis)| {
                    axis.categorical().is_some_and(|(_, mass)| mass == 0.0)
                        && matches!(marginal, Marginal::Categorical(p) if p[*value as usize] == 0.0)
                }) { continue; }
                let mut product = 1.0;
                for (d, axis) in self.axes.iter().enumerate() {
                    let inner = match *axis {
                        Axis::Continuous { degree } => inner_product(
                            &target_basis[self.offsets[d]..][..degree + 1],
                            &self.moments[i * self.width + self.offsets[d]..][..degree + 1],
                        ),
                        Axis::Categorical { .. } | Axis::SmoothedCategorical { .. } => {
                            let Marginal::Categorical(probabilities) = &self.prototypes[i].0[d]
                            else {
                                unreachable!()
                            };
                            axis.categorical_inner(probabilities[targets[d] as usize])
                        }
                    };
                    product = checked_product(product, inner)?;
                }
                sum.add(weight * product);
            }
            Ok(())
        })?;
        let denominator = normalizer.value();
        let normalizer_fraction = denominator / absolute.value();
        if rows == 0
            || !normalizer_fraction.is_finite()
            || denominator <= 0.0
            || normalizer_fraction < options.minimum_normalizer_fraction
        {
            return Err(Error::Numerical {
                context: "nonpositive or cancellation-dominated high-order conditioning normalizer",
            });
        }
        let q: Vec<f64> = q.iter().map(|sum| sum.value() / denominator).collect();
        if q.iter().any(|x| !x.is_finite()) {
            return Err(Error::Numerical {
                context: "high-order conditional moments",
            });
        }
        let (weights, iterations, gap) = solver::solve(&self.gram, &q, &options)?;
        Ok(Conditional {
            plan: self,
            weights,
            diagnostics: Diagnostics {
                rows,
                nonzero_kernel_rows: contributing,
                normalizer_fraction,
                negative_weight_fraction: negative.value() / absolute.value(),
                log_normalizer: log_scale + denominator.ln(),
                iterations,
                relative_optimality_gap: gap,
            },
            _reservation: reservation,
        })
    }
}

/// Nonnegative joint mixture. Density is with respect to Lebesgue measure on
/// continuous axes and counting measure on categorical axes.
#[derive(Debug)]
pub struct Conditional<'a> {
    plan: &'a Plan,
    weights: Vec<f64>,
    pub diagnostics: Diagnostics,
    _reservation: Reservation,
}
impl<'a> Conditional<'a> {
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// Condition the fitted JOINT mixture on exact known target attributes.
    /// Component weights are reweighted by their marginal likelihoods, using
    /// log arithmetic. This is distinct from smoothing a neighboring event.
    /// Unknown marginals in each product prototype are unchanged, while one
    /// shared posterior component retains their joint dependence.
    ///
    /// Continuous observations use density, categorical observations use mass.
    /// Zero evidence is an explicit error. This is exact conditioning of the
    /// fitted mixture, not a claim that the fitted mixture is the true CPDF.
    pub fn condition_targets(
        &self,
        known: &[Option<f64>],
        budget: &MemoryBudget,
        cancellation: &Cancellation,
    ) -> Result<TargetConditional<'a>> {
        cancellation.check()?;
        if known.len() != self.plan.axes.len() {
            return Err(invalid("known target dimension"));
        }
        for (&axis, &value) in self.plan.axes.iter().zip(known) {
            if let Some(value) = value {
                axis.validate_value(value)?;
            }
        }
        let bytes = checked_len(&[self.weights.len(), 8])?
            .checked_add(checked_len(&[known.len(), size_of::<Option<f64>>()])?)
            .ok_or(Error::SizeOverflow)?;
        let reservation = budget.reserve(u64::try_from(bytes).map_err(|_| Error::SizeOverflow)?)?;
        if known.iter().all(Option::is_none) {
            return Ok(TargetConditional {
                plan: self.plan,
                weights: self.weights.clone(),
                known: known.to_vec(),
                log_evidence: 0.0,
                _reservation: reservation,
            });
        }
        let mut weights = Vec::with_capacity(self.weights.len());
        let mut maximum = f64::NEG_INFINITY;
        for (prototype, &weight) in self.plan.prototypes.iter().zip(&self.weights) {
            cancellation.check()?;
            let mut log = weight.ln();
            for (marginal, value) in prototype.0.iter().zip(known) {
                if log == f64::NEG_INFINITY {
                    break;
                }
                if let Some(value) = value {
                    log += marginal.log_density(*value);
                }
            }
            if log.is_nan() || log == f64::INFINITY {
                return Err(Error::Numerical {
                    context: "known target log likelihood",
                });
            }
            maximum = maximum.max(log);
            weights.push(log);
        }
        if !maximum.is_finite() {
            return Err(Error::Numerical {
                context: "known targets have zero fitted probability or density",
            });
        }
        let mut sum = Sum::default();
        for weight in &mut weights {
            *weight = (*weight - maximum).exp();
            sum.add(*weight);
        }
        let total = sum.value();
        for weight in &mut weights {
            *weight /= total;
        }
        Ok(TargetConditional {
            plan: self.plan,
            weights,
            known: known.to_vec(),
            log_evidence: maximum + total.ln(),
            _reservation: reservation,
        })
    }

    pub fn log_probability(&self, values: &[f64]) -> Result<f64> {
        if values.len() != self.plan.axes.len() {
            return Err(invalid("joint probability dimension"));
        }
        for (&axis, &value) in self.plan.axes.iter().zip(values) {
            match axis {
                Axis::Continuous { .. } if value.is_finite() => {}
                _ => axis.validate_value(value)?,
            }
        }
        let (mut maximum, mut sum) = (f64::NEG_INFINITY, 0.0);
        for (prototype, weight) in self.plan.prototypes.iter().zip(&self.weights) {
            if *weight == 0.0 {
                continue;
            }
            let log = weight.ln()
                + prototype
                    .0
                    .iter()
                    .zip(values)
                    .map(|(p, x)| p.log_density(*x))
                    .sum::<f64>();
            if log == f64::NEG_INFINITY {
                continue;
            }
            if !log.is_finite() {
                return Err(Error::Numerical {
                    context: "high-order joint log density",
                });
            }
            if log > maximum {
                sum *= (maximum - log).exp();
                maximum = log;
            }
            sum += (log - maximum).exp();
        }
        Ok(maximum + sum.ln())
    }

    /// One jointly sampled vector, keyed by logical query/sample identity. Purpose
    /// 10 is reserved for this engine. Repeated identical identifiers repeat draws.
    /// An error can leave a partially written output slice; the grid simulator
    /// commits values only after the entire draw succeeds.
    pub fn draw_into(
        &self,
        seed: u64,
        realization: u64,
        item: u64,
        values: &mut [f64],
        cancellation: &Cancellation,
    ) -> Result<()> {
        sample_mixture(
            self.plan,
            &self.weights,
            &[],
            [seed, realization, item],
            values,
            cancellation,
        )
    }
}

/// Exact target conditioning of a fitted mixture, with bit-preserved known
/// values. Its support includes Dirac constraints on known continuous axes;
/// `log_evidence` is their likelihood under the original fitted distribution.
#[derive(Debug)]
pub struct TargetConditional<'a> {
    plan: &'a Plan,
    weights: Vec<f64>,
    known: Vec<Option<f64>>,
    pub log_evidence: f64,
    _reservation: Reservation,
}
impl TargetConditional<'_> {
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }
    /// Draw only unknown coordinates, copying known coordinates bit-for-bit.
    /// Uses the same logical random-stream keys as `Conditional::draw_into`.
    pub fn draw_into(
        &self,
        seed: u64,
        realization: u64,
        item: u64,
        values: &mut [f64],
        cancellation: &Cancellation,
    ) -> Result<()> {
        sample_mixture(
            self.plan,
            &self.weights,
            &self.known,
            [seed, realization, item],
            values,
            cancellation,
        )
    }
}
fn sample_mixture(
    plan: &Plan,
    weights: &[f64],
    known: &[Option<f64>],
    key: [u64; 3],
    values: &mut [f64],
    cancellation: &Cancellation,
) -> Result<()> {
    cancellation.check()?;
    if values.len() != plan.axes.len() {
        return Err(invalid("joint draw dimension"));
    }
    let mut rng = random_stream(key[0], key[1], key[2], 10);
    let u: f64 = rng.random();
    let (mut cdf, mut selected) = (0.0, 0);
    for (i, &weight) in weights.iter().enumerate() {
        if weight > 0.0 {
            selected = i;
        }
        cdf += weight;
        if u < cdf {
            break;
        }
    }
    for (i, (out, marginal)) in values
        .iter_mut()
        .zip(&plan.prototypes[selected].0)
        .enumerate()
    {
        *out = if let Some(value) = known.get(i).copied().flatten() {
            value
        } else {
            marginal.sample(&mut rng, cancellation)?
        };
    }
    Ok(())
}

fn inner_product(a: &[f64], b: &[f64]) -> f64 {
    let mut sum = Sum::default();
    for (w, (a, b)) in a.iter().zip(b).enumerate() {
        sum.add((w as f64 + 0.5) * a * b);
    }
    sum.value()
}
fn checked_product(a: f64, b: f64) -> Result<f64> {
    let product = a * b;
    if !product.is_finite() || (product == 0.0 && a != 0.0 && b != 0.0) {
        return Err(Error::Numerical {
            context: "high-order product kernel overflow or underflow",
        });
    }
    Ok(product)
}
#[derive(Clone, Copy, Debug)]
struct LogWeight {
    sign: f64,
    log: f64,
}
type WeightedVisitor<'a> = dyn FnMut(&[f64], Option<LogWeight>) -> Result<()> + 'a;

#[derive(Clone, Copy, Default)]
struct Sum {
    total: f64,
    correction: f64,
}
impl Sum {
    fn add(&mut self, x: f64) {
        let next = self.total + x;
        self.correction += if self.total.abs() >= x.abs() {
            (self.total - next) + x
        } else {
            (x - next) + self.total
        };
        self.total = next;
    }
    fn scale(&mut self, factor: f64) {
        self.total *= factor;
        self.correction *= factor;
    }
    fn value(self) -> f64 {
        self.total + self.correction
    }
}
