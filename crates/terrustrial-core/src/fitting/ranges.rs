//! Bounded local range fitting by profiling the convex PSD sill subproblem.
//! Families and principal frames are supplied; parameters may be shared between
//! axes or kernels. This is not automatic family/orientation selection or a global
//! optimum guarantee. Sensitivity diagnostics are local, not confidence intervals.
use super::{fit_lmc_from, project_psd, CrossVariogramBin, LmcFit, LmcFitOptions, SpatialKernel};
use crate::{
    data::Point3,
    error::{checked_len, invalid},
    model::{Anisotropy, Coregionalization, Correlation, Lmc, Structure},
    Error, Result,
};
use faer::{Mat, Side};
use std::mem::size_of;

#[derive(Clone, Copy, Debug)]
pub struct Parameter {
    pub initial: f64,
    pub bounds: [f64; 2],
}
#[derive(Clone, Copy, Debug)]
pub enum Scale {
    Fixed(f64),
    /// Physical axis scale = multiplier * positive fitted parameter.
    Parameter {
        index: usize,
        multiplier: f64,
    },
}
#[derive(Clone, Debug)]
pub struct Kernel {
    pub correlation: Correlation,
    pub frame: [[f64; 3]; 3],
    pub scales: [Scale; 3],
}
impl Kernel {
    pub fn isotropic(correlation: Correlation, parameter: usize) -> Self {
        Self {
            correlation,
            frame: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            scales: [Scale::Parameter {
                index: parameter,
                multiplier: 1.0,
            }; 3],
        }
    }
    pub fn fixed(kernel: SpatialKernel) -> Self {
        Self {
            correlation: kernel.correlation,
            frame: kernel.anisotropy.frame(),
            scales: kernel.anisotropy.scales().map(Scale::Fixed),
        }
    }
    fn resolve(&self, parameters: &[f64]) -> Result<SpatialKernel> {
        let mut scales = [0.0; 3];
        for (out, scale) in scales.iter_mut().zip(self.scales) {
            *out = match scale {
                Scale::Fixed(value) => value,
                Scale::Parameter { index, multiplier } => {
                    parameters
                        .get(index)
                        .ok_or_else(|| invalid("range parameter index"))?
                        * multiplier
                }
            };
        }
        Ok(SpatialKernel {
            correlation: self.correlation,
            anisotropy: Anisotropy::new(self.frame, scales)?,
        })
    }
}
pub struct Options<'a> {
    /// Total profile evaluations, including up to two sensitivity probes per
    /// parameter. Must be at least 1 + 2 * number of fitted parameters.
    pub maximum_evaluations: usize,
    pub initial_log_step: f64,
    pub log_tolerance: f64,
    /// Required SSE improvement relative to normalized weighted data energy.
    pub improvement_tolerance: f64,
    pub diagnostic_log_step: f64,
    /// Relative singular-value threshold for the local profiled sensitivity rank.
    pub sensitivity_relative_tolerance: f64,
    /// Applied after normalizing all semivariances by their largest observed
    /// magnitude. Every inner solve must converge, or fitting returns an error.
    pub sill: LmcFitOptions<'a>,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stop {
    LogStepTolerance,
    EvaluationLimit,
    NoFreeParameters,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bound {
    Interior,
    Lower,
    Upper,
}
#[derive(Clone, Debug)]
pub struct Fit {
    /// Final sills and objective in the input's original semivariance scale.
    pub fit: LmcFit,
    pub parameters: Vec<f64>,
    pub bounds: Vec<Bound>,
    pub stop: Stop,
    pub evaluations: usize,
    pub total_sill_iterations: usize,
    pub final_log_step: f64,
    pub semivariance_scale: f64,
    pub normalized_weighted_sse: f64,
    /// Initial and accepted normalized objective values, in decreasing order.
    pub objective_history: Vec<f64>,
    /// Eigenvalues of J^T J, where J is the pair-weighted finite-difference
    /// sensitivity of profiled predictions to log parameters, divided by the
    /// square root of normalized data energy. Sills are refitted at every probe.
    /// A full rank does not establish global identification or calibrated errors.
    pub sensitivity_eigenvalues: Vec<f64>,
    pub sensitivity_rank: usize,
    pub parameter_sensitivity: Vec<f64>,
}
struct Profile<'a, 'b> {
    variables: usize,
    bins: &'a [CrossVariogramBin],
    kernels: &'a [Kernel],
    parameters: &'a [Parameter],
    options: &'a Options<'b>,
    evaluations: usize,
    iterations: usize,
}
impl Profile<'_, '_> {
    fn values(&self, logs: &[f64]) -> Vec<f64> {
        logs.iter()
            .zip(self.parameters)
            .map(|(x, p)| parameter_value(*x, p))
            .collect()
    }
    fn evaluate(&mut self, logs: &[f64], initial: Option<&Lmc>) -> Result<LmcFit> {
        self.options.sill.cancellation.check()?;
        if self.evaluations >= self.options.maximum_evaluations {
            return Err(invalid("range profile evaluation budget exhausted"));
        }
        let values = self.values(logs);
        let kernels = self
            .kernels
            .iter()
            .map(|k| k.resolve(&values))
            .collect::<Result<Vec<_>>>()?;
        let fit = fit_lmc_from(
            self.variables,
            self.bins,
            &kernels,
            LmcFitOptions {
                maximum_iterations: self.options.sill.maximum_iterations,
                tolerance: self.options.sill.tolerance,
                budget: self.options.sill.budget,
                cancellation: self.options.sill.cancellation,
            },
            initial,
        )?;
        self.evaluations += 1;
        self.iterations = self
            .iterations
            .checked_add(fit.iterations)
            .ok_or(Error::SizeOverflow)?;
        if !fit.converged {
            return Err(Error::Convergence {
                context: "range-profile PSD sill solve",
                iterations: fit.iterations,
                residual: fit.projected_gradient_max,
            });
        }
        Ok(fit)
    }
}
fn parameter_value(log: f64, parameter: &Parameter) -> f64 {
    if log == parameter.bounds[0].ln() {
        parameter.bounds[0]
    } else if log == parameter.bounds[1].ln() {
        parameter.bounds[1]
    } else {
        log.exp().clamp(parameter.bounds[0], parameter.bounds[1])
    }
}

/// Fit supplied range parameters jointly with admissible LMC sills. A fixed-order
/// coordinate pattern search operates in log space and halves its step after a
/// sweep without sufficient improvement. Warm starts reuse the previous feasible
/// sill model. All accepted objectives decrease. Stopping on the evaluation limit
/// returns the best converged sill model with an explicit incomplete range status.
///
/// Different initial points may reach different local solutions. Explicitly
/// compare starts/model families when required. Parameters unused by any axis, or
/// attached to a nugget kernel, are rejected. Frames, range bounds and positivity
/// must remain valid over the entire parameter box.
pub fn fit(
    variables: usize,
    bins: &[CrossVariogramBin],
    kernels: &[Kernel],
    parameters: &[Parameter],
    options: Options<'_>,
) -> Result<Fit> {
    options.sill.cancellation.check()?;
    let p = parameters.len();
    let probes = p.checked_mul(2).ok_or(Error::SizeOverflow)?;
    if bins.is_empty()
        || kernels.is_empty()
        || variables == 0
        || options.maximum_evaluations < probes.checked_add(1).ok_or(Error::SizeOverflow)?
        || !options.initial_log_step.is_finite()
        || options.initial_log_step <= 0.0
        || !options.log_tolerance.is_finite()
        || options.log_tolerance <= 0.0
        || options.log_tolerance >= options.initial_log_step
        || !options.improvement_tolerance.is_finite()
        || options.improvement_tolerance < 0.0
        || !options.diagnostic_log_step.is_finite()
        || options.diagnostic_log_step <= 0.0
        || !options.sensitivity_relative_tolerance.is_finite()
        || !(0.0..1.0).contains(&options.sensitivity_relative_tolerance)
        || options.sensitivity_relative_tolerance == 0.0
    {
        return Err(invalid(
            "range fitting dimensions, evaluation budget or tolerances",
        ));
    }
    let bytes = checked_len(&[bins.len(), size_of::<CrossVariogramBin>()])?
        .checked_add(checked_len(&[bins.len(), p, 8])?)
        .and_then(|x| x.checked_add(p.checked_mul(p)?.checked_mul(64)?))
        .and_then(|x| x.checked_add(bins.len().checked_mul(32)?))
        .and_then(|x| x.checked_add(p.checked_mul(128)?))
        .and_then(|x| x.checked_add(options.maximum_evaluations.checked_mul(8)?))
        .and_then(|x| {
            x.checked_add(
                variables
                    .checked_mul(variables)?
                    .checked_mul(kernels.len())?
                    .checked_mul(64)?,
            )
        })
        .and_then(|x| x.checked_add(kernels.len().checked_mul(size_of::<SpatialKernel>() * 3)?))
        .ok_or(Error::SizeOverflow)?;
    let _reservation = options
        .sill
        .budget
        .reserve(u64::try_from(bytes).map_err(|_| Error::SizeOverflow)?)?;
    let mut lower = Vec::with_capacity(p);
    let mut upper = Vec::with_capacity(p);
    let mut logs = Vec::with_capacity(p);
    for parameter in parameters {
        let [a, b] = parameter.bounds;
        if !a.is_finite()
            || !b.is_finite()
            || a <= 0.0
            || a >= b
            || !parameter.initial.is_finite()
            || parameter.initial < a
            || parameter.initial > b
            || a.ln() >= b.ln()
        {
            return Err(invalid(
                "positive representable range bounds and an in-bounds initial value required",
            ));
        }
        lower.push(a.ln());
        upper.push(b.ln());
        logs.push(parameter.initial.ln());
    }
    let mut used = vec![false; p];
    for kernel in kernels {
        for scale in kernel.scales {
            if let Scale::Parameter { index, multiplier } = scale {
                if index >= p
                    || !multiplier.is_finite()
                    || multiplier <= 0.0
                    || matches!(kernel.correlation, Correlation::Nugget)
                {
                    return Err(invalid(
                        "range parameter index/multiplier or nugget range parameter",
                    ));
                }
                used[index] = true;
            }
        }
        kernel.resolve(&parameters.iter().map(|p| p.bounds[0]).collect::<Vec<_>>())?;
        kernel.resolve(&parameters.iter().map(|p| p.bounds[1]).collect::<Vec<_>>())?;
    }
    if used.contains(&false) {
        return Err(invalid("unused range parameter"));
    }
    let maximum_count = bins.iter().map(|b| b.pairs).max().unwrap();
    if maximum_count == 0 {
        return Err(invalid("range fitting requires nonempty bins"));
    }
    let mut scale = 0.0_f64;
    for bin in bins {
        options.sill.cancellation.check()?;
        let [a, b] = bin.variables;
        if a >= variables
            || b >= variables
            || !bin.semivariance.is_finite()
            || (a == b && bin.semivariance < 0.0)
        {
            return Err(invalid("range fit variogram variable or semivariance"));
        }
        Point3::new(bin.lag)?;
        if bin.pairs > 0 {
            scale = scale.max(bin.semivariance.abs());
        }
    }
    if scale == 0.0 {
        scale = 1.0;
    }
    let normalized: Vec<_> = bins
        .iter()
        .map(|b| CrossVariogramBin {
            variables: b.variables,
            lag: b.lag,
            pairs: b.pairs,
            semivariance: if b.pairs == 0 {
                0.0
            } else {
                b.semivariance / scale
            },
        })
        .collect();
    let energy = normalized
        .iter()
        .map(|b| (b.pairs as f64 / maximum_count as f64) * b.semivariance.powi(2))
        .sum::<f64>();
    let mut profile = Profile {
        variables,
        bins: &normalized,
        kernels,
        parameters,
        options: &options,
        evaluations: 0,
        iterations: 0,
    };
    let mut best = profile.evaluate(&logs, None)?;
    let mut history = Vec::with_capacity(options.maximum_evaluations);
    history.push(best.weighted_sse);
    let mut trial = logs.clone();
    let mut step = options.initial_log_step;
    let optimization_limit = options.maximum_evaluations - probes;
    let mut stop = if p == 0 {
        Stop::NoFreeParameters
    } else {
        Stop::EvaluationLimit
    };
    if p > 0 {
        'search: while step > options.log_tolerance {
            let mut improved = false;
            for j in 0..p {
                let center = logs[j];
                for direction in [-1.0, 1.0] {
                    trial.copy_from_slice(&logs);
                    trial[j] = (center + direction * step).clamp(lower[j], upper[j]);
                    if trial[j] == logs[j] {
                        continue;
                    }
                    if profile.evaluations >= optimization_limit {
                        break 'search;
                    }
                    let candidate = profile.evaluate(&trial, Some(&best.model))?;
                    if candidate.weighted_sse
                        < best.weighted_sse - options.improvement_tolerance * energy
                    {
                        logs.copy_from_slice(&trial);
                        best = candidate;
                        improved = true;
                        history.push(best.weighted_sse);
                    }
                }
            }
            if !improved {
                step *= 0.5;
            }
        }
        if step <= options.log_tolerance {
            stop = Stop::LogStepTolerance;
        }
    }
    // Finite-difference the profiled predictions, not the raw fixed-sill model.
    // One-sided differences are used when the optimum touches a bound.
    let mut jacobian = Mat::<f64>::zeros(bins.len(), p);
    let mut left = vec![0.0; bins.len()];
    let mut right = left.clone();
    for j in 0..p {
        let a = (logs[j] - options.diagnostic_log_step).max(lower[j]);
        let b = (logs[j] + options.diagnostic_log_step).min(upper[j]);
        if a >= b || parameter_value(a, &parameters[j]) >= parameter_value(b, &parameters[j]) {
            return Err(invalid(
                "diagnostic log step is not representable at fitted range",
            ));
        }
        for (x, output) in [(a, &mut left), (b, &mut right)] {
            if x == logs[j] {
                predictions(&best.model, &normalized, output)?;
            } else {
                trial.copy_from_slice(&logs);
                trial[j] = x;
                let probe = profile.evaluate(&trial, Some(&best.model))?;
                predictions(&probe.model, &normalized, output)?;
            }
        }
        for i in 0..bins.len() {
            jacobian[(i, j)] = (right[i] - left[i]) / (b - a)
                * (bins[i].pairs as f64 / maximum_count as f64).sqrt()
                / energy.sqrt().max(f64::MIN_POSITIVE);
        }
    }
    crate::compute::linalg::finite(jacobian.as_ref(), "profiled range sensitivity")?;
    let gram = Mat::from_fn(p, p, |i, j| {
        (0..bins.len())
            .map(|k| jacobian[(k, i)] * jacobian[(k, j)])
            .sum::<f64>()
    });
    crate::compute::linalg::finite(gram.as_ref(), "range sensitivity Gram matrix")?;
    let parameter_sensitivity = (0..p).map(|i| gram[(i, i)].sqrt()).collect();
    let eigenvalues = if p == 0 {
        Vec::new()
    } else {
        gram.self_adjoint_eigenvalues(Side::Lower)
            .map_err(|_| Error::Numerical {
                context: "range sensitivity eigenvalues",
            })?
            .into_iter()
            .map(|v| v.max(0.0))
            .collect::<Vec<_>>()
    };
    let largest = eigenvalues.iter().copied().fold(0.0, f64::max);
    if eigenvalues.iter().any(|v| !v.is_finite()) {
        return Err(Error::Numerical {
            context: "range sensitivity eigenvalues",
        });
    }
    let rank = eigenvalues
        .iter()
        .filter(|&&v| v > largest * options.sensitivity_relative_tolerance.powi(2))
        .count();
    let bounds = logs
        .iter()
        .enumerate()
        .map(|(j, &x)| {
            if x == lower[j] {
                Bound::Lower
            } else if x == upper[j] {
                Bound::Upper
            } else {
                Bound::Interior
            }
        })
        .collect();
    let values = profile.values(&logs);
    let normalized_sse = best.weighted_sse;
    rescale(&mut best, scale)?;
    Ok(Fit {
        fit: best,
        parameters: values,
        bounds,
        stop,
        evaluations: profile.evaluations,
        total_sill_iterations: profile.iterations,
        final_log_step: step,
        semivariance_scale: scale,
        normalized_weighted_sse: normalized_sse,
        objective_history: history,
        sensitivity_eigenvalues: eigenvalues,
        sensitivity_rank: rank,
        parameter_sensitivity,
    })
}
fn predictions(model: &Lmc, bins: &[CrossVariogramBin], out: &mut [f64]) -> Result<()> {
    let origin = Point3::new([0.0; 3])?;
    for (bin, out) in bins.iter().zip(out) {
        let lag = Point3::new(bin.lag)?;
        let [a, b] = bin.variables;
        *out = 0.0;
        for structure in model.structures() {
            let feature = match structure.correlation {
                Correlation::Nugget => {
                    if origin == lag {
                        0.0
                    } else {
                        1.0
                    }
                }
                kernel => {
                    1.0 - kernel.evaluate(structure.anisotropy.normalized_distance(origin, lag))
                }
            };
            *out += structure.coregionalization.covariance(a, b)? * feature;
        }
        if !out.is_finite() {
            return Err(Error::Numerical {
                context: "profiled variogram prediction",
            });
        }
    }
    Ok(())
}
fn rescale(fit: &mut LmcFit, scale: f64) -> Result<()> {
    let mut structures = Vec::with_capacity(fit.model.structures().len());
    for structure in fit.model.structures() {
        let n = structure.coregionalization.variables();
        let matrix = Mat::from_fn(n, n, |i, j| {
            structure.coregionalization.covariance(i, j).unwrap()
        });
        let (_, factor) = project_psd(&matrix)?;
        let mut flat = Vec::with_capacity(checked_len(&[n, n])?);
        for i in 0..n {
            for j in 0..n {
                flat.push(factor[(i, j)] * scale.sqrt());
            }
        }
        structures.push(Structure {
            correlation: structure.correlation,
            anisotropy: structure.anisotropy.clone(),
            coregionalization: Coregionalization::from_factor(n, n, &flat)?,
        });
    }
    fit.model = Lmc::new(structures)?;
    fit.weighted_sse = (fit.weighted_sse.sqrt() * scale).powi(2);
    fit.projected_gradient_max *= scale;
    if !fit.weighted_sse.is_finite() || !fit.projected_gradient_max.is_finite() {
        return Err(Error::Numerical {
            context: "rescaled joint fit diagnostics",
        });
    }
    Ok(())
}
