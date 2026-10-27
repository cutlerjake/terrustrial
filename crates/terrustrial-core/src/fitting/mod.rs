//! Joint least-squares LMC fitting with positive-semidefinite sill constraints.
//! Projected gradient solves the convex PSD sill subproblem. The ranges module
//! profiles those sills within a bounded local search over supplied range parameters.
use crate::{
    data::Point3,
    error::{checked_len, invalid},
    execution::{Cancellation, MemoryBudget},
    model::{Coregionalization, Correlation, Lmc, SpatialKernel, Structure},
    Error, Result,
};
use faer::{Mat, Side};

pub mod ranges;

#[derive(Clone, Debug)]
pub struct CrossVariogramBin {
    /// Unordered variable pair. Either orientation is accepted.
    pub variables: [usize; 2],
    /// Representative displacement, in the coordinate units of the kernels.
    pub lag: [f64; 3],
    pub semivariance: f64,
    pub pairs: u64,
}
pub struct LmcFitOptions<'a> {
    pub maximum_iterations: usize,
    /// Relative projected-gradient stationarity tolerance.
    pub tolerance: f64,
    pub budget: &'a MemoryBudget,
    pub cancellation: &'a Cancellation,
}
#[derive(Clone, Debug)]
pub struct LmcFit {
    pub model: Lmc,
    pub iterations: usize,
    pub converged: bool,
    /// Sum of weighted squared residuals. Pair counts are divided by their maximum.
    pub weighted_sse: f64,
    pub projected_gradient_max: f64,
    /// No empirical information was supplied for these pairs. A feasible result
    /// does not imply identification, and no uncertainty interval is inferred.
    pub unobserved_pairs: Vec<[usize; 2]>,
    /// Kernels whose feature is zero at every positively weighted lag. Their
    /// sills are set to zero by convention and cannot be inferred from these bins.
    pub zero_feature_kernels: Vec<usize>,
}

/// Fit all auto/cross sills together. Each spatial kernel has its own PSD sill
/// matrix, so the resulting LMC is admissible even when empirical cross-variograms
/// violate covariance bounds. Repeated bins retain their individual weights.
/// Runtime O(iterations * (bins * kernels + kernels * variables^3)).
pub fn fit_lmc(
    variables: usize,
    bins: &[CrossVariogramBin],
    kernels: &[SpatialKernel],
    options: LmcFitOptions<'_>,
) -> Result<LmcFit> {
    fit_lmc_from(variables, bins, kernels, options, None)
}
fn fit_lmc_from(
    variables: usize,
    bins: &[CrossVariogramBin],
    kernels: &[SpatialKernel],
    options: LmcFitOptions<'_>,
    initial: Option<&Lmc>,
) -> Result<LmcFit> {
    if variables == 0
        || bins.is_empty()
        || kernels.is_empty()
        || options.maximum_iterations == 0
        || !options.tolerance.is_finite()
        || options.tolerance <= 0.0
    {
        return Err(invalid("joint fit dimensions or convergence options"));
    }
    options.cancellation.check()?;
    let matrix_size = checked_len(&[variables, variables])?;
    let matrices = checked_len(&[matrix_size, kernels.len(), 8, 8])?;
    let design = checked_len(&[bins.len(), kernels.len(), 8])?;
    let bytes = matrices
        .checked_add(design)
        .and_then(|v| v.checked_add(matrix_size.checked_mul(24)?))
        .ok_or(Error::SizeOverflow)?;
    let _reservation = options
        .budget
        .reserve(u64::try_from(bytes).map_err(|_| Error::SizeOverflow)?)?;
    let maximum_count = bins.iter().map(|b| b.pairs).max().unwrap();
    if maximum_count == 0 {
        return Err(invalid("joint fitting requires nonempty bins"));
    }
    let origin = Point3::new([0.0; 3])?;
    let mut features = Vec::with_capacity(checked_len(&[bins.len(), kernels.len()])?);
    let mut bounds = vec![0.0_f64; matrix_size];
    let mut observed = vec![false; matrix_size];
    let mut active = vec![false; kernels.len()];
    for bin in bins {
        options.cancellation.check()?;
        let [a, b] = bin.variables;
        if a >= variables
            || b >= variables
            || !bin.semivariance.is_finite()
            || (a == b && bin.semivariance < 0.0)
        {
            return Err(invalid("joint variogram bin variable or semivariance"));
        }
        let lag = Point3::new(bin.lag)?;
        let first = features.len();
        features.extend(kernels.iter().map(|k| match k.correlation {
            Correlation::Nugget => {
                if origin == lag {
                    0.0
                } else {
                    1.0
                }
            }
            kernel => 1.0 - kernel.evaluate(k.anisotropy.normalized_distance(origin, lag)),
        }));
        let feature = &features[first..];
        let weight = bin.pairs as f64 / maximum_count as f64;
        let index = a.min(b) * variables + a.max(b);
        bounds[index] += weight * feature.iter().map(|x| x * x).sum::<f64>();
        observed[index] |= bin.pairs > 0;
        for (active, feature) in active.iter_mut().zip(feature) {
            *active |= bin.pairs > 0 && *feature != 0.0;
        }
    }
    let lipschitz = bounds.into_iter().fold(0.0_f64, f64::max);
    if !lipschitz.is_finite() {
        return Err(Error::Numerical {
            context: "joint fitting step bound",
        });
    }
    for a in 0..variables {
        if !observed[a * variables + a] {
            return Err(invalid(
                "every variable needs at least one auto-variogram bin",
            ));
        }
    }
    // A zero design has zero gradient for every sill matrix. Choose zero sills
    // canonically, retaining the residual objective and explicit nonidentification.
    let step = if lipschitz == 0.0 {
        1.0
    } else {
        1.0 / lipschitz
    };
    let mut matrices: Vec<_> = if let Some(initial) = initial {
        initial
            .structures()
            .iter()
            .map(|structure| {
                Mat::from_fn(variables, variables, |i, j| {
                    structure.coregionalization.covariance(i, j).unwrap()
                })
            })
            .collect()
    } else {
        kernels
            .iter()
            .map(|_| Mat::<f64>::zeros(variables, variables))
            .collect()
    };
    for (matrix, active) in matrices.iter_mut().zip(&active) {
        if !active {
            matrix.as_mut().fill(0.0);
        }
    }
    let mut gradient: Vec<_> = kernels
        .iter()
        .map(|_| Mat::<f64>::zeros(variables, variables))
        .collect();
    let mut iterations = 0;
    let mut converged = false;
    let mut stationarity = f64::INFINITY;
    for iteration in 0..options.maximum_iterations {
        options.cancellation.check()?;
        for matrix in &mut gradient {
            matrix.as_mut().fill(0.0);
        }
        for (bin, feature) in bins.iter().zip(features.chunks_exact(kernels.len())) {
            let [a, b] = bin.variables;
            let weight = bin.pairs as f64 / maximum_count as f64;
            let prediction = matrices
                .iter()
                .zip(feature)
                .map(|(matrix, f)| matrix[(a, b)] * f)
                .sum::<f64>();
            // Off-diagonal derivatives are shared equally by the two symmetric
            // entries in the ambient Frobenius inner product.
            let residual =
                weight * (prediction - bin.semivariance) * if a == b { 1.0 } else { 0.5 };
            for (gradient, f) in gradient.iter_mut().zip(feature) {
                gradient[(a, b)] += residual * f;
                if a != b {
                    gradient[(b, a)] += residual * f;
                }
            }
        }
        let mut change = 0.0_f64;
        let mut magnitude = 0.0_f64;
        for (matrix, gradient) in matrices.iter_mut().zip(&gradient) {
            let trial = Mat::from_fn(variables, variables, |i, j| {
                matrix[(i, j)] - step * gradient[(i, j)]
            });
            let (next, _) = project_psd(&trial)?;
            for j in 0..variables {
                for i in 0..variables {
                    change = change.max((next[(i, j)] - matrix[(i, j)]).abs());
                    magnitude = magnitude.max(next[(i, j)].abs());
                }
            }
            *matrix = next;
        }
        iterations = iteration + 1;
        stationarity = change / step;
        if stationarity <= options.tolerance * (1.0 + magnitude) {
            converged = true;
            break;
        }
    }
    let mut structures = Vec::with_capacity(kernels.len());
    for (kernel, matrix) in kernels.iter().zip(&matrices) {
        let (_, factor) = project_psd(matrix)?;
        let factor_ref = &factor;
        let flat: Vec<_> = (0..variables)
            .flat_map(|i| (0..variables).map(move |j| factor_ref[(i, j)]))
            .collect();
        structures.push(Structure {
            correlation: kernel.correlation,
            anisotropy: kernel.anisotropy.clone(),
            coregionalization: Coregionalization::from_factor(variables, variables, &flat)?,
        });
    }
    let weighted_sse = bins
        .iter()
        .zip(features.chunks_exact(kernels.len()))
        .map(|(bin, features)| {
            let [a, b] = bin.variables;
            let fitted = matrices
                .iter()
                .zip(features)
                .map(|(matrix, f)| matrix[(a, b)] * f)
                .sum::<f64>();
            (bin.pairs as f64 / maximum_count as f64) * (fitted - bin.semivariance).powi(2)
        })
        .sum::<f64>();
    if !weighted_sse.is_finite() {
        return Err(Error::Numerical {
            context: "joint fitting objective",
        });
    }
    let mut unobserved_pairs = Vec::new();
    for a in 0..variables {
        for b in a..variables {
            if !observed[a * variables + b] {
                unobserved_pairs.push([a, b]);
            }
        }
    }
    Ok(LmcFit {
        model: Lmc::new(structures)?,
        iterations,
        converged,
        weighted_sse,
        projected_gradient_max: stationarity,
        unobserved_pairs,
        zero_feature_kernels: active
            .iter()
            .enumerate()
            .filter_map(|(i, &active)| (!active).then_some(i))
            .collect(),
    })
}
fn project_psd(matrix: &Mat<f64>) -> Result<(Mat<f64>, Mat<f64>)> {
    crate::compute::linalg::finite(matrix.as_ref(), "joint fitting iterate")?;
    let eigen = matrix
        .self_adjoint_eigen(Side::Lower)
        .map_err(|_| Error::Numerical {
            context: "PSD projection eigendecomposition",
        })?;
    let n = matrix.nrows();
    let factor = Mat::from_fn(n, n, |i, j| {
        eigen.U()[(i, j)] * eigen.S()[j].max(0.0).sqrt()
    });
    let projected = Mat::from_fn(n, n, |i, j| {
        (0..n).map(|k| factor[(i, k)] * factor[(j, k)]).sum::<f64>()
    });
    crate::compute::linalg::finite(projected.as_ref(), "joint fitting PSD projection")?;
    Ok((projected, factor))
}
