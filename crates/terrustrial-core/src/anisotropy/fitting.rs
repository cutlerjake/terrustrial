//! Directional Gaussian metric fitting for a local window with known sill/nugget.
//! This fits transformed directional evidence, not an arbitrary covariance family.
use super::Tensor;
use crate::{
    compute::linalg::{factor_spd, solve_spd},
    data::norm3,
    error::{checked_len, invalid},
    execution::{Cancellation, MemoryBudget},
    model::Anisotropy,
    Error, Result,
};
use faer::{Mat, Side};
#[derive(Clone, Debug)]
pub struct DirectionalBin {
    pub lag: [f64; 3],
    pub semivariance: f64,
    pub pairs: u64,
}
pub struct TensorFitOptions<'a> {
    pub structured_sill: f64,
    pub nugget: f64,
    pub range_bounds: [f64; 2],
    pub maximum_iterations: usize,
    pub tolerance: f64,
    pub budget: &'a MemoryBudget,
    pub cancellation: &'a Cancellation,
}
#[derive(Clone, Debug)]
pub struct TensorFit {
    pub anisotropy: Anisotropy,
    pub tensor: Tensor,
    pub converged: bool,
    pub iterations: usize,
    pub weighted_transformed_sse: f64,
    pub design_condition_number: f64,
    pub used_bins: usize,
    pub excluded_bins: usize,
    pub axes_at_bound: usize,
}
/// gamma(h)=nugget+sill*(1-exp(-h^T M h)). Fit -ln(1-(gamma-nugget)/sill)
/// against h^T M h with pair-count weights and bounded SPD eigenvalues. Range
/// constraints are explicit. Bins at/beyond sill or below nugget are excluded and
/// counted. Six identifiable directions are required; planar evidence does not
/// identify a full 3D tensor without additional constraints.
///
/// This single-window function does not select windows or regularization. Use
/// `field::fit_family` for a coupled field, `evidence::build` for grouped spatial
/// windows, and `selection::select_families_by` for spatial prediction selection.
pub fn fit_gaussian_tensor(
    bins: &[DirectionalBin],
    options: TensorFitOptions<'_>,
) -> Result<TensorFit> {
    let [minimum, maximum] = options.range_bounds;
    if !options.structured_sill.is_finite()
        || options.structured_sill <= 0.0
        || !options.nugget.is_finite()
        || options.nugget < 0.0
        || !minimum.is_finite()
        || minimum <= 0.0
        || !maximum.is_finite()
        || maximum <= minimum
        || options.maximum_iterations == 0
        || !options.tolerance.is_finite()
        || options.tolerance <= 0.0
    {
        return Err(invalid("tensor fitting parameters"));
    }
    options.cancellation.check()?;
    let bytes = checked_len(&[bins.len(), 64])?
        .checked_add(16384)
        .ok_or(Error::SizeOverflow)?;
    let _reservation = options
        .budget
        .reserve(u64::try_from(bytes).map_err(|_| Error::SizeOverflow)?)?;
    let mut data = Vec::new();
    let mut coordinate_scale = 0.0_f64;
    let mut maximum_pairs = 0_u64;
    let mut excluded = 0;
    for bin in bins {
        if bin.lag.iter().any(|v| !v.is_finite())
            || !bin.semivariance.is_finite()
            || bin.semivariance < 0.0
        {
            return Err(invalid("directional tensor evidence"));
        }
        let length = norm3(bin.lag);
        let fraction = (bin.semivariance - options.nugget) / options.structured_sill;
        if bin.pairs == 0 || length == 0.0 || !(0.0..1.0).contains(&fraction) {
            excluded += 1;
            continue;
        }
        if !length.is_finite() {
            return Err(invalid("tensor fitting coordinate scale overflow"));
        }
        coordinate_scale = coordinate_scale.max(length);
        maximum_pairs = maximum_pairs.max(bin.pairs);
        data.push((bin.lag, -(-fraction).ln_1p(), bin.pairs as f64));
    }
    if data.len() < 6 {
        return Err(invalid(
            "six usable directional bins required for a 3D tensor",
        ));
    }
    let lower = (coordinate_scale / maximum).powi(2);
    let upper = (coordinate_scale / minimum).powi(2);
    if lower <= 0.0 || !upper.is_finite() {
        return Err(invalid("tensor metric bounds exceed floating-point range"));
    }
    let mut gram = Mat::<f64>::zeros(6, 6);
    let mut rhs = Mat::<f64>::zeros(6, 1);
    let mut lipschitz = 0.0;
    for (lag, target, weight) in &mut data {
        *lag = lag.map(|h| h / coordinate_scale);
        *weight /= maximum_pairs as f64;
        let [x, y, z] = *lag;
        let design = [
            x * x,
            y * y,
            z * z,
            2.0_f64.sqrt() * x * y,
            2.0_f64.sqrt() * x * z,
            2.0_f64.sqrt() * y * z,
        ];
        lipschitz += *weight * (x * x + y * y + z * z).powi(2);
        for i in 0..6 {
            rhs[(i, 0)] += *weight * design[i] * *target;
            for j in 0..6 {
                gram[(i, j)] += *weight * design[i] * design[j];
            }
        }
    }
    let eigenvalues = gram
        .self_adjoint_eigenvalues(Side::Lower)
        .map_err(|_| Error::Numerical {
            context: "tensor design spectrum",
        })?;
    if eigenvalues[0] <= 1e-10 * eigenvalues[5] {
        return Err(invalid(
            "directional evidence does not identify a well-conditioned 3D tensor",
        ));
    }
    let condition = (eigenvalues[5] / eigenvalues[0]).sqrt();
    factor_spd(gram.as_mut(), "tensor design rank")?;
    solve_spd(gram.as_ref(), rhs.as_mut());
    let mut metric = Mat::<f64>::zeros(3, 3);
    for i in 0..3 {
        metric[(i, i)] = rhs[(i, 0)];
    }
    for (index, (i, j)) in [(0, 1), (0, 2), (1, 2)].into_iter().enumerate() {
        metric[(i, j)] = rhs[(3 + index, 0)] / 2.0_f64.sqrt();
        metric[(j, i)] = metric[(i, j)];
    }
    metric = project(&metric, lower, upper)?;
    let mut iterations = 0;
    let mut converged = false;
    for iteration in 0..options.maximum_iterations {
        options.cancellation.check()?;
        let mut gradient = Mat::<f64>::zeros(3, 3);
        for (lag, target, weight) in &data {
            let residual = (quadratic(&metric, *lag) - target) * weight;
            for i in 0..3 {
                for j in 0..3 {
                    gradient[(i, j)] += residual * lag[i] * lag[j];
                }
            }
        }
        let trial = Mat::from_fn(3, 3, |i, j| metric[(i, j)] - gradient[(i, j)] / lipschitz);
        let next = project(&trial, lower, upper)?;
        let mut change = 0.0_f64;
        let mut magnitude = 0.0_f64;
        for i in 0..3 {
            for j in 0..3 {
                change = change.max((next[(i, j)] - metric[(i, j)]).abs());
                magnitude = magnitude.max(next[(i, j)].abs());
            }
        }
        metric = next;
        iterations = iteration + 1;
        if change * lipschitz <= options.tolerance * (1.0 + magnitude) {
            converged = true;
            break;
        }
    }
    let weighted_transformed_sse = data
        .iter()
        .map(|(lag, y, w)| w * (quadratic(&metric, *lag) - y).powi(2))
        .sum::<f64>();
    if !weighted_transformed_sse.is_finite() {
        return Err(Error::Numerical {
            context: "tensor fitting objective",
        });
    }
    let eigen = metric
        .self_adjoint_eigen(Side::Lower)
        .map_err(|_| Error::Numerical {
            context: "fitted tensor spectrum",
        })?;
    let frame = std::array::from_fn(|i| std::array::from_fn(|j| eigen.U()[(j, i)]));
    let scales = std::array::from_fn(|i| coordinate_scale / eigen.S()[i].sqrt());
    let axes_at_bound = scales
        .iter()
        .filter(|&&r| (r / minimum - 1.0).abs() < 1e-6 || (r / maximum - 1.0).abs() < 1e-6)
        .count();
    let anisotropy = Anisotropy::new(frame, scales)?;
    let tensor = Tensor::from_anisotropy(&anisotropy)?;
    Ok(TensorFit {
        anisotropy,
        tensor,
        converged,
        iterations,
        weighted_transformed_sse,
        design_condition_number: condition,
        used_bins: data.len(),
        excluded_bins: excluded,
        axes_at_bound,
    })
}
fn quadratic(matrix: &Mat<f64>, lag: [f64; 3]) -> f64 {
    (0..3)
        .map(|i| {
            (0..3)
                .map(|j| lag[i] * matrix[(i, j)] * lag[j])
                .sum::<f64>()
        })
        .sum()
}
pub(super) fn project(matrix: &Mat<f64>, lower: f64, upper: f64) -> Result<Mat<f64>> {
    crate::compute::linalg::finite(matrix.as_ref(), "tensor fitting iterate")?;
    let eigen = matrix
        .self_adjoint_eigen(Side::Lower)
        .map_err(|_| Error::Numerical {
            context: "tensor constraint projection",
        })?;
    let matrix = Mat::from_fn(3, 3, |i, j| {
        (0..3)
            .map(|k| eigen.U()[(i, k)] * eigen.S()[k].clamp(lower, upper) * eigen.U()[(j, k)])
            .sum::<f64>()
    });
    crate::compute::linalg::finite(matrix.as_ref(), "projected tensor")?;
    Ok(matrix)
}
