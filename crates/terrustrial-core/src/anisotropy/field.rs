//! Joint local metric fitting with explicit spatial regularization.
//!
//! Windows, known local sills/nuggets, and undirected links are supplied by the
//! caller. See `evidence` for grouped spatial windows and `selection` for family
//! and smoothing selection. This solver does not infer faults or uncertainty.
use super::{
    fitting::{project, DirectionalBin},
    AnchorField, Family, Tensor,
};
use crate::{
    data::{norm3, Point3},
    error::{checked_len, invalid},
    execution::{Cancellation, MemoryBudget},
    model::Anisotropy,
    Error, Result,
};
use faer::{Mat, Side};

pub struct Window<'a> {
    pub position: Point3,
    pub bins: &'a [DirectionalBin],
    pub structured_sill: f64,
    pub nugget: f64,
}
#[derive(Clone, Copy, Debug)]
pub struct Edge {
    pub nodes: [usize; 2],
    pub weight: f64,
}
pub struct Options<'a> {
    pub range_bounds: [f64; 2],
    /// Strength in the normalized metric coordinates reported by the result.
    pub regularization: f64,
    pub maximum_iterations: usize,
    /// Relative projected-gradient mapping tolerance.
    pub tolerance: f64,
    pub maximum_cg_iterations: usize,
    pub cg_tolerance: f64,
    /// Eigenvalues below this fraction of the largest design eigenvalue do not
    /// count toward rank. Applies to the Gram matrix, not its square root.
    pub rank_tolerance: f64,
    pub interpolation_neighbors: usize,
    pub interpolation_power: f64,
    pub budget: &'a MemoryBudget,
    pub cancellation: &'a Cancellation,
}
#[derive(Clone, Debug)]
pub struct NodeDiagnostics {
    pub design_rank: usize,
    pub used_bins: usize,
    pub excluded_bins: usize,
    pub axes_at_bound: usize,
    pub weighted_transformed_sse: f64,
    /// svec of coordinate_scale^2 times the inverse tensor. Off-diagonal
    /// entries are multiplied by sqrt(2), preserving the Frobenius norm.
    pub normalized_metric: [f64; 6],
}
pub struct FieldFit {
    pub family: Family,
    pub field: AnchorField,
    pub nodes: Vec<NodeDiagnostics>,
    pub coordinate_scale: f64,
    pub converged: bool,
    pub iterations: usize,
    pub cg_iterations: usize,
    pub cg_relative_residual: f64,
    pub weighted_transformed_sse: f64,
    /// Sum of edge weight times squared metric differences, before lambda.
    pub roughness: f64,
    /// Half of data SSE plus regularization times roughness.
    pub objective: f64,
    /// Infinity norm of the projected-gradient mapping, divided by 1+||x||inf.
    pub stationarity: f64,
}
#[derive(Clone, Default)]
struct Block {
    gram: [[f64; 6]; 6],
    rhs: [f64; 6],
    used: usize,
    excluded: usize,
}

/// Minimize 0.5 * sum(w * (h^T M h - y)^2) +
/// 0.5 * lambda * sum(edge_weight * ||M_i-M_j||_F^2), with bounded positive
/// eigenvalues. Here y=-ln(1-(gamma-nugget)/sill), h is divided by the global
/// maximum usable lag length, and pair counts are divided by the global maximum
/// pair count. All windows therefore share metric units and evidence weights.
///
/// Each linked component must have full pooled directional rank. A window may
/// have lower local rank, including no usable bins; diagnostics identify this.
/// With lambda=0 every window must have full rank. Duplicate positions and
/// duplicate/self links are rejected. O(windows + links) storage, independent of
/// bin count; no dense global matrix. A deterministic preconditioned conjugate
/// gradient warm start is followed by spectral projected gradient. A returned
/// unconverged fit is feasible but must not be mistaken for an optimum.
/// MemoryBudget accounts for logical transient workspace, not retained output or
/// process RSS. The returned index and all window tensors are resident in memory.
pub fn fit(windows: &[Window<'_>], edges: &[Edge], options: Options<'_>) -> Result<FieldFit> {
    fit_family(windows, edges, Family::Gaussian, options)
}

/// Fit local tensor directions and scales for a supported correlation family.
/// The inverse radial correlation replaces the Gaussian log transform; rank,
/// bounds, shared metric units and regularization semantics are unchanged.
/// As with `fit`, local experimental bins approximate stationary behavior within
/// their windows; use independent spatial predictions to check that assumption.
pub fn fit_family(
    windows: &[Window<'_>],
    edges: &[Edge],
    family: Family,
    options: Options<'_>,
) -> Result<FieldFit> {
    let n = windows.len();
    let [minimum, maximum] = options.range_bounds;
    if n == 0
        || !minimum.is_finite()
        || minimum <= 0.0
        || !maximum.is_finite()
        || maximum <= minimum
        || !options.regularization.is_finite()
        || options.regularization < 0.0
        || options.maximum_iterations == 0
        || options.maximum_cg_iterations == 0
        || !options.tolerance.is_finite()
        || options.tolerance <= 0.0
        || !options.cg_tolerance.is_finite()
        || options.cg_tolerance <= 0.0
        || options.cg_tolerance >= 1.0
        || !options.rank_tolerance.is_finite()
        || !(0.0..1.0).contains(&options.rank_tolerance)
        || options.rank_tolerance == 0.0
        || options.interpolation_neighbors == 0
        || !options.interpolation_power.is_finite()
        || options.interpolation_power <= 0.0
    {
        return Err(invalid("local tensor field fitting parameters"));
    }
    options.cancellation.check()?;
    let bytes = checked_len(&[n, 2048])?
        .checked_add(checked_len(&[edges.len(), 32])?)
        .and_then(|v| v.checked_add(16384))
        .ok_or(Error::SizeOverflow)?;
    let _reservation = options
        .budget
        .reserve(u64::try_from(bytes).map_err(|_| Error::SizeOverflow)?)?;
    let mut positions = std::collections::HashSet::new();
    positions
        .try_reserve(n)
        .map_err(|_| invalid("unable to allocate field positions"))?;
    let mut coordinate_scale = 0.0_f64;
    let mut maximum_pairs = 0;
    for window in windows {
        if !window.structured_sill.is_finite()
            || window.structured_sill <= 0.0
            || !window.nugget.is_finite()
            || window.nugget < 0.0
        {
            return Err(invalid("local tensor sill/nugget"));
        }
        if !positions.insert(window.position.coordinates().map(|x| {
            if x == 0.0 {
                0
            } else {
                x.to_bits()
            }
        })) {
            return Err(invalid("duplicate local tensor positions"));
        }
        for bin in window.bins {
            options.cancellation.check()?;
            if let Some((_, length)) = evidence(window, bin, family)? {
                coordinate_scale = coordinate_scale.max(length);
                maximum_pairs = maximum_pairs.max(bin.pairs);
            }
        }
    }
    let lower = (coordinate_scale / maximum).powi(2);
    let upper = (coordinate_scale / minimum).powi(2);
    if lower <= 0.0 || !upper.is_finite() || upper <= lower {
        return Err(invalid(
            "local tensor metric bounds exceed floating-point range or no usable evidence",
        ));
    }
    let mut links = Vec::new();
    links
        .try_reserve_exact(edges.len())
        .map_err(|_| invalid("unable to allocate field links"))?;
    for edge in edges {
        options.cancellation.check()?;
        let [a, b] = edge.nodes;
        if a >= n || b >= n || a == b || !edge.weight.is_finite() || edge.weight <= 0.0 {
            return Err(invalid("local tensor field link"));
        }
        let weight = edge.weight * options.regularization;
        if !weight.is_finite() || (options.regularization > 0.0 && weight == 0.0) {
            return Err(invalid(
                "local tensor link strength exceeds floating-point range",
            ));
        }
        links.push(Edge {
            nodes: [a.min(b), a.max(b)],
            weight,
        });
    }
    links.sort_unstable_by_key(|e| e.nodes);
    if links.windows(2).any(|pair| pair[0].nodes == pair[1].nodes) {
        return Err(invalid("duplicate local tensor field links"));
    }
    let mut blocks = vec![Block::default(); n];
    for (block, window) in blocks.iter_mut().zip(windows) {
        for bin in window.bins {
            options.cancellation.check()?;
            let Some((target, _)) = evidence(window, bin, family)? else {
                block.excluded += 1;
                continue;
            };
            block.used += 1;
            let phi = design(bin.lag.map(|v| v / coordinate_scale));
            let w = bin.pairs as f64 / maximum_pairs as f64;
            for i in 0..6 {
                block.rhs[i] += w * phi[i] * target;
                for j in 0..6 {
                    block.gram[i][j] += w * phi[i] * phi[j];
                }
            }
        }
    }
    let mut parent: Vec<_> = (0..n).collect();
    for edge in &links {
        if edge.weight > 0.0 {
            let a = root(&mut parent, edge.nodes[0]);
            let b = root(&mut parent, edge.nodes[1]);
            parent[b] = a;
        }
    }
    let mut pooled = vec![[[0.0; 6]; 6]; n];
    for (i, block) in blocks.iter().enumerate() {
        let r = root(&mut parent, i);
        for (row, source) in pooled[r].iter_mut().zip(&block.gram) {
            for (value, source) in row.iter_mut().zip(source) {
                *value += source;
            }
        }
    }
    for (i, gram) in pooled.iter().enumerate() {
        if parent[i] == i {
            let r = rank(gram, options.rank_tolerance)?;
            if r < 6 {
                return Err(invalid(format!(
                    "local tensor component at node {i} has directional rank {r}, requires 6"
                )));
            }
        }
    }
    drop(pooled);
    let mut diagonal = vec![[0.0; 6]; n];
    let mut degree = vec![0.0; n];
    for e in &links {
        for i in e.nodes {
            degree[i] += e.weight;
        }
    }
    let mut lipschitz = 0.0_f64;
    for i in 0..n {
        let trace = (0..6).map(|j| blocks[i].gram[j][j]).sum::<f64>();
        lipschitz = lipschitz.max(trace + 2.0 * degree[i]);
        for (j, value) in diagonal[i].iter_mut().enumerate() {
            *value = blocks[i].gram[j][j] + degree[i];
            if *value <= 0.0 || !value.is_finite() {
                return Err(invalid("local tensor field diagonal"));
            }
        }
    }
    if !lipschitz.is_finite() || lipschitz <= 0.0 {
        return Err(invalid("local tensor field spectral bound"));
    }
    let rhs: Vec<_> = blocks.iter().map(|b| b.rhs).collect();
    let (mut x, cg_iterations, cg_relative_residual) =
        cg(&blocks, &links, &diagonal, &rhs, &options)?;
    for value in &mut x {
        *value = project_vector(*value, lower, upper)?;
    }
    let mut product = vec![[0.0; 6]; n];
    let mut next = vec![[0.0; 6]; n];
    let mut stationarity = f64::INFINITY;
    let mut converged = false;
    let mut iterations = 0;
    // Test stationarity at the returned iterate, including after the last update.
    for iteration in 0..=options.maximum_iterations {
        options.cancellation.check()?;
        multiply(&blocks, &links, &x, &mut product, options.cancellation)?;
        let mut change = 0.0_f64;
        let mut magnitude = 0.0_f64;
        for i in 0..n {
            options.cancellation.check()?;
            next[i] = project_vector(
                std::array::from_fn(|j| x[i][j] - (product[i][j] - rhs[i][j]) / lipschitz),
                lower,
                upper,
            )?;
            for j in 0..6 {
                change = change.max((next[i][j] - x[i][j]).abs());
                magnitude = magnitude.max(x[i][j].abs());
            }
        }
        stationarity = change / (1.0 + magnitude) * lipschitz;
        if !stationarity.is_finite() {
            return Err(Error::Numerical {
                context: "local tensor field stationarity",
            });
        }
        if stationarity <= options.tolerance {
            converged = true;
            break;
        }
        if iteration == options.maximum_iterations {
            break;
        }
        std::mem::swap(&mut x, &mut next);
        iterations += 1;
    }
    let mut nodes = Vec::with_capacity(n);
    let mut anchors = Vec::with_capacity(n);
    let mut weighted_transformed_sse = 0.0;
    for (i, window) in windows.iter().enumerate() {
        options.cancellation.check()?;
        let metric = matrix(x[i]);
        let eigen = metric
            .self_adjoint_eigen(Side::Lower)
            .map_err(|_| Error::Numerical {
                context: "local tensor field spectrum",
            })?;
        let frame = std::array::from_fn(|a| std::array::from_fn(|b| eigen.U()[(b, a)]));
        let scales = std::array::from_fn(|a| coordinate_scale / eigen.S()[a].sqrt());
        let axes_at_bound = scales
            .iter()
            .filter(|&&r| (r / minimum - 1.0).abs() < 1e-6 || (r / maximum - 1.0).abs() < 1e-6)
            .count();
        let tensor = Tensor::from_anisotropy(&Anisotropy::new(frame, scales)?)?;
        let mut sse = 0.0;
        for bin in window.bins {
            options.cancellation.check()?;
            if let Some((target, _)) = evidence(window, bin, family)? {
                let phi = design(bin.lag.map(|v| v / coordinate_scale));
                sse += bin.pairs as f64 / maximum_pairs as f64 * (dot6(phi, x[i]) - target).powi(2);
            }
        }
        weighted_transformed_sse += sse;
        nodes.push(NodeDiagnostics {
            design_rank: rank(&blocks[i].gram, options.rank_tolerance)?,
            used_bins: blocks[i].used,
            excluded_bins: blocks[i].excluded,
            axes_at_bound,
            weighted_transformed_sse: sse,
            normalized_metric: x[i],
        });
        anchors.push((window.position, tensor));
    }
    // Original edge weights report roughness independently of lambda, including lambda=0.
    let roughness = edges
        .iter()
        .map(|e| {
            e.weight
                * (0..6)
                    .map(|j| (x[e.nodes[0]][j] - x[e.nodes[1]][j]).powi(2))
                    .sum::<f64>()
        })
        .sum::<f64>();
    let objective = 0.5 * weighted_transformed_sse + 0.5 * options.regularization * roughness;
    if !weighted_transformed_sse.is_finite() || !roughness.is_finite() || !objective.is_finite() {
        return Err(Error::Numerical {
            context: "local tensor field objective",
        });
    }
    let field = AnchorField::new(
        anchors,
        options.interpolation_neighbors,
        options.interpolation_power,
    )?;
    Ok(FieldFit {
        family,
        field,
        nodes,
        coordinate_scale,
        converged,
        iterations,
        cg_iterations,
        cg_relative_residual,
        weighted_transformed_sse,
        roughness,
        objective,
        stationarity,
    })
}
fn evidence(
    window: &Window<'_>,
    bin: &DirectionalBin,
    family: Family,
) -> Result<Option<(f64, f64)>> {
    if bin.lag.iter().any(|v| !v.is_finite())
        || !bin.semivariance.is_finite()
        || bin.semivariance < 0.0
    {
        return Err(invalid("local tensor directional evidence"));
    }
    let length = norm3(bin.lag);
    let fraction = (bin.semivariance - window.nugget) / window.structured_sill;
    if bin.pairs == 0 || length == 0.0 || !(0.0..1.0).contains(&fraction) {
        return Ok(None);
    }
    if !length.is_finite() {
        return Err(invalid("local tensor coordinate scale overflow"));
    }
    let target = family.metric_target(fraction);
    if fraction > 0.0 && target == 0.0 {
        return Err(invalid(
            "local tensor transformed lag underflows for this family",
        ));
    }
    Ok(Some((target, length)))
}
fn design([x, y, z]: [f64; 3]) -> [f64; 6] {
    [
        x * x,
        y * y,
        z * z,
        std::f64::consts::SQRT_2 * x * y,
        std::f64::consts::SQRT_2 * x * z,
        std::f64::consts::SQRT_2 * y * z,
    ]
}
fn dot6(a: [f64; 6], b: [f64; 6]) -> f64 {
    (0..6).map(|j| a[j] * b[j]).sum()
}
fn matrix(x: [f64; 6]) -> Mat<f64> {
    let mut m = Mat::zeros(3, 3);
    for i in 0..3 {
        m[(i, i)] = x[i];
    }
    for (k, (i, j)) in [(0, 1), (0, 2), (1, 2)].into_iter().enumerate() {
        m[(i, j)] = x[k + 3] / std::f64::consts::SQRT_2;
        m[(j, i)] = m[(i, j)];
    }
    m
}
fn project_vector(x: [f64; 6], lower: f64, upper: f64) -> Result<[f64; 6]> {
    let m = project(&matrix(x), lower, upper)?;
    Ok([
        m[(0, 0)],
        m[(1, 1)],
        m[(2, 2)],
        std::f64::consts::SQRT_2 * m[(0, 1)],
        std::f64::consts::SQRT_2 * m[(0, 2)],
        std::f64::consts::SQRT_2 * m[(1, 2)],
    ])
}
fn rank(g: &[[f64; 6]; 6], tolerance: f64) -> Result<usize> {
    let m = Mat::from_fn(6, 6, |i, j| g[i][j]);
    crate::compute::linalg::finite(m.as_ref(), "local tensor design")?;
    let eigen = m
        .self_adjoint_eigenvalues(Side::Lower)
        .map_err(|_| Error::Numerical {
            context: "local tensor design rank",
        })?;
    if eigen.iter().any(|v| !v.is_finite()) {
        return Err(Error::Numerical {
            context: "local tensor design spectrum",
        });
    }
    Ok(eigen.iter().filter(|&&v| v > tolerance * eigen[5]).count())
}
fn root(parent: &mut [usize], mut i: usize) -> usize {
    while parent[i] != i {
        parent[i] = parent[parent[i]];
        i = parent[i];
    }
    i
}
fn multiply(
    blocks: &[Block],
    edges: &[Edge],
    x: &[[f64; 6]],
    out: &mut [[f64; 6]],
    cancel: &Cancellation,
) -> Result<()> {
    for (i, b) in blocks.iter().enumerate() {
        cancel.check()?;
        for (j, value) in out[i].iter_mut().enumerate() {
            *value = dot6(b.gram[j], x[i]);
        }
    }
    for e in edges {
        cancel.check()?;
        let [a, b] = e.nodes;
        for j in 0..6 {
            let d = e.weight * (x[a][j] - x[b][j]);
            out[a][j] += d;
            out[b][j] -= d;
        }
    }
    if out.iter().flatten().any(|v| !v.is_finite()) {
        return Err(Error::Numerical {
            context: "local tensor field product",
        });
    }
    Ok(())
}
fn dot(a: &[[f64; 6]], b: &[[f64; 6]]) -> f64 {
    a.iter().zip(b).map(|(&a, &b)| dot6(a, b)).sum()
}
fn norm(a: &[[f64; 6]]) -> f64 {
    a.iter().flatten().fold(0.0, |a, &b| a.hypot(b))
}
fn cg(
    blocks: &[Block],
    edges: &[Edge],
    diagonal: &[[f64; 6]],
    rhs: &[[f64; 6]],
    options: &Options<'_>,
) -> Result<(Vec<[f64; 6]>, usize, f64)> {
    let n = blocks.len();
    let mut x = vec![[0.0; 6]; n];
    let mut r = rhs.to_vec();
    let mut z = vec![[0.0; 6]; n];
    for i in 0..n {
        for j in 0..6 {
            z[i][j] = r[i][j] / diagonal[i][j];
        }
    }
    let mut p = z.clone();
    let mut product = vec![[0.0; 6]; n];
    let scale = norm(rhs);
    if scale == 0.0 {
        return Ok((x, 0, 0.0));
    }
    if !scale.is_finite() {
        return Err(Error::Numerical {
            context: "local tensor field right hand side",
        });
    }
    let mut rz = dot(&r, &z);
    let mut relative = 1.0;
    for iteration in 0..options.maximum_cg_iterations {
        options.cancellation.check()?;
        multiply(blocks, edges, &p, &mut product, options.cancellation)?;
        let denominator = dot(&p, &product);
        if denominator <= 0.0 || !denominator.is_finite() || rz <= 0.0 || !rz.is_finite() {
            return Err(Error::Numerical {
                context: "local tensor field CG curvature",
            });
        }
        let alpha = rz / denominator;
        for i in 0..n {
            for j in 0..6 {
                x[i][j] += alpha * p[i][j];
                r[i][j] -= alpha * product[i][j];
            }
        }
        relative = norm(&r) / scale;
        let restart = (iteration + 1) % 25 == 0
            || relative <= options.cg_tolerance
            || iteration + 1 == options.maximum_cg_iterations;
        if restart {
            multiply(blocks, edges, &x, &mut product, options.cancellation)?;
            for i in 0..n {
                for j in 0..6 {
                    r[i][j] = rhs[i][j] - product[i][j];
                }
            }
            relative = norm(&r) / scale;
        }
        if !relative.is_finite() {
            return Err(Error::Numerical {
                context: "local tensor field CG residual",
            });
        }
        if relative <= options.cg_tolerance {
            return Ok((x, iteration + 1, relative));
        }
        for i in 0..n {
            for j in 0..6 {
                z[i][j] = r[i][j] / diagonal[i][j];
            }
        }
        let next = dot(&r, &z);
        let beta = if restart { 0.0 } else { next / rz };
        for i in 0..n {
            for j in 0..6 {
                p[i][j] = z[i][j] + beta * p[i][j];
            }
        }
        rz = next;
    }
    Ok((x, options.maximum_cg_iterations, relative))
}
