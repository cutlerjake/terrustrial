//! Valid nonstationary covariance, local tensor fitting and spatial selection.
//! Formula: Paciorek and Schervish (2006), doi:10.1002/env.785.
pub mod evidence;
mod family;
pub mod field;
pub mod fitting;
pub use family::Family;
mod prepared;
pub mod selection;
pub use prepared::PreparedField;

use crate::spatial::hierarchy::{Bounded, Bounds, Hierarchy};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};

use crate::{
    data::Point3,
    error::invalid,
    model::{Anisotropy, Coregionalization, Covariance},
    Error, Result,
};

#[derive(Clone, Debug)]
pub struct Tensor {
    matrix: [[f64; 3]; 3],
    lower: [[f64; 3]; 3],
    log_determinant: f64,
}
impl Tensor {
    pub fn new(matrix: [[f64; 3]; 3]) -> Result<Self> {
        if matrix.iter().flatten().any(|v| !v.is_finite()) {
            return Err(invalid("nonfinite anisotropy tensor"));
        }
        let mut lower = [[0.0_f64; 3]; 3];
        for i in 0..3 {
            for j in 0..=i {
                let symmetry_scale = matrix[i][i].abs().max(matrix[j][j].abs());
                if (matrix[i][j] - matrix[j][i]).abs() > 1e-12 * symmetry_scale {
                    return Err(invalid("anisotropy tensor must be symmetric"));
                }
                let residual =
                    matrix[i][j] - (0..j).map(|k| lower[i][k] * lower[j][k]).sum::<f64>();
                lower[i][j] = if i == j {
                    if residual <= 0.0 || !residual.is_finite() {
                        return Err(invalid("anisotropy tensor must be positive definite"));
                    }
                    residual.sqrt()
                } else {
                    residual / lower[j][j]
                };
            }
        }
        if lower.iter().flatten().any(|v| !v.is_finite()) {
            return Err(invalid("anisotropy factor overflow"));
        }
        let log_determinant = 2.0 * (0..3).map(|i| lower[i][i].ln()).sum::<f64>();
        Ok(Self {
            matrix,
            lower,
            log_determinant,
        })
    }
    pub fn from_anisotropy(anisotropy: &Anisotropy) -> Result<Self> {
        let r = anisotropy.frame();
        let s = anisotropy.scales();
        let mut matrix = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                matrix[i][j] = (0..3).map(|k| r[k][i] * r[k][j] * s[k] * s[k]).sum();
            }
        }
        Self::new(matrix)
    }
    pub fn matrix(&self) -> [[f64; 3]; 3] {
        self.matrix
    }
    fn quadratic(&self, delta: [f64; 3]) -> Result<f64> {
        let mut y = [0.0; 3];
        for i in 0..3 {
            y[i] = (delta[i] - (0..i).map(|j| self.lower[i][j] * y[j]).sum::<f64>())
                / self.lower[i][i];
        }
        let norm = y[0].hypot(y[1]).hypot(y[2]);
        if norm.is_nan() {
            return Err(Error::Numerical {
                context: "anisotropic distance",
            });
        }
        Ok(norm * norm)
    }
}

pub trait TensorField: Sync {
    fn at(&self, point: Point3) -> Result<Tensor>;
}
impl<F: Fn(Point3) -> Result<Tensor> + Sync> TensorField for F {
    fn at(&self, point: Point3) -> Result<Tensor> {
        self(point)
    }
}

#[derive(Clone)]
struct AnchorEntry {
    point: [f64; 3],
    index: usize,
}
impl Bounded for AnchorEntry {
    fn bounds(&self) -> Bounds {
        Bounds::point(self.point)
    }
}
#[derive(Clone, Copy)]
struct AnchorDistance {
    distance: f64,
    index: usize,
}
impl PartialEq for AnchorDistance {
    fn eq(&self, rhs: &Self) -> bool {
        self.distance == rhs.distance && self.index == rhs.index
    }
}
impl Eq for AnchorDistance {}
impl PartialOrd for AnchorDistance {
    fn partial_cmp(&self, rhs: &Self) -> Option<Ordering> {
        Some(self.cmp(rhs))
    }
}
impl Ord for AnchorDistance {
    fn cmp(&self, rhs: &Self) -> Ordering {
        self.distance
            .total_cmp(&rhs.distance)
            .then(self.index.cmp(&rhs.index))
    }
}

/// Convex tensor interpolation over indexed nearest anchors. This interpolates
/// supplied evidence; it does not by itself infer tensors from assays. Exact
/// anchors are honored, and input anchor indices break equal-distance ties.
/// Query workspace is O(neighbors + tree depth). A depth-first traversal uses
/// robust, unsquared distances for pruning, including tiny/large coordinates.
/// Degenerate layouts or distant queries can still require visiting all anchors.
/// The index and tensors themselves remain resident in memory.
/// Truncating inverse-distance interpolation to a changing nearest-neighbor set
/// can introduce jumps at set boundaries. Use all anchors when continuity is
/// required and the anchor count permits it; prepared tensors amortize that work.
pub struct AnchorField {
    anchors: Vec<(Point3, Tensor)>,
    tree: Hierarchy<AnchorEntry>,
    neighbors: usize,
    power: f64,
}
impl AnchorField {
    pub fn new(anchors: Vec<(Point3, Tensor)>, neighbors: usize, power: f64) -> Result<Self> {
        if anchors.is_empty() || neighbors == 0 || !power.is_finite() || power <= 0.0 {
            return Err(invalid("anisotropy anchor field parameters"));
        }
        let mut positions = HashSet::new();
        positions
            .try_reserve(anchors.len())
            .map_err(|_| invalid("unable to allocate anchor positions"))?;
        for (point, _) in &anchors {
            let key = point
                .coordinates()
                .map(|x| if x == 0.0 { 0 } else { x.to_bits() });
            if !positions.insert(key) {
                return Err(invalid("duplicate anisotropy anchors"));
            }
        }
        drop(positions);
        let tree = Hierarchy::new(
            anchors
                .iter()
                .enumerate()
                .map(|(index, (point, _))| AnchorEntry {
                    point: point.coordinates(),
                    index,
                })
                .collect(),
        )?;
        Ok(Self {
            neighbors: neighbors.min(anchors.len()),
            anchors,
            tree,
            power,
        })
    }
    pub fn anchors(&self) -> &[(Point3, Tensor)] {
        &self.anchors
    }
    fn visit(
        &self,
        point: Point3,
        nearest: &mut BinaryHeap<AnchorDistance>,
    ) -> Result<Option<usize>> {
        let mut state = (nearest, None);
        self.tree.traverse(
            &mut state,
            |(nearest, exact), bounds| {
                if exact.is_some() {
                    return None;
                }
                let d = crate::data::norm3(bounds.delta(point.coordinates()));
                let cutoff = if nearest.len() == self.neighbors {
                    (nearest.peek().unwrap().distance * (1. + 8. * f64::EPSILON)).next_up()
                } else {
                    f64::INFINITY
                };
                (d <= cutoff).then_some(d)
            },
            |(nearest, exact), entry, distance| {
                if distance == 0. {
                    *exact = Some(entry.index);
                    return Ok(());
                }
                let candidate = AnchorDistance {
                    distance,
                    index: entry.index,
                };
                if nearest.len() < self.neighbors {
                    nearest.push(candidate);
                } else if nearest.peek().is_some_and(|worst| candidate < *worst) {
                    *nearest.peek_mut().unwrap() = candidate;
                }
                Ok(())
            },
        )?;
        Ok(state.1)
    }
}
impl TensorField for AnchorField {
    fn at(&self, point: Point3) -> Result<Tensor> {
        let mut nearest = BinaryHeap::new();
        nearest
            .try_reserve_exact(self.neighbors)
            .map_err(|_| invalid("unable to allocate anchor-query workspace"))?;
        if let Some(index) = self.visit(point, &mut nearest)? {
            return Ok(self.anchors[index].1.clone());
        }
        self.interpolate(&nearest.into_sorted_vec())
    }
}

/// Reusable interpolation storage. One workspace per concurrent caller; it can
/// be shared between fields whose neighbor count does not exceed its capacity.
pub struct AnchorWorkspace {
    nearest: BinaryHeap<AnchorDistance>,
    sorted: Vec<AnchorDistance>,
}
impl AnchorField {
    pub fn workspace(&self) -> Result<AnchorWorkspace> {
        let mut nearest = BinaryHeap::new();
        let mut sorted = Vec::new();
        nearest
            .try_reserve_exact(self.neighbors)
            .map_err(|_| invalid("unable to allocate anchor-query workspace"))?;
        sorted
            .try_reserve_exact(self.neighbors)
            .map_err(|_| invalid("unable to allocate anchor-query ordering"))?;
        Ok(AnchorWorkspace { nearest, sorted })
    }
    /// No allocation after workspace construction, including exact-anchor and
    /// failed queries. Summation order is nearest first, with stable index ties.
    pub fn at_with_workspace(
        &self,
        point: Point3,
        workspace: &mut AnchorWorkspace,
    ) -> Result<Tensor> {
        if workspace.nearest.capacity() < self.neighbors
            || workspace.sorted.capacity() < self.neighbors
        {
            return Err(invalid("insufficient anchor workspace capacity"));
        }
        let nearest = &mut workspace.nearest;
        nearest.clear();
        workspace.sorted.clear();
        if let Some(index) = self.visit(point, nearest)? {
            return Ok(self.anchors[index].1.clone());
        }
        while let Some(entry) = nearest.pop() {
            workspace.sorted.push(entry);
        }
        workspace.sorted.reverse();
        self.interpolate(&workspace.sorted)
    }
    fn interpolate(&self, distances: &[AnchorDistance]) -> Result<Tensor> {
        let closest = distances[0].distance;
        if !closest.is_finite() {
            return Err(invalid("anchor distances overflow"));
        }
        let total = distances
            .iter()
            .map(|a| (closest / a.distance).powf(self.power))
            .sum::<f64>();
        let mut matrix = [[0.0; 3]; 3];
        for anchor in distances {
            let weight = (closest / anchor.distance).powf(self.power) / total;
            let tensor = &self.anchors[anchor.index].1;
            for (i, row) in matrix.iter_mut().enumerate() {
                for (j, value) in row.iter_mut().enumerate() {
                    *value += weight * tensor.matrix[i][j];
                }
            }
        }
        Tensor::new(matrix)
    }
}

/// Gaussian member of the Paciorek-Schervish construction, multiplied by a PSD
/// cross-variable sill matrix. A shared tensor field applies to all variables.
pub struct NonstationaryGaussian<F> {
    pub field: F,
    pub coregionalization: Coregionalization,
}
/// Paciorek-Schervish covariance with a shared local tensor field and a PSD
/// cross-variable sill matrix. Families are restricted to those valid in every
/// dimension. A separate point nugget/noise component is not implied.
pub struct Nonstationary<F> {
    pub family: Family,
    pub field: F,
    pub coregionalization: Coregionalization,
}
impl<F: TensorField> Covariance for Nonstationary<F> {
    fn variables(&self) -> usize {
        self.coregionalization.variables()
    }
    fn point_covariance(&self, a: Point3, va: usize, b: Point3, vb: usize) -> Result<f64> {
        let sill = self.coregionalization.covariance(va, vb)?;
        Ok(sill * correlation(a, &self.field.at(a)?, b, &self.field.at(b)?, self.family)?)
    }
}
impl<F: TensorField> Covariance for NonstationaryGaussian<F> {
    fn variables(&self) -> usize {
        self.coregionalization.variables()
    }
    fn point_covariance(&self, a: Point3, va: usize, b: Point3, vb: usize) -> Result<f64> {
        let sill = self.coregionalization.covariance(va, vb)?;
        let sa = self.field.at(a)?;
        let sb = self.field.at(b)?;
        Ok(sill * gaussian_correlation(a, &sa, b, &sb)?)
    }
}

fn gaussian_correlation(a: Point3, sa: &Tensor, b: Point3, sb: &Tensor) -> Result<f64> {
    correlation(a, sa, b, sb, Family::Gaussian)
}
fn correlation(a: Point3, sa: &Tensor, b: Point3, sb: &Tensor, family: Family) -> Result<f64> {
    let mut average = [[0.0; 3]; 3];
    for (i, row) in average.iter_mut().enumerate() {
        for (j, value) in row.iter_mut().enumerate() {
            *value = 0.5 * sa.matrix[i][j] + 0.5 * sb.matrix[i][j];
        }
    }
    let average = Tensor::new(average)?;
    let a = a.coordinates();
    let b = b.coordinates();
    let delta = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    if delta.iter().any(|d| d.is_infinite()) {
        return Ok(0.0);
    }
    let log_normalization =
        0.25 * sa.log_determinant + 0.25 * sb.log_determinant - 0.5 * average.log_determinant;
    let correlation = (log_normalization + family.log_correlation(average.quadratic(delta)?)).exp();
    if !correlation.is_finite() {
        return Err(Error::Numerical {
            context: "nonstationary covariance",
        });
    }
    Ok(correlation)
}
