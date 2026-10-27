//! Bounded, common-support midpoint refinement diagnostics.
//!
//! Supply the union of observation and target sites, then use the returned sites
//! consistently in the Gaussian plan, mean and drift evaluations. A rule is never
//! chosen separately for each matrix entry, which could destroy covariance
//! consistency. This does not alter a model or silently refine ordinary calls.
//!
//! Successive-rule agreement is an empirical diagnostic, not an integration
//! error bound. Discontinuous/nugget kernels, oscillation, aliasing and coordinate
//! rounding may give false agreement. Validate against problem-specific reference
//! integrals and check downstream predictions as well as matrix entries. Nugget
//! behavior retains the model's explicit discrete node-coincidence semantics.
use crate::{
    data::{Site, Support},
    error::{checked_len, invalid},
    execution::{Cancellation, MemoryBudget, Reservation},
    model::Covariance,
    Error, Result,
};
use std::mem::size_of;

#[derive(Clone, Copy, Debug)]
pub struct RefinementOptions {
    /// Entrywise test: |fine-coarse| <= absolute + relative * max(|fine|, |coarse|).
    /// Absolute tolerance has covariance units. Both tolerances may be zero.
    pub absolute_tolerance: f64,
    pub relative_tolerance: f64,
    /// Number of consecutive successful comparisons required before stopping.
    pub stable_comparisons: usize,
    pub maximum_refinements: usize,
    /// Cumulative coarse/fine midpoint node-pair estimate across all comparisons.
    /// Point-point entries do not change and are skipped. Custom `covariance`
    /// overrides are honored, but their actual internal work is not measurable here.
    pub maximum_point_pairs: u64,
}
impl Default for RefinementOptions {
    fn default() -> Self {
        Self {
            absolute_tolerance: 0.,
            relative_tolerance: 1e-4,
            stable_comparisons: 2,
            maximum_refinements: 6,
            maximum_point_pairs: 10_000_000,
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RefinementStop {
    /// Every support is a point; no quadrature comparison was necessary.
    PointsOnly,
    /// The requested consecutive comparisons agree. This is not an error bound.
    Stable,
    RefinementLimit,
    WorkLimit,
    /// Doubling a noncollapsed axis would overflow its divisions or node count.
    DimensionLimit,
    /// Adjacent edge nodes on a noncollapsed axis have identical coordinates.
    ResolutionLimit,
}
#[derive(Clone, Copy, Debug)]
pub struct RefinementReport {
    pub stop: RefinementStop,
    pub completed_refinements: usize,
    pub stable_comparisons: usize,
    pub covariance_evaluations: u64,
    pub estimated_point_pairs: u64,
    /// Maxima from the last complete comparison, or None if no comparison ran.
    /// Absolute change can be infinite when two finite values differ by more
    /// than f64::MAX. The normalized relative comparison still remains valid.
    pub maximum_absolute_change: Option<f64>,
    pub maximum_relative_change: Option<f64>,
}
/// Owns the selected sites and their memory reservation. Dropping releases both.
/// Limits return the last complete comparison's sites, or the original sites if
/// no comparison ran, never a partly assessed next level.
pub struct RefinedSites {
    sites: Vec<Site>,
    report: RefinementReport,
    _reservation: Reservation,
}
impl RefinedSites {
    pub fn sites(&self) -> &[Site] {
        &self.sites
    }
    pub fn report(&self) -> RefinementReport {
        self.report
    }
    pub fn accounted_bytes(&self) -> u64 {
        self._reservation.bytes()
    }
}

fn refine(site: Site) -> Result<Site> {
    let extents = site.support.half_extents();
    let mut divisions = site.support.divisions();
    for axis in 0..3 {
        if extents[axis] != 0. {
            divisions[axis] = divisions[axis].checked_mul(2).ok_or(Error::SizeOverflow)?;
        }
    }
    Ok(Site {
        support: Support::block(site.support.center(), extents, divisions)?,
        variable: site.variable,
    })
}
fn point_pairs(sites: &[Site], cancel: &Cancellation) -> Result<u128> {
    let mut total = 0_u128;
    let mut points = 0_u128;
    for site in sites {
        cancel.check()?;
        total = total
            .checked_add(site.support.node_count() as u128)
            .ok_or(Error::SizeOverflow)?;
        points += u128::from(site.support.is_point());
    }
    total
        .checked_mul(total)
        .and_then(|n| n.checked_sub(points * points))
        .ok_or(Error::SizeOverflow)
}
fn distinct_edge_nodes(site: &Site) -> bool {
    let center = site.support.center().coordinates();
    let half = site.support.half_extents();
    let divisions = site.support.divisions();
    (0..3).all(|axis| {
        let n = divisions[axis];
        if half[axis] == 0. || n < 2 {
            return true;
        }
        // Match Support::nodes arithmetic. The edge comparisons detect loss of
        // spatial resolution without traversing a potentially enormous rule.
        let node = |digit: usize| {
            center[axis] + half[axis] * (2. * ((digit as f64 + 0.5) / n as f64) - 1.)
        };
        node(0) != node(1) && node(n - 2) != node(n - 1)
    })
}
fn buffer(sites: &[Site]) -> Result<Vec<Site>> {
    let mut out = Vec::new();
    out.try_reserve_exact(sites.len())
        .map_err(|_| invalid("quadrature site allocation"))?;
    out.extend_from_slice(sites);
    Ok(out)
}
/// Assess every ordered covariance entry that involves a block while doubling
/// each noncollapsed axis at every level. Storage is O(number of sites), with two
/// charged site buffers and no retained covariance matrix. Each level recomputes
/// coarse and fine entries to avoid quadratic retained memory. With B block
/// supports and N total sites, selection visits O(B*N) entries, never N*N
/// point-point combinations. Both orientations are assessed independently.
///
/// Cancellation is checked between model calls; an individual custom covariance
/// call is not interruptible by this function. The work estimate is checked before
/// starting a level. Model/callback values must be deterministic and immutable.
/// Mean and drift quadrature need their own accuracy checks; covariance agreement
/// does not bound their integration error or an ill-conditioned solve's error.
/// Errors return no selected sites and release all reservations. Work/refinement
/// limits are successful diagnostic results whose `stop` must be inspected.
///
/// ```
/// use terrustrial_core::{data::{Point3, Site, Support}, execution::{Cancellation, MemoryBudget},
///     model::{Anisotropy, Coregionalization, Correlation, Lmc, Structure,
///         quadrature::{refine_covariance, RefinementOptions}}};
/// # fn main() -> terrustrial_core::Result<()> {
/// let model = Lmc::new(vec![Structure { correlation: Correlation::Gaussian,
///     anisotropy: Anisotropy::isotropic(2.)?, coregionalization: Coregionalization::scalar(1.)? }])?;
/// let site = Site { variable: 0, support: Support::block(Point3::new([0.; 3])?,
///     [0.5, 0., 0.], [2, 1, 1])? };
/// let refined = refine_covariance(&model, &[site], RefinementOptions::default(),
///     &MemoryBudget::new(4096), &Cancellation::default())?;
/// // Inspect refined.report().stop; use refined.sites() consistently in the plan.
/// # Ok(()) }
/// ```
pub fn refine_covariance(
    model: &(impl Covariance + ?Sized),
    sites: &[Site],
    options: RefinementOptions,
    budget: &MemoryBudget,
    cancel: &Cancellation,
) -> Result<RefinedSites> {
    cancel.check()?;
    if !options.absolute_tolerance.is_finite()
        || options.absolute_tolerance < 0.
        || !options.relative_tolerance.is_finite()
        || options.relative_tolerance < 0.
        || options.stable_comparisons == 0
    {
        return Err(invalid("quadrature tolerances or stable-comparison count"));
    }
    let variables = model.variables();
    for site in sites {
        cancel.check()?;
        if site.variable >= variables {
            return Err(invalid("quadrature variable index"));
        }
    }
    let bytes = u64::try_from(checked_len(&[sites.len(), size_of::<Site>()])?)
        .map_err(|_| Error::SizeOverflow)?;
    let reservation = budget.reserve(bytes)?;
    let mut current = buffer(sites)?;
    let mut report = RefinementReport {
        stop: RefinementStop::PointsOnly,
        completed_refinements: 0,
        stable_comparisons: 0,
        covariance_evaluations: 0,
        estimated_point_pairs: 0,
        maximum_absolute_change: None,
        maximum_relative_change: None,
    };
    if sites.iter().any(|s| !s.support.is_point()) {
        let _scratch_reservation = budget.reserve(bytes)?;
        let mut next = buffer(sites)?;
        report.stop = RefinementStop::RefinementLimit;
        'levels: for _ in 0..options.maximum_refinements {
            cancel.check()?;
            for (to, from) in next.iter_mut().zip(&current) {
                cancel.check()?;
                match refine(*from) {
                    Ok(site) => {
                        if !distinct_edge_nodes(&site) {
                            report.stop = RefinementStop::ResolutionLimit;
                            break 'levels;
                        }
                        *to = site;
                    }
                    Err(Error::SizeOverflow) => {
                        report.stop = RefinementStop::DimensionLimit;
                        break 'levels;
                    }
                    Err(error) => return Err(error),
                }
            }
            let work = point_pairs(&current, cancel)?
                .checked_add(point_pairs(&next, cancel)?)
                .ok_or(Error::SizeOverflow)?;
            if work > u128::from(options.maximum_point_pairs - report.estimated_point_pairs) {
                report.stop = RefinementStop::WorkLimit;
                break;
            }
            let mut stable = true;
            let mut absolute = 0_f64;
            let mut relative = 0_f64;
            let mut compare = |i: usize, j: usize| -> Result<()> {
                cancel.check()?;
                let coarse = model.covariance(&current[i], &current[j])?;
                cancel.check()?;
                let fine = model.covariance(&next[i], &next[j])?;
                cancel.check()?;
                if !coarse.is_finite() || !fine.is_finite() {
                    return Err(Error::Numerical {
                        context: "quadrature covariance comparison",
                    });
                }
                report.covariance_evaluations = report
                    .covariance_evaluations
                    .checked_add(2)
                    .ok_or(Error::SizeOverflow)?;
                let scale = coarse.abs().max(fine.abs());
                // Normalize before subtracting when the unscaled difference
                // overflows. Do not overflow the tolerance product either.
                let difference = (fine - coarse).abs();
                let change = if scale == 0. {
                    0.
                } else if difference.is_finite() {
                    difference / scale
                } else {
                    (fine / scale - coarse / scale).abs()
                };
                absolute = absolute.max(difference);
                relative = relative.max(change);
                stable &= scale == 0.
                    || change <= options.relative_tolerance + options.absolute_tolerance / scale;
                Ok(())
            };
            // Visit block rows and their point-column transposes. Scanning every
            // point-point combination just to skip it would cost O(n^2) even
            // when a large point dataset contains only one block target.
            for (i, a) in current.iter().enumerate() {
                cancel.check()?;
                if a.support.is_point() {
                    continue;
                }
                for (j, b) in current.iter().enumerate() {
                    compare(i, j)?;
                    if b.support.is_point() {
                        compare(j, i)?;
                    }
                }
            }
            report.estimated_point_pairs += u64::try_from(work).map_err(|_| Error::SizeOverflow)?;
            report.completed_refinements += 1;
            report.maximum_absolute_change = Some(absolute);
            report.maximum_relative_change = Some(relative);
            report.stable_comparisons = if stable {
                report.stable_comparisons + 1
            } else {
                0
            };
            std::mem::swap(&mut current, &mut next);
            if report.stable_comparisons >= options.stable_comparisons {
                report.stop = RefinementStop::Stable;
                break;
            }
        }
    }
    Ok(RefinedSites {
        sites: current,
        report,
        _reservation: reservation,
    })
}
