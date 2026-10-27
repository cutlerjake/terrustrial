//! Resident, grouped spatial windows for shared multivariate local anisotropy.
//! Geometry chooses windows; attribute values never choose a partition. Call
//! separately for geological regions that must not share an anisotropy field.
use super::{
    field::{Edge, Window},
    fitting::DirectionalBin,
};
use crate::{
    data::{Observation, Point3, Site},
    error::{checked_len, invalid},
    execution::{Cancellation, MemoryBudget, Reservation},
    spatial::{Neighborhood, ObservationSource, SpatialIndex},
    variography::{
        indexed::{SampleIndex, Workspace},
        Direction, Sample,
    },
    Error, Result,
};

#[derive(Clone, Copy, Debug)]
pub struct Marginal {
    pub variable: usize,
    /// Fixed external or training-only prior, in the sample value units squared.
    pub structured_sill: f64,
    pub nugget: f64,
}
pub struct Options<'a> {
    /// Whole groups (e.g. drillholes) are kept together. Geometry median splits
    /// the longest axis of group centroids until this bound is met.
    pub maximum_groups: usize,
    /// Refuse a window exceeding this working set, including an oversized group.
    pub maximum_samples: usize,
    pub link_neighbors: usize,
    pub lag_edges: &'a [f64],
    pub directions: &'a [Direction],
    pub workspace_blocks: usize,
    pub budget: &'a MemoryBudget,
    pub cancellation: &'a Cancellation,
}
pub struct EvidenceWindow {
    pub position: Point3,
    pub groups: Vec<u64>,
    pub samples: usize,
    /// Each marginal is normalized as (gamma-nugget)/structured_sill. The
    /// shared-field solve uses sill=1 and nugget=0. Pair weights remain explicit.
    pub bins: Vec<DirectionalBin>,
    pub empty_bins: usize,
    pub below_nugget_bins: usize,
    pub zero_distance_bins: usize,
}
pub struct Evidence {
    windows: Vec<EvidenceWindow>,
    edges: Vec<Edge>,
    _reservation: Reservation,
}
impl Evidence {
    pub fn windows(&self) -> &[EvidenceWindow] {
        &self.windows
    }
    pub fn edges(&self) -> &[Edge] {
        &self.edges
    }
    /// Borrowed window descriptors. Their small Vec belongs to the caller.
    pub fn fitting_windows(&self) -> Vec<Window<'_>> {
        self.windows
            .iter()
            .map(|w| Window {
                position: w.position,
                bins: &w.bins,
                structured_sill: 1.,
                nugget: 0.,
            })
            .collect()
    }
    pub fn accounted_bytes(&self) -> u64 {
        self._reservation.bytes()
    }
}
struct Group {
    id: u64,
    first: usize,
    last: usize,
    center: [f64; 3],
}

/// Construct disjoint windows automatically, enumerate exact within-window
/// directional pairs using the shared spatial index, and form nearest-window
/// regularization links. All attributes contribute to a shared metric; this
/// does not fit a cross-sill matrix or independent per-attribute geometries.
///
/// The input is point-supported. Composite/block support must not be silently
/// passed as point evidence in a production acceptance study. Cone bins use the
/// directional axis times the observed mean distance as their representative
/// lag; cone-width approximation remains subject to validation. Overlapping
/// cones can count a pair more than once, so these are not independent likelihood
/// observations. Each group occurs in exactly one window, permitting whole-window
/// spatial validation without shared samples. All supplied marginal priors and
/// transforms must be independent of that validation evidence.
///
/// Resident geometry workspace is O(samples); each local pair index is bounded
/// by maximum_samples. Pair lists are never materialized. Returned bin and link
/// capacities retain their memory charge. Logical charges exclude allocator
/// bookkeeping and caller-owned input; this is not an out-of-core loader.
pub fn build(
    samples: &[Sample],
    group_ids: &[u64],
    marginals: &[Marginal],
    options: Options<'_>,
) -> Result<Evidence> {
    options.cancellation.check()?;
    if samples.is_empty()
        || samples.len() != group_ids.len()
        || marginals.is_empty()
        || options.maximum_groups == 0
        || options.maximum_samples == 0
        || options.link_neighbors == 0
        || options.directions.is_empty()
        || options.workspace_blocks == 0
        || options.lag_edges.is_empty()
        || options.lag_edges.iter().any(|h| !h.is_finite() || *h <= 0.)
        || options.lag_edges.windows(2).any(|h| h[0] >= h[1])
    {
        return Err(invalid("grouped local anisotropy evidence configuration"));
    }
    for (i, m) in marginals.iter().enumerate() {
        if !m.structured_sill.is_finite()
            || m.structured_sill <= 0.
            || !m.nugget.is_finite()
            || m.nugget < 0.
            || marginals[..i].iter().any(|a| a.variable == m.variable)
        {
            return Err(invalid("local anisotropy marginal priors"));
        }
    }
    let variables = marginals
        .iter()
        .map(|m| m.variable)
        .max()
        .unwrap()
        .checked_add(1)
        .ok_or(Error::SizeOverflow)?;
    for s in samples {
        if s.values.len() < variables || s.values.iter().flatten().any(|x| !x.is_finite()) {
            return Err(invalid("local anisotropy sample attributes"));
        }
    }
    let scratch = checked_len(&[samples.len(), 256])?
        .checked_add(checked_len(&[options.maximum_samples, variables, 16])?)
        .and_then(|n| {
            n.checked_add(
                options
                    .maximum_samples
                    .checked_mul(std::mem::size_of::<Sample>())?,
            )
        })
        .ok_or(Error::SizeOverflow)?;
    let _scratch = options
        .budget
        .reserve(u64::try_from(scratch).map_err(|_| Error::SizeOverflow)?)?;
    let mut order: Vec<_> = (0..samples.len()).collect();
    order.sort_unstable_by_key(|&i| (group_ids[i], i));
    let mut groups = Vec::new();
    let mut first = 0;
    while first < order.len() {
        options.cancellation.check()?;
        let id = group_ids[order[first]];
        let mut last = first + 1;
        while last < order.len() && group_ids[order[last]] == id {
            last += 1;
        }
        if last - first > options.maximum_samples {
            return Err(invalid(format!(
                "group {id} exceeds maximum_samples; refusing to split a validation group"
            )));
        }
        let mut center = [0.; 3];
        for &i in &order[first..last] {
            for (j, c) in center.iter_mut().enumerate() {
                *c += samples[i].point.coordinates()[j] / (last - first) as f64;
            }
        }
        Point3::new(center)?;
        groups.push(Group {
            id,
            first,
            last,
            center,
        });
        first = last;
    }
    let mut partitions = Vec::new();
    partition(
        &mut groups,
        0,
        options.maximum_groups,
        options.maximum_samples,
        &mut partitions,
        options.cancellation,
    )?;
    let bin_capacity = checked_len(&[
        marginals.len(),
        options.directions.len(),
        options.lag_edges.len(),
    ])?;
    let retained = checked_len(&[
        partitions.len(),
        bin_capacity,
        std::mem::size_of::<DirectionalBin>(),
    ])?
    .checked_add(checked_len(&[
        partitions.len(),
        std::mem::size_of::<EvidenceWindow>(),
    ])?)
    .and_then(|n| n.checked_add(groups.len().checked_mul(8)?))
    .and_then(|n| {
        n.checked_add(
            partitions
                .len()
                .checked_mul(options.link_neighbors)?
                .checked_mul(std::mem::size_of::<Edge>())?,
        )
    })
    .ok_or(Error::SizeOverflow)?;
    let reservation = options
        .budget
        .reserve(u64::try_from(retained).map_err(|_| Error::SizeOverflow)?)?;
    let mut windows = Vec::with_capacity(partitions.len());
    let mut workspace = Workspace::new(
        options.lag_edges.len(),
        options.workspace_blocks,
        options.budget,
    )?;
    for range in partitions {
        options.cancellation.check()?;
        let groups = &groups[range];
        let count = groups.iter().map(|g| g.last - g.first).sum::<usize>();
        let mut local = Vec::with_capacity(count);
        let mut center = [0.; 3];
        for g in groups {
            for &i in &order[g.first..g.last] {
                let s = &samples[i];
                for (j, c) in center.iter_mut().enumerate() {
                    *c += s.point.coordinates()[j] / count as f64;
                }
                local.push(Sample {
                    point: s.point,
                    values: s.values[..variables].to_vec(),
                });
            }
        }
        let index = SampleIndex::new(&local, options.budget, options.cancellation)?;
        let mut window = EvidenceWindow {
            position: Point3::new(center)?,
            groups: groups.iter().map(|g| g.id).collect(),
            samples: count,
            bins: Vec::with_capacity(bin_capacity),
            empty_bins: 0,
            below_nugget_bins: 0,
            zero_distance_bins: 0,
        };
        for direction in options.directions {
            for marginal in marginals {
                let variogram = index.experimental(
                    [marginal.variable; 2],
                    options.lag_edges,
                    Some(direction),
                    &mut workspace,
                    options.cancellation,
                )?;
                for bin in variogram.bins {
                    let (Some(h), Some(gamma)) = (bin.mean_distance, bin.semivariance) else {
                        window.empty_bins += 1;
                        continue;
                    };
                    if h == 0. {
                        window.zero_distance_bins += 1;
                        continue;
                    }
                    let normalized = (gamma - marginal.nugget) / marginal.structured_sill;
                    if !normalized.is_finite() {
                        return Err(Error::Numerical {
                            context: "local variogram marginal normalization",
                        });
                    }
                    if normalized < 0. {
                        window.below_nugget_bins += 1;
                        continue;
                    }
                    window.bins.push(DirectionalBin {
                        lag: direction.axis().map(|v| v * h),
                        semivariance: normalized,
                        pairs: bin.pairs,
                    });
                }
            }
        }
        windows.push(window);
    }
    let observations = windows
        .iter()
        .enumerate()
        .map(|(i, w)| Observation::new(i as u64, Site::point(w.position, 0), 0., 0.))
        .collect::<Result<Vec<_>>>()?;
    let index = SpatialIndex::with_budget(observations, options.budget)?;
    let mut edges = Vec::with_capacity(checked_len(&[windows.len(), options.link_neighbors])?);
    for (i, w) in windows.iter().enumerate() {
        let neighbors = index.nearest(
            w.position,
            Neighborhood {
                maximum: options.link_neighbors.saturating_add(1).min(windows.len()),
                radius: f64::MAX,
            },
            options.cancellation,
        )?;
        for n in neighbors
            .into_iter()
            .filter(|n| n.id() != i as u64)
            .take(options.link_neighbors)
        {
            let j = usize::try_from(n.id()).map_err(|_| Error::SizeOverflow)?;
            if i != j {
                edges.push(Edge {
                    nodes: [i.min(j), i.max(j)],
                    weight: 1.,
                });
            }
        }
    }
    edges.sort_unstable_by_key(|e| e.nodes);
    edges.dedup_by_key(|e| e.nodes);
    Ok(Evidence {
        windows,
        edges,
        _reservation: reservation,
    })
}
fn partition(
    groups: &mut [Group],
    offset: usize,
    maximum_groups: usize,
    maximum_samples: usize,
    out: &mut Vec<std::ops::Range<usize>>,
    cancel: &Cancellation,
) -> Result<()> {
    cancel.check()?;
    if groups.len() <= maximum_groups
        && groups.iter().map(|g| g.last - g.first).sum::<usize>() <= maximum_samples
    {
        out.push(offset..offset + groups.len());
        return Ok(());
    }
    let axis = (0..3)
        .max_by(|&a, &b| {
            let extent = |axis| {
                let lo = groups
                    .iter()
                    .map(|g| g.center[axis])
                    .fold(f64::INFINITY, f64::min);
                let hi = groups
                    .iter()
                    .map(|g| g.center[axis])
                    .fold(f64::NEG_INFINITY, f64::max);
                hi - lo
            };
            extent(a).total_cmp(&extent(b))
        })
        .unwrap();
    groups.sort_unstable_by(|a, b| {
        a.center[axis]
            .total_cmp(&b.center[axis])
            .then(a.id.cmp(&b.id))
    });
    let middle = groups.len() / 2;
    let (left, right) = groups.split_at_mut(middle);
    partition(left, offset, maximum_groups, maximum_samples, out, cancel)?;
    partition(
        right,
        offset + middle,
        maximum_groups,
        maximum_samples,
        out,
        cancel,
    )
}
