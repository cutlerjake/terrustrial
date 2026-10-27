//! Bounded deterministic ellipsoid selection with crossing variable/octant caps.
mod quotas;
use super::{Candidate, Neighborhood, ObservationSource};
use crate::{
    data::{norm3, Observation, Point3},
    error::{checked_len, invalid},
    execution::{Cancellation, MemoryBudget, Reservation},
    model::Anisotropy,
    Error, Result,
};
use quotas::QuotaWorkspace;
use std::{borrow::Cow, collections::BinaryHeap, mem::size_of};

#[derive(Clone, Copy, Debug)]
pub struct VariableLimit {
    pub variable: usize,
    pub maximum: usize,
}

/// Minimum count for one explicitly admitted variable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VariableMinimum {
    pub variable: usize,
    pub minimum: usize,
}
/// Lower bounds on the selected neighborhood, in addition to the existing caps.
/// Octant numbering follows principal-frame sign bits x + 2*y + 4*z.
#[derive(Clone, Debug, Default)]
pub struct MinimumQuotas {
    pub total: usize,
    pub variables: Vec<VariableMinimum>,
    pub octants: [usize; 8],
}
/// Diagnostics for the last selection. Counts describe retained candidates, not
/// all rows in the input. An incomplete scan or cancelled selection is not complete.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SearchDiagnostics {
    /// Number of attempted geometries, including the original ellipsoid.
    pub attempts: usize,
    /// Zero is the original geometry; one is the first expansion.
    pub level: usize,
    pub stop: Option<SearchStop>,
    pub retained_candidates: usize,
    pub selected: usize,
    pub minimums_satisfied: bool,
    pub complete: bool,
}

/// Why a completed neighborhood search stopped. Errors and cancellation have no stop.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SearchStop {
    Satisfied,
    Exhausted,
}

#[derive(Clone, Debug)]
struct Expansion {
    geometry: Anisotropy,
    extent: [f64; 3],
}
fn enclosing_extent(geometry: &Anisotropy) -> [f64; 3] {
    let scales = geometry.scales();
    let frame = geometry.frame();
    // F F^T differs from I by at most 1e-10 per entry. Its inverse
    // correction has norm below 3.1e-10, so a 1e-9 * largest-radius
    // margin encloses the actual accepted frame as well as rounding.
    let margin = scales.into_iter().fold(0.0, f64::max) * 1e-9;
    std::array::from_fn(|axis| {
        (norm3(std::array::from_fn(|i| frame[i][axis] * scales[i])) + margin).next_up()
    })
}

/// Validated search configuration. Radii are the anisotropy scales; distance <= 1
/// is inside. This search geometry does not change the covariance model.
///
/// Without minimum quotas, selection is greedy in (normalized distance, unsigned
/// record ID) order, subject to total, per-variable and per-octant maxima. This
/// default policy is not a maximum-cardinality assignment. `with_minimums` enables
/// exact feasible maximum-cardinality selection. An empty variable list admits every variable without a variable
/// cap; a nonempty list admits only listed variables. Zero variable/octant caps
/// exclude their observations. Octants use signs in the supplied principal frame;
/// exact zero (including -0) lies on the nonnegative side of each plane.
#[derive(Clone, Debug)]
pub struct SearchNeighborhood {
    pub(crate) geometry: Anisotropy,
    extent: [f64; 3],
    maximum: usize,
    octant_maximum: Option<usize>,
    variables: Vec<VariableLimit>,
    minimums: Option<MinimumQuotas>,
    expansion: Vec<Expansion>,
}
impl SearchNeighborhood {
    pub fn new(
        maximum: usize,
        geometry: Anisotropy,
        octant_maximum: Option<usize>,
        mut variables: Vec<VariableLimit>,
    ) -> Result<Self> {
        if maximum == 0 {
            return Err(invalid("positive total neighborhood maximum required"));
        }
        variables.sort_unstable_by_key(|v| v.variable);
        if variables.windows(2).any(|v| v[0].variable == v[1].variable) {
            return Err(invalid("duplicate neighborhood variable limit"));
        }
        let extent = enclosing_extent(&geometry);
        Ok(Self {
            geometry,
            extent,
            maximum,
            octant_maximum,
            variables,
            minimums: None,
            expansion: Vec::new(),
        })
    }
    /// Enable exact lower-bound selection. Among all subsets satisfying the
    /// minimums and maxima, maximize cardinality, then choose the lexicographically
    /// nearest subset in (normalized distance, unsigned ID) order. This can differ
    /// from the legacy greedy maximum-only policy even when a minimum is small.
    ///
    /// Variable minimums require corresponding entries in `variable_limits`.
    /// Impossible geometry returns `Error::InsufficientNeighborhood`, never a
    /// silently underfilled result. All-zero minimums retain the greedy policy.
    pub fn with_minimums(mut self, mut minimums: MinimumQuotas) -> Result<Self> {
        if minimums.total > self.maximum {
            return Err(invalid("neighborhood minimum exceeds total maximum"));
        }
        minimums.variables.sort_unstable_by_key(|v| v.variable);
        if minimums
            .variables
            .windows(2)
            .any(|v| v[0].variable == v[1].variable)
        {
            return Err(invalid("duplicate neighborhood variable minimum"));
        }
        let mut variable_total = 0_usize;
        for minimum in &minimums.variables {
            let class = self
                .variables
                .binary_search_by_key(&minimum.variable, |v| v.variable)
                .map_err(|_| invalid("variable minimum requires an explicit variable limit"))?;
            if minimum.minimum > self.variable_cap(class) {
                return Err(invalid("variable minimum exceeds maximum"));
            }
            variable_total = variable_total
                .checked_add(minimum.minimum)
                .ok_or(Error::SizeOverflow)?;
        }
        let mut octant_total = 0_usize;
        for &minimum in &minimums.octants {
            if minimum
                > self
                    .octant_maximum
                    .unwrap_or(self.maximum)
                    .min(self.maximum)
            {
                return Err(invalid("octant minimum exceeds maximum"));
            }
            octant_total = octant_total
                .checked_add(minimum)
                .ok_or(Error::SizeOverflow)?;
        }
        if variable_total > self.maximum || octant_total > self.maximum {
            return Err(invalid("neighborhood minimum sum exceeds total maximum"));
        }
        minimums.variables.retain(|v| v.minimum != 0);
        if minimums.total == 0 && minimums.variables.is_empty() && octant_total == 0 {
            if !self.expansion.is_empty() {
                return Err(invalid("adaptive search requires nonzero minimum quotas"));
            }
            self.minimums = None;
        } else {
            self.minimums = Some(minimums);
        }
        Ok(self)
    }
    /// Try the original ellipsoid, then these absolute principal-axis radii,
    /// stopping at the first geometry satisfying all minimum quotas. Each stage
    /// must enlarge at least one axis and shrink none; the frame remains fixed.
    /// The final stage is a hard limit, with `InsufficientNeighborhood` on failure.
    ///
    /// Nonempty expansion requires nonzero minimums to be configured first.
    /// An empty list removes expansion. Each attempt reuses the same budgeted
    /// buffers and repeats indexed candidate traversal. Sources must remain
    /// immutable for the entire search; SQLite binds all stages to one snapshot.
    pub fn with_expansion(mut self, scales: Vec<[f64; 3]>) -> Result<Self> {
        if !scales.is_empty() && self.minimums.is_none() {
            return Err(invalid("adaptive search requires nonzero minimum quotas"));
        }
        let mut expansion = Vec::new();
        expansion
            .try_reserve_exact(scales.len())
            .map_err(|_| invalid("adaptive search configuration allocation"))?;
        let mut previous = self.geometry.scales();
        for scales in scales {
            let geometry = Anisotropy::new(self.geometry.frame(), scales)?;
            if (0..3).any(|i| scales[i] < previous[i]) || scales == previous {
                return Err(invalid(
                    "each search expansion must enlarge at least one axis and shrink none",
                ));
            }
            expansion.push(Expansion {
                extent: enclosing_extent(&geometry),
                geometry,
            });
            previous = scales;
        }
        self.expansion = expansion;
        Ok(self)
    }
    /// Absolute principal-axis radii after the original geometry, in attempt order.
    pub fn expansion_scales(&self) -> impl ExactSizeIterator<Item = [f64; 3]> + '_ {
        self.expansion.iter().map(|step| step.geometry.scales())
    }
    fn geometry_at(&self, level: usize) -> &Anisotropy {
        if level == 0 {
            &self.geometry
        } else {
            &self.expansion[level - 1].geometry
        }
    }
    pub fn minimums(&self) -> Option<&MinimumQuotas> {
        self.minimums.as_ref()
    }
    fn variable_minimum(&self, class: usize) -> usize {
        let Some(minimums) = &self.minimums else {
            return 0;
        };
        let Some(variable) = self.variables.get(class) else {
            return 0;
        };
        minimums
            .variables
            .binary_search_by_key(&variable.variable, |v| v.variable)
            .map_or(0, |i| minimums.variables[i].minimum)
    }
    pub fn maximum(&self) -> usize {
        self.maximum
    }
    pub fn geometry(&self) -> &Anisotropy {
        &self.geometry
    }
    pub fn variable_limits(&self) -> &[VariableLimit] {
        &self.variables
    }
    pub fn octant_maximum(&self) -> Option<usize> {
        self.octant_maximum
    }
    fn sectors(&self) -> usize {
        if self.octant_maximum.is_some()
            || self
                .minimums
                .as_ref()
                .is_some_and(|m| m.octants.iter().any(|&n| n != 0))
        {
            8
        } else {
            1
        }
    }
    fn classes(&self) -> usize {
        self.variables.len().max(1)
    }
    fn variable_cap(&self, class: usize) -> usize {
        if self.variables.is_empty() {
            self.maximum
        } else {
            self.variables[class].maximum.min(self.maximum)
        }
    }
    fn bucket_cap(&self, class: usize) -> usize {
        self.variable_cap(class)
            .min(self.octant_maximum.unwrap_or(self.maximum))
    }
    fn class(&self, variable: usize) -> Option<usize> {
        if self.variables.is_empty() {
            Some(0)
        } else {
            self.variables
                .binary_search_by_key(&variable, |v| v.variable)
                .ok()
        }
    }
    /// Conservative enclosing box. The small inflation covers the accepted
    /// orthonormal-frame tolerance (1e-10 per Gram entry) and rounding. Infinite
    /// bounds are valid and cause an index to visit every intersecting entry.
    fn bounds(&self, point: Point3, level: usize) -> ([f64; 3], [f64; 3]) {
        let p = point.coordinates();
        let extent = if level == 0 {
            self.extent
        } else {
            self.expansion[level - 1].extent
        };
        (
            std::array::from_fn(|i| (p[i] - extent[i]).next_down()),
            std::array::from_fn(|i| (p[i] + extent[i]).next_up()),
        )
    }
    fn metric(&self, point: Point3, center: Point3, geometry: &Anisotropy) -> (f64, usize) {
        let (normalized, octant) =
            geometry.projected_displacement(center.coordinates(), point.coordinates());
        (
            norm3(normalized),
            if self.sectors() == 8 { octant } else { 0 },
        )
    }
}

#[derive(Clone, Copy, Debug)]
pub enum NeighborhoodQuery<'a> {
    Spherical(Neighborhood),
    Constrained(&'a SearchNeighborhood),
}
impl From<Neighborhood> for NeighborhoodQuery<'_> {
    fn from(value: Neighborhood) -> Self {
        Self::Spherical(value)
    }
}
impl NeighborhoodQuery<'_> {
    pub fn validate(self) -> Result<()> {
        match self {
            Self::Spherical(n) => n.validate(),
            Self::Constrained(_) => Ok(()),
        }
    }
}

/// Reusable, budgeted selection buffers, independent of source row count.
/// Retain at most min(total, variable cap, octant cap) per variable/octant bucket.
/// A discarded candidate cannot enter the greedy result: all earlier candidates
/// in its bucket have identical constraints, so either a shared cap already fills
/// or at least that many have been accepted. Cross-bucket caps are applied only
/// after globally sorting the retained candidates. For minimum quotas, any
/// discarded member can be exchanged with an earlier retained member in the same
/// bucket without changing constraint counts. This preserves feasibility and the
/// lexicographically best maximum-cardinality result. Quota flow scratch is also
/// preallocated and charged; quota selection allocates nothing after construction.
///
/// Source observation IDs must be unique. Returned observations borrow this
/// workspace until its next search. No record IDs or full input are retained
/// across queries. The reservation accounts for buffer payloads and the fixed
/// center metric cache, not allocator metadata, source caches, or the caller-owned
/// search configuration.
pub struct SearchWorkspace<'a> {
    query: &'a SearchNeighborhood,
    point: Point3,
    level: usize,
    geometry: &'a Anisotropy,
    buckets: Vec<BinaryHeap<Candidate>>,
    candidates: Vec<Candidate>,
    selected: Vec<Observation>,
    variable_counts: Vec<usize>,
    sector_counts: [usize; 8],
    quotas: Option<QuotaWorkspace>,
    diagnostics: SearchDiagnostics,
    probe_radius: f64,
    last_metric: Option<CenterMetric>,
    _reservation: Reservation,
}
#[derive(Clone, Copy)]
struct CenterMetric {
    center: [u64; 3],
    distance: f64,
    sector: usize,
}
impl<'a> SearchWorkspace<'a> {
    pub fn new(query: &'a SearchNeighborhood, budget: &MemoryBudget) -> Result<Self> {
        let buckets = checked_len(&[query.classes(), query.sectors()])?;
        let retained = (0..query.classes()).try_fold(0usize, |sum, class| {
            sum.checked_add(checked_len(&[query.bucket_cap(class), query.sectors()])?)
                .ok_or(Error::SizeOverflow)
        })?;
        let bytes = checked_len(&[retained, size_of::<Candidate>(), 2])?
            .checked_add(checked_len(&[buckets, size_of::<BinaryHeap<Candidate>>()])?)
            .and_then(|x| {
                x.checked_add(
                    query
                        .maximum
                        .min(retained)
                        .checked_mul(size_of::<Observation>())?,
                )
            })
            .and_then(|x| x.checked_add(query.classes().checked_mul(size_of::<usize>())?))
            .and_then(|x| x.checked_add(size_of::<Option<CenterMetric>>()))
            .ok_or(Error::SizeOverflow)?;
        let reservation = budget.reserve(u64::try_from(bytes).map_err(|_| Error::SizeOverflow)?)?;
        let mut heaps = Vec::new();
        heaps
            .try_reserve_exact(buckets)
            .map_err(|_| invalid("unable to allocate neighborhood buckets"))?;
        for class in 0..query.classes() {
            for _ in 0..query.sectors() {
                let mut heap = BinaryHeap::new();
                heap.try_reserve_exact(query.bucket_cap(class))
                    .map_err(|_| invalid("unable to allocate neighborhood candidates"))?;
                heaps.push(heap);
            }
        }
        let mut candidates = Vec::new();
        candidates
            .try_reserve_exact(retained)
            .map_err(|_| invalid("unable to allocate neighborhood merge buffer"))?;
        let mut selected = Vec::new();
        selected
            .try_reserve_exact(query.maximum.min(retained))
            .map_err(|_| invalid("unable to allocate neighborhood result"))?;
        let mut variable_counts = Vec::new();
        variable_counts
            .try_reserve_exact(query.classes())
            .map_err(|_| invalid("unable to allocate neighborhood counts"))?;
        variable_counts.resize(query.classes(), 0);
        let quotas = if query.minimums.is_some() {
            Some(QuotaWorkspace::new(
                query.classes(),
                query.sectors(),
                retained,
                budget,
            )?)
        } else {
            None
        };
        Ok(Self {
            query,
            point: Point3::new([0.0; 3])?,
            level: 0,
            geometry: &query.geometry,
            buckets: heaps,
            candidates,
            selected,
            variable_counts,
            sector_counts: [0; 8],
            quotas,
            diagnostics: SearchDiagnostics::default(),
            probe_radius: 0.125,
            last_metric: None,
            _reservation: reservation,
        })
    }
    pub fn accounted_bytes(&self) -> u64 {
        self._reservation.bytes()
            + self
                .quotas
                .as_ref()
                .map_or(0, QuotaWorkspace::accounted_bytes)
    }
    pub fn diagnostics(&self) -> SearchDiagnostics {
        self.diagnostics
    }
    pub(crate) fn reset(&mut self, point: Point3) {
        self.point = point;
        self.level = 0;
        self.geometry = &self.query.geometry;
        self.diagnostics = SearchDiagnostics::default();
        self.clear_selection();
    }
    fn clear_selection(&mut self) {
        // Reset also covers a new query point and every geometry expansion.
        self.last_metric = None;
        for heap in &mut self.buckets {
            heap.clear();
        }
        self.candidates.clear();
        self.selected.clear();
        self.variable_counts.fill(0);
        self.sector_counts.fill(0);
    }
    pub(crate) fn bounds(&self) -> ([f64; 3], [f64; 3]) {
        self.query.bounds(self.point, self.level)
    }
    pub(crate) fn offer(&mut self, observation: &Observation) -> Result<()> {
        let Some(class) = self.query.class(observation.site().variable) else {
            return Ok(());
        };
        let cap = self.query.bucket_cap(class);
        if cap == 0 {
            return Ok(());
        }
        let center = observation.site().support.center();
        let bits = center.coordinates().map(f64::to_bits);
        let (distance, sector) = match self.last_metric {
            Some(metric) if metric.center == bits => (metric.distance, metric.sector),
            _ => {
                let (distance, sector) = self.query.metric(self.point, center, self.geometry);
                if distance.is_nan() {
                    return Err(Error::Numerical {
                        context: "ellipsoid search distance",
                    });
                }
                // Colocated variables share geometry, but retain independent
                // identity, cap and heap processing. Bit keys preserve -0.0.
                self.last_metric = Some(CenterMetric {
                    center: bits,
                    distance,
                    sector,
                });
                (distance, sector)
            }
        };
        if distance > 1.0 {
            return Ok(());
        }
        let heap = &mut self.buckets[class * self.query.sectors() + sector];
        let candidate = Candidate {
            distance,
            observation: *observation,
        };
        if heap.len() < cap {
            heap.push(candidate);
        } else if heap.peek().is_some_and(|worst| candidate < *worst) {
            *heap.peek_mut().unwrap() = candidate;
        }
        Ok(())
    }
    #[cfg(feature = "sqlite")]
    pub(crate) fn has_expansion(&self) -> bool {
        !self.query.expansion.is_empty()
    }
    pub(crate) fn search_source<'w>(
        &'w mut self,
        source: &(impl ObservationSource + ?Sized),
        point: Point3,
        cancel: &Cancellation,
    ) -> Result<&'w [Observation]> {
        self.search_indexed(point, cancel, |lower, upper, workspace| {
            source.visit_bounds(lower, upper, cancel, &mut |o| workspace.offer(o))
        })
    }
    /// Internal index traversal can retain its read-side state while offering
    /// candidates directly. Public visitors still retain their reentrant contract.
    pub(crate) fn search_indexed<'w>(
        &'w mut self,
        point: Point3,
        cancel: &Cancellation,
        mut visit: impl FnMut([f64; 3], [f64; 3], &mut Self) -> Result<()>,
    ) -> Result<&'w [Observation]> {
        self.search_with(point, cancel, |workspace| {
            // A probe is exact only when every nonempty quota bucket is full
            // and its worst retained candidate lies within the enclosed metric
            // radius. All omitted candidates are then worse in their own bucket.
            // Missing variables/sectors fall back to the full configured extent.
            let mut radius = workspace.probe_radius;
            loop {
                cancel.check()?;
                let (lower, upper) = if radius >= 1.0 {
                    workspace.bounds()
                } else {
                    let extent = if workspace.level == 0 {
                        workspace.query.extent
                    } else {
                        workspace.query.expansion[workspace.level - 1].extent
                    };
                    let p = point.coordinates();
                    (
                        std::array::from_fn(|a| {
                            (p[a] - (extent[a] * radius).next_up()).next_down()
                        }),
                        std::array::from_fn(|a| (p[a] + (extent[a] * radius).next_up()).next_up()),
                    )
                };
                visit(lower, upper, workspace)?;
                let complete = workspace.buckets.iter().enumerate().all(|(i, heap)| {
                    let cap = workspace.query.bucket_cap(i / workspace.query.sectors());
                    cap == 0
                        || (heap.len() == cap && heap.peek().is_some_and(|c| c.distance < radius))
                });
                if radius >= 1.0 || complete {
                    // Reuse density only as a hint. The next query independently
                    // proves completeness, even across unrelated sources.
                    workspace.probe_radius = if complete {
                        workspace
                            .buckets
                            .iter()
                            .filter_map(|h| h.peek())
                            .map(|c| c.distance)
                            .fold(0.0, f64::max)
                            .mul_add(1.25, 1e-12)
                            .clamp(1e-6, 1.0)
                    } else {
                        1.0
                    };
                    return Ok(());
                }
                // Restart, rather than offering duplicate observations from
                // overlapping boxes into the bounded heaps.
                workspace.clear_selection();
                radius = (radius * 2.0).min(1.0);
            }
        })
    }
    pub(crate) fn search_scan<'w>(
        &'w mut self,
        source: &(impl ObservationSource + ?Sized),
        point: Point3,
        cancel: &Cancellation,
    ) -> Result<&'w [Observation]> {
        // A contiguous scan already computes the ellipsoid membership in offer.
        // Avoid a redundant per-row bounding-box test on this path.
        self.search_with(point, cancel, |workspace| {
            source.scan(&mut |o| {
                cancel.check()?;
                workspace.offer(o)
            })
        })
    }
    fn search_with<'w>(
        &'w mut self,
        point: Point3,
        cancel: &Cancellation,
        mut visit: impl FnMut(&mut Self) -> Result<()>,
    ) -> Result<&'w [Observation]> {
        self.reset(point);
        loop {
            cancel.check()?;
            self.diagnostics = SearchDiagnostics {
                attempts: self.level + 1,
                level: self.level,
                ..Default::default()
            };
            visit(self)?;
            match self.finish(cancel) {
                Ok(()) => return Ok(&self.selected),
                Err(Error::InsufficientNeighborhood) if self.level < self.query.expansion.len() => {
                    self.level += 1;
                    self.geometry = self.query.geometry_at(self.level);
                    self.clear_selection();
                    // The previous stage was infeasible, but the overall search
                    // is still in progress if cancellation arrives between stages.
                    self.diagnostics.complete = false;
                    self.diagnostics.stop = None;
                }
                Err(error) => return Err(error),
            }
        }
    }
    fn finish(&mut self, cancel: &Cancellation) -> Result<()> {
        cancel.check()?;
        for heap in &mut self.buckets {
            self.candidates.extend(heap.drain());
        }
        self.candidates.sort_unstable();
        self.diagnostics.retained_candidates = self.candidates.len();
        if let Some(quotas) = &mut self.quotas {
            let result = quotas.select(
                self.query,
                self.point,
                self.geometry,
                &self.candidates,
                &mut self.selected,
                &mut self.variable_counts,
                &mut self.sector_counts,
                cancel,
            );
            match result {
                Ok(()) => {
                    self.diagnostics.selected = self.selected.len();
                    self.diagnostics.minimums_satisfied = true;
                    self.diagnostics.complete = true;
                    self.diagnostics.stop = Some(SearchStop::Satisfied);
                    return Ok(());
                }
                Err(Error::InsufficientNeighborhood) => {
                    self.diagnostics.complete = true;
                    self.diagnostics.stop = Some(SearchStop::Exhausted);
                    return Err(Error::InsufficientNeighborhood);
                }
                Err(error) => return Err(error),
            }
        }
        if self.query.sectors() == 1 {
            // Without octant caps or minimums, each variable's single heap
            // already enforces its cap. Any prefix of their sorted union also
            // obeys those caps, so only the total maximum remains to apply.
            for candidate in self.candidates.iter().take(self.query.maximum) {
                cancel.check()?;
                self.selected.push(candidate.observation);
            }
        } else {
            for candidate in &self.candidates {
                cancel.check()?;
                let class = self
                    .query
                    .class(candidate.observation.site().variable)
                    .unwrap();
                let (_, sector) = self.query.metric(
                    self.point,
                    candidate.observation.site().support.center(),
                    self.geometry,
                );
                if self.variable_counts[class] >= self.query.variable_cap(class)
                    || self.sector_counts[sector]
                        >= self.query.octant_maximum.unwrap_or(self.query.maximum)
                {
                    continue;
                }
                self.selected.push(candidate.observation);
                self.variable_counts[class] += 1;
                self.sector_counts[sector] += 1;
                if self.selected.len() == self.query.maximum {
                    break;
                }
            }
        }
        self.diagnostics.selected = self.selected.len();
        self.diagnostics.minimums_satisfied = true;
        self.diagnostics.complete = true;
        self.diagnostics.stop = Some(SearchStop::Satisfied);
        Ok(())
    }
}

// Persistent per-worker state. Keep the workspace inline without an extra
// unaccounted heap allocation just to equalize enum variant sizes.
#[allow(clippy::large_enum_variant)]
pub(crate) enum QueryWorkspace<'a> {
    Spherical(super::NearestWorkspace),
    Constrained(SearchWorkspace<'a>),
}
impl<'a> QueryWorkspace<'a> {
    pub(crate) fn new(query: NeighborhoodQuery<'a>, budget: &MemoryBudget) -> Result<Self> {
        query.validate()?;
        match query {
            NeighborhoodQuery::Spherical(query) => Ok(Self::Spherical(
                super::NearestWorkspace::new(query, budget)?,
            )),
            NeighborhoodQuery::Constrained(query) => {
                Ok(Self::Constrained(SearchWorkspace::new(query, budget)?))
            }
        }
    }
    pub(crate) fn select(
        &mut self,
        source: &(impl ObservationSource + ?Sized),
        point: Point3,
        cancel: &Cancellation,
    ) -> Result<Cow<'_, [Observation]>> {
        match self {
            Self::Spherical(workspace) => source
                .nearest_into(point, workspace, cancel)
                .map(Cow::Borrowed),
            Self::Constrained(workspace) => {
                source.search(point, workspace, cancel).map(Cow::Borrowed)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::Site;

    fn row(id: u64, coordinates: [f64; 3], variable: usize) -> Observation {
        Observation::new(
            id,
            Site::point(Point3::new(coordinates).unwrap(), variable),
            if id % 2 == 0 { -0.0 } else { id as f64 },
            0.0,
        )
        .unwrap()
    }

    // Full scalar materialization, without bounded heaps or metric reuse.
    fn reference(query: &SearchNeighborhood, point: Point3, rows: &[Observation]) -> Vec<u64> {
        let mut candidates = Vec::new();
        for observation in rows {
            let Some(class) = query.class(observation.site().variable) else {
                continue;
            };
            if query.bucket_cap(class) == 0 {
                continue;
            }
            let (projected, sector) = query.geometry.projected_displacement(
                observation.site().support.center().coordinates(),
                point.coordinates(),
            );
            let distance = norm3(projected);
            assert!(!distance.is_nan());
            if distance <= 1.0 {
                candidates.push((
                    distance,
                    observation.id(),
                    class,
                    if query.sectors() == 8 { sector } else { 0 },
                ));
            }
        }
        candidates.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        let mut counts = vec![0; query.classes()];
        let mut sectors = [0; 8];
        let mut selected = Vec::new();
        for (_, id, class, sector) in candidates {
            if counts[class] < query.variable_cap(class)
                && sectors[sector] < query.octant_maximum.unwrap_or(query.maximum)
            {
                selected.push(id);
                counts[class] += 1;
                sectors[sector] += 1;
                if selected.len() == query.maximum {
                    break;
                }
            }
        }
        selected
    }

    fn compare(selected: &[Observation], expected: &[u64], source: &[Observation]) {
        assert_eq!(
            selected.iter().map(Observation::id).collect::<Vec<_>>(),
            expected
        );
        for actual in selected {
            let original = source.iter().find(|row| row.id() == actual.id()).unwrap();
            assert_eq!(actual.value().to_bits(), original.value().to_bits());
            assert_eq!(
                actual.noise_variance().to_bits(),
                original.noise_variance().to_bits()
            );
            assert_eq!(actual.site().variable, original.site().variable);
            assert_eq!(
                actual
                    .site()
                    .support
                    .center()
                    .coordinates()
                    .map(f64::to_bits),
                original
                    .site()
                    .support
                    .center()
                    .coordinates()
                    .map(f64::to_bits)
            );
        }
    }

    #[test]
    fn colocated_metrics_match_scalar_selection_across_layouts_and_extremes() {
        let cancel = Cancellation::default();
        let budget = MemoryBudget::new(1 << 20);
        let centers = [
            [0.0; 3],
            [-0.0, 0.0, -0.0],
            [0.5, 0.25, -0.125],
            [-0.5, -0.25, 0.125],
            [2.0, -1.0, 0.5],
            [-2.0, 1.0, -0.5],
            [1.2e308, 0.0, 0.0],
            [-1.2e308, 0.0, 0.0],
            [2e-200, -1e-200, 0.0],
            [-2e-200, 1e-200, -0.0],
        ];
        let mut rows: Vec<_> = centers
            .iter()
            .enumerate()
            .flat_map(|(center, &coordinates)| {
                (0..8)
                    .map(move |variable| row((center * 8 + variable) as u64, coordinates, variable))
            })
            .collect();
        for scales in [[4., 2., 1.], [4e-200, 2e-200, 1e-200], [1.7e308; 3]] {
            let query = SearchNeighborhood::new(
                9,
                Anisotropy::new([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]], scales).unwrap(),
                Some(3),
                (0..7)
                    .map(|variable| VariableLimit {
                        variable,
                        maximum: if variable == 6 { 0 } else { 2 },
                    })
                    .collect(),
            )
            .unwrap();
            let mut workspace = SearchWorkspace::new(&query, &budget).unwrap();
            for center in [
                [0.; 3],
                [-0.0, 0.0, -0.0],
                [0.5, 0.25, -0.125],
                [-1.2e308, 0., 0.],
            ] {
                let point = Point3::new(center).unwrap();
                let expected = reference(&query, point, &rows);
                for layout in 0..3 {
                    match layout {
                        0 => rows.sort_unstable_by_key(Observation::id),
                        1 => rows.sort_unstable_by_key(|row| (row.site().variable, row.id())),
                        _ => rows.reverse(),
                    }
                    compare(
                        workspace
                            .search_scan(rows.as_slice(), point, &cancel)
                            .unwrap(),
                        &expected,
                        &rows,
                    );
                    compare(
                        workspace
                            .search_source(rows.as_slice(), point, &cancel)
                            .unwrap(),
                        &expected,
                        &rows,
                    );
                }
            }
        }
        assert_eq!(budget.used(), 0);
    }

    #[test]
    fn variable_only_caps_match_scalar_greedy_across_layouts_and_indexes() {
        let cancel = Cancellation::default();
        let budget = MemoryBudget::new(1 << 20);
        let mut rows = vec![
            row(u64::MAX, [1., 0., 0.], 1),
            row(1 << 63, [-1., 0., 0.], 0),
            row(5, [0., 1., 0.], 1),
            row(1, [0., -1., 0.], 0),
            row(3, [1., 0., 0.], 0),
            row(9, [-1., 0., 0.], 1),
            row(0, [0.; 3], 2),
            row(2, [0.; 3], 3),
            row(11, [2., 0., 0.], 1),
            row(4, [8., 0., 0.], 1),
        ];
        for maximum in [1, 3, 8] {
            for limited in [false, true] {
                let limits = if limited {
                    vec![
                        VariableLimit {
                            variable: 0,
                            maximum: 1,
                        },
                        VariableLimit {
                            variable: 1,
                            maximum: 3,
                        },
                        VariableLimit {
                            variable: 2,
                            maximum: 0,
                        },
                    ]
                } else {
                    vec![]
                };
                let query = SearchNeighborhood::new(
                    maximum,
                    Anisotropy::isotropic(4.).unwrap(),
                    None,
                    limits,
                )
                .unwrap();
                let mut workspace = SearchWorkspace::new(&query, &budget).unwrap();
                for coordinates in [[0.; 3], [0.5, 0., 0.], [-2., 0., 0.]] {
                    let point = Point3::new(coordinates).unwrap();
                    let expected = reference(&query, point, &rows);
                    if limited && maximum == 3 && coordinates == [0.; 3] {
                        assert_eq!(expected, [1, 5, 9]);
                    }
                    for layout in 0..3 {
                        match layout {
                            0 => rows.sort_unstable_by_key(Observation::id),
                            1 => rows.sort_unstable_by_key(|row| (row.site().variable, row.id())),
                            _ => rows.reverse(),
                        }
                        compare(
                            workspace
                                .search_scan(rows.as_slice(), point, &cancel)
                                .unwrap(),
                            &expected,
                            &rows,
                        );
                        let index = crate::spatial::SpatialIndex::new(rows.clone()).unwrap();
                        compare(
                            index.search(point, &mut workspace, &cancel).unwrap(),
                            &expected,
                            &rows,
                        );
                        assert_eq!(workspace.diagnostics().selected, expected.len());
                        assert_eq!(workspace.diagnostics().stop, Some(SearchStop::Satisfied));
                    }
                }
            }
        }
        assert_eq!(budget.used(), 0);
    }

    #[test]
    fn colocated_metrics_reset_for_query_points_and_adaptive_geometries() {
        let cancel = Cancellation::default();
        let budget = MemoryBudget::new(1 << 20);
        let query = SearchNeighborhood::new(7, Anisotropy::isotropic(1.).unwrap(), None, vec![])
            .unwrap()
            .with_minimums(MinimumQuotas {
                total: 7,
                ..Default::default()
            })
            .unwrap()
            .with_expansion(vec![[2.; 3], [4.; 3]])
            .unwrap();
        let rows: Vec<_> = (0..7)
            .map(|variable| row(variable as u64, [3., 0., 0.], variable))
            .collect();
        let mut workspace = SearchWorkspace::new(&query, &budget).unwrap();
        for (center, level) in [
            ([0.; 3], 2),
            ([3., 0., 0.], 0),
            ([1.5, 0., 0.], 1),
            ([0.; 3], 2),
        ] {
            let point = Point3::new(center).unwrap();
            for indexed in [false, true] {
                let selected = if indexed {
                    workspace
                        .search_source(rows.as_slice(), point, &cancel)
                        .unwrap()
                } else {
                    workspace
                        .search_scan(rows.as_slice(), point, &cancel)
                        .unwrap()
                };
                compare(selected, &[0, 1, 2, 3, 4, 5, 6], &rows);
                assert_eq!(workspace.diagnostics().level, level);
                assert_eq!(workspace.diagnostics().attempts, level + 1);
                assert_eq!(workspace.diagnostics().stop, Some(SearchStop::Satisfied));
            }
        }
    }

    #[test]
    fn colocated_metrics_allocate_nothing_and_keep_budget_accounting() {
        let budget = MemoryBudget::new(1 << 20);
        let query =
            SearchNeighborhood::new(7, Anisotropy::isotropic(4.).unwrap(), None, vec![]).unwrap();
        let rows: Vec<_> = (0..70)
            .map(|id| row(id, [(id / 7) as f64, 0., 0.], (id % 7) as usize))
            .collect();
        let mut workspace = SearchWorkspace::new(&query, &budget).unwrap();
        let bytes = workspace.accounted_bytes();
        assert_eq!(budget.used(), bytes);
        let short = MemoryBudget::new(bytes - 1);
        assert!(matches!(
            SearchWorkspace::new(&query, &short),
            Err(Error::MemoryBudget { .. })
        ));
        assert_eq!(short.used(), 0);
        let cancel = Cancellation::default();
        let allocations = allocation_counter::measure(|| {
            for center in [[0.; 3], [2., 0., 0.], [-0.0, 0.0, -0.0]] {
                let selected = workspace
                    .search_scan(rows.as_slice(), Point3::new(center).unwrap(), &cancel)
                    .unwrap();
                assert_eq!(selected.len(), 7);
            }
        });
        assert_eq!(allocations.count_total, 0);
        assert_eq!(budget.used(), bytes);
        drop(workspace);
        assert_eq!(budget.used(), 0);
    }
}
