//! Stable neighborhood selection and resident spatial indexing.
use crate::{
    data::{Observation, Point3, Site},
    error::invalid,
    execution::Cancellation,
    Error, Result,
};
use foldhash::{HashSet, HashSetExt};
use hierarchy::{Bounded, Bounds, Hierarchy};
use std::{cmp::Ordering, collections::BinaryHeap};

pub(crate) mod exact;
pub mod hierarchy;
pub(crate) mod nearest;
#[cfg(feature = "sqlite")]
pub(crate) mod paged;
mod traversal;
pub(crate) use traversal::visit_envelope;
pub mod search;
pub(crate) use nearest::visit_tree;
pub use nearest::NearestWorkspace;
pub use search::{
    MinimumQuotas, NeighborhoodQuery, SearchDiagnostics, SearchNeighborhood, SearchStop,
    SearchWorkspace, VariableLimit, VariableMinimum,
};

#[derive(Clone, Copy, Debug)]
pub struct Neighborhood {
    pub maximum: usize,
    pub radius: f64,
}
impl Neighborhood {
    pub fn validate(self) -> Result<()> {
        if self.maximum == 0 || !self.radius.is_finite() || self.radius <= 0.0 {
            return Err(invalid(
                "positive maximum and finite positive search radius required",
            ));
        }
        Ok(())
    }
}

/// Sources can stream without materializing the complete dataset. Selection is by
/// support-center distance, with stable record IDs breaking ties.
pub trait ObservationSource {
    /// Whether this source participates in the execution driver's deferred
    /// packed-page reads. Wrappers must forward this capability to their source.
    fn deferred_page_reads(&self) -> bool {
        false
    }
    fn scan(&self, visitor: &mut dyn FnMut(&Observation) -> Result<()>) -> Result<()>;
    /// Visit candidates whose support centers lie in a closed Cartesian box.
    /// Implementations may return extra candidates, but must not omit any inside.
    /// This composable traversal lets path-filtered overlays share exact ranking.
    fn visit_bounds(
        &self,
        lower: [f64; 3],
        upper: [f64; 3],
        cancel: &Cancellation,
        visitor: &mut dyn FnMut(&Observation) -> Result<()>,
    ) -> Result<()> {
        cancel.check()?;
        self.scan(&mut |observation| {
            cancel.check()?;
            let p = observation.site().support.center().coordinates();
            if (0..3).all(|j| p[j] >= lower[j] && p[j] <= upper[j]) {
                visitor(observation)?;
            }
            Ok(())
        })
    }
    /// Deterministic constrained selection with reusable bounded workspace.
    /// The default is a scan oracle; indexed sources override candidate pruning.
    fn search<'a>(
        &self,
        point: Point3,
        workspace: &'a mut SearchWorkspace<'_>,
        cancel: &Cancellation,
    ) -> Result<&'a [Observation]> {
        workspace.search_source(self, point, cancel)
    }
    fn at_site(&self, site: &Site) -> Result<Vec<Observation>> {
        let mut found = Vec::new();
        self.scan(&mut |observation| {
            if observation.site() == site {
                found.push(*observation);
            }
            Ok(())
        })?;
        found.sort_by_key(Observation::id);
        Ok(found)
    }
    /// Exact-site matches plus an optional proof that the entire support center
    /// is empty. `true` requires a complete successful query proving that no
    /// observation of any variable or support has that center. An empty exact
    /// result alone is insufficient. The proof is valid only for this source
    /// view; wrappers must include every overlay before forwarding it.
    /// The returned observations and their order must match `at_site(site)`.
    /// Supplying a proof permits callers to skip further exact queries at the
    /// same center while the source view remains unchanged.
    ///
    /// The default preserves custom `at_site` behavior and supplies no proof.
    fn at_site_with_center_status(&self, site: &Site) -> Result<(Vec<Observation>, bool)> {
        Ok((self.at_site(site)?, false))
    }
    fn nearest(
        &self,
        point: Point3,
        query: Neighborhood,
        cancel: &Cancellation,
    ) -> Result<Vec<Observation>> {
        query.validate()?;
        cancel.check()?;
        let mut heap = BinaryHeap::with_capacity(query.maximum);
        self.scan(&mut |observation| {
            cancel.check()?;
            offer(&mut heap, point, query, observation);
            Ok(())
        })?;
        Ok(finish(heap))
    }
    /// Reuse bounded selection storage. The default preserves custom nearest()
    /// overrides; indexed sources can also eliminate traversal allocations.
    fn nearest_into<'a>(
        &self,
        point: Point3,
        workspace: &'a mut NearestWorkspace,
        cancel: &Cancellation,
    ) -> Result<&'a [Observation]> {
        let found = self.nearest(point, workspace.query(), cancel)?;
        workspace.reset(point);
        for observation in &found {
            workspace.offer(observation);
        }
        workspace.finish(cancel)
    }
}
pub trait ObservationSink {
    fn append(&mut self, observation: Observation) -> Result<()>;
}

#[derive(Clone, Debug)]
pub(crate) struct Candidate {
    pub(crate) distance: f64,
    observation: Observation,
}
impl PartialEq for Candidate {
    fn eq(&self, rhs: &Self) -> bool {
        self.distance == rhs.distance && self.observation.id() == rhs.observation.id()
    }
}
impl Eq for Candidate {}
impl PartialOrd for Candidate {
    fn partial_cmp(&self, rhs: &Self) -> Option<Ordering> {
        Some(self.cmp(rhs))
    }
}
impl Ord for Candidate {
    fn cmp(&self, rhs: &Self) -> Ordering {
        self.distance
            .total_cmp(&rhs.distance)
            .then_with(|| self.observation.id().cmp(&rhs.observation.id()))
    }
}
// The heap stores an unsquared, rounded distance. Squaring it can round below
// an equal-distance candidate's squared key, dropping a lower-ID tie. Expand the
// cutoff before squaring, then round outward again; final offer() still applies
// the exact distance/radius predicate. Infinite bounds merely disable pruning.
pub(crate) fn squared_search_bound(distance: f64) -> f64 {
    let outward = distance.next_up();
    (outward * outward).next_up()
}

pub(crate) fn offer(
    heap: &mut BinaryHeap<Candidate>,
    point: Point3,
    query: Neighborhood,
    observation: &Observation,
) -> bool {
    let distance = point.distance(observation.site().support.center());
    offer_distance(heap, query, observation, distance)
}

pub(crate) fn offer_distance(
    heap: &mut BinaryHeap<Candidate>,
    query: Neighborhood,
    observation: &Observation,
    distance: f64,
) -> bool {
    if distance > query.radius {
        return false;
    }
    if heap.len() < query.maximum {
        heap.push(Candidate {
            distance,
            observation: *observation,
        });
        true
    } else if let Some(mut worst) = heap.peek_mut() {
        if distance
            .total_cmp(&worst.distance)
            .then(observation.id().cmp(&worst.observation.id()))
            .is_lt()
        {
            *worst = Candidate {
                distance,
                observation: *observation,
            };
            true
        } else {
            false
        }
    } else {
        false
    }
}
pub(crate) fn finish(heap: BinaryHeap<Candidate>) -> Vec<Observation> {
    heap.into_sorted_vec()
        .into_iter()
        .map(|x| x.observation)
        .collect()
}

/// Resident index with a packed immutable base and logarithmically merged append
/// segments. Known simulation targets use immutable prefix-filtered windows.
/// Unknown inserts never modify the base hierarchy or degrade its split quality.
struct ResidentTree {
    tree: Hierarchy<Observation>,
    _reservation: crate::execution::Reservation,
}
impl std::ops::Deref for ResidentTree {
    type Target = Hierarchy<Observation>;
    fn deref(&self) -> &Self::Target {
        &self.tree
    }
}

pub struct SpatialIndex {
    tree: std::sync::Arc<ResidentTree>,
    bounds: Option<Bounds>,
    runs: Vec<std::sync::Arc<ResidentTree>>,
    pending: Vec<Observation>,
    ids: HashSet<u64>,
    budget: crate::execution::MemoryBudget,
    reservation: crate::execution::Reservation,
}
impl SpatialIndex {
    pub fn new(observations: Vec<Observation>) -> Result<Self> {
        Self::with_budget(observations, &crate::execution::MemoryBudget::new(u64::MAX))
    }
    /// Account for resident entries, hierarchy nodes, identities, append buffers,
    /// and overlapping merge allocations. Allocator bookkeeping is excluded.
    pub fn with_budget(
        observations: Vec<Observation>,
        budget: &crate::execution::MemoryBudget,
    ) -> Result<Self> {
        let bytes = observations
            .capacity()
            .checked_mul(std::mem::size_of::<Observation>())
            .and_then(|n| n.checked_add(observations.len().checked_mul(64)?))
            .and_then(|n| {
                n.checked_add(Hierarchy::<Observation>::scratch_bytes(observations.len()).ok()?)
            })
            .and_then(|n| n.checked_add(32 * std::mem::size_of::<Observation>()))
            .and_then(|n| {
                n.checked_add(Hierarchy::<Observation>::node_bytes(observations.len()).ok()?)
            })
            .ok_or(Error::SizeOverflow)?;
        let mut reservation = budget.reserve(bytes as u64)?;
        let mut ids = HashSet::new();
        ids.try_reserve(observations.len())
            .map_err(|_| invalid("spatial identity allocation"))?;
        for obs in &observations {
            if !ids.insert(obs.id()) {
                return Err(invalid("duplicate observation ID"));
            }
        }
        let tree = Hierarchy::new(observations)?;
        let bounds = tree.bounds();
        let tree_charge = reservation.split(tree.allocated_bytes() as u64)?;
        let tree = std::sync::Arc::new(ResidentTree {
            tree,
            _reservation: tree_charge,
        });
        let mut pending = Vec::new();
        pending
            .try_reserve_exact(32)
            .map_err(|_| invalid("spatial append buffer allocation"))?;
        let mut out = Self {
            tree,
            bounds,
            runs: Vec::new(),
            pending,
            ids,
            budget: budget.clone(),
            reservation,
        };
        out.reservation.resize(out.allocated_bytes() as u64)?;
        Ok(out)
    }
    fn record_bounds(&mut self, observation: &Observation) {
        if let Some(bounds) = &mut self.bounds {
            bounds.include(observation.bounds());
        } else {
            self.bounds = Some(observation.bounds());
        }
    }
    fn allocated_bytes(&self) -> usize {
        self.runs.capacity() * std::mem::size_of::<std::sync::Arc<ResidentTree>>()
            + self.pending.capacity() * std::mem::size_of::<Observation>()
            + self.ids.capacity() * 32
    }
    /// Immutable view sharing tree allocations. Only the bounded append tail and
    /// logarithmic segment handles are copied; retained segments keep their charge.
    pub fn snapshot(&self) -> Result<ResidentSnapshot> {
        let bytes = self.pending.len() * std::mem::size_of::<Observation>()
            + self.runs.len() * std::mem::size_of::<std::sync::Arc<ResidentTree>>()
            + 256;
        let reservation = self.budget.reserve(bytes as u64)?;
        let mut pending = Vec::new();
        pending
            .try_reserve_exact(self.pending.len())
            .map_err(|_| invalid("resident snapshot tail allocation"))?;
        pending.extend_from_slice(&self.pending);
        let mut runs = Vec::new();
        runs.try_reserve_exact(self.runs.len())
            .map_err(|_| invalid("resident snapshot segments allocation"))?;
        runs.extend(self.runs.iter().cloned());
        Ok(ResidentSnapshot {
            count: self.len(),
            index: SpatialIndex {
                tree: self.tree.clone(),
                bounds: self.bounds,
                runs,
                pending,
                ids: HashSet::new(),
                budget: self.budget.clone(),
                reservation,
            },
        })
    }
    pub fn len(&self) -> usize {
        self.ids.len()
    }
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }
    fn visit_filtered(
        &self,
        workspace: &mut NearestWorkspace,
        cancel: &Cancellation,
        select: impl Fn(&Observation) -> Option<&Observation>,
    ) -> Result<()> {
        // Recent simulated values are often spatially close to the next target.
        // Offer them first so the exact distance cutoff tightens early.
        for entry in self.pending.iter().rev() {
            cancel.check()?;
            if let Some(o) = select(entry) {
                workspace.offer(o);
            }
        }
        for tree in self.runs.iter().rev() {
            visit_tree(tree, workspace, cancel, &select)?;
        }
        visit_tree(&self.tree, workspace, cancel, &select)?;
        Ok(())
    }
}
impl ObservationSource for SpatialIndex {
    fn nearest_into<'a>(
        &self,
        point: Point3,
        workspace: &'a mut NearestWorkspace,
        cancel: &Cancellation,
    ) -> Result<&'a [Observation]> {
        let probe = workspace.locality_radius(point);
        workspace.reset(point);
        if let Some(radius) = probe {
            workspace.restrict_radius(radius);
            self.visit_filtered(workspace, cancel, |entry| Some(entry))?;
            if workspace.filled_within(radius) {
                return workspace.finish(cancel);
            }
            workspace.reset(point);
        }
        self.visit_filtered(workspace, cancel, |entry| Some(entry))?;
        workspace.finish(cancel)
    }
    fn visit_bounds(
        &self,
        lower: [f64; 3],
        upper: [f64; 3],
        cancel: &Cancellation,
        visitor: &mut dyn FnMut(&Observation) -> Result<()>,
    ) -> Result<()> {
        cancel.check()?;
        let bounds = Bounds { lower, upper };
        if (0..3).any(|a| lower[a].is_nan() || upper[a].is_nan() || lower[a] > upper[a]) {
            return Err(invalid("spatial query bounds"));
        }
        if self.bounds.is_none_or(|b| !b.intersects(bounds)) {
            return Ok(());
        }
        let mut offer = |entry: &Observation| {
            cancel.check()?;
            visitor(entry)
        };
        self.tree.visit_bounds(bounds, &mut offer)?;
        for tree in &self.runs {
            tree.visit_bounds(bounds, &mut offer)?;
        }
        for entry in &self.pending {
            if entry.bounds().intersects(bounds) {
                offer(entry)?;
            }
        }
        Ok(())
    }
    fn scan(&self, visitor: &mut dyn FnMut(&Observation) -> Result<()>) -> Result<()> {
        for entry in self
            .tree
            .iter()
            .chain(self.runs.iter().flat_map(|t| t.iter()))
            .chain(&self.pending)
        {
            visitor(entry)?;
        }
        Ok(())
    }
    fn at_site(&self, site: &Site) -> Result<Vec<Observation>> {
        let p = site.support.center().coordinates();
        let mut found = Vec::new();
        let bounds = Bounds::point(p);
        if self.bounds.is_none_or(|b| !b.intersects(bounds)) {
            return Ok(found);
        }
        let mut visit = |entry: &Observation| {
            if entry.site() == site {
                found.push(*entry);
            }
            Ok(())
        };
        self.tree.visit_bounds(bounds, &mut visit)?;
        for tree in &self.runs {
            tree.visit_bounds(bounds, &mut visit)?;
        }
        for entry in &self.pending {
            visit(entry)?;
        }
        found.sort_unstable_by_key(Observation::id);
        Ok(found)
    }
    fn nearest(
        &self,
        point: Point3,
        query: Neighborhood,
        cancel: &Cancellation,
    ) -> Result<Vec<Observation>> {
        let budget = crate::execution::MemoryBudget::new(u64::MAX);
        let mut workspace = NearestWorkspace::new(query, &budget)?;
        self.nearest_into(point, &mut workspace, cancel)?;
        Ok(workspace.into_selected())
    }
}
/// Immutable resident view. Appends and merges on its source cannot alter it.
pub struct ResidentSnapshot {
    index: SpatialIndex,
    count: usize,
}
impl ResidentSnapshot {
    pub fn len(&self) -> usize {
        self.count
    }
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }
}
impl ObservationSource for ResidentSnapshot {
    fn scan(&self, visitor: &mut dyn FnMut(&Observation) -> Result<()>) -> Result<()> {
        self.index.scan(visitor)
    }
    fn visit_bounds(
        &self,
        lower: [f64; 3],
        upper: [f64; 3],
        cancel: &Cancellation,
        visitor: &mut dyn FnMut(&Observation) -> Result<()>,
    ) -> Result<()> {
        self.index.visit_bounds(lower, upper, cancel, visitor)
    }
    fn nearest_into<'a>(
        &self,
        point: Point3,
        workspace: &'a mut NearestWorkspace,
        cancel: &Cancellation,
    ) -> Result<&'a [Observation]> {
        self.index.nearest_into(point, workspace, cancel)
    }
    fn at_site(&self, site: &Site) -> Result<Vec<Observation>> {
        self.index.at_site(site)
    }
}

impl ObservationSink for SpatialIndex {
    fn append(&mut self, observation: Observation) -> Result<()> {
        // Existing capacity is already charged. Most insertions change no
        // allocation, so they require neither a second hash nor budget atomics.
        if self.pending.len() < 31 && self.ids.len() < self.ids.capacity() {
            if !self.ids.insert(observation.id()) {
                return Err(invalid("duplicate observation ID"));
            }
            self.pending.push(observation);
            self.record_bounds(&observation);
            return Ok(());
        }

        if self.ids.contains(&observation.id()) {
            return Err(invalid("duplicate observation ID"));
        }
        let merge_count = if self.pending.len() == 31 {
            let mut count = 32usize;
            for run in self.runs.iter().rev() {
                if run.len() > count {
                    break;
                }
                count = count.checked_add(run.len()).ok_or(Error::SizeOverflow)?;
            }
            count
        } else {
            0
        };
        let identity_growth = if self.ids.len() == self.ids.capacity() {
            self.ids
                .capacity()
                .checked_add(1)
                .and_then(|n| n.checked_mul(64))
                .ok_or(Error::SizeOverflow)?
        } else {
            0
        };
        let peak = self
            .allocated_bytes()
            .checked_add(identity_growth)
            .and_then(|n| {
                n.checked_add(merge_count.checked_mul(std::mem::size_of::<Observation>())?)
            })
            .and_then(|n| n.checked_add(Hierarchy::<Observation>::node_bytes(merge_count).ok()?))
            .and_then(|n| n.checked_add(Hierarchy::<Observation>::scratch_bytes(merge_count).ok()?))
            .and_then(|n| {
                n.checked_add(
                    (self.runs.capacity() + 1)
                        * 2
                        * std::mem::size_of::<std::sync::Arc<ResidentTree>>(),
                )
            })
            .ok_or(Error::SizeOverflow)?;
        self.reservation.resize(peak as u64)?;
        let result = (|| -> Result<()> {
            self.ids
                .try_reserve(1)
                .map_err(|_| invalid("spatial identity allocation"))?;
            if self.pending.len() < 31 {
                self.pending.push(observation);
            } else {
                let mut start = self.runs.len();
                let mut count = self.pending.len() + 1;
                while start > 0 && self.runs[start - 1].len() <= count {
                    start -= 1;
                    count = count
                        .checked_add(self.runs[start].len())
                        .ok_or(Error::SizeOverflow)?;
                }
                self.runs
                    .try_reserve_exact(1)
                    .map_err(|_| invalid("spatial segment allocation"))?;
                let mut entries = Vec::new();
                entries
                    .try_reserve_exact(count)
                    .map_err(|_| invalid("spatial merge allocation"))?;
                for tree in &self.runs[start..] {
                    entries.extend(tree.iter().copied());
                }
                entries.extend(self.pending.iter().copied());
                entries.push(observation);
                // Construction can fail without losing any previously accepted record.
                let merged = Hierarchy::new(entries)?;
                let charge = self.reservation.split(merged.allocated_bytes() as u64)?;
                let merged = std::sync::Arc::new(ResidentTree {
                    tree: merged,
                    _reservation: charge,
                });
                self.runs.truncate(start);
                self.runs.push(merged);
                self.pending.clear();
            }
            self.ids.insert(observation.id());
            self.record_bounds(&observation);
            Ok(())
        })();
        self.reservation.resize(self.allocated_bytes() as u64)?;
        result
    }
}
impl ObservationSource for [Observation] {
    fn search<'a>(
        &self,
        point: Point3,
        workspace: &'a mut SearchWorkspace<'_>,
        cancel: &Cancellation,
    ) -> Result<&'a [Observation]> {
        workspace.search_scan(self, point, cancel)
    }

    fn nearest_into<'a>(
        &self,
        point: Point3,
        workspace: &'a mut NearestWorkspace,
        cancel: &Cancellation,
    ) -> Result<&'a [Observation]> {
        workspace.reset(point);
        for observation in self {
            cancel.check()?;
            workspace.offer(observation);
        }
        workspace.finish(cancel)
    }
    fn scan(&self, visitor: &mut dyn FnMut(&Observation) -> Result<()>) -> Result<()> {
        for x in self {
            visitor(x)?;
        }
        Ok(())
    }
}

impl hierarchy::Bounded for Observation {
    fn bounds(&self) -> hierarchy::Bounds {
        hierarchy::Bounds::point(self.site().support.center().coordinates())
    }
}
impl ObservationSource for hierarchy::Hierarchy<Observation> {
    fn scan(&self, visitor: &mut dyn FnMut(&Observation) -> Result<()>) -> Result<()> {
        for observation in self.iter() {
            visitor(observation)?;
        }
        Ok(())
    }
    fn visit_bounds(
        &self,
        lower: [f64; 3],
        upper: [f64; 3],
        cancel: &Cancellation,
        visitor: &mut dyn FnMut(&Observation) -> Result<()>,
    ) -> Result<()> {
        cancel.check()?;
        self.visit_bounds(hierarchy::Bounds { lower, upper }, |o| {
            cancel.check()?;
            visitor(o)
        })
    }
    fn nearest_into<'a>(
        &self,
        point: Point3,
        workspace: &'a mut NearestWorkspace,
        cancel: &Cancellation,
    ) -> Result<&'a [Observation]> {
        workspace.reset(point);
        let p = point.coordinates();
        self.traverse(
            workspace,
            |w, b| {
                let d = b.distance_squared(p);
                (d <= w.cutoff_squared()).then_some(d)
            },
            |w, o, d| {
                cancel.check()?;
                let distance = if d.is_normal() {
                    d.sqrt()
                } else {
                    point.distance(o.site().support.center())
                };
                w.offer_distance(o, distance);
                Ok(())
            },
        )?;
        workspace.finish(cancel)
    }
}
