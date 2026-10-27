use super::*;
use crate::{
    error::checked_len,
    execution::{MemoryBudget, Reservation},
};

#[derive(Clone, Copy, Debug)]
struct KeyCandidate {
    distance: f64,
    id: u64,
    slot: usize,
}
impl PartialEq for KeyCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance && self.id == other.id
    }
}
impl Eq for KeyCandidate {}
impl PartialOrd for KeyCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for KeyCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance
            .total_cmp(&other.distance)
            .then(self.id.cmp(&other.id))
    }
}

#[cfg(feature = "sqlite")]
#[derive(Clone, Copy)]
pub(crate) struct Location {
    pub page: u64,
    pub slot: usize,
    pub count: usize,
    pub center: [f64; 3],
}

/// Fixed-capacity spherical nearest-neighbor buffers. Results borrow this arena
/// until the next query. Its logical reservation covers all retained buffers;
/// traversal uses bounded, reusable scratch.
pub struct NearestWorkspace {
    query: Neighborhood,
    point: Point3,
    heap: BinaryHeap<KeyCandidate>,
    records: Vec<Observation>,
    selected: Vec<Observation>,
    #[cfg(feature = "sqlite")]
    locations: Vec<Location>,
    #[cfg(feature = "sqlite")]
    pub(crate) packed_frontier: Option<super::paged::Frontier>,
    cutoff_squared: f64,
    #[cfg(feature = "sqlite")]
    sqlite_probe_cooldown: u8,
    _reservation: Reservation,
}
impl NearestWorkspace {
    pub fn new(query: Neighborhood, budget: &MemoryBudget) -> Result<Self> {
        query.validate()?;
        let bytes = checked_len(&[
            query.maximum,
            std::mem::size_of::<KeyCandidate>() + 2 * std::mem::size_of::<Observation>(),
        ])?;
        #[cfg(feature = "sqlite")]
        let bytes = bytes
            .checked_add(checked_len(&[
                query.maximum,
                std::mem::size_of::<Location>(),
            ])?)
            .ok_or(Error::SizeOverflow)?;
        #[cfg(feature = "sqlite")]
        let bytes = bytes
            .checked_add(super::paged::Frontier::bytes())
            .ok_or(Error::SizeOverflow)?;
        let reservation = budget.reserve(u64::try_from(bytes).map_err(|_| Error::SizeOverflow)?)?;
        let mut heap = BinaryHeap::new();
        heap.try_reserve_exact(query.maximum)
            .map_err(|_| invalid("nearest heap allocation"))?;
        let mut records = Vec::new();
        records
            .try_reserve_exact(query.maximum)
            .map_err(|_| invalid("nearest payload allocation"))?;
        let mut selected = Vec::new();
        selected
            .try_reserve_exact(query.maximum)
            .map_err(|_| invalid("nearest result allocation"))?;
        #[cfg(feature = "sqlite")]
        let mut locations = Vec::new();
        #[cfg(feature = "sqlite")]
        locations
            .try_reserve_exact(query.maximum)
            .map_err(|_| invalid("nearest location allocation"))?;
        Ok(Self {
            query,
            point: Point3::new([0.; 3])?,
            heap,
            records,
            selected,
            #[cfg(feature = "sqlite")]
            locations,
            #[cfg(feature = "sqlite")]
            packed_frontier: Some(super::paged::Frontier::new()?),
            cutoff_squared: squared_search_bound(query.radius),
            #[cfg(feature = "sqlite")]
            sqlite_probe_cooldown: 0,
            _reservation: reservation,
        })
    }
    pub fn query(&self) -> Neighborhood {
        self.query
    }
    pub(crate) fn reset(&mut self, point: Point3) {
        self.point = point;
        self.heap.clear();
        self.records.clear();
        self.selected.clear();
        #[cfg(feature = "sqlite")]
        self.locations.clear();
        self.cutoff_squared = squared_search_bound(self.query.radius);
    }
    pub(crate) fn offer(&mut self, observation: &Observation) {
        self.offer_distance(
            observation,
            self.point.distance(observation.site().support.center()),
        );
    }
    pub(crate) fn cutoff_squared(&self) -> f64 {
        self.cutoff_squared
    }
    fn offer_key(&mut self, id: u64, distance: f64) -> Option<usize> {
        if distance > self.query.radius {
            return None;
        }
        let slot;
        if self.heap.len() < self.query.maximum {
            slot = self.heap.len();
            self.heap.push(KeyCandidate { distance, id, slot });
        } else {
            let mut worst = self.heap.peek_mut().unwrap();
            if distance
                .total_cmp(&worst.distance)
                .then(id.cmp(&worst.id))
                .is_ge()
            {
                return None;
            }
            slot = worst.slot;
            worst.distance = distance;
            worst.id = id;
        }
        Some(slot)
    }
    pub(crate) fn offer_distance(&mut self, observation: &Observation, distance: f64) {
        if let Some(slot) = self.offer_key(observation.id(), distance) {
            if slot == self.records.len() {
                self.records.push(*observation);
            } else {
                self.records[slot] = *observation;
            }
            self.update_cutoff();
        }
    }
    #[cfg(feature = "sqlite")]
    pub(crate) fn offer_colocated(
        &mut self,
        distance: f64,
        locations: impl Iterator<Item = (u64, Location)>,
    ) {
        let mut changed = false;
        for (id, location) in locations {
            if let Some(slot) = self.offer_key(id, distance) {
                if slot == self.locations.len() {
                    self.locations.push(location);
                } else {
                    self.locations[slot] = location;
                }
                changed = true;
            }
        }
        if changed {
            self.update_cutoff();
        }
    }
    #[cfg(feature = "sqlite")]
    pub(crate) fn resolve_locations(
        &mut self,
        mut resolve: impl FnMut(Location) -> Result<Observation>,
    ) -> Result<()> {
        self.records.clear();
        for &location in &self.locations {
            let observation = resolve(location)?;
            if observation.site().support.center().coordinates() != location.center {
                return Err(invalid("packed geometry/payload mismatch"));
            }
            self.records.push(observation);
        }
        for key in &self.heap {
            if self.records[key.slot].id() != key.id {
                return Err(invalid("packed geometry/payload identity mismatch"));
            }
        }
        Ok(())
    }
    pub(crate) fn filled_within(&self, radius: f64) -> bool {
        self.heap.len() == self.query.maximum
            && self
                .heap
                .peek()
                .is_some_and(|worst| worst.distance <= radius)
    }
    #[cfg(feature = "sqlite")]
    pub(crate) fn sqlite_probe_radius(&mut self) -> Option<f64> {
        if self.sqlite_probe_cooldown != 0 {
            self.sqlite_probe_cooldown -= 1;
            return None;
        }
        let radius = self.cutoff_squared.sqrt() * 1.25;
        (radius > 0. && radius < self.query.radius).then_some(radius)
    }
    #[cfg(feature = "sqlite")]
    pub(crate) fn sqlite_probe_failed(&mut self) {
        // Abrupt density changes can make every other probe miss. Bound that
        // repeated work, while periodically retrying if density stabilizes.
        self.sqlite_probe_cooldown = 7;
    }
    fn update_cutoff(&mut self) {
        if self.heap.len() == self.query.maximum {
            self.cutoff_squared = self.cutoff_squared.min(squared_search_bound(
                self.heap.peek().unwrap().distance.min(self.query.radius),
            ));
        }
    }
    /// A previous complete selection suggests a local radius. This is only a
    /// probe: callers must retry the full radius unless it finds k observations.
    /// No source identity or persistence of the old records is assumed.
    pub(crate) fn locality_radius(&self, point: Point3) -> Option<f64> {
        if self.selected.len() != self.query.maximum {
            return None;
        }
        let last = self.selected.last()?.site().support.center();
        let previous = self.point.distance(last);
        if self.point.distance(point) > previous {
            return None;
        }
        // A small amount of slack avoids retries for ordinary density changes.
        // This is a heuristic, not an assumed geometric upper bound.
        let radius = (previous * 1.125).next_up();
        (radius > 0. && radius < self.query.radius).then_some(radius)
    }
    pub(crate) fn restrict_radius(&mut self, radius: f64) {
        self.cutoff_squared = self.cutoff_squared.min(squared_search_bound(radius));
    }
    /// Resume selection over a union after the first source has finished.
    pub(crate) fn resume(&mut self) {
        self.heap.clear();
        self.records.clear();
        for i in 0..self.selected.len() {
            let observation = self.selected[i];
            self.offer(&observation);
        }
        self.selected.clear();
        self.update_cutoff();
    }
    pub(crate) fn finish(&mut self, cancel: &Cancellation) -> Result<&[Observation]> {
        cancel.check()?;
        self.selected.clear();
        let mut keys = std::mem::take(&mut self.heap).into_vec();
        keys.sort_unstable();
        self.selected
            .extend(keys.iter().map(|key| self.records[key.slot]));
        keys.clear();
        self.heap = BinaryHeap::from(keys);
        Ok(&self.selected)
    }
    pub(crate) fn into_selected(self) -> Vec<Observation> {
        self.selected
    }
}

/// Exact near-first traversal over contiguous spatial nodes.
pub(crate) fn visit_tree<T: Bounded, F: Fn(&T) -> Option<&Observation>>(
    tree: &Hierarchy<T>,
    workspace: &mut NearestWorkspace,
    cancel: &Cancellation,
    select: F,
) -> Result<()> {
    cancel.check()?;
    let point = workspace.point;
    let p = point.coordinates();
    tree.traverse_leaves(
        workspace,
        |w, b| {
            let d = b.distance_squared(p);
            (d <= w.cutoff_squared()).then_some(d)
        },
        |w, entries| {
            for entry in entries {
                cancel.check()?;
                if let Some(o) = select(entry) {
                    let q = o.site().support.center().coordinates();
                    let delta = [p[0] - q[0], p[1] - q[1], p[2] - q[2]];
                    let d = delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2];
                    if d <= w.cutoff_squared() {
                        let distance = if d.is_normal() {
                            d.sqrt()
                        } else {
                            point.distance(o.site().support.center())
                        };
                        w.offer_distance(o, distance);
                    }
                }
            }
            Ok(())
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forest_segments_preserve_filtered_full_payloads() {
        let records: Vec<_> = (0..4000)
            .map(|i| {
                let p = Point3::new([(i % 23) as f64, ((i / 23) % 19) as f64, (i / 437) as f64])
                    .unwrap();
                let support = if i % 2 == 0 {
                    crate::data::Support::point(p)
                } else {
                    crate::data::Support::block(p, [0.1; 3], [2, 1, 1]).unwrap()
                };
                Observation::new(
                    4000 - i,
                    Site {
                        support,
                        variable: i as usize % 3,
                    },
                    -(i as f64),
                    (i % 7) as f64 * 0.125,
                )
                .unwrap()
            })
            .collect();
        let mut index = SpatialIndex::new(vec![]).unwrap();
        // A permutation interleaves distant branches during incremental growth.
        for i in 0..records.len() {
            index.append(records[i * 7919 % records.len()]).unwrap();
        }
        fn check<const COMPLETE: bool>(index: &SpatialIndex, records: &[Observation]) {
            let budget = MemoryBudget::new(1 << 22);
            let cancel = Cancellation::default();
            let maxima = if COMPLETE {
                [records.len(), records.len() + 1, records.len() + 7]
            } else {
                [1, 7, 67]
            };
            for maximum in maxima {
                for radius in [0.5, 3., 100.] {
                    let mut workspace =
                        NearestWorkspace::new(Neighborhood { maximum, radius }, &budget).unwrap();
                    for coordinates in [[0.; 3], [11.5, 9.5, 2.], [-100.; 3]] {
                        let point = Point3::new(coordinates).unwrap();
                        for variable in [0, 2, 99] {
                            let mut expected: Vec<_> = records
                                .iter()
                                .filter(|o| o.site().variable == variable)
                                .filter_map(|o| {
                                    let distance = point.distance(o.site().support.center());
                                    (distance <= radius).then_some((distance, *o))
                                })
                                .collect();
                            expected.sort_unstable_by(|a, b| {
                                a.0.total_cmp(&b.0).then(a.1.id().cmp(&b.1.id()))
                            });
                            expected.truncate(maximum);
                            workspace.reset(point);
                            let allocations = allocation_counter::measure(|| {
                                index
                                    .visit_filtered(&mut workspace, &cancel, |entry| {
                                        (entry.site().variable == variable).then_some(entry)
                                    })
                                    .unwrap();
                                workspace.finish(&cancel).unwrap();
                            });
                            assert_eq!(allocations.count_total, 0);
                            assert_eq!(workspace.selected.len(), expected.len());
                            for (actual, (_, expected)) in workspace.selected.iter().zip(expected) {
                                assert_eq!(actual.id(), expected.id());
                                assert_eq!(actual.site(), expected.site());
                                assert_eq!(actual.value().to_bits(), expected.value().to_bits());
                                assert_eq!(
                                    actual.noise_variance().to_bits(),
                                    expected.noise_variance().to_bits()
                                );
                            }
                        }
                    }
                }
            }
            assert_eq!(budget.used(), 0);
        }
        check::<false>(&index, &records);
        check::<true>(&index, &records);
    }
    #[test]
    fn resumed_selection_preserves_complete_payloads_and_reuses_storage() {
        let observations: Vec<_> = (0..300)
            .map(|i| {
                let center = Point3::new([
                    (i % 13) as f64 - 6.,
                    ((i / 13) % 11) as f64 - 5.,
                    (i / 143) as f64,
                ])
                .unwrap();
                let support = if i % 2 == 0 {
                    crate::data::Support::point(center)
                } else {
                    crate::data::Support::block(center, [0.1; 3], [2, 1, 1]).unwrap()
                };
                Observation::new(
                    300 - i,
                    Site {
                        support,
                        variable: i as usize % 3,
                    },
                    -(i as f64),
                    (i % 5) as f64 * 0.125,
                )
                .unwrap()
            })
            .collect();
        let budget = MemoryBudget::new(1 << 22);
        let cancel = Cancellation::default();
        for maximum in [1, 2, 7, 67, 511] {
            let query = Neighborhood {
                maximum,
                radius: 7.,
            };
            let mut workspace = NearestWorkspace::new(query, &budget).unwrap();
            for coordinates in [[0.; 3], [2.5, -3.5, 1.], [100.; 3]] {
                let point = Point3::new(coordinates).unwrap();
                workspace.reset(point);
                assert!(workspace.finish(&cancel).unwrap().is_empty());
                let mut count = 0;
                for chunk in observations.chunks(31) {
                    let allocation = allocation_counter::measure(|| {
                        workspace.resume();
                        for observation in chunk {
                            workspace.offer(observation);
                        }
                        workspace.finish(&cancel).unwrap();
                    });
                    assert_eq!(allocation.count_total, 0);
                    count += chunk.len();
                    let expected = observations[..count]
                        .nearest(point, query, &cancel)
                        .unwrap();
                    assert_eq!(workspace.selected.len(), expected.len());
                    for (actual, expected) in workspace.selected.iter().zip(expected) {
                        assert_eq!(actual.id(), expected.id());
                        assert_eq!(actual.site(), expected.site());
                        assert_eq!(actual.value().to_bits(), expected.value().to_bits());
                        assert_eq!(
                            actual.noise_variance().to_bits(),
                            expected.noise_variance().to_bits()
                        );
                    }
                }
            }
        }
        assert_eq!(budget.used(), 0);
    }
}
