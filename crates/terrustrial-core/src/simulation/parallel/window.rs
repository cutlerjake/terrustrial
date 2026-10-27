//! Within-window simulation dependencies and checked source overlays.
#[cfg(test)]
use crate::data::Target;
use crate::{
    data::{Group, Observation, Point3, Site},
    error::invalid,
    execution::{Cancellation, MemoryBudget},
    spatial::{
        hierarchy::{Bounded, Bounds, Hierarchy},
        visit_envelope, visit_tree, NearestWorkspace, Neighborhood, ObservationSource,
        SearchWorkspace,
    },
    Error, Result,
};
use foldhash::HashMap;

#[derive(Clone)]
struct WindowEntry {
    pub(super) rank: usize,
    observation: Observation,
}
impl Bounded for WindowEntry {
    fn bounds(&self) -> Bounds {
        Bounds::point(self.observation.site().support.center().coordinates())
    }
}
pub(super) struct Window {
    tree: Option<Hierarchy<WindowEntry>>,
    entries: Vec<WindowEntry>,
    pub(super) bindings: WindowBindings,
}
// Capacity belongs to the parent window's metadata allowance. IDs map to dense
// value slots, rather than allocating a second hash table for generated values.
#[derive(Default)]
pub(super) struct WindowBindings {
    pub(super) ids: HashMap<u64, usize>,
    values: Vec<Option<f64>>,
}
impl WindowBindings {
    fn reset(&mut self, count: usize) -> Result<()> {
        self.ids.clear();
        self.values.clear();
        self.ids
            .try_reserve(count)
            .map_err(|_| invalid("simulation window ID allocation"))?;
        self.values
            .try_reserve_exact(count)
            .map_err(|_| invalid("simulation window value allocation"))?;
        Ok(())
    }
    fn register(&mut self, id: u64) -> Result<()> {
        match self.ids.entry(id) {
            std::collections::hash_map::Entry::Occupied(_) => {
                return Err(invalid("duplicate target ID within simulation window"));
            }
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert(self.values.len());
            }
        }
        self.values.push(None);
        Ok(())
    }
    pub(super) fn value(&self, id: u64) -> Result<Option<f64>> {
        self.ids
            .get(&id)
            .map(|&slot| {
                self.values[slot].ok_or_else(|| invalid("unresolved simulation dependency"))
            })
            .transpose()
    }
    pub(super) fn record(&mut self, observation: &Observation) -> Result<()> {
        let slot = *self
            .ids
            .get(&observation.id())
            .ok_or_else(|| invalid("generated observation is absent from simulation window"))?;
        self.values[slot] = Some(observation.value());
        Ok(())
    }
}
impl Window {
    pub(super) fn empty() -> Self {
        Self {
            tree: None,
            entries: Vec::new(),
            bindings: WindowBindings::default(),
        }
    }
    pub(super) fn rebuild(&mut self, batch: &[Group]) -> Result<()> {
        // Release the preceding spatial tree before constructing its replacement
        // so two complete trees do not overlap the window metadata allowance.
        self.tree = None;
        self.entries.clear();
        let count = batch.iter().try_fold(0usize, |n, g| {
            n.checked_add(g.targets.len()).ok_or(Error::SizeOverflow)
        })?;
        self.bindings.reset(count)?;
        self.entries
            .try_reserve_exact(count)
            .map_err(|_| invalid("simulation window entry allocation"))?;
        for (rank, group) in batch.iter().enumerate() {
            for target in &group.targets {
                self.bindings.register(target.id)?;
                self.entries.push(WindowEntry {
                    rank,
                    observation: Observation::new(target.id, target.site, 0.0, 0.0)?,
                });
            }
        }
        // One leaf needs no hierarchy. Keep its entries in reusable path order,
        // allowing queries to stop as soon as they reach a future group.
        if count > 128 {
            self.tree = Some(Hierarchy::new(std::mem::take(&mut self.entries))?);
        }
        Ok(())
    }
    fn visit(
        &self,
        rank: usize,
        envelope: Bounds,
        mut visitor: impl FnMut(&WindowEntry) -> Result<()>,
    ) -> Result<()> {
        if let Some(tree) = &self.tree {
            visit_envelope(tree, envelope, |entry| {
                if entry.rank < rank {
                    visitor(entry)?;
                }
                Ok(())
            })
        } else {
            for entry in self.entries.iter().take_while(|entry| entry.rank < rank) {
                if envelope.contains_point(&entry.observation.site().support.center().coordinates())
                {
                    visitor(entry)?;
                }
            }
            Ok(())
        }
    }
    fn nearest_into(
        &self,
        point: Point3,
        rank: usize,
        workspace: &mut NearestWorkspace,
        cancel: &Cancellation,
    ) -> Result<()> {
        if let Some(tree) = &self.tree {
            visit_tree(tree, workspace, cancel, |entry| {
                (entry.rank < rank).then_some(&entry.observation)
            })
        } else {
            cancel.check()?;
            for entry in self.entries.iter().take_while(|entry| entry.rank < rank) {
                cancel.check()?;
                if entry.bounds().distance_squared(point.coordinates())
                    <= workspace.cutoff_squared()
                {
                    workspace.offer(&entry.observation);
                }
            }
            Ok(())
        }
    }
    #[cfg(test)]
    fn new(batch: &[Group]) -> Result<Self> {
        let mut window = Self::empty();
        window.rebuild(batch)?;
        Ok(window)
    }
}
pub(super) struct Checked<'a, S: ?Sized> {
    pub(super) source: &'a S,
    pub(super) ids: &'a HashMap<u64, usize>,
}
impl<S: ObservationSource + ?Sized> Checked<'_, S> {
    fn check(&self, o: &Observation) -> Result<()> {
        if self.ids.contains_key(&o.id()) {
            Err(invalid("target ID collides with conditioning observation"))
        } else {
            Ok(())
        }
    }
}
impl<S: ObservationSource + ?Sized> ObservationSource for Checked<'_, S> {
    fn nearest_into<'a>(
        &self,
        point: Point3,
        workspace: &'a mut NearestWorkspace,
        cancel: &Cancellation,
    ) -> Result<&'a [Observation]> {
        let found = self.source.nearest_into(point, workspace, cancel)?;
        for observation in found {
            self.check(observation)?;
        }
        Ok(found)
    }
    fn scan(&self, visitor: &mut dyn FnMut(&Observation) -> Result<()>) -> Result<()> {
        self.source.scan(&mut |o| {
            self.check(o)?;
            visitor(o)
        })
    }
    fn visit_bounds(
        &self,
        lower: [f64; 3],
        upper: [f64; 3],
        cancel: &Cancellation,
        visitor: &mut dyn FnMut(&Observation) -> Result<()>,
    ) -> Result<()> {
        self.source.visit_bounds(lower, upper, cancel, &mut |o| {
            self.check(o)?;
            visitor(o)
        })
    }
    fn at_site(&self, site: &Site) -> Result<Vec<Observation>> {
        let found = self.source.at_site(site)?;
        for o in &found {
            self.check(o)?;
        }
        Ok(found)
    }
    fn at_site_with_center_status(&self, site: &Site) -> Result<(Vec<Observation>, bool)> {
        let (found, center_empty) = self.source.at_site_with_center_status(site)?;
        for o in &found {
            self.check(o)?;
        }
        Ok((found, center_empty))
    }
    fn nearest(
        &self,
        point: Point3,
        query: Neighborhood,
        cancel: &Cancellation,
    ) -> Result<Vec<Observation>> {
        let found = self.source.nearest(point, query, cancel)?;
        for o in &found {
            self.check(o)?;
        }
        Ok(found)
    }
    fn search<'a>(
        &self,
        point: Point3,
        workspace: &'a mut SearchWorkspace<'_>,
        cancel: &Cancellation,
    ) -> Result<&'a [Observation]> {
        let found = self.source.search(point, workspace, cancel)?;
        for o in found {
            self.check(o)?;
        }
        Ok(found)
    }
}
pub(super) struct Overlay<'a, S: ?Sized> {
    pub(super) base: &'a S,
    pub(super) window: &'a Window,
    pub(super) rank: usize,
}
impl<S: ObservationSource + ?Sized> ObservationSource for Overlay<'_, S> {
    fn nearest_into<'a>(
        &self,
        point: Point3,
        workspace: &'a mut NearestWorkspace,
        cancel: &Cancellation,
    ) -> Result<&'a [Observation]> {
        // The first group has no within-window predecessors. The base source
        // has already ranked its result, so avoid rebuilding and sorting it.
        if self.rank == 0 {
            cancel.check()?;
            return self.base.nearest_into(point, workspace, cancel);
        }
        self.base.nearest_into(point, workspace, cancel)?;
        workspace.resume();
        self.window
            .nearest_into(point, self.rank, workspace, cancel)?;
        workspace.finish(cancel)
    }
    fn scan(&self, visitor: &mut dyn FnMut(&Observation) -> Result<()>) -> Result<()> {
        self.base.scan(visitor)?;
        self.window.visit(
            self.rank,
            Bounds::from_corners([f64::NEG_INFINITY; 3], [f64::INFINITY; 3]),
            |entry| visitor(&entry.observation),
        )
    }
    fn visit_bounds(
        &self,
        lower: [f64; 3],
        upper: [f64; 3],
        cancel: &Cancellation,
        visitor: &mut dyn FnMut(&Observation) -> Result<()>,
    ) -> Result<()> {
        self.base.visit_bounds(lower, upper, cancel, visitor)?;
        self.window
            .visit(self.rank, Bounds::from_corners(lower, upper), |entry| {
                cancel.check()?;
                visitor(&entry.observation)?;
                Ok(())
            })
    }
    fn at_site(&self, site: &Site) -> Result<Vec<Observation>> {
        let mut found = self.base.at_site(site)?;
        self.window.visit(
            self.rank,
            Bounds::point(site.support.center().coordinates()),
            |entry| {
                if entry.observation.site() == site {
                    found.push(entry.observation);
                }
                Ok(())
            },
        )?;
        found.sort_by_key(Observation::id);
        Ok(found)
    }
    fn nearest(
        &self,
        point: Point3,
        query: Neighborhood,
        cancel: &Cancellation,
    ) -> Result<Vec<Observation>> {
        let mut workspace = NearestWorkspace::new(query, &MemoryBudget::new(u64::MAX))?;
        self.nearest_into(point, &mut workspace, cancel)?;
        Ok(workspace.into_selected())
    }
}

#[cfg(test)]
mod window_tests {
    use super::*;

    #[test]
    fn checked_exact_center_proof_reuses_misses_and_preserves_id_validation() {
        use crate::spatial::exact::ExactWorkspace;
        use std::cell::Cell;

        struct Source {
            rows: Vec<Observation>,
            calls: Cell<usize>,
            fail: Cell<bool>,
        }
        impl ObservationSource for Source {
            fn scan(&self, _: &mut dyn FnMut(&Observation) -> Result<()>) -> Result<()> {
                panic!("preserve exact override")
            }
            fn at_site(&self, site: &Site) -> Result<Vec<Observation>> {
                self.calls.set(self.calls.get() + 1);
                if self.fail.get() {
                    return Err(Error::Cancelled);
                }
                Ok(self
                    .rows
                    .iter()
                    .filter(|o| o.site() == site)
                    .copied()
                    .collect())
            }
            fn at_site_with_center_status(&self, site: &Site) -> Result<(Vec<Observation>, bool)> {
                Ok((
                    self.at_site(site)?,
                    !self
                        .rows
                        .iter()
                        .any(|o| o.site().support.center() == site.support.center()),
                ))
            }
        }
        let site = Site::point(Point3::new([0.; 3]).unwrap(), 1);
        let source = Source {
            rows: vec![
                Observation::new(9, site, 5., 0.).unwrap(),
                Observation::new(7, site, 5., 0.).unwrap(),
            ],
            calls: Cell::new(0),
            fail: Cell::new(false),
        };
        let mut ids = HashMap::default();
        let mut workspace = ExactWorkspace::default();
        let checked = Checked {
            source: &source,
            ids: &ids,
        };
        for variable in 0..7 {
            let absent = Site::point(Point3::new([1.; 3]).unwrap(), variable);
            assert!(workspace.at_site(&checked, &absent).unwrap().is_empty());
        }
        assert_eq!(
            source.calls.get(),
            1,
            "the production wrapper must forward proof"
        );
        let (found, center_empty) = checked.at_site_with_center_status(&site).unwrap();
        assert!(!center_empty);
        assert_eq!(
            found.iter().map(Observation::id).collect::<Vec<_>>(),
            [9, 7]
        );

        ids.insert(7, 0);
        let checked = Checked {
            source: &source,
            ids: &ids,
        };
        assert!(checked.at_site_with_center_status(&site).is_err());
        assert!(checked.at_site(&site).is_err());
        source.fail.set(true);
        assert!(matches!(
            checked.at_site_with_center_status(&site),
            Err(Error::Cancelled)
        ));
    }

    #[test]
    fn checked_exact_center_custom_default_never_invents_empty_center_proof() {
        use crate::spatial::exact::ExactWorkspace;
        use std::cell::Cell;

        struct Custom(Cell<usize>);
        impl ObservationSource for Custom {
            fn scan(&self, _: &mut dyn FnMut(&Observation) -> Result<()>) -> Result<()> {
                panic!("preserve custom at_site override")
            }
            fn at_site(&self, _: &Site) -> Result<Vec<Observation>> {
                self.0.set(self.0.get() + 1);
                Ok(Vec::new())
            }
        }
        let custom = Custom(Cell::new(0));
        let ids = HashMap::default();
        let checked = Checked {
            source: &custom,
            ids: &ids,
        };
        let mut workspace = ExactWorkspace::default();
        for variable in 0..7 {
            let site = Site::point(Point3::new([0.; 3]).unwrap(), variable);
            assert!(workspace.at_site(&checked, &site).unwrap().is_empty());
        }
        assert_eq!(custom.0.get(), 7);
    }

    #[test]
    fn flat_windows_rebuild_and_query_without_allocation_and_transition_to_trees() {
        let groups: Vec<_> = (0..129)
            .map(|i| Group {
                id: i,
                targets: vec![Target {
                    id: 1000 + i,
                    site: Site::point(
                        Point3::new([(i % 17) as f64, (i / 17) as f64, 0.]).unwrap(),
                        0,
                    ),
                }],
            })
            .collect();
        let mut window = Window::new(&groups[..128]).unwrap();
        let budget = MemoryBudget::new(1 << 20);
        let cancel = Cancellation::default();
        let point = Point3::new([0.; 3]).unwrap();
        let mut workspace = NearestWorkspace::new(
            Neighborhood {
                maximum: 8,
                radius: 100.,
            },
            &budget,
        )
        .unwrap();
        let envelope = Bounds::from_corners([-1.; 3], [100.; 3]);
        for n in [128, 7, 0, 64, 127, 128] {
            let allocations = allocation_counter::measure(|| {
                window.rebuild(&groups[..n]).unwrap();
                assert!(window.tree.is_none());
                workspace.reset(point);
                window
                    .nearest_into(point, n / 2, &mut workspace, &cancel)
                    .unwrap();
                assert_eq!(workspace.finish(&cancel).unwrap().len(), (n / 2).min(8));
                let mut count = 0;
                window
                    .visit(n / 2, envelope, |_| {
                        count += 1;
                        Ok(())
                    })
                    .unwrap();
                assert_eq!(count, n / 2);
            });
            assert_eq!(allocations.count_total, 0);
        }
        for n in [129, 128, 129, 0, 128] {
            window.rebuild(&groups[..n]).unwrap();
            assert_eq!(window.tree.is_some(), n > 128);
            let mut ids = Vec::new();
            window
                .visit(n, envelope, |entry| {
                    ids.push(entry.observation.id());
                    Ok(())
                })
                .unwrap();
            ids.sort_unstable();
            assert_eq!(ids, (1000..1000 + n as u64).collect::<Vec<_>>());
        }
        cancel.cancel();
        assert!(matches!(
            window.nearest_into(point, 0, &mut workspace, &cancel),
            Err(Error::Cancelled)
        ));
    }

    #[test]
    fn window_bindings_reuse_capacity_without_stale_values_or_allocations() {
        let site = Site::point(Point3::new([0.; 3]).unwrap(), 0);
        let mut bindings = WindowBindings::default();
        bindings.reset(64).unwrap();
        for count in [64, 16, 0, 8, 64] {
            let allocations = allocation_counter::measure(|| {
                bindings.reset(count).unwrap();
                for i in 0..count {
                    let id = u64::MAX - i as u64;
                    bindings.register(id).unwrap();
                    let value = if i % 2 == 0 { -0. } else { i as f64 };
                    bindings
                        .record(&Observation::new(id, site, value, 0.).unwrap())
                        .unwrap();
                    assert_eq!(
                        bindings.value(id).unwrap().unwrap().to_bits(),
                        value.to_bits()
                    );
                }
                assert_eq!(bindings.value(u64::MAX - count as u64).unwrap(), None);
            });
            assert_eq!(allocations.count_total, 0);
        }
        bindings.reset(1).unwrap();
        bindings.register(u64::MAX).unwrap();
        assert!(bindings.value(u64::MAX).is_err());
        assert_eq!(bindings.value(u64::MAX - 1).unwrap(), None);
        bindings
            .record(&Observation::new(u64::MAX, site, -0., 0.).unwrap())
            .unwrap();
        assert!(bindings.register(u64::MAX).is_err());
        assert_eq!(
            bindings.value(u64::MAX).unwrap().unwrap().to_bits(),
            (-0_f64).to_bits()
        );
        assert!(bindings
            .record(&Observation::new(7, site, 1., 0.).unwrap())
            .is_err());
        assert!(bindings.reset(usize::MAX).is_err());
        assert_eq!(bindings.value(u64::MAX).unwrap(), None);
        bindings.reset(1).unwrap();
        bindings.register(u64::MAX).unwrap();
        assert!(bindings.value(u64::MAX).is_err());
    }

    #[test]
    fn rebuilt_windows_discard_prior_geometry_and_recover_after_duplicate_ids() {
        let mut groups: Vec<_> = (0..16)
            .map(|i| Group {
                id: i,
                targets: vec![Target {
                    id: 100 + i,
                    site: Site::point(Point3::new([i as f64, 0., 0.]).unwrap(), 0),
                }],
            })
            .collect();
        let mut window = Window::new(&groups).unwrap();
        for group in &groups {
            let target = &group.targets[0];
            window
                .bindings
                .record(&Observation::new(target.id, target.site, 19., 0.).unwrap())
                .unwrap();
        }
        groups[3].targets[0].id = groups[0].targets[0].id;
        assert!(window.rebuild(&groups).is_err());
        assert!(window.tree.is_none());
        groups.truncate(1);
        groups[0].targets[0].site = Site::point(Point3::new([-100., 0., 0.]).unwrap(), 0);
        window.rebuild(&groups).unwrap();
        assert_eq!(window.entries.len(), 1);
        assert_eq!(window.bindings.ids.len(), 1);
        assert!(window.bindings.value(100).is_err());
        assert_eq!(window.bindings.value(101).unwrap(), None);
        let entry = &window.entries[0];
        assert_eq!(entry.observation.site(), &groups[0].targets[0].site);
        assert_eq!(entry.rank, 0);
        window.rebuild(&[]).unwrap();
        assert!(window.tree.is_none());
        assert!(window.entries.is_empty());
        assert!(window.bindings.ids.is_empty());
        assert!(window.bindings.values.is_empty());
    }

    #[test]
    fn window_layout_matches_full_sort_across_ranks_ties_and_coordinate_extremes() {
        let budget = MemoryBudget::new(1 << 24);
        let cancel = Cancellation::default();
        for scale in [1e-200, 1., 1e200] {
            for n in [4, 16, 64, 256] {
                let groups: Vec<_> = (0..n)
                    .map(|i| Group {
                        id: i as u64,
                        targets: (0..8)
                            .map(|j| Target {
                                id: u64::MAX - (i * 8 + j) as u64,
                                site: Site::point(
                                    Point3::new([
                                        ((i % 8) as f64 - 3.) * scale,
                                        (((i / 8) % 8) as f64 - 3.) * scale,
                                        0.,
                                    ])
                                    .unwrap(),
                                    j,
                                ),
                            })
                            .collect(),
                    })
                    .collect();
                let window = Window::new(&groups).unwrap();
                let hard =
                    [
                        Observation::new(
                            0,
                            Site::point(Point3::new([0.; 3]).unwrap(), 0),
                            2.,
                            0.01,
                        )
                        .unwrap(),
                    ];
                for rank in [0, 1, n / 2, n] {
                    let overlay = Overlay {
                        base: hard.as_slice(),
                        window: &window,
                        rank,
                    };
                    let candidates: Vec<_> = hard
                        .iter()
                        .copied()
                        .chain(groups[..rank].iter().flat_map(|g| {
                            g.targets
                                .iter()
                                .map(|t| Observation::new(t.id, t.site, 0., 0.).unwrap())
                        }))
                        .collect();
                    for coordinates in [
                        [0.; 3],
                        [scale, -scale, 0.],
                        [2.7 * scale, 0.2 * scale, scale],
                    ] {
                        let point = Point3::new(coordinates).unwrap();
                        for maximum in [1, 16, 256] {
                            let query = Neighborhood {
                                maximum,
                                radius: 3. * scale,
                            };
                            let mut expected: Vec<_> = candidates
                                .iter()
                                .filter_map(|o| {
                                    let distance = point.distance(o.site().support.center());
                                    (distance <= query.radius).then_some((distance, o.id()))
                                })
                                .collect();
                            expected
                                .sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
                            expected.truncate(maximum);
                            let mut workspace = NearestWorkspace::new(query, &budget).unwrap();
                            let actual: Vec<_> = overlay
                                .nearest_into(point, &mut workspace, &cancel)
                                .unwrap()
                                .iter()
                                .map(Observation::id)
                                .collect();
                            assert_eq!(
                                actual,
                                expected.iter().map(|v| v.1).collect::<Vec<_>>(),
                                "scale={scale}, groups={n}, rank={rank}, maximum={maximum}"
                            );
                        }
                    }
                    let site = Site::point(Point3::new([0.; 3]).unwrap(), 0);
                    let mut expected: Vec<_> = candidates
                        .iter()
                        .filter(|o| o.site() == &site)
                        .map(Observation::id)
                        .collect();
                    expected.sort_unstable();
                    assert_eq!(
                        overlay
                            .at_site(&site)
                            .unwrap()
                            .iter()
                            .map(Observation::id)
                            .collect::<Vec<_>>(),
                        expected
                    );
                    let lower = [-scale; 3];
                    let upper = [scale; 3];
                    let mut actual = Vec::new();
                    overlay
                        .visit_bounds(lower, upper, &cancel, &mut |o| {
                            actual.push(o.id());
                            Ok(())
                        })
                        .unwrap();
                    actual.sort_unstable();
                    let mut expected: Vec<_> = candidates
                        .iter()
                        .filter(|o| {
                            o.site()
                                .support
                                .center()
                                .coordinates()
                                .into_iter()
                                .all(|v| v >= -scale && v <= scale)
                        })
                        .map(Observation::id)
                        .collect();
                    expected.sort_unstable();
                    assert_eq!(actual, expected);
                }
            }
        }
        assert_eq!(budget.used(), 0);
    }
}
