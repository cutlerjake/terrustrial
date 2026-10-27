use terrustrial_core::{
    data::{Observation, Point3, Site},
    execution::{Cancellation, MemoryBudget},
    spatial::{
        hierarchy::{Bounded, Bounds, Hierarchy, PointDistance},
        Neighborhood, ObservationSink, ObservationSource, SpatialIndex,
    },
    Error,
};

#[test]
fn resident_snapshots_share_geometry_and_retain_charges_across_merges() {
    let budget = MemoryBudget::new(16 << 20);
    let observation = |id| {
        Observation::new(
            id,
            Site::point(Point3::new([id as f64, 0., 0.]).unwrap(), 0),
            id as f64,
            0.,
        )
        .unwrap()
    };
    let mut store =
        SpatialIndex::with_budget((0..4096).map(observation).collect(), &budget).unwrap();
    let initial = budget.used();
    let old = store.snapshot().unwrap();
    assert!(
        budget.used() - initial < 4096,
        "snapshot must not copy the base tree"
    );
    for id in 4096..4175 {
        store.append(observation(id)).unwrap();
    }
    let middle = store.snapshot().unwrap();
    for id in 4175..4600 {
        store.append(observation(id)).unwrap();
    }
    let latest = store.snapshot().unwrap();
    drop(store);
    assert!(
        budget.used() >= 4096 * std::mem::size_of::<Observation>() as u64,
        "shared geometry still owns its charge after the writer's ID table is freed"
    );
    for (snapshot, count) in [(&old, 4096), (&middle, 4175), (&latest, 4600)] {
        assert_eq!(snapshot.len(), count);
        let mut ids = Vec::new();
        snapshot
            .scan(&mut |o| {
                ids.push(o.id());
                Ok(())
            })
            .unwrap();
        ids.sort_unstable();
        assert_eq!(ids, (0..count as u64).collect::<Vec<_>>());
        assert_eq!(
            snapshot
                .at_site(observation(count as u64 - 1).site())
                .unwrap()
                .len(),
            1
        );
        assert!(snapshot
            .at_site(observation(count as u64).site())
            .unwrap()
            .is_empty());
    }
    drop((old, middle, latest));
    assert_eq!(budget.used(), 0);
}
#[derive(Clone, Copy)]
struct BoxItem {
    id: usize,
    bounds: Bounds,
}
impl Bounded for BoxItem {
    fn bounds(&self) -> Bounds {
        self.bounds
    }
}
impl PointDistance for BoxItem {
    fn distance_squared(&self, p: [f64; 3]) -> f64 {
        self.bounds.distance_squared(p)
    }
}
#[test]
fn point_and_extended_envelopes_match_independent_range_and_nearest_oracles() {
    for count in [0, 1, 7, 8, 9, 31, 257, 8193] {
        let items: Vec<_> = (0..count)
            .map(|i| {
                let p = [(i % 31) as f64, ((i / 31) % 17) as f64, (i / 527) as f64];
                BoxItem {
                    id: i,
                    bounds: Bounds::from_corners(
                        p,
                        p.map(|x| x + if i % 2 == 0 { 0. } else { 0.75 }),
                    ),
                }
            })
            .collect();
        let tree = Hierarchy::new(items.clone()).unwrap();
        for p in [[0.; 3], [5.25, 9.25, 2.25], [100.; 3]] {
            let query = Bounds::from_corners(p.map(|x| x - 2.), p.map(|x| x + 2.));
            let mut expected: Vec<_> = items
                .iter()
                .filter(|o| {
                    (0..3).all(|a| {
                        o.bounds.lower[a] <= query.upper[a] && o.bounds.upper[a] >= query.lower[a]
                    })
                })
                .map(|o| o.id)
                .collect();
            expected.sort_unstable();
            let mut actual = Vec::with_capacity(expected.len());
            let allocated = allocation_counter::measure(|| {
                tree.visit_bounds(query, |o| {
                    actual.push(o.id);
                    Ok(())
                })
                .unwrap();
            });
            assert_eq!(allocated.count_total, 0);
            actual.sort_unstable();
            assert_eq!(actual, expected);
            actual.clear();
            let allocated =
                allocation_counter::measure(|| actual.extend(tree.query(query).map(|o| o.id)));
            assert_eq!(allocated.count_total, 0);
            actual.sort_unstable();
            assert_eq!(actual, expected);
            let mut distances: Vec<_> = items
                .iter()
                .map(|o| {
                    let d: [f64; 3] = std::array::from_fn(|a| {
                        if p[a] < o.bounds.lower[a] {
                            o.bounds.lower[a] - p[a]
                        } else if p[a] > o.bounds.upper[a] {
                            p[a] - o.bounds.upper[a]
                        } else {
                            0.
                        }
                    });
                    d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
                })
                .collect();
            distances.sort_unstable_by(f64::total_cmp);
            let found: Vec<_> = tree.nearest_iter(p).map(|(_, d)| d).collect();
            assert_eq!(found, distances);
            let mut ids: Vec<_> = tree.nearest_iter(p).map(|(o, _)| o.id).collect();
            ids.sort_unstable();
            assert_eq!(ids, (0..count).collect::<Vec<_>>());
        }
    }
    for b in [
        Bounds::point([f64::NAN, 0., 0.]),
        Bounds::point([f64::INFINITY; 3]),
        Bounds::from_corners([1.; 3], [0.; 3]),
    ] {
        assert!(Hierarchy::new(vec![BoxItem { id: 0, bounds: b }]).is_err());
    }
}
fn observation(i: u64) -> Observation {
    Observation::new(
        i,
        Site::point(Point3::new([i as f64, 0., 0.]).unwrap(), 0),
        i as f64,
        0.,
    )
    .unwrap()
}
#[test]
fn resident_budget_failures_preserve_records_ids_and_release_merge_reservations() {
    let tiny = MemoryBudget::new(1);
    assert!(matches!(
        SpatialIndex::with_budget(vec![observation(0)], &tiny),
        Err(Error::MemoryBudget { .. })
    ));
    assert_eq!(tiny.used(), 0);
    let budget = MemoryBudget::new(1 << 20);
    {
        let mut index = SpatialIndex::with_budget(vec![], &budget).unwrap();
        for i in 0..31 {
            index.append(observation(i)).unwrap();
        }
        let retained = budget.used();
        let pressure = budget.reserve(budget.limit() - retained).unwrap();
        assert!(matches!(
            index.append(observation(31)),
            Err(Error::MemoryBudget { .. })
        ));
        assert_eq!(index.len(), 31);
        assert_eq!(budget.used(), budget.limit());
        drop(pressure);
        assert_eq!(budget.used(), retained);
        index.append(observation(31)).unwrap();
        let retained = budget.used();
        assert!(index.append(observation(31)).is_err());
        assert_eq!(index.len(), 32);
        assert_eq!(budget.used(), retained);
        // Multiple merges, including monotone insertion, must not degrade exactness.
        for i in 32..2048 {
            index.append(observation(i)).unwrap();
        }
        for i in [0, 17, 31, 255, 1023, 2047] {
            let found = index
                .nearest(
                    Point3::new([i as f64, 0., 0.]).unwrap(),
                    Neighborhood {
                        maximum: 1,
                        radius: 1.,
                    },
                    &Cancellation::default(),
                )
                .unwrap();
            assert_eq!(found[0].id(), i);
        }
        assert!(budget.used() < budget.limit());
    }
    assert_eq!(budget.used(), 0);
}

#[test]
fn append_uses_already_charged_capacity_without_requesting_spare_budget() {
    let budget = MemoryBudget::new(1 << 20);
    let mut index = SpatialIndex::with_budget(vec![], &budget).unwrap();
    index.append(observation(0)).unwrap();
    let used = budget.used();
    let pressure = budget.reserve(budget.limit() - used).unwrap();
    // A one-element HashSet has capacity for at least three IDs. Existing
    // storage remains usable even when unrelated work owns the remaining budget.
    index.append(observation(1)).unwrap();
    index.append(observation(2)).unwrap();
    assert_eq!(budget.used(), budget.limit());
    assert_eq!(index.len(), 3);
    assert!(index.append(observation(1)).is_err());
    assert_eq!(index.len(), 3);
    assert!(matches!(
        index.append(observation(3)),
        Err(Error::MemoryBudget { .. })
    ));
    drop(pressure);
    for i in 0..3 {
        assert_eq!(index.at_site(observation(i).site()).unwrap()[0].id(), i);
    }
    index.append(observation(3)).unwrap();
    drop(index);
    assert_eq!(budget.used(), 0);
}
