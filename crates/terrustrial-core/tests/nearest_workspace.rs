use terrustrial_core::{
    data::{Observation, Point3, Site},
    execution::{Cancellation, MemoryBudget},
    spatial::{NearestWorkspace, Neighborhood, ObservationSink, ObservationSource, SpatialIndex},
    Result,
};
fn point(p: [f64; 3]) -> Point3 {
    Point3::new(p).unwrap()
}
fn observations(count: usize, scale: f64) -> Vec<Observation> {
    (0..count)
        .map(|i| {
            Observation::new(
                (count - i) as u64,
                Site::point(
                    point([
                        (i % 23) as f64 * scale,
                        ((i / 23) % 19) as f64 * scale,
                        (i / 437) as f64 * scale,
                    ]),
                    i % 3,
                ),
                i as f64,
                0.,
            )
            .unwrap()
        })
        .collect()
}
fn check(
    source: &impl ObservationSource,
    records: &[Observation],
    p: Point3,
    workspace: &mut NearestWorkspace,
) {
    let query = workspace.query();
    let mut expected: Vec<_> = records
        .iter()
        .filter_map(|o| {
            let distance = p.distance(o.site().support.center());
            (distance <= query.radius).then_some((distance, o))
        })
        .collect();
    expected.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.id().cmp(&b.1.id())));
    expected.truncate(query.maximum);
    let found = source
        .nearest_into(p, workspace, &Cancellation::default())
        .unwrap();
    let owned = source.nearest(p, query, &Cancellation::default()).unwrap();
    for found in [found, owned.as_slice()] {
        assert_eq!(found.len(), expected.len());
        for (actual, (_, expected)) in found.iter().zip(&expected) {
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

#[test]
fn bulk_incremental_ties_radius_extremes_and_reuse_match_full_sort() {
    let budget = MemoryBudget::new(1 << 24);
    for scale in [1e-200, 1., 1e200] {
        let records = observations(3000, scale);
        let bulk = SpatialIndex::new(records.clone()).unwrap();
        let mut incremental = SpatialIndex::new(vec![]).unwrap();
        for o in records.iter().rev() {
            incremental.append(*o).unwrap();
        }
        for maximum in [1, 7, 32, 67, 257, 3000, 3001] {
            for radius in [0.5, 2., 20.] {
                let mut workspace = NearestWorkspace::new(
                    Neighborhood {
                        maximum,
                        radius: radius * scale,
                    },
                    &budget,
                )
                .unwrap();
                for center in [
                    [0., 0., 0.],
                    [2., 2., 0.],
                    [11.5, 9.5, 2.],
                    [-10., -10., -10.],
                    [23., 19., 7.],
                ] {
                    let p = point(center.map(|v| v * scale));
                    check(&bulk, &records, p, &mut workspace);
                    check(&incremental, &records, p, &mut workspace);
                }
            }
        }
    }
    assert_eq!(budget.used(), 0);
}

#[test]
fn near_maximum_finite_coordinates_match_full_sort_after_bulk_and_incremental_builds() {
    let records: Vec<_> = (0..256)
        .map(|i| {
            let sign = if i < 128 { 1. } else { -1. };
            Observation::new(
                u64::MAX - i as u64,
                Site::point(
                    point([
                        sign * (1e308 + (i % 7) as f64 * 1e306),
                        sign * (1e308 + ((i / 7) % 7) as f64 * 1e306),
                        sign * (1e308 + (i / 49) as f64 * 1e306),
                    ]),
                    0,
                ),
                i as f64,
                0.,
            )
            .unwrap()
        })
        .collect();
    let bulk = SpatialIndex::new(records.clone()).unwrap();
    let mut incremental = SpatialIndex::new(vec![]).unwrap();
    for &record in &records {
        incremental.append(record).unwrap();
    }
    let budget = MemoryBudget::new(1 << 20);
    for radius in [1e306, 1e307, f64::MAX] {
        let mut workspace = NearestWorkspace::new(
            Neighborhood {
                maximum: 17,
                radius,
            },
            &budget,
        )
        .unwrap();
        for coordinates in [[1e308; 3], [-1e308; 3], [0.; 3]] {
            for index in [&bulk, &incremental] {
                check(index, &records, point(coordinates), &mut workspace);
                let owned = index
                    .nearest(
                        point(coordinates),
                        workspace.query(),
                        &Cancellation::default(),
                    )
                    .unwrap();
                let reused = index
                    .nearest_into(point(coordinates), &mut workspace, &Cancellation::default())
                    .unwrap();
                assert_eq!(
                    owned.iter().map(Observation::id).collect::<Vec<_>>(),
                    reused.iter().map(Observation::id).collect::<Vec<_>>()
                );
            }
        }
    }
    assert_eq!(budget.used(), 0);
}

#[test]
fn repeated_indexed_queries_allocate_nothing_and_preserve_owned_reference() {
    let records = observations(20000, 0.3);
    let index = SpatialIndex::new(records).unwrap();
    let query = Neighborhood {
        maximum: 32,
        radius: 8.,
    };
    let budget = MemoryBudget::new(1 << 20);
    let mut workspace = NearestWorkspace::new(query, &budget).unwrap();
    let cancel = Cancellation::default();
    let points: Vec<_> = (0..1000)
        .map(|i| {
            point([
                (i % 23) as f64 * 0.3 + 0.13,
                ((i / 23) % 19) as f64 * 0.3 + 0.17,
                ((i / 437) % 17) as f64 * 0.3 + 0.19,
            ])
        })
        .collect();
    for &p in &points {
        let expected = index.nearest(p, query, &cancel).unwrap();
        let found = index.nearest_into(p, &mut workspace, &cancel).unwrap();
        assert_eq!(
            expected.iter().map(Observation::id).collect::<Vec<_>>(),
            found.iter().map(Observation::id).collect::<Vec<_>>()
        );
    }
    let owned = allocation_counter::measure(|| {
        for &p in &points {
            std::hint::black_box(index.nearest(p, query, &cancel).unwrap());
        }
    });
    let reused = allocation_counter::measure(|| {
        for &p in &points {
            std::hint::black_box(index.nearest_into(p, &mut workspace, &cancel).unwrap());
        }
    });
    eprintln!("1000 nearest queries: owned {owned:?}; workspace {reused:?}");
    assert_eq!(reused.count_total, 0);
    assert_eq!(reused.bytes_total, 0);
    assert!(owned.count_total >= 1000);
    assert!(budget.used() > 0);
    drop(workspace);
    assert_eq!(budget.used(), 0);
}

#[test]
fn limits_cancellation_empty_recovery_and_custom_override() {
    let budget = MemoryBudget::new(1 << 20);
    assert!(NearestWorkspace::new(
        Neighborhood {
            maximum: 0,
            radius: 1.
        },
        &budget
    )
    .is_err());
    assert!(NearestWorkspace::new(
        Neighborhood {
            maximum: usize::MAX,
            radius: 1.
        },
        &budget
    )
    .is_err());
    assert_eq!(budget.used(), 0);
    let query = Neighborhood {
        maximum: 32,
        radius: 8.,
    };
    assert!(NearestWorkspace::new(query, &MemoryBudget::new(1)).is_err());
    let mut workspace = NearestWorkspace::new(query, &budget).unwrap();
    let empty = SpatialIndex::new(vec![]).unwrap();
    let cancel = Cancellation::default();
    assert!(matches!(
        empty.nearest(
            point([0.; 3]),
            Neighborhood {
                maximum: usize::MAX,
                radius: 1.,
            },
            &cancel,
        ),
        Err(terrustrial_core::Error::SizeOverflow)
    ));
    assert!(empty
        .nearest_into(point([0.; 3]), &mut workspace, &cancel)
        .unwrap()
        .is_empty());
    cancel.cancel();
    assert!(empty
        .nearest_into(point([0.; 3]), &mut workspace, &cancel)
        .is_err());
    let records = observations(5, 1.);
    // Default nearest_into must honor an existing source's nearest override.
    struct Custom(Vec<Observation>);
    impl ObservationSource for Custom {
        fn scan(&self, _: &mut dyn FnMut(&Observation) -> Result<()>) -> Result<()> {
            panic!("nearest override must be used")
        }
        fn nearest(
            &self,
            _: Point3,
            _: Neighborhood,
            _: &Cancellation,
        ) -> Result<Vec<Observation>> {
            Ok(self.0.clone())
        }
    }
    let custom = Custom(records.clone());
    check(&custom, &records, point([0.; 3]), &mut workspace);
    check(
        &SpatialIndex::new(records.clone()).unwrap(),
        &records,
        point([0.; 3]),
        &mut workspace,
    );
}

#[test]
fn locality_probe_retries_changed_sources_and_density_without_stale_payloads() {
    let budget = MemoryBudget::new(1 << 20);
    let dense = observations(300, 0.001);
    let sparse: Vec<_> = observations(300, 0.25)
        .into_iter()
        .map(|o| Observation::new(o.id(), *o.site(), -o.value() - 1., 0.5).unwrap())
        .collect();
    let dense_index = SpatialIndex::new(dense.clone()).unwrap();
    let sparse_index = SpatialIndex::new(sparse.clone()).unwrap();
    let mut workspace = NearestWorkspace::new(
        Neighborhood {
            maximum: 7,
            radius: 100.,
        },
        &budget,
    )
    .unwrap();
    // The same center and IDs deliberately conceal the changed source. Only a
    // full-radius retry can find seven sparse neighbors after the dense query.
    for _ in 0..3 {
        check(&dense_index, &dense, point([0.; 3]), &mut workspace);
        check(&sparse_index, &sparse, point([0.; 3]), &mut workspace);
    }
    let cancel = Cancellation::default();
    let measured = allocation_counter::measure(|| {
        for source in [&dense_index, &sparse_index, &dense_index, &sparse_index] {
            assert_eq!(
                source
                    .nearest_into(point([0.; 3]), &mut workspace, &cancel)
                    .unwrap()
                    .len(),
                7
            );
        }
    });
    assert_eq!(measured.count_total, 0);
    cancel.cancel();
    assert!(matches!(
        dense_index.nearest_into(point([0.; 3]), &mut workspace, &cancel),
        Err(terrustrial_core::Error::Cancelled)
    ));
    check(&sparse_index, &sparse, point([0.; 3]), &mut workspace);
    check(&dense_index, &dense, point([100.; 3]), &mut workspace);
}
