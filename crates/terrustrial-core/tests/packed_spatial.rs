#![cfg(feature = "sqlite")]
use terrustrial_core::{
    data::{Observation, Point3, Site, Support},
    execution::{Cancellation, MemoryBudget},
    model::Anisotropy,
    spatial::{
        NearestWorkspace, Neighborhood, ObservationSource, SearchNeighborhood, SearchWorkspace,
        VariableLimit,
    },
    storage::packed::{Options, PackedBuilder, PackedStore, Snapshot},
    Error,
};
fn point(p: [f64; 3]) -> Point3 {
    Point3::new(p).unwrap()
}
fn records(n: usize) -> Vec<Observation> {
    (0..n)
        .map(|i| {
            let location = i / 4;
            let p = point([
                (location % 19) as f64,
                ((location / 19) % 13) as f64,
                (location / 247) as f64,
            ]);
            let support = if i % 9 == 0 {
                Support::block(p, [0.1, 0.2, 0.], [2, 3, 1]).unwrap()
            } else {
                Support::point(p)
            };
            Observation::new(
                u64::MAX - i as u64,
                Site {
                    support,
                    variable: i % 4,
                },
                if i % 11 == 0 { -0. } else { (i as f64).sin() },
                if i % 7 == 0 { -0. } else { 0.01 },
            )
            .unwrap()
        })
        .collect()
}
fn options() -> Options {
    Options {
        build_window_leaves: 256,
        leaf_records: 23,
        cached_pages: 12,
        readers: 4,
        directory_pages: 8,
        sqlite_cache_bytes: 16384,
        wal_checkpoint_bytes: 64 << 20,
        max_database_bytes: 1 << 28,
    }
}
fn build(
    path: &std::path::Path,
    records: &[Observation],
    ranks: impl Fn(usize) -> u64,
    budget: &MemoryBudget,
) -> PackedStore {
    let mut b = PackedBuilder::create(path, options(), budget).unwrap();
    // Reverse input deliberately; physical order must not affect rank or IDs.
    for (i, o) in records.iter().enumerate().rev() {
        b.push(*o, ranks(i), &Cancellation::default()).unwrap();
    }
    b.finish(&Cancellation::default()).unwrap()
}
fn equal(actual: &[Observation], expected: &[Observation]) {
    assert_eq!(actual.len(), expected.len());
    for (a, b) in actual.iter().zip(expected) {
        assert_eq!(a.id(), b.id());
        assert_eq!(a.site(), b.site());
        assert_eq!(a.value().to_bits(), b.value().to_bits());
        assert_eq!(a.noise_variance().to_bits(), b.noise_variance().to_bits());
        for (a, b) in a
            .site()
            .support
            .center()
            .coordinates()
            .iter()
            .zip(b.site().support.center().coordinates())
        {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }
}
fn check(snapshot: &impl ObservationSource, observations: &[Observation]) {
    let budget = MemoryBudget::new(1 << 24);
    for maximum in [1, 7, 32, observations.len().max(1) + 1] {
        for radius in [0.5, 2.5, 100.] {
            let q = Neighborhood { maximum, radius };
            let mut w = NearestWorkspace::new(q, &budget).unwrap();
            for p in [
                [0., 0., 0.],
                [2.5, 1.5, 0.],
                [9.9, 7.1, 0.5],
                [-100., 10., 0.],
            ] {
                let p = point(p);
                let mut expected: Vec<_> = observations
                    .iter()
                    .filter(|o| p.distance(o.site().support.center()) <= radius)
                    .copied()
                    .collect();
                expected.sort_unstable_by(|a, b| {
                    p.distance(a.site().support.center())
                        .total_cmp(&p.distance(b.site().support.center()))
                        .then(a.id().cmp(&b.id()))
                });
                expected.truncate(maximum);
                equal(
                    snapshot
                        .nearest_into(p, &mut w, &Cancellation::default())
                        .unwrap(),
                    &expected,
                );
            }
        }
    }
}

#[test]
fn exact_disk_queries_empty_ties_payloads_and_eviction() {
    let dir = tempfile::tempdir().unwrap();
    for n in [0, 1, 24, 513] {
        let budget = MemoryBudget::new(8 << 20);
        let observations = records(n);
        let store = build(
            &dir.path().join(format!("{n}.sqlite")),
            &observations,
            |_| 0,
            &budget,
        );
        let snapshot = store.snapshot().unwrap();
        assert_eq!(snapshot.len(), n as u64);
        check(&snapshot, &observations);
        store.clear_cache().unwrap();
        check(&snapshot, &observations);
        if n > 0 {
            assert!(store.statistics().page_reads > 0);
        }
        drop(snapshot);
        drop(store);
        assert_eq!(budget.used(), 0);
    }
}

#[test]
fn constrained_anisotropic_search_uses_all_eligible_candidates() {
    let dir = tempfile::tempdir().unwrap();
    let budget = MemoryBudget::new(8 << 20);
    let observations = records(777);
    let store = build(
        &dir.path().join("search.sqlite"),
        &observations,
        |i| if i % 3 == 0 { 5 } else { 0 },
        &budget,
    );
    let visible: Vec<_> = observations
        .iter()
        .enumerate()
        .filter(|(i, _)| i % 3 != 0)
        .map(|(_, o)| *o)
        .collect();
    let a = std::f64::consts::FRAC_1_SQRT_2;
    let query = SearchNeighborhood::new(
        23,
        Anisotropy::new([[a, a, 0.], [-a, a, 0.], [0., 0., 1.]], [4., 2., 0.75]).unwrap(),
        Some(5),
        vec![
            VariableLimit {
                variable: 0,
                maximum: 7,
            },
            VariableLimit {
                variable: 2,
                maximum: 13,
            },
        ],
    )
    .unwrap();
    let mut actual = SearchWorkspace::new(&query, &budget).unwrap();
    let mut reference = SearchWorkspace::new(&query, &budget).unwrap();
    for p in [[0., 0., 0.], [3., 2., 0.], [10., 5., 0.5]] {
        equal(
            store
                .snapshot()
                .unwrap()
                .search(point(p), &mut actual, &Cancellation::default())
                .unwrap(),
            visible
                .as_slice()
                .search(point(p), &mut reference, &Cancellation::default())
                .unwrap(),
        );
    }
}

#[test]
fn internal_search_preserves_adaptive_minimums_diagnostics_and_cancellation() {
    use terrustrial_core::spatial::{MinimumQuotas, VariableMinimum};
    let dir = tempfile::tempdir().unwrap();
    let budget = MemoryBudget::new(16 << 20);
    let data = records(777);
    let store = build(&dir.path().join("adaptive"), &data, |_| 0, &budget);
    let snapshot = store.snapshot().unwrap();
    let a = std::f64::consts::FRAC_1_SQRT_2;
    let query = SearchNeighborhood::new(
        23,
        Anisotropy::new([[a, a, 0.], [-a, a, 0.], [0., 0., 1.]], [0.5, 0.25, 0.1]).unwrap(),
        Some(7),
        vec![
            VariableLimit {
                variable: 0,
                maximum: 11,
            },
            VariableLimit {
                variable: 2,
                maximum: 13,
            },
        ],
    )
    .unwrap()
    .with_minimums(MinimumQuotas {
        total: 12,
        variables: vec![
            VariableMinimum {
                variable: 0,
                minimum: 4,
            },
            VariableMinimum {
                variable: 2,
                minimum: 4,
            },
        ],
        octants: [0; 8],
    })
    .unwrap()
    .with_expansion(vec![[4., 2., 0.75], [16., 8., 3.]])
    .unwrap();
    let mut actual = SearchWorkspace::new(&query, &budget).unwrap();
    let mut reference = SearchWorkspace::new(&query, &budget).unwrap();
    let cancel = Cancellation::default();
    for p in [[0., 0., 0.], [3., 2., 0.], [10., 5., 0.5], [500.; 3]] {
        store.clear_cache().unwrap();
        match (
            snapshot.search(point(p), &mut actual, &cancel),
            data.as_slice().search(point(p), &mut reference, &cancel),
        ) {
            (Ok(a), Ok(b)) => equal(a, b),
            (Err(Error::InsufficientNeighborhood), Err(Error::InsufficientNeighborhood)) => (),
            (a, b) => panic!("packed/scan result mismatch: {a:?}, {b:?}"),
        }
        assert_eq!(actual.diagnostics(), reference.diagnostics());
    }
    cancel.cancel();
    assert!(matches!(
        snapshot.search(point([0.; 3]), &mut actual, &cancel),
        Err(Error::Cancelled)
    ));
    assert!(!actual.diagnostics().complete);
}

#[test]
fn unchanged_payloads_survive_publication_without_staling_historical_or_external_views() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("payload-generations");
    let budget = MemoryBudget::new(16 << 20);
    let cancel = Cancellation::default();
    let opts = Options {
        leaf_records: 8,
        cached_pages: 128,
        ..options()
    };
    let data: Vec<_> = (0..128)
        .map(|i| {
            Observation::new(
                i,
                Site::point(point([i as f64 * 10., 0., 0.]), 0),
                i as f64,
                0.,
            )
            .unwrap()
        })
        .collect();
    let mut builder = PackedBuilder::create(&path, opts, &budget).unwrap();
    for o in &data {
        let rank = match o.id() {
            120 => 1,
            121 => 2,
            _ => 0,
        };
        builder.push(*o, rank, &cancel).unwrap();
    }
    let store = builder.finish(&cancel).unwrap();
    let old = store.snapshot().unwrap();
    let query = |snapshot: &Snapshot, id: usize| snapshot.at_site(data[id].site()).unwrap();
    equal(&query(&old, 0), &data[..1]);
    // Warm the changed leaf too, through a visible neighbor.
    equal(&query(&old, 122), &data[122..123]);
    assert!(query(&old, 120).is_empty());
    let next = store.publish_group(1, &[(120, -0.)], &cancel).unwrap();
    let reads = store.statistics().page_reads;
    equal(&query(&next, 0), &data[..1]);
    assert_eq!(store.statistics().page_reads, reads);
    assert_eq!(query(&next, 120)[0].value().to_bits(), (-0.0_f64).to_bits());
    assert_eq!(
        store.statistics().page_reads,
        reads,
        "committed payload is already decoded"
    );
    assert!(query(&old, 120).is_empty());

    // Rejected publication leaves both cached data and the visible prefix intact.
    assert!(store.publish_group(2, &[(121, f64::NAN)], &cancel).is_err());
    let reads = store.statistics().page_reads;
    equal(&query(&store.snapshot().unwrap(), 0), &data[..1]);
    assert_eq!(store.statistics().page_reads, reads);

    let other = PackedStore::open(&path, opts, &budget).unwrap();
    let external = other.publish_group(2, &[(121, 987.)], &cancel).unwrap();
    let fresh = store.snapshot().unwrap();
    assert_eq!(query(&fresh, 121)[0].value(), 987.);
    assert!(query(&next, 121).is_empty());
    assert_eq!(query(&next, 120)[0].value().to_bits(), (-0.0_f64).to_bits());
    store.clear_cache().unwrap();
    assert_eq!(query(&fresh, 121)[0].value(), 987.);
    assert!(query(&old, 121).is_empty());
    drop((old, next, fresh, external, other, store));
    assert_eq!(budget.used(), 0);
}

#[test]
fn historical_snapshots_share_actual_payload_versions_after_eviction() {
    let dir = tempfile::tempdir().unwrap();
    let budget = MemoryBudget::new(16 << 20);
    let cancel = Cancellation::default();
    let opts = Options {
        leaf_records: 4,
        cached_pages: 128,
        ..options()
    };
    let mut builder =
        PackedBuilder::create(dir.path().join("version-intervals"), opts, &budget).unwrap();
    let data: Vec<_> = (0..16)
        .map(|id| {
            Observation::new(
                id,
                Site::point(point([id as f64 * 100., 0., 0.]), 0),
                id as f64,
                0.,
            )
            .unwrap()
        })
        .collect();
    for o in &data {
        builder
            .push(
                *o,
                match o.id() {
                    0 => 1,
                    15 => 2,
                    1 => 3,
                    _ => 0,
                },
                &cancel,
            )
            .unwrap();
    }
    let store = builder.finish(&cancel).unwrap();
    let initial = store.snapshot().unwrap();
    let first = store.publish_group(1, &[(0, -0.)], &cancel).unwrap();
    let second = store.publish_group(2, &[(15, 200.)], &cancel).unwrap();
    let third = store.publish_group(3, &[(1, 300.)], &cancel).unwrap();
    store.clear_cache().unwrap();
    assert_eq!(
        second.at_site(data[0].site()).unwrap()[0].value().to_bits(),
        (-0.0_f64).to_bits()
    );
    let reads = store.statistics().page_reads;
    assert_eq!(
        first.at_site(data[0].site()).unwrap()[0].value().to_bits(),
        (-0.0_f64).to_bits()
    );
    assert_eq!(
        store.statistics().page_reads,
        reads,
        "generations one and two share the same physical payload version"
    );
    assert!(initial.at_site(data[0].site()).unwrap().is_empty());
    assert!(second.at_site(data[1].site()).unwrap().is_empty());
    assert_eq!(third.at_site(data[1].site()).unwrap()[0].value(), 300.);
    // A failed suffix cannot install speculative cache values.
    assert!(store.publish_groups(&[(4, &[(2, 999.)])], &cancel).is_err());
    assert_eq!(
        store.snapshot().unwrap().at_site(data[2].site()).unwrap()[0].value(),
        2.
    );
    drop((initial, first, second, third, store));
    assert_eq!(budget.used(), 0);
}

#[test]
fn publication_cursor_rejects_bad_prefixes_atomically_and_allows_rank_gaps() {
    let dir = tempfile::tempdir().unwrap();
    let budget = MemoryBudget::new(16 << 20);
    let cancel = Cancellation::default();
    let data = records(12);
    let store = build(
        &dir.path().join("prefix-cursor"),
        &data,
        |i| [2, 7, 11][i / 4],
        &budget,
    );
    let values: Vec<Vec<_>> = data
        .chunks(4)
        .map(|part| part.iter().map(|o| (o.id(), -0.)).collect())
        .collect();
    let mut duplicate = values[0].clone();
    duplicate[1] = duplicate[0];
    let mut nonfinite = values[1].clone();
    nonfinite[3].1 = f64::NAN;
    let invalid: Vec<Vec<(u64, &[(u64, f64)])>> = vec![
        vec![(7, &values[1])],
        vec![(2, &values[0]), (11, &values[2])],
        vec![(2, &values[0][..3]), (7, &values[1])],
        vec![(2, &values[0]), (7, &values[1][..3])],
        vec![(2, &values[0]), (7, &[])],
        vec![(2, &duplicate)],
        vec![(2, &values[0]), (2, &values[0])],
        vec![(2, &values[0]), (7, &nonfinite)],
    ];
    for groups in invalid {
        assert!(store.publish_groups(&groups, &cancel).is_err());
        let view = store.snapshot().unwrap();
        assert_eq!(view.generation(), 0);
        assert_eq!(view.completed_group(), 0);
        assert!(view.is_empty());
    }
    let first = store
        .publish_groups(&[(2, &values[0]), (7, &values[1])], &cancel)
        .unwrap();
    assert_eq!(first.completed_group(), 7);
    assert_eq!(first.len(), 8);
    let final_view = store.publish_group(11, &values[2], &cancel).unwrap();
    assert_eq!(final_view.len(), 12);
    assert_eq!(first.len(), 8);
    drop((first, final_view, store));
    assert_eq!(budget.used(), 0);
}

#[test]
fn compact_versions_stay_inline_in_new_large_leaf_stores_and_old_layouts_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let data = records(256);
    let cancel = Cancellation::default();
    let opts = Options {
        leaf_records: 256,
        cached_pages: 64,
        ..options()
    };
    for old_layout in [false, true] {
        let budget = MemoryBudget::new(16 << 20);
        let path = dir.path().join(format!("inline-{old_layout}"));
        let mut builder = PackedBuilder::create(&path, opts, &budget).unwrap();
        for o in &data {
            builder.push(*o, 1, &cancel).unwrap();
        }
        drop(builder.finish(&cancel).unwrap());
        assert_eq!(budget.used(), 0);
        let c = rusqlite::Connection::open(&path).unwrap();
        if old_layout {
            // Recreate the previous physical schema as well as its page size.
            c.execute_batch("ALTER TABLE payloads RENAME TO payloads_rowid;
                CREATE TABLE payloads(page INTEGER NOT NULL,generation INTEGER NOT NULL,body BLOB NOT NULL,crc INTEGER NOT NULL,encoding INTEGER NOT NULL DEFAULT 0,PRIMARY KEY(page,generation)) WITHOUT ROWID;
                INSERT INTO payloads SELECT page,generation,body,crc,encoding FROM payloads_rowid;
                DROP TABLE payloads_rowid;").unwrap();
            c.execute_batch("PRAGMA journal_mode=DELETE; PRAGMA page_size=4096; VACUUM;")
                .unwrap();
        }
        let page_size: i64 = c
            .pragma_query_value(None, "page_size", |r| r.get(0))
            .unwrap();
        assert_eq!(page_size, if old_layout { 4096 } else { 16384 });
        let overflow = || {
            c.query_row(
                "SELECT count(*) FROM dbstat WHERE name='payloads' AND pagetype='overflow'",
                [],
                |r| r.get::<_, i64>(0),
            )
            .unwrap()
        };
        let before = overflow();
        let store = PackedStore::open(&path, opts, &budget).unwrap();
        let old = store.snapshot().unwrap();
        let values: Vec<_> = data.iter().map(|o| (o.id(), -0.)).collect();
        let view = store.publish_group(1, &values, &cancel).unwrap();
        assert!(old.is_empty());
        assert_eq!(view.len(), 256);
        if old_layout {
            assert!(overflow() > before);
        } else {
            assert_eq!(
                overflow(),
                before,
                "compact values must not allocate overflow pages"
            );
        }
        let mut expected = data.clone();
        for o in &mut expected {
            *o = Observation::new(o.id(), *o.site(), -0., o.noise_variance()).unwrap();
        }
        store.clear_cache().unwrap();
        check(&view, &expected);
        drop((view, old, store, c));
        assert_eq!(budget.used(), 0);
        let reopened = PackedStore::open(&path, opts, &budget).unwrap();
        check(&reopened.snapshot().unwrap(), &expected);
        drop(reopened);
        assert_eq!(budget.used(), 0);
    }
}

#[test]
fn multi_group_publication_is_atomic_and_rewrites_each_payload_once() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bulk-publish");
    let budget = MemoryBudget::new(16 << 20);
    let cancel = Cancellation::default();
    let data = records(16);
    let store = build(&path, &data, |i| if i < 8 { 1 } else { 3 }, &budget);
    let old = store.snapshot().unwrap();
    let first: Vec<_> = data[..8].iter().map(|o| (o.id(), -0.)).collect();
    let second: Vec<_> = data[8..].iter().map(|o| (o.id(), 42.)).collect();
    assert!(store
        .publish_groups(&[(1, &first), (3, &second[..7])], &cancel)
        .is_err());
    assert_eq!(store.snapshot().unwrap().completed_group(), 0);
    let new = store
        .publish_groups(&[(1, &first), (3, &second)], &cancel)
        .unwrap();
    assert_eq!(new.completed_group(), 3);
    assert!(old.is_empty());
    let mut count = 0;
    new.scan(&mut |o| {
        count += 1;
        assert!(o.value().to_bits() == (-0.0_f64).to_bits() || o.value() == 42.);
        Ok(())
    })
    .unwrap();
    assert_eq!(count, 16);
    let connection = rusqlite::Connection::open(&path).unwrap();
    let duplicate_pages: i64 = connection.query_row("SELECT count(*) FROM (SELECT page FROM payloads WHERE generation=?1 GROUP BY page HAVING count(*)>1)", [new.generation() as i64], |r| r.get(0)).unwrap();
    assert_eq!(duplicate_pages, 0);
    assert_eq!(new.generation(), old.generation() + 1);
    drop((connection, old, new, store));
    assert_eq!(budget.used(), 0);
}

#[test]
fn publication_failure_after_staging_pages_cannot_poison_retry_cache() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("staged-rollback");
    let budget = MemoryBudget::new(16 << 20);
    let cancel = Cancellation::default();
    let data = records(96);
    let store = build(&path, &data, |_| 1, &budget);
    let c = rusqlite::Connection::open(&path).unwrap();
    let (page, body): (i64, Vec<u8>) = c
        .query_row(
            "SELECT page,body FROM payloads ORDER BY page DESC LIMIT 1",
            [],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .unwrap();
    c.execute(
        "UPDATE payloads SET body=zeroblob(65536) WHERE page=?1",
        [page],
    )
    .unwrap();
    let first: Vec<_> = data.iter().map(|o| (o.id(), 42.)).collect();
    assert!(store.publish_group(1, &first, &cancel).is_err());
    assert_eq!(store.snapshot().unwrap().completed_group(), 0);
    c.execute(
        "UPDATE payloads SET body=?1 WHERE page=?2",
        rusqlite::params![body, page],
    )
    .unwrap();
    let retry: Vec<_> = data.iter().map(|o| (o.id(), 84.)).collect();
    let snapshot = store.publish_group(1, &retry, &cancel).unwrap();
    let mut count = 0;
    snapshot
        .scan(&mut |o| {
            assert_eq!(o.value(), 84.);
            count += 1;
            Ok(())
        })
        .unwrap();
    assert_eq!(count, data.len());
    drop((c, snapshot, store));
    assert_eq!(budget.used(), 0);
}

#[test]
fn scheduled_writer_views_include_pending_values_and_survive_commits_and_drop() {
    use terrustrial_core::{
        data::{Group, Target},
        simulation::parallel::SnapshotStore,
        storage::packed::{ScheduledOptions, ScheduledWriter},
    };
    let dir = tempfile::tempdir().unwrap();
    let budget = MemoryBudget::new(16 << 20);
    let cancel = Cancellation::default();
    let data = records(40);
    let store = build(&dir.path().join("queued"), &data, |i| i as u64 + 1, &budget);
    let mut writer = ScheduledWriter::new(
        store.clone(),
        ScheduledOptions {
            maximum_group_observations: 1,
            queued_groups: 4,
            commit_groups: 3,
            retained_groups: 12,
        },
        &budget,
        &cancel,
    )
    .unwrap();
    let mut retained = None;
    for (i, original) in data.iter().enumerate() {
        let group = Group {
            id: i as u64,
            targets: vec![Target {
                id: original.id(),
                site: *original.site(),
            }],
        };
        let observation = Observation::new(
            original.id(),
            *original.site(),
            if i == 0 { -0. } else { i as f64 + 100. },
            original.noise_variance(),
        )
        .unwrap();
        writer.append_group(&group, &[observation]).unwrap();
        if i == 1 {
            assert_eq!(
                writer.durable_group().unwrap(),
                0,
                "partial batch remains in the bounded overlay"
            );
            retained = Some(writer.read_view().unwrap());
            drop(writer.read_view().unwrap());
            let measured = allocation_counter::measure(|| {
                for _ in 0..32 {
                    drop(writer.read_view().unwrap());
                }
            });
            assert_eq!(measured.count_total, 0, "recycled pending-view metadata");
        }
    }
    writer.finish().unwrap();
    assert_eq!(writer.drawn_group(), 40);
    assert_eq!(writer.durable_group().unwrap(), 40);
    let current = writer.read_view().unwrap();
    drop(writer);
    let mut current_values = Vec::new();
    current
        .scan(&mut |o| {
            current_values.push((o.id(), o.value().to_bits()));
            Ok(())
        })
        .unwrap();
    current_values.sort_unstable();
    let mut expected: Vec<_> = data
        .iter()
        .enumerate()
        .map(|(i, o)| {
            (
                o.id(),
                if i == 0 {
                    (-0.0_f64).to_bits()
                } else {
                    (i as f64 + 100.).to_bits()
                },
            )
        })
        .collect();
    expected.sort_unstable();
    assert_eq!(current_values, expected);
    let mut old_values = Vec::new();
    retained
        .as_ref()
        .unwrap()
        .scan(&mut |o| {
            old_values.push(o.id());
            Ok(())
        })
        .unwrap();
    assert_eq!(
        old_values,
        data[..2].iter().map(Observation::id).collect::<Vec<_>>()
    );
    drop((retained, current, store));
    assert_eq!(budget.used(), 0);
}

#[test]
fn per_group_acknowledgment_and_writer_failure_keep_the_durable_prefix() {
    use terrustrial_core::{
        data::{Group, Target},
        simulation::parallel::SnapshotStore,
        storage::packed::{ScheduledOptions, ScheduledWriter},
    };
    let dir = tempfile::tempdir().unwrap();
    let budget = MemoryBudget::new(16 << 20);
    let cancel = Cancellation::default();
    let data = records(4);
    let store = build(
        &dir.path().join("writer-failure"),
        &data,
        |i| i as u64 + 1,
        &budget,
    );
    let mut writer = ScheduledWriter::new(
        store.clone(),
        ScheduledOptions {
            maximum_group_observations: 1,
            queued_groups: 2,
            commit_groups: 1,
            retained_groups: 4,
        },
        &budget,
        &cancel,
    )
    .unwrap();
    let group = |id, o: &Observation| Group {
        id,
        targets: vec![Target {
            id: o.id(),
            site: *o.site(),
        }],
    };
    writer
        .append_group(&group(0, &data[0]), &data[..1])
        .unwrap();
    assert_eq!(
        writer.last_durable_group(),
        1,
        "single-group mode acknowledges the commit before returning"
    );
    // The writer validates pre-indexed identities independently of the caller's
    // targets. This observation is scheduled for rank 4, not rank 2.
    assert!(writer
        .append_group(&group(1, &data[3]), &data[3..])
        .is_err());
    assert!(writer.finish().is_err());
    assert_eq!(writer.last_durable_group(), 1);
    assert_eq!(store.snapshot().unwrap().len(), 1);
    drop((writer, store));
    assert_eq!(budget.used(), 0);
}

#[test]
fn large_scheduled_batches_flush_suffix_preserve_views_and_reopen_durably() {
    use terrustrial_core::{
        data::{Group, Target},
        simulation::parallel::SnapshotStore,
        storage::packed::{ScheduledOptions, ScheduledWriter},
    };
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("large-scheduled-batches");
    let budget = MemoryBudget::new(16 << 20);
    let cancel = Cancellation::default();
    let data = records(2081); // Four full 512-group transactions and a suffix.
    let store = build(&path, &data, |i| i as u64 + 1, &budget);
    let settings = ScheduledOptions {
        maximum_group_observations: 1,
        commit_groups: 512,
        queued_groups: 1024,
        retained_groups: 1089,
    };
    let denied = MemoryBudget::new(1);
    assert!(matches!(
        ScheduledWriter::new(store.clone(), settings, &denied, &cancel),
        Err(Error::MemoryBudget { .. })
    ));
    assert_eq!(denied.used(), 0);
    let mut writer = ScheduledWriter::new(store.clone(), settings, &budget, &cancel).unwrap();
    let mut old = None;
    for (i, o) in data.iter().enumerate() {
        let group = Group {
            id: i as u64,
            targets: vec![Target {
                id: o.id(),
                site: *o.site(),
            }],
        };
        writer.append_group(&group, &[*o]).unwrap();
        if i == 16 {
            old = Some(writer.read_view().unwrap());
        }
        if i == 510 {
            assert_eq!(writer.durable_group().unwrap(), 0);
        }
    }
    writer.finish().unwrap();
    assert_eq!(writer.durable_group().unwrap(), data.len() as u64);
    assert_eq!(
        store.snapshot().unwrap().generation(),
        5,
        "backpressure must not split full commit batches"
    );
    let mut retained = Vec::new();
    old.as_ref()
        .unwrap()
        .scan(&mut |o| {
            retained.push(*o);
            Ok(())
        })
        .unwrap();
    equal(&retained, &data[..17]);
    check(&store.snapshot().unwrap(), &data);
    drop((writer, old, store));
    assert_eq!(budget.used(), 0);
    let reopened = PackedStore::open(&path, options(), &budget).unwrap();
    assert_eq!(
        reopened.snapshot().unwrap().completed_group(),
        data.len() as u64
    );
    check(&reopened.snapshot().unwrap(), &data);
    drop(reopened);
    assert_eq!(budget.used(), 0);
}

#[test]
fn scheduled_chunk_views_survive_partial_finish_and_resumed_rank_gaps() {
    use terrustrial_core::{
        data::{Group, Target},
        simulation::parallel::SnapshotStore,
        storage::packed::{ScheduledOptions, ScheduledWriter},
    };
    for commit_groups in [1, 7, 32, 64] {
        let dir = tempfile::tempdir().unwrap();
        let budget = MemoryBudget::new(16 << 20);
        let data = records(139);
        let store = build(
            &dir.path().join("resumed"),
            &data,
            |i| 3 * i as u64 + 2,
            &budget,
        );
        let mut writer = ScheduledWriter::new(
            store.clone(),
            ScheduledOptions {
                maximum_group_observations: 1,
                commit_groups,
                queued_groups: 128,
                retained_groups: 256,
            },
            &budget,
            &Cancellation::default(),
        )
        .unwrap();
        let mut views = Vec::new();
        writer.finish().unwrap();
        writer.finish().unwrap();
        for (i, o) in data.iter().enumerate() {
            if i <= 1 {
                // Exercise both first-chunk admission and copy-on-write of the
                // tail retained by view zero. Failure accepts no new group.
                let occupied = budget.reserve(budget.limit() - budget.used()).unwrap();
                let before = writer.drawn_group();
                assert!(matches!(
                    writer.append_group(
                        &Group {
                            id: 3 * i as u64 + 1,
                            targets: vec![Target {
                                id: o.id(),
                                site: *o.site()
                            }],
                        },
                        &[*o]
                    ),
                    Err(Error::MemoryBudget { .. })
                ));
                assert_eq!(writer.drawn_group(), before);
                drop(occupied);
            }
            writer
                .append_group(
                    &Group {
                        id: 3 * i as u64 + 1,
                        targets: vec![Target {
                            id: o.id(),
                            site: *o.site(),
                        }],
                    },
                    &[*o],
                )
                .unwrap();
            if [0, 15, 31, 32, 36, 63, 64, 96, 138].contains(&i) {
                views.push((i + 1, writer.read_view().unwrap()));
            }
            if i == 36 || i == 96 {
                writer.finish().unwrap();
                writer.finish().unwrap();
                check(&writer.read_view().unwrap(), &data[..=i]);
            }
        }
        writer.finish().unwrap();
        for (count, view) in &views {
            check(view, &data[..*count]);
        }
        check(&store.snapshot().unwrap(), &data);
        drop((writer, store));
        for (count, view) in &views {
            check(view, &data[..*count]);
        }
        drop(views);
        assert_eq!(budget.used(), 0);
    }
}

#[test]
fn retained_generations_share_directory_budget_and_external_appends_change_geometry() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("shared-geometry");
    let budget = MemoryBudget::new(16 << 20);
    let data = records(256);
    let store = build(
        &path,
        &data,
        |i| if i < 128 { 0 } else { (i - 128) as u64 / 8 + 1 },
        &budget,
    );
    let mut snapshots = vec![store.snapshot().unwrap()];
    let initial = budget.used();
    for group in 1..=16 {
        let values: Vec<_> = data[128 + (group - 1) * 8..128 + group * 8]
            .iter()
            .map(|o| (o.id(), o.value()))
            .collect();
        snapshots.push(
            store
                .publish_group(group as u64, &values, &Cancellation::default())
                .unwrap(),
        );
        store.clear_cache().unwrap();
        // Retained value generations need only small manifests; immutable branch
        // directories must not be charged and copied again for every generation.
        assert!(budget.used() <= initial + group as u64 * 4096);
    }
    for i in [0, 8, 16] {
        check(&snapshots[i], &data[..128 + i * 8]);
    }
    let other = PackedStore::open(&path, options(), &budget).unwrap();
    let added = Observation::new(1, Site::point(point([100.; 3]), 0), 7., 0.).unwrap();
    let latest = other
        .append_batch([Ok(added)], &Cancellation::default())
        .unwrap();
    let current = store.snapshot().unwrap();
    let mut all = data.clone();
    all.push(added);
    check(&current, &all);
    check(snapshots.last().unwrap(), &data);
    drop((snapshots, current, latest, other, store));
    assert_eq!(budget.used(), 0);
}

#[test]
fn publication_is_atomic_and_old_snapshots_survive_compaction_and_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("versions.sqlite");
    let budget = MemoryBudget::new(16 << 20);
    let original = records(149);
    let store = build(
        &path,
        &original,
        |i| {
            if i < 20 {
                0
            } else if i < 81 {
                3
            } else {
                9
            }
        },
        &budget,
    );
    let old = store.snapshot().unwrap();
    check(&old, &original[..20]);
    let values: Vec<_> = original[20..81]
        .iter()
        .enumerate()
        .map(|(i, o)| (o.id(), if i == 0 { -0. } else { i as f64 + 100. }))
        .collect();
    assert!(store
        .publish_group(9, &[], &Cancellation::default())
        .is_err());
    assert!(store
        .publish_group(3, &values[..values.len() - 1], &Cancellation::default())
        .is_err());
    let mut duplicates = values.clone();
    duplicates[1] = duplicates[0];
    assert!(store
        .publish_group(3, &duplicates, &Cancellation::default())
        .is_err());
    assert_eq!(store.snapshot().unwrap().generation(), 0);
    let next = store
        .publish_group(3, &values, &Cancellation::default())
        .unwrap();
    let mut current = original.clone();
    for (o, (_, v)) in current[20..81].iter_mut().zip(values) {
        *o = Observation::new(o.id(), *o.site(), v, o.noise_variance()).unwrap();
    }
    check(&old, &original[..20]);
    check(&next, &current[..81]);
    let extra = Observation::new(17, Site::point(point([0.1, 0.1, 0.]), 3), 17., 0.).unwrap();
    let appended = store
        .append_batch([Ok(extra)], &Cancellation::default())
        .unwrap();
    let mut visible = current[..81].to_vec();
    visible.push(extra);
    check(&appended, &visible);
    assert!(store
        .append_batch([Ok(extra)], &Cancellation::default())
        .is_err());
    let compact = store.compact(&Cancellation::default()).unwrap();
    check(&compact, &visible);
    check(&next, &current[..81]);
    check(&old, &original[..20]);
    let reopened = PackedStore::open(&path, options(), &budget).unwrap();
    check(&reopened.snapshot().unwrap(), &visible);
    let exported = store
        .export_compacted(
            dir.path().join("export.sqlite"),
            options(),
            &budget,
            &Cancellation::default(),
        )
        .unwrap();
    assert_eq!(exported.snapshot().unwrap().completed_group(), 3);
    check(&exported.snapshot().unwrap(), &visible);
    let last: Vec<_> = original[81..]
        .iter()
        .map(|o| (o.id(), o.value() + 1.))
        .collect();
    exported
        .publish_group(9, &last, &Cancellation::default())
        .unwrap();
    check(&old, &original[..20]);
}

#[test]
fn bounded_segments_consolidate_without_losing_old_views() {
    let dir = tempfile::tempdir().unwrap();
    let budget = MemoryBudget::new(8 << 20);
    let store = build(&dir.path().join("segments.sqlite"), &[], |_| 0, &budget);
    let observations = records(35);
    let empty = store.snapshot().unwrap();
    for o in &observations {
        store
            .append_batch([Ok(*o)], &Cancellation::default())
            .unwrap();
    }
    check(&store.snapshot().unwrap(), &observations);
    check(&empty, &[]);
}

#[test]
fn parallel_readers_see_a_stable_prefix_during_publication() {
    let dir = tempfile::tempdir().unwrap();
    let budget = MemoryBudget::new(16 << 20);
    let original = records(900);
    let store = build(
        &dir.path().join("parallel.sqlite"),
        &original,
        |i| if i < 400 { 0 } else { 1 },
        &budget,
    );
    let snapshot = store.snapshot().unwrap();
    std::thread::scope(|scope| {
        for _ in 0..4 {
            let snapshot = &snapshot;
            let original = &original;
            scope.spawn(move || check(snapshot, &original[..400]));
        }
        let values: Vec<_> = original[400..]
            .iter()
            .map(|o| (o.id(), o.value()))
            .collect();
        store
            .publish_group(1, &values, &Cancellation::default())
            .unwrap();
    });
    check(&store.snapshot().unwrap(), &original);
}

#[test]
fn cancelled_build_writes_and_query_recover_and_never_overwrite() {
    let dir = tempfile::tempdir().unwrap();
    let budget = MemoryBudget::new(8 << 20);
    let path = dir.path().join("abort.sqlite");
    let observations = records(90);
    let cancelled = Cancellation::default();
    cancelled.cancel();
    let mut b = PackedBuilder::create(&path, options(), &budget).unwrap();
    b.push(observations[0], 0, &Cancellation::default())
        .unwrap();
    assert!(matches!(b.finish(&cancelled), Err(Error::Cancelled)));
    assert!(!path.exists());
    assert_eq!(budget.used(), 0);
    let store = build(&path, &observations, |_| 0, &budget);
    let mut duplicate = PackedBuilder::create(&path, options(), &budget).unwrap();
    duplicate
        .push(observations[0], 0, &Cancellation::default())
        .unwrap();
    assert!(duplicate.finish(&Cancellation::default()).is_err());
    check(&store.snapshot().unwrap(), &observations);
    let before = store.snapshot().unwrap().generation();
    assert!(matches!(
        store.append_batch(std::iter::empty(), &cancelled),
        Err(Error::Cancelled)
    ));
    assert_eq!(store.snapshot().unwrap().generation(), before);
    let mut workspace = NearestWorkspace::new(
        Neighborhood {
            maximum: 7,
            radius: 10.,
        },
        &budget,
    )
    .unwrap();
    assert!(matches!(
        store
            .snapshot()
            .unwrap()
            .nearest_into(point([0.; 3]), &mut workspace, &cancelled),
        Err(Error::Cancelled)
    ));
    check(&store.snapshot().unwrap(), &observations);
}

#[test]
fn checksums_and_budget_exhaustion_fail_explicitly() {
    let dir = tempfile::tempdir().unwrap();
    let budget = MemoryBudget::new(8 << 20);
    assert!(matches!(
        PackedBuilder::create(dir.path().join("denied"), options(), &MemoryBudget::new(1)),
        Err(Error::MemoryBudget { .. })
    ));
    let path = dir.path().join("corrupt.sqlite");
    let store = build(&path, &records(20), |_| 0, &budget);
    let c = rusqlite::Connection::open(&path).unwrap();
    c.execute("UPDATE payloads SET crc=crc+1", []).unwrap();
    assert!(store
        .snapshot()
        .unwrap()
        .nearest(
            point([0.; 3]),
            Neighborhood {
                maximum: 1,
                radius: 100.
            },
            &Cancellation::default()
        )
        .is_err());
    c.execute("UPDATE manifest SET roots_crc=roots_crc+1", [])
        .unwrap();
    assert!(store.snapshot().is_err());
}

#[test]
fn randomized_geometry_and_radius_boundaries_match_independent_full_sort() {
    use rand::{RngExt, SeedableRng};
    let dir = tempfile::tempdir().unwrap();
    let budget = MemoryBudget::new(32 << 20);
    let cancel = Cancellation::default();
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(908172);
    for (case, leaf, directory, scale) in [
        (0, 1, 0, 1.),
        (1, 7, 4, 1e-170),
        (2, 64, 0, 1e170),
        (3, 256, 32, 1.),
    ] {
        let observations: Vec<_> = (0..769)
            .map(|i| {
                let xyz = match i % 5 {
                    0 => [0.; 3],
                    1 => [rng.random_range(-30.0..30.0), 0., 0.],
                    _ => std::array::from_fn(|_| rng.random_range(-30.0..30.0)),
                };
                Observation::new(
                    u64::MAX - i,
                    Site::point(point(xyz.map(|x| x * scale)), (i % 4) as usize),
                    i as f64,
                    0.,
                )
                .unwrap()
            })
            .collect();
        let mut b = PackedBuilder::create(
            dir.path().join(format!("random-{case}")),
            Options {
                leaf_records: leaf,
                directory_pages: directory,
                cached_pages: 3,
                ..options()
            },
            &budget,
        )
        .unwrap();
        for o in observations.iter().rev() {
            b.push(*o, 0, &cancel).unwrap();
        }
        let store = b.finish(&cancel).unwrap();
        let snapshot = store.snapshot().unwrap();
        for i in 0..96 {
            let p = point(std::array::from_fn(|_| {
                rng.random_range(-35.0..35.0) * scale
            }));
            let distance = p.distance(
                observations[i * 7 % observations.len()]
                    .site()
                    .support
                    .center(),
            );
            for radius in [distance.next_down(), distance, distance.next_up()] {
                let maximum = [1, 7, 32, 257][i % 4];
                let mut w =
                    NearestWorkspace::new(Neighborhood { maximum, radius }, &budget).unwrap();
                let mut expected: Vec<_> = observations
                    .iter()
                    .copied()
                    .filter(|o| p.distance(o.site().support.center()) <= radius)
                    .collect();
                expected.sort_unstable_by(|a, b| {
                    p.distance(a.site().support.center())
                        .total_cmp(&p.distance(b.site().support.center()))
                        .then(a.id().cmp(&b.id()))
                });
                expected.truncate(maximum);
                equal(
                    snapshot.nearest_into(p, &mut w, &cancel).unwrap(),
                    &expected,
                );
            }
        }
    }
    assert_eq!(budget.used(), 0);
}

#[test]
fn mid_batch_failure_and_memory_pressure_are_recoverable() {
    let dir = tempfile::tempdir().unwrap();
    let budget = MemoryBudget::new(8 << 20);
    let data = records(150);
    let store = build(&dir.path().join("rollback"), &data[..50], |_| 0, &budget);
    let original = store.snapshot().unwrap();
    let cancel = Cancellation::default();
    let attempt = data[50..].iter().enumerate().map(|(i, o)| {
        if i == 31 {
            cancel.cancel();
        }
        Ok(*o)
    });
    assert!(matches!(
        store.append_batch(attempt, &cancel),
        Err(Error::Cancelled)
    ));
    let attempt = data[50..].iter().enumerate().map(|(i, o)| {
        if i == 31 {
            Err(Error::Backend("injected input failure".into()))
        } else {
            Ok(*o)
        }
    });
    assert!(store
        .append_batch(attempt, &Cancellation::default())
        .is_err());
    assert_eq!(
        store.snapshot().unwrap().generation(),
        original.generation()
    );
    check(&original, &data[..50]);
    let mut workspace = NearestWorkspace::new(
        Neighborhood {
            maximum: 7,
            radius: 100.,
        },
        &budget,
    )
    .unwrap();
    store.clear_cache().unwrap();
    let held = budget.reserve(budget.limit() - budget.used()).unwrap();
    assert!(matches!(
        original.nearest_into(point([0.; 3]), &mut workspace, &Cancellation::default()),
        Err(Error::MemoryBudget { .. })
    ));
    drop(held);
    check(&original, &data[..50]);
    let updated = store
        .append_batch(data[50..].iter().copied().map(Ok), &Cancellation::default())
        .unwrap();
    check(&updated, &data);
    check(&original, &data[..50]);
}

#[test]
fn warmed_nearest_and_constrained_queries_do_not_allocate() {
    let dir = tempfile::tempdir().unwrap();
    let budget = MemoryBudget::new(16 << 20);
    let mut b = PackedBuilder::create(
        dir.path().join("allocations"),
        Options {
            cached_pages: 1024,
            ..options()
        },
        &budget,
    )
    .unwrap();
    for o in records(2000) {
        b.push(o, 0, &Cancellation::default()).unwrap();
    }
    let store = b.finish(&Cancellation::default()).unwrap();
    let snapshot = store.snapshot().unwrap();
    let cancel = Cancellation::default();
    let mut w = NearestWorkspace::new(
        Neighborhood {
            maximum: 32,
            radius: 5.,
        },
        &budget,
    )
    .unwrap();
    let queries: Vec<_> = (0..40)
        .map(|i| point([(i % 19) as f64 + 0.1, (i % 13) as f64 + 0.2, 0.3]))
        .collect();
    let query =
        SearchNeighborhood::new(32, Anisotropy::isotropic(5.).unwrap(), None, vec![]).unwrap();
    let mut search = SearchWorkspace::new(&query, &budget).unwrap();
    for p in &queries {
        snapshot.nearest_into(*p, &mut w, &cancel).unwrap();
        snapshot.search(*p, &mut search, &cancel).unwrap();
    }
    let reads = store.statistics().page_reads;
    let measured = allocation_counter::measure(|| {
        for _ in 0..10 {
            for p in &queries {
                std::hint::black_box(snapshot.nearest_into(*p, &mut w, &cancel).unwrap());
                std::hint::black_box(snapshot.search(*p, &mut search, &cancel).unwrap());
            }
        }
    });
    assert_eq!(store.statistics().page_reads, reads);
    assert_eq!(measured.count_total, 0, "{measured:?}");
}

#[test]
fn warmed_empty_exact_site_queries_do_not_allocate_or_change_budget() {
    let dir = tempfile::tempdir().unwrap();
    let budget = MemoryBudget::new(16 << 20);
    let data = records(20);
    let store = build(&dir.path().join("empty-exact"), &data, |_| 0, &budget);
    let view = store.snapshot().unwrap();
    let mut site = *data[1].site();
    site.variable = 99;
    assert!(view.at_site(&site).unwrap().is_empty());
    let used = budget.used();
    let reads = store.statistics().page_reads;
    let measured = allocation_counter::measure(|| {
        for _ in 0..100 {
            assert!(view.at_site(&site).unwrap().is_empty());
        }
    });
    assert_eq!(measured.count_total, 0, "{measured:?}");
    assert_eq!(budget.used(), used);
    assert_eq!(store.statistics().page_reads, reads);
    drop((view, store));
    assert_eq!(budget.used(), 0);
}
#[test]
fn oversized_latest_payload_cannot_fall_back_to_an_older_version() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("oversized");
    let budget = MemoryBudget::new(8 << 20);
    let data = records(20);
    let store = build(&path, &data, |i| if i < 10 { 0 } else { 1 }, &budget);
    let old = store.snapshot().unwrap();
    let values: Vec<_> = data[10..].iter().map(|o| (o.id(), 9.)).collect();
    let new = store
        .publish_group(1, &values, &Cancellation::default())
        .unwrap();
    let c = rusqlite::Connection::open(path).unwrap();
    c.execute(
        "UPDATE payloads SET body=zeroblob(65536) WHERE generation=1",
        [],
    )
    .unwrap();
    // Publication now warms decoded pages. Force this corruption check to read disk.
    store.clear_cache().unwrap();
    assert!(new
        .nearest(
            point([0.; 3]),
            Neighborhood {
                maximum: 20,
                radius: 100.
            },
            &Cancellation::default()
        )
        .is_err());
    check(&old, &data[..10]);
    assert!(store.compact(&Cancellation::default()).is_err());
}

#[test]
fn joint_estimation_and_simulation_match_resident_conditioning() {
    use terrustrial_core::{
        data::{Group, Target},
        estimation::{estimate_groups, EstimationOptions},
        execution::ExecutionOptions,
        gaussian::{NumericalTolerance, Simple, Uncertainty},
        model::{Coregionalization, Correlation, Lmc, Structure},
        simulation::{simulate_groups, SimulationOptions},
        spatial::SpatialIndex,
    };
    let dir = tempfile::tempdir().unwrap();
    let budget = MemoryBudget::new(128 << 20);
    let cancel = Cancellation::default();
    let data = records(100);
    let store = build(&dir.path().join("gaussian"), &data, |_| 0, &budget);
    let snapshot = store.snapshot().unwrap();
    let resident = SpatialIndex::new(data.clone()).unwrap();
    let model = Lmc::new(vec![Structure {
        correlation: Correlation::Exponential,
        anisotropy: Anisotropy::isotropic(5.).unwrap(),
        coregionalization: Coregionalization::from_factor(
            4,
            4,
            &[
                1., 0., 0., 0., 0.3, 1., 0., 0., -0.2, 0.4, 1., 0., 0.1, 0.2, 0.3, 1.,
            ],
        )
        .unwrap(),
    }])
    .unwrap();
    let groups: Vec<_> = (0..12)
        .map(|i| Group {
            id: i,
            targets: (0..4)
                .map(|v| Target {
                    id: i * 4 + v as u64,
                    site: Site::point(point([i as f64 + 0.3, 0.4, 0.1]), v),
                })
                .collect(),
        })
        .collect();
    let query = Neighborhood {
        maximum: 32,
        radius: 10.,
    };
    let estimate = |source: &(dyn ObservationSource + Sync), workers| {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .unwrap();
        let mut out = Vec::new();
        pool.install(|| {
            estimate_groups(
                &model,
                source,
                groups.clone(),
                &Simple,
                EstimationOptions {
                    neighborhood: query.into(),
                    mean: None,
                    uncertainty: Uncertainty::Joint,
                    tolerance: NumericalTolerance::default(),
                    budget: &budget,
                    cancellation: &cancel,
                },
                ExecutionOptions {
                    batch_groups: 4,
                    maximum_targets_per_group: 4,
                    bytes_per_group: 8 << 20,
                },
                |result| {
                    out.extend(result.values.iter().map(|v| v.to_bits()));
                    let c = result.covariance().unwrap();
                    for j in 0..c.ncols() {
                        for i in 0..c.nrows() {
                            out.push(c[(i, j)].to_bits());
                        }
                    }
                    Ok(())
                },
            )
        })
        .unwrap();
        out
    };
    let expected = estimate(&resident, 1);
    for workers in [1, 2, 4, 8] {
        assert_eq!(estimate(&snapshot, workers), expected);
    }
    let simulate = |source: &(dyn ObservationSource + Sync)| {
        let mut simulated = SpatialIndex::new(vec![]).unwrap();
        let mut out = Vec::new();
        simulate_groups(
            &model,
            source,
            &mut simulated,
            groups.clone(),
            &Simple,
            SimulationOptions {
                seed: 81,
                realization: 2,
                mean: None,
                hard_neighborhood: query.into(),
                simulated_neighborhood: query.into(),
                budget: &budget,
                cancellation: &cancel,
            },
            |observations| {
                out.extend(observations.iter().map(|o| o.value().to_bits()));
                Ok(())
            },
        )
        .unwrap();
        out
    };
    assert_eq!(simulate(&snapshot), simulate(&resident));
}

// The child exits without running destructors, including SQLite transaction Drop.
#[test]
fn asynchronous_exit_child() {
    use terrustrial_core::{
        data::{Group, Target},
        simulation::parallel::SnapshotStore,
        storage::packed::{ScheduledOptions, ScheduledWriter},
    };
    let Ok(path) = std::env::var("TERRUSTRIAL_ASYNC_CRASH_PATH") else {
        return;
    };
    let budget = MemoryBudget::new(8 << 20);
    let cancel = Cancellation::default();
    let store = PackedStore::open(path, options(), &budget).unwrap();
    let mut writer = ScheduledWriter::new(
        store,
        ScheduledOptions {
            maximum_group_observations: 1,
            queued_groups: 4,
            commit_groups: 3,
            retained_groups: 8,
        },
        &budget,
        &cancel,
    )
    .unwrap();
    for (i, observation) in records(2).iter().enumerate() {
        let group = Group {
            id: i as u64,
            targets: vec![Target {
                id: observation.id(),
                site: *observation.site(),
            }],
        };
        writer.append_group(&group, &[*observation]).unwrap();
        if i == 0 || std::env::var("TERRUSTRIAL_ASYNC_CRASH_FLUSH").unwrap() == "yes" {
            writer.finish().unwrap();
        }
    }
    std::process::exit(75);
}

#[test]
fn asynchronous_process_exit_recovers_only_acknowledged_groups() {
    let dir = tempfile::tempdir().unwrap();
    let budget = MemoryBudget::new(8 << 20);
    let data = records(4);
    for (flush, count) in [("no", 1), ("yes", 2)] {
        let path = dir.path().join(flush);
        drop(build(&path, &data, |i| i as u64 + 1, &budget));
        let result = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "asynchronous_exit_child"])
            .env("TERRUSTRIAL_ASYNC_CRASH_PATH", &path)
            .env("TERRUSTRIAL_ASYNC_CRASH_FLUSH", flush)
            .output()
            .unwrap();
        assert_eq!(result.status.code(), Some(75), "{result:?}");
        let reopened = PackedStore::open(&path, options(), &budget).unwrap();
        let snapshot = reopened.snapshot().unwrap();
        assert_eq!(snapshot.completed_group(), count as u64);
        check(&snapshot, &data[..count]);
    }
    assert_eq!(budget.used(), 0);
}

#[test]
fn process_exit_child() {
    let Ok(path) = std::env::var("TERRUSTRIAL_PACKED_CRASH_PATH") else {
        return;
    };
    let budget = MemoryBudget::new(8 << 20);
    let store = PackedStore::open(path, options(), &budget).unwrap();
    let data = records(150);
    let commit = std::env::var("TERRUSTRIAL_PACKED_CRASH_COMMIT").unwrap() == "yes";
    store
        .append_batch(
            data[50..].iter().enumerate().map(|(i, o)| {
                if !commit && i == 31 {
                    std::process::exit(73);
                }
                Ok(*o)
            }),
            &Cancellation::default(),
        )
        .unwrap();
    std::process::exit(74);
}

#[test]
fn abrupt_process_exit_rolls_back_incomplete_writes_and_retains_commits() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("process-exit");
    let budget = MemoryBudget::new(8 << 20);
    let data = records(150);
    drop(build(&path, &data[..50], |_| 0, &budget));
    for (commit, code, count) in [("no", 73, 50), ("yes", 74, 150)] {
        let result = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "process_exit_child"])
            .env("TERRUSTRIAL_PACKED_CRASH_PATH", &path)
            .env("TERRUSTRIAL_PACKED_CRASH_COMMIT", commit)
            .output()
            .unwrap();
        assert_eq!(result.status.code(), Some(code), "{result:?}");
        let reopened = PackedStore::open(&path, options(), &budget).unwrap();
        check(&reopened.snapshot().unwrap(), &data[..count]);
        let c = rusqlite::Connection::open(&path).unwrap();
        let integrity: String = c
            .query_row("PRAGMA integrity_check", [], |r| r.get(0))
            .unwrap();
        assert_eq!(integrity, "ok");
    }
    assert_eq!(budget.used(), 0);
}

#[test]
fn exact_site_admission_obeys_budget_for_pinned_and_evictable_pages() {
    for count in [16, 600] {
        let dir = tempfile::tempdir().unwrap();
        let budget = MemoryBudget::new(16 << 20);
        let site = Site::point(point([0.; 3]), 0);
        let data: Vec<_> = (0..count)
            .map(|id| Observation::new(id, site, id as f64, 0.).unwrap())
            .collect();
        let mut b = PackedBuilder::create(
            dir.path().join("site"),
            Options {
                cached_pages: 1024,
                ..options()
            },
            &budget,
        )
        .unwrap();
        for o in &data {
            b.push(*o, 0, &Cancellation::default()).unwrap();
        }
        let store = b.finish(&Cancellation::default()).unwrap();
        let snapshot = store.snapshot().unwrap();
        equal(&snapshot.at_site(&site).unwrap(), &data);
        let held = budget.reserve(budget.limit() - budget.used()).unwrap();
        if count == 16 {
            // The only leaf and payload are pinned by this query. Eviction cannot
            // make room for even the first output-buffer allocation.
            assert!(matches!(
                snapshot.at_site(&site),
                Err(Error::MemoryBudget { .. })
            ));
        } else {
            // Other warm leaves are evictable and can fund output construction.
            equal(&snapshot.at_site(&site).unwrap(), &data);
        }
        assert!(budget.used() <= budget.limit());
        drop(held);
        equal(&snapshot.at_site(&site).unwrap(), &data);
    }
}

#[test]
fn bounds_visitors_can_reenter_queries_and_clear_cache() {
    let dir = tempfile::tempdir().unwrap();
    let budget = MemoryBudget::new(16 << 20);
    let data = records(80);
    let store = build(&dir.path().join("reenter"), &data, |_| 0, &budget);
    let snapshot = store.snapshot().unwrap();
    let mut seen = 0;
    snapshot
        .scan(&mut |_| {
            if seen == 0 {
                store.clear_cache()?;
                let result = snapshot.nearest(
                    point([0.; 3]),
                    Neighborhood {
                        maximum: 7,
                        radius: 10.,
                    },
                    &Cancellation::default(),
                )?;
                assert_eq!(result.len(), 7);
            }
            seen += 1;
            Ok(())
        })
        .unwrap();
    assert_eq!(seen, data.len());
}

#[test]
fn cached_pages_yield_to_snapshot_query_and_publication_admission() {
    let dir = tempfile::tempdir().unwrap();
    let budget = MemoryBudget::new(32 << 20);
    let input = records(4096);
    let mut opts = options();
    opts.cached_pages = 2048;
    opts.leaf_records = 32;
    let mut builder =
        PackedBuilder::create(dir.path().join("admission.sqlite"), opts, &budget).unwrap();
    let cancel = Cancellation::default();
    for (i, &o) in input.iter().enumerate() {
        builder
            .push(o, u64::from(i >= input.len() - 4), &cancel)
            .unwrap();
    }
    let store = builder.finish(&cancel).unwrap();
    let old = store.snapshot().unwrap();
    old.scan(&mut |_| Ok(())).unwrap();
    let retained = budget.used();
    let pressure = budget.reserve(budget.limit() - retained).unwrap();
    // There is no unreserved byte. These operations must evict cached pages,
    // not require the caller to clear all useful geometry and payloads manually.
    let exact = old.at_site(input[0].site()).unwrap();
    assert!(exact.iter().any(|o| o.id() == input[0].id()));
    drop(exact);
    let another = store.snapshot().unwrap();
    assert_eq!(another.len(), 4092);
    let query = Neighborhood {
        maximum: 32,
        radius: 8.,
    };
    let mut workspace = store.nearest_workspace(query).unwrap();
    let p = point([0.1, 0.2, 0.3]);
    let found = old.nearest_into(p, &mut workspace, &cancel).unwrap();
    let expected = input[..4092].nearest(p, query, &cancel).unwrap();
    assert_eq!(
        found.iter().map(Observation::id).collect::<Vec<_>>(),
        expected.iter().map(Observation::id).collect::<Vec<_>>()
    );
    let values: Vec<_> = input[4092..].iter().map(|o| (o.id(), -0.)).collect();
    let published = store.publish_group(1, &values, &cancel).unwrap();
    assert_eq!(old.len(), 4092);
    assert_eq!(published.len(), 4096);
    for o in &input[4092..] {
        assert!(old
            .at_site(o.site())
            .unwrap()
            .iter()
            .all(|v| v.id() != o.id()));
        let matches = published.at_site(o.site()).unwrap();
        assert_eq!(
            matches
                .iter()
                .find(|v| v.id() == o.id())
                .unwrap()
                .value()
                .to_bits(),
            (-0.0f64).to_bits()
        );
    }
    assert!(budget.used() <= budget.limit());
    let before = budget.used();
    assert!(matches!(
        store.nearest_workspace(Neighborhood {
            maximum: 1_000_000,
            radius: 8.
        }),
        Err(Error::MemoryBudget { .. })
    ));
    assert_eq!(
        budget.used(),
        before,
        "impossible requests must not flush the cache"
    );
    drop(pressure);
    drop(workspace);
    drop(published);
    drop(another);
    drop(old);
    drop(store);
    assert_eq!(budget.used(), 0);
}

#[test]
fn compact_value_versions_and_legacy_full_records_preserve_snapshot_and_export_bits() {
    let dir = tempfile::tempdir().unwrap();
    let cancel = Cancellation::default();
    let budget = MemoryBudget::new(32 << 20);
    let data = records(64);
    for legacy in [false, true] {
        let path = dir.path().join(format!("encoding-{legacy}"));
        drop(build(&path, &data, |i| if i < 32 { 1 } else { 3 }, &budget));
        if legacy {
            // Recreate the previous on-disk schema, whose payloads contain full
            // records and whose identity index is not covering.
            let c = rusqlite::Connection::open(&path).unwrap();
            c.execute_batch("ALTER TABLE payloads DROP COLUMN encoding; DROP INDEX identities_rank; CREATE INDEX identities_rank ON identities(rank,id); PRAGMA user_version=2;").unwrap();
        }
        let store = PackedStore::open(&path, options(), &budget).unwrap();
        let initial = store.snapshot().unwrap();
        let first_values: Vec<_> = data[..32].iter().map(|o| (o.id(), -0.)).collect();
        let first = store.publish_group(1, &first_values, &cancel).unwrap();
        let second_values: Vec<_> = data[32..].iter().map(|o| (o.id(), 42.)).collect();
        let second = store.publish_group(3, &second_values, &cancel).unwrap();
        let expected: Vec<_> = data
            .iter()
            .enumerate()
            .map(|(i, o)| {
                Observation::new(
                    o.id(),
                    *o.site(),
                    if i < 32 { -0. } else { 42. },
                    o.noise_variance(),
                )
                .unwrap()
            })
            .collect();
        store.clear_cache().unwrap();
        check(&initial, &[]);
        check(&first, &expected[..32]);
        check(&second, &expected);
        let c = rusqlite::Connection::open(&path).unwrap();
        let (base_bytes, version_bytes): (i64, i64) = c.query_row("SELECT sum(length(body)),sum((SELECT length(v.body) FROM payloads v WHERE v.page=b.page ORDER BY v.generation DESC LIMIT 1)) FROM payloads b WHERE generation=0", [], |r| Ok((r.get(0)?, r.get(1)?))).unwrap();
        assert_eq!(base_bytes, 64 * 104);
        assert_eq!(version_bytes, 64 * if legacy { 104 } else { 8 });
        let exported = store
            .export_compacted(
                dir.path().join(format!("export-{legacy}")),
                options(),
                &budget,
                &cancel,
            )
            .unwrap();
        check(&exported.snapshot().unwrap(), &expected);
        let compacted = store.compact(&cancel).unwrap();
        store.clear_cache().unwrap();
        check(&compacted, &expected);
        check(&first, &expected[..32]);
        drop((c, initial, first, second, compacted, exported, store));
        assert_eq!(budget.used(), 0);
    }
}

#[test]
fn compact_value_corruption_and_base_association_fail_closed() {
    let dir = tempfile::tempdir().unwrap();
    let budget = MemoryBudget::new(16 << 20);
    let cancel = Cancellation::default();
    let data = records(16);
    let path = dir.path().join("compact-corruption");
    let store = build(&path, &data, |_| 1, &budget);
    let values: Vec<_> = data.iter().map(|o| (o.id(), 42.)).collect();
    let view = store.publish_group(1, &values, &cancel).unwrap();
    let c = rusqlite::Connection::open(&path).unwrap();
    let original: Vec<u8> = c
        .query_row("SELECT body FROM payloads WHERE generation=1", [], |r| {
            r.get(0)
        })
        .unwrap();
    for invalid in [vec![0; 8], vec![0; 16 * 8 + 1], vec![0; 24 * 8]] {
        c.execute(
            "UPDATE payloads SET body=?1,crc=?2 WHERE generation=1",
            rusqlite::params![invalid, crc32fast::hash(&invalid)],
        )
        .unwrap();
        store.clear_cache().unwrap();
        assert!(view.scan(&mut |_| Ok(())).is_err());
    }
    c.execute(
        "UPDATE payloads SET body=?1,crc=?2 WHERE generation=1",
        rusqlite::params![original, crc32fast::hash(&original)],
    )
    .unwrap();
    c.execute("UPDATE payloads SET crc=crc+1 WHERE generation=0", [])
        .unwrap();
    store.clear_cache().unwrap();
    assert!(view.scan(&mut |_| Ok(())).is_err());
}

#[test]
fn cached_publication_preserves_shared_page_history_and_matches_cold_writes() {
    let dir = tempfile::tempdir().unwrap();
    let cancel = Cancellation::default();
    let data = records(20); // All three ranks share one payload page.
    let mut encoded = Vec::new();
    for warm in [false, true] {
        let budget = MemoryBudget::new(16 << 20);
        let path = dir.path().join(format!("cached-publication-{warm}"));
        let store = build(&path, &data, |i| (i % 3) as u64, &budget);
        let initial = store.snapshot().unwrap();
        let mut expected: Vec<_> = data.iter().step_by(3).copied().collect();
        check(&initial, &expected);
        let mut views = Vec::new();
        for rank in [1, 2] {
            if !warm {
                store.clear_cache().unwrap();
            }
            let values: Vec<_> = data
                .iter()
                .enumerate()
                .filter(|(i, _)| i % 3 == rank)
                .map(|(_, o)| (o.id(), if rank == 1 { -0. } else { 42. }))
                .collect();
            // Validation must still reject incomplete groups before publication.
            assert!(store
                .publish_group(rank as u64, &values[..values.len() - 1], &cancel)
                .is_err());
            assert_eq!(store.snapshot().unwrap().completed_group(), rank as u64 - 1);
            let view = store.publish_group(rank as u64, &values, &cancel).unwrap();
            for (i, o) in data.iter().enumerate().filter(|(i, _)| i % 3 == rank) {
                expected.push(
                    Observation::new(
                        o.id(),
                        *o.site(),
                        if i % 3 == 1 { -0. } else { 42. },
                        o.noise_variance(),
                    )
                    .unwrap(),
                );
            }
            check(&view, &expected);
            views.push((view, expected.clone()));
        }
        store.clear_cache().unwrap();
        check(
            &initial,
            &data.iter().step_by(3).copied().collect::<Vec<_>>(),
        );
        for (view, expected) in &views {
            check(view, expected);
        }
        drop((views, initial, store));
        assert_eq!(budget.used(), 0);
        let reopened = PackedStore::open(&path, options(), &budget).unwrap();
        check(&reopened.snapshot().unwrap(), &expected);
        let c = rusqlite::Connection::open(&path).unwrap();
        let bodies: Vec<(i64, Vec<u8>, u32)> = c
            .prepare("SELECT generation,body,crc FROM payloads ORDER BY page,generation")
            .unwrap()
            .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))
            .unwrap()
            .collect::<std::result::Result<_, _>>()
            .unwrap();
        encoded.push(bodies);
        drop((reopened, c));
        assert_eq!(budget.used(), 0);
    }
    assert_eq!(encoded[0], encoded[1]);
}
