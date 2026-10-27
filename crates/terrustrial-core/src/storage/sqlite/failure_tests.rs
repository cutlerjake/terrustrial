use super::*;

fn record(id: u64) -> Observation {
    Observation::new(
        id,
        Site {
            support: Support::block(
                Point3::new([id as f64 * 0.25, -0., 0.]).unwrap(),
                if id.is_multiple_of(2) {
                    [0.; 3]
                } else {
                    [0.2, -0., 0.]
                },
                if id.is_multiple_of(2) {
                    [1; 3]
                } else {
                    [2, 1, 1]
                },
            )
            .unwrap(),
            variable: 0,
        },
        -0.,
        0.,
    )
    .unwrap()
}
type Payload = (u64, usize, [u64; 8], [usize; 3]);
fn payload(o: &Observation) -> Payload {
    let [x, y, z] = o.site().support.center().coordinates();
    let [hx, hy, hz] = o.site().support.half_extents();
    (
        o.id(),
        o.site().variable,
        [x, y, z, hx, hy, hz, o.value(), o.noise_variance()].map(f64::to_bits),
        o.site().support.divisions(),
    )
}
fn records(store: &SqliteStore) -> Vec<Payload> {
    let mut out = Vec::new();
    store
        .scan(&mut |o| {
            out.push(payload(o));
            Ok(())
        })
        .unwrap();
    out.sort_unstable();
    out
}

#[test]
fn sqlite_full_rolls_back_groups_and_imports_and_allows_retry() {
    for checkpointed in [false, true] {
        for extra_pages in [0, 2, 8] {
            let directory = tempfile::tempdir().unwrap();
            let path = directory.path().join("limited.sqlite");
            let budget = MemoryBudget::new(1 << 20);
            let mut store = SqliteStore::create(&path, 4096, &budget).unwrap();
            let first = Checkpoint {
                identity: [31; 32],
                completed_group: 0,
            };
            let next = Checkpoint {
                completed_group: 1,
                ..first.clone()
            };
            if checkpointed {
                store
                    .commit_group(&[record(0)], None, &first, &Cancellation::default())
                    .unwrap();
            } else {
                store
                    .import(&[record(0)][..], &Cancellation::default())
                    .unwrap();
            }
            let initial = records(&store);
            let pages: i64 = store
                .connection
                .pragma_query_value(None, "page_count", |r| r.get(0))
                .unwrap();
            let maximum: i64 = store
                .connection
                .pragma_query_value(None, "max_page_count", |r| r.get(0))
                .unwrap();
            store
                .connection
                .pragma_update(None, "max_page_count", pages + extra_pages)
                .unwrap();
            let input: Vec<_> = (1..4096).map(record).collect();
            let result = if checkpointed {
                store.commit_group(&input, Some(&first), &next, &Cancellation::default())
            } else {
                store.import(input.as_slice(), &Cancellation::default())
            };
            assert!(
                matches!(result, Err(Error::Storage(rusqlite::Error::SqliteFailure(error, _))) if error.code == rusqlite::ErrorCode::DiskFull),
                "expected SQLITE_FULL, got {result:?}"
            );
            assert_eq!(records(&store), initial);
            assert_eq!(
                store.checkpoint().unwrap(),
                checkpointed.then_some(first.clone())
            );
            store.integrity_check().unwrap();
            store
                .connection
                .pragma_update(None, "max_page_count", maximum)
                .unwrap();
            if checkpointed {
                store
                    .commit_group(&input, Some(&first), &next, &Cancellation::default())
                    .unwrap();
            } else {
                store
                    .import(input.as_slice(), &Cancellation::default())
                    .unwrap();
            }
            let expected = records(&store);
            assert_eq!(expected.len(), 4096);
            for (actual, reference) in expected.iter().zip((0..4096).map(record)) {
                assert_eq!(*actual, payload(&reference));
            }
            store.integrity_check().unwrap();
            drop(store);
            let reopened = SqliteStore::open(&path, 4096, &budget).unwrap();
            assert_eq!(records(&reopened), expected);
            assert_eq!(reopened.checkpoint().unwrap(), checkpointed.then_some(next));
            drop(reopened);
            assert_eq!(budget.used(), 0);
        }
    }
}
