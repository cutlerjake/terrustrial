#![cfg(feature = "sqlite")]
use std::time::Duration;
use terrustrial_core::{
    data::{Observation, Point3, Site},
    execution::{Cancellation, MemoryBudget},
    identity,
    model::Anisotropy,
    spatial::{
        NearestWorkspace, Neighborhood, ObservationSource, SearchNeighborhood, SearchWorkspace,
    },
    storage::{SqliteReadPool, SqliteStore},
};

fn fixture(path: &std::path::Path, budget: &MemoryBudget) {
    let mut store = SqliteStore::create(path, 4096, budget).unwrap();
    let data: Vec<_> = (0..16)
        .map(|i| {
            Observation::new(
                i,
                Site::point(Point3::new([i as f64, 0., 0.]).unwrap(), 0),
                i as f64,
                0.,
            )
            .unwrap()
        })
        .collect();
    store
        .import(data.as_slice(), &Cancellation::default())
        .unwrap();
}
fn rows(source: &impl ObservationSource) -> Vec<(u64, u64)> {
    let mut out = Vec::new();
    source
        .scan(&mut |o| {
            out.push((o.id(), o.value().to_bits()));
            Ok(())
        })
        .unwrap();
    out.sort_unstable();
    out
}

#[test]
fn wal_snapshot_retains_all_readers_and_query_paths_while_writers_continue() {
    for readers in [1, 4] {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("hard.sqlite");
        let budget = MemoryBudget::new(1 << 24);
        let cancel = Cancellation::default();
        fixture(&path, &budget);
        let writer = rusqlite::Connection::open(&path).unwrap();
        writer
            .execute_batch("PRAGMA journal_mode=WAL; PRAGMA wal_autocheckpoint=0")
            .unwrap();
        let snapshot =
            SqliteReadPool::open_snapshot(&path, readers, 4096, &budget, &cancel).unwrap();
        assert!(snapshot.is_snapshot());
        assert_eq!(budget.used(), snapshot.accounted_bytes());
        let original = rows(&snapshot);
        let fingerprint = identity::observations(&snapshot, &cancel).unwrap();
        writer.execute_batch("BEGIN IMMEDIATE; UPDATE points SET value=value+100; DELETE FROM bounds WHERE id=0; DELETE FROM points WHERE id=0; INSERT INTO points VALUES(100,0,100,0,0,100,0,1,0,NULL); INSERT INTO bounds VALUES(100,100,100,0,0,0,0); COMMIT;").unwrap();
        std::thread::scope(|scope| {
            let mut handles = Vec::new();
            for _ in 0..readers * 2 {
                let snapshot = &snapshot;
                let original = &original;
                let budget = &budget;
                let cancel = &cancel;
                handles.push(scope.spawn(move || {
                    for _ in 0..8 {
                        assert_eq!(rows(snapshot), *original);
                        assert_eq!(snapshot.len().unwrap(), 16);
                        let site = Site::point(Point3::new([0.; 3]).unwrap(), 0);
                        let exact = snapshot.at_site(&site).unwrap();
                        assert_eq!(exact.len(), 1);
                        assert_eq!(exact[0].id(), 0);
                        let query = Neighborhood {
                            maximum: 16,
                            radius: 200.,
                        };
                        let mut workspace = NearestWorkspace::new(query, budget).unwrap();
                        let selected = snapshot
                            .nearest_into(site.support.center(), &mut workspace, cancel)
                            .unwrap();
                        assert_eq!(
                            selected
                                .iter()
                                .map(|o| (o.id(), o.value().to_bits()))
                                .collect::<Vec<_>>(),
                            *original
                        );
                        assert_eq!(
                            snapshot
                                .nearest(site.support.center(), query, cancel)
                                .unwrap()
                                .len(),
                            16
                        );
                        let search = SearchNeighborhood::new(
                            16,
                            Anisotropy::isotropic(200.).unwrap(),
                            None,
                            Vec::new(),
                        )
                        .unwrap();
                        let mut workspace = SearchWorkspace::new(&search, budget).unwrap();
                        assert_eq!(
                            snapshot
                                .search(site.support.center(), &mut workspace, cancel)
                                .unwrap()
                                .iter()
                                .map(|o| o.id())
                                .collect::<Vec<_>>(),
                            (0..16).collect::<Vec<_>>()
                        );
                    }
                }));
            }
            for handle in handles {
                handle.join().unwrap();
            }
        });
        assert_eq!(
            identity::observations(&snapshot, &cancel).unwrap(),
            fingerprint
        );
        let live = SqliteReadPool::open(&path, 1, 4096, &budget).unwrap();
        assert!(!live.is_snapshot());
        assert_ne!(rows(&live), original);
        assert_ne!(identity::observations(&live, &cancel).unwrap(), fingerprint);
        let new_snapshot =
            SqliteReadPool::open_snapshot(&path, readers, 4096, &budget, &cancel).unwrap();
        assert_eq!(rows(&new_snapshot), rows(&live));
        drop((snapshot, new_snapshot, live, writer));
        assert_eq!(budget.used(), 0);
    }
}

#[test]
fn empty_wal_snapshot_is_pinned_and_rollback_snapshots_release_writer_locks() {
    let directory = tempfile::tempdir().unwrap();
    let budget = MemoryBudget::new(1 << 20);
    let cancel = Cancellation::default();
    let empty_path = directory.path().join("empty.sqlite");
    drop(SqliteStore::create(&empty_path, 4096, &budget).unwrap());
    let writer = rusqlite::Connection::open(&empty_path).unwrap();
    writer.execute_batch("PRAGMA journal_mode=WAL").unwrap();
    let empty = SqliteReadPool::open_snapshot(&empty_path, 3, 4096, &budget, &cancel).unwrap();
    writer.execute_batch("BEGIN; INSERT INTO points VALUES(0,0,0,0,0,1,0,1,0,NULL); INSERT INTO bounds VALUES(0,0,0,0,0,0,0); COMMIT").unwrap();
    for _ in 0..12 {
        assert!(empty.is_empty().unwrap());
        assert!(rows(&empty).is_empty());
    }
    drop((empty, writer));
    let path = directory.path().join("rollback.sqlite");
    fixture(&path, &budget);
    let snapshot = SqliteReadPool::open_snapshot(&path, 3, 4096, &budget, &cancel).unwrap();
    let before = rows(&snapshot);
    let writer = rusqlite::Connection::open(&path).unwrap();
    writer.busy_timeout(Duration::ZERO).unwrap();
    writer
        .execute_batch("BEGIN IMMEDIATE; UPDATE points SET value=99")
        .unwrap();
    assert!(
        matches!(writer.execute_batch("COMMIT"),Err(rusqlite::Error::SqliteFailure(e,_)) if e.code==rusqlite::ErrorCode::DatabaseBusy)
    );
    assert_eq!(rows(&snapshot), before);
    drop(snapshot);
    writer.execute_batch("COMMIT").unwrap();
    let live = SqliteReadPool::open(&path, 1, 4096, &budget).unwrap();
    assert!(rows(&live).iter().all(|(_, v)| *v == 99_f64.to_bits()));
    drop((live, writer));
    assert_eq!(budget.used(), 0);
}

#[test]
fn identified_run_uses_the_same_conditioning_snapshot_after_fingerprinting() {
    use std::cell::Cell;
    use terrustrial_core::{
        data::{Group, Target},
        gaussian::Simple,
        model::{Coregionalization, Correlation, Lmc, Structure},
        simulation::{
            identity::RunProvenance, simulate_checkpointed_identified, SimulationOptions,
        },
    };
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("hard.sqlite");
    let budget = MemoryBudget::new(1 << 24);
    let cancel = Cancellation::default();
    fixture(&path, &budget);
    let writer = rusqlite::Connection::open(&path).unwrap();
    writer.execute_batch("PRAGMA journal_mode=WAL").unwrap();
    let hard = SqliteReadPool::open_snapshot(&path, 4, 4096, &budget, &cancel).unwrap();
    let mut original = Vec::new();
    hard.scan(&mut |o| {
        original.push(*o);
        Ok(())
    })
    .unwrap();
    let model = Lmc::new(vec![Structure {
        correlation: Correlation::Exponential,
        anisotropy: Anisotropy::isotropic(3.).unwrap(),
        coregionalization: Coregionalization::scalar(1.).unwrap(),
    }])
    .unwrap();
    let provenance = RunProvenance {
        implementation: identity::opaque("test build", b"snapshot v1").unwrap(),
        backend: identity::opaque("test backend", b"cpu-f64").unwrap(),
        mean: None,
    };
    let options = || SimulationOptions {
        seed: 4,
        realization: 2,
        mean: None,
        hard_neighborhood: Neighborhood {
            maximum: 8,
            radius: 100.,
        }
        .into(),
        simulated_neighborhood: Neighborhood {
            maximum: 8,
            radius: 100.,
        }
        .into(),
        budget: &budget,
        cancellation: &cancel,
    };
    let groups = || {
        (0..8).map(|i| {
            Ok(Group {
                id: i,
                targets: vec![Target {
                    id: 100 + i,
                    site: Site::point(Point3::new([i as f64 + 0.25, 0., 0.]).unwrap(), 0),
                }],
            })
        })
    };
    let opens = Cell::new(0);
    let changing_writer = || {
        opens.set(opens.get() + 1);
        if opens.get() == 2 {
            writer
                .execute_batch("UPDATE points SET value=value+1000")
                .unwrap();
        }
        Ok(groups())
    };
    let mut output =
        SqliteStore::create(directory.path().join("output.sqlite"), 4096, &budget).unwrap();
    simulate_checkpointed_identified(
        &model,
        &hard,
        &mut output,
        changing_writer,
        &Simple,
        options(),
        provenance,
        |_| Ok(()),
    )
    .unwrap();
    let mut reference =
        SqliteStore::create(directory.path().join("reference.sqlite"), 4096, &budget).unwrap();
    simulate_checkpointed_identified(
        &model,
        original.as_slice(),
        &mut reference,
        || Ok(groups()),
        &Simple,
        options(),
        provenance,
        |_| Ok(()),
    )
    .unwrap();
    assert_eq!(rows(&output), rows(&reference));
    assert_eq!(
        output.checkpoint().unwrap(),
        reference.checkpoint().unwrap()
    );
    drop((hard, output, reference, writer));
    assert_eq!(budget.used(), 0);
}
