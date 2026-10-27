use super::*;
use crate::{
    model::Anisotropy,
    spatial::{MinimumQuotas, SearchNeighborhood, SearchStop},
};
use std::{
    cell::Cell,
    panic::{catch_unwind, AssertUnwindSafe},
};

#[test]
fn expansion_snapshot_survives_a_wal_commit_between_attempts_and_releases_on_error() {
    for after_full_level in [false, true] {
        check_expansion_snapshot(after_full_level);
    }
}
fn check_expansion_snapshot(after_full_level: bool) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("adaptive.sqlite");
    let budget = MemoryBudget::new(1 << 20);
    let cancel = Cancellation::default();
    let mut store = SqliteStore::create(&path, 4096, &budget).unwrap();
    let data = [
        Observation::new(
            1,
            Site::point(Point3::new([0.25, 0., 0.]).unwrap(), 0),
            1.,
            0.,
        )
        .unwrap(),
        Observation::new(
            2,
            Site::point(Point3::new([1.5, 0., 0.]).unwrap(), 0),
            2.,
            0.,
        )
        .unwrap(),
    ];
    store.import(data.as_slice(), &cancel).unwrap();
    let writer = rusqlite::Connection::open(&path).unwrap();
    writer
        .execute_batch("PRAGMA journal_mode=WAL; PRAGMA wal_autocheckpoint=0")
        .unwrap();
    let query = SearchNeighborhood::new(3, Anisotropy::isotropic(1.).unwrap(), None, vec![])
        .unwrap()
        .with_minimums(MinimumQuotas {
            total: 2,
            ..Default::default()
        })
        .unwrap()
        .with_expansion(vec![[2.; 3]])
        .unwrap();
    let mut workspace = SearchWorkspace::new(&query, &budget).unwrap();
    struct Between<'a> {
        store: &'a SqliteStore,
        writer: &'a rusqlite::Connection,
        attempts: Cell<usize>,
        after_full_level: bool,
        written: Cell<bool>,
        expanded_after_commit: Cell<bool>,
    }
    impl ObservationSource for Between<'_> {
        fn scan(&self, _: &mut dyn FnMut(&Observation) -> Result<()>) -> Result<()> {
            panic!("bounded path expected")
        }
        fn visit_bounds(
            &self,
            lower: [f64; 3],
            upper: [f64; 3],
            cancel: &Cancellation,
            visitor: &mut dyn FnMut(&Observation) -> Result<()>,
        ) -> Result<()> {
            assert!(!self.store.connection.is_autocommit());
            if self.written.get() && upper[0] > 1.5 {
                self.expanded_after_commit.set(true);
            }
            self.store.visit_bounds(lower, upper, cancel, visitor)?;
            self.attempts.set(self.attempts.get() + 1);
            if !self.written.get() && (!self.after_full_level || upper[0] >= 1.0) {
                self.writer.execute_batch("BEGIN IMMEDIATE; UPDATE points SET value=99; DELETE FROM bounds WHERE id=2; DELETE FROM points WHERE id=2; INSERT INTO points VALUES(3,0,0.5,0,0,3,0,1,0,NULL); INSERT INTO bounds VALUES(3,0.5,0.5,0,0,0,0); COMMIT;")?;
                self.written.set(true);
            }
            Ok(())
        }
    }
    let between = Between {
        store: &store,
        writer: &writer,
        attempts: Cell::new(0),
        after_full_level,
        written: Cell::new(false),
        expanded_after_commit: Cell::new(false),
    };
    // Exercise the same read-snapshot wrapper used by SqliteStore::search,
    // Inject after either the first density probe or the complete initial level.
    // Internal probe count is an optimization detail; the snapshot must survive
    // both probing and the subsequent public radius expansion.
    let found = store
        .read_snapshot(|| between.search(Point3::new([0.; 3]).unwrap(), &mut workspace, &cancel))
        .unwrap();
    assert_eq!(
        found
            .iter()
            .map(|o| (o.id(), o.value().to_bits()))
            .collect::<Vec<_>>(),
        [(1, 1_f64.to_bits()), (2, 2_f64.to_bits())]
    );
    assert!(between.attempts.get() >= 2);
    assert!(between.written.get());
    assert!(between.expanded_after_commit.get());
    assert_eq!(workspace.diagnostics().level, 1);
    assert_eq!(workspace.diagnostics().stop, Some(SearchStop::Satisfied));
    assert!(store.connection.is_autocommit());
    let found = store
        .search(Point3::new([0.; 3]).unwrap(), &mut workspace, &cancel)
        .unwrap();
    assert_eq!(
        found
            .iter()
            .map(|o| (o.id(), o.value().to_bits()))
            .collect::<Vec<_>>(),
        [(1, 99_f64.to_bits()), (3, 3_f64.to_bits())]
    );
    assert_eq!(workspace.diagnostics().level, 0);
    assert!(store.connection.is_autocommit());
    assert!(matches!(
        store.read_snapshot(|| {
            assert!(!store.connection.is_autocommit());
            Err::<(), _>(Error::Cancelled)
        }),
        Err(Error::Cancelled)
    ));
    assert!(store.connection.is_autocommit());
    assert!(catch_unwind(AssertUnwindSafe(
        || store.read_snapshot::<()>(|| panic!("snapshot unwind"))
    ))
    .is_err());
    assert!(store.connection.is_autocommit());
    drop((workspace, store));
    assert_eq!(budget.used(), 0);
}
