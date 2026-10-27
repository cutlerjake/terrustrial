#![cfg(feature = "sqlite")]
use terrustrial_core::{
    data::{Observation, Point3, Site},
    execution::{Cancellation, MemoryBudget},
    spatial::ObservationSource,
    storage::{Checkpoint, SqliteStore},
};

fn fixture(path: &std::path::Path, budget: &MemoryBudget) {
    let observations: Vec<_> = [
        (0, [1.25, -2.5, 3.75]),
        (1, [1e100, 0., 0.]),
        (2, [f64::from_bits(1), 0., 0.]),
        (3, [0.; 3]),
        (4, [f32::MIN_POSITIVE as f64; 3]),
        (5, [f32::MAX as f64 / 4.; 3]),
        (u64::MAX, [-1.25, 2.5, -3.75]),
    ]
    .into_iter()
    .map(|(id, xyz)| {
        Observation::new(
            id,
            Site::point(Point3::new(xyz).unwrap(), usize::MAX),
            f64::MAX,
            f64::MAX,
        )
        .unwrap()
    })
    .collect();
    SqliteStore::create(path, 4096, budget)
        .unwrap()
        .commit_group(
            &observations,
            None,
            &Checkpoint {
                identity: [19; 32],
                completed_group: u64::MAX,
            },
            &Cancellation::default(),
        )
        .unwrap();
}

#[test]
fn integrity_rejects_logical_corruption_that_structural_checks_accept() {
    let directory = tempfile::tempdir().unwrap();
    let budget = MemoryBudget::new(1 << 20);
    let original = directory.path().join("original.sqlite");
    fixture(&original, &budget);
    for (name, mutation) in [
        (
            "false x zero sign",
            "UPDATE points SET zero_signs=1 WHERE id=0",
        ),
        (
            "false y zero sign",
            "UPDATE points SET zero_signs=2 WHERE id=0",
        ),
        (
            "false z zero sign",
            "UPDATE points SET zero_signs=4 WHERE id=0",
        ),
        (
            "false value zero sign",
            "UPDATE points SET zero_signs=8 WHERE id=0",
        ),
        (
            "false noise zero sign",
            "UPDATE points SET zero_signs=16 WHERE id=0",
        ),
        (
            "noninteger zero signs",
            "UPDATE points SET zero_signs=0.5 WHERE id=0",
        ),
        ("missing index", "DELETE FROM bounds WHERE id=0"),
        (
            "orphaned index",
            "INSERT INTO bounds VALUES(100,0,0,0,0,0,0)",
        ),
        (
            "unexpected index",
            "INSERT INTO bounds VALUES(1,0,0,0,0,0,0)",
        ),
        ("wrong index flag", "UPDATE points SET indexed=0 WHERE id=0"),
        (
            "unsafe indexed coordinate",
            "UPDATE points SET x=1e100 WHERE id=0",
        ),
        (
            "unsafe subnormal index",
            "UPDATE points SET x=1e-200 WHERE id=0",
        ),
        ("excluded x", "UPDATE bounds SET min_x=8,max_x=9 WHERE id=0"),
        ("excluded y", "UPDATE bounds SET min_y=8,max_y=9 WHERE id=0"),
        ("excluded z", "UPDATE bounds SET min_z=8,max_z=9 WHERE id=0"),
        ("nonfinite box", "UPDATE bounds SET max_x=1e999 WHERE id=0"),
        (
            "nonfinite coordinate",
            "UPDATE points SET x=1e999 WHERE id=1",
        ),
        (
            "nonfinite value",
            "UPDATE points SET value=-1e999 WHERE id=0",
        ),
        ("negative noise", "UPDATE points SET noise=-1 WHERE id=0"),
        (
            "nonfinite noise",
            "UPDATE points SET noise=1e999 WHERE id=0",
        ),
        ("text payload", "UPDATE points SET value='bad' WHERE id=0"),
        ("blob coordinate", "UPDATE points SET x=x'0123' WHERE id=1"),
        (
            "noninteger variable",
            "UPDATE points SET variable=1.25 WHERE id=0",
        ),
        (
            "short checkpoint",
            "UPDATE metadata SET value=zeroblob(39) WHERE key='checkpoint'",
        ),
        (
            "long checkpoint",
            "UPDATE metadata SET value=zeroblob(41) WHERE key='checkpoint'",
        ),
        (
            "text checkpoint",
            "UPDATE metadata SET value='bad' WHERE key='checkpoint'",
        ),
    ] {
        let path = directory.path().join("case.sqlite");
        std::fs::copy(&original, &path).unwrap();
        let connection = rusqlite::Connection::open(&path).unwrap();
        connection.execute_batch(mutation).unwrap();
        let sql: String = connection
            .query_row("PRAGMA integrity_check", [], |r| r.get(0))
            .unwrap();
        let rtree: String = connection
            .query_row("SELECT rtreecheck('bounds')", [], |r| r.get(0))
            .unwrap();
        assert_eq!(sql, "ok", "{name}: must exercise application integrity");
        assert_eq!(rtree, "ok", "{name}: must exercise application integrity");
        drop(connection);
        let store = SqliteStore::open(&path, 4096, &budget).unwrap();
        assert!(store.integrity_check().is_err(), "accepted {name}");
        drop(store);
        assert_eq!(budget.used(), 0);
    }
}

#[test]
fn integrity_accepts_extremes_empty_stores_and_releases_failed_snapshots() {
    let directory = tempfile::tempdir().unwrap();
    let budget = MemoryBudget::new(1 << 20);
    let path = directory.path().join("store.sqlite");
    fixture(&path, &budget);
    let store = SqliteStore::open(&path, 4096, &budget).unwrap();
    store.integrity_check().unwrap();
    let connection = rusqlite::Connection::open(&path).unwrap();
    // A larger conservative box can add candidates but cannot omit neighbors.
    connection
        .execute_batch("UPDATE bounds SET min_x=-10,max_x=10 WHERE id=0")
        .unwrap();
    store.integrity_check().unwrap();
    connection
        .execute_batch("UPDATE points SET noise=-1 WHERE id=0")
        .unwrap();
    assert!(store.integrity_check().is_err());
    // A failed check must release its read lock and allow repair by the caller.
    connection
        .execute_batch("UPDATE points SET noise=0 WHERE id=0")
        .unwrap();
    store.integrity_check().unwrap();
    let mut rows = 0;
    store
        .scan(&mut |_| {
            rows += 1;
            Ok(())
        })
        .unwrap();
    assert_eq!(rows, 7);
    drop((store, connection));
    let empty = SqliteStore::create(directory.path().join("empty.sqlite"), 4096, &budget).unwrap();
    empty.integrity_check().unwrap();
    drop(empty);
    assert_eq!(budget.used(), 0);
}

#[test]
fn truncation_and_header_corruption_are_rejected_without_rewriting_the_file() {
    let directory = tempfile::tempdir().unwrap();
    let budget = MemoryBudget::new(1 << 20);
    let original = directory.path().join("original.sqlite");
    fixture(&original, &budget);
    let pristine = std::fs::read(&original).unwrap();
    let lengths = [0, 1, 99, 512, pristine.len() / 2, pristine.len() - 1];
    for length in lengths {
        let path = directory.path().join("truncated.sqlite");
        std::fs::write(&path, &pristine[..length]).unwrap();
        assert!(
            SqliteStore::open(&path, 4096, &budget)
                .and_then(|s| s.integrity_check())
                .is_err(),
            "accepted length {length}"
        );
        assert_eq!(std::fs::read(&path).unwrap(), pristine[..length]);
        assert_eq!(budget.used(), 0);
    }
    for offset in [0, 16, 68] {
        let path = directory.path().join("corrupt.sqlite");
        let mut bytes = pristine.clone();
        bytes[offset] ^= 0xff;
        std::fs::write(&path, &bytes).unwrap();
        assert!(
            SqliteStore::open(&path, 4096, &budget)
                .and_then(|s| s.integrity_check())
                .is_err(),
            "accepted header offset {offset}"
        );
        assert_eq!(std::fs::read(&path).unwrap(), bytes);
        assert_eq!(budget.used(), 0);
    }
}

#[test]
fn integrity_uses_committed_wal_pages_without_requiring_a_checkpoint() {
    let directory = tempfile::tempdir().unwrap();
    let budget = MemoryBudget::new(1 << 20);
    let path = directory.path().join("wal.sqlite");
    fixture(&path, &budget);
    let connection = rusqlite::Connection::open(&path).unwrap();
    connection
        .execute_batch(
            "PRAGMA journal_mode=WAL; PRAGMA wal_autocheckpoint=0;
         BEGIN;
         WITH RECURSIVE ids(i) AS (VALUES(10) UNION ALL SELECT i+1 FROM ids WHERE i<2010)
         INSERT INTO points SELECT i,0,i,0,0,1,0,1,0,NULL FROM ids;
         INSERT INTO bounds SELECT id,x,x,y,y,z,z FROM points WHERE id>=10;
         COMMIT;",
        )
        .unwrap();
    let size: u32 = connection
        .pragma_query_value(None, "page_size", |r| r.get(0))
        .unwrap();
    let pages: u32 = connection
        .pragma_query_value(None, "page_count", |r| r.get(0))
        .unwrap();
    assert!(std::fs::metadata(&path).unwrap().len() < u64::from(size) * u64::from(pages));
    let store = SqliteStore::open(&path, 4096, &budget).unwrap();
    store.integrity_check().unwrap();
    assert_eq!(store.len().unwrap(), 2008);
    drop((store, connection));
    assert_eq!(budget.used(), 0);
}

// APFS requires valid UTF-8 names. Linux CI exercises byte-valued filenames.
#[cfg(target_os = "linux")]
#[test]
fn integrity_accepts_non_utf8_database_names() {
    use std::os::unix::ffi::OsStrExt;
    let directory = tempfile::tempdir().unwrap();
    let path = directory
        .path()
        .join(std::ffi::OsStr::from_bytes(b"store-\xff.sqlite"));
    let budget = MemoryBudget::new(1 << 20);
    fixture(&path, &budget);
    let store = SqliteStore::open(&path, 4096, &budget).unwrap();
    assert_eq!(store.path(), std::fs::canonicalize(&path).unwrap());
    store.integrity_check().unwrap();
    drop(store);
    assert_eq!(budget.used(), 0);
}

#[test]
fn integrity_resolves_relative_unicode_paths() {
    let directory = tempfile::tempdir_in(".").unwrap();
    let path = directory.path().join("données.sqlite");
    let budget = MemoryBudget::new(1 << 20);
    fixture(&path, &budget);
    let store = SqliteStore::open(&path, 4096, &budget).unwrap();
    assert!(store.path().is_absolute());
    assert_eq!(store.path(), std::fs::canonicalize(&path).unwrap());
    store.integrity_check().unwrap();
    drop(store);
    assert_eq!(budget.used(), 0);
}
