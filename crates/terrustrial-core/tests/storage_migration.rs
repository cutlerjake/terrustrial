#![cfg(feature = "sqlite")]
use terrustrial_core::{
    data::{Observation, Point3, Site},
    execution::{Cancellation, MemoryBudget},
    spatial::ObservationSource,
    storage::{SqliteReadPool, SqliteStore},
};

// Frozen version-1 layout, independent of the current create() implementation.
fn version_one(path: &std::path::Path, checkpointed: bool) {
    let connection = rusqlite::Connection::open(path).unwrap();
    connection
        .execute_batch(
            "PRAGMA journal_mode=DELETE; PRAGMA synchronous=FULL;
         CREATE TABLE metadata(key TEXT PRIMARY KEY,value BLOB NOT NULL) WITHOUT ROWID;
         CREATE TABLE points(id INTEGER PRIMARY KEY,variable INTEGER NOT NULL,
             x REAL NOT NULL,y REAL NOT NULL,z REAL NOT NULL,value REAL NOT NULL,
             noise REAL NOT NULL,indexed INTEGER NOT NULL CHECK(indexed IN (0,1)));
         CREATE INDEX unindexed_points ON points(indexed) WHERE indexed=0;
         CREATE VIRTUAL TABLE bounds USING rtree(id,min_x,max_x,min_y,max_y,min_z,max_z);
         INSERT INTO points VALUES(-1,2,1.25,0,1e100,-3.5,0,0),(2,0,0,0,0,0,0,1);
         INSERT INTO bounds VALUES(2,0,0,0,0,0,0);
         PRAGMA application_id=0x54524753; PRAGMA user_version=1;",
        )
        .unwrap();
    if checkpointed {
        connection
            .execute(
                "INSERT INTO metadata VALUES('checkpoint',?1)",
                [&[9_u8; 40][..]],
            )
            .unwrap();
    }
}

#[test]
fn explicit_uncheckpointed_migration_preserves_stored_values_and_accepts_new_zero_signs() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("old.sqlite");
    version_one(&path, false);
    let original = std::fs::read(&path).unwrap();
    let budget = MemoryBudget::new(1 << 20);
    assert!(SqliteStore::open(&path, 4096, &budget).is_err());
    assert!(SqliteReadPool::open(&path, 2, 4096, &budget).is_err());
    assert_eq!(std::fs::read(&path).unwrap(), original);
    assert_eq!(budget.used(), 0);
    let mut store =
        SqliteStore::migrate_v1(&path, 4096, &budget, &Cancellation::default()).unwrap();
    let mut actual = Vec::new();
    store
        .scan(&mut |o| {
            actual.push((
                o.id(),
                o.site().variable,
                o.site().support.center().coordinates().map(f64::to_bits),
                o.value().to_bits(),
                o.noise_variance().to_bits(),
            ));
            Ok(())
        })
        .unwrap();
    actual.sort_unstable();
    assert_eq!(
        actual,
        vec![
            (2, 0, [0; 3], 0, 0),
            (
                u64::MAX,
                2,
                [1.25_f64.to_bits(), 0, 1e100_f64.to_bits()],
                (-3.5_f64).to_bits(),
                0
            ),
        ]
    );
    assert!(store.checkpoint().unwrap().is_none());
    let new =
        Observation::new(3, Site::point(Point3::new([-0.; 3]).unwrap(), 0), -0., -0.).unwrap();
    store.import(&[new][..], &Cancellation::default()).unwrap();
    let found = store.at_site(new.site()).unwrap();
    let found = found.iter().find(|o| o.id() == 3).unwrap();
    assert_eq!(
        found
            .site()
            .support
            .center()
            .coordinates()
            .map(f64::to_bits),
        [(-0.0_f64).to_bits(); 3]
    );
    assert_eq!(found.value().to_bits(), (-0.0_f64).to_bits());
    assert_eq!(found.noise_variance().to_bits(), (-0.0_f64).to_bits());
    store.integrity_check().unwrap();
    drop(store);
    let before = std::fs::read(&path).unwrap();
    assert!(SqliteStore::migrate_v1(&path, 4096, &budget, &Cancellation::default()).is_err());
    assert_eq!(std::fs::read(&path).unwrap(), before);
    let store = SqliteStore::open(&path, 4096, &budget).unwrap();
    assert_eq!(store.len().unwrap(), 3);
    drop(store);
    assert_eq!(budget.used(), 0);
}

#[test]
fn refused_failed_and_cancelled_migrations_leave_version_one_bytes_unchanged() {
    for (checkpointed, mutation, cancel_now) in [
        (true, "", false),
        (false, "UPDATE points SET noise=-1 WHERE id=2", false),
        (false, "DELETE FROM bounds", false),
        (false, "PRAGMA application_id=0", false),
        (false, "", true),
    ] {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("old.sqlite");
        version_one(&path, checkpointed);
        rusqlite::Connection::open(&path)
            .unwrap()
            .execute_batch(mutation)
            .unwrap();
        let original = std::fs::read(&path).unwrap();
        let budget = MemoryBudget::new(1 << 20);
        let cancel = Cancellation::default();
        if cancel_now {
            cancel.cancel();
        }
        assert!(SqliteStore::migrate_v1(&path, 4096, &budget, &cancel).is_err());
        assert_eq!(std::fs::read(&path).unwrap(), original);
        assert_eq!(budget.used(), 0);
        let connection = rusqlite::Connection::open(&path).unwrap();
        let version: i64 = connection
            .pragma_query_value(None, "user_version", |r| r.get(0))
            .unwrap();
        assert_eq!(version, 1);
        let columns: i64 = connection
            .query_row(
                "SELECT count(*) FROM pragma_table_info('points')",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(columns, 8);
    }
}

// Frozen version-2 schema and data, independent of current create/migration code.
fn version_two(path: &std::path::Path, checkpointed: bool) {
    let db = rusqlite::Connection::open(path).unwrap();
    db.execute_batch(
        "PRAGMA journal_mode=DELETE; PRAGMA synchronous=FULL;
        CREATE TABLE metadata(key TEXT PRIMARY KEY,value BLOB NOT NULL) WITHOUT ROWID;
        CREATE TABLE points(id INTEGER PRIMARY KEY,variable INTEGER NOT NULL,
            x REAL NOT NULL,y REAL NOT NULL,z REAL NOT NULL,value REAL NOT NULL,
            noise REAL NOT NULL,indexed INTEGER NOT NULL CHECK(indexed IN (0,1)),
            zero_signs INTEGER NOT NULL CHECK(zero_signs BETWEEN 0 AND 31));
        CREATE INDEX unindexed_points ON points(indexed) WHERE indexed=0;
        CREATE VIRTUAL TABLE bounds USING rtree(id,min_x,max_x,min_y,max_y,min_z,max_z);
        INSERT INTO points VALUES(-1,2,1.25,0,1e100,-3.5,0,0,18),(2,0,0,0,0,0,0,1,31);
        INSERT INTO bounds VALUES(2,0,0,0,0,0,0);
        PRAGMA application_id=0x54524753; PRAGMA user_version=2;",
    )
    .unwrap();
    if checkpointed {
        let mut bytes = [0; 40];
        bytes[..32].fill(9);
        db.execute(
            "INSERT INTO metadata VALUES('checkpoint',?1)",
            [bytes.as_slice()],
        )
        .unwrap();
    }
}
#[test]
fn version_two_migration_preserves_bits_checkpoints_and_accepts_blocks() {
    use terrustrial_core::{data::Support, storage::Checkpoint};
    for checkpointed in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("v2.sqlite");
        version_two(&path, checkpointed);
        let budget = MemoryBudget::new(1 << 20);
        let cancel = Cancellation::default();
        let original = std::fs::read(&path).unwrap();
        assert!(SqliteStore::open(&path, 4096, &budget).is_err());
        assert!(SqliteReadPool::open(&path, 2, 4096, &budget).is_err());
        assert_eq!(std::fs::read(&path).unwrap(), original);
        let mut store = SqliteStore::migrate_v2(&path, 4096, &budget, &cancel).unwrap();
        let expected = vec![
            Observation::new(2, Site::point(Point3::new([-0.; 3]).unwrap(), 0), -0., -0.).unwrap(),
            Observation::new(
                u64::MAX,
                Site::point(Point3::new([1.25, -0., 1e100]).unwrap(), 2),
                -3.5,
                -0.,
            )
            .unwrap(),
        ];
        let mut found = Vec::new();
        store
            .scan(&mut |o| {
                found.push(*o);
                Ok(())
            })
            .unwrap();
        found.sort_by_key(Observation::id);
        assert_eq!(
            terrustrial_core::identity::observations(found.as_slice(), &cancel).unwrap(),
            terrustrial_core::identity::observations(expected.as_slice(), &cancel).unwrap()
        );
        let checkpoint = Checkpoint {
            identity: [9; 32],
            completed_group: 0,
        };
        assert_eq!(
            store.checkpoint().unwrap(),
            checkpointed.then_some(checkpoint.clone())
        );
        let block = Observation::new(
            3,
            Site {
                variable: 0,
                support: Support::block(Point3::new([1.; 3]).unwrap(), [0.1, 0.2, 0.3], [2, 3, 4])
                    .unwrap(),
            },
            2.,
            0.,
        )
        .unwrap();
        if checkpointed {
            store
                .commit_group(
                    &[block],
                    Some(&checkpoint),
                    &Checkpoint {
                        completed_group: 1,
                        ..checkpoint
                    },
                    &cancel,
                )
                .unwrap();
        } else {
            store.import(&[block][..], &cancel).unwrap();
        }
        assert_eq!(store.at_site(block.site()).unwrap()[0].site(), block.site());
        store.integrity_check().unwrap();
        drop(store);
        let before = std::fs::read(&path).unwrap();
        assert!(SqliteStore::migrate_v2(&path, 4096, &budget, &cancel).is_err());
        assert_eq!(std::fs::read(&path).unwrap(), before);
        assert_eq!(budget.used(), 0);
    }
}
#[test]
fn version_two_failed_and_cancelled_migration_preserves_original_bytes() {
    for (sql, stopped) in [
        ("UPDATE points SET noise=-1 WHERE id=2", false),
        ("DELETE FROM bounds", false),
        ("UPDATE metadata SET value=x'00'", false),
        ("PRAGMA application_id=0", false),
        ("", true),
    ] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("v2.sqlite");
        version_two(&path, true);
        rusqlite::Connection::open(&path)
            .unwrap()
            .execute_batch(sql)
            .unwrap();
        let before = std::fs::read(&path).unwrap();
        let budget = MemoryBudget::new(1 << 20);
        let cancel = Cancellation::default();
        if stopped {
            cancel.cancel();
        }
        assert!(SqliteStore::migrate_v2(&path, 4096, &budget, &cancel).is_err());
        assert_eq!(std::fs::read(&path).unwrap(), before);
        assert_eq!(budget.used(), 0);
        let db = rusqlite::Connection::open(&path).unwrap();
        let version: i64 = db
            .pragma_query_value(None, "user_version", |r| r.get(0))
            .unwrap();
        assert_eq!(version, 2);
        let columns: i64 = db
            .query_row(
                "SELECT count(*) FROM pragma_table_info('points')",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(columns, 9);
    }
}
#[test]
fn migrated_version_two_checkpoint_resumes_the_same_point_simulation() {
    use terrustrial_core::{
        data::{Group, Target},
        gaussian::Simple,
        model::{Anisotropy, Coregionalization, Correlation, Lmc, Structure},
        simulation::{simulate_checkpointed, CheckpointedOptions, SimulationOptions},
        spatial::Neighborhood,
    };
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("v2.sqlite");
    version_two(&path, true);
    rusqlite::Connection::open(&path).unwrap().execute_batch("DELETE FROM points; DELETE FROM bounds;
        INSERT INTO points VALUES(100,0,0,0,0,0,0,1,8); INSERT INTO bounds VALUES(100,0,0,0,0,0,0);").unwrap();
    let budget = MemoryBudget::new(1 << 24);
    let cancel = Cancellation::default();
    let mut migrated = SqliteStore::migrate_v2(&path, 4096, &budget, &cancel).unwrap();
    let mut reference =
        SqliteStore::create(dir.path().join("reference.sqlite"), 4096, &budget).unwrap();
    let model = Lmc::new(vec![Structure {
        correlation: Correlation::Exponential,
        anisotropy: Anisotropy::isotropic(2.).unwrap(),
        coregionalization: Coregionalization::scalar(1.).unwrap(),
    }])
    .unwrap();
    let hard =
        [Observation::new(0, Site::point(Point3::new([0.; 3]).unwrap(), 0), -0., 0.).unwrap()];
    for store in [&mut migrated, &mut reference] {
        let groups = (0..4).map(|i| Group {
            id: i,
            targets: vec![Target {
                id: 100 + i,
                site: Site::point(Point3::new([i as f64, 0., 0.]).unwrap(), 0),
            }],
        });
        simulate_checkpointed(
            &model,
            &hard[..],
            store,
            groups,
            &Simple,
            CheckpointedOptions {
                identity: [9; 32],
                simulation: SimulationOptions {
                    seed: 153,
                    realization: 0,
                    mean: None,
                    hard_neighborhood: Neighborhood {
                        maximum: 1,
                        radius: 10.,
                    }
                    .into(),
                    simulated_neighborhood: Neighborhood {
                        maximum: 4,
                        radius: 10.,
                    }
                    .into(),
                    budget: &budget,
                    cancellation: &cancel,
                },
            },
            |_| Ok(()),
        )
        .unwrap();
    }
    assert_eq!(
        terrustrial_core::identity::observations(&migrated, &cancel).unwrap(),
        terrustrial_core::identity::observations(&reference, &cancel).unwrap()
    );
    assert_eq!(
        migrated.checkpoint().unwrap(),
        reference.checkpoint().unwrap()
    );
    drop((migrated, reference));
    assert_eq!(budget.used(), 0);
}
