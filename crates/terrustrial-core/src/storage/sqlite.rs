//! Optional durable point and block store with a disk-backed R-tree and atomic checkpoints.
//!
//! The SQLite page cache is bounded by configuration. Payloads stay in binary64;
//! outward-rounded R-tree boxes only prune candidates, never determine distances.
//! Blocks retain their full support metadata. Neighborhoods rank support centers,
//! matching resident sources; exact-site lookup compares the entire support.
//! Coordinates outside the safe binary32 index range use an exact scan fallback.
//! Queries use O(k) Rust working storage and stable (distance, unsigned ID) ordering.
use crate::{
    data::{Observation, Point3, Site, Support},
    error::invalid,
    execution::{Cancellation, MemoryBudget, Reservation},
    spatial::{
        finish, offer, Candidate, NearestWorkspace, Neighborhood, ObservationSink,
        ObservationSource, SearchWorkspace,
    },
    Error, Result,
};
use rusqlite::{params, Connection, OpenFlags, OptionalExtension, TransactionBehavior};
use std::{
    collections::BinaryHeap,
    fs::OpenOptions,
    path::{Path, PathBuf},
};

const APPLICATION_ID: i64 = 0x54524753;
const SCHEMA_VERSION: i64 = 3;
mod integrity;
mod migration;
mod pool;
mod support;
pub use pool::SqliteReadPool;
#[cfg(test)]
mod adaptive_tests;
#[cfg(test)]
mod crash_tests;
#[cfg(test)]
mod failure_tests;

macro_rules! point_select {
    () => {
        "SELECT p.id,p.variable,p.x,p.y,p.z,p.value,p.noise,p.zero_signs,p.support FROM "
    };
}
const BOUNDS_QUERY: &str = concat!(point_select!(), "bounds AS b JOIN points AS p ON p.id=b.id
    WHERE b.max_x>=?1 AND b.min_x<=?2 AND b.max_y>=?3 AND b.min_y<=?4 AND b.max_z>=?5 AND b.min_z<=?6");
const UNINDEXED_QUERY: &str = concat!(point_select!(), "points AS p WHERE indexed=0");
const SCAN_QUERY: &str = concat!(point_select!(), "points AS p");
const SITE_QUERY: &str = concat!(point_select!(), "bounds AS b JOIN points AS p ON p.id=b.id
    WHERE b.max_x>=?1 AND b.min_x<=?1 AND b.max_y>=?2 AND b.min_y<=?2 AND b.max_z>=?3 AND b.min_z<=?3
    AND p.x=?1 AND p.y=?2 AND p.z=?3 AND p.variable=?4
    UNION ALL ", point_select!(), "points AS p WHERE indexed=0 AND x=?1 AND y=?2 AND z=?3 AND variable=?4");

use super::{Checkpoint, CheckpointStore};
impl Checkpoint {
    fn encode(&self) -> [u8; 40] {
        let mut bytes = [0; 40];
        bytes[..32].copy_from_slice(&self.identity);
        bytes[32..].copy_from_slice(&self.completed_group.to_le_bytes());
        bytes
    }
    fn decode(bytes: &[u8]) -> Result<Self> {
        if bytes.len() != 40 {
            return Err(invalid("checkpoint length"));
        }
        Ok(Self {
            identity: bytes[..32].try_into().unwrap(),
            completed_group: u64::from_le_bytes(bytes[32..].try_into().unwrap()),
        })
    }
}

/// A single connection; use SqliteReadPool for concurrent read access.
/// SQLite serializes writers; every reader
/// query uses a transaction so concurrent writes cannot change its candidate set.
/// Cache accounting excludes SQLite metadata, query heap and operating-system caches.
pub struct SqliteStore {
    connection: Connection,
    path: PathBuf,
    _cache: Reservation,
}
impl SqliteStore {
    pub fn create(path: impl AsRef<Path>, cache_bytes: u64, budget: &MemoryBudget) -> Result<Self> {
        let path = path.as_ref();
        // Reserve the name exclusively. Never truncate an existing user file.
        let file = OpenOptions::new().write(true).create_new(true).open(path)?;
        drop(file);
        let result = (|| {
            let out = Self::connect(path, cache_bytes, budget)?;
            out.connection.execute_batch(
                "PRAGMA journal_mode=DELETE; PRAGMA synchronous=FULL;
                CREATE TABLE metadata(key TEXT PRIMARY KEY,value BLOB NOT NULL) WITHOUT ROWID;
                CREATE TABLE points(id INTEGER PRIMARY KEY,variable INTEGER NOT NULL,
                    x REAL NOT NULL,y REAL NOT NULL,z REAL NOT NULL,value REAL NOT NULL,
                    noise REAL NOT NULL,indexed INTEGER NOT NULL CHECK(indexed IN (0,1)),
                    zero_signs INTEGER NOT NULL CHECK(zero_signs BETWEEN 0 AND 31),
                    support BLOB CHECK(support IS NULL OR (typeof(support)='blob' AND length(support)=48)));
                CREATE INDEX unindexed_points ON points(indexed) WHERE indexed=0;
                CREATE VIRTUAL TABLE bounds USING rtree(id,min_x,max_x,min_y,max_y,min_z,max_z);",
            )?;
            out.connection
                .pragma_update(None, "application_id", APPLICATION_ID)?;
            out.connection
                .pragma_update(None, "user_version", SCHEMA_VERSION)?;
            Ok(out)
        })();
        if result.is_err() {
            let _ = std::fs::remove_file(path);
        }
        result
    }
    pub fn open(path: impl AsRef<Path>, cache_bytes: u64, budget: &MemoryBudget) -> Result<Self> {
        let out = Self::connect(path.as_ref(), cache_bytes, budget)?;
        out.validate_format()?;
        Ok(out)
    }
    fn validate_format(&self) -> Result<()> {
        let app: i64 = self
            .connection
            .pragma_query_value(None, "application_id", |row| row.get(0))?;
        let version: i64 = self
            .connection
            .pragma_query_value(None, "user_version", |row| row.get(0))?;
        if app != APPLICATION_ID || version != SCHEMA_VERSION {
            if app == APPLICATION_ID && version == 1 {
                return Err(invalid("spatial database version 1 loses negative-zero signs; explicitly migrate uncheckpointed data or restart checkpointed simulations"));
            }
            if app == APPLICATION_ID && version == 2 {
                return Err(invalid("spatial database version 2 requires explicit migrate_v2 before block support is available"));
            }
            return Err(invalid("unrecognized spatial database format"));
        }
        Ok(())
    }
    fn connect(path: &Path, cache_bytes: u64, budget: &MemoryBudget) -> Result<Self> {
        Self::connect_mode(path, cache_bytes, budget, OpenFlags::SQLITE_OPEN_READ_WRITE)
    }
    fn connect_mode(
        path: &Path,
        cache_bytes: u64,
        budget: &MemoryBudget,
        mode: OpenFlags,
    ) -> Result<Self> {
        let (connection, path, reservation) =
            configured_connection(path, cache_bytes, budget, mode)?;
        Ok(Self {
            connection,
            path,
            _cache: reservation,
        })
    }
    /// The resolved absolute database path, including non-UTF-8 names on Unix.
    pub fn path(&self) -> &Path {
        &self.path
    }
    pub fn len(&self) -> Result<u64> {
        Ok(self
            .connection
            .query_row("SELECT COUNT(*) FROM points", [], |row| {
                row.get::<_, i64>(0)
            })? as u64)
    }
    pub fn is_empty(&self) -> Result<bool> {
        Ok(self.len()? == 0)
    }
    pub fn checkpoint(&self) -> Result<Option<Checkpoint>> {
        checkpoint(&self.connection)
    }
    /// One transaction streams all records; cancellation or validation errors roll
    /// back both payload and spatial index, including earlier records in this import.
    pub fn import(
        &mut self,
        source: &(impl ObservationSource + ?Sized),
        cancel: &Cancellation,
    ) -> Result<()> {
        let transaction = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        if checkpoint(&transaction)?.is_some() {
            return Err(invalid("checkpointed stores require commit_group"));
        }
        source.scan(&mut |observation| {
            cancel.check()?;
            insert(&transaction, observation)
        })?;
        cancel.check()?;
        transaction.commit()?;
        Ok(())
    }
    /// Compare-and-swap of checkpoint and output. A stale or mismatched writer fails
    /// without leaving partial output, even if failure occurs after several inserts.
    pub fn commit_group(
        &mut self,
        observations: &[Observation],
        expected: Option<&Checkpoint>,
        next: &Checkpoint,
        cancel: &Cancellation,
    ) -> Result<()> {
        cancel.check()?;
        if expected.is_some_and(|old| {
            old.identity != next.identity || old.completed_group >= next.completed_group
        }) {
            return Err(invalid(
                "checkpoint identity changed or group did not advance",
            ));
        }
        let transaction = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        if checkpoint(&transaction)?.as_ref() != expected {
            return Err(invalid("stale checkpoint or simulation identity mismatch"));
        }
        for observation in observations {
            cancel.check()?;
            insert(&transaction, observation)?;
        }
        transaction.execute("INSERT INTO metadata(key,value) VALUES('checkpoint',?1) ON CONFLICT(key) DO UPDATE SET value=excluded.value",[next.encode().as_slice()])?;
        cancel.check()?;
        #[cfg(test)]
        crash_tests::hit("before_commit", next.completed_group);
        transaction.commit()?;
        #[cfg(test)]
        crash_tests::hit("after_commit", next.completed_group);
        Ok(())
    }
}
/// Shared connection policy for durable observation and variogram stores.
pub(crate) fn configured_connection(
    path: &Path,
    cache_bytes: u64,
    budget: &MemoryBudget,
    mode: OpenFlags,
) -> Result<(Connection, PathBuf, Reservation)> {
    let kib = i32::try_from(cache_bytes / 1024)
        .map_err(|_| invalid("SQLite cache exceeds supported size"))?;
    if kib < 4 {
        return Err(invalid("SQLite cache must be at least 4096 bytes"));
    }
    let reservation = budget.reserve(cache_bytes)?;
    let path = std::fs::canonicalize(path)?;
    let connection = Connection::open_with_flags(&path, mode | OpenFlags::SQLITE_OPEN_NO_MUTEX)?;
    connection.busy_timeout(std::time::Duration::from_secs(5))?;
    connection.pragma_update(None, "cache_size", -kib)?;
    connection.pragma_update(None, "mmap_size", 0)?;
    connection.pragma_update(None, "temp_store", "FILE")?;
    connection.pragma_update(None, "synchronous", "FULL")?;
    Ok((connection, path, reservation))
}
fn checkpoint(connection: &Connection) -> Result<Option<Checkpoint>> {
    connection
        .query_row(
            "SELECT value FROM metadata WHERE key='checkpoint'",
            [],
            |row| {
                Ok(match row.get_ref(0)? {
                    rusqlite::types::ValueRef::Blob(bytes) => Checkpoint::decode(bytes),
                    _ => Err(invalid("checkpoint must be a blob")),
                })
            },
        )
        .optional()?
        .transpose()
}
pub(crate) fn indexable(point: [f64; 3]) -> bool {
    // Subnormals and enormous coordinates bypass binary32 pruning altogether.
    point.iter().all(|x| {
        *x == 0.0 || (x.abs() >= f32::MIN_POSITIVE as f64 && x.abs() <= f32::MAX as f64 / 4.0)
    })
}
fn insert(connection: &Connection, observation: &Observation) -> Result<()> {
    let support = support::encode(&observation.site().support)?;
    let p = observation.site().support.center().coordinates();
    let indexed = indexable(p);
    let id = observation.id() as i64;
    let variable =
        u64::try_from(observation.site().variable).map_err(|_| Error::SizeOverflow)? as i64;
    connection
        .prepare_cached("INSERT INTO points VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9,?10)")?
        .execute(params![
            id,
            variable,
            p[0],
            p[1],
            p[2],
            observation.value(),
            observation.noise_variance(),
            indexed,
            [
                p[0],
                p[1],
                p[2],
                observation.value(),
                observation.noise_variance()
            ]
            .into_iter()
            .enumerate()
            .fold(0_u8, |mask, (i, value)| {
                mask | (u8::from(value.to_bits() == (-0.0_f64).to_bits()) << i)
            }),
            support.as_ref().map(|bytes| bytes.as_slice())
        ])?;
    #[cfg(test)]
    crash_tests::hit("payload", observation.id());
    if indexed {
        connection
            .prepare_cached("INSERT INTO bounds VALUES(?1,?2,?2,?3,?3,?4,?4)")?
            .execute(params![id, p[0], p[1], p[2]])?;
    }
    Ok(())
}
fn observation(row: &rusqlite::Row<'_>) -> Result<Observation> {
    let variable =
        usize::try_from(row.get::<_, i64>(1)? as u64).map_err(|_| Error::SizeOverflow)?;
    let mask: u8 = row.get(7)?;
    if mask > 31 {
        return Err(invalid("invalid negative-zero sign mask"));
    }
    let mut fields: [f64; 5] = [
        row.get(2)?,
        row.get(3)?,
        row.get(4)?,
        row.get(5)?,
        row.get(6)?,
    ];
    if mask != 0 {
        for (i, value) in fields.iter_mut().enumerate() {
            if mask & (1 << i) != 0 {
                if *value != 0.0 {
                    return Err(invalid("negative-zero sign mask marks a nonzero payload"));
                }
                *value = -0.0;
            }
        }
    }
    Observation::new(
        row.get::<_, i64>(0)? as u64,
        Site {
            support: support::decode(
                Point3::new([fields[0], fields[1], fields[2]])?,
                row.get_ref(8)?,
            )?,
            variable,
        },
        fields[3],
        fields[4],
    )
}
impl SqliteStore {
    fn read_snapshot<T>(&self, operation: impl FnOnce() -> Result<T>) -> Result<T> {
        // Snapshot-pool connections already hold a lifetime transaction.
        let transaction = self
            .connection
            .is_autocommit()
            .then(|| self.connection.unchecked_transaction())
            .transpose()?;
        let value = operation()?;
        if let Some(transaction) = transaction {
            transaction.commit()?;
        }
        Ok(value)
    }
    fn visit_bounds(
        &self,
        lower: [f64; 3],
        upper: [f64; 3],
        cancel: &Cancellation,
        visitor: &mut dyn FnMut(&Observation) -> Result<()>,
    ) -> Result<()> {
        cancel.check()?;
        self.read_snapshot(|| {
            let mut statement = self.connection.prepare_cached(BOUNDS_QUERY)?;
            let mut rows = statement.query(params![
                lower[0], upper[0], lower[1], upper[1], lower[2], upper[2]
            ])?;
            while let Some(row) = rows.next()? {
                cancel.check()?;
                visitor(&observation(row)?)?;
            }
            let mut statement = self.connection.prepare_cached(UNINDEXED_QUERY)?;
            let mut rows = statement.query([])?;
            while let Some(row) = rows.next()? {
                cancel.check()?;
                visitor(&observation(row)?)?;
            }
            Ok(())
        })
    }
}
impl ObservationSource for SqliteStore {
    fn visit_bounds(
        &self,
        lower: [f64; 3],
        upper: [f64; 3],
        cancel: &Cancellation,
        visitor: &mut dyn FnMut(&Observation) -> Result<()>,
    ) -> Result<()> {
        SqliteStore::visit_bounds(self, lower, upper, cancel, visitor)
    }
    fn nearest_into<'a>(
        &self,
        point: Point3,
        workspace: &'a mut NearestWorkspace,
        cancel: &Cancellation,
    ) -> Result<&'a [Observation]> {
        let radius = workspace.query().radius;
        // Prior query density supplies a hint only. A successful probe must
        // contain k records whose exact distances lie inside its sphere. Then
        // every excluded indexed record is farther away. Otherwise discard the
        // probe and execute the full-radius query, preserving exact membership.
        let probe = workspace.sqlite_probe_radius();
        workspace.reset(point);
        let p = point.coordinates();
        if let Some(probe) = probe {
            self.visit_bounds(
                p.map(|x| (x - probe).next_down()),
                p.map(|x| (x + probe).next_up()),
                cancel,
                &mut |observation| {
                    workspace.offer(observation);
                    Ok(())
                },
            )?;
            if workspace.filled_within(probe) {
                return workspace.finish(cancel);
            }
            workspace.sqlite_probe_failed();
            workspace.reset(point);
        }
        self.visit_bounds(
            p.map(|x| (x - radius).next_down()),
            p.map(|x| (x + radius).next_up()),
            cancel,
            &mut |observation| {
                workspace.offer(observation);
                Ok(())
            },
        )?;
        workspace.finish(cancel)
    }

    fn search<'a>(
        &self,
        point: Point3,
        workspace: &'a mut SearchWorkspace<'_>,
        cancel: &Cancellation,
    ) -> Result<&'a [Observation]> {
        workspace.reset(point);
        cancel.check()?;
        // Keep all expansion attempts in one read snapshot. Snapshot-pool
        // readers already hold a transaction; a fixed query retains its usual
        // transaction inside visit_bounds without an additional wrapper.
        if workspace.has_expansion() {
            self.read_snapshot(|| workspace.search_source(self, point, cancel))
        } else {
            workspace.search_source(self, point, cancel)
        }
    }

    fn at_site(&self, site: &Site) -> Result<Vec<Observation>> {
        let p = site.support.center().coordinates();
        let variable = u64::try_from(site.variable).map_err(|_| Error::SizeOverflow)? as i64;
        let mut statement = self.connection.prepare_cached(SITE_QUERY)?;
        let mut rows = statement.query(params![p[0], p[1], p[2], variable])?;
        let mut found = Vec::new();
        while let Some(row) = rows.next()? {
            let observation = observation(row)?;
            if observation.site() == site {
                found.push(observation);
            }
        }
        found.sort_by_key(Observation::id);
        Ok(found)
    }
    fn scan(&self, visitor: &mut dyn FnMut(&Observation) -> Result<()>) -> Result<()> {
        let mut statement = self.connection.prepare_cached(SCAN_QUERY)?;
        let mut rows = statement.query([])?;
        while let Some(row) = rows.next()? {
            visitor(&observation(row)?)?;
        }
        Ok(())
    }
    fn nearest(
        &self,
        point: Point3,
        query: Neighborhood,
        cancel: &Cancellation,
    ) -> Result<Vec<Observation>> {
        query.validate()?;
        cancel.check()?;
        let mut heap = BinaryHeap::<Candidate>::new();
        heap.try_reserve(query.maximum)
            .map_err(|_| invalid("unable to allocate neighborhood workspace"))?;
        let p = point.coordinates();
        let lower = p.map(|x| (x - query.radius).next_down());
        let upper = p.map(|x| (x + query.radius).next_up());
        self.visit_bounds(lower, upper, cancel, &mut |observation| {
            offer(&mut heap, point, query, observation);
            Ok(())
        })?;
        Ok(finish(heap))
    }
}
impl ObservationSink for SqliteStore {
    fn append(&mut self, observation: Observation) -> Result<()> {
        self.import(std::slice::from_ref(&observation), &Cancellation::default())
    }
}

impl CheckpointStore for SqliteStore {
    fn checkpoint(&self) -> Result<Option<Checkpoint>> {
        SqliteStore::checkpoint(self)
    }
    fn is_empty(&self) -> Result<bool> {
        SqliteStore::is_empty(self)
    }
    fn commit_group(
        &mut self,
        observations: &[Observation],
        expected: Option<&Checkpoint>,
        next: &Checkpoint,
        cancel: &Cancellation,
    ) -> Result<()> {
        SqliteStore::commit_group(self, observations, expected, next, cancel)
    }
}
