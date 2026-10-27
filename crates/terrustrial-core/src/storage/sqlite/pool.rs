//! Bounded read-only connections with short pool locks and cancellation-aware waits.
use super::*;
use crate::error::checked_len;
use std::{
    sync::{Condvar, Mutex},
    thread::{self, ThreadId},
    time::Duration,
};

struct State {
    idle: Vec<SqliteStore>,
    borrowers: Vec<ThreadId>,
    failed_snapshot: bool,
}
/// Concurrent read-only access for parallel estimation and hard conditioning.
/// Each query exclusively leases one connection and releases the pool mutex before
/// executing SQL. The connection count and configured page caches are bounded.
/// Waiting queries poll supplied cancellation tokens at 10 ms intervals.
/// SQLite's own lock wait retains the store's five-second timeout.
///
/// `open` gives each query a transaction; `open_snapshot` retains a common
/// database snapshot across queries and readers. Use the latter for conditioning
/// data that may receive concurrent writes. Otherwise keep the database unchanged
/// during a Gaussian driver call. Callbacks must not reenter the same pool on the same
/// thread; that returns an error instead of waiting on its own lease. Cross-thread
/// callback cycles must also be avoided. Cache accounting excludes SQLite metadata,
/// query output, allocator overhead and operating-system caches.
///
/// ```no_run
/// use terrustrial_core::{execution::{Cancellation, MemoryBudget}, storage::SqliteReadPool};
/// # fn main() -> terrustrial_core::Result<()> {
/// let budget = MemoryBudget::new(64 << 20);
/// let cancel = Cancellation::default();
/// let conditioning = SqliteReadPool::open_snapshot(
///     "observations.sqlite", 4, 8 << 20, &budget, &cancel)?;
/// assert_eq!(conditioning.readers(), 4);
/// // Share `&conditioning` with the parallel Gaussian driver.
/// # Ok(())
/// # }
/// ```
pub struct SqliteReadPool {
    state: Mutex<State>,
    available: Condvar,
    readers: usize,
    snapshot: bool,
    _reservation: Reservation,
}
impl SqliteReadPool {
    pub fn open(
        path: impl AsRef<Path>,
        readers: usize,
        cache_bytes_per_reader: u64,
        budget: &MemoryBudget,
    ) -> Result<Self> {
        Self::open_mode(
            path.as_ref(),
            readers,
            cache_bytes_per_reader,
            budget,
            false,
            None,
        )
    }

    /// Open a pool whose readers share one immutable database view until drop.
    /// Retain this pool across fingerprinting and the entire estimation/simulation
    /// call. Reopening creates a new snapshot and requires a new fingerprint.
    ///
    /// With multiple readers, a temporary read-write coordinator uses BEGIN
    /// IMMEDIATE while reader snapshots are established. It changes no data and
    /// is released before return, but requires write access and an additional
    /// 4096-byte cache reservation during construction. One reader needs neither.
    /// WAL writers can continue after construction; long-lived readers can retain
    /// WAL history. Rollback-journal writer commits wait until this pool is dropped.
    /// Cancellation is checked between connection operations; SQLite lock waits
    /// retain the five-second timeout. Any lost reader transaction fails the pool
    /// rather than silently opening a newer snapshot. External file replacement or
    /// corruption is outside SQLite's transactional protection.
    pub fn open_snapshot(
        path: impl AsRef<Path>,
        readers: usize,
        cache_bytes_per_reader: u64,
        budget: &MemoryBudget,
        cancel: &Cancellation,
    ) -> Result<Self> {
        cancel.check()?;
        if readers == 0 {
            return Err(invalid("SQLite read pool needs at least one connection"));
        }
        if readers == 1 {
            return Self::open_mode(
                path.as_ref(),
                readers,
                cache_bytes_per_reader,
                budget,
                true,
                Some(cancel),
            );
        }
        let coordinator = SqliteStore::connect_mode(
            path.as_ref(),
            4096,
            budget,
            OpenFlags::SQLITE_OPEN_READ_WRITE | OpenFlags::SQLITE_OPEN_PRIVATE_CACHE,
        )?;
        let transaction = rusqlite::Transaction::new_unchecked(
            &coordinator.connection,
            TransactionBehavior::Immediate,
        )?;
        coordinator.validate_format()?;
        cancel.check()?;
        let pool = Self::open_mode(
            &coordinator.path,
            readers,
            cache_bytes_per_reader,
            budget,
            true,
            Some(cancel),
        )?;
        cancel.check()?;
        transaction.rollback()?;
        Ok(pool)
    }

    fn open_mode(
        path: &Path,
        readers: usize,
        cache_bytes_per_reader: u64,
        budget: &MemoryBudget,
        snapshot: bool,
        cancel: Option<&Cancellation>,
    ) -> Result<Self> {
        if readers == 0 {
            return Err(invalid("SQLite read pool needs at least one connection"));
        }
        let cache_bytes = u64::try_from(readers)
            .map_err(|_| Error::SizeOverflow)?
            .checked_mul(cache_bytes_per_reader)
            .ok_or(Error::SizeOverflow)?;
        let slots = checked_len(&[
            readers,
            std::mem::size_of::<SqliteStore>() + std::mem::size_of::<ThreadId>(),
        ])?;
        let total = cache_bytes
            .checked_add(u64::try_from(slots).map_err(|_| Error::SizeOverflow)?)
            .ok_or(Error::SizeOverflow)?;
        let reservation = budget.reserve(total)?;
        let caches = MemoryBudget::new(cache_bytes);
        let mut idle = Vec::new();
        let mut borrowers = Vec::new();
        idle.try_reserve_exact(readers)
            .map_err(|_| invalid("SQLite pool slots allocation"))?;
        borrowers
            .try_reserve_exact(readers)
            .map_err(|_| invalid("SQLite pool borrowers allocation"))?;
        for _ in 0..readers {
            if let Some(cancel) = cancel {
                cancel.check()?;
            }
            let store = SqliteStore::connect_mode(
                path,
                cache_bytes_per_reader,
                &caches,
                OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_PRIVATE_CACHE,
            )?;
            if snapshot {
                let mut transaction = store.connection.unchecked_transaction()?;
                store.validate_format()?;
                // BEGIN is deferred. Reading even an empty table establishes
                // the snapshot while the coordinator excludes all writers.
                transaction
                    .query_row("SELECT id FROM points LIMIT 1", [], |r| r.get::<_, i64>(0))
                    .optional()?;
                if let Some(cancel) = cancel {
                    cancel.check()?;
                }
                // The read-only connection owns this transaction until close.
                transaction.set_drop_behavior(rusqlite::DropBehavior::Ignore);
            } else {
                store.validate_format()?;
            }
            idle.push(store);
        }
        Ok(Self {
            state: Mutex::new(State {
                idle,
                borrowers,
                failed_snapshot: false,
            }),
            available: Condvar::new(),
            readers,
            snapshot,
            _reservation: reservation,
        })
    }
    pub fn len(&self) -> Result<u64> {
        self.lease(None)?.store().len()
    }
    pub fn is_empty(&self) -> Result<bool> {
        self.lease(None)?.store().is_empty()
    }
    pub fn readers(&self) -> usize {
        self.readers
    }
    pub fn is_snapshot(&self) -> bool {
        self.snapshot
    }
    pub fn accounted_bytes(&self) -> u64 {
        self._reservation.bytes()
    }
    fn lease(&self, cancel: Option<&Cancellation>) -> Result<Lease<'_>> {
        let owner = thread::current().id();
        let mut state = self
            .state
            .lock()
            .map_err(|_| invalid("SQLite read pool poisoned"))?;
        loop {
            if state.failed_snapshot {
                return Err(invalid("SQLite read snapshot was lost; reopen the pool"));
            }
            if let Some(cancel) = cancel {
                if let Err(error) = cancel.check() {
                    // A cancelled waiter may consume the notification for an
                    // available connection. Pass it on to an uncancelled waiter.
                    if !state.idle.is_empty() {
                        self.available.notify_one();
                    }
                    return Err(error);
                }
            }
            if state.borrowers.contains(&owner) {
                return Err(invalid("reentrant SQLite read pool query"));
            }
            if let Some(store) = state.idle.pop() {
                if self.snapshot && store.connection.is_autocommit() {
                    state.failed_snapshot = true;
                    state.idle.push(store);
                    self.available.notify_all();
                    return Err(invalid("SQLite read snapshot was lost; reopen the pool"));
                }
                state.borrowers.push(owner);
                return Ok(Lease {
                    pool: self,
                    store: Some(store),
                    owner,
                });
            }
            state = if cancel.is_some() {
                self.available
                    .wait_timeout(state, Duration::from_millis(10))
                    .map_err(|_| invalid("SQLite read pool poisoned"))?
                    .0
            } else {
                self.available
                    .wait(state)
                    .map_err(|_| invalid("SQLite read pool poisoned"))?
            };
        }
    }
}
struct Lease<'a> {
    pool: &'a SqliteReadPool,
    store: Option<SqliteStore>,
    owner: ThreadId,
}
impl Lease<'_> {
    fn store(&self) -> &SqliteStore {
        self.store.as_ref().expect("leased SQLite connection")
    }
}
impl Drop for Lease<'_> {
    fn drop(&mut self) {
        // No user code runs while this mutex is held. Recover on unwinding so a
        // callback panic cannot strand a connection or leak its cache reservation.
        let mut state = self
            .pool
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if let Some(index) = state
            .borrowers
            .iter()
            .position(|owner| *owner == self.owner)
        {
            state.borrowers.swap_remove(index);
        }
        let store = self.store.take().expect("leased SQLite connection");
        if self.pool.snapshot && store.connection.is_autocommit() {
            state.failed_snapshot = true;
        }
        state.idle.push(store);
        let failed = state.failed_snapshot;
        drop(state);
        if failed {
            self.pool.available.notify_all();
        } else {
            self.pool.available.notify_one();
        }
    }
}
impl ObservationSource for SqliteReadPool {
    fn scan(&self, visitor: &mut dyn FnMut(&Observation) -> Result<()>) -> Result<()> {
        self.lease(None)?.store().scan(visitor)
    }
    fn visit_bounds(
        &self,
        lower: [f64; 3],
        upper: [f64; 3],
        cancel: &Cancellation,
        visitor: &mut dyn FnMut(&Observation) -> Result<()>,
    ) -> Result<()> {
        self.lease(Some(cancel))?
            .store()
            .visit_bounds(lower, upper, cancel, visitor)
    }
    fn at_site(&self, site: &Site) -> Result<Vec<Observation>> {
        self.lease(None)?.store().at_site(site)
    }
    fn nearest(
        &self,
        point: Point3,
        query: Neighborhood,
        cancel: &Cancellation,
    ) -> Result<Vec<Observation>> {
        self.lease(Some(cancel))?
            .store()
            .nearest(point, query, cancel)
    }
    fn nearest_into<'a>(
        &self,
        point: Point3,
        workspace: &'a mut NearestWorkspace,
        cancel: &Cancellation,
    ) -> Result<&'a [Observation]> {
        self.lease(Some(cancel))?
            .store()
            .nearest_into(point, workspace, cancel)
    }
    fn search<'a>(
        &self,
        point: Point3,
        workspace: &'a mut SearchWorkspace<'_>,
        cancel: &Cancellation,
    ) -> Result<&'a [Observation]> {
        workspace.reset(point);
        self.lease(Some(cancel))?
            .store()
            .search(point, workspace, cancel)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        panic::{catch_unwind, AssertUnwindSafe},
        sync::mpsc,
    };
    fn database() -> (tempfile::TempDir, PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("points.sqlite");
        let budget = MemoryBudget::new(1 << 20);
        let mut store = SqliteStore::create(&path, 4096, &budget).unwrap();
        let rows: Vec<_> = (0..16)
            .map(|id| {
                Observation::new(
                    id,
                    Site::point(Point3::new([id as f64, 0., 0.]).unwrap(), 0),
                    id as f64,
                    0.,
                )
                .unwrap()
            })
            .collect();
        store
            .import(rows.as_slice(), &Cancellation::default())
            .unwrap();
        drop(store);
        assert_eq!(budget.used(), 0);
        (dir, path)
    }
    #[test]
    fn snapshot_admission_all_readers_error_recovery_and_lost_transaction() {
        let (_directory, path) = database();
        let budget = MemoryBudget::new(1 << 20);
        let cancel = Cancellation::default();
        let writer = Connection::open(&path).unwrap();
        writer.execute_batch("PRAGMA journal_mode=WAL").unwrap();
        assert!(SqliteReadPool::open_snapshot(&path, 0, 4096, &budget, &cancel).is_err());
        assert!(
            SqliteReadPool::open_snapshot(&path, usize::MAX, u64::MAX, &budget, &cancel).is_err()
        );
        assert!(SqliteReadPool::open_snapshot(&path, 2, 1, &budget, &cancel).is_err());
        let stopped = Cancellation::default();
        stopped.cancel();
        assert!(matches!(
            SqliteReadPool::open_snapshot(&path, 2, 4096, &budget, &stopped),
            Err(Error::Cancelled)
        ));
        assert_eq!(budget.used(), 0);
        let pool = SqliteReadPool::open_snapshot(&path, 4, 4096, &budget, &cancel).unwrap();
        writer
            .execute_batch("UPDATE points SET value=value+100")
            .unwrap();
        {
            let state = pool.state.lock().unwrap();
            assert_eq!(state.idle.len(), 4);
            for store in &state.idle {
                assert!(!store.connection.is_autocommit());
                let value: f64 = store
                    .connection
                    .query_row("SELECT value FROM points WHERE id=0", [], |r| r.get(0))
                    .unwrap();
                assert_eq!(value, 0.);
                assert!(store
                    .connection
                    .execute("UPDATE points SET value=200", [])
                    .is_err());
            }
        }
        let retained = pool.accounted_bytes();
        drop(pool);
        assert_eq!(budget.used(), 0);
        let small = MemoryBudget::new(retained + 4095);
        assert!(matches!(
            SqliteReadPool::open_snapshot(&path, 4, 4096, &small, &cancel),
            Err(Error::MemoryBudget { .. })
        ));
        assert_eq!(small.used(), 0);
        let exact = MemoryBudget::new(retained + 4096);
        let pool = SqliteReadPool::open_snapshot(&path, 4, 4096, &exact, &cancel).unwrap();
        assert_eq!(exact.used(), retained);
        assert!(pool
            .visit_bounds([0.; 3], [20.; 3], &cancel, &mut |_| Err(Error::Cancelled))
            .is_err());
        assert!(catch_unwind(AssertUnwindSafe(|| {
            let _ = pool.scan(&mut |_| panic!("snapshot visitor"));
        }))
        .is_err());
        assert_eq!(pool.state.lock().unwrap().idle.len(), 4);
        assert!(!pool.state.lock().unwrap().failed_snapshot);
        let mut workspace = NearestWorkspace::new(
            Neighborhood {
                maximum: 8,
                radius: 20.,
            },
            &budget,
        )
        .unwrap();
        let point = Point3::new([0.; 3]).unwrap();
        pool.nearest_into(point, &mut workspace, &cancel).unwrap();
        let measured = allocation_counter::measure(|| {
            for _ in 0..100 {
                assert_eq!(
                    pool.nearest_into(point, &mut workspace, &cancel)
                        .unwrap()
                        .len(),
                    8
                );
            }
        });
        // rusqlite 0.40.2 replaces an Arc<str> cache key on every cached
        // statement checkout. Snapshot handling must add no allocations over
        // the same warmed ordinary query; SQL queries are not allocation-free.
        let live = SqliteReadPool::open(&path, 1, 4096, &budget).unwrap();
        live.nearest_into(point, &mut workspace, &cancel).unwrap();
        let ordinary = allocation_counter::measure(|| {
            for _ in 0..100 {
                assert_eq!(
                    live.nearest_into(point, &mut workspace, &cancel)
                        .unwrap()
                        .len(),
                    8
                );
            }
        });
        assert_eq!(measured.count_total, ordinary.count_total);
        let leases = allocation_counter::measure(|| {
            for _ in 0..100 {
                drop(pool.lease(Some(&cancel)).unwrap());
            }
        });
        assert_eq!(leases.count_total, 0);
        drop(live);
        // Simulate SQLite losing one reader transaction. No newer view may be
        // admitted by that connection or by another reader in the same pool.
        {
            let lease = pool.lease(None).unwrap();
            lease.store().connection.execute_batch("ROLLBACK").unwrap();
        }
        writer
            .execute_batch("UPDATE points SET value=value+100")
            .unwrap();
        assert!(pool.len().is_err());
        assert!(pool.nearest_into(point, &mut workspace, &cancel).is_err());
        drop(pool);
        assert_eq!(exact.used(), 0);
        drop(workspace);
        assert_eq!(budget.used(), 0);
    }
    #[test]
    fn admission_read_only_failure_unwinding_and_reentrancy_release_resources() {
        let (_dir, path) = database();
        let budget = MemoryBudget::new(1 << 20);
        assert!(SqliteReadPool::open(&path, 0, 4096, &budget).is_err());
        assert!(SqliteReadPool::open(&path, usize::MAX, u64::MAX, &budget).is_err());
        assert!(SqliteReadPool::open(&path, 2, 1, &budget).is_err());
        assert!(SqliteReadPool::open(&path, 2, 4096, &MemoryBudget::new(8191)).is_err());
        assert!(SqliteReadPool::open(path.with_extension("missing"), 2, 4096, &budget).is_err());
        assert!(!path.with_extension("missing").exists());
        assert_eq!(budget.used(), 0);
        let pool = SqliteReadPool::open(&path, 2, 4096, &budget).unwrap();
        assert_eq!(pool.len().unwrap(), 16);
        assert!(!pool.is_empty().unwrap());
        assert_eq!(pool.readers(), 2);
        assert_eq!(budget.used(), pool.accounted_bytes());
        assert!(pool.accounted_bytes() >= 8192);
        {
            let lease = pool.lease(None).unwrap();
            assert!(lease
                .store()
                .connection
                .execute("DELETE FROM points", [])
                .is_err());
            assert!(pool.lease(None).is_err());
        }
        let cancel = Cancellation::default();
        assert!(pool
            .visit_bounds([0.; 3], [20.; 3], &cancel, &mut |_| Err(Error::Cancelled))
            .is_err());
        assert!(catch_unwind(AssertUnwindSafe(|| {
            let _ = pool.visit_bounds([0.; 3], [20.; 3], &cancel, &mut |_| panic!("visitor panic"));
        }))
        .is_err());
        let mut count = 0;
        pool.scan(&mut |_| {
            count += 1;
            Ok(())
        })
        .unwrap();
        assert_eq!(count, 16);
        assert_eq!(pool.state.lock().unwrap().idle.len(), 2);
        let measured = allocation_counter::measure(|| {
            for _ in 0..100 {
                drop(pool.lease(Some(&cancel)).unwrap());
            }
        });
        assert_eq!(measured.count_total, 0);
        drop(pool);
        assert_eq!(budget.used(), 0);
    }
    #[test]
    fn separate_connections_overlap_and_capacity_waits_can_be_cancelled() {
        let (_dir, path) = database();
        let budget = MemoryBudget::new(1 << 20);
        let pool = SqliteReadPool::open(&path, 2, 4096, &budget).unwrap();
        thread::scope(|scope| {
            let (entered_tx, entered_rx) = mpsc::channel();
            let mut releases = Vec::new();
            let mut handles = Vec::new();
            for _ in 0..2 {
                let (release_tx, release_rx) = mpsc::channel();
                releases.push(release_tx);
                let entered = entered_tx.clone();
                let pool = &pool;
                handles.push(scope.spawn(move || {
                    let mut first = true;
                    pool.scan(&mut |_| {
                        if first {
                            first = false;
                            entered.send(()).unwrap();
                            release_rx.recv_timeout(Duration::from_secs(5)).unwrap();
                        }
                        Ok(())
                    })
                }));
            }
            // Both callbacks must enter before either lease is returned.
            for _ in 0..2 {
                entered_rx.recv_timeout(Duration::from_secs(5)).unwrap();
            }
            assert_eq!(pool.state.lock().unwrap().borrowers.len(), 2);
            for release in releases {
                release.send(()).unwrap();
            }
            for handle in handles {
                handle.join().unwrap().unwrap();
            }
        });
        drop(pool);
        let pool = SqliteReadPool::open(&path, 1, 4096, &budget).unwrap();
        let held = pool.lease(None).unwrap();
        let cancel = Cancellation::default();
        thread::scope(|scope| {
            let (tx, rx) = mpsc::channel();
            let pool = &pool;
            let cancel = &cancel;
            scope.spawn(move || {
                let result = pool.lease(Some(cancel));
                tx.send(matches!(result, Err(Error::Cancelled))).unwrap();
            });
            assert!(matches!(
                rx.recv_timeout(Duration::from_millis(30)),
                Err(mpsc::RecvTimeoutError::Timeout)
            ));
            cancel.cancel();
            assert!(rx.recv_timeout(Duration::from_secs(5)).unwrap());
        });
        drop(held);
        assert_eq!(pool.state.lock().unwrap().idle.len(), 1);
        drop(pool);
        assert_eq!(budget.used(), 0);
    }
}
