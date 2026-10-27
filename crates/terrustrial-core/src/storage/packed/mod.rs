//! Packed, versioned spatial pages with a shared decoded-page cache.
//!
//! Geometry pages are immutable. Values are versioned separately; a snapshot
//! captures both a root set and a completed simulation prefix. Construction
//! streams through an on-disk staging index, so Rust working memory does not
//! grow with input size. SQLite and allocator metadata and the OS cache are not
//! covered by the logical MemoryBudget. This backend is experimental until the
//! complete-workload and actual beyond-RAM acceptance matrix is measured.
use crate::execution::profiling::{Metric, Span, Stamp};
use crate::spatial::{NearestWorkspace, Neighborhood, ObservationSource};
use crate::{
    Error, Result,
    data::{Observation, Point3},
    error::invalid,
    execution::{Cancellation, MemoryBudget, Reservation},
};
// Cache keys are internal page identifiers, never persisted hash values.
use foldhash::{HashMap, HashMapExt};
use rusqlite::{Connection, OpenFlags, OptionalExtension, params};
use std::{
    collections::VecDeque,
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex, RwLock, Weak,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
};
mod builder;
mod codec;
mod payload;
mod publication_decode;
mod query;
mod readers;
mod scheduled;
mod updates;
pub use builder::PackedBuilder;
use codec::*;
pub use publication_decode::PublicationDecodeStatistics;
pub use scheduled::{ScheduledOptions, ScheduledView, ScheduledWriter};

const FORMAT: i64 = 0x54525031;
const MAX_ROOTS: usize = 32;

/// Construction and cache capacities are independent of total record count.
#[derive(Clone, Copy, Debug)]
pub struct Options {
    pub leaf_records: usize,
    pub cached_pages: usize,
    /// Bounded construction window, in leaf capacities, for spatial repartitioning.
    pub build_window_leaves: usize,
    pub readers: usize,
    /// Upper branch pages shared by snapshots with identical geometry roots.
    pub directory_pages: usize,
    /// Per SQLite connection, including the writer. A suggested SQLite page
    /// cache allowance, not a hard bound on SQLite's complete allocator usage.
    pub sqlite_cache_bytes: u64,
    /// Approximate WAL byte threshold for automatic checkpointing, rounded up
    /// to database pages. This is a trigger, not a disk-space cap: active readers
    /// may delay checkpoint progress. FULL synchronous commits remain enabled.
    pub wal_checkpoint_bytes: u64,
    /// Cap on the main database's page count times page size. Journals, WAL and
    /// temporary staging files need additional disk space.
    pub max_database_bytes: u64,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            leaf_records: 64,
            cached_pages: 256,
            build_window_leaves: 256,
            readers: 4,
            directory_pages: 32,
            sqlite_cache_bytes: 1 << 20,
            wal_checkpoint_bytes: 64 << 20,
            max_database_bytes: 1 << 40,
        }
    }
}
impl Options {
    fn validate(self) -> Result<()> {
        if !(1..=MAX_LEAF).contains(&self.leaf_records)
            || !(1..=4096).contains(&self.build_window_leaves)
            || self.cached_pages == 0
            || self.cached_pages > 1 << 20
            || self.readers == 0
            || self.directory_pages > 4096
            || self.readers > 256
            || self.sqlite_cache_bytes < 4096
            || !(4096..=1 << 40).contains(&self.wal_checkpoint_bytes)
            || self.max_database_bytes < 4096
        {
            return Err(invalid("packed engine capacities"));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Statistics {
    pub cache_hits: u64,
    pub page_reads: u64,
    pub bytes_read: u64,
    pub geometry_groups: u64,
    pub observations_tested: u64,
}
#[derive(Default)]
struct Counters {
    hits: AtomicU64,
    reads: AtomicU64,
    bytes: AtomicU64,
    groups: AtomicU64,
    observations: AtomicU64,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
struct Key {
    page: u64,
    generation: u64,
    payload: bool,
}
struct Cached {
    page: Page,
    _reservation: Reservation,
}
// Immutable identity, support and noise are shared by cached value versions.
// The original values in these decoded records are superseded by Payload::values.
// Keeping the charge inside the Arc makes its lifetime independent of the first
// cached version: eviction cannot uncharge a base still used by another version.
struct PayloadBase {
    observations: Vec<Observation>,
    _reservation: Reservation,
}
struct Payload {
    base: Arc<PayloadBase>,
    values: Vec<f64>,
}
impl Payload {
    fn len(&self) -> usize {
        self.values.len()
    }
    fn observation(&self, slot: usize) -> Result<Observation> {
        let base = self
            .base
            .observations
            .get(slot)
            .ok_or_else(|| invalid("packed payload slot"))?;
        let value = self
            .values
            .get(slot)
            .ok_or_else(|| invalid("packed payload slot"))?;
        Observation::new(base.id(), *base.site(), *value, base.noise_variance())
    }
    fn update(&mut self, slot: usize, id: u64, value: f64) -> Result<()> {
        let base = self
            .base
            .observations
            .get(slot)
            .ok_or_else(|| invalid("packed value slot"))?;
        if base.id() != id {
            return Err(invalid("packed value identity"));
        }
        let updated = Observation::new(base.id(), *base.site(), value, base.noise_variance())?;
        *self
            .values
            .get_mut(slot)
            .ok_or_else(|| invalid("packed value slot"))? = updated.value();
        Ok(())
    }
}
impl Cached {
    fn payload(bytes: &[u8], capacity: usize, mut reservation: Reservation) -> Result<Self> {
        let observations = decode_payload(bytes, capacity)?;
        let mut values = Vec::new();
        values
            .try_reserve_exact(observations.len())
            .map_err(|_| invalid("packed value cache allocation"))?;
        values.extend(observations.iter().map(Observation::value));
        let base_bytes = observations
            .capacity()
            .checked_mul(std::mem::size_of::<Observation>())
            .and_then(|bytes| bytes.checked_add(256))
            .ok_or(Error::SizeOverflow)?;
        let base_reservation = reservation.split(base_bytes as u64)?;
        Ok(Self {
            page: Page::Payload(Payload {
                base: Arc::new(PayloadBase {
                    observations,
                    _reservation: base_reservation,
                }),
                values,
            }),
            _reservation: reservation,
        })
    }
}
struct Entry {
    value: Arc<Cached>,
    recent: AtomicBool,
    // Inclusive snapshot generation through which this stored version is proven valid.
    valid_through: AtomicU64,
}
impl Entry {
    fn mark_recent(&self) {
        // The cache read guard excludes eviction. Avoid dirtying a shared cache
        // line on every hit when its clock bit is already set.
        if !self.recent.load(Ordering::Relaxed) {
            self.recent.store(true, Ordering::Relaxed);
        }
    }
}
struct Cache {
    entries: HashMap<Key, Entry>,
    // Fast current-version lookup plus a bounded, sorted historical-version index.
    versions: Vec<(u64, u64)>,
    payload_heads: HashMap<u64, Key>,
    generation: Option<u64>,
    queue: VecDeque<Key>,
    capacity: usize,
    _reservation: Reservation,
}
struct Reader {
    connection: Connection,
    _reservation: Reservation,
}
struct PublicationScratch {
    changed: Vec<u64>,
    reservation: Reservation,
    updates: Vec<updates::ValueUpdate>,
    update_reservation: Reservation,
    warmed: Vec<(Key, Arc<Cached>)>,
    warm_reservation: Reservation,
}
struct Inner {
    path: PathBuf,
    options: Options,
    compact_values: bool,
    budget: MemoryBudget,
    writer: Mutex<Connection>,
    publication: Mutex<PublicationScratch>,
    readers: readers::Pool,
    cache: Vec<RwLock<Cache>>,
    geometry: Mutex<Weak<Geometry>>,
    counters: Counters,
    // Private token for source methods that have no cancellation parameter.
    uncancelled: Cancellation,
    _writer_cache: Reservation,
}

/// Shared engine. Read snapshots are immutable and may be queried concurrently.
/// Writers publish complete transactions. Retained historical versions are
/// currently reclaimed by export_compacted into a new store, not in place.
#[derive(Clone)]
pub struct PackedStore {
    inner: Arc<Inner>,
}
/// Geometry and value generation captured atomically. Later commits cannot
/// change this view, including while a parallel Gaussian driver is running.
#[derive(Clone)]
pub struct Snapshot {
    inner: Arc<Inner>,
    manifest: Arc<Manifest>,
}
struct Manifest {
    generation: u64,
    visible: u64,
    count: u64,
    geometry: Arc<Geometry>,
    _reservation: Reservation,
}
struct Geometry {
    roots: Vec<Node>,
    directory: HashMap<u64, Page>,
    _reservation: Reservation,
}

fn lock<T>(mutex: &Mutex<T>) -> Result<std::sync::MutexGuard<'_, T>> {
    mutex
        .lock()
        .map_err(|_| Error::Backend("packed engine lock poisoned".into()))
}
fn sql_u64(row: &rusqlite::Row<'_>, column: usize) -> rusqlite::Result<u64> {
    let value: i64 = row.get(column)?;
    u64::try_from(value).map_err(|_| rusqlite::Error::IntegralValueOutOfRange(column, value))
}
fn sql_usize(row: &rusqlite::Row<'_>, column: usize) -> rusqlite::Result<usize> {
    let value: i64 = row.get(column)?;
    usize::try_from(value).map_err(|_| rusqlite::Error::IntegralValueOutOfRange(column, value))
}
fn sql_id(value: u64) -> Result<i64> {
    i64::try_from(value).map_err(|_| Error::SizeOverflow)
}
fn open_connection(
    path: &Path,
    options: Options,
    budget: &MemoryBudget,
    write: bool,
) -> Result<(Connection, Reservation)> {
    let mode = if write {
        OpenFlags::SQLITE_OPEN_READ_WRITE
    } else {
        OpenFlags::SQLITE_OPEN_READ_ONLY
    };
    let (c, _, reservation) = crate::storage::sqlite::configured_connection(
        path,
        options.sqlite_cache_bytes,
        budget,
        mode | OpenFlags::SQLITE_OPEN_PRIVATE_CACHE,
    )?;
    if write {
        configure_writer_limits(&c, options)?;
    }
    Ok((c, reservation))
}
fn configure_writer_limits(c: &Connection, options: Options) -> Result<()> {
    let page_size: u64 = c.pragma_query_value(None, "page_size", |r| sql_u64(r, 0))?;
    let maximum_pages = options.max_database_bytes / page_size;
    if maximum_pages == 0 {
        return Err(invalid("packed database page exceeds database byte limit"));
    }
    c.pragma_update(
        None,
        "wal_autocheckpoint",
        i32::try_from(options.wal_checkpoint_bytes.div_ceil(page_size))
            .map_err(|_| invalid("WAL checkpoint threshold exceeds SQLite page count"))?,
    )?;
    c.pragma_update(None, "max_page_count", sql_id(maximum_pages)?)?;
    Ok(())
}
impl PackedStore {
    pub fn open(
        path: impl AsRef<Path>,
        mut options: Options,
        budget: &MemoryBudget,
    ) -> Result<Self> {
        options.validate()?;
        let path = std::fs::canonicalize(path)?;
        let (writer, writer_reservation) = open_connection(&path, options, budget, true)?;
        let format: i64 = writer.pragma_query_value(None, "application_id", |r| r.get(0))?;
        let version: i64 = writer.pragma_query_value(None, "user_version", |r| r.get(0))?;
        if format != FORMAT || ![2, 3].contains(&version) {
            return Err(invalid("unrecognized packed spatial store"));
        }
        options.leaf_records = writer.query_row(
            "SELECT leaf_records FROM manifest WHERE singleton=1",
            [],
            |r| sql_usize(r, 0),
        )?;
        options.validate()?;
        writer.pragma_update(None, "journal_mode", "WAL")?;
        let mut readers = Vec::with_capacity(options.readers);
        for _ in 0..options.readers {
            let (connection, reservation) = open_connection(&path, options, budget, false)?;
            readers.push(Reader {
                connection,
                _reservation: reservation,
            });
        }
        // Keep small caches unified; large caches spread unrelated insertions
        // over independent maps. Capacities sum exactly to cached_pages.
        let partitions = (options.cached_pages / 64)
            .clamp(1, 16)
            .min(options.readers * 2);
        let mut cache = Vec::with_capacity(partitions);
        for partition in 0..partitions {
            let capacity = options.cached_pages / partitions
                + usize::from(partition < options.cached_pages % partitions);

            let bytes = capacity
                .checked_mul(
                    (std::mem::size_of::<(Key, Entry)>() + 32) * 2
                        + std::mem::size_of::<Key>()
                        + std::mem::size_of::<(u64, u64)>()
                        + (std::mem::size_of::<(u64, Key)>() + 32) * 2,
                )
                .and_then(|bytes| bytes.checked_add(512))
                .ok_or(Error::SizeOverflow)?;
            let reservation = budget.reserve(bytes as u64)?;
            let mut entries = HashMap::new();
            entries
                .try_reserve(capacity)
                .map_err(|_| invalid("packed cache allocation"))?;
            let mut payload_heads = HashMap::new();
            payload_heads
                .try_reserve(capacity)
                .map_err(|_| invalid("packed payload cache allocation"))?;
            let mut queue = VecDeque::new();
            queue
                .try_reserve_exact(capacity)
                .map_err(|_| invalid("packed cache clock allocation"))?;
            let mut versions = Vec::new();
            versions
                .try_reserve_exact(capacity)
                .map_err(|_| invalid("packed version index allocation"))?;
            cache.push(RwLock::new(Cache {
                versions,
                entries,
                payload_heads,
                generation: None,
                queue,
                capacity,
                _reservation: reservation,
            }));
        }
        Ok(Self {
            inner: Arc::new(Inner {
                path,
                options,
                compact_values: version == 3,
                budget: budget.clone(),
                writer: Mutex::new(writer),
                publication: Mutex::new(PublicationScratch {
                    changed: Vec::new(),
                    reservation: budget.reserve(0)?,
                    updates: Vec::new(),
                    update_reservation: budget.reserve(0)?,
                    warmed: Vec::new(),
                    warm_reservation: budget.reserve(0)?,
                }),
                readers: readers::Pool::new(readers, budget)?,
                cache,
                geometry: Mutex::new(Weak::new()),
                counters: Counters::default(),
                uncancelled: Cancellation::default(),
                _writer_cache: writer_reservation,
            }),
        })
    }
    pub fn snapshot(&self) -> Result<Snapshot> {
        let c = lock(&self.inner.writer)?;
        let snapshot = self.snapshot_at(&c)?;
        self.inner.observe_generation(snapshot.generation())?;
        Ok(snapshot)
    }
    fn snapshot_at(&self, c: &Connection) -> Result<Snapshot> {
        let reservation = self.inner.reserve_work(
            (MAX_ROOTS * std::mem::size_of::<Node>() + std::mem::size_of::<Manifest>() + 64) as u64,
        )?;
        let (generation, visible, count, roots) = updates::manifest(c)?;
        let mut cached = lock(&self.inner.geometry)?;
        let geometry = if let Some(geometry) = cached.upgrade().filter(|g| g.matches(&roots)) {
            geometry
        } else {
            let geometry = Arc::new(self.load_geometry(c, roots)?);
            // A weak reference neither pins the directory nor its budget charge.
            // Uncommitted geometry cannot escape: failed transactions drop their
            // candidate snapshot, and a later root set must match before reuse.
            *cached = Arc::downgrade(&geometry);
            geometry
        };
        Ok(Snapshot {
            inner: self.inner.clone(),
            manifest: Arc::new(Manifest {
                generation,
                visible,
                count,
                geometry,
                _reservation: reservation,
            }),
        })
    }
    fn load_geometry(&self, c: &Connection, roots: Vec<Node>) -> Result<Geometry> {
        let reservation = self.inner.reserve_work(
            (MAX_ROOTS * (std::mem::size_of::<Node>() + 72)
                + std::mem::size_of::<Geometry>()
                + 64
                + self.inner.options.directory_pages
                    * (FANOUT * (std::mem::size_of::<Node>() * 2 + 72) + 256)) as u64,
        )?;
        let mut directory = HashMap::new();
        directory
            .try_reserve(self.inner.options.directory_pages)
            .map_err(|_| invalid("packed directory allocation"))?;
        let mut queue: VecDeque<_> = roots.iter().copied().collect();
        let mut s = c.prepare("SELECT CASE WHEN length(body)<=?2 THEN body ELSE NULL END,crc FROM geometry WHERE id=?1 AND kind=1")?;
        while directory.len() < self.inner.options.directory_pages {
            let Some(node) = queue.pop_front() else { break };
            let row: Option<(Vec<u8>, u32)> = s
                .query_row(params![sql_id(node.id)?, (8 + FANOUT * 72) as i64], |r| {
                    Ok((r.get(0)?, r.get(1)?))
                })
                .optional()?;
            if let Some((body, crc)) = row {
                if crc32fast::hash(&body) != crc {
                    return Err(invalid("packed directory integrity"));
                }
                let page = decode_page(1, &body, self.inner.options.leaf_records)?;
                let Page::Branch(children) = &page else {
                    unreachable!()
                };
                if children.iter().any(|child| child.id >= node.id) {
                    return Err(invalid("packed directory cycle"));
                }
                queue.extend(children.iter().copied());
                directory.insert(node.id, page);
            }
        }

        Ok(Geometry {
            roots,
            directory,
            _reservation: reservation,
        })
    }
    /// Admit reusable query scratch, evicting unpinned cache pages if needed.
    /// Retain the workspace across queries to avoid hot-path allocations.
    pub fn nearest_workspace(&self, query: Neighborhood) -> Result<NearestWorkspace> {
        self.inner.nearest_workspace(query)
    }
    pub fn path(&self) -> &Path {
        &self.inner.path
    }
    pub fn statistics(&self) -> Statistics {
        self.inner.statistics()
    }
    /// Evict cache entries. Active queries retain pinned pages and their charges.
    pub fn clear_cache(&self) -> Result<()> {
        for partition in &self.inner.cache {
            let started = Stamp::now();
            let mut cache = partition
                .write()
                .map_err(|_| invalid("packed cache lock poisoned"))?;
            started.record(Metric::CacheWriteLock);
            cache.entries.clear();
            cache.versions.clear();
            cache.payload_heads.clear();
            cache.queue.clear();
        }
        Ok(())
    }
}
impl Geometry {
    fn matches(&self, roots: &[Node]) -> bool {
        self.roots.len() == roots.len()
            && self.roots.iter().zip(roots).all(|(a, b)| {
                a.id == b.id
                    && a.first_rank == b.first_rank
                    && a.records == b.records
                    && a.bounds.lower.map(f64::to_bits) == b.bounds.lower.map(f64::to_bits)
                    && a.bounds.upper.map(f64::to_bits) == b.bounds.upper.map(f64::to_bits)
            })
    }
}
impl Inner {
    fn partition(&self, page: u64) -> usize {
        // All versions of a page use the same partition, including its head.
        (page ^ (page >> 32)) as usize % self.cache.len()
    }
    fn observe_generation(&self, generation: u64) -> Result<()> {
        for partition in &self.cache {
            let started = Stamp::now();
            let mut cache = partition
                .write()
                .map_err(|_| invalid("packed cache lock poisoned"))?;
            started.record(Metric::CacheWriteLock);
            if cache.generation != Some(generation) {
                if let Some(previous) = cache.generation {
                    for head in cache.payload_heads.values() {
                        if let Some(entry) = cache.entries.get(head) {
                            entry.valid_through.fetch_max(previous, Ordering::Relaxed);
                        }
                    }
                }
                cache.payload_heads.clear();
                cache.generation = Some(generation);
            }
        }
        Ok(())
    }
    fn published(&self, previous: u64, generation: u64, changed: &[u64]) -> Result<()> {
        for (index, partition) in self.cache.iter().enumerate() {
            let started = Stamp::now();
            let mut cache = partition
                .write()
                .map_err(|_| invalid("packed cache lock poisoned"))?;
            started.record(Metric::CacheWriteLock);
            if cache.generation == Some(previous) {
                for page in changed
                    .iter()
                    .filter(|&&page| self.partition(page) == index)
                {
                    if let Some(key) = cache.payload_heads.remove(page) {
                        if let Some(entry) = cache.entries.get(&key) {
                            entry.valid_through.fetch_max(previous, Ordering::Relaxed);
                        }
                    }
                }
            } else {
                cache.payload_heads.clear();
            }
            cache.generation = Some(generation);
        }
        Ok(())
    }
    fn install_published(&self, warmed: &mut Vec<(Key, Arc<Cached>)>) -> Result<()> {
        for (key, value) in warmed.drain(..) {
            let started = Stamp::now();
            let mut cache = self.cache[self.partition(key.page)]
                .write()
                .map_err(|_| invalid("packed cache lock poisoned"))?;
            started.record(Metric::CacheWriteLock);
            if !cache.entries.contains_key(&key) {
                cache.insert(key, value, key.generation);
            }
        }
        Ok(())
    }
    fn evict_for_admission(&self) -> Result<bool> {
        for partition in &self.cache {
            if partition
                .write()
                .map_err(|_| invalid("packed cache lock poisoned"))?
                .evict(true)
            {
                return Ok(true);
            }
        }
        Ok(false)
    }
    // Useful cached data yields to admitted work. Pinned pages and externally
    // retained snapshots keep their reservations and cannot be evicted here.
    fn admit<T>(&self, mut action: impl FnMut() -> Result<T>) -> Result<T> {
        loop {
            match action() {
                Ok(value) => return Ok(value),
                Err(e @ Error::MemoryBudget { requested, .. }) => {
                    if requested > self.budget.limit() || !self.evict_for_admission()? {
                        return Err(e);
                    }
                }
                Err(e) => return Err(e),
            }
        }
    }
    fn reserve_work(&self, bytes: u64) -> Result<Reservation> {
        self.admit(|| self.budget.reserve(bytes))
    }
    fn resize_work(&self, reservation: &mut Reservation, bytes: u64) -> Result<()> {
        if bytes > self.budget.limit() {
            return reservation.resize(bytes);
        }
        self.admit(|| reservation.resize(bytes))
    }
    fn nearest_workspace(&self, query: Neighborhood) -> Result<NearestWorkspace> {
        self.admit(|| NearestWorkspace::new(query, &self.budget))
    }

    fn statistics(&self) -> Statistics {
        Statistics {
            cache_hits: self.counters.hits.load(Ordering::Relaxed),
            page_reads: self.counters.reads.load(Ordering::Relaxed),
            bytes_read: self.counters.bytes.load(Ordering::Relaxed),
            geometry_groups: self.counters.groups.load(Ordering::Relaxed),
            observations_tested: self.counters.observations.load(Ordering::Relaxed),
        }
    }
    fn page_charge(&self, payload: bool) -> u64 {
        // Covers serialized input plus decoded arrays at their peak overlap,
        // including a conservative allowance for Arc and Vec structures.
        (self.options.leaf_records
            * (if payload {
                RECORD_BYTES + 8 + std::mem::size_of::<Observation>()
            } else {
                128
            })
            + FANOUT * std::mem::size_of::<Node>()
            + 2048) as u64
    }
    fn value_charge(&self) -> u64 {
        // Later cached versions share an already charged base and allocate only
        // one complete values vector, with allowance for the Vec/Arc structures.
        (self.options.leaf_records * std::mem::size_of::<f64>() + 256) as u64
    }
    fn page(
        &self,
        page: u64,
        generation: u64,
        payload: bool,
        cancel: &Cancellation,
    ) -> Result<Arc<Cached>> {
        if crate::execution::paging::cache_only() {
            return Err(Error::WouldBlock);
        }
        let key = Key {
            page,
            generation: if payload { generation } else { 0 },
            payload,
        };
        let _load_span = Span::new(Metric::PageLoad);
        // Claim a page load, not a fixed connection shard. Duplicate misses
        // recheck the cache after the owner completes. No cache lock spans I/O.
        let acquire = Stamp::now();
        let acquired = self.readers.acquire(key, cancel, || {
            let cache = self.cache[self.partition(page)]
                .read()
                .map_err(|_| invalid("packed cache lock poisoned"))?;
            Ok(cache.get(key).map(|entry| {
                entry.mark_recent();
                entry.value.clone()
            }))
        })?;
        acquire.record(Metric::PageAcquire);
        let mut load = match acquired {
            readers::Acquired::Cached(value) => return Ok(value),
            readers::Acquired::Load(load) => load,
        };
        let reserve = Stamp::now();
        let reservation = loop {
            match self.budget.reserve(self.page_charge(payload)) {
                Ok(r) => break r,
                Err(error) => {
                    let evicted = self.evict_for_admission()?;
                    if !evicted {
                        return Err(error);
                    }
                }
            }
        };
        reserve.record(Metric::PageReserve);
        let maximum = self.options.leaf_records * RECORD_BYTES + FANOUT * 72 + 32;
        let c = load.connection();
        let (decoded, stored_generation, bytes, blobs) = if payload {
            let sql = Stamp::now();
            let encoded = payload::fetch(
                c,
                page,
                generation,
                self.options.leaf_records,
                self.compact_values,
            )?;
            sql.record(Metric::PageSqlPayload);
            load.release_reader();
            cancel.check()?;
            let bytes = encoded.bytes();
            let blobs = encoded.blobs();
            let generation = encoded.generation;
            let checksum = Stamp::now();
            let body = encoded.materialize(self.options.leaf_records)?;
            checksum.record(Metric::PageChecksum);
            let decode = Stamp::now();
            let decoded = Cached::payload(&body, self.options.leaf_records, reservation)?;
            decode.record(Metric::PageDecodePayload);
            (decoded, generation, bytes, blobs)
        } else {
            let sql = Stamp::now();
            let (kind, body, crc): (u8, Vec<u8>, u32) = c.prepare_cached("SELECT kind,CASE WHEN length(body)<=?2 THEN body ELSE NULL END,crc FROM geometry WHERE id=?1")?
                .query_row(params![sql_id(page)?, maximum as i64], |r| {
                    let _span = Span::new(Metric::PageBlobCopy);
                    Ok((r.get(0)?, r.get(1)?, r.get(2)?))
                })?;
            sql.record(Metric::PageSqlGeometry);
            load.release_reader();
            cancel.check()?;
            let checksum = Stamp::now();
            if crc32fast::hash(&body) != crc {
                return Err(invalid("packed page checksum mismatch"));
            }
            checksum.record(Metric::PageChecksum);
            let decode = Stamp::now();
            let decoded = decode_page(kind, &body, self.options.leaf_records)?;
            decode.record(Metric::PageDecodeGeometry);
            (
                Cached {
                    page: decoded,
                    _reservation: reservation,
                },
                0,
                body.len(),
                1,
            )
        };
        self.counters.reads.fetch_add(blobs, Ordering::Relaxed);
        self.counters
            .bytes
            .fetch_add(bytes as u64, Ordering::Relaxed);
        let value = Arc::new(decoded);
        let key = Key {
            generation: stored_generation,
            ..key
        };
        let cache_lock = Stamp::now();
        let mut cache = self.cache[self.partition(page)]
            .write()
            .map_err(|_| invalid("packed cache lock poisoned"))?;
        cache_lock.record(Metric::PageCacheLock);
        let _insert = Span::new(Metric::PageCacheInsert);
        if let Some(existing) = cache.entries.get(&key) {
            existing
                .valid_through
                .fetch_max(generation, Ordering::Relaxed);
            return Ok(existing.value.clone());
        }
        cache.insert(key, value.clone(), generation);

        Ok(value)
    }
}
impl Snapshot {
    /// Admit reusable query scratch against this snapshot's shared store budget.
    pub fn nearest_workspace(&self, query: Neighborhood) -> Result<NearestWorkspace> {
        self.inner.nearest_workspace(query)
    }
    pub fn generation(&self) -> u64 {
        self.manifest.generation
    }
    pub fn completed_group(&self) -> u64 {
        self.manifest.visible
    }
    pub fn len(&self) -> u64 {
        self.manifest.count
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn statistics(&self) -> Statistics {
        self.inner.statistics()
    }
}

impl Cache {
    fn get(&self, key: Key) -> Option<&Entry> {
        if !key.payload {
            return self.entries.get(&key);
        }
        if let Some(head) = self.payload_heads.get(&key.page) {
            if head.generation <= key.generation
                && self
                    .generation
                    .is_some_and(|latest| key.generation <= latest)
            {
                return self.entries.get(head);
            }
        }
        let end = self
            .versions
            .partition_point(|version| *version <= (key.page, key.generation));
        let &(page, generation) = self.versions.get(end.checked_sub(1)?)?;
        if page != key.page {
            return None;
        }
        self.entries
            .get(&Key {
                page,
                generation,
                payload: true,
            })
            .filter(|entry| key.generation <= entry.valid_through.load(Ordering::Relaxed))
    }
    fn insert(&mut self, key: Key, value: Arc<Cached>, through: u64) {
        if self.entries.len() == self.capacity {
            self.evict(false);
        }
        if key.payload {
            let position = self
                .versions
                .partition_point(|version| *version < (key.page, key.generation));
            self.versions.insert(position, (key.page, key.generation));
            if self.generation == Some(through) {
                self.payload_heads.insert(key.page, key);
            }
        }
        self.queue.push_back(key);
        self.entries.insert(
            key,
            Entry {
                value,
                recent: AtomicBool::new(true),
                valid_through: AtomicU64::new(through),
            },
        );
    }
    // Clock replacement costs amortized O(1), instead of scanning the complete
    // cache on each miss. Pinned entries may be removed for slot pressure, but
    // only unpinned entries actually release memory for budget pressure.
    fn evict(&mut self, unpinned: bool) -> bool {
        for _ in 0..self.queue.len().saturating_mul(2) {
            let Some(key) = self.queue.pop_front() else {
                return false;
            };
            let Some(entry) = self.entries.get(&key) else {
                continue;
            };
            if (unpinned && Arc::strong_count(&entry.value) != 1)
                || entry.recent.swap(false, Ordering::Relaxed)
            {
                self.queue.push_back(key);
            } else {
                self.entries.remove(&key);
                if key.payload {
                    if let Ok(position) = self.versions.binary_search(&(key.page, key.generation)) {
                        self.versions.remove(position);
                    }
                }
                if self.payload_heads.get(&key.page) == Some(&key) {
                    self.payload_heads.remove(&key.page);
                }
                return true;
            }
        }
        false
    }
}

struct PageView<'a> {
    inner: &'a Inner,
    hits: u64,
}
impl<'a> PageView<'a> {
    fn new(inner: &'a Inner) -> Self {
        Self { inner, hits: 0 }
    }
    fn page(
        &mut self,
        page: u64,
        generation: u64,
        payload: bool,
        cancel: &Cancellation,
    ) -> Result<Arc<Cached>> {
        let key = Key {
            page,
            generation: if payload { generation } else { 0 },
            payload,
        };
        {
            let started = Stamp::now();
            let cache = self.inner.cache[self.inner.partition(page)]
                .read()
                .map_err(|_| invalid("packed cache lock poisoned"))?;
            started.record(Metric::CacheReadLock);
            if let Some(entry) = cache.get(key) {
                entry.mark_recent();
                self.hits += 1;
                return Ok(entry.value.clone());
            }
        }
        // Only the immutable page Arc survives lookup. Traversal, SIMD distance
        // evaluation, caller callbacks and waits never hold a cache lock.
        self.inner.page(page, generation, payload, cancel)
    }
}
impl Drop for PageView<'_> {
    fn drop(&mut self) {
        self.inner
            .counters
            .hits
            .fetch_add(self.hits, Ordering::Relaxed);
    }
}
