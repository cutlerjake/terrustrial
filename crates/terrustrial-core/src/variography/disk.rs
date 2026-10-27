//! Streamed wide sample files with exact disk-indexed irregular variograms.
//!
//! SQLite owns the disk R-tree and bounded page caches. Payloads preserve f64
//! bits, including missingness and signed zero. Queries keep fixed 256-row buffers
//! and share the resident engine's statistics/workspace. No full source or pair
//! list is loaded. Very dense searches and extreme-coordinate fallbacks can
//! still take quadratic work. This does not impose an operating-system RSS cap.
use super::{
    accumulation::half_increment_product,
    indexed::{buffer, validate_request, Accumulator, Variogram, Workspace, ROWS_PER_BLOCK},
    Direction,
};
use crate::{
    data::Point3,
    error::{checked_len, invalid},
    execution::{Cancellation, MemoryBudget, Reservation},
    storage::sqlite::{configured_connection, indexable},
    Error, Result,
};
use rayon::prelude::*;
use rusqlite::{params, Connection, OpenFlags};
use std::path::{Path, PathBuf};

const APPLICATION_ID: i64 = 0x54525647;
const VERSION: i64 = 1;
const BLOCK_QUERY: &str =
    "SELECT ordinal,payload FROM samples WHERE ordinal>=?1 AND ordinal<?2 ORDER BY ordinal";
const INDEX_QUERY: &str = "SELECT p.ordinal,p.payload FROM bounds b JOIN samples p ON p.ordinal=b.ordinal WHERE b.max_x>=?1 AND b.min_x<=?2 AND b.max_y>=?3 AND b.min_y<=?4 AND b.max_z>=?5 AND b.min_z<=?6 AND p.ordinal>?7";
const UNINDEXED_QUERY: &str =
    "SELECT ordinal,payload FROM samples WHERE indexed=0 AND ordinal>?1 ORDER BY ordinal";

/// Fill the entire fixed-width attribute row and return its stable ID/position.
/// Missing values are None. IDs need not be contiguous or sorted, but must be
/// unique. Coincident coordinates with distinct IDs are distinct samples.
/// Source-owned buffers and I/O are outside the importer's logical budget.
pub trait Rows {
    fn next(&mut self, values: &mut [Option<f64>]) -> Result<Option<(u64, Point3)>>;
}
impl<F> Rows for F
where
    F: FnMut(&mut [Option<f64>]) -> Result<Option<(u64, Point3)>>,
{
    fn next(&mut self, values: &mut [Option<f64>]) -> Result<Option<(u64, Point3)>> {
        self(values)
    }
}

pub struct BuildOptions<'a> {
    pub variables: usize,
    pub cache_bytes: u64,
    /// SQLite main database pages only, rounded down to 4096 bytes. Journals,
    /// filesystem metadata, source data and allocator overhead are excluded.
    pub maximum_database_bytes: u64,
    pub budget: &'a MemoryBudget,
    pub cancellation: &'a Cancellation,
}
#[derive(Debug)]
pub struct BuildReport {
    pub samples: u64,
    pub variables: usize,
    pub unindexed_samples: u64,
    pub database_bytes: u64,
}

/// Build a new file from a single streaming pass. Scratch lives beside the
/// destination; errors/cancellation leave no destination and cleanup scratch on
/// normal return. Existing destinations are never replaced. The final database
/// is synchronized before no-clobber persistence. Directory-fsync/power-loss
/// durability and resumable construction are not provided by this operation.
/// SQLite supports at most i64::MAX rows here; record IDs cover all u64 values.
pub fn build(
    rows: &mut impl Rows,
    destination: &Path,
    options: BuildOptions<'_>,
) -> Result<BuildReport> {
    options.cancellation.check()?;
    let bytes = row_bytes(options.variables)?;
    let owned = checked_len(&[options.variables, std::mem::size_of::<Option<f64>>()])?
        .checked_add(bytes)
        .ok_or(Error::SizeOverflow)?;
    let _buffers = options
        .budget
        .reserve(u64::try_from(owned).map_err(|_| Error::SizeOverflow)?)?;
    let mut values = buffer(options.variables, || None)?;
    let mut payload = buffer(bytes, || 0_u8)?;
    if destination.file_name().is_none() || destination.exists() {
        return Err(invalid("variogram destination must be a new file"));
    }
    let pages = (options.maximum_database_bytes / 4096).min(u32::MAX as u64 - 1);
    if pages == 0 {
        return Err(Error::DiskBudget {
            requested: 4096,
            available: options.maximum_database_bytes,
        });
    }
    let parent = destination
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let staging = tempfile::Builder::new()
        .prefix("terrustrial-variogram-")
        .tempdir_in(parent)?;
    let file = tempfile::NamedTempFile::new_in(staging.path())?;
    let (mut connection, _, _cache) = configured_connection(
        file.path(),
        options.cache_bytes,
        options.budget,
        OpenFlags::SQLITE_OPEN_READ_WRITE,
    )?;
    connection.pragma_update(None, "page_size", 4096)?;
    connection.pragma_update(None, "max_page_count", pages as i64)?;
    connection.execute_batch("PRAGMA journal_mode=DELETE;
        CREATE TABLE metadata(singleton INTEGER PRIMARY KEY CHECK(singleton=1),variables INTEGER NOT NULL,samples INTEGER NOT NULL);
        CREATE TABLE samples(ordinal INTEGER PRIMARY KEY,id INTEGER NOT NULL UNIQUE,x REAL NOT NULL,y REAL NOT NULL,z REAL NOT NULL,indexed INTEGER NOT NULL CHECK(indexed IN (0,1)),payload BLOB NOT NULL);
        CREATE INDEX unindexed_samples ON samples(indexed) WHERE indexed=0;
        CREATE VIRTUAL TABLE bounds USING rtree(ordinal,min_x,max_x,min_y,max_y,min_z,max_z);")?;
    connection.pragma_update(None, "application_id", APPLICATION_ID)?;
    connection.pragma_update(None, "user_version", VERSION)?;
    let mut count = 0_u64;
    let mut unindexed = 0_u64;
    {
        let transaction = connection.transaction()?;
        let mut insert = transaction.prepare("INSERT INTO samples VALUES(?1,?2,?3,?4,?5,?6,?7)")?;
        let mut bounds = transaction.prepare("INSERT INTO bounds VALUES(?1,?2,?2,?3,?3,?4,?4)")?;
        loop {
            options.cancellation.check()?;
            values.fill(None);
            let Some((id, point)) = rows.next(&mut values)? else {
                break;
            };
            if count >= i64::MAX as u64 {
                return Err(Error::SizeOverflow);
            }
            if values.iter().flatten().any(|v| !v.is_finite()) {
                return Err(invalid(format!(
                    "nonfinite variogram attribute in sample {id}"
                )));
            }
            let p = point.coordinates();
            for (i, value) in p.iter().enumerate() {
                payload[i * 8..i * 8 + 8].copy_from_slice(&value.to_bits().to_le_bytes());
            }
            for (chunk, value) in payload[24..].as_chunks_mut::<9>().0.iter_mut().zip(&values) {
                chunk[0] = u8::from(value.is_some());
                chunk[1..].copy_from_slice(&value.unwrap_or(0.).to_bits().to_le_bytes());
            }
            let indexed = indexable(p);
            insert.execute(params![
                count as i64,
                id as i64,
                p[0],
                p[1],
                p[2],
                indexed,
                &payload
            ])?;
            if indexed {
                bounds.execute(params![count as i64, p[0], p[1], p[2]])?;
            } else {
                unindexed += 1;
            }
            count += 1;
        }
        options.cancellation.check()?;
        drop(bounds);
        drop(insert);
        transaction.execute(
            "INSERT INTO metadata VALUES(1,?1,?2)",
            params![options.variables as i64, count as i64],
        )?;
        transaction.commit()?;
    }
    let database_bytes = file.as_file().metadata()?.len();
    if database_bytes > options.maximum_database_bytes {
        return Err(Error::DiskBudget {
            requested: database_bytes,
            available: options.maximum_database_bytes,
        });
    }
    drop(connection);
    file.as_file().sync_all()?;
    options.cancellation.check()?;
    file.persist_noclobber(destination)
        .map_err(|e| Error::Io(e.error))?;
    Ok(BuildReport {
        samples: count,
        variables: options.variables,
        unindexed_samples: unindexed,
        database_bytes,
    })
}

fn row_bytes(variables: usize) -> Result<usize> {
    if variables == 0 {
        return Err(invalid("positive variogram variable count required"));
    }
    let bytes = variables
        .checked_mul(9)
        .and_then(|x| x.checked_add(24))
        .ok_or(Error::SizeOverflow)?;
    if bytes > i32::MAX as usize {
        return Err(invalid("variogram row exceeds SQLite blob size range"));
    }
    Ok(bytes)
}

#[derive(Clone, Copy)]
struct Pair {
    ordinal: i64,
    point: Point3,
    values: [Option<f64>; 2],
}
fn decode(ordinal: i64, payload: &[u8], variables: usize, selected: [usize; 2]) -> Result<Pair> {
    if ordinal < 0 || payload.len() != row_bytes(variables)? {
        return Err(invalid("variogram ordinal or payload length"));
    }
    let point = Point3::new(std::array::from_fn(|i| {
        f64::from_bits(u64::from_le_bytes(
            payload[i * 8..i * 8 + 8].try_into().unwrap(),
        ))
    }))?;
    let mut values = [None; 2];
    for (slot, variable) in selected.into_iter().enumerate() {
        values[slot] = attribute(&payload[24 + variable * 9..24 + (variable + 1) * 9])?;
    }
    Ok(Pair {
        ordinal,
        point,
        values,
    })
}
fn attribute(bytes: &[u8]) -> Result<Option<f64>> {
    let bits = u64::from_le_bytes(bytes[1..].try_into().unwrap());
    match bytes[0] {
        0 if bits == 0 => Ok(None),
        1 => {
            let value = f64::from_bits(bits);
            if value.is_finite() {
                Ok(Some(value))
            } else {
                Err(invalid("nonfinite variogram payload value"))
            }
        }
        _ => Err(invalid("invalid variogram missing-value encoding")),
    }
}

struct Reader {
    connection: Connection,
    rows: Vec<Pair>,
    _cache: Reservation,
    _rows: Reservation,
}

/// An immutable SQLite snapshot shared by bounded independent readers.
/// Open performs a streaming integrity check of all attributes and index coverage.
/// Each reader reserves its configured cache, fixed row-buffer payload and one
/// serialized-row allowance. SQLite
/// metadata/query heap, allocator overhead and OS caches are not budgeted. Multiple
/// readers temporarily require a write-capable coordinator to pin one snapshot,
/// as for SqliteReadPool; no rows are changed. External file replacement and I/O
/// failure are outside transaction protection. Drop releases all snapshots.
pub struct SampleFile {
    path: PathBuf,
    readers: Vec<Reader>,
    variables: usize,
    samples: u64,
    unindexed_samples: u64,
    _slots: Reservation,
}
impl SampleFile {
    pub fn open(
        path: &Path,
        readers: usize,
        cache_bytes_per_reader: u64,
        budget: &MemoryBudget,
        cancel: &Cancellation,
    ) -> Result<Self> {
        cancel.check()?;
        if readers == 0 {
            return Err(invalid("positive variogram reader count required"));
        }
        let slots = budget.reserve(
            u64::try_from(checked_len(&[readers, std::mem::size_of::<Reader>()])?)
                .map_err(|_| Error::SizeOverflow)?,
        )?;
        let mut connections = Vec::new();
        connections
            .try_reserve_exact(readers)
            .map_err(|_| invalid("unable to allocate variogram reader slots"))?;
        let coordinator = if readers > 1 {
            Some(configured_connection(
                path,
                4096,
                budget,
                OpenFlags::SQLITE_OPEN_READ_WRITE,
            )?)
        } else {
            None
        };
        if let Some((connection, _, _)) = &coordinator {
            connection.execute_batch("BEGIN IMMEDIATE")?;
        }
        let mut schema = None;
        let mut resolved = PathBuf::new();
        for _ in 0..readers {
            cancel.check()?;
            let (connection, path, cache) = configured_connection(
                path,
                cache_bytes_per_reader,
                budget,
                OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_PRIVATE_CACHE,
            )?;
            connection.execute_batch("BEGIN")?;
            let app: i64 = connection.pragma_query_value(None, "application_id", |r| r.get(0))?;
            let version: i64 = connection.pragma_query_value(None, "user_version", |r| r.get(0))?;
            if app != APPLICATION_ID || version != VERSION {
                return Err(invalid("unrecognized variogram sample format"));
            }
            let (variables, samples): (i64, i64) = connection.query_row(
                "SELECT variables,samples FROM metadata WHERE singleton=1",
                [],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )?;
            let variables = usize::try_from(variables)
                .map_err(|_| invalid("invalid variogram variable count"))?;
            row_bytes(variables)?;
            let samples =
                u64::try_from(samples).map_err(|_| invalid("invalid variogram sample count"))?;
            if schema.is_some_and(|old| old != (variables, samples)) {
                return Err(invalid("inconsistent variogram reader snapshots"));
            }
            schema = Some((variables, samples));
            let rows = budget.reserve(
                u64::try_from(
                    checked_len(&[ROWS_PER_BLOCK, std::mem::size_of::<Pair>()])?
                        .checked_add(row_bytes(variables)?)
                        .ok_or(Error::SizeOverflow)?,
                )
                .map_err(|_| Error::SizeOverflow)?,
            )?;
            connections.push(Reader {
                connection,
                rows: buffer(ROWS_PER_BLOCK, || Pair {
                    ordinal: 0,
                    point: Point3::new([0.; 3]).unwrap(),
                    values: [None; 2],
                })?,
                _cache: cache,
                _rows: rows,
            });
            resolved = path;
        }
        if let Some((connection, _, _)) = &coordinator {
            connection.execute_batch("ROLLBACK")?;
        }
        drop(coordinator);
        let (variables, samples) = schema.unwrap();
        let mut file = Self {
            path: resolved,
            readers: connections,
            variables,
            samples,
            unindexed_samples: 0,
            _slots: slots,
        };
        file.unindexed_samples = file.verify(cancel)?;
        Ok(file)
    }
    pub fn path(&self) -> &Path {
        &self.path
    }
    pub fn variables(&self) -> usize {
        self.variables
    }
    pub fn len(&self) -> u64 {
        self.samples
    }
    pub fn is_empty(&self) -> bool {
        self.samples == 0
    }

    fn verify(&self, cancel: &Cancellation) -> Result<u64> {
        let connection = &self.readers[0].connection;
        let mut statement = connection.prepare("SELECT p.ordinal,p.payload,p.x,p.y,p.z,p.indexed,b.min_x,b.max_x,b.min_y,b.max_y,b.min_z,b.max_z FROM samples p LEFT JOIN bounds b ON p.ordinal=b.ordinal ORDER BY p.ordinal")?;
        let mut rows = statement.query([])?;
        let mut count = 0_u64;
        let mut indexed_count = 0_i64;
        while let Some(row) = rows.next()? {
            cancel.check()?;
            let ordinal: i64 = row.get(0)?;
            let payload = payload(row)?;
            let pair = decode(ordinal, payload, self.variables, [0, 0])?;
            if ordinal as u64 != count {
                return Err(invalid("variogram ordinal sequence has gaps"));
            }
            for attr in payload[24..].as_chunks::<9>().0 {
                attribute(attr)?;
            }
            let p = pair.point.coordinates();
            let indexed: i64 = row.get(5)?;
            if indexed != i64::from(indexable(p)) {
                return Err(invalid("variogram spatial-index flag mismatch"));
            }
            for (axis, value) in p.iter().enumerate() {
                if row.get::<_, f64>(2 + axis)? != *value {
                    return Err(invalid("variogram coordinate/payload mismatch"));
                }
                let low: Option<f64> = row.get(6 + 2 * axis)?;
                let high: Option<f64> = row.get(7 + 2 * axis)?;
                if indexed == 1 {
                    if !low.is_some_and(|low| low.is_finite() && low <= *value)
                        || !high.is_some_and(|high| high.is_finite() && high >= *value)
                    {
                        return Err(invalid("variogram index bounds omit a sample"));
                    }
                } else if low.is_some() || high.is_some() {
                    return Err(invalid("unexpected variogram index entry"));
                }
            }
            indexed_count = indexed_count
                .checked_add(indexed)
                .ok_or(Error::SizeOverflow)?;
            count = count.checked_add(1).ok_or(Error::SizeOverflow)?;
        }
        let bounds: i64 = connection.query_row("SELECT COUNT(*) FROM bounds", [], |r| r.get(0))?;
        if count != self.samples || bounds != indexed_count {
            return Err(invalid("variogram sample/index count mismatch"));
        }
        cancel.check()?;
        Ok(count - indexed_count as u64)
    }

    /// Uses fixed input-ordinal blocks and ordered reduction. Readers and
    /// workspace windows may change without changing bits for the same file,
    /// backend and request. Pair order need not match the resident R-tree engine.
    /// Candidate counts include ordered pairs inside each source's outward f64
    /// radius box, before target missingness, exact distance and direction tests.
    /// All reader snapshots stay pinned for the full operation and later calls.
    pub fn experimental<'w>(
        &mut self,
        variables: [usize; 2],
        edges: &[f64],
        direction: Option<&Direction>,
        workspace: &'w mut Workspace,
        cancel: &Cancellation,
    ) -> Result<Variogram<'w>> {
        cancel.check()?;
        validate_request(variables, self.variables, edges, workspace.bin_capacity)?;
        let capacity = workspace.bin_capacity;
        let request = Request {
            total_variables: self.variables,
            selected: variables,
            edges,
            direction,
            cancel,
            unindexed: self.unindexed_samples != 0,
        };
        let samples = self.samples;
        workspace.reduce(
            samples,
            self.readers.len(),
            edges.len(),
            cancel,
            |start, active, partial, status| {
                let evaluate = |slot: usize,
                                reader: &mut Reader,
                                bins: &mut [Accumulator],
                                status: &mut Result<u128>| {
                    let first = (start + slot as u64) * ROWS_PER_BLOCK as u64;
                    let end = samples.min(first + ROWS_PER_BLOCK as u64);
                    *status = reader.evaluate(first..end, &request, bins);
                };
                if active == 1 {
                    evaluate(
                        0,
                        &mut self.readers[0],
                        &mut partial[..capacity],
                        &mut status[0],
                    );
                } else {
                    self.readers[..active]
                        .par_iter_mut()
                        .zip(partial.par_chunks_mut(capacity))
                        .zip(status.par_iter_mut())
                        .enumerate()
                        .for_each(|(slot, ((reader, bins), status))| {
                            evaluate(slot, reader, bins, status)
                        });
                }
            },
        )
    }
}
struct Request<'a> {
    total_variables: usize,
    selected: [usize; 2],
    edges: &'a [f64],
    direction: Option<&'a Direction>,
    cancel: &'a Cancellation,
    unindexed: bool,
}

impl Reader {
    fn evaluate(
        &mut self,
        range: std::ops::Range<u64>,
        request: &Request<'_>,
        bins: &mut [Accumulator],
    ) -> Result<u128> {
        let (first, end) = (range.start, range.end);
        let Request {
            total_variables,
            selected,
            edges,
            direction,
            cancel,
            unindexed: has_unindexed,
        } = *request;
        if self.connection.is_autocommit() {
            return Err(invalid("variogram reader lost its snapshot"));
        }
        let mut block = self.connection.prepare_cached(BLOCK_QUERY)?;
        let mut rows = block.query(params![first as i64, end as i64])?;
        let mut n = 0;
        while let Some(row) = rows.next()? {
            cancel.check()?;
            let pair = decode(row.get(0)?, payload(row)?, total_variables, selected)?;
            if n >= self.rows.len() || pair.ordinal as u64 != first + n as u64 {
                return Err(invalid("invalid variogram row block"));
            }
            self.rows[n] = pair;
            n += 1;
        }
        if n as u64 != end - first {
            return Err(invalid("incomplete variogram row block"));
        }
        drop(rows);
        drop(block);
        // Keep one fixed source buffer. Missing source attributes cannot contribute
        // any pairs and must not enlarge a batched search envelope.
        let mut eligible = 0;
        for i in 0..n {
            if self.rows[i].values.iter().all(Option::is_some) {
                self.rows[eligible] = self.rows[i];
                eligible += 1;
            }
        }
        if eligible == 0 {
            return Ok(0);
        }
        let sources = &mut self.rows[..eligible];
        let radius = edges[edges.len() - 1];
        let outward = radius.next_up();
        let mut low = sources[0].point.coordinates();
        let mut high = low;
        let mut first_ordinal = sources[0].ordinal;
        for a in sources.iter() {
            first_ordinal = first_ordinal.min(a.ordinal);
            for (axis, coordinate) in a.point.coordinates().into_iter().enumerate() {
                low[axis] = low[axis].min(coordinate);
                high[axis] = high[axis].max(coordinate);
            }
        }
        let half_span: [f64; 3] = std::array::from_fn(|axis| high[axis] * 0.5 - low[axis] * 0.5);
        // Estimate the union-box volume relative to one radius box without
        // multiplying physical lengths. Scattered blocks retain individual
        // queries instead of fetching nearly the entire database for each block.
        // Overflow simply selects this conservative performance fallback.
        let expansion: f64 = half_span.iter().map(|span| 1. + span / radius).product();
        let batch = expansion <= eligible as f64;
        let axis = (0..3)
            .max_by(|&a, &b| half_span[a].total_cmp(&half_span[b]))
            .unwrap();
        if batch {
            sources.sort_unstable_by(|a, b| {
                a.point.coordinates()[axis]
                    .total_cmp(&b.point.coordinates()[axis])
                    .then(a.ordinal.cmp(&b.ordinal))
            });
        }
        let mut spatial = self.connection.prepare_cached(INDEX_QUERY)?;
        let mut unindexed = self.connection.prepare_cached(UNINDEXED_QUERY)?;
        let mut candidates = 0_u128;
        for group in sources.chunks(if batch { eligible } else { 1 }) {
            cancel.check()?;
            let (low, high, minimum) = if batch {
                (low, high, first_ordinal)
            } else {
                let p = group[0].point.coordinates();
                (p, p, group[0].ordinal)
            };
            let low = low.map(|x| (x - outward).next_down());
            let high = high.map(|x| (x + outward).next_up());
            let mut accept = |rows: &mut rusqlite::Rows<'_>| -> Result<()> {
                while let Some(row) = rows.next()? {
                    cancel.check()?;
                    let b = decode(row.get(0)?, payload(row)?, total_variables, selected)?;
                    if b.ordinal <= minimum {
                        return Err(invalid("variogram query violated pair ordering"));
                    }
                    let q = b.point.coordinates();
                    let lower = (q[axis] - outward).next_down();
                    let upper = (q[axis] + outward).next_up();
                    let start = group.partition_point(|a| a.point.coordinates()[axis] < lower);
                    let end = group.partition_point(|a| a.point.coordinates()[axis] <= upper);
                    for a in &group[start..end] {
                        if a.ordinal >= b.ordinal {
                            continue;
                        }
                        let p = a.point.coordinates();
                        if (0..3).any(|axis| {
                            q[axis] < (p[axis] - outward).next_down()
                                || q[axis] > (p[axis] + outward).next_up()
                        }) {
                            continue;
                        }
                        candidates += 1;
                        let [Some(ba), Some(bb)] = b.values else {
                            continue;
                        };
                        let [Some(aa), Some(ab)] = a.values else {
                            unreachable!()
                        };
                        let distance = a.point.distance(b.point);
                        if distance > radius
                            || direction.is_some_and(|d| !d.accepts(a.point, b.point, distance))
                        {
                            continue;
                        }
                        let bin = &mut bins[edges.partition_point(|e| distance > *e)];
                        bin.distance.add(distance)?;
                        bin.gamma.add(half_increment_product(aa, ba, ab, bb)?)?;
                    }
                }
                Ok(())
            };
            accept(&mut spatial.query(params![
                low[0], high[0], low[1], high[1], low[2], high[2], minimum
            ])?)?;
            if has_unindexed {
                accept(&mut unindexed.query([minimum])?)?;
            }
        }
        Ok(candidates)
    }
}

fn payload<'a>(row: &'a rusqlite::Row<'_>) -> Result<&'a [u8]> {
    row.get_ref(1)?
        .as_blob()
        .map_err(|_| invalid("variogram payload must be a blob"))
}
