use super::*;
use std::fs::File;

/// Streaming builder. The destination is published without overwriting an
/// existing file only after all input, pages and the initial manifest commit.
/// Dropping or cancelling before publication removes the temporary database.
pub struct PackedBuilder {
    connection: Connection,
    temporary: tempfile::TempPath,
    destination: PathBuf,
    options: Options,
    budget: MemoryBudget,
    _cache: Reservation,
    _scratch: Reservation,
}
pub(super) fn page_scratch_bytes(options: Options) -> u64 {
    (options.leaf_records * (std::mem::size_of::<(Observation, u64)>() + RECORD_BYTES * 4 + 96)
        + 64 * FANOUT * std::mem::size_of::<Node>()
        + 16384) as u64
}
pub(super) fn scratch_bytes(options: Options) -> u64 {
    page_scratch_bytes(options)
        + ((options.build_window_leaves - 1)
            * options.leaf_records
            * std::mem::size_of::<(Observation, u64)>()) as u64
}

#[cfg(test)]
mod page_policy_tests {
    use super::*;

    #[test]
    fn larger_pages_preserve_byte_limits_and_fail_below_one_page() {
        let dir = tempfile::tempdir().unwrap();
        let budget = MemoryBudget::new(16 << 20);
        for (leaf_records, expected_page, cache_bytes) in [
            (64, 4096u64, 65536),
            (128, 8192, 65536),
            (256, 16384, 65536),
            (256, 4096, 4096),
        ] {
            let options = Options {
                leaf_records,
                sqlite_cache_bytes: cache_bytes,
                max_database_bytes: (8 << 20) + 4095,
                wal_checkpoint_bytes: (1 << 20) + 4096,
                ..Options::default()
            };
            let builder = PackedBuilder::create(
                dir.path().join(format!("leaf-{leaf_records}")),
                options,
                &budget,
            )
            .unwrap();
            let read = |name| {
                builder
                    .connection
                    .pragma_query_value(None, name, |r| r.get::<_, i64>(0))
                    .unwrap()
            };
            assert_eq!(read("page_size") as u64, expected_page);
            assert_eq!(read("cache_size"), -(cache_bytes as i64 / 1024));
            assert_eq!(
                read("max_page_count") as u64,
                options.max_database_bytes / expected_page
            );
            assert_eq!(
                read("wal_autocheckpoint") as u64,
                options.wal_checkpoint_bytes.div_ceil(expected_page)
            );
            assert_eq!(read("synchronous"), 2);
            drop(builder);
            assert_eq!(budget.used(), 0);
        }
        let too_small = Options {
            leaf_records: 256,
            max_database_bytes: 4096,
            ..Options::default()
        };
        assert!(PackedBuilder::create(dir.path().join("too-small"), too_small, &budget).is_err());
        assert_eq!(budget.used(), 0);
    }
}
pub(super) fn prepare_staging(c: &Connection) -> Result<()> {
    c.execute_batch("CREATE TABLE staging(id BLOB PRIMARY KEY CHECK(length(id)=8),x REAL NOT NULL,y REAL NOT NULL,z REAL NOT NULL,rank INTEGER NOT NULL CHECK(rank>=0),body BLOB NOT NULL) WITHOUT ROWID;
        CREATE TABLE ordered(code INTEGER NOT NULL,x REAL NOT NULL,y REAL NOT NULL,z REAL NOT NULL,id BLOB NOT NULL,rank INTEGER NOT NULL,body BLOB NOT NULL,PRIMARY KEY(code,x,y,z,id)) WITHOUT ROWID;")?;
    Ok(())
}
pub(super) fn stage(c: &Connection, observation: Observation, rank: u64) -> Result<()> {
    let p = observation.site().support.center().coordinates();
    c.prepare_cached("INSERT INTO staging VALUES(?1,?2,?3,?4,?5,?6)")?
        .execute(params![
            observation.id().to_be_bytes().as_slice(),
            p[0],
            p[1],
            p[2],
            sql_id(rank)?,
            encode_record(&observation).as_slice()
        ])?;
    Ok(())
}
impl PackedBuilder {
    pub fn create(path: impl AsRef<Path>, options: Options, budget: &MemoryBudget) -> Result<Self> {
        options.validate()?;
        let destination = path.as_ref().to_owned();
        let parent = destination
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        let scratch = budget.reserve(scratch_bytes(options))?;
        let temporary = tempfile::NamedTempFile::new_in(parent)?.into_temp_path();
        let (connection, cache) = open_connection(&temporary, options, budget, true)?;
        // Pack several compact value versions (including record headers/keys)
        // into each page, avoiding overflow storage for small value updates.
        // This affects new files only; existing files retain their page size.
        let mut page_size = ((options.leaf_records * 8 + 64) * 4)
            .next_power_of_two()
            .max(4096);
        // A tiny configured cache must still be able to hold one database page.
        // Retain overflow storage when the byte allowance cannot fit a larger page.
        while page_size as u64 > options.sqlite_cache_bytes {
            page_size /= 2;
        }
        connection.pragma_update(None, "page_size", page_size as i64)?;
        // Cache and disk limits are expressed in bytes. Reapply them after a
        // page-size change so their effective allowances cannot multiply.
        let kib = i32::try_from(options.sqlite_cache_bytes / 1024)
            .map_err(|_| invalid("SQLite cache exceeds supported size"))?;
        connection.pragma_update(None, "cache_size", -kib)?;
        configure_writer_limits(&connection, options)?;
        connection.pragma_update(None, "auto_vacuum", "INCREMENTAL")?;
        // Keep payload bodies in append-ordered rowid pages, with only small
        // keys in the unique lookup index. WITHOUT ROWID interleaves new value
        // versions with large immutable bases in the same tree, making those
        // inserts rebalance and journal substantially more payload pages.
        // Both layouts have the same format-3 columns and read/write contract;
        // existing stores retain their layout without migration.
        connection.execute_batch("BEGIN IMMEDIATE;
            CREATE TABLE geometry(id INTEGER PRIMARY KEY,kind INTEGER NOT NULL,body BLOB NOT NULL,crc INTEGER NOT NULL);
            CREATE TABLE payloads(page INTEGER NOT NULL,generation INTEGER NOT NULL,body BLOB NOT NULL,crc INTEGER NOT NULL,encoding INTEGER NOT NULL DEFAULT 0,PRIMARY KEY(page,generation));
            CREATE TABLE identities(id BLOB PRIMARY KEY,page INTEGER NOT NULL,slot INTEGER NOT NULL,rank INTEGER NOT NULL) WITHOUT ROWID;
            CREATE INDEX identities_rank ON identities(rank,id,page,slot);
            CREATE UNIQUE INDEX identities_page ON identities(page,slot);
            CREATE TABLE manifest(singleton INTEGER PRIMARY KEY CHECK(singleton=1),generation INTEGER NOT NULL,visible INTEGER NOT NULL,eligible INTEGER NOT NULL,roots BLOB NOT NULL,leaf_records INTEGER NOT NULL,roots_crc INTEGER NOT NULL);")?;
        connection.pragma_update(None, "application_id", FORMAT)?;
        connection.pragma_update(None, "user_version", 3)?;
        prepare_staging(&connection)?;
        Ok(Self {
            connection,
            temporary,
            destination,
            options,
            budget: budget.clone(),
            _cache: cache,
            _scratch: scratch,
        })
    }
    /// Rank zero is immediately visible conditioning data. Positive ranks are
    /// predetermined simulation groups, published together with their values.
    pub fn push(
        &mut self,
        observation: Observation,
        available_at: u64,
        cancel: &Cancellation,
    ) -> Result<()> {
        cancel.check()?;
        stage(&self.connection, observation, available_at)
    }
    pub fn finish(self, cancel: &Cancellation) -> Result<PackedStore> {
        self.finish_visible(0, cancel)
    }
    pub(super) fn finish_visible(self, visible: u64, cancel: &Cancellation) -> Result<PackedStore> {
        cancel.check()?;
        let root = build_segment(&self.connection, self.options, 0, cancel)?;
        let roots: Vec<_> = root.into_iter().collect();
        let eligible: u64 = self.connection.query_row(
            "SELECT count(*) FROM identities WHERE rank<=?1",
            [sql_id(visible)?],
            |r| sql_u64(r, 0),
        )?;
        let body = encode_nodes(&roots);
        self.connection.execute(
            "INSERT INTO manifest VALUES(1,0,?1,?2,?3,?4,?5)",
            params![
                sql_id(visible)?,
                sql_id(eligible)?,
                body,
                self.options.leaf_records as i64,
                crc32fast::hash(&body)
            ],
        )?;
        cancel.check()?;
        self.connection
            .execute_batch("COMMIT; PRAGMA incremental_vacuum;")?;
        self.connection
            .close()
            .map_err(|(_, e)| Error::Storage(e))?;
        // Windows requires write access for FlushFileBuffers (sync_all).
        File::options()
            .write(true)
            .open(&self.temporary)?
            .sync_all()?;
        self.temporary
            .persist_noclobber(&self.destination)
            .map_err(|e| Error::Io(e.error))?;
        // Release construction allowances before admitting the runtime cache.
        drop(self._scratch);
        drop(self._cache);
        PackedStore::open(self.destination, self.options, &self.budget)
    }
}

struct Levels {
    levels: [Vec<Node>; 64],
}
impl Levels {
    fn new() -> Self {
        Self {
            levels: std::array::from_fn(|_| Vec::with_capacity(FANOUT)),
        }
    }
    fn push(&mut self, c: &Connection, node: Node, level: usize, next: &mut u64) -> Result<()> {
        if level == self.levels.len() {
            return Err(Error::SizeOverflow);
        }
        self.levels[level].push(node);
        if self.levels[level].len() == FANOUT {
            let parent = write_branch(c, &self.levels[level], next)?;
            self.levels[level].clear();
            self.push(c, parent, level + 1, next)?;
        }
        Ok(())
    }
    fn finish(mut self, c: &Connection, next: &mut u64) -> Result<Option<Node>> {
        for level in 0..self.levels.len() {
            let count: usize = self.levels.iter().map(Vec::len).sum();
            if count == 0 {
                return Ok(None);
            }
            if count == 1 {
                return Ok(self.levels.iter().find_map(|v| v.first()).copied());
            }
            if !self.levels[level].is_empty() {
                let parent = write_branch(c, &self.levels[level], next)?;
                self.levels[level].clear();
                self.push(c, parent, level + 1, next)?;
            }
        }
        Err(Error::SizeOverflow)
    }
}
fn allocate_id(next: &mut u64) -> Result<u64> {
    let out = *next;
    sql_id(out)?;
    *next = out.checked_add(1).ok_or(Error::SizeOverflow)?;
    Ok(out)
}
fn write_branch(c: &Connection, children: &[Node], next: &mut u64) -> Result<Node> {
    let id = allocate_id(next)?;
    let mut bounds = children[0].bounds;
    let mut first_rank = u64::MAX;
    let mut records = 0u64;
    for child in children {
        bounds.include(child.bounds);
        first_rank = first_rank.min(child.first_rank);
        records = records
            .checked_add(child.records)
            .ok_or(Error::SizeOverflow)?;
    }
    let body = encode_nodes(children);
    c.prepare_cached("INSERT INTO geometry VALUES(?1,1,?2,?3)")?
        .execute(params![sql_id(id)?, body, crc32fast::hash(&body)])?;
    Ok(Node {
        id,
        bounds,
        first_rank,
        records,
    })
}
fn write_leaf(
    c: &Connection,
    records: &[(Observation, u64)],
    generation: u64,
    next: &mut u64,
) -> Result<Node> {
    let id = allocate_id(next)?;
    let mut bounds = Bounds::point(records[0].0.site().support.center().coordinates());
    let mut first_rank = u64::MAX;
    let mut payload = Vec::with_capacity(records.len() * RECORD_BYTES);
    for (slot, (observation, rank)) in records.iter().enumerate() {
        bounds.include(Bounds::point(
            observation.site().support.center().coordinates(),
        ));
        first_rank = first_rank.min(*rank);
        payload.extend_from_slice(&encode_record(observation));
        c.prepare_cached("INSERT INTO identities VALUES(?1,?2,?3,?4)")?
            .execute(params![
                observation.id().to_be_bytes().as_slice(),
                sql_id(id)?,
                slot as i64,
                sql_id(*rank)?
            ])?;
    }
    let geometry = encode_leaf(records);
    c.prepare_cached("INSERT INTO geometry VALUES(?1,2,?2,?3)")?
        .execute(params![sql_id(id)?, geometry, crc32fast::hash(&geometry)])?;
    write_payload(c, id, generation, &payload)?;
    Ok(Node {
        id,
        bounds,
        first_rank,
        records: records.len() as u64,
    })
}
// Repartition a bounded Morton-local construction window. Median splits give
// each disk subtree tight spatial bounds without a global in-memory sort.
fn write_cluster(
    c: &Connection,
    records: &mut [(Observation, u64)],
    leaf: usize,
    generation: u64,
    next: &mut u64,
    cancel: &Cancellation,
) -> Result<Node> {
    cancel.check()?;
    if records.len() <= leaf {
        records.sort_unstable_by(|a, b| {
            let pa = a.0.site().support.center().coordinates();
            let pb = b.0.site().support.center().coordinates();
            pa[0]
                .total_cmp(&pb[0])
                .then(pa[1].total_cmp(&pb[1]))
                .then(pa[2].total_cmp(&pb[2]))
                .then(a.0.id().cmp(&b.0.id()))
        });
        return write_leaf(c, records, generation, next);
    }
    fn split(
        records: &mut [(Observation, u64)],
        offset: usize,
        levels: usize,
        leaf: usize,
        ranges: &mut [(usize, usize); FANOUT],
        count: &mut usize,
    ) {
        if levels == 0 || records.len() <= leaf {
            ranges[*count] = (offset, offset + records.len());
            *count += 1;
            return;
        }
        let mut bounds = Bounds::point(records[0].0.site().support.center().coordinates());
        for (o, _) in records.iter().skip(1) {
            bounds.include(Bounds::point(o.site().support.center().coordinates()));
        }
        let axis = (0..3)
            .max_by(|&a, &b| {
                (bounds.upper[a] - bounds.lower[a]).total_cmp(&(bounds.upper[b] - bounds.lower[b]))
            })
            .unwrap();
        let mid = records.len() / 2;
        records.select_nth_unstable_by(mid, |a, b| {
            a.0.site().support.center().coordinates()[axis]
                .total_cmp(&b.0.site().support.center().coordinates()[axis])
                .then(a.0.id().cmp(&b.0.id()))
        });
        let (left, right) = records.split_at_mut(mid);
        split(left, offset, levels - 1, leaf, ranges, count);
        split(right, offset + mid, levels - 1, leaf, ranges, count);
    }
    let mut ranges = [(0, 0); FANOUT];
    let mut count = 0;
    split(records, 0, 3, leaf, &mut ranges, &mut count);
    let mut children = Vec::with_capacity(count);
    for &(first, last) in &ranges[..count] {
        children.push(write_cluster(
            c,
            &mut records[first..last],
            leaf,
            generation,
            next,
            cancel,
        )?);
    }
    write_branch(c, &children, next)
}

pub(super) fn write_payload(c: &Connection, page: u64, generation: u64, body: &[u8]) -> Result<()> {
    c.prepare_cached("INSERT INTO payloads(page,generation,body,crc) VALUES(?1,?2,?3,?4)")?
        .execute(params![
            sql_id(page)?,
            sql_id(generation)?,
            body,
            crc32fast::hash(body)
        ])?;
    Ok(())
}
pub(super) fn build_segment(
    c: &Connection,
    options: Options,
    generation: u64,
    cancel: &Cancellation,
) -> Result<Option<Node>> {
    // Bounds aggregation and B-tree staging use SQLite pages, not a Rust array
    // of all records or an unbounded in-memory sort. No ORDER BY sort is needed:
    // ordered's primary key supplies Morton, coordinate and ID order directly.
    let bounds: Option<Bounds> = c.query_row(
        "SELECT min(x),min(y),min(z),max(x),max(y),max(z) FROM staging",
        [],
        |r| {
            let x: Option<f64> = r.get(0)?;
            Ok(if let Some(x) = x {
                Some(Bounds {
                    lower: [x, r.get(1)?, r.get(2)?],
                    upper: [r.get(3)?, r.get(4)?, r.get(5)?],
                })
            } else {
                None
            })
        },
    )?;
    if let Some(bounds) = bounds {
        let mut s = c.prepare("SELECT id,x,y,z,rank,body FROM staging")?;
        let mut rows = s.query([])?;
        let mut insert = c.prepare("INSERT INTO ordered VALUES(?1,?2,?3,?4,?5,?6,?7)")?;
        while let Some(r) = rows.next()? {
            cancel.check()?;
            let p = [r.get::<_, f64>(1)?, r.get(2)?, r.get(3)?];
            let id = r
                .get_ref(0)?
                .as_blob()
                .map_err(|_| invalid("staging identity"))?;
            let body = r
                .get_ref(5)?
                .as_blob()
                .map_err(|_| invalid("staging payload"))?;
            insert.execute(params![
                morton(p, bounds) as i64,
                p[0],
                p[1],
                p[2],
                id,
                r.get::<_, i64>(4)?,
                body
            ])?;
        }
    }
    let mut next: u64 = c.query_row("SELECT coalesce(max(id),0)+1 FROM geometry", [], |r| {
        sql_u64(r, 0)
    })?;
    let mut levels = Levels::new();
    let mut records = Vec::with_capacity(options.leaf_records * options.build_window_leaves);
    {
        let mut s = c.prepare("SELECT body,rank FROM ordered ORDER BY code,x,y,z,id")?;
        let mut rows = s.query([])?;
        while let Some(r) = rows.next()? {
            cancel.check()?;
            records.push((
                decode_record(
                    r.get_ref(0)?
                        .as_blob()
                        .map_err(|_| invalid("ordered payload"))?,
                )?,
                sql_u64(r, 1)?,
            ));
            if records.len() == options.leaf_records * options.build_window_leaves {
                levels.push(
                    c,
                    write_cluster(
                        c,
                        &mut records,
                        options.leaf_records,
                        generation,
                        &mut next,
                        cancel,
                    )?,
                    0,
                    &mut next,
                )?;
                records.clear();
            }
        }
    }
    if !records.is_empty() {
        levels.push(
            c,
            write_cluster(
                c,
                &mut records,
                options.leaf_records,
                generation,
                &mut next,
                cancel,
            )?,
            0,
            &mut next,
        )?;
    }
    let root = levels.finish(c, &mut next)?;
    c.execute_batch("DROP TABLE ordered; DROP TABLE staging;")?;
    Ok(root)
}
