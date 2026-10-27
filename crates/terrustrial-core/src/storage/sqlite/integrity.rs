use super::*;

impl SqliteStore {
    /// Check structural and application integrity in one consistent read snapshot.
    ///
    /// In addition to SQLite and R-tree structure, this validates every observation,
    /// checkpoint encoding, index membership and finite boxes enclosing each indexed
    /// point. Conservative boxes wider than required remain valid. Extra Rust
    /// working storage does not grow with record count; SQLite manages its own
    /// integrity-check resources. This is a full-store diagnostic, not a cheap open
    /// check, a checksum of original values or an automatic repair operation.
    ///
    /// A valid-looking change to a value or checkpoint identity cannot be detected
    /// without an independent trusted digest. Ordinary opens validate the format
    /// header, not the complete store. Run this diagnostic after suspected damage
    /// before relying on indexed queries. Writers may wait until the check finishes.
    /// In rollback-journal mode the physical file must contain complete pages and
    /// be long enough for the snapshot. WAL mode delegates physical recovery to
    /// SQLite because committed pages can exist only in the WAL.
    pub fn integrity_check(&self) -> Result<()> {
        let transaction = self.connection.unchecked_transaction()?;
        self.validate_format()?;
        check_extent(&transaction, &self.path)?;
        check_contents(&transaction, &Cancellation::default())?;
        transaction.commit()?;
        Ok(())
    }
}

pub(super) fn check_extent(connection: &Connection, path: &Path) -> Result<()> {
    let journal: String = connection.pragma_query_value(None, "journal_mode", |r| r.get(0))?;
    if journal != "wal" {
        let page_size =
            u64::from(connection.pragma_query_value(None, "page_size", |r| r.get::<_, u32>(0))?);
        let pages =
            u64::from(connection.pragma_query_value(None, "page_count", |r| r.get::<_, u32>(0))?);
        let minimum = page_size.checked_mul(pages).ok_or(Error::SizeOverflow)?;
        // The retained path is resolved before the connection opens and does
        // not require UTF-8. The shared lock excludes rollback-mode writes.
        let length = std::fs::metadata(path)?.len();
        if page_size == 0 || length < minimum || !length.is_multiple_of(page_size) {
            return Err(invalid("database file contains incomplete pages"));
        }
    }
    Ok(())
}

pub(super) fn check_contents(connection: &Connection, cancel: &Cancellation) -> Result<()> {
    cancel.check()?;
    let result: String = connection.query_row("PRAGMA integrity_check", [], |row| row.get(0))?;
    if result != "ok" {
        return Err(invalid(format!("SQLite integrity check: {result}")));
    }
    let result: String =
        connection.query_row("SELECT rtreecheck('bounds')", [], |row| row.get(0))?;
    if result != "ok" {
        return Err(invalid(format!("R-tree integrity check: {result}")));
    }
    cancel.check()?;
    checkpoint(connection)?;
    {
        let mut statement = connection.prepare(
            "SELECT p.id,p.variable,p.x,p.y,p.z,p.value,p.noise,p.zero_signs,p.support,p.indexed,
             b.id,b.min_x,b.max_x,b.min_y,b.max_y,b.min_z,b.max_z
             FROM points AS p LEFT JOIN bounds AS b ON p.id=b.id",
        )?;
        let mut rows = statement.query([])?;
        while let Some(row) = rows.next()? {
            cancel.check()?;
            let id = row.get::<_, i64>(0)? as u64;
            let point = observation(row)?.site().support.center().coordinates();
            let indexed: i64 = row.get(9)?;
            let bound_id: Option<i64> = row.get(10)?;
            if indexed != i64::from(indexable(point)) || (indexed == 1) != bound_id.is_some() {
                return Err(invalid(format!(
                    "spatial index membership disagrees with payload for observation {id}"
                )));
            }
            if indexed == 1 {
                for (axis, coordinate) in point.into_iter().enumerate() {
                    let lower: f64 = row.get(11 + axis * 2)?;
                    let upper: f64 = row.get(12 + axis * 2)?;
                    if !lower.is_finite()
                        || !upper.is_finite()
                        || lower > coordinate
                        || upper < coordinate
                    {
                        return Err(invalid(format!(
                            "spatial index box does not enclose observation {id} on axis {axis}"
                        )));
                    }
                }
            }
        }
    }
    let orphaned: bool = connection.query_row(
        "SELECT EXISTS(SELECT 1 FROM bounds AS b LEFT JOIN points AS p
         ON p.id=b.id WHERE p.id IS NULL)",
        [],
        |row| row.get(0),
    )?;
    if orphaned {
        return Err(invalid("spatial index entry has no payload"));
    }
    cancel.check()?;
    Ok(())
}
