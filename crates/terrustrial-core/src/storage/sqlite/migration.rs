use super::*;

impl SqliteStore {
    /// Explicitly upgrade an uncheckpointed version-1 store in one transaction.
    ///
    /// Version 1 discarded negative-zero signs. Migration preserves values as
    /// currently stored, using positive zero for those unrecoverable signs; use
    /// the original input instead when exact original bits are required. Stores
    /// with simulation checkpoints are refused and must be rebuilt by rerunning
    /// the simulation. No incompatible checkpoint is silently carried forward.
    ///
    /// Full integrity validation precedes commit. Cancellation is checked between
    /// SQLite statements and each payload row, not inside SQLite's structural
    /// checking statements. Failures roll back the schema and version together.
    pub fn migrate_v1(
        path: impl AsRef<Path>,
        cache_bytes: u64,
        budget: &MemoryBudget,
        cancel: &Cancellation,
    ) -> Result<Self> {
        cancel.check()?;
        let mut store = Self::connect(path.as_ref(), cache_bytes, budget)?;
        let transaction = store
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        let app: i64 = transaction.pragma_query_value(None, "application_id", |r| r.get(0))?;
        let version: i64 = transaction.pragma_query_value(None, "user_version", |r| r.get(0))?;
        if app != APPLICATION_ID || version != 1 {
            return Err(invalid("migration requires a version-1 spatial database"));
        }
        if checkpoint(&transaction)?.is_some() {
            return Err(invalid("version-1 checkpoint cannot retain original negative-zero signs; restart the simulation from its original inputs"));
        }
        integrity::check_extent(&transaction, &store.path)?;
        cancel.check()?;
        transaction.execute_batch("ALTER TABLE points ADD COLUMN zero_signs INTEGER NOT NULL DEFAULT 0 CHECK(zero_signs BETWEEN 0 AND 31)")?;
        add_support(&transaction)?;
        transaction.pragma_update(None, "user_version", SCHEMA_VERSION)?;
        integrity::check_contents(&transaction, cancel)?;
        transaction.commit()?;
        Ok(store)
    }
}

fn add_support(connection: &Connection) -> Result<()> {
    connection.execute_batch("ALTER TABLE points ADD COLUMN support BLOB CHECK(support IS NULL OR (typeof(support)='blob' AND length(support)=48))")?;
    Ok(())
}
impl SqliteStore {
    /// Upgrade a version-2 point store atomically, retaining stored values and
    /// checkpoints. Existing rows get NULL support, meaning point support.
    /// Previously discarded metadata cannot be reconstructed. The caller must
    /// still verify input/model/path and implementation identity before resuming;
    /// format migration does not authorize running a different simulation.
    /// Integrity validation precedes commit; cancellation and failures roll back
    /// the column and version together. No dataset copy is retained in Rust.
    pub fn migrate_v2(
        path: impl AsRef<Path>,
        cache_bytes: u64,
        budget: &MemoryBudget,
        cancel: &Cancellation,
    ) -> Result<Self> {
        cancel.check()?;
        let mut store = Self::connect(path.as_ref(), cache_bytes, budget)?;
        let transaction = store
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        let app: i64 = transaction.pragma_query_value(None, "application_id", |r| r.get(0))?;
        let version: i64 = transaction.pragma_query_value(None, "user_version", |r| r.get(0))?;
        if app != APPLICATION_ID || version != 2 {
            return Err(invalid("migration requires a version-2 spatial database"));
        }
        integrity::check_extent(&transaction, &store.path)?;
        cancel.check()?;
        add_support(&transaction)?;
        transaction.pragma_update(None, "user_version", SCHEMA_VERSION)?;
        integrity::check_contents(&transaction, cancel)?;
        transaction.commit()?;
        Ok(store)
    }
}
