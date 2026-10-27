//! Persistent observation adapters. Binary point files need no optional feature.
//! SQLite persistence and transactional checkpoints are enabled by `sqlite`.
//! Spatial selection contracts live in `spatial`; adapters implement those
//! contracts without introducing persistence into the resident index.
mod point_file;
pub use point_file::{PointFile, PointReader};
#[cfg(feature = "sqlite")]
pub(crate) mod sqlite;
#[cfg(feature = "sqlite")]
pub use sqlite::{SqliteReadPool, SqliteStore};

pub mod grid;
use crate::{data::Observation, execution::Cancellation, spatial::ObservationSource, Result};
/// Caller-supplied digest must cover inputs, covariance, known mean and drift,
/// ordered path, seed, realization, numerical/backend options, and implementation
/// version. Arbitrary model or mean closures
/// cannot be fingerprinted automatically. Group output and this record commit together.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Checkpoint {
    pub identity: [u8; 32],
    pub completed_group: u64,
}

/// Atomic grouped-output storage used by restartable drivers.
/// Implementations must compare `expected` with the persisted checkpoint and
/// commit every observation plus `next` as one transaction. Before-commit errors
/// leave the prior state intact. Interruption or lost acknowledgement may leave
/// the old or fully committed group, never partial output; resumption reads the
/// checkpoint to determine which. Reject stale writers, identity changes and
/// non-increasing completed groups. Durability is an adapter-specific guarantee;
/// an in-memory implementation can satisfy atomicity without surviving a process.
pub trait CheckpointStore: ObservationSource {
    fn checkpoint(&self) -> Result<Option<Checkpoint>>;
    fn is_empty(&self) -> Result<bool>;
    fn commit_group(
        &mut self,
        observations: &[Observation],
        expected: Option<&Checkpoint>,
        next: &Checkpoint,
        cancel: &Cancellation,
    ) -> Result<()>;
}

#[cfg(feature = "sqlite")]
pub mod packed;

