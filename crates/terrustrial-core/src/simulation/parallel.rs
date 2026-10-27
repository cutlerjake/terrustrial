//! Bounded parallel preparation, dependency-ready draws, and ordered publication.
//!
//! `simulate_pipelined` retains immutable source views in a rolling lookahead
//! window. Selected predecessors form a DAG; a ready draw can overtake unrelated
//! work while publication remains ordered. Group-keyed random streams and actual
//! conditioning values preserve sequential results. The older `simulate_groups`
//! entry point uses batch preparation and ordered draws for generic mutable stores.
//!
//! Sources, covariance, drift and mean must be immutable/deterministic during a
//! call. Target IDs must be globally unique and disjoint from hard and existing
//! simulation IDs. Sources must implement geometric neighborhood semantics, not
//! value-dependent filtering. A source's visit_bounds override preserves indexed
//! pruning in the path overlay; the default scans. Stores must support concurrent
//! immutable reads (Sync). SQLite connection pooling is a separate adapter.
//!
//! Execution reserves per-group and per-worker allowances before dispatch. Each
//! slot has an independent sub-budget, so memory admission cannot race between
//! workers. Search and Gaussian arenas persist across batches. Source
//! caches, caller-owned iterator storage and allocator overhead are outside these
//! logical reservations. The rolling path additionally admits its bounded graph,
//! predecessor storage, and up to four blocking-service arenas for packed reads.
//! Result capacities persist; slots are recycled after ordered consumption.
use super::*;
mod rolling;
mod simulation;
#[cfg(feature = "experimental-two-phase")]
pub mod two_phase;
mod window;
use crate::execution::{parallel::prepare_batches, ExecutionOptions};
#[cfg(test)]
use crate::spatial::Neighborhood;
pub use rolling::{simulate_pipelined, simulate_pipelined_with_residuals, SnapshotStore};
pub use simulation::{simulate_groups, simulate_groups_with_residuals};
use window::{Checked, Overlay, Window};

struct Worker<'a, M: Mode = Unchecked> {
    diagnostic: M,
    budget: MemoryBudget,
    searches: Searches<'a>,
    simulation: Option<SimulationInputs>,
}
fn workers<'a, M: Mode>(
    execution: &ExecutionOptions,
    hard: NeighborhoodQuery<'a>,
    simulated: NeighborhoodQuery<'a>,
) -> Result<Vec<Worker<'a, M>>> {
    workers_n(execution, hard, simulated, execution.worker_count())
}
fn workers_n<'a, M: Mode>(
    execution: &ExecutionOptions,
    hard: NeighborhoodQuery<'a>,
    simulated: NeighborhoodQuery<'a>,
    count: usize,
) -> Result<Vec<Worker<'a, M>>> {
    (0..count)
        .map(|_| {
            let budget = MemoryBudget::new(execution.bytes_per_group);
            let searches = Searches::with_neighborhoods(hard, simulated, &budget)?;
            Ok(Worker {
                diagnostic: M::default(),
                budget,
                searches,
                simulation: None,
            })
        })
        .collect()
}
