//! Resource admission, cancellation, deterministic streams, and bounded execution.
pub(crate) mod paging;
pub(crate) mod parallel;
pub(crate) mod pipeline;
#[doc(hidden)]
pub mod profiling;
use crate::{Error, Result};
pub use parallel::ExecutionOptions;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

/// A native thread owned by an estimation, rolling simulation, or packed writer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ThreadRole {
    Compute,
    BlockingRead,
    Publish,
    /// Optional packed-publication payload decoding, separate from the writer.
    /// A caller enabling this role coordinates its CPU allowance with compute
    /// and publication; the library does not reserve or select a core.
    PublicationPrepare,
}

/// Startup identity within one invocation, independent of operating-system IDs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ThreadInfo {
    pub role: ThreadRole,
    /// Zero-based index within this role.
    pub index: usize,
    /// Number of native threads with this role in this invocation.
    pub count: usize,
}

/// Optional initialization on each newly created native worker, before work.
///
/// The callback runs concurrently, must return promptly, and may configure only
/// its current worker. It never runs on the caller or on existing Rayon threads.
/// Errors and panics reject startup; pipeline workers join without emitting any
/// output. A packed writer constructor also waits for successful startup.
/// Empty operations need not create workers or call the hook.
///
/// Clones share the callback, allowing a caller-owned placement plan to serve
/// both execution and a packed writer. There is no global registration or
/// automatic core reservation: the caller coordinates overlap between concurrent
/// runs and preserves the host's CPU restrictions. Without a hook, native threads
/// retain the operating system's inherited scheduling behavior.
#[derive(Clone)]
pub struct ThreadStart(Arc<dyn Fn(ThreadInfo) -> Result<()> + Send + Sync>);
impl ThreadStart {
    pub fn new(callback: impl Fn(ThreadInfo) -> Result<()> + Send + Sync + 'static) -> Self {
        Self(Arc::new(callback))
    }
    pub(crate) fn run(&self, info: ThreadInfo) -> Result<()> {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| (self.0)(info))).unwrap_or_else(
            |_| {
                Err(Error::Backend(format!(
                    "thread startup hook panicked for {:?} worker {} of {}",
                    info.role, info.index, info.count
                )))
            },
        )
    }
}
impl std::fmt::Debug for ThreadStart {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ThreadStart").finish_non_exhaustive()
    }
}

/// Resource limits with optional per-run native-thread initialization.
///
/// Ordinary [`ExecutionOptions`] convert directly and keep inherited scheduling.
/// Estimation (including residual/component variants) and `simulate_pipelined`
/// accept either form. Resource limits remain `Copy`; callbacks are owned by the
/// invocation and are never installed globally.
#[derive(Clone, Debug)]
pub struct ExecutionConfig {
    pub(crate) options: ExecutionOptions,
    pub(crate) thread_start: Option<ThreadStart>,
}
impl From<ExecutionOptions> for ExecutionConfig {
    fn from(options: ExecutionOptions) -> Self {
        Self {
            options,
            thread_start: None,
        }
    }
}

#[derive(Clone, Default, Debug)]
pub struct Cancellation(Arc<AtomicBool>);
impl Cancellation {
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Release);
    }
    pub fn check(&self) -> Result<()> {
        if self.0.load(Ordering::Acquire) {
            Err(Error::Cancelled)
        } else {
            Ok(())
        }
    }
}

/// ChaCha8 with an injective encoding of four 64-bit identifiers in its 256-bit key.
/// Version this mapping if reproducibility ever changes. No worker ID is involved.
pub fn random_stream(seed: u64, realization: u64, item: u64, purpose: u64) -> ChaCha8Rng {
    let mut key = [0_u8; 32];
    for (chunk, id) in key
        .as_chunks_mut::<8>()
        .0
        .iter_mut()
        .zip([seed, realization, item, purpose])
    {
        chunk.copy_from_slice(&id.to_le_bytes());
    }
    ChaCha8Rng::from_seed(key)
}

pub(crate) mod buffer;
mod memory;
pub use memory::{MemoryBudget, Reservation};

pub(crate) mod grid;
