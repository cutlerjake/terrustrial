//! Opt-in, process-wide aggregate diagnostics. Durations overlap across workers;
//! they are not wall time or a claim that a queue delay is on the critical path.
//! Normal builds compile stamps and spans away. No per-job trace is retained.
//!
//! `dependency_wait` measures only jobs with unresolved predecessor draws.
//! `draw_queue` starts when those dependencies are resolved; `factor_queue`
//! measures selected work awaiting compute service. `io_queue` is deferred
//! selection awaiting a reader worker, not proof that its pages are resident.
//! Idle spans include condition-variable wake/reacquisition latency. Counts are
//! completed spans, not operating-system context-switch counts.
//! Numerical spans are nested within preparation: `data_covariance` assembles
//! the symmetric data matrix, `data_factor` includes drift/scaling and solves,
//! and `target_system` includes target covariance assembly and its solves.
//! Retention spans distinguish matrix copies from ownership exchanges.
//! `local_draw` is a subset of `draw`, not an additional stage duration.
#[allow(dead_code)] // Storage-specific stages are absent without `sqlite`.
#[derive(Clone, Copy, Debug)]
#[repr(usize)]
pub(crate) enum Metric {
    PipelineLock,
    ComputeIdle,
    IoIdle,
    PrepareQueue,
    IoQueue,
    FactorQueue,
    DependencyWait,
    DrawQueue,
    Prepare,
    BlockingPrepare,
    Factor,
    Draw,
    OrderedWait,
    Consume,
    Admit,
    WriterBackpressure,
    WriterFinish,
    Publish,
    Snapshot,
    CacheReadLock,
    CacheWriteLock,
    PageLoad,
    ReaderWait,
    PublishValidate,
    PublishPayload,
    PublishCommit,
    PublishCache,
    DataCovariance,
    DataFactor,
    TargetSystem,
    PublishCached,
    PublishCold,
    PublishFetch,
    PublishEncode,
    PublishInsert,
    PublishDecode,
    PublishPrepare,
    PublishWarm,
    SimulationSetup,
    PipelineSetup,
    PipelineJoin,
    RetainCopy,
    RetainSwap,
    LocalDraw,
    GroupInput,
    PageAcquire,
    PageReserve,
    PageSqlGeometry,
    PageSqlPayload,
    PageBlobCopy,
    PageRelease,
    PageChecksum,
    PageDecodeGeometry,
    PageDecodePayload,
    PageCacheLock,
    PageCacheInsert,
    PagePoolLock,
}
#[cfg(feature = "profiling")]
mod enabled {
    use std::sync::atomic::{AtomicU64, Ordering};
    pub(super) const COUNT: usize = super::Metric::PagePoolLock as usize + 1;
    // Separate cache lines avoid sharing between unrelated stages.
    #[repr(align(64))]
    pub(super) struct Counter(pub AtomicU64, pub AtomicU64);
    pub(super) static COUNTERS: [Counter; COUNT] =
        [const { Counter(AtomicU64::new(0), AtomicU64::new(0)) }; COUNT];
    pub(super) fn add(metric: super::Metric, start: std::time::Instant) {
        let counter = &COUNTERS[metric as usize];
        counter.0.fetch_add(1, Ordering::Relaxed);
        counter
            .1
            .fetch_add(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
}
#[derive(Clone, Copy, Default)]
pub(crate) struct Stamp {
    #[cfg(feature = "profiling")]
    start: Option<std::time::Instant>,
}
impl Stamp {
    #[inline]
    pub(crate) fn now() -> Self {
        Self {
            #[cfg(feature = "profiling")]
            start: Some(std::time::Instant::now()),
        }
    }
    #[inline]
    pub(crate) fn record(self, _metric: Metric) {
        #[cfg(feature = "profiling")]
        if let Some(start) = self.start {
            enabled::add(_metric, start);
        }
    }
}
pub(crate) struct Span(Stamp, Metric);
impl Span {
    #[inline]
    pub(crate) fn new(metric: Metric) -> Self {
        Self(Stamp::now(), metric)
    }
}
impl Drop for Span {
    #[inline]
    fn drop(&mut self) {
        self.0.record(self.1);
    }
}
/// Reset only while no profiled operation is active.
#[cfg(feature = "profiling")]
pub fn reset() {
    use std::sync::atomic::Ordering;
    for counter in &enabled::COUNTERS {
        counter.0.store(0, Ordering::Relaxed);
        counter.1.store(0, Ordering::Relaxed);
    }
}
/// Aggregate counts and summed seconds, in a stable order.
#[cfg(feature = "profiling")]
pub fn snapshot() -> Vec<(&'static str, u64, f64)> {
    use std::sync::atomic::Ordering;
    const NAMES: [&str; enabled::COUNT] = [
        "pipeline_lock",
        "compute_idle",
        "io_idle",
        "prepare_queue",
        "io_queue",
        "factor_queue",
        "dependency_wait",
        "draw_queue",
        "prepare",
        "blocking_prepare",
        "factor",
        "draw",
        "ordered_wait",
        "consume",
        "admit",
        "writer_backpressure",
        "writer_finish",
        "publish",
        "snapshot",
        "cache_read_lock",
        "cache_write_lock",
        "page_load",
        "reader_wait",
        "publish_validate",
        "publish_payload",
        "publish_commit",
        "publish_cache",
        "data_covariance",
        "data_factor",
        "target_system",
        "publish_cached",
        "publish_cold",
        "publish_fetch",
        "publish_encode",
        "publish_insert",
        "publish_decode",
        "publish_prepare",
        "publish_warm",
        "simulation_setup",
        "pipeline_setup",
        "pipeline_join",
        "retain_copy",
        "retain_swap",
        "local_draw",
        "group_input",
        "page_acquire",
        "page_reserve",
        "page_sql_geometry",
        "page_sql_payload",
        "page_blob_copy",
        "page_release",
        "page_checksum",
        "page_decode_geometry",
        "page_decode_payload",
        "page_cache_lock",
        "page_cache_insert",
        "page_pool_lock",
    ];
    NAMES
        .into_iter()
        .zip(&enabled::COUNTERS)
        .map(|(name, counter)| {
            (
                name,
                counter.0.load(Ordering::Relaxed),
                counter.1.load(Ordering::Relaxed) as f64 / 1e9,
            )
        })
        .collect()
}
