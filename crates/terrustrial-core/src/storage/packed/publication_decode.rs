//! Bounded, owned-buffer cold-payload decoding for scheduled publication.
use super::*;
use crate::execution::{ThreadInfo, ThreadRole, ThreadStart};
use std::{
    sync::Condvar,
    thread::JoinHandle,
    time::{Duration, Instant},
};

const SLOTS: usize = 4;

/// Diagnostic snapshot of the currently retained publication decode helper.
///
/// Sampled after startup and successful publications. Counters accumulate for
/// that helper's lifetime. All fields reset to their defaults if memory pressure
/// retires the helper for serial fallback, or the writer reports a failure.
/// Durations are elapsed wall time, not CPU time, and can overlap publication.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PublicationDecodeStatistics {
    /// Whether the writer retains an admitted helper at the sampling point.
    pub admitted_current: bool,
    /// Number of payload decodes completed by that helper.
    pub decoded_pages: u64,
    /// Elapsed time inside helper payload decoding, in nanoseconds.
    pub decode_elapsed_ns: u64,
    /// Elapsed writer time collecting a result with waiting enabled, in nanoseconds.
    pub collect_wait_elapsed_ns: u64,
    /// Highest number of queued, running, or completed but uncollected jobs.
    pub max_inflight: u64,
}

struct Job {
    key: Key,
    bytes: Vec<u8>,
    reservation: Reservation,
}
struct Decoded {
    key: Key,
    bytes: Vec<u8>,
    value: Result<Arc<Cached>>,
}
struct State {
    jobs: VecDeque<Job>,
    results: VecDeque<Decoded>,
    starting: bool,
    stop: bool,
    error: Option<Error>,
}
struct Shared {
    state: Mutex<State>,
    changed: Condvar,
    decoded: AtomicU64,
    decode_ns: AtomicU64,
}

pub(super) struct Decoder {
    shared: Arc<Shared>,
    worker: Option<JoinHandle<()>>,
    free: Vec<Vec<u8>>,
    inflight: usize,
    wait_ns: u64,
    maximum_inflight: usize,
    // Covers every extra byte-buffer/queue capacity through helper shutdown.
    _memory: Reservation,
}

impl Decoder {
    pub(super) fn try_new(
        inner: &Inner,
        cancel: &Cancellation,
        start: &ThreadStart,
    ) -> Result<Option<Self>> {
        let bytes_per_buffer = inner
            .options
            .leaf_records
            .checked_mul(RECORD_BYTES)
            .ok_or(Error::SizeOverflow)?;
        let overhead = std::mem::size_of::<Self>() + std::mem::size_of::<Shared>() + 1024;
        let estimate = builder::page_scratch_bytes(inner.options)
            .checked_add(
                (SLOTS
                    * (bytes_per_buffer
                        + std::mem::size_of::<Vec<u8>>()
                        + std::mem::size_of::<Job>()
                        + std::mem::size_of::<Decoded>())
                    + overhead) as u64,
            )
            .ok_or(Error::SizeOverflow)?;
        // Optional setup does not evict useful cache entries to obtain a helper.
        let Ok(mut memory) = inner.budget.reserve(estimate) else {
            return Ok(None);
        };
        let mut free = Vec::new();
        free.try_reserve_exact(SLOTS)
            .map_err(|_| invalid("decode buffer handles"))?;
        for _ in 0..SLOTS {
            let mut bytes = Vec::new();
            bytes
                .try_reserve_exact(bytes_per_buffer)
                .map_err(|_| invalid("decode payload buffer"))?;
            free.push(bytes);
        }
        let mut jobs = VecDeque::new();
        jobs.try_reserve_exact(SLOTS)
            .map_err(|_| invalid("decode job queue"))?;
        let mut results = VecDeque::new();
        results
            .try_reserve_exact(SLOTS)
            .map_err(|_| invalid("decode result queue"))?;
        let actual = builder::page_scratch_bytes(inner.options)
            + (free.iter().map(Vec::capacity).sum::<usize>()
                + free.capacity() * std::mem::size_of::<Vec<u8>>()
                + jobs.capacity() * std::mem::size_of::<Job>()
                + results.capacity() * std::mem::size_of::<Decoded>()
                + overhead) as u64;
        if let Err(error) = memory.resize(actual) {
            return match error {
                Error::MemoryBudget { .. } => Ok(None),
                error => Err(error),
            };
        }
        let shared = Arc::new(Shared {
            state: Mutex::new(State {
                jobs,
                results,
                starting: true,
                stop: false,
                error: None,
            }),
            changed: Condvar::new(),
            decoded: AtomicU64::new(0),
            decode_ns: AtomicU64::new(0),
        });
        let helper_shared = shared.clone();
        let helper_cancel = cancel.clone();
        let helper_start = start.clone();
        let capacity = inner.options.leaf_records;
        let worker = std::thread::Builder::new()
            .name("terrustrial-publish-decode".into())
            .spawn(move || {
                let outcome =
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<()> {
                        helper_start.run(ThreadInfo {
                            role: ThreadRole::PublicationPrepare,
                            index: 0,
                            count: 1,
                        })?;
                        {
                            let mut state = lock(&helper_shared.state)?;
                            state.starting = false;
                            helper_shared.changed.notify_all();
                        }
                        loop {
                            let job = {
                                let mut state = lock(&helper_shared.state)?;
                                loop {
                                    if state.stop {
                                        return Ok(());
                                    }
                                    helper_cancel.check()?;
                                    if let Some(job) = state.jobs.pop_front() {
                                        break job;
                                    }
                                    state = helper_shared
                                        .changed
                                        .wait_timeout(state, Duration::from_millis(10))
                                        .map_err(|_| invalid("decode queue poisoned"))?
                                        .0;
                                }
                            };
                            helper_cancel.check()?;
                            let began = Instant::now();
                            let value = Cached::payload(&job.bytes, capacity, job.reservation)
                                .map(Arc::new);
                            helper_shared.decode_ns.fetch_add(
                                began.elapsed().as_nanos().min(u64::MAX as u128) as u64,
                                Ordering::Relaxed,
                            );
                            helper_shared.decoded.fetch_add(1, Ordering::Relaxed);
                            let mut state = lock(&helper_shared.state)?;
                            if state.stop {
                                return Ok(());
                            }
                            // All queued, running and completed jobs share SLOTS tokens.
                            assert!(state.results.len() < SLOTS);
                            state.results.push_back(Decoded {
                                key: job.key,
                                bytes: job.bytes,
                                value,
                            });
                            helper_shared.changed.notify_all();
                        }
                    }));
                let error = match outcome {
                    Ok(Ok(())) => None,
                    Ok(Err(error)) => Some(error),
                    Err(_) => Some(invalid("publication decode helper panicked")),
                };
                let mut state = helper_shared
                    .state
                    .lock()
                    .unwrap_or_else(|e| e.into_inner());
                if let Some(error) = error {
                    state.error = Some(error);
                }
                state.starting = false;
                state.stop = true;
                helper_shared.changed.notify_all();
            })?;
        let mut decoder = Self {
            shared,
            worker: Some(worker),
            free,
            inflight: 0,
            wait_ns: 0,
            maximum_inflight: 0,
            _memory: memory,
        };
        let started = (|| {
            let mut state = lock(&decoder.shared.state)?;
            while state.starting {
                cancel.check()?;
                state = decoder
                    .shared
                    .changed
                    .wait_timeout(state, Duration::from_millis(10))
                    .map_err(|_| invalid("decode startup poisoned"))?
                    .0;
            }
            if let Some(error) = state.error.take() {
                return Err(error);
            }
            if state.stop {
                return Err(invalid("decode helper stopped during startup"));
            }
            Ok(())
        })();
        if let Err(error) = started {
            decoder.stop_and_join();
            return Err(error);
        }
        Ok(Some(decoder))
    }

    pub(super) fn statistics(&self) -> PublicationDecodeStatistics {
        PublicationDecodeStatistics {
            admitted_current: self.active(),
            decoded_pages: self.shared.decoded.load(Ordering::Relaxed),
            decode_elapsed_ns: self.shared.decode_ns.load(Ordering::Relaxed),
            collect_wait_elapsed_ns: self.wait_ns,
            max_inflight: self.maximum_inflight as u64,
        }
    }
    pub(super) fn active(&self) -> bool {
        self.worker.is_some()
    }
    pub(super) fn inflight(&self) -> usize {
        self.inflight
    }

    fn collect(
        &mut self,
        warmed: &mut Vec<(Key, Arc<Cached>)>,
        cancel: &Cancellation,
        wait: bool,
    ) -> Result<bool> {
        let began = Instant::now();
        let result = (|| {
            let mut state = lock(&self.shared.state)?;
            loop {
                cancel.check()?;
                if let Some(error) = state.error.take() {
                    return Err(error);
                }
                if let Some(result) = state.results.pop_front() {
                    return Ok(Some(result));
                }
                if state.stop {
                    return Err(invalid("decode helper stopped"));
                }
                if !wait {
                    return Ok(None);
                }
                state = self
                    .shared
                    .changed
                    .wait_timeout(state, Duration::from_millis(10))
                    .map_err(|_| invalid("decode result queue poisoned"))?
                    .0;
            }
        })();
        if wait {
            self.wait_ns = self
                .wait_ns
                .saturating_add(began.elapsed().as_nanos().min(u64::MAX as u128) as u64);
        }
        if let Some(result) = result? {
            self.inflight -= 1;
            assert!(self.free.len() < SLOTS);
            self.free.push(result.bytes);
            assert!(warmed.len() < warmed.capacity());
            warmed.push((result.key, result.value?));
            Ok(true)
        } else {
            Ok(false)
        }
    }

    pub(super) fn stage(
        &mut self,
        inner: &Inner,
        page: u64,
        generation: u64,
        bytes: &mut Vec<u8>,
        warmed: &mut Vec<(Key, Arc<Cached>)>,
        cancel: &Cancellation,
    ) -> Result<()> {
        cancel.check()?;
        while self.collect(warmed, cancel, false)? {}
        if warmed.len() + self.inflight >= warmed.capacity() {
            return Ok(());
        }
        let Ok(reservation) = inner.budget.reserve(inner.page_charge(true)) else {
            return Ok(());
        };
        while self.free.is_empty() {
            self.collect(warmed, cancel, true)?;
        }
        let replacement = self.free.pop().expect("admitted decode buffer");
        let owned = std::mem::replace(bytes, replacement);
        let mut state = lock(&self.shared.state)?;
        if let Some(error) = state.error.take() {
            return Err(error);
        }
        if state.stop {
            return Err(invalid("decode helper stopped"));
        }
        assert!(state.jobs.len() < SLOTS);
        state.jobs.push_back(Job {
            key: Key {
                page,
                generation,
                payload: true,
            },
            bytes: owned,
            reservation,
        });
        self.inflight += 1;
        self.maximum_inflight = self.maximum_inflight.max(self.inflight);
        self.shared.changed.notify_one();
        Ok(())
    }

    pub(super) fn drain(
        &mut self,
        warmed: &mut Vec<(Key, Arc<Cached>)>,
        cancel: &Cancellation,
    ) -> Result<()> {
        while self.inflight != 0 {
            self.collect(warmed, cancel, true)?;
        }
        cancel.check()?;
        let mut state = lock(&self.shared.state)?;
        if let Some(error) = state.error.take() {
            return Err(error);
        }
        if state.stop {
            return Err(invalid("decode helper stopped"));
        }
        Ok(())
    }

    pub(super) fn stop_and_join(&mut self) {
        {
            let mut state = self.shared.state.lock().unwrap_or_else(|e| e.into_inner());
            state.stop = true;
            self.shared.changed.notify_all();
        }
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
        let mut state = self.shared.state.lock().unwrap_or_else(|e| e.into_inner());
        state.jobs.clear();
        state.results.clear();
        self.inflight = 0;
    }
}
impl Drop for Decoder {
    fn drop(&mut self) {
        self.stop_and_join();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::Site;
    use std::{cell::RefCell, sync::atomic::AtomicUsize};

    #[derive(Default)]
    struct ThreadCounts {
        live: AtomicUsize,
        exited: AtomicUsize,
    }
    struct ThreadLifetime(Arc<ThreadCounts>);
    impl Drop for ThreadLifetime {
        fn drop(&mut self) {
            self.0.live.fetch_sub(1, Ordering::AcqRel);
            self.0.exited.fetch_add(1, Ordering::Release);
        }
    }
    thread_local! {
        // The guard is dropped when the native thread exits, not when its
        // startup callback returns. This distinguishes joining from signaling.
        static THREAD_LIFETIME: RefCell<Option<ThreadLifetime>> = const { RefCell::new(None) };
    }
    fn track_thread(counts: &Arc<ThreadCounts>) {
        THREAD_LIFETIME.with(|slot| {
            let mut slot = slot.borrow_mut();
            assert!(slot.is_none());
            counts.live.fetch_add(1, Ordering::AcqRel);
            *slot = Some(ThreadLifetime(counts.clone()));
        });
    }
    fn scheduled_options() -> ScheduledOptions {
        ScheduledOptions {
            maximum_group_observations: 8,
            queued_groups: 2,
            commit_groups: 1,
            retained_groups: 3,
        }
    }

    fn fixture(path: &Path, budget: &MemoryBudget) -> PackedStore {
        let mut builder = PackedBuilder::create(
            path,
            Options {
                leaf_records: 4,
                ..Options::default()
            },
            budget,
        )
        .unwrap();
        for id in 0..16 {
            builder
                .push(
                    Observation::new(
                        id,
                        Site::point(Point3::new([id as f64, 0., 0.]).unwrap(), 0),
                        0.,
                        0.,
                    )
                    .unwrap(),
                    id % 2 + 1,
                    &Cancellation::default(),
                )
                .unwrap();
        }
        builder.finish(&Cancellation::default()).unwrap()
    }
    fn helper(store: &PackedStore, cancel: &Cancellation) -> Option<Decoder> {
        Decoder::try_new(
            &store.inner,
            cancel,
            &ThreadStart::new(|info| {
                assert_eq!(info.role, ThreadRole::PublicationPrepare);
                assert_eq!((info.index, info.count), (0, 1));
                Ok(())
            }),
        )
        .unwrap()
    }
    fn payload_rows(store: &PackedStore) -> Vec<(u64, u64, Vec<u8>, u32, u8)> {
        let connection = lock(&store.inner.writer).unwrap();
        let mut query = connection
            .prepare(
                "SELECT page,generation,body,crc,encoding FROM payloads ORDER BY page,generation",
            )
            .unwrap();
        let rows = query
            .query_map([], |r| {
                Ok((
                    sql_u64(r, 0)?,
                    sql_u64(r, 1)?,
                    r.get(2)?,
                    r.get(3)?,
                    r.get(4)?,
                ))
            })
            .unwrap()
            .collect::<std::result::Result<_, _>>()
            .unwrap();
        rows
    }

    #[test]
    fn publication_decode_matches_serial_bytes_reopen_and_old_snapshots() {
        let directory = tempfile::tempdir().unwrap();
        let budget = MemoryBudget::new(32 << 20);
        let serial = fixture(&directory.path().join("serial"), &budget);
        let parallel = fixture(&directory.path().join("parallel"), &budget);
        parallel.clear_cache().unwrap();
        let cancel = Cancellation::default();
        let mut decoder = helper(&parallel, &cancel);
        let mut history = Vec::new();
        for rank in 1..=2 {
            let values: Vec<_> = (0..16)
                .filter(|id| id % 2 + 1 == rank)
                .map(|id| (id, id as f64 + 0.25))
                .collect();
            let expected = serial.publish_groups(&[(rank, &values)], &cancel).unwrap();
            let actual = parallel
                .publish_staged_with_decode(&[(rank, &values)], &cancel, &mut None, &mut decoder)
                .unwrap();
            assert_eq!(
                (actual.generation(), actual.completed_group()),
                (expected.generation(), rank)
            );
            assert_eq!(payload_rows(&serial), payload_rows(&parallel));
            history.push(actual);
        }
        let counts = decoder.as_ref().unwrap().statistics();
        assert!(
            counts.decoded_pages > 0,
            "the cold decode path must actually execute"
        );
        assert!(counts.max_inflight <= SLOTS as u64);
        let mut old_ids = Vec::new();
        history[0]
            .scan(&mut |o| {
                old_ids.push(o.id());
                assert_eq!(o.value(), o.id() as f64 + 0.25);
                Ok(())
            })
            .unwrap();
        old_ids.sort_unstable();
        assert_eq!(old_ids, (0..16).step_by(2).collect::<Vec<_>>());
        drop((decoder, history, serial, parallel));
        assert_eq!(budget.used(), 0);
        let reopened = PackedStore::open(
            &directory.path().join("parallel"),
            Options::default(),
            &budget,
        )
        .unwrap();
        assert_eq!(reopened.snapshot().unwrap().completed_group(), 2);
        let mut count = 0;
        reopened
            .snapshot()
            .unwrap()
            .scan(&mut |o| {
                count += 1;
                assert_eq!(o.value(), o.id() as f64 + 0.25);
                Ok(())
            })
            .unwrap();
        assert_eq!(count, 16);
        drop(reopened);
        assert_eq!(budget.used(), 0);
    }

    #[test]
    fn publication_decode_corruption_rolls_back_and_joins_before_acknowledgment() {
        let directory = tempfile::tempdir().unwrap();
        let budget = MemoryBudget::new(32 << 20);
        let store = fixture(&directory.path().join("corrupt"), &budget);
        store.clear_cache().unwrap();
        {
            let connection = lock(&store.inner.writer).unwrap();
            let (page, mut body): (u64, Vec<u8>) = connection
                .query_row(
                    "SELECT page,body FROM payloads ORDER BY page DESC LIMIT 1",
                    [],
                    |r| Ok((sql_u64(r, 0)?, r.get(1)?)),
                )
                .unwrap();
            // Leave the checksum valid; corruption must be found by decoding,
            // after the transaction has already inserted some earlier pages.
            body[96..104].copy_from_slice(&f64::NAN.to_bits().to_le_bytes());
            connection
                .execute(
                    "UPDATE payloads SET body=?1,crc=?2 WHERE page=?3",
                    params![&body, crc32fast::hash(&body), sql_id(page).unwrap()],
                )
                .unwrap();
        }
        let before = payload_rows(&store);
        let cancel = Cancellation::default();
        let mut decoder = helper(&store, &cancel);
        let values: Vec<_> = (0..16).step_by(2).map(|id| (id, id as f64 + 1.)).collect();
        assert!(
            store
                .publish_staged_with_decode(&[(1, &values)], &cancel, &mut None, &mut decoder)
                .is_err()
        );
        assert!(!decoder.as_ref().unwrap().active());
        assert_eq!(store.snapshot().unwrap().completed_group(), 0);
        assert_eq!(payload_rows(&store), before);
        drop((decoder, store));
        assert_eq!(budget.used(), 0);
    }

    #[test]
    fn publication_decode_memory_fallback_releases_the_entire_optional_arena() {
        let directory = tempfile::tempdir().unwrap();
        let budget = MemoryBudget::new(32 << 20);
        let store = fixture(&directory.path().join("budget"), &budget);
        // ScheduledWriter retains a read view. Keep that same geometry charge
        // live; this test targets helper admission, not reloading a directory.
        let old = store.snapshot().unwrap();
        let baseline_charge = budget.used();
        let cancel = Cancellation::default();
        let held = budget.reserve(budget.limit() - budget.used()).unwrap();
        assert!(helper(&store, &cancel).is_none());
        drop(held);
        let mut decoder = helper(&store, &cancel);
        assert!(decoder.is_some());
        // Insufficient for transaction scratch while the helper is retained,
        // sufficient for normal publication after releasing its whole arena.
        let headroom = builder::page_scratch_bytes(store.inner.options) / 2;
        let held_bytes = budget.limit() - budget.used() - headroom;
        let held = budget.reserve(held_bytes).unwrap();
        let values: Vec<_> = (0..16).step_by(2).map(|id| (id, id as f64 + 1.)).collect();
        // An identical store under the same limit and external held charge
        // proves the ordinary path fits before testing optional reclamation.
        let serial_budget = MemoryBudget::new(budget.limit());
        let serial = fixture(&directory.path().join("budget-serial"), &serial_budget);
        let serial_old = serial.snapshot().unwrap();
        assert_eq!(serial_budget.used(), baseline_charge);
        let serial_held = serial_budget.reserve(held_bytes).unwrap();
        let serial_result = serial.publish_groups(&[(1, &values)], &cancel).unwrap();
        assert_eq!(serial_result.completed_group(), 1);
        drop((serial_result, serial_held, serial_old, serial));
        assert_eq!(serial_budget.used(), 0);
        store
            .publish_staged_with_decode(&[(1, &values)], &cancel, &mut None, &mut decoder)
            .unwrap();
        assert!(
            decoder.is_none(),
            "optional helper must release memory before serial retry"
        );
        assert_eq!(store.snapshot().unwrap().completed_group(), 1);
        assert_eq!(old.completed_group(), 0);
        drop((held, decoder, old, store));
        assert_eq!(budget.used(), 0);
    }

    #[test]
    fn publication_decode_cancellation_drains_owned_buffers_without_leaking() {
        let directory = tempfile::tempdir().unwrap();
        let budget = MemoryBudget::new(32 << 20);
        let store = fixture(&directory.path().join("cancel"), &budget);
        let cancel = Cancellation::default();
        let before = budget.used();
        let mut decoder = helper(&store, &cancel).unwrap();
        let mut warmed = Vec::with_capacity(SLOTS);
        let row =
            Observation::new(1, Site::point(Point3::new([0.; 3]).unwrap(), 0), 2., 0.).unwrap();
        let mut bytes = encode_record(&row).to_vec();
        for page in 1..=SLOTS as u64 {
            bytes.clear();
            bytes.extend_from_slice(&encode_record(&row));
            decoder
                .stage(&store.inner, page, 1, &mut bytes, &mut warmed, &cancel)
                .unwrap();
        }
        cancel.cancel();
        assert!(matches!(
            decoder.drain(&mut warmed, &cancel),
            Err(Error::Cancelled)
        ));
        decoder.stop_and_join();
        assert!(!decoder.active());
        drop((decoder, warmed, bytes));
        assert_eq!(budget.used(), before);
        drop(store);
        assert_eq!(budget.used(), 0);
    }

    #[test]
    fn publication_decode_startup_failure_or_panic_rejects_scheduled_writer() {
        let directory = tempfile::tempdir().unwrap();
        let budget = MemoryBudget::new(32 << 20);
        let store = fixture(&directory.path().join("startup"), &budget);
        for panic in [false, true] {
            let before = budget.used();
            let hook = ThreadStart::new(move |info| {
                if info.role == ThreadRole::PublicationPrepare {
                    if panic {
                        panic!("injected helper startup panic")
                    }
                    return Err(invalid("injected helper startup failure"));
                }
                Ok(())
            });
            assert!(
                ScheduledWriter::new_with_publication_decode(
                    store.clone(),
                    ScheduledOptions {
                        maximum_group_observations: 8,
                        queued_groups: 2,
                        commit_groups: 1,
                        retained_groups: 3,
                    },
                    &budget,
                    &Cancellation::default(),
                    hook
                )
                .is_err()
            );
            assert_eq!(budget.used(), before);
            assert_eq!(store.snapshot().unwrap().completed_group(), 0);
        }
        drop(store);
        assert_eq!(budget.used(), 0);
    }

    #[test]
    fn publication_decode_startup_initializes_both_roles_and_drop_joins_them() {
        let directory = tempfile::tempdir().unwrap();
        let budget = MemoryBudget::new(32 << 20);
        let store = fixture(&directory.path().join("startup-success"), &budget);
        let before = budget.used();
        let counts = Arc::new(ThreadCounts::default());
        let observed = Arc::new(Mutex::new(Vec::new()));
        let caller = std::thread::current().id();
        let hook = ThreadStart::new({
            let counts = counts.clone();
            let observed = observed.clone();
            move |info| {
                assert_eq!((info.index, info.count), (0, 1));
                assert_ne!(std::thread::current().id(), caller);
                track_thread(&counts);
                lock(&observed)?.push((info.role, std::thread::current().id()));
                Ok(())
            }
        });
        let writer = ScheduledWriter::new_with_publication_decode(
            store.clone(),
            scheduled_options(),
            &budget,
            &Cancellation::default(),
            hook,
        )
        .unwrap();
        let events = lock(&observed).unwrap();
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].0, ThreadRole::PublicationPrepare);
        assert_eq!(events[1].0, ThreadRole::Publish);
        assert_ne!(events[0].1, events[1].1);
        drop(events);
        assert_eq!(counts.live.load(Ordering::Acquire), 2);
        assert_eq!(counts.exited.load(Ordering::Acquire), 0);
        assert_eq!(
            writer.publication_decode_statistics(),
            PublicationDecodeStatistics {
                admitted_current: true,
                ..PublicationDecodeStatistics::default()
            }
        );
        drop(writer);
        assert_eq!(counts.live.load(Ordering::Acquire), 0);
        assert_eq!(counts.exited.load(Ordering::Acquire), 2);
        assert_eq!(budget.used(), before);
        drop(store);
        assert_eq!(budget.used(), 0);
    }

    #[test]
    fn publication_decode_publisher_startup_failure_joins_already_started_helper() {
        let directory = tempfile::tempdir().unwrap();
        let budget = MemoryBudget::new(32 << 20);
        let store = fixture(&directory.path().join("publisher-failure"), &budget);
        for panic in [false, true] {
            let before = budget.used();
            let counts = Arc::new(ThreadCounts::default());
            let hook = ThreadStart::new({
                let counts = counts.clone();
                move |info| {
                    assert_eq!((info.index, info.count), (0, 1));
                    track_thread(&counts);
                    if info.role == ThreadRole::Publish {
                        // The helper's callback completed and its thread is
                        // still alive when publisher initialization fails.
                        assert_eq!(counts.live.load(Ordering::Acquire), 2);
                        if panic {
                            panic!("injected publisher startup panic");
                        }
                        return Err(invalid("injected publisher startup failure"));
                    }
                    assert_eq!(info.role, ThreadRole::PublicationPrepare);
                    Ok(())
                }
            });
            assert!(
                ScheduledWriter::new_with_publication_decode(
                    store.clone(),
                    scheduled_options(),
                    &budget,
                    &Cancellation::default(),
                    hook,
                )
                .is_err()
            );
            assert_eq!(counts.live.load(Ordering::Acquire), 0);
            assert_eq!(counts.exited.load(Ordering::Acquire), 2);
            assert_eq!(budget.used(), before);
            assert_eq!(store.snapshot().unwrap().completed_group(), 0);
        }
        drop(store);
        assert_eq!(budget.used(), 0);
    }
}
