//! Bounded asynchronous publication of pre-indexed simulation groups.
use super::*;
use crate::execution::profiling::{Metric, Span};
use crate::{
    data::{Group, Site},
    execution::{ThreadInfo, ThreadRole, ThreadStart},
    simulation::parallel::SnapshotStore,
};
use std::{sync::Condvar, thread::JoinHandle, time::Duration};

#[derive(Clone, Copy, Debug)]
pub struct ScheduledOptions {
    pub maximum_group_observations: usize,
    /// Maximum drawn groups waiting for durable publication.
    pub queued_groups: usize,
    /// Number of groups per transaction. One requires durable acknowledgment
    /// before append_group returns. Larger batches acknowledge on finish;
    /// finalization flushes a shorter suffix.
    pub commit_groups: usize,
    /// Payload slots, including those pinned by historical preparation views.
    /// Use at least queued_groups + the preparation window + 1.
    pub retained_groups: usize,
}
struct Batch {
    rank: u64,
    observations: Vec<Observation>,
    bounds: crate::spatial::hierarchy::Bounds,
    values: Vec<(u64, f64)>,
}
// At most 32 groups share a chunk. For eight-center groups, 512 bytes give
// sixteen bits per center. Four probes reject most exact-site misses; saturation
// only causes more ordinary lookups. The existing chunk reservation includes
// these inline bytes, including historical copies retained by read views.
#[derive(Clone)]
struct CenterFilter([u64; 64]);
impl Default for CenterFilter {
    fn default() -> Self {
        Self([0; 64])
    }
}
impl CenterFilter {
    fn key(center: [f64; 3]) -> [usize; 4] {
        // Site equality treats signed zeros alike. No other distinct finite
        // f64 bit patterns compare equal, and Point3 excludes NaNs.
        let bits = center.map(|v| if v == 0. { 0 } else { v.to_bits() });
        let mut hash = bits[0] ^ bits[1].rotate_left(21) ^ bits[2].rotate_left(42);
        hash = (hash ^ (hash >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        hash = (hash ^ (hash >> 27)).wrapping_mul(0x94d049bb133111eb);
        hash ^= hash >> 31;
        std::array::from_fn(|i| ((hash >> (i * 12)) & 4095) as usize)
    }
    fn include(&mut self, observations: &[Observation]) {
        let mut previous = None;
        for observation in observations {
            let center = observation.site().support.center().coordinates();
            if previous == Some(center) {
                continue;
            }
            previous = Some(center);
            for bit in Self::key(center) {
                self.0[bit / 64] |= 1 << (bit % 64);
            }
        }
    }
    fn may_contain(&self, key: &[usize; 4]) -> bool {
        key.iter()
            .all(|&bit| self.0[bit / 64] & (1 << (bit % 64)) != 0)
    }
}
// Full chunks align with transaction boundaries. Sharing their metadata avoids
// cloning every pending payload Arc for every preparation view. Only the small
// unfinished tail needs copying when an older view still owns it.
struct BatchChunk {
    groups: Vec<Arc<Batch>>,
    centers: CenterFilter,
    _reservation: Reservation,
}
struct Pending {
    chunks: VecDeque<Arc<BatchChunk>>,
    count: usize,
    chunk_groups: usize,
}
impl Pending {
    fn append(&mut self, batch: Arc<Batch>, budget: &MemoryBudget) -> Result<()> {
        let extend = self
            .chunks
            .back()
            .is_some_and(|c| c.groups.len() < self.chunk_groups);
        if extend {
            if let Some(chunk) = Arc::get_mut(self.chunks.back_mut().unwrap()) {
                chunk.centers.include(&batch.observations);
                chunk.groups.push(batch);
                self.count += 1;
                return Ok(());
            }
        }
        let bytes = self.chunk_groups * std::mem::size_of::<Arc<Batch>>()
            + std::mem::size_of::<BatchChunk>()
            + 64;
        let reservation = budget.reserve(bytes as u64)?;
        let mut groups = Vec::new();
        groups
            .try_reserve_exact(self.chunk_groups)
            .map_err(|_| invalid("scheduled pending chunk allocation"))?;
        if extend {
            groups.extend(self.chunks.back().unwrap().groups.iter().cloned());
        }
        let mut centers = if extend {
            self.chunks.back().unwrap().centers.clone()
        } else {
            CenterFilter::default()
        };
        centers.include(&batch.observations);
        groups.push(batch);
        let chunk = Arc::new(BatchChunk {
            groups,
            centers,
            _reservation: reservation,
        });
        if extend {
            *self.chunks.back_mut().unwrap() = chunk;
        } else {
            self.chunks.push_back(chunk);
        }
        self.count += 1;
        Ok(())
    }
    fn discard_committed(&mut self, rank: u64) {
        while self
            .chunks
            .front()
            .is_some_and(|c| c.groups.last().unwrap().rank <= rank)
        {
            self.count -= self.chunks.pop_front().unwrap().groups.len();
        }
        // A full commit contains an integer number of chunks. A partial commit
        // occurs only at finish, which flushes the entire accepted suffix.
        debug_assert!(self.chunks.front().is_none_or(|c| c.groups[0].rank > rank));
    }
}
struct State {
    queue: VecDeque<Arc<Batch>>,
    snapshot: Snapshot,
    error: Option<Error>,
    failed: bool,
    starting: bool,
    decode_statistics: PublicationDecodeStatistics,
    flush: bool,
    close: bool,
}
struct Shared {
    state: Mutex<State>,
    changed: Condvar,
}
/// Writer for stores built with simulation rank = Group.id + 1. Read views merge
/// the committed snapshot and accepted pending groups without gaps or duplicates.
/// Output callbacks may see drawn data before it is durable; finish acknowledges
/// durability. This adapter does not create simulation identity checkpoints.
pub struct ScheduledWriter {
    shared: Arc<Shared>,
    worker: Option<JoinHandle<()>>,
    snapshot: Snapshot,
    pending: Pending,
    slots: Vec<Arc<Batch>>,
    next_slot: usize,
    drawn: u64,
    options: ScheduledOptions,
    views: Arc<ViewPool>,
    cancel: Cancellation,
    // Shared with views: their payload charges must survive writer destruction.
    reservation: Arc<Reservation>,
}
pub struct ScheduledView {
    base: Snapshot,
    buffer: Option<ViewBuffer>,
    pool: Arc<ViewPool>,
    _payloads: Arc<Reservation>,
}
struct ViewBuffer {
    pending: Vec<Arc<BatchChunk>>,
    _reservation: Reservation,
}
struct ViewPool {
    available: Mutex<Vec<ViewBuffer>>,
    budget: MemoryBudget,
    capacity: usize,
    retained: usize,
}
impl ViewPool {
    fn take(&self) -> Result<ViewBuffer> {
        if let Some(buffer) = lock(&self.available)?.pop() {
            return Ok(buffer);
        }
        let bytes = self
            .capacity
            .checked_mul(std::mem::size_of::<Arc<BatchChunk>>())
            .and_then(|n| n.checked_add(128))
            .ok_or(Error::SizeOverflow)?;
        let reservation = self.budget.reserve(bytes as u64)?;
        let mut pending = Vec::new();
        pending
            .try_reserve_exact(self.capacity)
            .map_err(|_| invalid("scheduled view allocation"))?;
        Ok(ViewBuffer {
            pending,
            _reservation: reservation,
        })
    }
}
impl ScheduledView {
    fn pending(&self) -> impl Iterator<Item = &Batch> {
        self.buffer
            .as_ref()
            .expect("live view buffer")
            .pending
            .iter()
            .flat_map(|chunk| chunk.groups.iter().map(Arc::as_ref))
    }
    fn pending_at_center(&self, center: [f64; 3]) -> impl Iterator<Item = &Batch> {
        let key = CenterFilter::key(center);
        self.buffer
            .as_ref()
            .expect("live view buffer")
            .pending
            .iter()
            .filter(move |chunk| chunk.centers.may_contain(&key))
            .flat_map(|chunk| chunk.groups.iter().map(Arc::as_ref))
    }
}
impl Drop for ScheduledView {
    fn drop(&mut self) {
        if let Some(mut buffer) = self.buffer.take() {
            buffer.pending.clear();
            let mut available = self
                .pool
                .available
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            if available.len() < self.pool.retained {
                available.push(buffer);
            }
        }
    }
}
impl ScheduledWriter {
    pub fn new(
        store: PackedStore,
        options: ScheduledOptions,
        budget: &MemoryBudget,
        cancel: &Cancellation,
    ) -> Result<Self> {
        Self::new_inner(store, options, budget, cancel, None, false)
    }
    /// Initialize the publication thread before accepting any groups. The
    /// constructor waits for the hook and returns its error (including a caught
    /// panic) after joining the worker. Clone the same caller-owned hook into
    /// execution options when placement must coordinate compute and publication.
    /// This does not configure the caller or other writers; see [`ThreadStart`].
    pub fn new_with_thread_start(
        store: PackedStore,
        options: ScheduledOptions,
        budget: &MemoryBudget,
        cancel: &Cancellation,
        thread_start: ThreadStart,
    ) -> Result<Self> {
        Self::new_inner(store, options, budget, cancel, Some(thread_start), false)
    }
    /// Enable one additional worker for cold-payload cache preparation.
    ///
    /// The helper has four owned payload slots, with buffers, metadata and
    /// decode scratch charged to the store's memory budget. Insufficient memory
    /// keeps publication serial; later budget pressure can retire the helper
    /// before retrying publication. Allocation, thread-start and callback errors
    /// still reject startup. Ordinary constructors keep serial preparation.
    ///
    /// The same caller-owned hook initializes [`ThreadRole::PublicationPrepare`]
    /// and [`ThreadRole::Publish`]. The constructor waits for both admitted
    /// threads to initialize. The caller must account for this extra worker in
    /// its CPU allowance; compute workers are not automatically reduced.
    /// The helper starts before the publisher's hook so it inherits the caller's
    /// allowed scheduling domain. No core or process affinity is selected here.
    pub fn new_with_publication_decode(
        store: PackedStore,
        options: ScheduledOptions,
        budget: &MemoryBudget,
        cancel: &Cancellation,
        thread_start: ThreadStart,
    ) -> Result<Self> {
        Self::new_inner(store, options, budget, cancel, Some(thread_start), true)
    }
    fn new_inner(
        store: PackedStore,
        options: ScheduledOptions,
        budget: &MemoryBudget,
        cancel: &Cancellation,
        thread_start: Option<ThreadStart>,
        decode: bool,
    ) -> Result<Self> {
        if options.maximum_group_observations == 0
            || options.queued_groups == 0
            || options.commit_groups == 0
            || options.commit_groups > options.queued_groups
            || options.retained_groups < options.queued_groups
        {
            return Err(invalid("scheduled writer capacities"));
        }
        let bytes = options
            .maximum_group_observations
            .checked_mul(std::mem::size_of::<Observation>() + std::mem::size_of::<(u64, f64)>())
            .and_then(|n| n.checked_add(256))
            .and_then(|n| n.checked_mul(options.retained_groups))
            // Two queue Arcs, a writer-batch Arc and a borrowed publication
            // descriptor per group; commit_groups cannot exceed queued_groups.
            .and_then(|n| n.checked_add(options.queued_groups.checked_mul(64)?))
            .and_then(|n| {
                n.checked_add(
                    options
                        .retained_groups
                        .checked_mul(std::mem::size_of::<ViewBuffer>())?,
                )
            })
            .ok_or(Error::SizeOverflow)?;
        let reservation = Arc::new(budget.reserve(bytes as u64)?);
        let chunk_groups = (1..=options.commit_groups.min(32))
            .rev()
            .find(|n| options.commit_groups % n == 0)
            .unwrap();
        let chunk_capacity = options.queued_groups.div_ceil(chunk_groups);
        let mut available = Vec::new();
        available
            .try_reserve_exact(options.retained_groups)
            .map_err(|_| invalid("scheduled view pool allocation"))?;
        let views = Arc::new(ViewPool {
            available: Mutex::new(available),
            budget: budget.clone(),
            capacity: chunk_capacity,
            retained: options.retained_groups,
        });
        let snapshot = store.snapshot()?;
        let drawn = snapshot.completed_group();
        let mut slots = Vec::new();
        slots
            .try_reserve_exact(options.retained_groups)
            .map_err(|_| invalid("scheduled slots allocation"))?;
        for _ in 0..options.retained_groups {
            let mut observations = Vec::new();
            observations
                .try_reserve_exact(options.maximum_group_observations)
                .map_err(|_| invalid("scheduled observations allocation"))?;
            let mut values = Vec::new();
            values
                .try_reserve_exact(options.maximum_group_observations)
                .map_err(|_| invalid("scheduled values allocation"))?;
            slots.push(Arc::new(Batch {
                rank: 0,
                observations,
                bounds: crate::spatial::hierarchy::Bounds::point([0.; 3]),
                values,
            }));
        }
        let mut queue = VecDeque::new();
        queue
            .try_reserve_exact(options.queued_groups)
            .map_err(|_| invalid("scheduled queue allocation"))?;
        let mut pending = VecDeque::new();
        pending
            .try_reserve_exact(chunk_capacity)
            .map_err(|_| invalid("scheduled pending allocation"))?;
        let shared = Arc::new(Shared {
            state: Mutex::new(State {
                queue,
                snapshot: snapshot.clone(),
                error: None,
                failed: false,
                starting: thread_start.is_some(),
                decode_statistics: PublicationDecodeStatistics::default(),
                flush: false,
                close: false,
            }),
            changed: Condvar::new(),
        });
        let writer_shared = shared.clone();
        let writer_cancel = cancel.clone();
        let wait_for_start = thread_start.is_some();
        let worker = std::thread::Builder::new()
            .name("terrustrial-publish".into())
            .spawn(move || {
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    // Spawn before narrowing the publisher's own affinity. The
                    // helper must inherit the caller's complete allowed domain.
                    let mut decoder = if decode {
                        publication_decode::Decoder::try_new(
                            &store.inner,
                            &writer_cancel,
                            thread_start.as_ref().expect("explicit decode placement"),
                        )?
                    } else {
                        None
                    };
                    if let Some(start) = thread_start.as_ref() {
                        start.run(ThreadInfo {
                            role: ThreadRole::Publish,
                            index: 0,
                            count: 1,
                        })?;
                    }
                    if thread_start.is_some() {
                        let mut state = lock(&writer_shared.state)?;
                        state.decode_statistics = decoder
                            .as_ref()
                            .map_or_else(Default::default, |d| d.statistics());
                        state.starting = false;
                        writer_shared.changed.notify_all();
                    }
                    writer_loop(
                        &store,
                        &writer_shared,
                        options,
                        &writer_cancel,
                        &mut decoder,
                    )
                }));
                let error = match result {
                    Ok(Ok(())) => None,
                    Ok(Err(e)) => Some(e),
                    Err(_) => Some(invalid("scheduled writer panicked")),
                };
                if let Some(error) = error {
                    let mut state = writer_shared
                        .state
                        .lock()
                        .unwrap_or_else(|e| e.into_inner());
                    state.failed = true;
                    state.decode_statistics = PublicationDecodeStatistics::default();
                    state.error = Some(error);
                    state.queue.clear();
                    writer_shared.changed.notify_all();
                }
            })?;
        let writer = Self {
            shared,
            worker: Some(worker),
            snapshot,
            pending: Pending {
                chunks: pending,
                count: 0,
                chunk_groups,
            },
            slots,
            next_slot: 0,
            drawn,
            options,
            views,
            cancel: cancel.clone(),
            reservation,
        };
        if wait_for_start {
            let mut state = lock(&writer.shared.state)?;
            while state.starting && !state.failed {
                cancel.check()?;
                state = writer
                    .shared
                    .changed
                    .wait_timeout(state, Duration::from_millis(10))
                    .map_err(|_| invalid("scheduled writer state poisoned"))?
                    .0;
            }
            if let Some(error) = state.error.take() {
                return Err(error);
            }
            if state.failed {
                return Err(invalid("scheduled writer failed during startup"));
            }
        }
        Ok(writer)
    }
    fn update(&mut self, wait: bool) -> Result<()> {
        let mut state = lock(&self.shared.state)?;
        loop {
            if let Some(error) = state.error.take() {
                return Err(error);
            }
            if state.failed {
                return Err(invalid("scheduled writer failed"));
            }
            self.snapshot = state.snapshot.clone();
            self.pending
                .discard_committed(self.snapshot.completed_group());
            if !wait || self.pending.count < self.options.queued_groups {
                return Ok(());
            }
            self.cancel.check()?;
            // Pending is the union of queued and in-flight groups. Because
            // queued_groups >= commit_groups, a full admission queue already
            // guarantees a full batch or an in-flight commit. Forcing flush
            // here would keep publishing smaller batches after pressure ends.
            let _span = Span::new(Metric::WriterBackpressure);
            state = self
                .shared
                .changed
                .wait_timeout(state, Duration::from_millis(10))
                .map_err(|_| invalid("scheduled writer state poisoned"))?
                .0;
        }
    }
    pub fn drawn_group(&self) -> u64 {
        self.drawn
    }
    /// Last sampled helper statistics; see [`PublicationDecodeStatistics`] for
    /// sampling, elapsed-time and serial-fallback semantics.
    pub fn publication_decode_statistics(&self) -> PublicationDecodeStatistics {
        self.shared
            .state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .decode_statistics
    }
    /// Last successfully committed prefix, including after a writer failure.
    pub fn last_durable_group(&self) -> u64 {
        self.shared
            .state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .snapshot
            .completed_group()
    }
    pub fn durable_group(&mut self) -> Result<u64> {
        self.update(false)?;
        Ok(self.snapshot.completed_group())
    }
    pub fn finish(&mut self) -> Result<()> {
        let _span = Span::new(Metric::WriterFinish);
        let mut state = lock(&self.shared.state)?;
        if state.snapshot.completed_group() != self.drawn {
            state.flush = true;
            self.shared.changed.notify_all();
        }
        // Check the publication predicate under the same lock used for waiting.
        // A commit between update() and reacquiring the lock otherwise loses its
        // notification and makes successful finalization wait for the timeout.
        while !state.failed && state.snapshot.completed_group() != self.drawn {
            self.cancel.check()?;
            state = self
                .shared
                .changed
                .wait_timeout(state, Duration::from_millis(10))
                .map_err(|_| invalid("scheduled writer state poisoned"))?
                .0;
        }
        drop(state);
        self.update(false)
    }
}
impl SnapshotStore for ScheduledWriter {
    type View = ScheduledView;
    fn deferred_page_reads(&self) -> bool {
        true
    }
    fn read_view(&mut self) -> Result<ScheduledView> {
        let _span = Span::new(Metric::Snapshot);
        self.update(false)?;
        let mut buffer = self.views.take()?;
        buffer.pending.extend(self.pending.chunks.iter().cloned());
        Ok(ScheduledView {
            base: self.snapshot.clone(),
            buffer: Some(buffer),
            pool: self.views.clone(),
            _payloads: self.reservation.clone(),
        })
    }

    fn append_group(&mut self, group: &Group, observations: &[Observation]) -> Result<()> {
        self.cancel.check()?;
        self.update(true)?;
        let rank = group.id.checked_add(1).ok_or(Error::SizeOverflow)?;
        if rank <= self.drawn
            || observations.is_empty()
            || observations.len() != group.targets.len()
            || observations.len() > self.options.maximum_group_observations
        {
            return Err(invalid("scheduled group order or length"));
        }
        for (target, observation) in group.targets.iter().zip(observations) {
            if target.id != observation.id() || target.site != *observation.site() {
                return Err(invalid("scheduled output does not match target"));
            }
        }
        // Probe counts without taking Arc's weak-reference lock for every
        // pinned slot. Rotate admission so successive groups usually inspect
        // one slot, even while older views retain a long prefix.
        let index = (self.next_slot..self.slots.len())
            .chain(0..self.next_slot)
            .find(|&i| Arc::strong_count(&self.slots[i]) == 1)
            .ok_or_else(|| invalid("scheduled payload slots pinned; increase retained_groups"))?;
        let slot = Arc::get_mut(&mut self.slots[index])
            .expect("exclusive scheduled slot without weak references");
        slot.rank = rank;
        slot.observations.clear();
        slot.values.clear();
        slot.observations.extend_from_slice(observations);
        slot.bounds = crate::spatial::hierarchy::Bounds::point(
            observations[0].site().support.center().coordinates(),
        );
        for observation in &observations[1..] {
            slot.bounds
                .include(crate::spatial::hierarchy::Bounds::point(
                    observation.site().support.center().coordinates(),
                ));
        }
        slot.values
            .extend(observations.iter().map(|o| (o.id(), o.value())));
        let batch = self.slots[index].clone();
        self.next_slot = (index + 1) % self.slots.len();
        let mut state = lock(&self.shared.state)?;
        if state.failed {
            return Err(state
                .error
                .take()
                .unwrap_or_else(|| invalid("scheduled writer failed")));
        }
        self.pending.append(batch.clone(), &self.views.budget)?;
        state.queue.push_back(batch);
        self.drawn = rank;
        // The writer cannot publish a partial batch until flush/finalization.
        // Waking it for every appended group only adds lock handoffs.
        if state.queue.len() >= self.options.commit_groups
            || state.flush
            || state.queue.len() % self.options.commit_groups.min(64) == 0
        {
            self.shared.changed.notify_one();
        }
        drop(state);
        if self.options.commit_groups == 1 {
            self.finish()?;
        }
        Ok(())
    }
    fn finish(&mut self) -> Result<()> {
        ScheduledWriter::finish(self)
    }
}
impl Drop for ScheduledWriter {
    fn drop(&mut self) {
        {
            let mut state = self.shared.state.lock().unwrap_or_else(|e| e.into_inner());
            state.close = true;
            state.flush = true;
            self.shared.changed.notify_all();
        }
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}
fn writer_loop(
    store: &PackedStore,
    shared: &Shared,
    options: ScheduledOptions,
    cancel: &Cancellation,
    decoder: &mut Option<publication_decode::Decoder>,
) -> Result<()> {
    let mut batch = Vec::new();
    batch
        .try_reserve_exact(options.commit_groups)
        .map_err(|_| invalid("writer batch allocation"))?;
    let mut prepared = updates::PreparedIdentities::new(
        options
            .commit_groups
            .saturating_mul(options.maximum_group_observations),
        &store.inner.budget,
    )
    .ok();
    let mut attempted = 0;
    let prepare_step = options.commit_groups.min(64);
    loop {
        let mut state = lock(&shared.state)?;
        while state.queue.len() < options.commit_groups && !state.flush && !state.close {
            cancel.check()?;
            let count = state.queue.len() / prepare_step * prepare_step;
            if count != 0 && state.queue[count - 1].rank > attempted {
                attempted = state.queue[count - 1].rank;
                if let Some(plan) = prepared.as_mut() {
                    drop(state);
                    // Staging errors do not replace mandatory publication
                    // validation. The transaction rechecks freshness.
                    let _ = store.prepare_identities(plan, attempted, cancel);
                    state = lock(&shared.state)?;
                    continue;
                }
            }
            state = shared
                .changed
                .wait_timeout(state, Duration::from_millis(10))
                .map_err(|_| invalid("scheduled writer poisoned"))?
                .0;
        }
        if state.queue.is_empty() {
            if state.close {
                return Ok(());
            }
            state.flush = false;
            continue;
        }
        while batch.len() < options.commit_groups {
            let Some(group) = state.queue.pop_front() else {
                break;
            };
            batch.push(group);
        }
        drop(state);
        let snapshot = {
            // Covered by the queue metadata reservation. Borrowed descriptors
            // live only until publication completes, before releasing batches.
            let mut slices = Vec::new();
            slices
                .try_reserve_exact(batch.len())
                .map_err(|_| invalid("writer publication descriptors allocation"))?;
            slices.extend(
                batch
                    .iter()
                    .map(|group| (group.rank, group.values.as_slice())),
            );
            let _span = Span::new(Metric::Publish);
            store.publish_staged_with_decode(&slices, cancel, &mut prepared, decoder)?
        };
        batch.clear();
        let mut state = lock(&shared.state)?;
        state.decode_statistics = decoder
            .as_ref()
            .map_or_else(Default::default, |d| d.statistics());
        state.snapshot = snapshot;
        if state.queue.is_empty() {
            state.flush = false;
        }
        shared.changed.notify_all();
    }
}
impl ObservationSource for ScheduledView {
    fn deferred_page_reads(&self) -> bool {
        true
    }
    fn scan(&self, visitor: &mut dyn FnMut(&Observation) -> Result<()>) -> Result<()> {
        self.base.scan(visitor)?;
        for group in self.pending() {
            for observation in &group.observations {
                visitor(observation)?;
            }
        }
        Ok(())
    }
    fn visit_bounds(
        &self,
        lower: [f64; 3],
        upper: [f64; 3],
        cancel: &Cancellation,
        visitor: &mut dyn FnMut(&Observation) -> Result<()>,
    ) -> Result<()> {
        self.base.visit_bounds(lower, upper, cancel, visitor)?;
        for group in self.pending() {
            cancel.check()?;
            if !group
                .bounds
                .intersects(crate::spatial::hierarchy::Bounds { lower, upper })
            {
                continue;
            }
            for observation in &group.observations {
                let p = observation.site().support.center().coordinates();
                if (0..3).all(|i| lower[i] <= p[i] && p[i] <= upper[i]) {
                    visitor(observation)?;
                }
            }
        }
        Ok(())
    }
    fn nearest_into<'a>(
        &self,
        point: Point3,
        workspace: &'a mut NearestWorkspace,
        cancel: &Cancellation,
    ) -> Result<&'a [Observation]> {
        if self
            .buffer
            .as_ref()
            .expect("live view buffer")
            .pending
            .is_empty()
        {
            return self.base.nearest_into(point, workspace, cancel);
        }
        self.base.nearest_into(point, workspace, cancel)?;
        workspace.resume();
        for group in self.pending() {
            cancel.check()?;
            if group.bounds.distance_squared(point.coordinates()) > workspace.cutoff_squared() {
                continue;
            }
            for o in &group.observations {
                workspace.offer(o);
            }
        }
        workspace.finish(cancel)
    }
    fn at_site(&self, site: &Site) -> Result<Vec<Observation>> {
        self.at_site_with_center_status(site)
            .map(|(found, _)| found)
    }
    fn at_site_with_center_status(&self, site: &Site) -> Result<(Vec<Observation>, bool)> {
        let (mut found, mut center_empty) = self.base.at_site_with_center_status(site)?;
        let center = site.support.center().coordinates();
        for group in self.pending_at_center(center) {
            if !group
                .bounds
                .contains(crate::spatial::hierarchy::Bounds::point(
                    site.support.center().coordinates(),
                ))
            {
                continue;
            }
            for observation in &group.observations {
                center_empty &= observation.site().support.center().coordinates() != center;
                if observation.site() == site {
                    found.push(*observation);
                }
            }
        }
        found.sort_unstable_by_key(Observation::id);
        Ok((found, center_empty))
    }
}

#[cfg(test)]
mod thread_start_tests {
    use super::*;
    use crate::data::Target;
    use std::sync::atomic::AtomicUsize;

    fn options() -> ScheduledOptions {
        ScheduledOptions {
            maximum_group_observations: 1,
            queued_groups: 2,
            commit_groups: 1,
            retained_groups: 4,
        }
    }
    fn fixture(path: &Path, budget: &MemoryBudget) -> (PackedStore, Observation) {
        let observation =
            Observation::new(0, Site::point(Point3::new([0.; 3]).unwrap(), 0), 0., 0.).unwrap();
        let mut builder = PackedBuilder::create(path, Options::default(), budget).unwrap();
        builder
            .push(observation, 1, &Cancellation::default())
            .unwrap();
        (
            builder.finish(&Cancellation::default()).unwrap(),
            observation,
        )
    }
    #[test]
    fn thread_start_writer_initializes_on_its_own_thread_before_returning() {
        let dir = tempfile::tempdir().unwrap();
        let budget = MemoryBudget::new(32 << 20);
        let (store, observation) = fixture(&dir.path().join("startup"), &budget);
        let calls = Arc::new(AtomicUsize::new(0));
        let recorded = calls.clone();
        let placement_lease = Arc::new(());
        let retained_lease = Arc::downgrade(&placement_lease);
        let caller = std::thread::current().id();
        let hook = ThreadStart::new(move |info| {
            assert_eq!(Arc::strong_count(&placement_lease), 1);
            assert_eq!(
                info,
                ThreadInfo {
                    role: ThreadRole::Publish,
                    index: 0,
                    count: 1
                }
            );
            assert_ne!(std::thread::current().id(), caller);
            recorded.fetch_add(1, Ordering::SeqCst);
            Ok(())
        });
        let cancel = Cancellation::default();
        let mut writer = ScheduledWriter::new_with_thread_start(
            store.clone(),
            options(),
            &budget,
            &cancel,
            hook,
        )
        .unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert!(retained_lease.upgrade().is_some());
        writer
            .append_group(
                &Group {
                    id: 0,
                    targets: vec![Target {
                        id: 0,
                        site: *observation.site(),
                    }],
                },
                &[observation],
            )
            .unwrap();
        writer.finish().unwrap();
        assert!(retained_lease.upgrade().is_some());
        assert_eq!(writer.durable_group().unwrap(), 1);
        drop(writer);
        assert!(retained_lease.upgrade().is_none());
        drop(store);
        assert_eq!(budget.used(), 0);
    }
    #[test]
    fn thread_start_writer_failure_or_panic_joins_and_releases_admission() {
        let dir = tempfile::tempdir().unwrap();
        let budget = MemoryBudget::new(32 << 20);
        let (store, _) = fixture(&dir.path().join("failure"), &budget);
        for panics in [false, true] {
            let before = budget.used();
            let hook = ThreadStart::new(move |_| {
                if panics {
                    panic!("injected writer startup panic");
                }
                Err(Error::Backend("injected writer startup failure".into()))
            });
            let result = ScheduledWriter::new_with_thread_start(
                store.clone(),
                options(),
                &budget,
                &Cancellation::default(),
                hook,
            );
            let error = match result {
                Err(error) => error.to_string(),
                Ok(_) => panic!("writer accepted failed startup"),
            };
            assert!(
                error.contains(if panics {
                    "startup hook panicked"
                } else {
                    "injected writer startup failure"
                }),
                "{error}"
            );
            assert_eq!(
                budget.used(),
                before,
                "failed constructor must join and release its buffers"
            );
            assert_eq!(store.snapshot().unwrap().completed_group(), 0);
        }
        drop(store);
        assert_eq!(budget.used(), 0);
    }
}

#[cfg(test)]
mod center_filter_tests {
    use super::*;
    use crate::data::{Support, Target};

    fn point(p: [f64; 3]) -> Point3 {
        Point3::new(p).unwrap()
    }
    fn batch(rank: u64, observations: Vec<Observation>) -> Arc<Batch> {
        let mut bounds = crate::spatial::hierarchy::Bounds::point(
            observations[0].site().support.center().coordinates(),
        );
        for o in &observations {
            bounds.include(crate::spatial::hierarchy::Bounds::point(
                o.site().support.center().coordinates(),
            ));
        }
        Arc::new(Batch {
            rank,
            observations,
            bounds,
            values: Vec::new(),
        })
    }
    fn observation(id: u64, p: [f64; 3], variable: usize) -> Observation {
        Observation::new(id, Site::point(point(p), variable), id as f64, 0.).unwrap()
    }

    #[test]
    fn center_filter_keeps_all_inserted_centers_and_signed_zero_aliases() {
        let mut observations = Vec::new();
        for i in 0..256 {
            let center = point([i as f64 * 0.17, (i % 29) as f64, -(i as f64)]);
            for variable in 0..7 {
                let support = if variable % 2 == 0 {
                    Support::point(center)
                } else {
                    Support::block(center, [0.2, 0.3, 0.], [2, 3, 1]).unwrap()
                };
                observations.push(
                    Observation::new(
                        observations.len() as u64,
                        Site { support, variable },
                        1.,
                        0.,
                    )
                    .unwrap(),
                );
            }
        }
        for p in [
            [-0., 0., -0.],
            [1e300, -1e300, 1e-300],
            [-1e-300, 1e-300, -1e300],
        ] {
            observations.push(observation(observations.len() as u64, p, 0));
        }
        let mut filter = CenterFilter::default();
        filter.include(&observations);
        assert!(filter.0[32..].iter().any(|&word| word != 0));
        for o in &observations {
            assert!(
                filter.may_contain(&CenterFilter::key(o.site().support.center().coordinates()))
            );
        }
        for p in [[0., -0., 0.], [-0., -0., -0.], [0.; 3]] {
            assert!(filter.may_contain(&CenterFilter::key(p)));
        }
        // Check useful negative rejection, without making false positives an
        // error or depending on an exact Bloom-filter probability.
        let rejected = (0..2048)
            .filter(|i| !filter.may_contain(&CenterFilter::key([1e6 + *i as f64, 9., 19.])))
            .count();
        assert!(rejected > 1536);
    }

    #[test]
    fn center_filter_chunk_copies_keep_history_and_charge_each_live_version() {
        let groups = 32;
        let charge = (groups * std::mem::size_of::<Arc<Batch>>()
            + std::mem::size_of::<BatchChunk>()
            + 64) as u64;
        let budget = MemoryBudget::new(charge * 2);
        let mut pending = Pending {
            chunks: VecDeque::new(),
            count: 0,
            chunk_groups: groups,
        };
        let first = batch(1, vec![observation(1, [-0., 0., 0.], 0)]);
        pending.append(first, &budget).unwrap();
        assert_eq!(budget.used(), charge);
        let old = pending.chunks[0].clone();
        let next_center = (1..1000)
            .map(|i| [i as f64, 7., 13.])
            .find(|p| !old.centers.may_contain(&CenterFilter::key(*p)))
            .unwrap();
        pending
            .append(batch(2, vec![observation(2, next_center, 1)]), &budget)
            .unwrap();
        assert_eq!(budget.used(), charge * 2);
        assert_eq!(old.groups.len(), 1);
        assert!(!old.centers.may_contain(&CenterFilter::key(next_center)));
        assert!(
            pending.chunks[0]
                .centers
                .may_contain(&CenterFilter::key(next_center))
        );
        let second = pending.chunks[0].clone();
        assert!(matches!(
            pending.append(batch(3, vec![observation(3, [4., 5., 6.], 0)]), &budget),
            Err(Error::MemoryBudget { .. })
        ));
        assert_eq!(pending.count, 2);
        assert_eq!(pending.chunks[0].groups.len(), 2);
        assert_eq!(budget.used(), charge * 2);
        drop(old);
        pending
            .append(batch(3, vec![observation(3, [4., 5., 6.], 0)]), &budget)
            .unwrap();
        assert_eq!(second.groups.len(), 2);
        assert_eq!(budget.used(), charge * 2);
        drop(second);
        pending
            .append(batch(4, vec![observation(4, [9., 8., 7.], 0)]), &budget)
            .unwrap();
        assert_eq!(
            budget.used(),
            charge,
            "exclusive tail extends without a new charge"
        );
        for group in &pending.chunks[0].groups {
            for o in &group.observations {
                assert!(
                    pending.chunks[0]
                        .centers
                        .may_contain(&CenterFilter::key(o.site().support.center().coordinates()))
                );
            }
        }
        drop(pending);
        assert_eq!(budget.used(), 0);
    }

    fn assert_exact_matches_scan(view: &impl ObservationSource, queries: &[Site]) {
        for site in queries {
            let mut expected = Vec::new();
            let mut expected_center_empty = true;
            view.scan(&mut |o| {
                expected_center_empty &=
                    o.site().support.center().coordinates() != site.support.center().coordinates();
                if o.site() == site {
                    expected.push(*o);
                }
                Ok(())
            })
            .unwrap();
            expected.sort_unstable_by_key(Observation::id);
            let actual = view.at_site(site).unwrap();
            assert_eq!(actual.len(), expected.len());
            assert!(actual.iter().zip(&expected).all(|(a, b)| a.same_bits(b)));
            let (with_status, center_empty) = view.at_site_with_center_status(site).unwrap();
            assert_eq!(with_status.len(), expected.len());
            assert!(
                with_status
                    .iter()
                    .zip(&expected)
                    .all(|(a, b)| a.same_bits(b))
            );
            assert_eq!(center_empty, expected_center_empty, "site={site:?}");
        }
    }

    #[test]
    fn filtered_exact_sites_match_scan_across_supports_commits_and_old_views() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("filtered-exact-sites");
        let budget = MemoryBudget::new(16 << 20);
        let cancel = Cancellation::default();
        let options = Options {
            leaf_records: 16,
            build_window_leaves: 8,
            cached_pages: 32,
            readers: 2,
            directory_pages: 8,
            sqlite_cache_bytes: 16384,
            ..Options::default()
        };
        let groups: Vec<Vec<Observation>> = (0..8)
            .map(|g| {
                let center = point(if g % 2 == 0 {
                    [-0., 0., -0.]
                } else {
                    [0., -0., 0.]
                });
                [
                    Site::point(center, 0),
                    Site::point(center, 1),
                    Site {
                        support: Support::block(center, [0.1, 0.2, 0.], [2, 3, 1]).unwrap(),
                        variable: 0,
                    },
                    Site::point(point([10. + g as f64, 3., 4.]), g as usize % 2),
                ]
                .into_iter()
                .enumerate()
                .map(|(i, site)| {
                    Observation::new(
                        1000 - g * 4 - i as u64,
                        site,
                        if i == 0 {
                            -0.
                        } else {
                            (g * 4 + i as u64) as f64
                        },
                        0.01,
                    )
                    .unwrap()
                })
                .collect()
            })
            .collect();
        let mut builder = PackedBuilder::create(&path, options, &budget).unwrap();
        for (rank, group) in groups.iter().enumerate().rev() {
            for o in group {
                builder.push(*o, rank as u64 + 1, &cancel).unwrap();
            }
        }
        let store = builder.finish(&cancel).unwrap();
        let mut writer = ScheduledWriter::new(
            store.clone(),
            ScheduledOptions {
                maximum_group_observations: 4,
                queued_groups: 8,
                commit_groups: 8,
                retained_groups: 32,
            },
            &budget,
            &cancel,
        )
        .unwrap();
        let mut history = Vec::new();
        for (rank, observations) in groups.iter().enumerate() {
            let group = Group {
                id: rank as u64,
                targets: observations
                    .iter()
                    .map(|o| Target {
                        id: o.id(),
                        site: *o.site(),
                    })
                    .collect(),
            };
            writer.append_group(&group, observations).unwrap();
            if rank == 1 || rank == 3 {
                history.push(writer.read_view().unwrap());
            }
        }
        writer.finish().unwrap();
        assert_eq!(writer.durable_group().unwrap(), 8);
        let current = writer.read_view().unwrap();
        drop(writer);
        let mut queries: Vec<Site> = groups.iter().flatten().map(|o| *o.site()).collect();
        queries.extend([
            Site::point(point([0.; 3]), 0),
            Site::point(point([0.; 3]), 99),
            Site::point(point([1e100, -1e100, 3.]), 0),
            Site {
                support: Support::block(point([0.; 3]), [0.2, 0.1, 0.], [3, 2, 1]).unwrap(),
                variable: 0,
            },
        ]);
        for (i, view) in history.iter().enumerate() {
            let mut count = 0;
            view.scan(&mut |_| {
                count += 1;
                Ok(())
            })
            .unwrap();
            assert_eq!(count, (i + 1) * 8);
            assert_exact_matches_scan(view, &queries);
        }
        assert_exact_matches_scan(&current, &queries);
        drop((history, current, store));
        assert_eq!(budget.used(), 0);
        let reopened = PackedStore::open(&path, options, &budget).unwrap();
        let snapshot = reopened.snapshot().unwrap();
        assert_eq!(snapshot.completed_group(), 8);
        assert_exact_matches_scan(&snapshot, &queries);
        drop((snapshot, reopened));
        assert_eq!(budget.used(), 0);
    }
}
