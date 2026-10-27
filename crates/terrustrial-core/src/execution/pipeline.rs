//! Bounded dynamic preparation with ordered consumption on the caller's thread.
//!
//! A job owns its result capacity before admission. Workers own their numerical
//! arenas; queues transfer ownership, never hold a lock while preparing or
//! consuming. Every returned slot can admit another job immediately.
use super::{
    parallel::validate_order,
    profiling::{Metric, Span, Stamp},
    Cancellation, ExecutionOptions, ThreadInfo, ThreadRole, ThreadStart,
};
use crate::{data::Group, error::invalid, Result};
use std::{
    collections::VecDeque,
    sync::{Condvar, Mutex},
    time::Duration,
};

struct Job<T> {
    slot: usize,
    group: Group,
    value: T,
    queued: Stamp,
}
struct State<T> {
    jobs: VecDeque<Job<T>>,
    blocking: VecDeque<Job<T>>,
    selected: VecDeque<Job<T>>,
    waiting: Vec<Option<Job<T>>>,
    ready: Vec<Option<Job<T>>>,
    positions: VecDeque<(u64, usize)>,
    drawn: Vec<bool>,
    unresolved: Vec<usize>,
    successors: Vec<Vec<usize>>,
    runnable: VecDeque<usize>,
    starting: usize,
    startup_error: Option<crate::Error>,
    stop: bool,
}
fn job_ready<T>(state: &mut State<T>, slot: usize, waited: bool) {
    let job = state.waiting[slot].as_mut().expect("waiting job");
    if waited {
        job.queued.record(Metric::DependencyWait);
    }
    job.queued = Stamp::now();
    state.runnable.push_back(slot);
}
struct Shared<T> {
    state: Mutex<State<T>>,
    compute_changed: Condvar,
    io_changed: Condvar,
    consumed: Condvar,
}
impl<T> Shared<T> {
    fn stop(&self) {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if state.stop {
            return;
        }
        state.stop = true;
        drop(state);
        self.compute_changed.notify_all();
        self.io_changed.notify_all();
        self.consumed.notify_all();
    }
}
struct StopOnDrop<'a, T>(&'a Shared<T>);
impl<T> Drop for StopOnDrop<'_, T> {
    fn drop(&mut self) {
        self.0.stop();
    }
}
struct JoinStarted<'a>(&'a std::cell::Cell<Stamp>);
impl Drop for JoinStarted<'_> {
    fn drop(&mut self) {
        self.0.set(Stamp::now());
    }
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn run<W: Send, T: Send>(
    workers: Vec<W>,
    slots: Vec<T>,
    groups: impl IntoIterator<Item = Group>,
    execution: &ExecutionOptions,
    cancel: &Cancellation,
    prepare: impl Fn(&Group, &mut W, &mut T) + Sync,
    mut consume: impl FnMut(Group, &mut T) -> Result<()>,
) -> Result<()> {
    run_with(
        workers,
        slots,
        groups,
        execution,
        cancel,
        prepare,
        |event| match event {
            Event::Admit(_, _) => Ok(()),
            Event::Consume(group, value) => consume(group, value),
        },
    )
}

pub(crate) enum Event<'a, T> {
    Admit(&'a Group, &'a mut T),
    Consume(Group, &'a mut T),
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_with<W: Send, T: Send>(
    workers: Vec<W>,
    slots: Vec<T>,
    groups: impl IntoIterator<Item = Group>,
    execution: &ExecutionOptions,
    cancel: &Cancellation,
    prepare: impl Fn(&Group, &mut W, &mut T) + Sync,
    mut coordinate: impl FnMut(Event<'_, T>) -> Result<()>,
) -> Result<()> {
    run_staged(
        workers,
        Vec::new(),
        slots,
        groups,
        execution,
        None,
        cancel,
        |group, worker, value, _| {
            prepare(group, worker, value);
            Progress::Complete
        },
        0,
        |_| &[],
        &mut coordinate,
    )
}

#[derive(Clone, Copy)]
pub(crate) enum Stage {
    Prepare,
    BlockingPrepare,
    Factor,
    Draw,
}
pub(crate) enum Progress {
    Waiting,
    Blocking,
    Selected,
    Drawn,
    Complete,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run_staged<W: Send, T: Send>(
    workers: Vec<W>,
    io_workers: Vec<W>,
    slots: Vec<T>,
    groups: impl IntoIterator<Item = Group>,
    execution: &ExecutionOptions,
    thread_start: Option<&ThreadStart>,
    cancel: &Cancellation,
    prepare: impl Fn(&Group, &mut W, &mut T, Stage) -> Progress + Sync,
    maximum_dependencies: usize,
    dependencies: impl Fn(&T) -> &[u64] + Sync,
    mut coordinate: impl FnMut(Event<'_, T>) -> Result<()>,
) -> Result<()> {
    if workers.is_empty() || slots.is_empty() {
        return Err(invalid("empty execution pipeline"));
    }
    let compute_count = workers.len();
    let io_count = io_workers.len();
    let setup = Span::new(Metric::PipelineSetup);
    let capacity = slots.len();
    let mut jobs = VecDeque::new();
    jobs.try_reserve_exact(capacity)
        .map_err(|_| invalid("pipeline queue allocation"))?;
    let mut blocking = VecDeque::new();
    blocking
        .try_reserve_exact(capacity)
        .map_err(|_| invalid("pipeline blocking queue allocation"))?;
    let mut selected = VecDeque::new();
    selected
        .try_reserve_exact(capacity)
        .map_err(|_| invalid("pipeline selected queue allocation"))?;
    let mut ready = Vec::new();
    ready
        .try_reserve_exact(capacity)
        .map_err(|_| invalid("pipeline result allocation"))?;
    ready.resize_with(capacity, || None);
    let mut waiting = Vec::new();
    waiting
        .try_reserve_exact(capacity)
        .map_err(|_| invalid("pipeline dependencies allocation"))?;
    waiting.resize_with(capacity, || None);
    let mut groups = groups.into_iter().fuse();
    let mut previous = None;
    let mut next = || -> Result<Option<Group>> {
        let _span = Span::new(Metric::GroupInput);
        cancel.check()?;
        let Some(group) = groups.next() else {
            return Ok(None);
        };
        validate_order(group.id, &mut previous)?;
        if group.targets.is_empty() || group.targets.len() > execution.maximum_targets_per_group {
            return Err(invalid("parallel group target limit"));
        }
        Ok(Some(group))
    };
    for (slot, mut value) in slots.into_iter().enumerate() {
        let Some(group) = next()? else {
            break;
        };
        coordinate(Event::Admit(&group, &mut value))?;
        jobs.push_back(Job {
            slot,
            group,
            value,
            queued: Stamp::now(),
        });
    }
    let mut remaining = jobs.len();
    if remaining == 0 {
        return Ok(());
    }
    let mut positions = VecDeque::new();
    positions
        .try_reserve(capacity)
        .map_err(|_| invalid("pipeline identity allocation"))?;
    for job in &jobs {
        positions.push_back((job.group.id, job.slot));
    }
    let mut successors = Vec::new();
    successors
        .try_reserve_exact(capacity)
        .map_err(|_| invalid("pipeline successor allocation"))?;
    for _ in 0..capacity {
        let mut edges = Vec::new();
        edges
            .try_reserve_exact(if maximum_dependencies == 0 {
                0
            } else {
                capacity
            })
            .map_err(|_| invalid("pipeline edge allocation"))?;
        successors.push(edges);
    }
    let mut runnable = VecDeque::new();
    runnable
        .try_reserve_exact(capacity)
        .map_err(|_| invalid("pipeline ready queue allocation"))?;
    let shared = Shared {
        state: Mutex::new(State {
            jobs,
            blocking,
            selected,
            ready,
            waiting,
            positions,
            drawn: vec![false; capacity],
            unresolved: vec![0; capacity],
            successors,
            runnable,
            starting: if thread_start.is_some() {
                compute_count + io_count
            } else {
                0
            },
            startup_error: None,
            stop: false,
        }),
        compute_changed: Condvar::new(),
        io_changed: Condvar::new(),
        consumed: Condvar::new(),
    };
    // Some platforms lazily allocate their native mutex and condition variable.
    // Initialize both on the coordinator before workers race to use them; this
    // avoids first-query allocation and duplicate initialization under contention.
    drop(shared.state.lock().unwrap_or_else(|e| e.into_inner()));
    shared.compute_changed.notify_all();
    shared.io_changed.notify_all();
    shared.consumed.notify_all();
    let joining = std::cell::Cell::new(Stamp::default());
    let result = std::thread::scope(|scope| -> Result<()> {
        let _join_started = JoinStarted(&joining);
        // Drop before the scope joins threads, including callback panics and
        // partial thread-creation failure. No waiter outlives the driver.
        let _stop = StopOnDrop(&shared);
        let has_io = !io_workers.is_empty();
        for (index, (mut worker, blocking_worker)) in workers
            .into_iter()
            .map(|w| (w, false))
            .chain(io_workers.into_iter().map(|w| (w, true)))
            .enumerate()
        {
            let shared = &shared;
            let prepare = &prepare;
            let dependencies = &dependencies;
            std::thread::Builder::new()
                .name(format!(
                    "terrustrial-{}-{index}",
                    if blocking_worker { "read" } else { "compute" }
                ))
                .spawn_scoped(scope, move || {
                    // A panicking preparation wakes the coordinator before the
                    // scope propagates the panic; it cannot leave a missing slot.
                    let _stop = StopOnDrop(shared);
                    if let Some(start) = thread_start {
                        let info = ThreadInfo {
                            role: if blocking_worker {
                                ThreadRole::BlockingRead
                            } else {
                                ThreadRole::Compute
                            },
                            index: if blocking_worker {
                                index - compute_count
                            } else {
                                index
                            },
                            count: if blocking_worker {
                                io_count
                            } else {
                                compute_count
                            },
                        };
                        // No preparation can begin until every initializer has
                        // succeeded. Failed creation/startup wakes this gate via
                        // StopOnDrop, so partial startup cannot strand peers.
                        let result = start.run(info);
                        let mut state = shared.state.lock().unwrap_or_else(|e| e.into_inner());
                        if let Err(error) = result {
                            if state.startup_error.is_none() {
                                state.startup_error = Some(error);
                            }
                            drop(state);
                            shared.stop();
                            return;
                        }
                        state.starting -= 1;
                        if state.starting == 0 {
                            shared.consumed.notify_all();
                        }
                        while state.starting != 0 && !state.stop {
                            if cancel.check().is_err() {
                                return;
                            }
                            state = shared
                                .consumed
                                .wait_timeout(state, Duration::from_millis(10))
                                .unwrap_or_else(|e| e.into_inner())
                                .0;
                        }
                        if state.stop {
                            return;
                        }
                    }
                    let mut local_draw = None;
                    let lock_started = Stamp::now();
                    let mut state = shared.state.lock().unwrap_or_else(|e| e.into_inner());
                    lock_started.record(Metric::PipelineLock);
                    loop {
                        let mut drawing_local = false;
                        let (mut job, stage) = loop {
                            if state.stop || cancel.check().is_err() {
                                return;
                            }
                            if blocking_worker {
                                if let Some(job) = state.blocking.pop_front() {
                                    break (job, Stage::BlockingPrepare);
                                }
                            } else {
                                // Kahn's ready queue: completing a draw wakes only
                                // its direct successors. No full-window polling.
                                if let Some(slot) = state.runnable.pop_front() {
                                    break (state.waiting[slot].take().unwrap(), Stage::Draw);
                                }
                                if let Some(job) = local_draw.take() {
                                    drawing_local = true;
                                    break (job, Stage::Draw);
                                }
                                if let Some(job) = state.selected.pop_front() {
                                    break (job, Stage::Factor);
                                }
                                if let Some(job) = state.jobs.pop_front() {
                                    break (job, Stage::Prepare);
                                }
                            }
                            let idle = Stamp::now();
                            let changed = if blocking_worker {
                                &shared.io_changed
                            } else {
                                &shared.compute_changed
                            };
                            state = changed
                                .wait_timeout(state, Duration::from_millis(10))
                                .unwrap_or_else(|e| e.into_inner())
                                .0;
                            idle.record(if blocking_worker {
                                Metric::IoIdle
                            } else {
                                Metric::ComputeIdle
                            });
                        };
                        drop(state);
                        job.queued.record(match stage {
                            Stage::Prepare => Metric::PrepareQueue,
                            Stage::BlockingPrepare => Metric::IoQueue,
                            Stage::Factor => Metric::FactorQueue,
                            Stage::Draw => Metric::DrawQueue,
                        });
                        let active = Stamp::now();
                        let progress = prepare(&job.group, &mut worker, &mut job.value, stage);
                        active.record(match stage {
                            Stage::Prepare => Metric::Prepare,
                            Stage::BlockingPrepare => Metric::BlockingPrepare,
                            Stage::Factor => Metric::Factor,
                            Stage::Draw => Metric::Draw,
                        });
                        if drawing_local {
                            active.record(Metric::LocalDraw);
                        }
                        let lock_started = Stamp::now();
                        state = shared.state.lock().unwrap_or_else(|e| e.into_inner());
                        lock_started.record(Metric::PipelineLock);
                        if state.stop {
                            return;
                        }
                        job.queued = Stamp::now();
                        let slot = job.slot;
                        debug_assert!(state.ready[slot].is_none());
                        let runnable_before = state.runnable.len();
                        match progress {
                            Progress::Blocking => {
                                assert!(
                                    has_io && !blocking_worker,
                                    "disk fallback requires a blocking service"
                                );
                                state.blocking.push_back(job);
                                shared.io_changed.notify_one();
                            }
                            Progress::Selected => {
                                state.selected.push_back(job);
                                shared.compute_changed.notify_one();
                            }
                            Progress::Waiting => {
                                let edges = dependencies(&job.value);
                                assert!(edges.len() <= maximum_dependencies);
                                for &id in edges {
                                    assert!(
                                        id < job.group.id,
                                        "simulation edges must follow the path"
                                    );
                                    if let Ok(position) =
                                        state.positions.binary_search_by_key(&id, |&(id, _)| id)
                                    {
                                        let predecessor = state.positions[position].1;
                                        if !state.drawn[predecessor] {
                                            state.successors[predecessor].push(slot);
                                            state.unresolved[slot] += 1;
                                        }
                                    }
                                    // An older group absent from the bounded
                                    // graph has already been consumed/published.
                                }
                                state.waiting[slot] = Some(job);
                                if state.unresolved[slot] == 0 {
                                    // Keep freshly prepared matrices on this
                                    // compute worker when no older draw is ready.
                                    // Only ready work is retained; dependency
                                    // waits always release the worker. A newly
                                    // queued draw still takes priority above.
                                    if !blocking_worker && state.runnable.is_empty() {
                                        debug_assert!(local_draw.is_none());
                                        local_draw = state.waiting[slot].take();
                                    } else {
                                        job_ready(&mut state, slot, false);
                                    }
                                }
                            }
                            Progress::Drawn => {
                                state.drawn[slot] = true;
                                while let Some(successor) = state.successors[slot].pop() {
                                    state.unresolved[successor] -= 1;
                                    if state.unresolved[successor] == 0 {
                                        job_ready(&mut state, successor, true);
                                    }
                                }
                                state.ready[slot] = Some(job);
                                shared.consumed.notify_one();
                            }
                            // An unsuccessful draw is consumable in path order
                            // for error handling, but never releases dependents.
                            Progress::Complete => {
                                state.ready[slot] = Some(job);
                                shared.consumed.notify_one();
                            }
                        }
                        // Separate predicates keep an I/O waiter from stealing a
                        // compute wakeup. Wake only as many compute workers as
                        // newly runnable draws; dependency-only changes need none.
                        for _ in runnable_before..state.runnable.len() {
                            shared.compute_changed.notify_one();
                        }
                        // Select the next stage while still holding this brief
                        // queue lock. Numerical work above always drops it;
                        // an idle worker releases it in the condition wait.
                    }
                })?;
        }
        if thread_start.is_some() {
            let mut state = shared.state.lock().unwrap_or_else(|e| e.into_inner());
            while state.starting != 0 && !state.stop {
                cancel.check()?;
                state = shared
                    .consumed
                    .wait_timeout(state, Duration::from_millis(10))
                    .unwrap_or_else(|e| e.into_inner())
                    .0;
            }
            if let Some(error) = state.startup_error.take() {
                return Err(error);
            }
            if state.stop {
                cancel.check()?;
                return Err(invalid("preparation worker stopped during startup"));
            }
        }
        drop(setup);
        let mut slot = 0;
        while remaining != 0 {
            let lock_started = Stamp::now();
            let mut state = shared.state.lock().unwrap_or_else(|e| e.into_inner());
            lock_started.record(Metric::PipelineLock);
            let ordered_wait = Stamp::now();
            let mut job = loop {
                cancel.check()?;
                if state.stop {
                    return Err(invalid("preparation worker stopped"));
                }
                if let Some(job) = state.ready[slot].take() {
                    let removed = state.positions.pop_front();
                    debug_assert_eq!(removed, Some((job.group.id, slot)));
                    break job;
                }
                state = shared
                    .consumed
                    .wait_timeout(state, Duration::from_millis(10))
                    .unwrap_or_else(|e| e.into_inner())
                    .0;
            };
            drop(state);
            ordered_wait.record(Metric::OrderedWait);
            {
                let _span = Span::new(Metric::Consume);
                coordinate(Event::Consume(job.group, &mut job.value))?;
            }
            remaining -= 1;
            if let Some(group) = next()? {
                {
                    let _span = Span::new(Metric::Admit);
                    coordinate(Event::Admit(&group, &mut job.value))?;
                }
                job.group = group;
                let lock_started = Stamp::now();
                let mut state = shared.state.lock().unwrap_or_else(|e| e.into_inner());
                lock_started.record(Metric::PipelineLock);
                state.drawn[slot] = false;
                debug_assert_eq!(state.unresolved[slot], 0);
                debug_assert!(state.successors[slot].is_empty());
                state.positions.push_back((job.group.id, slot));
                job.queued = Stamp::now();
                state.jobs.push_back(job);
                remaining += 1;
                shared.compute_changed.notify_one();
            }
            slot = (slot + 1) % capacity;
        }
        Ok(())
    });
    joining.get().record(Metric::PipelineJoin);
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        data::{Point3, Site, Target},
        Error,
    };
    use std::sync::mpsc;
    fn groups(n: u64) -> impl Iterator<Item = Group> {
        (0..n).map(|id| Group {
            id,
            targets: vec![Target {
                id,
                site: Site::point(Point3::new([id as f64, 0., 0.]).unwrap(), 0),
            }],
        })
    }
    fn options() -> ExecutionOptions {
        ExecutionOptions {
            batch_groups: 3,
            maximum_targets_per_group: 1,
            bytes_per_group: 4096,
        }
    }
    #[test]
    fn thread_start_reports_roles_and_gates_work_until_all_workers_start() {
        use std::sync::Arc;
        let starts = Arc::new(Mutex::new(Vec::new()));
        let captured = starts.clone();
        let caller = std::thread::current().id();
        let start = ThreadStart::new(move |info| {
            assert_ne!(std::thread::current().id(), caller);
            captured.lock().unwrap().push(info);
            Ok(())
        });
        let mut output = Vec::new();
        run_staged(
            vec![(); 2],
            vec![()],
            vec![(); 3],
            groups(7),
            &options(),
            Some(&start),
            &Cancellation::default(),
            |_, _, _, stage| {
                assert_eq!(starts.lock().unwrap().len(), 3);
                match stage {
                    Stage::Prepare => Progress::Blocking,
                    Stage::BlockingPrepare => Progress::Selected,
                    Stage::Factor => Progress::Complete,
                    Stage::Draw => unreachable!(),
                }
            },
            0,
            |_| &[],
            |event| {
                if let Event::Consume(group, _) = event {
                    output.push(group.id);
                }
                Ok(())
            },
        )
        .unwrap();
        let starts = starts.lock().unwrap();
        assert_eq!(starts.len(), 3);
        for info in [
            ThreadInfo {
                role: ThreadRole::Compute,
                index: 0,
                count: 2,
            },
            ThreadInfo {
                role: ThreadRole::Compute,
                index: 1,
                count: 2,
            },
            ThreadInfo {
                role: ThreadRole::BlockingRead,
                index: 0,
                count: 1,
            },
        ] {
            assert_eq!(starts.iter().filter(|&&actual| actual == info).count(), 1);
        }
        assert_eq!(output, (0..7).collect::<Vec<_>>());
    }
    #[test]
    fn thread_start_failure_and_panic_join_all_workers_without_preparation_or_output() {
        use std::sync::{
            atomic::{AtomicUsize, Ordering},
            Arc,
        };
        struct Worker(Arc<AtomicUsize>);
        impl Drop for Worker {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }
        for role in [ThreadRole::Compute, ThreadRole::BlockingRead] {
            for panics in [false, true] {
                let dropped = Arc::new(AtomicUsize::new(0));
                let start = ThreadStart::new(move |info| {
                    if info.role == role && info.index == 1 {
                        if panics {
                            panic!("injected startup panic");
                        }
                        return Err(Error::Backend("injected startup failure".into()));
                    }
                    Ok(())
                });
                let result = run_staged(
                    (0..2).map(|_| Worker(dropped.clone())).collect(),
                    (0..2).map(|_| Worker(dropped.clone())).collect(),
                    vec![(); 3],
                    groups(7),
                    &options(),
                    Some(&start),
                    &Cancellation::default(),
                    |_, _, _, _| panic!("failed startup must not prepare"),
                    0,
                    |_| &[],
                    |event| {
                        assert!(
                            !matches!(event, Event::Consume(_, _)),
                            "failed startup emitted output"
                        );
                        Ok(())
                    },
                );
                let error = result.unwrap_err().to_string();
                assert!(
                    error.contains(if panics {
                        "startup hook panicked"
                    } else {
                        "injected startup failure"
                    }),
                    "{error}"
                );
                assert_eq!(
                    dropped.load(Ordering::SeqCst),
                    4,
                    "all workers must join before returning"
                );
            }
        }
    }
    #[test]
    fn thread_start_is_owned_by_each_concurrent_invocation() {
        use std::sync::{Arc, Barrier};
        let barrier = Arc::new(Barrier::new(4));
        let seen = [
            Arc::new(Mutex::new(Vec::new())),
            Arc::new(Mutex::new(Vec::new())),
        ];
        std::thread::scope(|scope| {
            let mut handles = Vec::new();
            for seen in &seen {
                let captured = seen.clone();
                let barrier = barrier.clone();
                handles.push(scope.spawn(move || {
                    let start = ThreadStart::new(move |info| {
                        captured.lock().unwrap().push(info);
                        barrier.wait();
                        Ok(())
                    });
                    run_staged(
                        vec![(); 2],
                        Vec::new(),
                        vec![(); 3],
                        groups(3),
                        &options(),
                        Some(&start),
                        &Cancellation::default(),
                        |_, _, _, _| Progress::Complete,
                        0,
                        |_| &[],
                        |_| Ok(()),
                    )
                }));
            }
            for handle in handles {
                handle.join().unwrap().unwrap();
            }
        });
        for seen in &seen {
            let seen = seen.lock().unwrap();
            assert_eq!(seen.len(), 2);
            assert!(seen
                .iter()
                .all(|info| info.role == ThreadRole::Compute && info.count == 2));
            assert_ne!(seen[0].index, seen[1].index);
        }
    }
    #[test]
    fn thread_start_public_estimation_and_simulation_reject_before_output() {
        use crate::{
            data::Observation,
            estimation::{estimate_groups, EstimationOptions},
            execution::MemoryBudget,
            gaussian::{NumericalTolerance, Simple, Uncertainty},
            model::{Anisotropy, Coregionalization, Correlation, Lmc, Structure},
            simulation::{parallel::simulate_pipelined, SimulationOptions},
            spatial::{Neighborhood, SpatialIndex},
        };
        use std::sync::{
            atomic::{AtomicUsize, Ordering},
            Arc,
        };
        let model = Lmc::new(vec![Structure {
            correlation: Correlation::Exponential,
            anisotropy: Anisotropy::isotropic(2.).unwrap(),
            coregionalization: Coregionalization::scalar(1.).unwrap(),
        }])
        .unwrap();
        let hard = SpatialIndex::new(vec![Observation::new(
            100,
            Site::point(Point3::new([-1., 0., 0.]).unwrap(), 0),
            1.,
            0.,
        )
        .unwrap()])
        .unwrap();
        let mut simulated = SpatialIndex::new(Vec::new()).unwrap();
        let budget = MemoryBudget::new(16 << 20);
        let cancel = Cancellation::default();
        let query = Neighborhood {
            maximum: 1,
            radius: 10.,
        };
        let limits = ExecutionOptions {
            bytes_per_group: 1 << 20,
            ..options()
        };
        let calls = Arc::new(AtomicUsize::new(0));
        let recorded = calls.clone();
        let hook = ThreadStart::new(move |info| {
            assert_eq!(
                info,
                ThreadInfo {
                    role: ThreadRole::Compute,
                    index: 0,
                    count: 1
                }
            );
            recorded.fetch_add(1, Ordering::SeqCst);
            Err(Error::Backend("public driver startup rejected".into()))
        });
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        pool.install(|| {
            let result = estimate_groups(
                &model,
                &hard,
                groups(1),
                &Simple,
                EstimationOptions {
                    neighborhood: query.into(),
                    mean: None,
                    uncertainty: Uncertainty::None,
                    tolerance: NumericalTolerance::default(),
                    budget: &budget,
                    cancellation: &cancel,
                },
                limits.with_thread_start(hook.clone()),
                |_| panic!("failed estimation startup emitted output"),
            );
            assert!(result
                .unwrap_err()
                .to_string()
                .contains("public driver startup rejected"));
            let result = simulate_pipelined(
                &model,
                &hard,
                &mut simulated,
                groups(1),
                &Simple,
                SimulationOptions {
                    seed: 1,
                    realization: 0,
                    mean: None,
                    hard_neighborhood: query.into(),
                    simulated_neighborhood: query.into(),
                    budget: &budget,
                    cancellation: &cancel,
                },
                limits.with_thread_start(hook),
                |_| panic!("failed simulation startup emitted output"),
            );
            assert!(result
                .unwrap_err()
                .to_string()
                .contains("public driver startup rejected"));
        });
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        assert!(simulated.is_empty());
        assert_eq!(budget.used(), 0);
    }
    #[test]
    fn replaces_consumed_slot_before_slow_original_batch_finishes() {
        let (release, wait) = mpsc::channel();
        let wait = Mutex::new(wait);
        let (later, later_rx) = mpsc::channel();
        let mut output = Vec::new();
        run(
            vec![0; 2],
            vec![0; 3],
            groups(9),
            &options(),
            &Cancellation::default(),
            |group, _, result| {
                if group.id == 2 {
                    wait.lock()
                        .unwrap()
                        .recv_timeout(Duration::from_secs(2))
                        .unwrap();
                }
                if group.id == 3 {
                    later.send(()).unwrap();
                }
                *result = group.id;
            },
            |group, result| {
                if group.id == 1 {
                    // Group 3 was admitted as soon as group 0 was consumed,
                    // despite group 2 still preparing. A batch join deadlocks.
                    later_rx.recv_timeout(Duration::from_secs(2)).unwrap();
                    release.send(()).unwrap();
                }
                assert_eq!(group.id, *result);
                output.push(group.id);
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(output, (0..9).collect::<Vec<_>>());
    }
    #[test]
    fn callback_failure_stops_successor_output_and_joins_workers() {
        let mut output = Vec::new();
        let error = run(
            vec![(); 4],
            vec![(); 3],
            groups(20),
            &options(),
            &Cancellation::default(),
            |_, _, _| {},
            |group, _| {
                if group.id == 2 {
                    return Err(Error::Cancelled);
                }
                output.push(group.id);
                Ok(())
            },
        )
        .unwrap_err();
        assert!(matches!(error, Error::Cancelled));
        assert_eq!(output, [0, 1]);
    }
    #[test]
    fn single_worker_short_and_empty_inputs() {
        for n in [0, 1, 2, 3, 8] {
            let mut count = 0;
            run(
                vec![()],
                vec![(); 3],
                groups(n),
                &options(),
                &Cancellation::default(),
                |_, _, _| {},
                |group, _| {
                    assert_eq!(group.id, count);
                    count += 1;
                    Ok(())
                },
            )
            .unwrap();
            assert_eq!(count, n);
        }
    }
    #[test]
    fn worker_panic_wakes_coordinator() {
        let result = std::panic::catch_unwind(|| {
            run(
                vec![()],
                vec![(); 3],
                groups(3),
                &options(),
                &Cancellation::default(),
                |_, _, _| panic!("injected preparation panic"),
                |_, _| Ok(()),
            )
        });
        assert!(result.is_err());
    }

    #[test]
    fn topological_draws_overtake_unrelated_work_and_preserve_output_order() {
        let (release, wait) = mpsc::channel();
        let wait = Mutex::new(wait);
        let draws = Mutex::new(Vec::new());
        let mut outputs = Vec::new();
        run_staged(
            vec![(); 2],
            Vec::new(),
            vec![Vec::<u64>::new(); 3],
            groups(7),
            &options(),
            None,
            &Cancellation::default(),
            |group, _, dependencies, stage| match stage {
                Stage::Prepare | Stage::BlockingPrepare | Stage::Factor => {
                    dependencies.clear();
                    dependencies.extend_from_slice(match group.id {
                        1 => &[0],
                        3 => &[2],
                        4 => &[1, 3],
                        5 => &[4],
                        6 => &[0],
                        _ => &[],
                    });
                    Progress::Waiting
                }
                Stage::Draw => {
                    if group.id == 0 {
                        wait.lock()
                            .unwrap()
                            .recv_timeout(Duration::from_secs(2))
                            .unwrap();
                    }
                    draws.lock().unwrap().push(group.id);
                    if group.id == 2 {
                        release.send(()).unwrap();
                    }
                    Progress::Drawn
                }
            },
            2,
            |dependencies| dependencies,
            |event| {
                if let Event::Consume(group, _) = event {
                    outputs.push(group.id);
                }
                Ok(())
            },
        )
        .unwrap();
        let draws = draws.into_inner().unwrap();
        assert_eq!(draws[0], 2, "independent draw overtakes stalled draw zero");
        let position = |id| draws.iter().position(|&actual| actual == id).unwrap();
        for (from, to) in [(0, 1), (2, 3), (1, 4), (3, 4), (4, 5), (0, 6)] {
            assert!(position(from) < position(to));
        }
        assert_eq!(outputs, (0..7).collect::<Vec<_>>());
    }

    #[test]
    fn immediately_ready_draw_stays_on_its_preparing_worker() {
        let mut consumed = 0;
        run_staged(
            vec![(); 2],
            Vec::new(),
            vec![None],
            groups(64),
            &options(),
            None,
            &Cancellation::default(),
            |_, _, owner, stage| match stage {
                Stage::Prepare => {
                    *owner = Some(std::thread::current().id());
                    Progress::Waiting
                }
                Stage::Draw => {
                    // With one admitted slot no older ready draw can interrupt
                    // this continuation, even when another worker is idle.
                    assert_eq!(*owner, Some(std::thread::current().id()));
                    Progress::Drawn
                }
                _ => unreachable!(),
            },
            0,
            |_| &[],
            |event| {
                if let Event::Consume(group, _) = event {
                    assert_eq!(group.id, consumed);
                    consumed += 1;
                }
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(consumed, 64);
    }

    #[test]
    fn a_stalled_blocking_job_does_not_occupy_the_compute_worker() {
        let (started, started_rx) = mpsc::channel();
        let started_rx = Mutex::new(started_rx);
        let (release, release_rx) = mpsc::channel();
        let release_rx = Mutex::new(release_rx);
        let mut output = Vec::new();
        run_staged(
            vec![()],
            vec![()],
            vec![(); 3],
            groups(3),
            &options(),
            None,
            &Cancellation::default(),
            |group, _, _, stage| match (group.id, stage) {
                (0, Stage::Prepare) => Progress::Blocking,
                (0, Stage::BlockingPrepare) => {
                    assert!(std::thread::current().name().unwrap().contains("read"));
                    started.send(()).unwrap();
                    release_rx
                        .lock()
                        .unwrap()
                        .recv_timeout(Duration::from_secs(2))
                        .unwrap();
                    Progress::Selected
                }
                (0, Stage::Factor) => {
                    assert!(std::thread::current().name().unwrap().contains("compute"));
                    Progress::Complete
                }
                (1, Stage::Prepare) => {
                    // This compute job runs while the blocking service is
                    // occupied by group zero. Neither consumes the other's slot.
                    started_rx
                        .lock()
                        .unwrap()
                        .recv_timeout(Duration::from_secs(2))
                        .unwrap();
                    release.send(()).unwrap();
                    Progress::Complete
                }
                _ => Progress::Complete,
            },
            0,
            |_| &[],
            |event| {
                if let Event::Consume(group, _) = event {
                    output.push(group.id);
                }
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(output, [0, 1, 2]);
    }
}
