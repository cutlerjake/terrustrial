//! Experimental all-plan simulation. Retains the entire path and all plans in
//! the supplied memory budget; it does not spill plans or implement streaming.
//! Geometry selection and Gaussian preparation precede all dependency-ready
//! draws. Inspection and durable publication remain in original path order.
//!
//! This intentionally uses a resident geometry index even with a packed output
//! store. Memory grows with the whole path, selected observations, graph edges,
//! and retained numerical plans. Insufficient budget returns an error before
//! publication; there is no automatic fallback. Phase timings exclude final
//! state destruction, which callers must include in end-to-end measurements.
//! Numerical preparations start largest-first; ReadyOrder only orders draws.
use super::{
    simulation::{prepare_selected, select_simulation, Prepared},
    *,
};
use crate::execution::Reservation;
use std::{
    cmp::Reverse,
    collections::BinaryHeap,
    sync::{Condvar, Mutex, OnceLock},
    time::{Duration, Instant},
};

#[derive(Clone, Copy, Debug)]
pub enum ReadyOrder {
    /// Original path rank among dependency-ready draws.
    Path,
    /// Increasing estimated draw work, (conditioning + targets) * targets.
    SmallFirst,
    /// Decreasing estimated draw work; not a critical-path priority.
    LargeFirst,
}

#[derive(Debug, Default)]
pub struct Statistics {
    pub geometry_seconds: f64,
    pub selection_seconds: f64,
    pub preparation_seconds: f64,
    pub graph_seconds: f64,
    pub drawing_seconds: f64,
    pub publication_seconds: f64,
    pub groups: usize,
    pub dependency_edges: usize,
    pub dependency_depth: usize,
    pub initial_ready: usize,
    pub maximum_ready: usize,
    pub retained_plan_bytes: u64,
    /// Shared logical usage, including source/store caches and reserved workers.
    /// This is a barrier sample, not RSS or a measured allocation high-water mark.
    pub budget_bytes_at_preparation_barrier: u64,
}

struct Node<M: Mode> {
    rank: usize,
    prepared: Prepared<M>,
    predecessors: Vec<usize>,
    draws: Vec<Observation>,
    _edges: Option<Reservation>,
}

fn allocate<T>(capacity: usize) -> Result<Vec<T>> {
    let mut v = Vec::new();
    v.try_reserve_exact(capacity)
        .map_err(|_| invalid("two-phase allocation"))?;
    Ok(v)
}

fn parallel<W: Send, T: Send>(
    workers: Vec<W>,
    jobs: Vec<T>,
    work: impl Fn(&mut W, T) + Sync,
) -> Result<Vec<W>> {
    let count = workers.len();
    let jobs = Mutex::new(jobs.into_iter());
    std::thread::scope(|scope| {
        let mut handles = allocate(count)?;
        for mut worker in workers {
            let jobs = &jobs;
            let work = &work;
            handles.push(std::thread::Builder::new().spawn_scoped(scope, move || {
                loop {
                    let job = jobs.lock().unwrap_or_else(|e| e.into_inner()).next();
                    let Some(job) = job else { break };
                    work(&mut worker, job);
                }
                worker
            })?);
        }
        let mut returned = allocate(count)?;
        for handle in handles {
            match handle.join() {
                Ok(worker) => returned.push(worker),
                Err(panic) => std::panic::resume_unwind(panic),
            }
        }
        Ok(returned)
    })
}

struct DrawState<M: Mode> {
    nodes: Vec<Option<Node<M>>>,
    ready: BinaryHeap<Reverse<(usize, usize)>>,
    unresolved: Vec<usize>,
    remaining: usize,
    maximum_ready: usize,
    stop: bool,
    error: Option<Error>,
}
struct DrawShared<M: Mode> {
    state: Mutex<DrawState<M>>,
    changed: Condvar,
}
struct Stop<'a, M: Mode>(&'a DrawShared<M>);
impl<M: Mode> Drop for Stop<'_, M> {
    fn drop(&mut self) {
        self.0.state.lock().unwrap_or_else(|e| e.into_inner()).stop = true;
        self.0.changed.notify_all();
    }
}

/// Prototype counterpart of unchecked rolling simulation. All groups are
/// caller-owned; their geometry and retained execution state are charged here.
/// Unlike rolling admission, `batch_groups` only caps worker count here.
/// `bytes_per_group` caps each worker arena; retained selections and plans are
/// charged directly to the shared parent budget for their entire lifetime.
#[allow(clippy::too_many_arguments)]
pub fn simulate<C: Covariance, D: Drift, S: SnapshotStore, H: ObservationSource + Sync + ?Sized>(
    model: &C,
    hard: &H,
    simulated: &mut S,
    groups: &[Group],
    drift: &D,
    options: SimulationOptions<'_>,
    execution: ExecutionOptions,
    order: ReadyOrder,
    output: impl FnMut(&[Observation]) -> Result<()>,
) -> Result<Statistics> {
    run::<Unchecked, _, _, _, _>(
        model,
        hard,
        simulated,
        groups,
        drift,
        options,
        execution,
        order,
        |_, _| Ok(()),
        output,
    )
}

/// Numerical inspection happens before each ordered write. No output is
/// published until all independent preparations and private draws succeed.
#[allow(clippy::too_many_arguments)]
pub fn simulate_with_residuals<
    C: Covariance,
    D: Drift,
    S: SnapshotStore,
    H: ObservationSource + Sync + ?Sized,
>(
    model: &C,
    hard: &H,
    simulated: &mut S,
    groups: &[Group],
    drift: &D,
    options: SimulationOptions<'_>,
    execution: ExecutionOptions,
    order: ReadyOrder,
    mut inspect: impl FnMut(&Group, &GroupResiduals) -> Result<()>,
    output: impl FnMut(&[Observation]) -> Result<()>,
) -> Result<Statistics> {
    run::<diagnostics::Checked, _, _, _, _>(
        model,
        hard,
        simulated,
        groups,
        drift,
        options,
        execution,
        order,
        |group, report| inspect(group, report.as_ref().expect("checked plan")),
        output,
    )
}

#[allow(clippy::too_many_arguments)]
fn run<M: Mode, C: Covariance, D: Drift, S: SnapshotStore, H: ObservationSource + Sync + ?Sized>(
    model: &C,
    hard: &H,
    simulated: &mut S,
    groups: &[Group],
    drift: &D,
    options: SimulationOptions<'_>,
    execution: ExecutionOptions,
    order: ReadyOrder,
    mut inspect: impl FnMut(&Group, &M::Report) -> Result<()>,
    mut output: impl FnMut(&[Observation]) -> Result<()>,
) -> Result<Statistics> {
    let start = Instant::now();
    options.validate()?;
    if execution.batch_groups == 0
        || execution.bytes_per_group == 0
        || execution.maximum_targets_per_group == 0
    {
        return Err(invalid("two-phase execution limits"));
    }
    let count = groups.len();
    if count == 0 {
        return Ok(Statistics::default());
    }
    let worker_count = execution.worker_count().min(count);
    let targets = groups.iter().try_fold(0usize, |n, g| {
        n.checked_add(g.targets.len()).ok_or(Error::SizeOverflow)
    })?;
    // Conservative charge includes input geometry, Window hierarchy/scratch,
    // identity table rounding, value cells, draws, graph headers and job lists.
    // Plans and selected observations have their own exact logical reservations.
    let metadata_bytes = checked_len(&[targets, 1024])?
        .checked_add(checked_len(&[
            count,
            1024 + std::mem::size_of::<Node<M>>(),
        ])?)
        .ok_or(Error::SizeOverflow)?;
    let _metadata = options.budget.reserve(metadata_bytes as u64)?;
    let _worker_memory = options.budget.reserve(
        (worker_count as u64)
            .checked_mul(execution.bytes_per_group)
            .ok_or(Error::SizeOverflow)?,
    )?;
    let mut previous = None;
    for group in groups {
        options.cancellation.check()?;
        validate_order(group.id, &mut previous)?;
        if group.targets.is_empty() || group.targets.len() > execution.maximum_targets_per_group {
            return Err(invalid("two-phase target limit"));
        }
    }
    let mut window = Window::empty();
    window.rebuild(groups)?;
    let base = simulated.read_view()?;
    let hard = Checked {
        source: hard,
        ids: &window.bindings.ids,
    };
    let base = Checked {
        source: &base,
        ids: &window.bindings.ids,
    };
    let mut owners = allocate(targets)?;
    let mut values = allocate(targets)?;
    for (rank, group) in groups.iter().enumerate() {
        for _ in &group.targets {
            owners.push(rank);
            values.push(OnceLock::<f64>::new());
        }
    }
    let mut nodes = allocate(count)?;
    for (rank, group) in groups.iter().enumerate() {
        nodes.push(Node::<M> {
            rank,
            prepared: Prepared::new(options.budget)?,
            predecessors: Vec::new(),
            draws: allocate(group.targets.len())?,
            _edges: None,
        });
    }
    let workers = workers_n::<M>(
        &execution,
        options.hard_neighborhood,
        options.simulated_neighborhood,
        worker_count,
    )?;
    let mut stats = Statistics {
        groups: count,
        geometry_seconds: start.elapsed().as_secs_f64(),
        ..Default::default()
    };

    let start = Instant::now();
    let mut jobs = allocate(count)?;
    jobs.extend(nodes.iter_mut());
    let workers = parallel(workers, jobs, |worker, node| {
        let source = Overlay {
            base: &base,
            window: &window,
            rank: node.rank,
        };
        node.prepared.status = select_simulation(
            &hard,
            &source,
            &groups[node.rank],
            &options,
            worker,
            &mut node.prepared,
        );
    })?;
    for node in &mut nodes {
        std::mem::replace(&mut node.prepared.status, Ok(()))?;
    }
    stats.selection_seconds = start.elapsed().as_secs_f64();

    let start = Instant::now();
    let mut jobs = allocate(count)?;
    jobs.extend(nodes.iter_mut());
    // Selection is already known: launch larger factorizations early to avoid
    // deliberately leaving expensive independent solves until the tail.
    jobs.sort_unstable_by_key(|n| {
        Reverse((
            n.prepared.selected.observations.len(),
            groups[n.rank].targets.len(),
        ))
    });
    let workers = parallel(workers, jobs, |worker, node| {
        node.prepared.status = prepare_selected(
            model,
            drift,
            &groups[node.rank],
            &options,
            worker,
            &mut node.prepared,
        );
    })?;
    for node in &mut nodes {
        std::mem::replace(&mut node.prepared.status, Ok(()))?;
    }
    stats.preparation_seconds = start.elapsed().as_secs_f64();
    for node in &nodes {
        let weights = node.prepared.plan.get()?.weights();
        let entries = checked_len(&[weights.nrows(), weights.ncols()])?
            .checked_add(checked_len(&[weights.ncols(), weights.ncols(), 2])?)
            .ok_or(Error::SizeOverflow)?;
        stats.retained_plan_bytes = stats
            .retained_plan_bytes
            .checked_add(checked_len(&[entries, 8])? as u64)
            .ok_or(Error::SizeOverflow)?;
    }
    stats.budget_bytes_at_preparation_barrier = options.budget.used();

    let start = Instant::now();
    let mut successor_counts = allocate(count)?;
    successor_counts.resize(count, 0usize);
    let mut depth = allocate(count)?;
    depth.resize(count, 1usize);
    let mut keys = allocate(count)?;
    for node in &mut nodes {
        let observations = &node.prepared.selected.observations;
        // Each raw selected ID, including exact aliases, remains a consistency
        // dependency even if its final prediction weight is zero.
        node._edges = Some(
            options
                .budget
                .reserve(checked_len(&[observations.len(), 32])? as u64)?,
        );
        node.predecessors = allocate(observations.len())?;
        for o in observations {
            if let Some(&index) = window.bindings.ids.get(&o.id()) {
                let predecessor = owners[index];
                if predecessor >= node.rank {
                    return Err(invalid("two-phase future dependency"));
                }
                node.predecessors.push(predecessor);
            }
        }
        node.predecessors.sort_unstable();
        node.predecessors.dedup();
        for &pred in &node.predecessors {
            successor_counts[pred] += 1;
            depth[node.rank] = depth[node.rank].max(depth[pred] + 1);
        }
        stats.dependency_edges += node.predecessors.len();
        let cost = checked_len(&[
            observations.len() + groups[node.rank].targets.len(),
            groups[node.rank].targets.len(),
        ])?;
        keys.push(match order {
            ReadyOrder::Path => node.rank,
            ReadyOrder::SmallFirst => cost,
            ReadyOrder::LargeFirst => usize::MAX - cost,
        });
    }
    stats.dependency_depth = *depth.iter().max().unwrap();
    let mut successors = allocate(count)?;
    for n in successor_counts {
        successors.push(allocate(n)?);
    }
    let mut unresolved = allocate(count)?;
    let mut ready = BinaryHeap::new();
    ready
        .try_reserve_exact(count)
        .map_err(|_| invalid("two-phase ready allocation"))?;
    for node in &nodes {
        unresolved.push(node.predecessors.len());
        for &pred in &node.predecessors {
            successors[pred].push(node.rank);
        }
        if node.predecessors.is_empty() {
            ready.push(Reverse((keys[node.rank], node.rank)));
        }
    }
    stats.initial_ready = ready.len();
    let mut retained = allocate(count)?;
    retained.extend(nodes.into_iter().map(Some));
    let shared = DrawShared {
        state: Mutex::new(DrawState {
            nodes: retained,
            ready,
            unresolved,
            remaining: count,
            maximum_ready: stats.initial_ready,
            stop: false,
            error: None,
        }),
        changed: Condvar::new(),
    };
    stats.graph_seconds = start.elapsed().as_secs_f64();

    let start = Instant::now();
    std::thread::scope(|scope| -> Result<()> {
        let _stop = Stop(&shared);
        for mut worker in workers {
            let mut drawing = DrawingWorkspace::new(&worker.budget)?;
            let shared = &shared;
            let values = &values;
            let ids = &window.bindings.ids;
            let successors = &successors;
            let keys = &keys;
            let options = &options;
            std::thread::Builder::new().spawn_scoped(scope, move || {
                let _stop = Stop(shared);
                loop {
                    let mut state = shared.state.lock().unwrap_or_else(|e| e.into_inner());
                    let mut node = loop {
                        if state.stop || state.remaining == 0 {
                            return;
                        }
                        if let Err(error) = options.cancellation.check() {
                            state.error = Some(error);
                            return;
                        }
                        if let Some(Reverse((_, rank))) = state.ready.pop() {
                            break state.nodes[rank].take().unwrap();
                        }
                        state = shared
                            .changed
                            .wait_timeout(state, Duration::from_millis(10))
                            .unwrap_or_else(|e| e.into_inner())
                            .0;
                    };
                    drop(state);
                    let result = (|| -> Result<()> {
                        let group = &groups[node.rank];
                        for o in &mut node.prepared.selected.observations {
                            if let Some(&index) = ids.get(&o.id()) {
                                let value = *values[index]
                                    .get()
                                    .ok_or_else(|| invalid("two-phase unresolved value"))?;
                                *o =
                                    Observation::new(o.id(), *o.site(), value, o.noise_variance())?;
                            }
                        }
                        let plan = node.prepared.plan.get()?;
                        if node.prepared.selected.observations.len() != plan.weights().nrows() {
                            worker.searches.canonical.compact(
                                &mut node.prepared.selected.observations,
                                true,
                                options.cancellation,
                            )?;
                        }
                        let result = drawing.draw(
                            plan,
                            &node.prepared.selected.observations,
                            group,
                            options,
                        )?;
                        options.cancellation.check()?;
                        node.draws.extend_from_slice(result);
                        for o in result {
                            values[ids[&o.id()]]
                                .set(o.value())
                                .map_err(|_| invalid("two-phase duplicate draw"))?;
                        }
                        Ok(())
                    })();
                    let mut state = shared.state.lock().unwrap_or_else(|e| e.into_inner());
                    if let Err(error) = result {
                        state.error = Some(error);
                        return;
                    }
                    let rank = node.rank;
                    state.nodes[rank] = Some(node);
                    state.remaining -= 1;
                    let mut released = 0;
                    for &next in &successors[rank] {
                        state.unresolved[next] -= 1;
                        if state.unresolved[next] == 0 {
                            state.ready.push(Reverse((keys[next], next)));
                            released += 1;
                        }
                    }
                    state.maximum_ready = state.maximum_ready.max(state.ready.len());
                    if state.remaining == 0 {
                        shared.changed.notify_all();
                    } else {
                        for _ in 0..released {
                            shared.changed.notify_one();
                        }
                    }
                }
            })?;
        }
        // Keep the stop guard alive while the workers complete normally.
        let mut state = shared.state.lock().unwrap_or_else(|e| e.into_inner());
        while !state.stop && state.remaining != 0 {
            state = shared
                .changed
                .wait_timeout(state, Duration::from_millis(10))
                .unwrap_or_else(|e| e.into_inner())
                .0;
        }
        Ok(())
    })?;
    stats.drawing_seconds = start.elapsed().as_secs_f64();
    let mut state = shared.state.into_inner().unwrap_or_else(|e| e.into_inner());
    if let Some(error) = state.error.take() {
        return Err(error);
    }
    if state.remaining != 0 {
        return Err(invalid("two-phase incomplete DAG"));
    }
    stats.maximum_ready = state.maximum_ready;

    let start = Instant::now();
    let published = (|| -> Result<()> {
        for (group, node) in groups.iter().zip(&state.nodes) {
            let node = node.as_ref().expect("completed draw");
            inspect(group, &node.prepared.report)?;
            options.cancellation.check()?;
            simulated.append_group(group, &node.draws)?;
            output(&node.draws)?;
        }
        Ok(())
    })();
    let finished = simulated.finish();
    published.and(finished)?;
    stats.publication_seconds = start.elapsed().as_secs_f64();
    Ok(stats)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        data::{Point3, Target},
        gaussian::Simple,
        model::{Anisotropy, Coregionalization, Correlation, Lmc, Structure},
        spatial::{Neighborhood, SpatialIndex},
    };

    fn model() -> Lmc {
        Lmc::new(vec![Structure {
            correlation: Correlation::Exponential,
            anisotropy: Anisotropy::isotropic(2.).unwrap(),
            coregionalization: Coregionalization::scalar(1.).unwrap(),
        }])
        .unwrap()
    }
    fn site(x: f64) -> Site {
        Site::point(Point3::new([x, 0., 0.]).unwrap(), 0)
    }
    fn groups() -> Vec<Group> {
        (0..24)
            .map(|i| Group {
                id: i * 3,
                targets: (0..2)
                    .map(|j| Target {
                        id: 100 + i * 2 + j,
                        // Repeated exact locations exercise alias binding and validation.
                        site: site(if j == 0 && i % 4 == 0 {
                            0.375
                        } else {
                            i as f64 * 0.125 + j as f64 * 0.03125
                        }),
                    })
                    .collect(),
            })
            .collect()
    }
    fn hard() -> Vec<Observation> {
        vec![
            Observation::new(1, site(-1.), 0.2, 0.01).unwrap(),
            Observation::new(2, site(4.), -0.4, 0.01).unwrap(),
        ]
    }
    fn options<'a>(budget: &'a MemoryBudget, cancel: &'a Cancellation) -> SimulationOptions<'a> {
        SimulationOptions {
            seed: 371,
            realization: 3,
            mean: None,
            hard_neighborhood: Neighborhood {
                maximum: 3,
                radius: 100.,
            }
            .into(),
            simulated_neighborhood: Neighborhood {
                maximum: 3,
                radius: 100.,
            }
            .into(),
            budget,
            cancellation: cancel,
        }
    }
    fn execution() -> ExecutionOptions {
        ExecutionOptions {
            batch_groups: 4,
            maximum_targets_per_group: 2,
            bytes_per_group: 1 << 20,
        }
    }
    fn bits(os: &[Observation]) -> Vec<(u64, u64)> {
        os.iter().map(|o| (o.id(), o.value().to_bits())).collect()
    }
    #[test]
    fn all_plan_orders_preserve_path_rng_exact_aliases_and_release_budget() {
        let m = model();
        let h = hard();
        let groups = groups();
        let budget = MemoryBudget::new(1 << 28);
        let cancel = Cancellation::default();
        let mut reference = Vec::new();
        let mut store = SpatialIndex::new(vec![]).unwrap();
        simulate_pipelined(
            &m,
            h.as_slice(),
            &mut store,
            groups.clone(),
            &Simple,
            options(&budget, &cancel),
            execution(),
            |os| {
                reference.extend(bits(os));
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(budget.used(), 0);
        for threads in [1, 2, 4] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            for order in [
                ReadyOrder::Path,
                ReadyOrder::SmallFirst,
                ReadyOrder::LargeFirst,
            ] {
                let mut store = SpatialIndex::new(vec![]).unwrap();
                let mut output = Vec::new();
                let stats = pool
                    .install(|| {
                        simulate(
                            &m,
                            h.as_slice(),
                            &mut store,
                            &groups,
                            &Simple,
                            options(&budget, &cancel),
                            execution(),
                            order,
                            |os| {
                                output.extend(bits(os));
                                Ok(())
                            },
                        )
                    })
                    .unwrap();
                assert_eq!(output, reference);
                assert!(stats.dependency_edges > 0);
                assert!(stats.dependency_depth > 1);
                assert!(stats.retained_plan_bytes > 0);
                assert_eq!(budget.used(), 0);
            }
        }
    }

    struct Sink {
        index: SpatialIndex,
        writes: usize,
        finishes: usize,
    }
    impl SnapshotStore for Sink {
        type View = crate::spatial::ResidentSnapshot;
        fn read_view(&mut self) -> Result<Self::View> {
            self.index.snapshot()
        }
        fn append_group(&mut self, _: &Group, os: &[Observation]) -> Result<()> {
            self.writes += 1;
            for o in os {
                self.index.append(*o)?;
            }
            Ok(())
        }
        fn finish(&mut self) -> Result<()> {
            self.finishes += 1;
            Ok(())
        }
    }
    fn sink() -> Sink {
        Sink {
            index: SpatialIndex::new(vec![]).unwrap(),
            writes: 0,
            finishes: 0,
        }
    }
    #[test]
    fn inspection_and_output_failure_preserve_publication_boundary_and_drain() {
        let m = model();
        let h = hard();
        let groups = groups();
        let budget = MemoryBudget::new(1 << 28);
        let cancel = Cancellation::default();
        let mut store = sink();
        let mut inspected = Vec::new();
        let mut emitted = 0;
        let error = simulate_with_residuals(
            &m,
            h.as_slice(),
            &mut store,
            &groups,
            &Simple,
            options(&budget, &cancel),
            execution(),
            ReadyOrder::SmallFirst,
            |g, _| {
                inspected.push(g.id);
                if g.id == 6 {
                    Err(Error::Cancelled)
                } else {
                    Ok(())
                }
            },
            |_| {
                emitted += 1;
                Ok(())
            },
        )
        .unwrap_err();
        assert!(matches!(error, Error::Cancelled));
        assert_eq!(inspected, [0, 3, 6]);
        assert_eq!(emitted, 2);
        assert_eq!(store.writes, 2);
        assert_eq!(store.finishes, 1);
        assert_eq!(budget.used(), 0);
        let mut store = sink();
        let error = simulate(
            &m,
            h.as_slice(),
            &mut store,
            &groups,
            &Simple,
            options(&budget, &cancel),
            execution(),
            ReadyOrder::LargeFirst,
            |_| Err(Error::Cancelled),
        )
        .unwrap_err();
        assert!(matches!(error, Error::Cancelled));
        assert_eq!(store.writes, 1);
        assert_eq!(store.finishes, 1);
        assert_eq!(budget.used(), 0);
    }
    #[test]
    fn budget_and_cancellation_fail_before_publication() {
        let m = model();
        let h = hard();
        let groups = groups();
        let budget = MemoryBudget::new(1024);
        let cancel = Cancellation::default();
        let mut store = sink();
        let error = simulate(
            &m,
            h.as_slice(),
            &mut store,
            &groups,
            &Simple,
            options(&budget, &cancel),
            execution(),
            ReadyOrder::Path,
            |_| Ok(()),
        )
        .unwrap_err();
        assert!(matches!(error, Error::MemoryBudget { .. }));
        assert_eq!(store.writes, 0);
        assert_eq!(budget.used(), 0);
        let budget = MemoryBudget::new(1 << 28);
        let mut store = sink();
        cancel.cancel();
        let error = simulate(
            &m,
            h.as_slice(),
            &mut store,
            &groups,
            &Simple,
            options(&budget, &cancel),
            execution(),
            ReadyOrder::Path,
            |_| Ok(()),
        )
        .unwrap_err();
        assert!(matches!(error, Error::Cancelled));
        assert_eq!(store.writes, 0);
        assert_eq!(budget.used(), 0);
    }
}
