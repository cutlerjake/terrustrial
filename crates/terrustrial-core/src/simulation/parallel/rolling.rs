//! Rolling preparation over immutable source views and bounded predecessor slots.
use super::simulation::{prepare_selected, select_simulation, Prepared};
use super::*;
use crate::{
    execution::{
        pipeline::{self, Event, Progress, Stage},
        ExecutionConfig,
    },
    spatial::{NearestWorkspace, ResidentSnapshot, SpatialIndex},
};
use foldhash::{HashMap, HashMapExt, HashSet, HashSetExt};
use std::{
    collections::VecDeque,
    sync::{Arc, OnceLock},
};

/// A writer that can hand preparation an immutable owned read view. A view must
/// remain unchanged after append and must not borrow or lock the mutable writer.
pub trait SnapshotStore {
    type View: ObservationSource + Send + Sync;
    fn deferred_page_reads(&self) -> bool {
        false
    }
    fn read_view(&mut self) -> Result<Self::View>;
    fn append_group(&mut self, group: &Group, observations: &[Observation]) -> Result<()>;
    /// Wait for accepted writes and report a background writer failure.
    fn finish(&mut self) -> Result<()> {
        Ok(())
    }
}
impl SnapshotStore for SpatialIndex {
    type View = ResidentSnapshot;
    fn read_view(&mut self) -> Result<Self::View> {
        self.snapshot()
    }
    fn append_group(&mut self, _: &Group, observations: &[Observation]) -> Result<()> {
        for observation in observations {
            self.append(*observation)?;
        }
        Ok(())
    }
}

struct Projection {
    observations: Vec<Observation>,
    bounds: crate::spatial::hierarchy::Bounds,
    ids: HashMap<u64, usize>,
    values: Vec<OnceLock<f64>>,
    id: u64,
    id_range: (u64, u64),
}
impl Projection {
    fn new(maximum: usize) -> Result<Self> {
        let mut observations = Vec::new();
        observations
            .try_reserve_exact(maximum)
            .map_err(|_| invalid("projection observations allocation"))?;
        let mut values = Vec::new();
        values
            .try_reserve_exact(maximum)
            .map_err(|_| invalid("projection values allocation"))?;
        let mut ids = HashMap::new();
        ids.try_reserve(maximum)
            .map_err(|_| invalid("projection identities allocation"))?;
        Ok(Self {
            observations,
            bounds: crate::spatial::hierarchy::Bounds::point([0.; 3]),
            ids,
            values,
            id: 0,
            id_range: (0, 0),
        })
    }
    fn reset(&mut self, group: &Group) -> Result<()> {
        self.id = group.id;
        self.id_range = (u64::MAX, 0);
        self.observations.clear();
        self.ids.clear();
        self.values.clear();
        self.bounds = crate::spatial::hierarchy::Bounds::point(
            group.targets[0].site.support.center().coordinates(),
        );
        for target in &group.targets {
            self.id_range.0 = self.id_range.0.min(target.id);
            self.id_range.1 = self.id_range.1.max(target.id);
            if self
                .ids
                .insert(target.id, self.observations.len())
                .is_some()
            {
                return Err(invalid("duplicate target ID in simulation group"));
            }
            self.observations
                .push(Observation::new(target.id, target.site, 0., 0.)?);
            self.values.push(OnceLock::new());
            self.bounds
                .include(crate::spatial::hierarchy::Bounds::point(
                    target.site.support.center().coordinates(),
                ));
        }
        Ok(())
    }
    fn contains_id(&self, id: u64) -> bool {
        self.id_range.0 <= id && id <= self.id_range.1 && self.ids.contains_key(&id)
    }
    fn bind(&self, observation: &mut Observation) -> Result<()> {
        if observation.id() < self.id_range.0 || observation.id() > self.id_range.1 {
            return Ok(());
        }
        if let Some(&index) = self.ids.get(&observation.id()) {
            let value = *self.values[index]
                .get()
                .ok_or_else(|| invalid("unresolved rolling dependency"))?;
            *observation = Observation::new(
                observation.id(),
                *observation.site(),
                value,
                observation.noise_variance(),
            )?;
        }
        Ok(())
    }
}
struct Context<V> {
    base: Option<V>,
    predecessors: Vec<Arc<Projection>>,
    own: Option<Arc<Projection>>,
}
impl<V: ObservationSource> ObservationSource for Context<V> {
    fn scan(&self, visitor: &mut dyn FnMut(&Observation) -> Result<()>) -> Result<()> {
        self.base.as_ref().expect("admitted view").scan(visitor)?;
        for group in &self.predecessors {
            for o in &group.observations {
                visitor(o)?;
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
        self.base
            .as_ref()
            .expect("admitted view")
            .visit_bounds(lower, upper, cancel, visitor)?;
        for group in &self.predecessors {
            cancel.check()?;
            if !group
                .bounds
                .intersects(crate::spatial::hierarchy::Bounds { lower, upper })
            {
                continue;
            }
            for o in &group.observations {
                let p = o.site().support.center().coordinates();
                if (0..3).all(|i| lower[i] <= p[i] && p[i] <= upper[i]) {
                    visitor(o)?;
                }
            }
        }
        Ok(())
    }
    fn nearest_into<'a>(
        &self,
        point: crate::data::Point3,
        workspace: &'a mut NearestWorkspace,
        cancel: &Cancellation,
    ) -> Result<&'a [Observation]> {
        if self.predecessors.is_empty() {
            return self
                .base
                .as_ref()
                .expect("admitted view")
                .nearest_into(point, workspace, cancel);
        }
        self.base
            .as_ref()
            .expect("admitted view")
            .nearest_into(point, workspace, cancel)?;
        workspace.resume();
        for group in &self.predecessors {
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
        self.at_site_with_center_status(site).map(|(found, _)| found)
    }
    fn at_site_with_center_status(&self, site: &Site) -> Result<(Vec<Observation>, bool)> {
        let (mut found, mut center_empty) = self
            .base
            .as_ref()
            .expect("admitted view")
            .at_site_with_center_status(site)?;
        let center = site.support.center().coordinates();
        for group in &self.predecessors {
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
struct Slot<M: Mode, V> {
    prepared: Prepared<M>,
    context: Context<V>,
    dependencies: Vec<Arc<Projection>>,
    dependency_ids: Vec<u64>,
    draws: Vec<Observation>,
}

/// Bounded dynamic preparation and dependency-ready drawing with ordered output.
/// Preserves conditioning order and random streams. Sink publication/output is
/// ordered; a sink may queue writes and acknowledge durability in finish. Packed
/// cache misses transfer preparation to a bounded blocking service. This API
/// alone does not create transactional simulation identity checkpoints.
#[allow(clippy::too_many_arguments)]
pub fn simulate_pipelined<
    C: Covariance,
    D: Drift,
    S: SnapshotStore,
    H: ObservationSource + Sync + ?Sized,
>(
    model: &C,
    hard: &H,
    simulated: &mut S,
    groups: impl IntoIterator<Item = Group>,
    drift: &D,
    options: SimulationOptions<'_>,
    execution: impl Into<ExecutionConfig>,
    output: impl FnMut(&[Observation]) -> Result<()>,
) -> Result<()> {
    run::<Unchecked, _, _, _, _>(
        model,
        hard,
        simulated,
        groups,
        drift,
        options,
        execution,
        |_, _| Ok(()),
        output,
    )
}
/// Checked rolling simulation. Private draws may execute out of order. Inspect
/// each group in path order before writes or output. Rejected groups and all
/// successors remain unpublished, even if their private draws already completed.
#[allow(clippy::too_many_arguments)]
pub fn simulate_pipelined_with_residuals<
    C: Covariance,
    D: Drift,
    S: SnapshotStore,
    H: ObservationSource + Sync + ?Sized,
>(
    model: &C,
    hard: &H,
    simulated: &mut S,
    groups: impl IntoIterator<Item = Group>,
    drift: &D,
    options: SimulationOptions<'_>,
    execution: impl Into<ExecutionConfig>,
    mut inspect: impl FnMut(&Group, &GroupResiduals) -> Result<()>,
    output: impl FnMut(&[Observation]) -> Result<()>,
) -> Result<()> {
    run::<diagnostics::Checked, _, _, _, _>(
        model,
        hard,
        simulated,
        groups,
        drift,
        options,
        execution,
        |group, report| inspect(group, report.as_ref().expect("checked plan")),
        output,
    )
}
#[allow(clippy::too_many_arguments)]
fn run<M: Mode, C: Covariance, D: Drift, S: SnapshotStore, H: ObservationSource + Sync + ?Sized>(
    model: &C,
    hard: &H,
    simulated: &mut S,
    groups: impl IntoIterator<Item = Group>,
    drift: &D,
    options: SimulationOptions<'_>,
    execution: impl Into<ExecutionConfig>,
    mut inspect: impl FnMut(&Group, &M::Report) -> Result<()>,
    mut output: impl FnMut(&[Observation]) -> Result<()>,
) -> Result<()> {
    let config = execution.into();
    let execution = config.options;
    options.validate()?;
    let _window_memory = execution.reserve(options.budget)?;
    let count = execution.batch_groups;
    let extra = checked_len(&[count, count, 4, std::mem::size_of::<Arc<Projection>>()])?
        .checked_add(checked_len(&[
            count,
            512 + std::mem::size_of::<Context<S::View>>() + std::mem::size_of::<M::Report>(),
        ])?)
        .and_then(|n| n.checked_add(std::mem::size_of::<M>() * execution.worker_count()))
        .ok_or(Error::SizeOverflow)?;
    let _metadata = options.budget.reserve(extra as u64)?;
    // The coordinator maintains the exact admitted ID set instead of comparing
    // every new target with every preceding group's map. Charge hash capacity
    // conservatively (including load-factor and power-of-two rounding) first.
    let id_capacity = checked_len(&[count, execution.maximum_targets_per_group])?;
    let id_bytes = id_capacity
        .checked_mul(64)
        .and_then(|n| n.checked_add(128))
        .ok_or(Error::SizeOverflow)?;
    let _id_memory = options.budget.reserve(id_bytes as u64)?;
    let mut admitted_ids = HashSet::new();
    admitted_ids
        .try_reserve(id_capacity)
        .map_err(|_| invalid("rolling admitted IDs allocation"))?;
    let setup = crate::execution::profiling::Span::new(
        crate::execution::profiling::Metric::SimulationSetup,
    );
    let io_count = if hard.deferred_page_reads() || simulated.deferred_page_reads() {
        execution.worker_count().min(4)
    } else {
        0
    };
    let _io_memory = options.budget.reserve(
        (io_count as u64)
            .checked_mul(execution.bytes_per_group + std::mem::size_of::<M>() as u64)
            .ok_or(Error::SizeOverflow)?,
    )?;
    let io_workers = workers_n::<M>(
        &execution,
        options.hard_neighborhood,
        options.simulated_neighborhood,
        io_count,
    )?
    .into_iter()
    .map(|worker| {
        let drawing = DrawingWorkspace::new(&worker.budget)?;
        Ok((worker, drawing))
    })
    .collect::<Result<Vec<_>>>()?;
    let workers = workers::<M>(
        &execution,
        options.hard_neighborhood,
        options.simulated_neighborhood,
    )?;
    let workers = workers
        .into_iter()
        .map(|worker| {
            let drawing = DrawingWorkspace::new(&worker.budget)?;
            Ok((worker, drawing))
        })
        .collect::<Result<Vec<_>>>()?;
    let mut slots = Vec::new();
    slots
        .try_reserve_exact(count)
        .map_err(|_| invalid("rolling slots allocation"))?;
    for _ in 0..count {
        let mut predecessors = Vec::new();
        predecessors
            .try_reserve_exact(count)
            .map_err(|_| invalid("rolling dependencies allocation"))?;
        let mut dependencies = Vec::new();
        dependencies
            .try_reserve_exact(count)
            .map_err(|_| invalid("draw dependencies allocation"))?;
        let mut draws = Vec::new();
        draws
            .try_reserve_exact(execution.maximum_targets_per_group)
            .map_err(|_| invalid("draw result allocation"))?;
        let mut dependency_ids = Vec::new();
        dependency_ids
            .try_reserve_exact(count)
            .map_err(|_| invalid("dependency IDs allocation"))?;
        slots.push(Slot {
            dependency_ids,
            dependencies,
            draws,
            prepared: Prepared::<M>::new(&MemoryBudget::new(execution.bytes_per_group))?,
            context: Context {
                base: None,
                predecessors,
                own: None,
            },
        });
    }
    // Any retained predecessor is within one window of the earliest admitted
    // job. Two windows suffice; reuse only after every dependent releases it.
    let mut projections = Vec::new();
    projections
        .try_reserve_exact(count * 2)
        .map_err(|_| invalid("projection pool allocation"))?;
    for _ in 0..count * 2 {
        projections.push(Arc::new(Projection::new(
            execution.maximum_targets_per_group,
        )?));
    }
    let mut pending = VecDeque::<Arc<Projection>>::new();
    pending
        .try_reserve_exact(count)
        .map_err(|_| invalid("rolling pending allocation"))?;
    drop(setup);
    let result = pipeline::run_staged(
        workers,
        io_workers,
        slots,
        groups,
        &execution,
        config.thread_start.as_ref(),
        options.cancellation,
        |group, (worker, drawing), slot, stage| {
            match stage {
                Stage::Prepare | Stage::BlockingPrepare | Stage::Factor => {
                    if !matches!(stage, Stage::Factor) {
                        let ids = &slot.context.own.as_ref().expect("admitted projection").ids;
                        let hard = Checked { source: hard, ids };
                        let source = Checked {
                            source: &slot.context,
                            ids,
                        };
                        slot.prepared.status = crate::execution::paging::with_cache_only(
                            io_count != 0 && matches!(stage, Stage::Prepare),
                            || {
                                select_simulation(
                                    &hard,
                                    &source,
                                    group,
                                    &options,
                                    worker,
                                    &mut slot.prepared,
                                )
                            },
                        );
                        if matches!(slot.prepared.status, Err(Error::WouldBlock)) {
                            return Progress::Blocking;
                        }
                        if slot.prepared.status.is_err() {
                            return Progress::Complete;
                        }
                        if matches!(stage, Stage::BlockingPrepare) {
                            return Progress::Selected;
                        }
                    }
                    slot.prepared.status =
                        prepare_selected(model, drift, group, &options, worker, &mut slot.prepared);
                    if slot.prepared.status.is_err() {
                        return Progress::Complete;
                    }
                    // Use the raw selection, including exact aliases. A zero
                    // weight is not a reason to discard a consistency dependency.
                    for previous in &slot.context.predecessors {
                        if slot
                            .prepared
                            .selected
                            .observations
                            .iter()
                            .any(|o| previous.contains_id(o.id()))
                        {
                            slot.dependency_ids.push(previous.id);
                            slot.dependencies.push(previous.clone());
                        }
                    }
                    Progress::Waiting
                }
                Stage::Draw => {
                    slot.prepared.status = (|| -> Result<()> {
                        for observation in &mut slot.prepared.selected.observations {
                            for previous in &slot.dependencies {
                                previous.bind(observation)?;
                            }
                        }
                        let selected = &mut slot.prepared.selected.observations;
                        let plan = slot.prepared.plan.get()?;
                        if selected.len() != plan.weights().nrows() {
                            worker.searches.canonical.compact(
                                selected,
                                true,
                                options.cancellation,
                            )?;
                        }
                        let result = drawing.draw(plan, selected, group, &options)?;
                        options.cancellation.check()?;
                        slot.draws.extend_from_slice(result);
                        let own = slot.context.own.as_ref().expect("admitted projection");
                        for observation in result {
                            let index = own.ids[&observation.id()];
                            own.values[index]
                                .set(observation.value())
                                .map_err(|_| invalid("duplicate rolling draw"))?;
                        }
                        // Publish private dependency values only after the
                        // complete joint draw succeeds. Durable writes are later.
                        Ok(())
                    })();
                    if slot.prepared.status.is_ok() {
                        Progress::Drawn
                    } else {
                        Progress::Complete
                    }
                }
            }
        },
        count,
        |slot| &slot.dependency_ids,
        |event| match event {
            Event::Admit(group, slot) => {
                let index = projections
                    .iter()
                    .position(|p| Arc::strong_count(p) == 1)
                    .ok_or_else(|| invalid("rolling projection capacity invariant"))?;
                let projection =
                    Arc::get_mut(&mut projections[index]).expect("exclusive projection");
                projection.reset(group)?;
                for target in &group.targets {
                    if !admitted_ids.insert(target.id) {
                        return Err(invalid("duplicate target ID within rolling window"));
                    }
                }
                slot.context.base = Some(simulated.read_view()?);
                slot.context.predecessors.extend(pending.iter().cloned());
                let own = projections[index].clone();
                pending.push_back(own.clone());
                slot.context.own = Some(own);
                Ok(())
            }
            Event::Consume(group, slot) => {
                let prepared = &mut slot.prepared;
                std::mem::replace(&mut prepared.status, Ok(()))?;
                inspect(&group, &prepared.report)?;
                options.cancellation.check()?;
                let own = pending.pop_front().expect("ordered projection");
                debug_assert!(Arc::ptr_eq(&own, slot.context.own.as_ref().unwrap()));
                simulated.append_group(&group, &slot.draws)?;
                output(&slot.draws)?;
                for target in &group.targets {
                    let removed = admitted_ids.remove(&target.id);
                    debug_assert!(removed);
                }
                slot.draws.clear();
                slot.dependencies.clear();
                slot.dependency_ids.clear();
                slot.context.base = None;
                slot.context.predecessors.clear();
                slot.context.own = None;
                Ok(())
            }
        },
    );
    // Always drain accepted writes, including on inspection/output failure.
    // Preserve the original error if drawing stopped before the writer did.
    let finished = simulated.finish();
    result.and(finished)
}

#[cfg(test)]
mod exact_center_tests {
    use super::*;
    use crate::data::{Point3, Site, Support, Target};
    use std::cell::Cell;

    struct ProvenEmpty;
    impl ObservationSource for ProvenEmpty {
        fn scan(&self, _: &mut dyn FnMut(&Observation) -> Result<()>) -> Result<()> {
            Ok(())
        }
        fn at_site_with_center_status(&self, _: &Site) -> Result<(Vec<Observation>, bool)> {
            Ok((Vec::new(), true))
        }
    }

    fn site(center: [f64; 3], variable: usize) -> Site {
        Site::point(Point3::new(center).unwrap(), variable)
    }

    fn projection(id: u64, targets: Vec<Target>) -> Arc<Projection> {
        let mut projection = Projection::new(targets.len()).unwrap();
        projection.reset(&Group { id, targets }).unwrap();
        Arc::new(projection)
    }

    #[test]
    fn predecessor_centers_require_all_variables_and_supports_to_be_absent() {
        let point_site = site([-0., 0., -0.], 7);
        let block_site = Site {
            support: Support::block(
                Point3::new([0., -0., 0.]).unwrap(),
                [0.1, 0.2, 0.],
                [2, 3, 1],
            )
            .unwrap(),
            variable: 9,
        };
        let context = Context {
            base: Some(ProvenEmpty),
            predecessors: vec![
                projection(
                    u64::MAX,
                    vec![
                        Target {
                            id: u64::MAX - 1,
                            site: point_site,
                        },
                        Target {
                            id: 23,
                            site: block_site,
                        },
                    ],
                ),
                projection(
                    2,
                    vec![Target {
                        id: 7,
                        site: point_site,
                    }],
                ),
            ],
            own: Some(projection(
                3,
                vec![Target {
                    id: 123,
                    site: site([3., 2., 1.], 0),
                }],
            )),
        };
        for query in [
            site([0., -0., 0.], 99),
            site([0.; 3], 9),
            Site {
                support: block_site.support,
                variable: 7,
            },
        ] {
            let (found, center_empty) = context.at_site_with_center_status(&query).unwrap();
            assert!(found.is_empty());
            assert!(!center_empty, "a different variable/support occupies the center");
        }
        let (aliases, center_empty) = context
            .at_site_with_center_status(&site([0.; 3], 7))
            .unwrap();
        assert!(!center_empty);
        assert_eq!(
            aliases.iter().map(Observation::id).collect::<Vec<_>>(),
            [7, u64::MAX - 1]
        );
        let (blocks, center_empty) = context.at_site_with_center_status(&block_site).unwrap();
        assert!(!center_empty);
        assert_eq!(blocks.iter().map(Observation::id).collect::<Vec<_>>(), [23]);
        let (found, center_empty) = context
            .at_site_with_center_status(&site([3., 2., 1.], 0))
            .unwrap();
        assert!(found.is_empty());
        assert!(center_empty, "the group's own targets are not predecessors");
    }

    #[test]
    fn custom_base_without_center_proof_disables_empty_center_status() {
        struct Custom {
            calls: Cell<usize>,
        }
        impl ObservationSource for Custom {
            fn scan(&self, _: &mut dyn FnMut(&Observation) -> Result<()>) -> Result<()> {
                panic!("preserve the custom at_site override")
            }
            fn at_site(&self, _: &Site) -> Result<Vec<Observation>> {
                self.calls.set(self.calls.get() + 1);
                Ok(Vec::new())
            }
        }
        let context = Context {
            base: Some(Custom {
                calls: Cell::new(0),
            }),
            predecessors: vec![projection(
                9,
                vec![Target {
                    id: 500,
                    site: site([10.; 3], 0),
                }],
            )],
            own: None,
        };
        let (found, center_empty) = context.at_site_with_center_status(&site([0.; 3], 0)).unwrap();
        assert!(found.is_empty());
        assert!(!center_empty, "a custom exact-site miss is not a center proof");
        assert_eq!(context.base.as_ref().unwrap().calls.get(), 1);
    }
}
