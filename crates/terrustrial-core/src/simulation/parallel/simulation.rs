//! Reusable retained plans and ordered simulation drawing.
use super::*;
use crate::gaussian::RetainedPlan;

pub(super) struct Prepared<M: Mode = Unchecked> {
    pub(super) report: M::Report,
    pub(super) selected: Selection,
    pub(super) plan: RetainedPlan,
    pub(super) status: Result<()>,
}
impl<M: Mode> Prepared<M> {
    pub(super) fn new(budget: &MemoryBudget) -> Result<Self> {
        Ok(Self {
            report: M::Report::default(),
            selected: Selection::new(budget)?,
            plan: RetainedPlan::new(budget)?,
            status: Ok(()),
        })
    }
}

/// Concurrent preparation, serial path-ordered drawing and writes. Within-window
/// predecessors are placeholders only during matrix construction. Actual draws
/// replace their values before prediction, with duplicate consistency rechecked.
/// As in simulate_groups, a failing store append may leave a partial group. This
/// entry point does not provide transactional checkpointing.
#[allow(clippy::too_many_arguments)]
pub fn simulate_groups<C, D, S, H>(
    model: &C,
    hard: &H,
    simulated: &mut S,
    groups: impl IntoIterator<Item = Group>,
    drift: &D,
    options: SimulationOptions<'_>,
    execution: ExecutionOptions,
    output: impl FnMut(&[Observation]) -> Result<()>,
) -> Result<()>
where
    C: Covariance,
    D: Drift,
    S: ObservationSource + ObservationSink + Sync,
    H: ObservationSource + Sync + ?Sized,
{
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
/// Checked parallel simulation. Numerical reports are prepared concurrently but
/// inspected in path order before drawing or mutation. A rejected/failed group
/// and every successor emit no output and make no store writes. Earlier groups
/// remain written. This is not transactional checkpointing; an append failure
/// can still leave a partial group. See the sequential checked API for policy,
/// report semantics and callback immutability requirements.
#[allow(clippy::too_many_arguments)]
pub fn simulate_groups_with_residuals<
    C: Covariance,
    D: Drift,
    S: ObservationSource + ObservationSink + Sync,
    H: ObservationSource + Sync + ?Sized,
>(
    model: &C,
    hard: &H,
    simulated: &mut S,
    groups: impl IntoIterator<Item = Group>,
    drift: &D,
    options: SimulationOptions<'_>,
    execution: ExecutionOptions,
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
fn run<
    M: Mode,
    C: Covariance,
    D: Drift,
    S: ObservationSource + ObservationSink + Sync,
    H: ObservationSource + Sync + ?Sized,
>(
    model: &C,
    hard: &H,
    simulated: &mut S,
    groups: impl IntoIterator<Item = Group>,
    drift: &D,
    options: SimulationOptions<'_>,
    execution: ExecutionOptions,
    mut inspect: impl FnMut(&Group, &M::Report) -> Result<()>,
    mut output: impl FnMut(&[Observation]) -> Result<()>,
) -> Result<()> {
    options.validate()?;
    let _window_memory = execution.reserve(options.budget)?;
    let diagnostic_bytes = checked_len(&[execution.worker_count(), std::mem::size_of::<M>()])?
        .checked_add(checked_len(&[
            execution.batch_groups,
            std::mem::size_of::<M::Report>(),
        ])?)
        .ok_or(Error::SizeOverflow)?;
    let _diagnostic_metadata = if diagnostic_bytes == 0 {
        None
    } else {
        Some(
            options.budget.reserve(
                diagnostic_bytes
                    .try_into()
                    .map_err(|_| Error::SizeOverflow)?,
            )?,
        )
    };
    let mut workers = workers::<M>(
        &execution,
        options.hard_neighborhood,
        options.simulated_neighborhood,
    )?;
    // Drawing is ordered after preparation; worker zero's allowance also holds
    // these persistent buffers. No additional uncharged arena is introduced.
    let mut drawing = DrawingWorkspace::new(&workers[0].budget)?;
    let mut slots = Vec::new();
    slots
        .try_reserve_exact(execution.batch_groups)
        .map_err(|_| invalid("simulation result slots allocation"))?;
    for _ in 0..execution.batch_groups {
        slots.push(Prepared::<M>::new(&MemoryBudget::new(
            execution.bytes_per_group,
        ))?);
    }
    let mut groups = groups.into_iter();
    let mut previous = None;
    let mut batch = Vec::new();
    let mut window = Window::empty();
    loop {
        execution.batch_into(&mut groups, &mut previous, options.cancellation, &mut batch)?;
        if batch.is_empty() {
            break;
        }
        window.rebuild(&batch)?;
        let hard = Checked {
            source: hard,
            ids: &window.bindings.ids,
        };
        let base = Checked {
            source: &*simulated,
            ids: &window.bindings.ids,
        };
        prepare_batches(
            &batch,
            &mut workers,
            &mut slots,
            |rank, group, worker, slot| {
                let overlay = Overlay {
                    base: &base,
                    window: &window,
                    rank,
                };
                slot.status = prepare_simulation(
                    model, drift, &hard, &overlay, group, &options, worker, slot,
                );
            },
        );
        for (group, slot) in batch.iter().zip(&mut slots) {
            options.cancellation.check()?;
            std::mem::replace(&mut slot.status, Ok(()))?;
            inspect(group, &slot.report)?;
            options.cancellation.check()?;
            let selected = &mut slot.selected.observations;
            for observation in selected.iter_mut() {
                if let Some(value) = window.bindings.value(observation.id())? {
                    *observation = Observation::new(
                        observation.id(),
                        *observation.site(),
                        value,
                        observation.noise_variance(),
                    )?;
                }
            }
            let plan = slot.plan.get()?;
            // Preparation validated IDs and geometry without using placeholder
            // values. Equal lengths prove that it found no duplicates or exact
            // aliases, so binding values cannot introduce a new conflict. Only
            // merged selections need another value-aware compaction pass.
            if selected.len() != plan.weights().nrows() {
                workers[0]
                    .searches
                    .canonical
                    .compact(selected, true, options.cancellation)?;
            }
            let result = drawing.draw(plan, selected, group, &options)?;
            options.cancellation.check()?;
            for observation in result {
                simulated.append(*observation)?;
                window.bindings.record(observation)?;
            }
            output(result)?;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn prepare_simulation<
    M: Mode,
    C: Covariance,
    D: Drift,
    H: ObservationSource + ?Sized,
    S: ObservationSource + ?Sized,
>(
    model: &C,
    drift: &D,
    hard: &H,
    simulated: &S,
    group: &Group,
    options: &SimulationOptions<'_>,
    worker: &mut Worker<'_, M>,
    slot: &mut Prepared<M>,
) -> Result<()> {
    select_simulation(hard, simulated, group, options, worker, slot)?;
    prepare_selected(model, drift, group, options, worker, slot)
}

pub(super) fn select_simulation<
    M: Mode,
    H: ObservationSource + ?Sized,
    S: ObservationSource + ?Sized,
>(
    hard: &H,
    simulated: &S,
    group: &Group,
    options: &SimulationOptions<'_>,
    worker: &mut Worker<'_, M>,
    slot: &mut Prepared<M>,
) -> Result<()> {
    slot.plan.invalidate();
    slot.selected.observations.clear();
    visit_group_selection(
        hard,
        simulated,
        group,
        &mut worker.searches,
        options.cancellation,
        |part| slot.selected.extend(&part),
    )?;
    Ok(())
}

pub(super) fn prepare_selected<M: Mode, C: Covariance, D: Drift>(
    model: &C,
    drift: &D,
    group: &Group,
    options: &SimulationOptions<'_>,
    worker: &mut Worker<'_, M>,
    slot: &mut Prepared<M>,
) -> Result<()> {
    // Keep the raw selection until binding: discarded exact aliases still need
    // their IDs and values checked against every actual predecessor draw.
    if worker.simulation.is_none() {
        worker.simulation = Some(SimulationInputs::new(&worker.budget)?);
    }
    let inputs = worker
        .simulation
        .as_mut()
        .expect("initialized simulation inputs");
    inputs.selection.observations.clear();
    inputs.selection.extend(&slot.selected.observations)?;
    worker.searches.canonical.compact(
        &mut inputs.selection.observations,
        false,
        options.cancellation,
    )?;
    inputs.fill_sites(group)?;
    let observations = &inputs.selection.observations;
    let solver = GaussianWorkspace::cached(
        &mut worker.searches.solver,
        observations.len(),
        inputs.sites.len(),
        drift.columns(),
        &worker.budget,
    )?;
    let (_, report) = worker.diagnostic.prepare(
        solver,
        model,
        observations,
        &inputs.sites,
        drift,
        &worker.budget,
        WorkspaceOptions {
            uncertainty: Uncertainty::Joint,
            tolerance: NumericalTolerance::default(),
            cancellation: options.cancellation,
        },
    )?;
    slot.plan.update_from_workspace(solver)?;
    slot.report = report;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        gaussian::Simple,
        model::{Anisotropy, Coregionalization, Correlation, Lmc, Structure},
    };

    #[test]
    fn warmed_preparation_retains_plans_without_allocating_and_invalidates_failures() {
        warmed::<Unchecked>();
        warmed::<diagnostics::Checked>();
    }
    fn warmed<M: Mode>() {
        let model = Lmc::new(vec![Structure {
            correlation: Correlation::Exponential,
            anisotropy: Anisotropy::isotropic(3.).unwrap(),
            coregionalization: Coregionalization::scalar(1.).unwrap(),
        }])
        .unwrap();
        let hard: Vec<_> = (0..32)
            .map(|i| {
                Observation::new(
                    i,
                    Site::point(Point3::new([i as f64 * 0.2, 0., 0.]).unwrap(), 0),
                    i as f64,
                    0.01,
                )
                .unwrap()
            })
            .collect();
        let groups: Vec<_> = (0..8)
            .map(|i| Group {
                id: i,
                targets: (0..1 + i % 4)
                    .map(|j| Target {
                        id: 100 + i * 4 + j,
                        site: Site::point(
                            Point3::new([0.13 + i as f64 * 0.13 + j as f64 * 0.17, 0.1, 0.1])
                                .unwrap(),
                            0,
                        ),
                    })
                    .collect(),
            })
            .collect();
        let query = Neighborhood {
            maximum: 32,
            radius: 100.,
        }
        .into();
        let budget = MemoryBudget::new(1 << 24);
        let cancel = Cancellation::default();
        let options = SimulationOptions {
            hard_neighborhood: query,
            simulated_neighborhood: query,
            seed: 1,
            realization: 0,
            mean: None,
            budget: &budget,
            cancellation: &cancel,
        };
        let execution = ExecutionOptions {
            batch_groups: 1,
            maximum_targets_per_group: 4,
            bytes_per_group: 1 << 20,
        };
        let mut workers = workers::<M>(&execution, query, query).unwrap();
        let slot_budget = MemoryBudget::new(1 << 20);
        let mut slot = Prepared::new(&slot_budget).unwrap();
        let empty: &[Observation] = &[];
        prepare_simulation(
            &model,
            &Simple,
            hard.as_slice(),
            empty,
            &groups[3],
            &options,
            &mut workers[0],
            &mut slot,
        )
        .unwrap();
        let used = slot_budget.used();
        let measured = allocation_counter::measure(|| {
            for _ in 0..100 {
                for group in &groups {
                    prepare_simulation(
                        &model,
                        &Simple,
                        hard.as_slice(),
                        empty,
                        group,
                        &options,
                        &mut workers[0],
                        &mut slot,
                    )
                    .unwrap();
                    assert_eq!(
                        slot.plan.get().unwrap().weights().ncols(),
                        group.targets.len()
                    );
                }
            }
        });
        assert_eq!(measured.count_total, 0, "{measured:?}");
        assert_eq!(used, slot_budget.used());
        cancel.cancel();
        assert!(prepare_simulation(
            &model,
            &Simple,
            hard.as_slice(),
            empty,
            &groups[0],
            &options,
            &mut workers[0],
            &mut slot
        )
        .is_err());
        assert!(slot.plan.get().is_err());
        drop(slot);
        assert_eq!(slot_budget.used(), 0);
    }
}
