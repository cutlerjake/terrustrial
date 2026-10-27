//! Reusable storage and ordered parallel estimation.
mod diagnostics;
use crate::execution::buffer::grow;
#[cfg(test)]
use crate::execution::ExecutionOptions;
use crate::{
    data::{Group, Observation, Site},
    error::{checked_len, invalid},
    execution::{pipeline, Cancellation, ExecutionConfig, MemoryBudget, Reservation},
    gaussian::{
        group::{group_center, CanonicalWorkspace, Selection},
        Drift, GaussianPlan, KnownMean, NeighborhoodConditioning, NumericalTolerance, Uncertainty,
        WorkspaceOptions,
    },
    model::Covariance,
    spatial::{search::QueryWorkspace, NeighborhoodQuery, ObservationSource},
    Error, Result,
};
#[cfg(test)]
use crate::{
    data::{Point3, Target},
    gaussian::{group::canonicalize_observations, GaussianWorkspace},
    spatial::Neighborhood,
};
pub use diagnostics::{estimate_groups_with_residuals, GroupResiduals};
use diagnostics::{Mode, Unchecked};

struct Worker<'a> {
    budget: MemoryBudget,
    search: QueryWorkspace<'a>,
    canonical: CanonicalWorkspace,
    estimation: Option<EstimationBuffers>,
}
impl<'a> Worker<'a> {
    fn new(query: NeighborhoodQuery<'a>, bytes: u64) -> Result<Self> {
        let budget = MemoryBudget::new(bytes);
        Ok(Self {
            search: QueryWorkspace::new(query, &budget)?,
            canonical: CanonicalWorkspace::new(&budget)?,
            estimation: None,
            budget,
        })
    }
}

struct EstimationBuffers {
    maximum_observations: usize,
    maximum_targets: usize,
    with_mean: bool,
    sites: Vec<Site>,
    values: Vec<f64>,
    target_mean: Vec<f64>,
    _reservation: Reservation,
}
impl EstimationBuffers {
    fn cached<'a>(
        slot: &'a mut Option<Self>,
        n: usize,
        m: usize,
        with_mean: bool,
        budget: &MemoryBudget,
    ) -> Result<&'a mut Self> {
        if !slot.as_ref().is_some_and(|v| {
            n <= v.maximum_observations && m <= v.maximum_targets && (!with_mean || v.with_mean)
        }) {
            let (n, m) = slot.as_ref().map_or((n, m), |v| {
                (n.max(v.maximum_observations), m.max(v.maximum_targets))
            });
            *slot = None;
            let means = if with_mean { m } else { 0 };
            let bytes = checked_len(&[m, std::mem::size_of::<Site>()])?
                .checked_add(checked_len(&[n, 8])?)
                .and_then(|v| v.checked_add(means.checked_mul(8)?))
                .ok_or(Error::SizeOverflow)?;
            let reservation =
                budget.reserve(u64::try_from(bytes).map_err(|_| Error::SizeOverflow)?)?;
            fn buffer<T>(capacity: usize) -> Result<Vec<T>> {
                let mut out = Vec::new();
                out.try_reserve_exact(capacity)
                    .map_err(|_| invalid("estimation input buffer allocation"))?;
                Ok(out)
            }
            *slot = Some(Self {
                maximum_observations: n,
                maximum_targets: m,
                with_mean,
                sites: buffer(m)?,
                values: buffer(n)?,
                target_mean: buffer(if with_mean { m } else { 0 })?,
                _reservation: reservation,
            });
        }
        Ok(slot.as_mut().expect("initialized estimation buffers"))
    }
    fn fill(&mut self, observations: &[Observation], group: &Group) {
        self.sites.clear();
        self.sites.extend(group.targets.iter().map(|t| t.site));
        self.values.clear();
        self.values
            .extend(observations.iter().map(Observation::value));
    }
    fn predict(
        &mut self,
        plan: &GaussianPlan,
        observations: &[Observation],
        mean: Option<&dyn KnownMean>,
        output: &mut [f64],
    ) -> Result<()> {
        self.predict_component(plan, observations, mean, mean, output)
    }
    fn predict_component(
        &mut self,
        plan: &GaussianPlan,
        observations: &[Observation],
        observation_mean: Option<&dyn KnownMean>,
        target_mean: Option<&dyn KnownMean>,
        output: &mut [f64],
    ) -> Result<()> {
        let evaluate = |mean: &dyn KnownMean, site: &Site| -> Result<f64> {
            let value = mean.evaluate(site)?;
            if !value.is_finite() {
                return Err(invalid(
                    "known mean must be finite on every selected support",
                ));
            }
            Ok(value)
        };
        if let Some(mean) = observation_mean {
            for (value, observation) in self.values.iter_mut().zip(observations) {
                *value -= evaluate(mean, observation.site())?;
                if !value.is_finite() {
                    return Err(Error::Numerical {
                        context: "known-mean observation residual",
                    });
                }
            }
        }
        if let Some(mean) = target_mean {
            self.target_mean.clear();
            for site in &self.sites {
                self.target_mean.push(evaluate(mean, site)?);
            }
        }
        plan.estimate_into(&self.values, output)?;
        if target_mean.is_some() {
            for (value, target_mean) in output.iter_mut().zip(&self.target_mean) {
                *value += target_mean;
                if !value.is_finite() {
                    return Err(Error::Numerical {
                        context: "known-mean conditional prediction",
                    });
                }
            }
        }
        Ok(())
    }
}
pub struct EstimationOptions<'a> {
    pub neighborhood: NeighborhoodQuery<'a>,
    pub mean: Option<&'a dyn KnownMean>,
    pub uncertainty: Uncertainty,
    pub tolerance: NumericalTolerance,
    pub budget: &'a MemoryBudget,
    pub cancellation: &'a Cancellation,
}
/// Values and optional joint error covariance, in target order. The callback
/// borrows the result so the batch memory reservation cannot escape the driver.
pub struct GroupEstimate {
    pub group: Group,
    pub values: Vec<f64>,
    uncertainty: RetainedUncertainty,
}
impl GroupEstimate {
    pub fn covariance(&self) -> Option<faer::MatRef<'_, f64>> {
        self.uncertainty.covariance.as_ref().map(faer::Mat::as_ref)
    }
    pub fn roundoff_corrections(&self) -> usize {
        self.uncertainty.roundoff_corrections
    }
}

// Prediction has consumed the weights. Estimation exposes no sampling operation,
// so neither weights nor the conditional sampling factor need to escape a worker.
struct RetainedUncertainty {
    covariance: Option<faer::Mat<f64>>,
    roundoff_corrections: usize,
    capacity: usize,
    reservation: Option<Reservation>,
}
impl RetainedUncertainty {
    fn empty() -> Self {
        Self {
            covariance: None,
            roundoff_corrections: 0,
            capacity: 0,
            reservation: None,
        }
    }
    fn update(&mut self, plan: &GaussianPlan, budget: &MemoryBudget) -> Result<()> {
        if let Some(source) = plan.covariance() {
            let m = source.nrows();
            if m > self.capacity {
                let bytes =
                    u64::try_from(checked_len(&[m, m, 8])?).map_err(|_| Error::SizeOverflow)?;
                if let Some(reservation) = &mut self.reservation {
                    reservation.resize(bytes)?;
                } else {
                    self.reservation = Some(budget.reserve(bytes)?);
                }
                let matrix = self.covariance.get_or_insert_with(faer::Mat::new);
                if matrix.try_reserve(m, m).is_err() {
                    self.reservation
                        .as_mut()
                        .expect("reserved covariance")
                        .resize((self.capacity * self.capacity * 8) as u64)?;
                    return Err(invalid("estimation covariance allocation"));
                }
                self.capacity = m;
            }
            let matrix = self.covariance.as_mut().expect("initialized covariance");
            matrix.resize_with(m, m, |_, _| 0.);
            matrix.as_mut().copy_from(source);
        } else {
            self.covariance = None;
            self.reservation = None;
            self.capacity = 0;
        }
        self.roundoff_corrections = plan.roundoff_corrections();
        Ok(())
    }
    #[cfg(test)]
    fn from_plan(plan: &GaussianPlan, budget: &MemoryBudget) -> Result<Self> {
        let mut out = Self::empty();
        out.update(plan, budget)?;
        Ok(out)
    }
}
struct Estimated<M: Mode = Unchecked> {
    selection: Selection,
    report: M::Report,
    values: Vec<f64>,
    uncertainty: RetainedUncertainty,
    values_reservation: Reservation,
    budget: MemoryBudget,
    status: Result<()>,
}
impl<M: Mode> Estimated<M> {
    fn new(budget: MemoryBudget) -> Result<Self> {
        Ok(Self {
            selection: Selection::new(&budget)?,
            report: M::Report::default(),
            values: Vec::new(),
            uncertainty: RetainedUncertainty::empty(),
            values_reservation: budget.reserve(0)?,
            budget,
            status: Ok(()),
        })
    }
}

/// Parallel neighborhood search, preparation and prediction with ordered callbacks.
/// IDs increase in input order. Errors stop output at a group boundary; already
/// emitted groups remain. Input is validated before admission to a bounded
/// rolling window. The calling Rayon pool determines compute concurrency;
/// scoped compute workers and a bounded disk-read service execute the jobs.
/// After selection and duplicate resolution, observations are ordered by ID.
/// Each worker retains one conditioning system and reuses it only for bitwise
/// identical observations in that order. Changing neighbor distances alone does
/// not force refactorization. Model, drift and source outputs must remain
/// deterministic and unchanged during the call, including interior mutable state.
/// Cache storage is bounded by worker capacity and charged to its memory budget.
pub fn estimate_groups<C, D, H>(
    model: &C,
    hard: &H,
    groups: impl IntoIterator<Item = Group>,
    drift: &D,
    options: EstimationOptions<'_>,
    execution: impl Into<ExecutionConfig>,
    mut output: impl FnMut(&GroupEstimate) -> Result<()>,
) -> Result<()>
where
    C: Covariance,
    D: Drift,
    H: ObservationSource + Sync + ?Sized,
{
    let target = factorial::Component {
        covariance: model,
        drift,
        mean: options.mean,
    };
    estimate_with_component::<Unchecked, _, _, _, _, _>(
        model,
        hard,
        groups,
        drift,
        target,
        true,
        options,
        execution,
        |out, ()| output(out),
    )
}

#[allow(clippy::too_many_arguments)]
fn estimate_with_component<M: Mode, C, D, H, T, TD>(
    model: &C,
    hard: &H,
    groups: impl IntoIterator<Item = Group>,
    drift: &D,
    target: factorial::Component<'_, T, TD>,
    honor_exact: bool,
    options: EstimationOptions<'_>,
    execution: impl Into<ExecutionConfig>,
    mut output: impl FnMut(&GroupEstimate, &M::Report) -> Result<()>,
) -> Result<()>
where
    C: Covariance,
    D: Drift,
    T: Covariance,
    TD: Drift,
    H: ObservationSource + Sync + ?Sized,
{
    let config = execution.into();
    let execution = config.options;
    if target.covariance.variables() != model.variables()
        || target.drift.columns() != drift.columns()
    {
        return Err(invalid("component covariance/drift dimensions"));
    }
    options.neighborhood.validate()?;
    options.cancellation.check()?;
    let _window = execution.reserve(options.budget)?;
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
            options
                .budget
                .reserve(u64::try_from(diagnostic_bytes).map_err(|_| Error::SizeOverflow)?)?,
        )
    };
    let io_count = if hard.deferred_page_reads() {
        execution.worker_count().min(4)
    } else {
        0
    };
    let _io_memory = options.budget.reserve(
        (io_count as u64)
            .checked_mul(execution.bytes_per_group + std::mem::size_of::<M>() as u64)
            .ok_or(Error::SizeOverflow)?,
    )?;
    let make_worker = |_| {
        let worker = Worker::new(options.neighborhood, execution.bytes_per_group)?;
        let conditioning = NeighborhoodConditioning::new(model, drift, &worker.budget)?;
        Ok((worker, conditioning, M::default()))
    };
    let workers: Vec<_> = (0..execution.worker_count())
        .map(make_worker)
        .collect::<Result<_>>()?;
    let io_workers: Vec<_> = (0..io_count).map(make_worker).collect::<Result<_>>()?;
    let mut slots = Vec::new();
    slots
        .try_reserve_exact(execution.batch_groups)
        .map_err(|_| invalid("estimation result slots allocation"))?;
    for _ in 0..execution.batch_groups {
        slots.push(Estimated::<M>::new(MemoryBudget::new(
            execution.bytes_per_group,
        ))?);
    }
    pipeline::run_staged(
        workers,
        io_workers,
        slots,
        groups,
        &execution,
        config.thread_start.as_ref(),
        options.cancellation,
        |group, (worker, conditioning, diagnostic), slot, stage| {
            if !matches!(stage, pipeline::Stage::Factor) {
                slot.status = crate::execution::paging::with_cache_only(
                    io_count != 0 && matches!(stage, pipeline::Stage::Prepare),
                    || select_estimate(hard, group, &options, worker, slot),
                );
                if matches!(slot.status, Err(Error::WouldBlock)) {
                    return pipeline::Progress::Blocking;
                }
                if slot.status.is_err() {
                    return pipeline::Progress::Complete;
                }
                if matches!(stage, pipeline::Stage::BlockingPrepare) {
                    return pipeline::Progress::Selected;
                }
            }
            slot.status = estimate_selected(
                group,
                &options,
                &target,
                honor_exact,
                worker,
                conditioning,
                slot,
                diagnostic,
            );
            pipeline::Progress::Complete
        },
        0,
        |_| &[],
        |event| {
            let pipeline::Event::Consume(group, slot) = event else {
                return Ok(());
            };
            options.cancellation.check()?;
            std::mem::replace(&mut slot.status, Ok(()))?;
            // Keep the reservation through the callback, then recycle buffers.
            let result = GroupEstimate {
                group,
                values: std::mem::take(&mut slot.values),
                uncertainty: std::mem::replace(&mut slot.uncertainty, RetainedUncertainty::empty()),
            };
            let outcome = output(&result, &slot.report);
            slot.values = result.values;
            slot.uncertainty = result.uncertainty;
            outcome
        },
    )
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn prepare_estimate<
    M: Mode,
    C: Covariance,
    D: Drift,
    H: ObservationSource + ?Sized,
    T: Covariance,
    TD: Drift,
>(
    hard: &H,
    group: &Group,
    options: &EstimationOptions<'_>,
    target: &factorial::Component<'_, T, TD>,
    honor_exact: bool,
    worker: &mut Worker<'_>,
    conditioning: &mut NeighborhoodConditioning<'_, C, D>,
    slot: &mut Estimated<M>,
    diagnostic: &mut M,
) -> Result<()> {
    select_estimate(hard, group, options, worker, slot)?;
    estimate_selected(
        group,
        options,
        target,
        honor_exact,
        worker,
        conditioning,
        slot,
        diagnostic,
    )
}

fn select_estimate<M: Mode, H: ObservationSource + ?Sized>(
    hard: &H,
    group: &Group,
    options: &EstimationOptions<'_>,
    worker: &mut Worker<'_>,
    slot: &mut Estimated<M>,
) -> Result<()> {
    let center = group_center(group, options.cancellation)?;
    let selection = &mut slot.selection;
    selection.observations.clear();
    selection.extend(&worker.search.select(hard, center, options.cancellation)?)?;
    for target in &group.targets {
        options.cancellation.check()?;
        selection.extend(&hard.at_site(&target.site)?)?;
    }
    worker
        .canonical
        .compact(&mut selection.observations, true, options.cancellation)?;
    worker
        .canonical
        .sort_by_id(&mut selection.observations, options.cancellation)?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn estimate_selected<M: Mode, C: Covariance, D: Drift, T: Covariance, TD: Drift>(
    group: &Group,
    options: &EstimationOptions<'_>,
    target: &factorial::Component<'_, T, TD>,
    honor_exact: bool,
    worker: &mut Worker<'_>,
    conditioning: &mut NeighborhoodConditioning<'_, C, D>,
    slot: &mut Estimated<M>,
    diagnostic: &mut M,
) -> Result<()> {
    let Worker {
        budget: worker_budget,
        estimation,
        ..
    } = worker;
    let selection = &slot.selection;
    let observations = &selection.observations;
    let n = observations.len();
    let m = group.targets.len();
    conditioning.release_if_insufficient(n, m);
    let buffers = EstimationBuffers::cached(
        estimation,
        n,
        m,
        options.mean.is_some() || target.mean.is_some(),
        worker_budget,
    )?;
    buffers.fill(observations, group);
    let preparation = WorkspaceOptions {
        uncertainty: options.uncertainty,
        tolerance: options.tolerance,
        cancellation: options.cancellation,
    };
    let (plan, report) = diagnostic.prepare(
        conditioning,
        target,
        observations,
        &buffers.sites,
        worker_budget,
        preparation,
    )?;
    slot.report = report;
    grow(&mut slot.values, m, &mut slot.values_reservation)?;
    slot.values.resize(m, 0.);
    if honor_exact {
        buffers.predict(plan, observations, options.mean, &mut slot.values)?;
    } else {
        buffers.predict_component(
            plan,
            observations,
            options.mean,
            target.mean,
            &mut slot.values,
        )?;
    }
    if honor_exact {
        for (site, prediction) in buffers.sites.iter().zip(&mut slot.values) {
            if let Some(hard) = observations
                .iter()
                .find(|o| o.noise_variance() == 0. && o.site() == site)
            {
                *prediction = hard.value();
            }
        }
    }
    options.cancellation.check()?;
    slot.uncertainty.update(plan, &slot.budget)
}

#[cfg(test)]
mod tests;

pub mod factorial;
