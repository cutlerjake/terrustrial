//! Opt-in numerical checks, accepted in path order before store mutation.
use super::*;
pub use crate::gaussian::GroupResiduals;
use crate::gaussian::ResidualWorkspace;

/// Sequential simulation with an ordered, pre-write numerical acceptance callback.
/// Every newly simulated group is checked. Reject by returning an error from
/// `inspect`; that group is never appended or passed to `output`. Earlier groups
/// remain written. The callback runs after drawing but before mutation. It must
/// inspect before performing external side effects. Store append failures can
/// still leave a partial group; use the checkpointed API for atomic commits.
///
/// Reports check the Gaussian plan before exact hard-value binding, not the
/// random draw's statistical quality. No threshold, repair or model change is
/// automatic. Model/drift/mean callbacks must remain deterministic and immutable.
/// Scratch is budgeted and reused; default simulation performs none of this work.
#[allow(clippy::too_many_arguments)]
pub fn simulate_groups_with_residuals<C, D, S, H>(
    model: &C,
    hard: &H,
    simulated: &mut S,
    groups: impl IntoIterator<Item = Group>,
    drift: &D,
    options: SimulationOptions<'_>,
    mut inspect: impl FnMut(&Group, &GroupResiduals) -> Result<()>,
    output: impl FnMut(&[Observation]) -> Result<()>,
) -> Result<()>
where
    C: Covariance,
    D: Drift,
    S: ObservationSource + ObservationSink,
    H: ObservationSource + ?Sized,
{
    run_groups::<Checked, _, _, _, _>(
        model,
        hard,
        simulated,
        groups,
        drift,
        options,
        |group, report| inspect(group, report.as_ref().expect("checked plan")),
        output,
    )
}

/// Checked transactional simulation. Rejection, diagnostic errors and cancellation
/// leave the last committed prefix intact. Notifications occur after commit and
/// can fail after durability, just as in `simulate_checkpointed`.
///
/// `policy` must fingerprint the callback's complete rules and external inputs.
/// It is combined with `options.identity` under a separate versioned identity.
/// Changing it, or resuming an unchecked run here, is rejected. Previously
/// committed groups are skipped, not rechecked. A fingerprint cannot verify the
/// caller's description or make callback side effects transactional.
#[allow(clippy::too_many_arguments)]
pub fn simulate_checkpointed_with_residuals<C, D, H, S>(
    model: &C,
    hard: &H,
    simulated: &mut S,
    groups: impl IntoIterator<Item = Group>,
    drift: &D,
    mut options: CheckpointedOptions<'_>,
    policy: crate::identity::Fingerprint,
    mut inspect: impl FnMut(&Group, &GroupResiduals) -> Result<()>,
    committed: impl FnMut(&[Observation]) -> Result<()>,
) -> Result<()>
where
    C: Covariance,
    D: Drift,
    H: ObservationSource + ?Sized,
    S: crate::storage::CheckpointStore,
{
    options.identity = identity::checked_run(options.identity, policy);
    run_checkpointed::<Checked, _, _, _, _>(
        model,
        hard,
        simulated,
        groups.into_iter().map(Ok),
        drift,
        options,
        |group, report| inspect(group, report.as_ref().expect("checked plan")),
        committed,
    )
}

/// Identified counterpart of checked checkpointing. Hashes the full hard data
/// and fallible full path before simulation or resume; also binds `policy`.
/// Path factories, callbacks and sources follow the same immutability contract
/// as `simulate_checkpointed_identified`.
#[allow(clippy::too_many_arguments)]
pub fn simulate_checkpointed_identified_with_residuals<C, D, H, G, S>(
    model: &C,
    hard: &H,
    simulated: &mut S,
    groups: impl Fn() -> Result<G>,
    drift: &D,
    options: SimulationOptions<'_>,
    provenance: identity::RunProvenance,
    policy: crate::identity::Fingerprint,
    mut inspect: impl FnMut(&Group, &GroupResiduals) -> Result<()>,
    committed: impl FnMut(&[Observation]) -> Result<()>,
) -> Result<()>
where
    C: Covariance + crate::identity::CanonicalFingerprint,
    D: Drift + crate::identity::CanonicalFingerprint,
    H: ObservationSource + ?Sized,
    S: crate::storage::CheckpointStore,
    G: IntoIterator<Item = Result<Group>>,
{
    options.validate()?;
    provenance.validate_mean(options.mean.is_some())?;
    let hard_id = crate::identity::observations(hard, options.cancellation)?;
    let path_id = crate::identity::try_path(groups()?, options.cancellation)?;
    let run = identity::gaussian_run(model, drift, hard_id, path_id, &options, provenance)?;
    run_checkpointed::<Checked, _, _, _, _>(
        model,
        hard,
        simulated,
        groups()?,
        drift,
        CheckpointedOptions {
            identity: identity::checked_run(run.to_bytes(), policy),
            simulation: options,
        },
        |group, report| inspect(group, report.as_ref().expect("checked plan")),
        committed,
    )
}

pub(super) trait Mode: Default + Send {
    type Report: Default + Send;
    #[allow(clippy::too_many_arguments)]
    fn prepare<'w, C: Covariance, D: Drift>(
        &mut self,
        solver: &'w mut GaussianWorkspace,
        model: &C,
        observations: &[Observation],
        sites: &[Site],
        drift: &D,
        budget: &MemoryBudget,
        options: WorkspaceOptions<'_>,
    ) -> Result<(&'w GaussianPlan, Self::Report)>;
}
#[derive(Default)]
pub(super) struct Unchecked;
impl Mode for Unchecked {
    type Report = ();
    fn prepare<'w, C: Covariance, D: Drift>(
        &mut self,
        solver: &'w mut GaussianWorkspace,
        model: &C,
        observations: &[Observation],
        sites: &[Site],
        drift: &D,
        _: &MemoryBudget,
        options: WorkspaceOptions<'_>,
    ) -> Result<(&'w GaussianPlan, ())> {
        Ok((
            solver.prepare(model, observations, sites, drift, options)?,
            (),
        ))
    }
}
#[derive(Default)]
pub(super) struct Checked {
    scratch: Option<ResidualWorkspace>,
}
impl Mode for Checked {
    type Report = Option<GroupResiduals>;
    fn prepare<'w, C: Covariance, D: Drift>(
        &mut self,
        solver: &'w mut GaussianWorkspace,
        model: &C,
        observations: &[Observation],
        sites: &[Site],
        drift: &D,
        budget: &MemoryBudget,
        options: WorkspaceOptions<'_>,
    ) -> Result<(&'w GaussianPlan, Self::Report)> {
        let scratch =
            ResidualWorkspace::cached(&mut self.scratch, observations.len(), sites.len(), budget)?;
        let (plan, equations) =
            solver.prepare_with_residuals(model, observations, sites, drift, scratch, options)?;
        let report = GroupResiduals {
            covariance_observation_id: equations
                .covariance_equations
                .worst
                .map(|w| observations[w.row].id()),
            observations: observations.len(),
            equations,
        };
        Ok((plan, Some(report)))
    }
}
