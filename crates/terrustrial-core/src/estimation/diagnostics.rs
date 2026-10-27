//! Optional numerical reports alongside ordered estimation output.
use super::*;
use crate::gaussian::ResidualWorkspace;

pub use crate::gaussian::GroupResiduals;

/// Estimate with all available residual checks on every group. Reports and
/// estimates arrive together in input order. No threshold or repair is applied.
/// Inspect the report before publishing output and return an error to reject a
/// group. Earlier callbacks remain committed; later groups are not emitted.
/// Callback/model/cancellation failures during checking emit no result for that
/// group. Scratch is reused per worker and charged to its `bytes_per_group`;
/// fixed report/scratch metadata is admitted separately from the parent budget.
/// Checking re-evaluates covariance and drift callbacks, which must be stable.
/// `Uncertainty::None` skips covariance/factor checks, but still checks weights
/// and both drift representations. Ordinary `estimate_groups` performs no checks.
#[allow(clippy::too_many_arguments)]
pub fn estimate_groups_with_residuals<
    C: Covariance,
    D: Drift,
    H: ObservationSource + Sync + ?Sized,
>(
    model: &C,
    hard: &H,
    groups: impl IntoIterator<Item = Group>,
    drift: &D,
    options: EstimationOptions<'_>,
    execution: impl Into<ExecutionConfig>,
    mut output: impl FnMut(&GroupEstimate, &GroupResiduals) -> Result<()>,
) -> Result<()> {
    let target = factorial::Component {
        covariance: model,
        drift,
        mean: options.mean,
    };
    estimate_with_component::<Checked, _, _, _, _, _>(
        model,
        hard,
        groups,
        drift,
        target,
        true,
        options,
        execution,
        |estimate, report| output(estimate, report.as_ref().expect("successful checked group")),
    )
}

// Static modes keep scratch, reports and checking code out of the default path.
pub(super) trait Mode: Default + Send {
    type Report: Default + Send;
    #[allow(clippy::too_many_arguments)]
    fn prepare<'w, C: Covariance, D: Drift, T: Covariance, TD: Drift>(
        &mut self,
        conditioning: &'w mut NeighborhoodConditioning<'_, C, D>,
        target: &factorial::Component<'_, T, TD>,
        observations: &[Observation],
        sites: &[Site],
        budget: &MemoryBudget,
        options: WorkspaceOptions<'_>,
    ) -> Result<(&'w GaussianPlan, Self::Report)>;
}
#[derive(Default)]
pub(super) struct Unchecked;
impl Mode for Unchecked {
    type Report = ();
    fn prepare<'w, C: Covariance, D: Drift, T: Covariance, TD: Drift>(
        &mut self,
        conditioning: &'w mut NeighborhoodConditioning<'_, C, D>,
        target: &factorial::Component<'_, T, TD>,
        observations: &[Observation],
        sites: &[Site],
        _: &MemoryBudget,
        options: WorkspaceOptions<'_>,
    ) -> Result<(&'w GaussianPlan, ())> {
        Ok((
            conditioning.prepare_component(
                target.covariance,
                target.drift,
                observations,
                sites,
                options,
            )?,
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
    fn prepare<'w, C: Covariance, D: Drift, T: Covariance, TD: Drift>(
        &mut self,
        conditioning: &'w mut NeighborhoodConditioning<'_, C, D>,
        target: &factorial::Component<'_, T, TD>,
        observations: &[Observation],
        sites: &[Site],
        budget: &MemoryBudget,
        options: WorkspaceOptions<'_>,
    ) -> Result<(&'w GaussianPlan, Self::Report)> {
        let scratch =
            ResidualWorkspace::cached(&mut self.scratch, observations.len(), sites.len(), budget)?;
        let (plan, equations) = conditioning.prepare_component_with_residuals(
            target.covariance,
            target.drift,
            observations,
            sites,
            scratch,
            options,
        )?;
        let covariance_observation_id = equations
            .covariance_equations
            .worst
            .map(|where_| observations[where_.row].id());
        Ok((
            plan,
            Some(GroupResiduals {
                equations,
                observations: observations.len(),
                covariance_observation_id,
            }),
        ))
    }
}
