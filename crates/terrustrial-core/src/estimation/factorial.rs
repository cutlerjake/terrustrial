//! Joint independent-component estimation through the shared parallel driver.
use super::*;

/// Target component of the total observation covariance model. The caller must
/// establish an independent decomposition: its covariance is both the component
/// prior and its cross covariance with total observations. A PSD check alone does
/// not establish that decomposition. Attributes retain their original IDs.
///
/// Drift columns correspond to the observation model's unknown coefficients.
/// To estimate a centered residual component under an unknown observation mean,
/// supply zero target drift values, with the same number of columns. To estimate
/// a component containing that mean, supply its target basis instead.
pub struct Component<'a, C, D> {
    pub covariance: &'a C,
    pub drift: &'a D,
    /// Component's known mean. None means zero; distinct from the total data mean.
    pub mean: Option<&'a dyn KnownMean>,
}

/// Parallel component prediction and joint error covariance within each group.
/// `options.mean` is the TOTAL observation mean; `component.mean` is the target
/// component mean. Unlike total-field estimation, component predictions at hard
/// sites are not overwritten with observed total values. The covariance, search,
/// canonicalization, conditioning cache, worker budgets and ordered output use
/// the same engine as ordinary estimation. Cross-component covariance is not
/// returned by separate calls; do not treat independently estimated components
/// as conditionally independent or independently draw their uncertainties.
#[allow(clippy::too_many_arguments)]
pub fn estimate_groups<C, D, H, T, TD>(
    model: &C,
    hard: &H,
    groups: impl IntoIterator<Item = Group>,
    drift: &D,
    component: Component<'_, T, TD>,
    options: EstimationOptions<'_>,
    execution: impl Into<ExecutionConfig>,
    mut output: impl FnMut(&GroupEstimate) -> Result<()>,
) -> Result<()>
where
    C: Covariance,
    D: Drift,
    T: Covariance,
    TD: Drift,
    H: ObservationSource + Sync + ?Sized,
{
    estimate_with_component::<Unchecked, _, _, _, _, _>(
        model,
        hard,
        groups,
        drift,
        component,
        false,
        options,
        execution,
        |out, ()| output(out),
    )
}

/// Component estimation with the same opt-in residual policy as
/// `super::estimate_groups_with_residuals`. Original drift checks use the total
/// observation basis and the component's target basis separately.
#[allow(clippy::too_many_arguments)]
pub fn estimate_groups_with_residuals<C, D, H, T, TD>(
    model: &C,
    hard: &H,
    groups: impl IntoIterator<Item = Group>,
    drift: &D,
    component: Component<'_, T, TD>,
    options: EstimationOptions<'_>,
    execution: impl Into<ExecutionConfig>,
    mut output: impl FnMut(&GroupEstimate, &GroupResiduals) -> Result<()>,
) -> Result<()>
where
    C: Covariance,
    D: Drift,
    T: Covariance,
    TD: Drift,
    H: ObservationSource + Sync + ?Sized,
{
    estimate_with_component::<diagnostics::Checked, _, _, _, _, _>(
        model,
        hard,
        groups,
        drift,
        component,
        false,
        options,
        execution,
        |out, report| output(out, report.as_ref().expect("successful checked group")),
    )
}
