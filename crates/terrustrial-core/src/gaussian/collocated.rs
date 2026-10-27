//! Explicit collocated neighborhood approximation and a valid Markov-I model.
use crate::{
    data::{Observation, Site},
    error::invalid,
    model::{Anisotropy, Coregionalization, Correlation, Lmc, SpatialKernel, Structure},
    Result,
};

#[derive(Clone, Debug)]
pub struct CollocatedSelection {
    /// Original input indices, in input order. Use these to map reusable weights.
    pub indices: Vec<usize>,
    pub observations: Vec<Observation>,
    pub omitted_secondary: usize,
}
/// Keep every selected primary observation and secondary observations on a target
/// support. For joint targets, take the union of all target supports. The resulting
/// estimate uses the specified full covariance model on this reduced neighborhood.
/// This is an approximation whenever omitted secondary data carry information.
pub fn select(
    observations: &[Observation],
    targets: &[Site],
    primary_variables: &[usize],
) -> Result<CollocatedSelection> {
    if targets.is_empty() || primary_variables.is_empty() {
        return Err(invalid("collocated targets and primary variables required"));
    }
    let mut indices = Vec::new();
    let mut selected = Vec::new();
    for (index, observation) in observations.iter().enumerate() {
        if primary_variables.contains(&observation.site().variable)
            || targets
                .iter()
                .any(|target| target.support == observation.site().support)
        {
            indices.push(index);
            selected.push(*observation);
        }
    }
    Ok(CollocatedSelection {
        omitted_secondary: observations.len() - selected.len(),
        indices,
        observations: selected,
    })
}
/// X_i(s) = loading_i * Y(s) + epsilon_i(s), with epsilon independent of Y and
/// epsilon_0 identically zero. Thus C_0i(h)=loading_i/loading_0 * C_00(h), the
/// Markov-I cross-covariance relation. The residual sill is PSD by construction.
/// A nugget residual is the default independent-site approximation. This model
/// states a substantive assumption; it is not implied by collocated selection.
pub fn markov_one(
    primary: SpatialKernel,
    loadings: &[f64],
    residual: Coregionalization,
) -> Result<Lmc> {
    if loadings.is_empty()
        || !loadings[0].is_finite()
        || loadings[0] <= 0.0
        || residual.variables() != loadings.len()
    {
        return Err(invalid("Markov-I primary loading or residual dimensions"));
    }
    for variable in 0..loadings.len() {
        if residual.covariance(0, variable)? != 0.0 || residual.covariance(variable, 0)? != 0.0 {
            return Err(invalid(
                "Markov-I residual must have a zero primary row and column",
            ));
        }
    }
    Lmc::new(vec![
        Structure {
            correlation: primary.correlation,
            anisotropy: primary.anisotropy,
            coregionalization: Coregionalization::from_factor(loadings.len(), 1, loadings)?,
        },
        Structure {
            correlation: Correlation::Nugget,
            anisotropy: Anisotropy::isotropic(1.0)?,
            coregionalization: residual,
        },
    ])
}
