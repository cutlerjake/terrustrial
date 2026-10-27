//! High-order grid simulation workflows; density inference remains in high_order.
use crate::{
    error::{checked_len, invalid},
    execution::grid::{simulate_grid_using, GridDraw, GridSimulationOptions},
    grid::{Boundary, Event, Grid, Variable},
    high_order::{
        aggregate::{self, Evidence, Unsupported},
        grid::{GridReplicates, Location},
        Axis, Diagnostics, FitOptions, Plan,
    },
    Error, Result,
};
pub struct SimulationOptions<'a> {
    /// Ordered neighbor priority; all known attributes at each chosen location
    /// enter the conditioning event. Center hard attributes always participate.
    pub template: &'a [[i64; 3]],
    pub maximum_neighbors: usize,
    pub training_boundary: Boundary,
    pub seed: u64,
    pub realization: u64,
    pub fit: FitOptions<'a>,
}
#[derive(Debug)]
pub struct Draw {
    pub values: Vec<(usize, f64)>,
    pub diagnostics: Diagnostics,
    pub aggregation: Option<Evidence>,
    /// Exact hard-target likelihood under the fitted joint mixture, when some
    /// target attributes were already known. This is not the event normalizer.
    pub hard_target_log_evidence: Option<f64>,
}
impl GridDraw for Draw {
    fn values(&self) -> &[(usize, f64)] {
        &self.values
    }
}

/// Sequential joint conditional-density simulation. The plan contains ALL grid
/// variables in grid order. A single mixture component supplies all unknown
/// attributes at a node. Fit the joint target distribution using spatial neighbors,
/// then condition that mixture on known target attributes by Bayes' rule. Hard
/// targets are retained bit-for-bit and never blurred by a smoothing kernel.
///
/// Continuous grid values must be pretransformed to [-1,1]; MPS mismatch scales
/// do not perform this transformation. The Legendre kernel smooths conditioning,
/// so exact hard-data retention is not a claim of exact continuous CPDF inference.
/// Failure leaves the completed prefix in state. Output errors occur after that
/// node's state update. No durable checkpoint or multiscale aggregation is implied.
pub fn simulate(
    plan: &Plan,
    training: &Grid,
    state: &mut Grid,
    path: impl IntoIterator<Item = usize>,
    options: SimulationOptions<'_>,
    output: impl FnMut(usize, &Draw) -> Result<()>,
) -> Result<()> {
    simulate_using(plan, training, state, path, options, None, output)
}

/// Controls how partially observed spatial patterns inform the high-order model.
#[derive(Clone, Copy)]
pub struct AggregationPolicy {
    pub minimum_replicates: u64,
    pub unsupported: Unsupported,
}
/// Simulate using nested-prefix evidence from a possibly sparse sample grid.
/// An exhaustive training image is not required. The template order determines
/// the hierarchy; use increasing distance when following the cited method.
/// Joint target attributes must all be observed at each retained source center.
/// Evidence diagnostics accompany every draw. The same prefix-preservation and
/// failure-state contract as `simulate` applies. Missing interaction coefficients
/// are never treated as empirically identified.
pub fn simulate_prefixes(
    plan: &Plan,
    samples: &Grid,
    state: &mut Grid,
    path: impl IntoIterator<Item = usize>,
    options: SimulationOptions<'_>,
    policy: AggregationPolicy,
    output: impl FnMut(usize, &Draw) -> Result<()>,
) -> Result<()> {
    simulate_using(plan, samples, state, path, options, Some(policy), output)
}
fn simulate_using(
    plan: &Plan,
    training: &Grid,
    state: &mut Grid,
    path: impl IntoIterator<Item = usize>,
    options: SimulationOptions<'_>,
    policy: Option<AggregationPolicy>,
    output: impl FnMut(usize, &Draw) -> Result<()>,
) -> Result<()> {
    options.fit.validate()?;
    if policy.is_some_and(|p| p.minimum_replicates == 0) {
        return Err(invalid("positive aggregation evidence floor required"));
    }
    if plan.axes().len() != training.variables().len() {
        return Err(invalid("high-order plan/grid variable count"));
    }
    for (&axis, variable) in plan.axes().iter().zip(training.variables()) {
        if !matches!(
            (axis, variable),
            (Axis::Continuous { .. }, Variable::Continuous { .. })
        ) && !matches!((axis.categorical(),variable), (Some((a,_)),Variable::Categorical {classes:b}) if a==*b)
        {
            return Err(invalid("high-order plan/grid variable definitions"));
        }
    }
    // Validate domain before mutation, including values in hard-only nodes and
    // incomplete TI rows which the complete-replicate policy might otherwise skip.
    for grid in [training, &*state] {
        for (i, value) in grid.values().iter().enumerate() {
            if i % 4096 == 0 {
                options.fit.cancellation.check()?;
            }
            if let Some(value) = value {
                plan.axes()[i % plan.axes().len()].validate_value(*value)?;
            }
        }
    }
    let n = plan.axes().len();
    let capacity = options
        .maximum_neighbors
        .min(options.template.len())
        .checked_add(1)
        .and_then(|v| v.checked_mul(n))
        .ok_or(Error::SizeOverflow)?;
    let bytes = checked_len(&[capacity, 64])?
        .checked_add(checked_len(&[n, 32])?)
        .ok_or(Error::SizeOverflow)?;
    let _reservation = options
        .fit
        .budget
        .reserve(u64::try_from(bytes).map_err(|_| Error::SizeOverflow)?)?;
    let targets: Vec<_> = (0..n).collect();
    let mut values = vec![0.0; n];
    let mut known = vec![None; n];
    simulate_grid_using(
        training,
        state,
        path,
        GridSimulationOptions {
            template: options.template,
            maximum_neighbors: options.maximum_neighbors,
            budget: options.fit.budget,
            cancellation: options.fit.cancellation,
        },
        |events: &[Event], requested, cell| {
            known.fill(None);
            for event in events.iter().filter(|e| e.offset == [0; 3]) {
                known[event.variable] = Some(event.value);
            }
            let locations: Vec<_> = events
                .iter()
                .filter(|e| e.offset != [0; 3])
                .map(|e| Location {
                    offset: e.offset,
                    variable: e.variable,
                })
                .collect();
            let event: Vec<_> = events
                .iter()
                .filter(|e| e.offset != [0; 3])
                .map(|e| (plan.axes()[e.variable], e.value))
                .collect();
            let source = GridReplicates::new(
                training,
                &targets,
                &locations,
                options.training_boundary,
                options.fit.budget,
                options.fit.cancellation,
            )?;
            let (fit, aggregation) = if let Some(policy) = policy {
                let result = plan.condition_prefixes(
                    &source,
                    &event,
                    aggregate::Options {
                        minimum_replicates: policy.minimum_replicates,
                        unsupported: policy.unsupported,
                        fit: options.fit,
                    },
                )?;
                (result.distribution, Some(result.evidence))
            } else {
                (plan.condition(&source, &event, options.fit)?, None)
            };
            let hard_target_log_evidence = if known.iter().any(Option::is_some) {
                let posterior =
                    fit.condition_targets(&known, options.fit.budget, options.fit.cancellation)?;
                posterior.draw_into(
                    options.seed,
                    options.realization,
                    cell as u64,
                    &mut values,
                    options.fit.cancellation,
                )?;
                Some(posterior.log_evidence)
            } else {
                fit.draw_into(
                    options.seed,
                    options.realization,
                    cell as u64,
                    &mut values,
                    options.fit.cancellation,
                )?;
                None
            };
            Ok(Draw {
                values: requested.iter().map(|&v| (v, values[v])).collect(),
                diagnostics: fit.diagnostics.clone(),
                aggregation,
                hard_target_log_evidence,
            })
        },
        output,
    )
}
