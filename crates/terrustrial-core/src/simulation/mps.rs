//! Pattern-sampling workflows over shared grid execution.
use crate::{
    error::invalid,
    execution::{
        grid::{simulate_grid_using, GridDraw, GridSimulationOptions},
        Cancellation, MemoryBudget,
    },
    grid::{Boundary, Event, Grid},
    mps::{draw, JointDraw, MatchOptions},
    Result,
};
impl GridDraw for JointDraw {
    fn values(&self) -> &[(usize, f64)] {
        &self.values
    }
}
pub struct SimulationOptions<'a> {
    /// Ordered offsets select neighbor priority. All available attributes at each
    /// chosen location participate in the joint event. The center is added separately.
    pub template: &'a [[i64; 3]],
    pub maximum_neighbors: usize,
    pub matching_neighbors: usize,
    pub training_boundary: Boundary,
    pub seed: u64,
    pub realization: u64,
    pub budget: &'a MemoryBudget,
    pub cancellation: &'a Cancellation,
}
/// Sequential joint simulation on a supplied path. Hard values and already known
/// attributes remain bit-for-bit unchanged. Simulation-grid boundaries never wrap.
/// The mutable grid is currently in memory; durable MPS checkpoints are not implied.
pub fn simulate(
    training: &Grid,
    state: &mut Grid,
    path: impl IntoIterator<Item = usize>,
    options: SimulationOptions<'_>,
    output: impl FnMut(usize, &JointDraw) -> Result<()>,
) -> Result<()> {
    simulate_using(
        training,
        state,
        path,
        options,
        |events, requested, options| draw(training, events, requested, options),
        output,
    )
}
pub(crate) fn simulate_using(
    training: &Grid,
    state: &mut Grid,
    path: impl IntoIterator<Item = usize>,
    options: SimulationOptions<'_>,
    mut sample: impl FnMut(&[Event], &[usize], &MatchOptions<'_>) -> Result<JointDraw>,
    output: impl FnMut(usize, &JointDraw) -> Result<()>,
) -> Result<()> {
    if options.matching_neighbors == 0 {
        return Err(invalid("MPS matching neighbor count"));
    }
    simulate_grid_using(
        training,
        state,
        path,
        GridSimulationOptions {
            template: options.template,
            maximum_neighbors: options.maximum_neighbors,
            budget: options.budget,
            cancellation: options.cancellation,
        },
        |events, requested, cell| {
            sample(
                events,
                requested,
                &MatchOptions {
                    neighbors: options.matching_neighbors,
                    boundary: options.training_boundary,
                    seed: options.seed,
                    realization: options.realization,
                    query_id: cell as u64,
                    budget: options.budget,
                    cancellation: options.cancellation,
                },
            )
        },
        output,
    )
}

#[cfg(feature = "fft")]
pub fn simulate_fft_approximate(
    matcher: &mut crate::mps::fft::Matcher<'_>,
    state: &mut Grid,
    path: impl IntoIterator<Item = usize>,
    options: SimulationOptions<'_>,
    refinement: crate::mps::fft::Refinement,
    output: impl FnMut(usize, &JointDraw) -> Result<()>,
) -> Result<()> {
    let training = matcher.training();
    simulate_using(
        training,
        state,
        path,
        options,
        |events, requested, options| {
            matcher.draw_approximate(events, requested, options, refinement)
        },
        output,
    )
}

#[cfg(feature = "gpu")]
pub fn simulate_gpu_approximate(
    matcher: &mut crate::mps::gpu::Matcher<'_>,
    state: &mut Grid,
    path: impl IntoIterator<Item = usize>,
    options: SimulationOptions<'_>,
    refinement_tolerance: f64,
    output: impl FnMut(usize, &JointDraw) -> Result<()>,
) -> Result<()> {
    let training = matcher.training();
    simulate_using(
        training,
        state,
        path,
        options,
        |events, requested, options| {
            matcher.draw_approximate(events, requested, options, refinement_tolerance)
        },
        output,
    )
}
