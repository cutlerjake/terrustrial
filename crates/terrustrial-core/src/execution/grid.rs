//! Shared in-memory grid path execution; the sampler owns its statistical method.
use crate::{
    error::{checked_len, invalid},
    execution::{Cancellation, MemoryBudget},
    grid::{Boundary, Event, Grid},
    Error, Result,
};
use std::collections::HashSet;
pub(crate) trait GridDraw {
    fn values(&self) -> &[(usize, f64)];
}
pub(crate) struct GridSimulationOptions<'a> {
    pub template: &'a [[i64; 3]],
    pub maximum_neighbors: usize,
    pub budget: &'a MemoryBudget,
    pub cancellation: &'a Cancellation,
}
// The path and exact-hard-data contract is shared by pattern sampling and
// moment-based conditional density sampling. No TI candidate is invented for a
// density draw. The sampler returns only the requested previously unknown values.
pub(crate) fn simulate_grid_using<D: GridDraw>(
    training: &Grid,
    state: &mut Grid,
    path: impl IntoIterator<Item = usize>,
    options: GridSimulationOptions<'_>,
    mut sample: impl FnMut(&[Event], &[usize], usize) -> Result<D>,
    mut output: impl FnMut(usize, &D) -> Result<()>,
) -> Result<()> {
    if training.variables != state.variables || options.maximum_neighbors == 0 {
        return Err(invalid(
            "grid variable definitions or neighborhood parameters",
        ));
    }
    let event_capacity = options
        .maximum_neighbors
        .min(options.template.len())
        .checked_add(1)
        .and_then(|n| n.checked_mul(state.variables.len()))
        .ok_or(Error::SizeOverflow)?;
    let bytes = state
        .shape
        .cells
        .checked_add(checked_len(&[event_capacity, 96])?)
        .and_then(|n| n.checked_add(checked_len(&[options.template.len(), 64]).ok()?))
        .ok_or(Error::SizeOverflow)?;
    let _reservation = options
        .budget
        .reserve(u64::try_from(bytes).map_err(|_| Error::SizeOverflow)?)?;
    let mut offsets = HashSet::new();
    for offset in options.template {
        if *offset == [0; 3] || !offsets.insert(*offset) {
            return Err(invalid(
                "grid template contains the center or a repeated offset",
            ));
        }
    }
    let mut visited = vec![false; state.shape.cells];
    let variables = state.variables.len();
    for cell in path {
        options.cancellation.check()?;
        if cell >= state.shape.cells || visited[cell] {
            return Err(invalid("grid path contains an invalid or repeated cell"));
        }
        visited[cell] = true;
        let requested: Vec<_> = (0..variables)
            .filter(|&v| state.values[cell * variables + v].is_none())
            .collect();
        if requested.is_empty() {
            continue;
        }
        let mut events = Vec::new();
        for variable in 0..variables {
            if let Some(value) = state.values[cell * variables + variable] {
                events.push(Event {
                    offset: [0; 3],
                    variable,
                    value,
                    weight: 1.0,
                });
            }
        }
        let mut neighbors = 0;
        for &offset in options.template {
            let Some(neighbor) = state.shape.offset(cell, offset, Boundary::Bounded) else {
                continue;
            };
            let before = events.len();
            for variable in 0..variables {
                if let Some(value) = state.values[neighbor * variables + variable] {
                    events.push(Event {
                        offset,
                        variable,
                        value,
                        weight: 1.0,
                    });
                }
            }
            if events.len() > before {
                neighbors += 1;
                if neighbors == options.maximum_neighbors {
                    break;
                }
            }
        }
        let matched = sample(&events, &requested, cell)?;
        options.cancellation.check()?;
        for &(variable, value) in matched.values() {
            state.values[cell * variables + variable] = Some(value);
        }
        output(cell, &matched)?;
    }
    Ok(())
}
