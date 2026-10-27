//! Joint mixed-variable k-best pattern sampling. This exhaustive scalar engine is
//! the correctness oracle for accelerated QuickSampling-style matching, not yet a
//! production-scale MPS implementation. Boundaries and missing data are explicit.
#[cfg(feature = "fft")]
pub mod fft;
#[cfg(feature = "gpu")]
pub mod gpu;
use crate::{
    error::{checked_len, invalid},
    execution::{random_stream, Cancellation, MemoryBudget},
    Error, Result,
};
use rand::RngExt;
use std::{
    cmp::Ordering,
    collections::{BinaryHeap, HashSet},
};

use crate::grid::{Boundary, Event, Grid, Variable};
pub struct MatchOptions<'a> {
    pub neighbors: usize,
    pub boundary: Boundary,
    pub seed: u64,
    pub realization: u64,
    pub query_id: u64,
    pub budget: &'a MemoryBudget,
    pub cancellation: &'a Cancellation,
}
#[derive(Clone, Debug)]
pub struct Match {
    pub index: usize,
    pub mismatch: f64,
}
#[derive(Clone, Debug)]
struct Candidate {
    matched: Match,
    tie: u128,
}
impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for Candidate {}
impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.matched
            .mismatch
            .total_cmp(&other.matched.mismatch)
            .then(self.tie.cmp(&other.tie))
            .then(self.matched.index.cmp(&other.matched.index))
    }
}
/// Exact scalar top-k for the specified score. All event cells must be present in
/// the TI. No valid candidate is an error, not permission to drop conditioning.
/// Runtime O(TI cells * events); heap storage O(k). Equal-score candidates use seeded
/// random priorities, avoiding first-index bias. Empty events sample the TI marginal.
pub fn best_matches(
    training: &Grid,
    events: &[Event],
    requested: &[usize],
    options: &MatchOptions<'_>,
) -> Result<Vec<Match>> {
    options.cancellation.check()?;
    if options.neighbors == 0
        || requested.is_empty()
        || requested.iter().any(|v| *v >= training.variables.len())
    {
        return Err(invalid("MPS neighbors or requested attributes"));
    }
    let count = options.neighbors.min(training.shape.cells);
    let bytes = checked_len(&[count, 64])?
        .checked_add(checked_len(&[events.len(), 80])?)
        .and_then(|bytes| bytes.checked_add(checked_len(&[requested.len(), 64]).ok()?))
        .ok_or(Error::SizeOverflow)?;
    let _reservation = options
        .budget
        .reserve(u64::try_from(bytes).map_err(|_| Error::SizeOverflow)?)?;
    let weights = validate_query(training, events, requested, options.neighbors)?;
    let mut heap = BinaryHeap::<Candidate>::new();
    heap.try_reserve(count)
        .map_err(|_| invalid("unable to allocate MPS match heap"))?;
    let mut rng = random_stream(options.seed, options.realization, options.query_id, 2);
    for index in 0..training.shape.cells {
        options.cancellation.check()?;
        let tie = rng.random::<u128>();
        if requested
            .iter()
            .any(|variable| training.values[index * training.variables.len() + variable].is_none())
        {
            continue;
        }
        let mut score = 0.0;
        let mut valid = true;
        for (event, weight) in events.iter().zip(&weights) {
            let Some(cell) = training.shape.offset(index, event.offset, options.boundary) else {
                valid = false;
                break;
            };
            let Some(value) = training.values[cell * training.variables.len() + event.variable]
            else {
                valid = false;
                break;
            };
            score += weight * training.variables[event.variable].mismatch(value, event.value)?;
        }
        if !valid {
            continue;
        }
        if !score.is_finite() {
            return Err(Error::Numerical {
                context: "MPS weighted score",
            });
        }
        let candidate = Candidate {
            matched: Match {
                index,
                mismatch: score,
            },
            tie,
        };
        push_candidate(&mut heap, count, candidate);
    }
    if heap.is_empty() {
        return Err(invalid(
            "no training candidate supports the full conditioning event and requested attributes",
        ));
    }
    Ok(heap
        .into_sorted_vec()
        .into_iter()
        .map(|candidate| candidate.matched)
        .collect())
}
#[derive(Clone, Debug)]
pub struct JointDraw {
    pub source_index: usize,
    pub mismatch: f64,
    pub values: Vec<(usize, f64)>,
}
pub fn draw(
    training: &Grid,
    events: &[Event],
    requested: &[usize],
    options: &MatchOptions<'_>,
) -> Result<JointDraw> {
    let matches = best_matches(training, events, requested, options)?;
    select_match(training, &matches, requested, options)
}
fn select_match(
    training: &Grid,
    matches: &[Match],
    requested: &[usize],
    options: &MatchOptions<'_>,
) -> Result<JointDraw> {
    let mut rng = random_stream(options.seed, options.realization, options.query_id, 3);
    let selected = &matches[rng.random_range(0..matches.len())];
    let values = requested
        .iter()
        .map(|&variable| {
            (
                variable,
                training.values[selected.index * training.variables.len() + variable].unwrap(),
            )
        })
        .collect();
    Ok(JointDraw {
        source_index: selected.index,
        mismatch: selected.mismatch,
        values,
    })
}
fn validate_query(
    training: &Grid,
    events: &[Event],
    requested: &[usize],
    neighbors: usize,
) -> Result<Vec<f64>> {
    if neighbors == 0
        || requested.is_empty()
        || requested.iter().any(|v| *v >= training.variables.len())
    {
        return Err(invalid("MPS neighbors or requested attributes"));
    }
    let mut seen = HashSet::new();
    if requested.iter().any(|v| !seen.insert(*v)) {
        return Err(invalid("repeated requested MPS attribute"));
    }
    let mut seen = HashSet::new();
    let mut maximum = 0.0_f64;
    for event in events {
        if !seen.insert((event.offset, event.variable)) {
            return Err(invalid("repeated MPS conditioning event"));
        }
        training
            .variables
            .get(event.variable)
            .ok_or_else(|| invalid("MPS event variable"))?
            .validate_value(event.value)?;
        if !event.weight.is_finite() || event.weight <= 0.0 {
            return Err(invalid("MPS event weight"));
        }
        maximum = maximum.max(event.weight);
    }
    let mut weights: Vec<_> = events.iter().map(|event| event.weight / maximum).collect();
    if weights.contains(&0.0) {
        return Err(invalid(
            "MPS weight ratio exceeds floating-point resolution",
        ));
    }
    let total: f64 = weights.iter().sum();
    for weight in &mut weights {
        *weight /= total;
    }
    if weights.contains(&0.0) {
        return Err(invalid(
            "normalized MPS weight is below floating-point resolution",
        ));
    }
    Ok(weights)
}

fn push_candidate(heap: &mut BinaryHeap<Candidate>, count: usize, candidate: Candidate) {
    if heap.len() < count {
        heap.push(candidate);
    } else if heap.peek().is_some_and(|worst| candidate < *worst) {
        heap.pop();
        heap.push(candidate);
    }
}
/// Direct validation/refinement shared by accelerators. The exhaustive oracle
/// retains its own event traversal so accelerated-score tests remain independent.
#[cfg(any(feature = "gpu", feature = "fft"))]
fn score_candidate(
    training: &Grid,
    events: &[Event],
    weights: &[f64],
    requested: &[usize],
    index: usize,
    boundary: Boundary,
) -> Result<Option<f64>> {
    if requested
        .iter()
        .any(|&v| training.values[index * training.variables.len() + v].is_none())
    {
        return Ok(None);
    }
    let mut score = 0.0;
    for (event, weight) in events.iter().zip(weights) {
        let Some(cell) = training.shape.offset(index, event.offset, boundary) else {
            return Ok(None);
        };
        let Some(value) = training.values[cell * training.variables.len() + event.variable] else {
            return Ok(None);
        };
        score += weight * training.variables[event.variable].mismatch(value, event.value)?;
    }
    if !score.is_finite() {
        return Err(Error::Numerical {
            context: "MPS refined pattern score",
        });
    }
    Ok(Some(score))
}

impl Variable {
    pub(crate) fn mismatch(&self, a: f64, b: f64) -> Result<f64> {
        let value = match self {
            Self::Categorical { .. } => {
                if a == b {
                    0.0
                } else {
                    1.0
                }
            }
            Self::Continuous { scale } => {
                let difference = a - b;
                let normalized = if difference.is_finite() {
                    difference / scale
                } else {
                    a / scale - b / scale
                };
                normalized * normalized
            }
        };
        if !value.is_finite() {
            return Err(Error::Numerical {
                context: "MPS mismatch exceeds floating-point range",
            });
        }
        Ok(value)
    }
}
