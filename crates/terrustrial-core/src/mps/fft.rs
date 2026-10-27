//! Cached Fourier pattern scores for repeated queries against an immutable TI.
//!
//! This backend screens candidates approximately, then recomputes selected scores
//! with the scalar metric. The screening tolerance is a user-controlled heuristic,
//! not a proven bound on RustFFT error. Use `mps::best_matches` when exact scalar
//! ranking is required. Returned candidates always satisfy the complete event and
//! requested-value availability, including bounded-grid edges and missing values.
use super::*;
use crate::{execution::Reservation, grid::Shape};
use rustfft::{num_complex::Complex64, Fft, FftPlanner};
use std::{collections::BTreeSet, sync::Arc};

/// Enlarges the approximate top-k score threshold before scalar refinement.
/// The relative term multiplies a cancellation scale for the whole query.
#[derive(Clone, Copy, Debug)]
pub struct Refinement {
    pub absolute: f64,
    pub relative: f64,
}
impl Default for Refinement {
    fn default() -> Self {
        Self {
            absolute: 1e-10,
            relative: 1e-10,
        }
    }
}
#[derive(Debug)]
pub struct MatchReport {
    pub matches: Vec<Match>,
    pub refined_candidates: usize,
    pub screening_tolerance: f64,
}

enum Feature {
    Linear { center: f64, scale: f64 },
    Square { center: f64, scale: f64 },
    Category { category: u32, reference: u32 },
    Present,
}
struct Spectrum {
    variable: usize,
    feature: Feature,
    values: Vec<Complex64>,
}
struct Transform {
    shape: Shape,
    forward: Vec<Arc<dyn Fft<f64>>>,
    inverse: Vec<Arc<dyn Fft<f64>>>,
    line: Vec<Complex64>,
    scratch: Vec<Complex64>,
}
impl Transform {
    fn new(shape: Shape, budget: &MemoryBudget) -> Result<(Self, Reservation)> {
        let mut planner = FftPlanner::new();
        let forward: Vec<_> = shape
            .dimensions
            .iter()
            .map(|&n| planner.plan_fft_forward(n))
            .collect();
        let inverse: Vec<_> = shape
            .dimensions
            .iter()
            .map(|&n| planner.plan_fft_inverse(n))
            .collect();
        let scratch = forward
            .iter()
            .chain(&inverse)
            .map(|plan| plan.get_inplace_scratch_len())
            .max()
            .unwrap();
        let line = *shape.dimensions.iter().max().unwrap();
        let bytes = checked_len(&[line.checked_add(scratch).ok_or(Error::SizeOverflow)?, 16])?;
        let reservation = budget.reserve(u64::try_from(bytes).map_err(|_| Error::SizeOverflow)?)?;
        Ok((
            Self {
                shape,
                forward,
                inverse,
                line: vec![Complex64::default(); line],
                scratch: vec![Complex64::default(); scratch],
            },
            reservation,
        ))
    }
    fn apply(
        &mut self,
        values: &mut [Complex64],
        inverse: bool,
        cancel: &Cancellation,
    ) -> Result<()> {
        let [nx, ny, nz] = self.shape.dimensions;
        let plans = if inverse {
            &self.inverse
        } else {
            &self.forward
        };
        cancel.check()?;
        if nx > 1 {
            for (row_index, row) in values.chunks_exact_mut(nx).enumerate() {
                if row_index % 4096 == 0 {
                    cancel.check()?;
                }
                plans[0].process_with_scratch(row, &mut self.scratch);
            }
        }
        if ny > 1 {
            for z in 0..nz {
                cancel.check()?;
                for x in 0..nx {
                    for y in 0..ny {
                        self.line[y] = values[x + nx * (y + ny * z)];
                    }
                    plans[1].process_with_scratch(&mut self.line[..ny], &mut self.scratch);
                    for y in 0..ny {
                        values[x + nx * (y + ny * z)] = self.line[y];
                    }
                }
            }
        }
        if nz > 1 {
            for y in 0..ny {
                cancel.check()?;
                for x in 0..nx {
                    for z in 0..nz {
                        self.line[z] = values[x + nx * (y + ny * z)];
                    }
                    plans[2].process_with_scratch(&mut self.line[..nz], &mut self.scratch);
                    for z in 0..nz {
                        values[x + nx * (y + ny * z)] = self.line[z];
                    }
                }
            }
        }
        if inverse {
            let divisor = self.shape.cells as f64;
            for value in values {
                *value /= divisor;
            }
        }
        cancel.check()
    }
}

/// In-memory spectra and reusable workspace. One matcher processes one query at
/// a time. It borrows the TI, so cached spectra cannot become stale after edits.
/// Storage is O(cells * (continuous channels * 2 + observed categories + masks)).
/// RustFFT plans/allocator metadata are outside the logical payload budget.
/// No GPU, out-of-core FFT, or bitwise cross-hardware agreement is implied.
pub struct Matcher<'a> {
    training: &'a Grid,
    spectra: Vec<Spectrum>,
    /// (center, scale, largest normalized absolute value), continuous channels only.
    normalization: Vec<Option<(f64, f64, f64)>>,
    categorical_reference: Vec<Option<u32>>,
    transform: Transform,
    pattern: Vec<Complex64>,
    scores: Vec<Complex64>,
    present: Vec<Complex64>,
    _reservation: Reservation,
    _scratch_reservation: Reservation,
}
fn normalized(value: f64, center: f64, scale: f64) -> f64 {
    let difference = value - center;
    if difference.is_finite() {
        difference / scale
    } else {
        value / scale - center / scale
    }
}
fn numerical() -> Error {
    Error::Numerical {
        context: "FFT MPS score range or roundoff; use the scalar matcher",
    }
}

impl<'a> Matcher<'a> {
    pub fn new(training: &'a Grid, budget: &MemoryBudget, cancel: &Cancellation) -> Result<Self> {
        cancel.check()?;
        let n = training.shape.cells;
        let variables = training.variables.len();
        // Upper bound on temporary ordered category sets and feature descriptors.
        let _metadata = budget.reserve(
            u64::try_from(checked_len(&[training.values.len(), 64])?)
                .map_err(|_| Error::SizeOverflow)?,
        )?;
        let mut features = Vec::new();
        let mut normalization = vec![None; variables];
        let mut categorical_reference = vec![None; variables];
        for (variable, kind) in training.variables.iter().enumerate() {
            cancel.check()?;
            let mut missing = false;
            match kind {
                Variable::Continuous { scale } => {
                    let mut minimum = f64::INFINITY;
                    let mut maximum = f64::NEG_INFINITY;
                    for cell in training.values.chunks_exact(variables) {
                        if let Some(value) = cell[variable] {
                            minimum = minimum.min(value);
                            maximum = maximum.max(value);
                        } else {
                            missing = true;
                        }
                    }
                    let center = if minimum.is_finite() {
                        minimum * 0.5 + maximum * 0.5
                    } else {
                        0.0
                    };
                    let amplitude = if minimum.is_finite() {
                        normalized(minimum, center, *scale)
                            .abs()
                            .max(normalized(maximum, center, *scale).abs())
                    } else {
                        0.0
                    };
                    // Keep both forward spectra and their products representable.
                    if !amplitude.is_finite() || amplitude > f64::MAX.sqrt() / (n as f64 * 16.0) {
                        return Err(numerical());
                    }
                    normalization[variable] = Some((center, *scale, amplitude));
                    features.push((
                        variable,
                        Feature::Linear {
                            center,
                            scale: *scale,
                        },
                    ));
                    features.push((
                        variable,
                        Feature::Square {
                            center,
                            scale: *scale,
                        },
                    ));
                }
                Variable::Categorical { .. } => {
                    let mut categories = BTreeSet::new();
                    for cell in training.values.chunks_exact(variables) {
                        if let Some(value) = cell[variable] {
                            categories.insert(value as u32);
                        } else {
                            missing = true;
                        }
                    }
                    if let Some(&reference) = categories.first() {
                        categorical_reference[variable] = Some(reference);
                        for category in categories
                            .into_iter()
                            .filter(|category| *category != reference)
                        {
                            features.push((
                                variable,
                                Feature::Category {
                                    category,
                                    reference,
                                },
                            ));
                        }
                    }
                }
            }
            if missing {
                features.push((variable, Feature::Present));
            }
        }
        let bytes = checked_len(&[
            n,
            features.len().checked_add(3).ok_or(Error::SizeOverflow)?,
            16,
        ])?
        .checked_add(checked_len(&[variables, 64])?)
        .and_then(|bytes| {
            bytes.checked_add(checked_len(&[features.len(), std::mem::size_of::<Spectrum>()]).ok()?)
        })
        .ok_or(Error::SizeOverflow)?;
        let reservation = budget.reserve(u64::try_from(bytes).map_err(|_| Error::SizeOverflow)?)?;
        let (mut transform, scratch_reservation) = Transform::new(training.shape, budget)?;
        let mut spectra = Vec::new();
        for (variable, feature) in features {
            let mut values = vec![Complex64::default(); n];
            for (cell, output) in training.values.chunks_exact(variables).zip(&mut values) {
                output.re = match (&feature, cell[variable]) {
                    (Feature::Linear { center, scale }, Some(value)) => {
                        normalized(value, *center, *scale)
                    }
                    (Feature::Square { center, scale }, Some(value)) => {
                        normalized(value, *center, *scale).powi(2)
                    }
                    (Feature::Category { category, .. }, Some(value)) => {
                        f64::from(*category == value as u32)
                    }
                    (Feature::Present, Some(_)) => 1.0,
                    _ => 0.0,
                };
            }
            transform.apply(&mut values, false, cancel)?;
            spectra.push(Spectrum {
                variable,
                feature,
                values,
            });
        }
        Ok(Self {
            training,
            spectra,
            normalization,
            categorical_reference,
            transform,
            pattern: vec![Complex64::default(); n],
            scores: vec![Complex64::default(); n],
            present: vec![Complex64::default(); n],
            _reservation: reservation,
            _scratch_reservation: scratch_reservation,
        })
    }

    /// Draw all requested attributes from one selected TI center, using the same
    /// stream contract as the reference sampler. Ranking is approximate as above.
    pub fn draw_approximate(
        &mut self,
        events: &[Event],
        requested: &[usize],
        options: &MatchOptions<'_>,
        refinement: Refinement,
    ) -> Result<JointDraw> {
        let report = self.best_matches_approximate(events, requested, options, refinement)?;
        select_match(self.training, &report.matches, requested, options)
    }

    /// Immutable training data used by this reusable matching engine.
    pub fn training(&self) -> &'a Grid {
        self.training
    }

    /// Approximate top-k screening followed by exact scalar score/validity checks.
    /// A wider tolerance refines more candidates. A tolerance of zero is permitted
    /// but can change ties and near-ties. This method does not promise the exact
    /// scalar global top-k for any finite tolerance. Errors leave the cache reusable.
    pub fn best_matches_approximate(
        &mut self,
        events: &[Event],
        requested: &[usize],
        options: &MatchOptions<'_>,
        refinement: Refinement,
    ) -> Result<MatchReport> {
        options.cancellation.check()?;
        if !refinement.absolute.is_finite()
            || refinement.absolute < 0.0
            || !refinement.relative.is_finite()
            || refinement.relative < 0.0
        {
            return Err(invalid("FFT refinement tolerances"));
        }
        if events.is_empty() {
            return Ok(MatchReport {
                matches: super::best_matches(self.training, events, requested, options)?,
                refined_candidates: self.training.shape.cells,
                screening_tolerance: 0.0,
            });
        }
        let n = self.training.shape.cells;
        let count = options.neighbors.min(n);
        let bytes = checked_len(&[events.len(), 128])?
            .checked_add(checked_len(&[count, 128])?)
            .and_then(|bytes| bytes.checked_add(checked_len(&[requested.len(), 64]).ok()?))
            .ok_or(Error::SizeOverflow)?;
        let _reservation = options
            .budget
            .reserve(u64::try_from(bytes).map_err(|_| Error::SizeOverflow)?)?;
        let weights = validate_query(self.training, events, requested, options.neighbors)?;
        // Intersection of all valid center boxes exactly implements bounded TI.
        let mut lower = [0_i128; 3];
        let mut upper = self.training.shape.dimensions.map(|n| n as i128);
        let mut pattern_cells = Vec::with_capacity(events.len());
        let mut constant = 0.0;
        let mut cancellation_scale = 0.0;
        for (event, weight) in events.iter().zip(&weights) {
            if options.boundary == Boundary::Bounded {
                for axis in 0..3 {
                    lower[axis] = lower[axis].max(-(event.offset[axis] as i128));
                    upper[axis] = upper[axis].min(
                        self.training.shape.dimensions[axis] as i128 - event.offset[axis] as i128,
                    );
                }
            }
            let p = std::array::from_fn(|axis| {
                (-(event.offset[axis] as i128))
                    .rem_euclid(self.training.shape.dimensions[axis] as i128)
                    as usize
            });
            pattern_cells.push(self.training.shape.index(p)?);
            if let Some((center, scale, amplitude)) = self.normalization[event.variable] {
                let value = normalized(event.value, center, scale);
                constant += weight * value * value;
                cancellation_scale += weight * (amplitude + value.abs()).powi(2);
            } else {
                if self.categorical_reference[event.variable] != Some(event.value as u32) {
                    constant += weight;
                }
                cancellation_scale += weight;
            }
        }
        if (0..3).any(|axis| lower[axis] >= upper[axis]) {
            return Err(invalid("no bounded TI center supports the full event"));
        }
        let tolerance = refinement.absolute + refinement.relative * cancellation_scale;
        if !constant.is_finite()
            || !cancellation_scale.is_finite()
            || !tolerance.is_finite()
            || cancellation_scale > f64::MAX / (n as f64 * 32.0)
        {
            return Err(numerical());
        }
        self.scores.fill(Complex64::default());
        self.present.fill(Complex64::default());
        let mut missing_events = 0_usize;
        for spectrum in &self.spectra {
            self.pattern.fill(Complex64::default());
            let mut active = false;
            for ((event, weight), &cell) in events.iter().zip(&weights).zip(&pattern_cells) {
                if event.variable != spectrum.variable {
                    continue;
                }
                let coefficient = match spectrum.feature {
                    Feature::Linear { center, scale } => {
                        -2.0 * weight * normalized(event.value, center, scale)
                    }
                    Feature::Square { .. } => *weight,
                    Feature::Category {
                        category,
                        reference,
                    } => {
                        if event.value as u32 == reference {
                            *weight
                        } else if event.value as u32 == category {
                            -*weight
                        } else {
                            0.0
                        }
                    }
                    Feature::Present => {
                        missing_events += 1;
                        1.0
                    }
                };
                if coefficient != 0.0 {
                    active = true;
                    self.pattern[cell].re += coefficient;
                }
            }
            if !active {
                continue;
            }
            self.transform
                .apply(&mut self.pattern, false, options.cancellation)?;
            let destination = if matches!(spectrum.feature, Feature::Present) {
                &mut self.present
            } else {
                &mut self.scores
            };
            for ((output, a), b) in destination
                .iter_mut()
                .zip(&spectrum.values)
                .zip(&self.pattern)
            {
                *output += a * b;
            }
        }
        self.transform
            .apply(&mut self.scores, true, options.cancellation)?;
        if missing_events > 0 {
            self.transform
                .apply(&mut self.present, true, options.cancellation)?;
        }
        let valid = |index: usize| -> bool {
            let p = self.training.shape.coordinates(index).unwrap();
            (0..3).all(|axis| (p[axis] as i128) >= lower[axis] && (p[axis] as i128) < upper[axis])
                && requested.iter().all(|&v| {
                    self.training.values[index * self.training.variables.len() + v].is_some()
                })
                && (missing_events == 0
                    || (self.present[index].re - missing_events as f64).abs() < 0.25)
        };
        let mut heap = BinaryHeap::<Candidate>::new();
        heap.try_reserve(count)
            .map_err(|_| invalid("unable to allocate FFT MPS heap"))?;
        for index in 0..n {
            if index % 4096 == 0 {
                options.cancellation.check()?;
            }
            if !valid(index) {
                continue;
            }
            let score = self.scores[index].re + constant;
            if !score.is_finite() {
                return Err(numerical());
            }
            push_candidate(
                &mut heap,
                count,
                Candidate {
                    matched: Match {
                        index,
                        mismatch: score,
                    },
                    tie: 0,
                },
            );
        }
        let threshold = heap
            .peek()
            .ok_or_else(|| {
                invalid(
                    "no TI candidate supports the full conditioning event and requested attributes",
                )
            })?
            .matched
            .mismatch
            + tolerance;
        if !threshold.is_finite() {
            return Err(numerical());
        }
        heap.clear();
        let mut rng = random_stream(options.seed, options.realization, options.query_id, 2);
        let mut refined = 0;
        for index in 0..n {
            if index % 4096 == 0 {
                options.cancellation.check()?;
            }
            let tie = rng.random::<u128>();
            if !valid(index) || self.scores[index].re + constant > threshold {
                continue;
            }
            let Some(score) = score_candidate(
                self.training,
                events,
                &weights,
                requested,
                index,
                options.boundary,
            )?
            else {
                continue;
            };
            refined += 1;
            push_candidate(
                &mut heap,
                count,
                Candidate {
                    matched: Match {
                        index,
                        mismatch: score,
                    },
                    tie,
                },
            );
        }
        // If numerical mask screening excluded too much, the scalar engine remains
        // a safe recovery. It never relaxes missing-data or event requirements.
        if heap.len() < count {
            return Ok(MatchReport {
                matches: super::best_matches(self.training, events, requested, options)?,
                refined_candidates: n,
                screening_tolerance: tolerance,
            });
        }
        Ok(MatchReport {
            matches: heap
                .into_sorted_vec()
                .into_iter()
                .map(|candidate| candidate.matched)
                .collect(),
            refined_candidates: refined,
            screening_tolerance: tolerance,
        })
    }
}
