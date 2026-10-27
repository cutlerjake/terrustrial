//! All-lag auto/cross-semivariograms with exact integer counts and Fourier sums.
//!
//! The finite grid is zero-padded, never periodic. Both attributes must exist at
//! both endpoints. Signed cross-semivariances are preserved. Plans and buffers
//! are reused; source input and requested output need not be resident together.
//! Transform workspace is O(padded eligible-extent cells). A complete source scan
//! finds the extent and attribute ranges; later scans and refinements visit only
//! that box. Use `grid::experimental` for
//! bounded requested-lag streaming when the full transform does not fit.
//!
//! The roundoff indicator is a conservative heuristic, not a certified RustFFT
//! error bound. Suspicious lags are recomputed in the exact streaming pair order.
//! Use `Map::lag_direct` to require that reduction for a particular lag, or zero
//! absolute and relative tolerances to require it for every nonanalytic lag.
use super::{counts, direct_lag, distance};
use crate::{
    compute::scaling::{exponent, shift},
    error::{checked_len, invalid},
    execution::{Cancellation, MemoryBudget, Reservation},
    grid::Source,
    variography::{indexed::buffer, Bin},
    Error, Result,
};
use rustfft::{num_complex::Complex64, Fft, FftPlanner};
use std::sync::Arc;

/// Physical-unit absolute and relative thresholds for the roundoff indicator.
/// These select refinement, not a guaranteed bound on the returned answer.
#[derive(Clone, Copy, Debug)]
pub struct Accuracy {
    pub absolute: f64,
    pub relative: f64,
}
impl Default for Accuracy {
    fn default() -> Self {
        Self {
            absolute: 1e-10,
            relative: 1e-10,
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Method {
    /// No pairs, zero lag, or at least one globally constant eligible attribute.
    Analytic,
    Fourier,
    Direct,
}
#[derive(Debug)]
pub struct Estimate {
    pub statistics: Bin,
    pub method: Method,
    /// Absolute physical-unit FFT roundoff indicator, when Fourier was used or
    /// screened. None for analytic answers and explicit direct evaluation.
    /// A direct fallback may report infinity when restoration was unsafe.
    pub roundoff_indicator: Option<f64>,
}

#[derive(Clone, Copy)]
struct Normalization {
    base: i32,
    center: f64,
    spread: i32,
}
impl Normalization {
    fn new(min: f64, max: f64) -> Self {
        let base = exponent(min.abs().max(max.abs()));
        let lo = shift(min, -base);
        let hi = shift(max, -base);
        let center = (lo + hi) * 0.5;
        let spread = exponent((lo - center).abs().max((hi - center).abs()));
        Self {
            base,
            center,
            spread,
        }
    }
    fn value(self, value: f64) -> (f64, bool) {
        let scaled = shift(value, -self.base);
        let centered = scaled - self.center;
        let result = shift(centered, -self.spread);
        (
            result,
            (value != 0.0 && scaled == 0.0) || (centered != 0.0 && result == 0.0),
        )
    }
    fn exponent(self) -> i32 {
        self.base + self.spread
    }
}
struct Backend {
    forward: Arc<dyn Fft<f64>>,
    inverse: Arc<dyn Fft<f64>>,
    left: Vec<Complex64>,
    right: Vec<Complex64>,
    scratch_left: Vec<Complex64>,
    scratch_right: Vec<Complex64>,
    _reservation: Reservation,
}
impl Backend {
    fn new(length: usize, budget: &MemoryBudget) -> Result<Self> {
        // Admit coefficient arrays and a two-array plan allowance before asking
        // RustFFT to plan. Its private plan heap is not exposed or fallible, so
        // this allowance is not a strict bound on plan allocations or process RSS.
        let base = checked_len(&[length, 4])?;
        let mut reservation = budget
            .reserve(u64::try_from(checked_len(&[base, 16])?).map_err(|_| Error::SizeOverflow)?)?;
        let mut planner = FftPlanner::new();
        let forward = planner.plan_fft_forward(length);
        let inverse = planner.plan_fft_inverse(length);
        let scratch_left = forward
            .get_inplace_scratch_len()
            .max(inverse.get_inplace_scratch_len());
        let scratch_right = forward.get_inplace_scratch_len();
        let total = base
            .checked_add(scratch_left)
            .and_then(|v| v.checked_add(scratch_right))
            .ok_or(Error::SizeOverflow)?;
        reservation
            .resize(u64::try_from(checked_len(&[total, 16])?).map_err(|_| Error::SizeOverflow)?)?;
        Ok(Self {
            forward,
            inverse,
            left: buffer(length, Complex64::default)?,
            right: buffer(length, Complex64::default)?,
            scratch_left: buffer(scratch_left, Complex64::default)?,
            scratch_right: buffer(scratch_right, Complex64::default)?,
            _reservation: reservation,
        })
    }
    fn transform(&mut self, cancel: &Cancellation) -> Result<()> {
        cancel.check()?;
        rayon::join(
            || {
                self.forward
                    .process_with_scratch(&mut self.left, &mut self.scratch_left)
            },
            || {
                self.forward
                    .process_with_scratch(&mut self.right, &mut self.scratch_right)
            },
        );
        cancel.check()?;
        let n = self.left.len();
        // Two complex transforms encode the four real channels M, M*A*B, M*A,
        // M*B. Recover conjugate spectra before overwriting either member of a
        // frequency pair. Re(P*conj(M) - A*conj(B)) is the symmetric half-increment
        // product sum spectrum. Its inverse is real and even, including cross A/B.
        for k in 0..=n / 2 {
            if k.is_multiple_of(4096) {
                cancel.check()?;
            }
            let other = (n - k) % n;
            let mask = (self.left[k] + self.left[other].conj()) * 0.5;
            let product = (self.left[k] - self.left[other].conj()) * Complex64::new(0.0, -0.5);
            let a = (self.right[k] + self.right[other].conj()) * 0.5;
            let b = (self.right[k] - self.right[other].conj()) * Complex64::new(0.0, -0.5);
            let value = Complex64::new((product * mask.conj() - a * b.conj()).re, 0.0);
            self.left[k] = value;
            self.left[other] = value;
        }
        self.inverse
            .process_with_scratch(&mut self.left, &mut self.scratch_left);
        cancel.check()
    }
}

/// Reusable integer counts, floating transforms and bounded direct-refinement
/// scratch. No per-lag allocations. Cached plans are released before a changed
/// transform length is admitted. Cancellation is checked between transforms,
/// during scans and frequency assembly, not inside a library transform.
pub struct Workspace {
    counts: counts::Workspace,
    buffers: [Vec<Option<f64>>; 4],
    backend: Option<Backend>,
    budget: MemoryBudget,
    _reservation: Reservation,
}
impl Workspace {
    pub fn new(tile_cells: usize, budget: &MemoryBudget) -> Result<Self> {
        if tile_cells == 0 {
            return Err(invalid("positive spectral tile size required"));
        }
        let reservation = budget.reserve(
            u64::try_from(checked_len(&[
                tile_cells,
                4,
                std::mem::size_of::<Option<f64>>(),
            ])?)
            .map_err(|_| Error::SizeOverflow)?,
        )?;
        Ok(Self {
            counts: counts::Workspace::new(tile_cells, budget)?,
            buffers: [
                buffer(tile_cells, || None)?,
                buffer(tile_cells, || None)?,
                buffer(tile_cells, || None)?,
                buffer(tile_cells, || None)?,
            ],
            backend: None,
            budget: budget.clone(),
            _reservation: reservation,
        })
    }

    /// The source must stay immutable during preparation and all lag queries.
    /// Same-cardinality mutations cannot be detected in general. No output map
    /// is allocated: query individual lags from the borrowed result. Zero lag
    /// includes eligible self-pairs and has exactly zero semivariance.
    pub fn compute<'w, S: Source + ?Sized>(
        &'w mut self,
        source: &'w mut S,
        variables: [usize; 2],
        spacing: [f64; 3],
        accuracy: Accuracy,
        cancel: &'w Cancellation,
    ) -> Result<Map<'w, S>> {
        cancel.check()?;
        if spacing.iter().any(|v| !v.is_finite() || *v <= 0.0)
            || [accuracy.absolute, accuracy.relative]
                .iter()
                .any(|v| !v.is_finite() || *v < 0.0)
        {
            return Err(invalid("spectral spacing or refinement tolerance"));
        }
        let mut min = [f64::INFINITY; 2];
        let mut max = [f64::NEG_INFINITY; 2];
        let mask = self.counts.prepare(source, variables, cancel, |a, b| {
            for (v, value) in [a, b].into_iter().enumerate() {
                min[v] = min[v].min(value);
                max[v] = max[v].max(value);
            }
        })?;
        let present = mask.present;
        let integer_length = mask.transform_length()?;
        let analytic = present == 0 || (0..2).any(|v| min[v] == max[v]);
        let mut geometry = None;
        let mut physical_exponent = 0;
        let mut force_direct = accuracy.absolute == 0.0 && accuracy.relative == 0.0;
        let normalization = if !analytic && !force_direct {
            let normalization =
                std::array::from_fn::<_, 2, _>(|v| Normalization::new(min[v], max[v]));
            physical_exponent = normalization[0].exponent() + normalization[1].exponent();
            // Preserve the direct reference's per-product overflow contract.
            force_direct = !shift(8.0, physical_exponent).is_finite();
            if !force_direct {
                geometry = Some(mask.geometry()?);
            }
            Some(normalization)
        } else {
            None
        };
        // Coordinate both caches before admitting either new plan. A retained
        // large floating plan must not block a smaller integer replacement.
        self.counts.retain_length(integer_length);
        let floating_length = geometry.as_ref().map(|g| g.length);
        if self
            .backend
            .as_ref()
            .is_some_and(|b| Some(b.left.len()) != floating_length)
        {
            self.backend = None;
        }
        let counts = self.counts.finish(source, variables, cancel, mask)?;
        let mut reads = counts.source_values_read;
        if let Some(shape) = &geometry {
            if self.backend.is_none() {
                self.backend = Some(Backend::new(shape.length, &self.budget)?);
            }
            let backend = self.backend.as_mut().unwrap();
            let normalization = normalization.unwrap();
            backend.left.fill(Complex64::default());
            backend.right.fill(Complex64::default());
            let mut populated = 0;
            let mut cropped = counts::Cropped::new(source, counts.eligible_extent().unwrap())?;
            let (first, rest) = self.buffers.split_at_mut(1);
            counts::scan(
                &mut cropped,
                variables,
                &mut first[0],
                &mut rest[0],
                cancel,
                |i, a, b| {
                    if let (Some(a), Some(b)) = (a, b) {
                        populated += 1;
                        let (a, lost_a) = normalization[0].value(a);
                        let (b, lost_b) = normalization[1].value(b);
                        force_direct |= lost_a || lost_b;
                        let i = shape.index(i);
                        backend.left[i] = Complex64::new(1.0, a * b);
                        backend.right[i] = Complex64::new(a, b);
                    }
                    Ok(())
                },
                &mut reads,
            )?;
            if populated != present {
                return Err(invalid("grid mask changed during spectral preparation"));
            }
            if !force_direct {
                backend.transform(cancel)?;
            }
        }
        let normalized_roundoff = geometry.as_ref().map_or(0.0, |g| {
            64.0 * f64::EPSILON * (1.0 + g.length.ilog2() as f64) * present as f64
        });
        Ok(Map {
            counts,
            source,
            variables,
            spacing,
            accuracy,
            cancel,
            buffers: &mut self.buffers,
            backend: self.backend.as_ref(),
            geometry,
            physical_exponent,
            normalized_roundoff,
            force_direct,
            analytic,
            source_values_read: reads,
        })
    }
}

/// Prepared all-lag map. Keeps both workspace and immutable-in-practice source
/// borrowed so numerical fallbacks can reread bounded tiles without allocation.
pub struct Map<'a, S: Source + ?Sized> {
    counts: counts::Counts<'a>,
    source: &'a mut S,
    variables: [usize; 2],
    spacing: [f64; 3],
    accuracy: Accuracy,
    cancel: &'a Cancellation,
    buffers: &'a mut [Vec<Option<f64>>; 4],
    backend: Option<&'a Backend>,
    geometry: Option<counts::Geometry>,
    physical_exponent: i32,
    normalized_roundoff: f64,
    force_direct: bool,
    analytic: bool,
    pub source_values_read: u128,
}
impl<S: Source + ?Sized> Map<'_, S> {
    pub fn eligible_cells(&self) -> u64 {
        self.counts.eligible_cells()
    }
    /// Smallest box containing cells eligible for both selected attributes.
    /// Lags reaching outside its dimensions have no pairs, even inside the source grid.
    pub fn eligible_extent(&self) -> Option<counts::Extent> {
        self.counts.eligible_extent()
    }
    pub fn layout(&self) -> crate::grid::Layout {
        self.counts.layout()
    }
    /// Exact count, including self-pairs at zero lag. Out-of-grid offsets give 0.
    pub fn pairs(&self, offset: [i64; 3]) -> u64 {
        self.counts.pairs(offset)
    }
    pub fn lag(&mut self, offset: [i64; 3]) -> Result<Estimate> {
        self.cancel.check()?;
        let pairs = self.pairs(offset);
        let mean_distance = if pairs == 0 {
            None
        } else if offset == [0; 3] {
            Some(0.0)
        } else {
            Some(distance(offset, self.spacing)?)
        };
        if pairs == 0 || offset == [0; 3] || self.analytic {
            return Ok(Estimate {
                statistics: Bin {
                    pairs,
                    mean_distance,
                    semivariance: (pairs > 0).then_some(0.0),
                },
                method: Method::Analytic,
                roundoff_indicator: None,
            });
        }
        if self.force_direct {
            return self.lag_direct(offset);
        }
        let geometry = self.geometry.as_ref().unwrap();
        let n = geometry.length as f64;
        let sum = self.backend.unwrap().left[geometry.cyclic_lag(offset)].re / n;
        let normalized = sum / pairs as f64;
        let value = shift(normalized, self.physical_exponent);
        // Normalized channels have magnitude <=2. The indicator accounts for
        // transform stages, global eligible mass, cancellation and pair dilution.
        // It deliberately does not claim a library-specific forward-error proof.
        let indicator = shift(
            self.normalized_roundoff / pairs as f64,
            self.physical_exponent,
        );
        let tolerance = self
            .accuracy
            .absolute
            .max(self.accuracy.relative * value.abs());
        if !value.is_finite()
            || !indicator.is_finite()
            || (normalized != 0.0 && value == 0.0)
            || (value != 0.0 && value.abs() < f64::MIN_POSITIVE)
            || (self.variables[0] == self.variables[1] && value < 0.0)
            || indicator > tolerance
        {
            let mut result = self.lag_direct(offset)?;
            result.roundoff_indicator = Some(indicator);
            return Ok(result);
        }
        Ok(Estimate {
            statistics: Bin {
                pairs,
                mean_distance,
                semivariance: Some(value),
            },
            method: Method::Fourier,
            roundoff_indicator: Some(indicator),
        })
    }
    /// Always evaluate the streaming pair reduction. This is also available to
    /// verify lags accepted by the heuristic Fourier screen.
    pub fn lag_direct(&mut self, offset: [i64; 3]) -> Result<Estimate> {
        self.cancel.check()?;
        let statistics = if let Some(extent) = self.counts.eligible_extent() {
            if extent.dimensions == self.source.layout().dimensions() {
                // Keep the original monomorphized read/reduction hot path for
                // whole grids. Even the identity adapter's checks measurably
                // cost repeated short reads during cancellation refinement.
                direct_lag(
                    self.source,
                    self.variables,
                    offset,
                    self.spacing,
                    self.buffers,
                    &mut self.source_values_read,
                    self.cancel,
                )?
            } else {
                let mut cropped = counts::Cropped::new(self.source, extent)?;
                direct_lag(
                    &mut cropped,
                    self.variables,
                    offset,
                    self.spacing,
                    self.buffers,
                    &mut self.source_values_read,
                    self.cancel,
                )?
            }
        } else {
            Bin {
                pairs: 0,
                mean_distance: None,
                semivariance: None,
            }
        };
        if statistics.pairs != self.pairs(offset) {
            return Err(invalid("grid mask changed during spectral refinement"));
        }
        Ok(Estimate {
            statistics,
            method: Method::Direct,
            roundoff_indicator: None,
        })
    }
}
