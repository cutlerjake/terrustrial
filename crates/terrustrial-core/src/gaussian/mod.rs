//! Generalized least squares and joint conditional Gaussian prediction.
use crate::compute::linalg::{factor_spd_with_parameters, finite, solve_spd};
use crate::{
    data::{Observation, Site},
    error::{checked_len, invalid},
    execution::{random_stream, Cancellation, MemoryBudget, Reservation},
    model::Covariance,
    Error, Result,
};
use dyn_stack::{MemBuffer, MemStack};
use faer::{linalg::cholesky::llt, Mat, MatMut, MatRef, Par};
use rand::RngExt;
use rand_distr::StandardNormal;
use std::collections::HashSet;

mod conditioning;
mod diagnostics;
pub(crate) mod group;
pub use diagnostics::{ConditioningDiagnostics, DiagnosticWorkspace, SpectralCondition};
mod residuals;
pub use residuals::{
    EquationResiduals, GroupResiduals, ResidualLocation, ResidualWorkspace, SolveResiduals,
};
mod drawing;
mod drift;
mod equilibration;
mod rank;
mod scaling;
mod semidefinite;
pub use conditioning::ConditionedWorkspace;
use semidefinite::factor as psd_factor_with_threshold;
mod neighborhood;
pub(crate) use neighborhood::NeighborhoodConditioning;
mod retained;
pub(crate) use retained::RetainedPlan;
#[cfg(test)]
mod prediction_tests;

/// Deterministic known mean in the same scale as observations and simulation.
/// Return the mean over the entire support, not necessarily its center. The
/// supplied covariance describes residuals after subtracting this mean. Mean
/// estimation uncertainty is not included. Values must be finite and stable
/// across repeated calls, worker counts, and resumed runs.
pub trait KnownMean: Sync {
    fn evaluate(&self, site: &Site) -> Result<f64>;
}
impl<F: Fn(&Site) -> Result<f64> + Sync> KnownMean for F {
    fn evaluate(&self, site: &Site) -> Result<f64> {
        self(site)
    }
}

/// Unbiasedness constraints. Basis values are support averages, not just center values.
/// Preparation scales each column by its maximum absolute conditioning value and
/// applies the same scale at targets. This preserves the represented constraints;
/// zero columns and unrepresentable normalized values return contextual errors.
/// Suspect drift normal equations use a column-pivoted QR change of basis.
/// Numerically rank-deficient drift is rejected; constraints are never dropped.
pub trait Drift: Sync {
    fn columns(&self) -> usize;
    fn evaluate(&self, site: &Site, values: &mut [f64]) -> Result<()>;
}
#[derive(Clone, Copy, Debug)]
pub struct Simple;
impl Drift for Simple {
    fn columns(&self) -> usize {
        0
    }
    fn evaluate(&self, _: &Site, _: &mut [f64]) -> Result<()> {
        Ok(())
    }
}
#[derive(Clone, Copy, Debug)]
pub struct Ordinary {
    pub variables: usize,
}
impl Drift for Ordinary {
    fn columns(&self) -> usize {
        self.variables
    }
    fn evaluate(&self, site: &Site, values: &mut [f64]) -> Result<()> {
        if site.variable >= self.variables || values.len() != self.variables {
            return Err(invalid("ordinary drift variable count"));
        }
        values.fill(0.0);
        values[site.variable] = 1.0;
        Ok(())
    }
}

/// Per-variable linear trend with centered/scaled coordinates for conditioning.
#[derive(Clone, Debug)]
pub struct LinearDrift {
    variables: usize,
    origin: [f64; 3],
    scale: f64,
}
impl LinearDrift {
    pub fn variables(&self) -> usize {
        self.variables
    }
    pub fn origin(&self) -> [f64; 3] {
        self.origin
    }
    pub fn scale(&self) -> f64 {
        self.scale
    }
    pub fn new(variables: usize, origin: [f64; 3], scale: f64) -> Result<Self> {
        checked_len(&[variables, 4])?;
        if variables == 0
            || origin.iter().any(|x| !x.is_finite())
            || !scale.is_finite()
            || scale <= 0.0
        {
            return Err(invalid("linear drift parameters"));
        }
        Ok(Self {
            variables,
            origin,
            scale,
        })
    }
}
impl Drift for LinearDrift {
    fn columns(&self) -> usize {
        self.variables * 4
    }
    fn evaluate(&self, site: &Site, values: &mut [f64]) -> Result<()> {
        if site.variable >= self.variables || values.len() != self.columns() {
            return Err(invalid("linear drift variable count"));
        }
        values.fill(0.0);
        let start = site.variable * 4;
        values[start] = 1.0;
        for axis in 0..3 {
            let coordinate = site.support.center().coordinates()[axis];
            let delta = coordinate - self.origin[axis];
            values[start + axis + 1] = if delta.is_finite() {
                delta / self.scale
            } else {
                // Opposite finite extremes can overflow before division even
                // when the requested normalized difference is representable.
                coordinate / self.scale - self.origin[axis] / self.scale
            };
        }
        Ok(())
    }
}

/// Custom support-aware basis for external drift or other universal kriging models.
pub struct ExternalDrift<F> {
    columns: usize,
    evaluate: F,
}
impl<F> ExternalDrift<F> {
    pub fn new(columns: usize, evaluate: F) -> Result<Self> {
        if columns == 0 {
            return Err(invalid("external drift needs at least one column"));
        }
        Ok(Self { columns, evaluate })
    }
}
impl<F: Fn(&Site, &mut [f64]) -> Result<()> + Sync> Drift for ExternalDrift<F> {
    fn columns(&self) -> usize {
        self.columns
    }
    fn evaluate(&self, site: &Site, values: &mut [f64]) -> Result<()> {
        (self.evaluate)(site, values)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Uncertainty {
    None,
    Joint,
}

/// Explicit roundoff policy in covariance units. No nugget is introduced.
#[derive(Clone, Copy, Debug)]
pub struct NumericalTolerance {
    pub absolute: f64,
    pub relative: f64,
}
impl Default for NumericalTolerance {
    fn default() -> Self {
        Self {
            absolute: 0.0,
            relative: 1e-12,
        }
    }
}
impl NumericalTolerance {
    pub(crate) fn threshold(self, scale: f64) -> Result<f64> {
        if !self.absolute.is_finite()
            || self.absolute < 0.0
            || !self.relative.is_finite()
            || !(0.0..1.0).contains(&self.relative)
        {
            return Err(invalid("numerical tolerance"));
        }
        let tolerance = self.absolute + self.relative * scale;
        if !tolerance.is_finite() {
            return Err(invalid("numerical tolerance overflow"));
        }
        Ok(tolerance)
    }
}

pub struct PrepareOptions<'a> {
    pub uncertainty: Uncertainty,
    pub tolerance: NumericalTolerance,
    pub budget: &'a MemoryBudget,
    pub cancellation: &'a Cancellation,
}

/// Immutable reusable weights and, optionally, joint prediction-error covariance.
/// UK/OK uncertainty includes uncertainty in the unknown drift coefficients under
/// a flat prior. Rank-deficient conditioning/drift is rejected explicitly.
pub struct GaussianPlan {
    weights: Mat<f64>,
    covariance: Mat<f64>,
    factor: Mat<f64>,
    joint: bool,
    roundoff_corrections: usize,
    reservation: Reservation,
}
impl GaussianPlan {
    pub fn prepare(
        model: &impl Covariance,
        observations: &[Observation],
        targets: &[Site],
        drift: &impl Drift,
        options: PrepareOptions<'_>,
    ) -> Result<Self> {
        Self::prepare_component(model, model, observations, targets, drift, options)
    }

    /// For factorial kriging, `target_model` is an independent component of `data_model`.
    /// The caller must establish that decomposition; mismatches can fail the PSD check.
    pub fn prepare_component(
        data_model: &impl Covariance,
        target_model: &impl Covariance,
        observations: &[Observation],
        targets: &[Site],
        drift: &impl Drift,
        options: PrepareOptions<'_>,
    ) -> Result<Self> {
        options.cancellation.check()?;
        let mut workspace = GaussianWorkspace::new(
            observations.len(),
            targets.len(),
            drift.columns(),
            options.budget,
        )?;
        workspace.prepare_component(
            data_model,
            target_model,
            observations,
            targets,
            drift,
            WorkspaceOptions {
                uncertainty: options.uncertainty,
                tolerance: options.tolerance,
                cancellation: options.cancellation,
            },
        )?;
        Ok(workspace.plan)
    }
    /// Retain a plan beyond the next workspace preparation. Only final weights,
    /// covariance and the conditional factor are copied, not solver scratch.
    pub fn to_owned_plan(&self, budget: &MemoryBudget) -> Result<Self> {
        let n = self.weights.nrows();
        let m = self.weights.ncols();
        let entries = checked_len(&[n, m])?
            .checked_add(if self.joint {
                checked_len(&[m, m, 2])?
            } else {
                0
            })
            .ok_or(Error::SizeOverflow)?;
        let reservation = budget.reserve(
            u64::try_from(checked_len(&[entries, 8])?).map_err(|_| Error::SizeOverflow)?,
        )?;
        Ok(Self {
            weights: copy_matrix(self.weights.as_ref())?,
            covariance: if self.joint {
                copy_matrix(self.covariance.as_ref())?
            } else {
                Mat::new()
            },
            factor: if self.joint {
                copy_matrix(self.factor.as_ref())?
            } else {
                Mat::new()
            },
            joint: self.joint,
            roundoff_corrections: self.roundoff_corrections,
            reservation,
        })
    }
    pub fn weights(&self) -> MatRef<'_, f64> {
        self.weights.as_ref()
    }
    pub fn covariance(&self) -> Option<MatRef<'_, f64>> {
        self.joint.then(|| self.covariance.as_ref())
    }
    pub fn roundoff_corrections(&self) -> usize {
        self.roundoff_corrections
    }
    pub fn accounted_bytes(&self) -> u64 {
        self.reservation.bytes()
    }
    pub fn estimate(&self, values: &[f64]) -> Result<Vec<f64>> {
        let mut output = vec![0.0; self.weights.ncols()];
        self.estimate_into(values, &mut output)?;
        Ok(output)
    }
    /// Allocation-free repeated prediction with prepared weights.
    pub fn estimate_into(&self, values: &[f64], output: &mut [f64]) -> Result<()> {
        if values.len() != self.weights.nrows()
            || output.len() != self.weights.ncols()
            || values.iter().any(|x| !x.is_finite())
        {
            return Err(invalid("prediction values or output dimensions"));
        }
        faer::linalg::matmul::matmul(
            faer::ColMut::from_slice_mut(output).as_mat_mut(),
            faer::Accum::Replace,
            self.weights.as_ref().transpose(),
            faer::ColRef::from_slice(values).as_mat(),
            1.0,
            Par::Seq,
        );
        if output.iter().any(|x| !x.is_finite()) {
            return Err(Error::Numerical {
                context: "kriging prediction",
            });
        }
        Ok(())
    }
    /// Known means implement SK with nonzero mean and locally varying mean kriging.
    /// When prepared with drift constraints, those constraints apply to residuals.
    pub fn estimate_with_mean(
        &self,
        values: &[f64],
        data_mean: &[f64],
        target_mean: &[f64],
    ) -> Result<Vec<f64>> {
        let mut out = vec![0.0; self.weights.ncols()];
        self.estimate_with_mean_into(values, data_mean, target_mean, &mut out)?;
        Ok(out)
    }
    /// Allocation-free prediction with explicit known means and caller output.
    pub fn estimate_with_mean_into(
        &self,
        values: &[f64],
        data_mean: &[f64],
        target_mean: &[f64],
        out: &mut [f64],
    ) -> Result<()> {
        if values.len() != self.weights.nrows()
            || out.len() != self.weights.ncols()
            || data_mean.len() != values.len()
            || target_mean.len() != self.weights.ncols()
            || values
                .iter()
                .chain(data_mean)
                .chain(target_mean)
                .any(|x| !x.is_finite())
        {
            return Err(invalid(
                "prediction values/means have invalid dimensions or nonfinite entries",
            ));
        }
        for (j, value) in out.iter_mut().enumerate() {
            *value = target_mean[j]
                + (0..values.len())
                    .map(|i| self.weights[(i, j)] * (values[i] - data_mean[i]))
                    .sum::<f64>();
        }
        if out.iter().any(|x| !x.is_finite()) {
            return Err(Error::Numerical {
                context: "kriging prediction",
            });
        }
        Ok(())
    }
    pub fn sample(
        &self,
        mean: &[f64],
        seed: u64,
        realization: u64,
        group_id: u64,
    ) -> Result<Vec<f64>> {
        if !self.joint {
            return Err(invalid("simulation requires joint uncertainty"));
        }
        if mean.len() != self.weights.ncols() || mean.iter().any(|x| !x.is_finite()) {
            return Err(invalid("conditional mean dimensions or values"));
        }
        let mut noise = vec![0.0; mean.len()];
        let mut out = vec![0.0; mean.len()];
        self.sample_into(mean, seed, realization, group_id, &mut noise, &mut out)?;
        Ok(out)
    }
    /// Allocation-free draw using caller-owned standard-normal scratch and output.
    /// Both slices must have exactly one entry per target. On numerical failure,
    /// buffers may contain partial results and must not be treated as a draw.
    pub fn sample_into(
        &self,
        mean: &[f64],
        seed: u64,
        realization: u64,
        group_id: u64,
        noise: &mut [f64],
        output: &mut [f64],
    ) -> Result<()> {
        if !self.joint {
            return Err(invalid("simulation requires joint uncertainty"));
        }
        let factor = &self.factor;
        if mean.len() != factor.nrows()
            || noise.len() != mean.len()
            || output.len() != mean.len()
            || mean.iter().any(|x| !x.is_finite())
        {
            return Err(invalid("conditional mean or draw buffer dimensions/values"));
        }
        let mut rng = random_stream(seed, realization, group_id, 0);
        for value in noise.iter_mut() {
            *value = rng.sample(StandardNormal);
        }
        drawing::apply(factor.as_ref(), mean, noise, output);
        if output.iter().any(|x| !x.is_finite()) {
            return Err(Error::Numerical {
                context: "Gaussian draw",
            });
        }
        Ok(())
    }
}

/// Fixed-capacity Gaussian preparation buffers. Construction allocates and reserves
/// storage; successful preparations within capacity reuse it, including faer's
/// factorization scratch. Covariance/drift callbacks may allocate independently.
/// Larger faer matrix kernels can also allocate internal packing buffers, so
/// workspace reuse does not guarantee zero allocations for every matrix shape.
/// The returned plan borrows the workspace until the next mutable preparation.
/// A failed preparation does not expose a partial plan; the workspace is reusable.
/// Extreme common covariance units and heterogeneous diagonal variances use
/// power-of-two scaling internally. Heterogeneous systems also scale drift
/// constraints and joint targets; relative roundoff tolerances then apply in
/// dimensionless target units. Returned
/// covariance, sampling factors and absolute roundoff tolerances retain physical
/// units. Covariance that exceeds the physical range only within the explicit
/// roundoff tolerance is contracted, together with its sampling factor, to fit.
/// Unrepresentable normalized inputs or larger output overflows return errors.
pub struct GaussianWorkspace {
    simd: fearless_simd::Level,
    maximum_observations: usize,
    maximum_targets: usize,
    maximum_drift: usize,
    c: Mat<f64>,
    covariance_units: f64,
    prior_row: Vec<f64>,
    equilibration: equilibration::Equilibration,
    k: Mat<f64>,
    fd: Mat<f64>,
    ft: Mat<f64>,
    correction: Mat<f64>,
    cinv_f: Mat<f64>,
    normal: Mat<f64>,
    drift_basis: rank::DriftBasis,
    // First p entries are callback values; the next p retain conditioning scales.
    buffer: Vec<f64>,
    ids: HashSet<u64>,
    scratch: MemBuffer,
    plan: GaussianPlan,
}
pub struct WorkspaceOptions<'a> {
    pub uncertainty: Uncertainty,
    pub tolerance: NumericalTolerance,
    pub cancellation: &'a Cancellation,
}

// Avoid splitting a short run of colocated variables across matrix batches.
// The cap remains fixed even for arbitrarily long coincident runs.
fn covariance_batch_end<'a>(first: usize, count: usize, site: impl Fn(usize) -> &'a Site) -> usize {
    let limit = (first + 8).min(count);
    if limit == count || site(limit).support.node_count() != 1 {
        return limit;
    }
    let point = site(limit).support.center().coordinates().map(f64::to_bits);
    let mut end = limit;
    while end > first
        && site(end - 1).support.node_count() == 1
        && site(end - 1)
            .support
            .center()
            .coordinates()
            .map(f64::to_bits)
            == point
    {
        end -= 1;
    }
    if end == first {
        limit
    } else {
        end
    }
}

impl GaussianWorkspace {
    pub fn new(
        maximum_observations: usize,
        maximum_targets: usize,
        maximum_drift: usize,
        budget: &MemoryBudget,
    ) -> Result<Self> {
        if maximum_targets == 0 || maximum_drift > maximum_observations {
            return Err(invalid("Gaussian workspace dimensions"));
        }
        let (n, m, p) = (maximum_observations, maximum_targets, maximum_drift);
        let size = n
            .checked_add(m)
            .and_then(|v| v.checked_add(p))
            .ok_or(Error::SizeOverflow)?;
        let bytes = checked_len(&[size, size, 8, 12])?
            .checked_add(checked_len(&[size, 2])?)
            .and_then(|v| v.checked_add(m.checked_mul(16)?))
            .ok_or(Error::SizeOverflow)?;
        let reservation = budget.reserve(u64::try_from(bytes).map_err(|_| Error::SizeOverflow)?)?;
        fn matrix(rows: usize, cols: usize) -> Result<Mat<f64>> {
            let mut out = Mat::new();
            out.try_reserve(rows, cols)
                .map_err(|_| invalid("Gaussian matrix allocation"))?;
            Ok(out)
        }
        let mut ids = HashSet::new();
        ids.try_reserve(n)
            .map_err(|_| invalid("Gaussian ID validation allocation"))?;
        let mut buffer = Vec::new();
        buffer
            .try_reserve_exact(checked_len(&[p, 2])?)
            .map_err(|_| invalid("Gaussian drift buffer allocation"))?;
        let mut prior_row = Vec::new();
        prior_row
            .try_reserve_exact(m)
            .map_err(|_| invalid("Gaussian prior row allocation"))?;
        prior_row.resize(m, 0.0);
        // faer 0.24.4 requests one contiguous column for sequential LLT.
        // Also check the actual request before every factorization, so a future
        // dependency change cannot turn an insufficient arena into a panic.
        let request = llt::factor::cholesky_in_place_scratch::<f64>(
            n,
            Par::Seq,
            gaussian_llt_parameters(n).into(),
        )
        .or(llt::factor::cholesky_in_place_scratch::<f64>(
            p,
            Par::Seq,
            gaussian_llt_parameters(p).into(),
        ))
        .or(rank::DriftBasis::scratch(n, p));
        let scratch = MemBuffer::try_new(request)
            .map_err(|_| invalid("Gaussian factor scratch allocation"))?;
        Ok(Self {
            simd: fearless_simd::Level::new(),
            maximum_observations: n,
            maximum_targets: m,
            maximum_drift: p,
            c: matrix(n, n)?,
            covariance_units: 1.0,
            prior_row,
            equilibration: equilibration::Equilibration::new(n, m, p)?,
            k: matrix(n, m)?,
            fd: matrix(n, p)?,
            ft: matrix(m, p)?,
            correction: matrix(p, m)?,
            cinv_f: matrix(n, p)?,
            normal: matrix(p, p)?,
            drift_basis: rank::DriftBasis::new(p)?,
            buffer,
            ids,
            scratch,
            plan: GaussianPlan {
                weights: matrix(n, m)?,
                covariance: matrix(m, m)?,
                factor: matrix(m, m)?,
                joint: false,
                roundoff_corrections: 0,
                reservation,
            },
        })
    }
    pub(crate) fn cached<'a>(
        slot: &'a mut Option<Self>,
        n: usize,
        m: usize,
        p: usize,
        budget: &MemoryBudget,
    ) -> Result<&'a mut Self> {
        if !slot
            .as_ref()
            .is_some_and(|workspace| workspace.supports_shape(n, m, p))
        {
            *slot = None;
            *slot = Some(Self::new(n, m, p, budget)?);
        }
        Ok(slot.as_mut().expect("initialized Gaussian workspace"))
    }
    pub(crate) fn supports_shape(&self, n: usize, m: usize, p: usize) -> bool {
        n <= self.maximum_observations && m <= self.maximum_targets && p <= self.maximum_drift
    }
    pub fn prepare(
        &mut self,
        model: &impl Covariance,
        observations: &[Observation],
        targets: &[Site],
        drift: &impl Drift,
        options: WorkspaceOptions<'_>,
    ) -> Result<&GaussianPlan> {
        self.prepare_component(model, model, observations, targets, drift, options)
    }
    #[allow(clippy::too_many_arguments)]
    pub fn prepare_component(
        &mut self,
        data_model: &impl Covariance,
        target_model: &impl Covariance,
        observations: &[Observation],
        targets: &[Site],
        drift: &impl Drift,
        options: WorkspaceOptions<'_>,
    ) -> Result<&GaussianPlan> {
        options.cancellation.check()?;
        options.tolerance.threshold(1.0)?;
        let n = observations.len();
        let m = targets.len();
        let p = drift.columns();
        if m == 0 || p > n || data_model.variables() != target_model.variables() {
            return Err(invalid(
                "empty targets, underdetermined drift, or inconsistent variable counts",
            ));
        }
        if !self.supports_shape(n, m, p) {
            return Err(invalid("Gaussian workspace capacity exceeded"));
        }
        if targets.iter().any(|t| t.variable >= data_model.variables()) {
            return Err(invalid("target variable out of range"));
        }
        let (variables, columns) =
            self.prepare_data(data_model, observations, drift, options.cancellation)?;
        self.prepare_targets(
            target_model,
            observations,
            targets,
            drift,
            variables,
            columns,
            options,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn prepare_targets(
        &mut self,
        target_model: &impl Covariance,
        observations: &[Observation],
        targets: &[Site],
        drift: &impl Drift,
        variables: usize,
        p: usize,
        options: WorkspaceOptions<'_>,
    ) -> Result<&GaussianPlan> {
        let _span = crate::execution::profiling::Span::new(
            crate::execution::profiling::Metric::TargetSystem,
        );
        options.cancellation.check()?;
        options.tolerance.threshold(1.0)?;
        let n = observations.len();
        let m = targets.len();
        if m == 0
            || m > self.maximum_targets
            || target_model.variables() != variables
            || drift.columns() != p
        {
            return Err(invalid(
                "prepared conditioning target dimensions or variable counts",
            ));
        }
        if targets.iter().any(|t| t.variable >= variables) {
            return Err(invalid("target variable out of range"));
        }
        let Self {
            simd,
            c,
            covariance_units,
            prior_row,
            equilibration,
            k,
            fd,
            ft,
            buffer,
            correction,
            cinv_f,
            normal,
            drift_basis,
            plan,
            ..
        } = self;
        let GaussianPlan {
            weights,
            covariance,
            factor,
            ..
        } = plan;
        k.resize_with(n, m, |_, _| 0.0);
        ft.resize_with(m, p, |_, _| 0.0);
        correction.resize_with(p, m, |_, _| 0.0);
        weights.resize_with(n, m, |_, _| 0.0);
        equilibration.prepare_targets(
            target_model,
            observations,
            targets,
            *covariance_units,
            options.cancellation,
        )?;
        let mut first = 0;
        while first < m {
            options.cancellation.check()?;
            let end = covariance_batch_end(first, m, |i| &targets[i]);
            let mut columns = k.as_mut().subcols_mut(first, end - first).col_iter_mut();
            let mut rows: [&mut [f64]; 8] = std::array::from_fn(|_| -> &mut [f64] {
                columns.next().map_or(&mut [] as &mut [f64], |column| {
                    column
                        .try_as_col_major_mut()
                        .expect("owned cross covariance column")
                        .as_slice_mut()
                })
            });
            target_model.covariance_columns(
                observations,
                &targets[first..end],
                &mut rows[..end - first],
            )?;
            first = end;
        }
        equilibration.normalize_cross(k.as_mut(), *covariance_units)?;
        let (values, scales) = buffer.split_at_mut(p);
        for (i, site) in targets.iter().enumerate() {
            options.cancellation.check()?;
            evaluate_drift(drift, site, values)?;
            for j in 0..p {
                ft[(i, j)] = values[j];
            }
        }
        // Ordinary drift already has unit scales. Check once per column, and
        // traverse contiguous target values only when normalization is needed.
        for j in 0..p {
            if scales[j] != 1.0 {
                for i in 0..m {
                    ft[(i, j)] = normalize_drift(ft[(i, j)], scales[j])?;
                }
            }
        }
        equilibration.normalize_drift(ft.as_mut())?;
        drift_basis.transform_targets(ft.as_mut(), values)?;
        if n > 0 {
            weights.as_mut().copy_from(k.as_ref());
            solve_spd(c.as_ref(), weights.as_mut());
            if p > 0 {
                // Matrix-shape crossover measured on complete preparations.
                // faer owns CPU specialization for wider target groups; retain
                // native-width reductions for small groups with lower overhead.
                let bulk_drift_product = m >= 8;
                if bulk_drift_product {
                    correction.as_mut().copy_from(ft.as_ref().transpose());
                    faer::linalg::matmul::matmul(
                        correction.as_mut(),
                        faer::Accum::Add,
                        fd.as_ref().transpose(),
                        weights.as_ref(),
                        -1.0,
                        Par::Seq,
                    );
                } else {
                    drift::residual(
                        *simd,
                        fd.as_ref(),
                        weights.as_ref(),
                        ft.as_ref(),
                        correction.as_mut(),
                    );
                }
                solve_spd(normal.as_ref(), correction.as_mut());
                if bulk_drift_product {
                    faer::linalg::matmul::matmul(
                        weights.as_mut(),
                        faer::Accum::Add,
                        cinv_f.as_ref(),
                        correction.as_ref(),
                        1.0,
                        Par::Seq,
                    );
                } else {
                    drift::add_weights(
                        *simd,
                        weights.as_mut(),
                        cinv_f.as_ref(),
                        correction.as_ref(),
                    );
                }
            }
        }
        finite(weights.as_ref(), "kriging weights")?;
        let roundoff_corrections = if options.uncertainty == Uncertainty::Joint {
            covariance.resize_with(m, m, |_, _| 0.0);
            factor.resize_with(m, m, |_, _| 0.0);
            if p > 0 {
                // The sampling factor is not needed until covariance assembly
                // finishes. Reuse its storage for the drift contribution.
                faer::linalg::matmul::matmul(
                    factor.as_mut(),
                    faer::Accum::Replace,
                    ft.as_ref(),
                    correction.as_ref(),
                    1.0,
                    Par::Seq,
                );
            }
            let bulk_cross_product = n >= 64 && m >= 16;
            if bulk_cross_product {
                // Reuse the final covariance buffer for K^T W. Each symmetric
                // pair is consumed together below before being overwritten.
                faer::linalg::matmul::matmul(
                    covariance.as_mut(),
                    faer::Accum::Replace,
                    k.as_ref().transpose(),
                    weights.as_ref(),
                    1.0,
                    Par::Seq,
                );
            }
            let mut prior_scale = 0.0_f64;
            // Ctt - K^T W + Ft G^-1 R equals the universal-kriging error covariance.
            for i in 0..m {
                options.cancellation.check()?;
                target_model.covariance_sites_row(
                    &targets[i],
                    &targets[..i],
                    &mut prior_row[..i],
                )?;
                for j in 0..=i {
                    let mut value = equilibration.prior(prior_row[j], i, j, *covariance_units)?;
                    if i == j {
                        prior_scale = prior_scale.max(value.abs());
                    }
                    value -= 0.5
                        * if bulk_cross_product {
                            covariance[(i, j)] + covariance[(j, i)]
                        } else {
                            (0..n)
                                .map(|r| k[(r, i)] * weights[(r, j)] + k[(r, j)] * weights[(r, i)])
                                .sum::<f64>()
                        };
                    if p > 0 {
                        value += 0.5 * (factor[(i, j)] + factor[(j, i)]);
                    }
                    covariance[(i, j)] = value;
                    covariance[(j, i)] = value;
                }
            }
            let rounded = equilibration.factor(
                covariance.as_ref(),
                factor.as_mut(),
                options.tolerance,
                prior_scale,
                *covariance_units,
            )?;
            // Expose the same covariance that sampling uses after roundoff repair.
            if rounded > 0 {
                for j in 0..m {
                    for i in j..m {
                        let value = (0..=j).map(|r| factor[(i, r)] * factor[(j, r)]).sum();
                        covariance[(i, j)] = value;
                        covariance[(j, i)] = value;
                    }
                }
            }
            rounded
                + equilibration.restore(
                    covariance.as_mut(),
                    factor.as_mut(),
                    *covariance_units,
                    options.tolerance,
                    prior_scale,
                )?
        } else {
            0
        };
        equilibration.restore_weights(weights.as_mut())?;
        self.plan.roundoff_corrections = roundoff_corrections;
        self.plan.joint = options.uncertainty == Uncertainty::Joint;
        Ok(&self.plan)
    }
}

fn evaluate_drift(drift: &impl Drift, site: &Site, buffer: &mut [f64]) -> Result<()> {
    buffer.fill(f64::NAN);
    drift.evaluate(site, buffer)?;
    if buffer.iter().any(|v| !v.is_finite()) {
        return Err(invalid(
            "drift basis did not fill every column with finite values",
        ));
    }
    Ok(())
}

fn normalize_drift(value: f64, scale: f64) -> Result<f64> {
    if scale == 1.0 {
        return Ok(value);
    }
    let normalized = value / scale;
    if !normalized.is_finite() || (value != 0.0 && normalized == 0.0) {
        return Err(Error::Numerical {
            context: "drift column dynamic range exceeds representable normalization",
        });
    }
    Ok(normalized)
}

/// Fallible copy of a final matrix, shared by plan and estimation retention.
pub(crate) fn copy_matrix(source: MatRef<'_, f64>) -> Result<Mat<f64>> {
    let mut out = Mat::new();
    out.try_reserve(source.nrows(), source.ncols())
        .map_err(|_| invalid("retained Gaussian matrix allocation"))?;
    out.resize_with(source.nrows(), source.ncols(), |i, j| source[(i, j)]);
    Ok(out)
}

/// Fixed-order semidefinite Cholesky. Exact zero variance remains deterministic.
fn psd_factor_into(
    matrix: MatRef<'_, f64>,
    factor: MatMut<'_, f64>,
    tolerance: NumericalTolerance,
    prior_scale: f64,
    covariance_units: f64,
) -> Result<usize> {
    finite(matrix, "conditional covariance")?;
    let n = matrix.nrows();
    let scale = (0..n)
        .map(|i| matrix[(i, i)].abs())
        .fold(prior_scale, f64::max);
    let threshold = scaling::threshold(tolerance, scale, covariance_units)?;
    psd_factor_with_threshold(matrix, factor, |_, _| threshold)
}

fn factor_spd_in(
    matrix: MatMut<'_, f64>,
    context: &'static str,
    scratch: &mut MemBuffer,
) -> Result<()> {
    let parameters = gaussian_llt_parameters(matrix.nrows());
    factor_spd_with_parameters(matrix, context, scratch, parameters)
}

// Measured Gaussian blocking retains faer's portable kernels. It reduces packing
// for larger neighborhoods. Keep the existing allocation-free factorization
// through 64 observations; choose solely from logical dimensions, not capacity.
fn gaussian_llt_parameters(dim: usize) -> llt::factor::LltParams {
    llt::factor::LltParams {
        recursion_threshold: if dim <= 64 { 64 } else { 32 },
        block_size: 256,
        ..faer::auto!(f64)
    }
}

pub mod collocated;
pub mod nonlinear;

pub mod disjunctive;
