//! Fixed conditioning preparation and its scoped reuse handle.
use super::*;

/// A scoped preparation of one fixed conditioning system, reused across targets.
///
/// The workspace, observations, data model and drift remain borrowed until this
/// handle is dropped. Changing any of them requires a new conditioning call.
/// Covariance and drift outputs must remain deterministic and unchanged, including
/// outputs derived from interior mutable state. This is explicit reuse, not a
/// pointer-identity cache. Prediction values and known means are supplied to the
/// returned plan and can vary without rebuilding covariance or drift factors.
/// Target errors do not invalidate the prepared conditioning system.
pub struct ConditionedWorkspace<'a, C, D> {
    workspace: &'a mut GaussianWorkspace,
    model: &'a C,
    observations: &'a [Observation],
    drift: &'a D,
    variables: usize,
    columns: usize,
}
impl<C: Covariance, D: Drift> ConditionedWorkspace<'_, C, D> {
    /// Diagnose the covariance and drift systems without modifying preparation.
    /// This performs faer SVDs using a caller-owned workspace; ordinary preparation
    /// and prediction do not compute these diagnostics or pay their cost.
    /// The report uses internal scaled coordinates, not physical-unit condition
    /// numbers. Failed diagnostics leave conditioning usable.
    pub fn diagnostics(
        &self,
        workspace: &mut DiagnosticWorkspace,
        cancellation: &Cancellation,
    ) -> Result<ConditioningDiagnostics> {
        workspace.evaluate(self.workspace, cancellation)
    }

    pub fn prepare(
        &mut self,
        targets: &[Site],
        options: WorkspaceOptions<'_>,
    ) -> Result<&GaussianPlan> {
        self.prepare_component(self.model, targets, options)
    }

    /// Prepare targets and check the resulting weights against the augmented
    /// equations. Diagnostics are opt-in and use separate reusable scratch.
    /// Callback failures/cancellation return no report or plan; this conditioning
    /// handle remains usable. No threshold, repair or regularization is applied.
    pub fn prepare_with_residuals(
        &mut self,
        targets: &[Site],
        diagnostics: &mut ResidualWorkspace,
        options: WorkspaceOptions<'_>,
    ) -> Result<(&GaussianPlan, SolveResiduals)> {
        self.prepare_component_with_residuals(self.model, targets, diagnostics, options)
    }

    /// Component-model counterpart of `prepare_with_residuals`. The caller must
    /// establish the same independent decomposition as for `prepare_component`.
    pub fn prepare_component_with_residuals(
        &mut self,
        target_model: &impl Covariance,
        targets: &[Site],
        diagnostics: &mut ResidualWorkspace,
        options: WorkspaceOptions<'_>,
    ) -> Result<(&GaussianPlan, SolveResiduals)> {
        let cancel = options.cancellation;
        self.prepare_component(target_model, targets, options)?;
        let mut report = diagnostics.evaluate(
            self.workspace,
            self.model,
            target_model,
            self.observations,
            targets,
            cancel,
        )?;
        report.original_drift_constraints = diagnostics.original_drift(
            self.workspace,
            self.observations,
            targets,
            self.drift,
            cancel,
        )?;
        Ok((&self.workspace.plan, report))
    }

    /// Prepare an independent component of the borrowed data model. As with
    /// GaussianPlan::prepare_component, the caller establishes the decomposition.
    pub fn prepare_component(
        &mut self,
        target_model: &impl Covariance,
        targets: &[Site],
        options: WorkspaceOptions<'_>,
    ) -> Result<&GaussianPlan> {
        self.prepare_component_with_drift(target_model, targets, self.drift, options)
    }

    /// Prepare an independent component with its own target drift response.
    /// Observation and target bases use the same coefficient ordering, but their
    /// values may differ. A zero-mean component uses a zero target basis with the
    /// same column count as the observation drift. Known means are supplied at
    /// prediction, separately from these unknown-coefficient constraints.
    pub fn prepare_component_with_drift(
        &mut self,
        target_model: &impl Covariance,
        targets: &[Site],
        target_drift: &impl Drift,
        options: WorkspaceOptions<'_>,
    ) -> Result<&GaussianPlan> {
        options.cancellation.check()?;
        if self.model.variables() != self.variables || self.drift.columns() != self.columns {
            return Err(invalid("prepared conditioning callback dimensions changed"));
        }
        self.workspace.prepare_targets(
            target_model,
            self.observations,
            targets,
            target_drift,
            self.variables,
            self.columns,
            options,
        )
    }
}

impl GaussianWorkspace {
    /// Assemble/factor the observation covariance and drift normal equations once.
    /// Repeated target preparations through the returned handle retain those
    /// factors. Dropping it permits ordinary workspace use or new conditioning.
    /// No conditioning observations are copied. Backend packing allocations are
    /// subject to the same shape-dependent limits as ordinary preparation.
    ///
    /// ```
    /// use terrustrial_core::{
    ///     data::{Observation, Point3, Site},
    ///     execution::{Cancellation, MemoryBudget},
    ///     gaussian::{GaussianWorkspace, DiagnosticWorkspace, ResidualWorkspace, Simple, WorkspaceOptions, NumericalTolerance, Uncertainty},
    ///     model::{Lmc, Structure, Correlation, Anisotropy, Coregionalization},
    /// };
    /// # fn main() -> terrustrial_core::Result<()> {
    /// let model = Lmc::new(vec![Structure {
    ///     correlation: Correlation::Exponential,
    ///     anisotropy: Anisotropy::isotropic(2.)?,
    ///     coregionalization: Coregionalization::scalar(1.)?,
    /// }])?;
    /// let data = [Observation::new(0, Site::point(Point3::new([0.; 3])?, 0), 2., 0.)?];
    /// let budget = MemoryBudget::new(1 << 16);
    /// let cancel = Cancellation::default();
    /// let mut workspace = GaussianWorkspace::new(1, 1, 0, &budget)?;
    /// let mut conditioned = workspace.condition(&model, &data, &Simple, &cancel)?;
    /// // Optional diagnostics run once for this conditioning system.
    /// let mut diagnostic = DiagnosticWorkspace::new(data.len(), &budget)?;
    /// let report = conditioned.diagnostics(&mut diagnostic, &cancel)?;
    /// assert_eq!(report.covariance.unwrap().reciprocal_condition, 1.);
    /// let mut residuals = ResidualWorkspace::new(data.len(), 1, &budget)?;
    /// for x in [0.5, 1., 1.5] {
    ///     let targets = [Site::point(Point3::new([x, 0., 0.])?, 0)];
    ///     let (plan, report) = conditioned.prepare_with_residuals(&targets, &mut residuals, WorkspaceOptions {
    ///         uncertainty: Uncertainty::Joint,
    ///         tolerance: NumericalTolerance::default(), cancellation: &cancel,
    ///     })?;
    ///     assert!(report.covariance_equations.max_componentwise_backward_error < 1e-12);
    ///     let prediction = plan.estimate(&[2.])?;
    ///     assert!((prediction[0] - 2. * (-x / 2.).exp()).abs() < 1e-12);
    /// }
    /// # Ok(()) }
    /// ```
    pub fn condition<'a, C: Covariance, D: Drift>(
        &'a mut self,
        model: &'a C,
        observations: &'a [Observation],
        drift: &'a D,
        cancellation: &Cancellation,
    ) -> Result<ConditionedWorkspace<'a, C, D>> {
        let (variables, columns) = self.prepare_data(model, observations, drift, cancellation)?;
        Ok(ConditionedWorkspace {
            workspace: self,
            model,
            observations,
            drift,
            variables,
            columns,
        })
    }
    pub(super) fn prepare_data(
        &mut self,
        data_model: &impl Covariance,
        observations: &[Observation],
        drift: &impl Drift,
        cancellation: &Cancellation,
    ) -> Result<(usize, usize)> {
        cancellation.check()?;
        let n = observations.len();
        let p = drift.columns();
        let variables = data_model.variables();
        if p > n {
            return Err(invalid("underdetermined drift"));
        }
        if n > self.maximum_observations || p > self.maximum_drift {
            return Err(invalid("Gaussian workspace capacity exceeded"));
        }
        self.ids.clear();
        for observation in observations {
            if !self.ids.insert(observation.id()) {
                return Err(invalid("duplicate observation ID"));
            }
            if observation.site().variable >= variables {
                return Err(invalid("observation variable out of range"));
            }
        }
        let Self {
            c,
            covariance_units,
            equilibration,
            fd,
            cinv_f,
            normal,
            drift_basis,
            buffer,
            scratch,
            ..
        } = self;
        *covariance_units = 1.0;
        drift_basis.reset();
        c.resize_with(n, n, |_, _| 0.0);
        fd.resize_with(n, p, |_, _| 0.0);
        cinv_f.resize_with(n, p, |_, _| 0.0);
        normal.resize_with(p, p, |_, _| 0.0);
        // A single retained allocation holds callback values and column scales.
        // Diagonal rescaling preserves the drift span and avoids squaring units
        // as large as 1e200 (or as small as 1e-200) in the normal equations.
        buffer.resize(2 * p, 0.0);
        let (values, scales) = buffer.split_at_mut(p);
        scales.fill(0.0);
        let covariance_span = crate::execution::profiling::Span::new(
            crate::execution::profiling::Metric::DataCovariance,
        );
        let mut first = 0;
        while first < n {
            cancellation.check()?;
            let end = covariance_batch_end(first, n, |i| observations[i].site());
            let mut sites = [*observations[first].site(); 8];
            let mut columns = c.as_mut().subcols_mut(first, end - first).col_iter_mut();
            let mut rows: [&mut [f64]; 8] = std::array::from_fn(|offset| -> &mut [f64] {
                let Some(column) = columns.next() else {
                    return &mut [];
                };
                sites[offset] = *observations[first + offset].site();
                &mut column
                    .try_as_col_major_mut()
                    .expect("owned covariance column")
                    .as_slice_mut()[..=first + offset]
            });
            data_model.covariance_rows(
                &sites[..end - first],
                &observations[..end],
                &mut rows[..end - first],
            )?;
            drop(columns);
            for i in first..end {
                for j in 0..i {
                    c[(i, j)] = c[(j, i)];
                }
            }
            first = end;
        }
        drop(covariance_span);
        let _factor_span =
            crate::execution::profiling::Span::new(crate::execution::profiling::Metric::DataFactor);
        for (i, observation) in observations.iter().enumerate() {
            cancellation.check()?;
            evaluate_drift(drift, observation.site(), values)?;
            for j in 0..p {
                fd[(i, j)] = values[j];
                scales[j] = scales[j].max(values[j].abs());
            }
        }
        for j in 0..p {
            if scales[j] == 0.0 {
                return Err(Error::Singular {
                    context: "drift column is zero throughout the conditioning neighborhood",
                });
            }
            if scales[j] != 1.0 {
                for i in 0..n {
                    fd[(i, j)] = normalize_drift(fd[(i, j)], scales[j])?;
                }
            }
        }
        finite(c.as_ref(), "conditioning covariance")?;
        *covariance_units = equilibration.prepare_data(c.as_mut(), fd.as_mut(), observations)?;
        if n > 0 {
            factor_spd_in(c.as_mut(), "conditioning covariance; merge consistent duplicates or specify measurement noise explicitly", scratch)?;
            if p > 0 {
                cinv_f.as_mut().copy_from(fd.as_ref());
                solve_spd(c.as_ref(), cinv_f.as_mut());
                let factor_result =
                    factor_drift_normal(fd.as_ref(), cinv_f.as_ref(), normal.as_mut(), scratch);
                let fallback = match factor_result {
                    Ok(()) => rank::DriftBasis::suspect(normal.as_ref()),
                    Err(Error::Singular { .. }) => true,
                    Err(error) => return Err(error),
                };
                if fallback {
                    drift_basis.orthogonalize(fd.as_mut(), cinv_f.as_mut(), scratch)?;
                    solve_spd(c.as_ref(), cinv_f.as_mut());
                    factor_drift_normal(fd.as_ref(), cinv_f.as_ref(), normal.as_mut(), scratch)?;
                }
            }
        }
        Ok((variables, p))
    }
}

fn factor_drift_normal(
    basis: MatRef<'_, f64>,
    cinv_basis: MatRef<'_, f64>,
    mut normal: MatMut<'_, f64>,
    scratch: &mut MemBuffer,
) -> Result<()> {
    faer::linalg::matmul::matmul(
        normal.as_mut(),
        faer::Accum::Replace,
        basis.transpose(),
        cinv_basis,
        1.0,
        Par::Seq,
    );
    finite(normal.as_ref(), "drift normal equations")?;
    factor_spd_in(
        normal,
        "drift columns are not identifiable in this neighborhood",
        scratch,
    )
}

impl GaussianWorkspace {
    /// Prepare and independently check one complete system with caller-owned
    /// residual scratch. Errors return no plan; subsequent preparation can reuse
    /// the workspace. No threshold or repair is applied.
    #[allow(clippy::too_many_arguments)]
    pub fn prepare_with_residuals(
        &mut self,
        model: &impl Covariance,
        observations: &[Observation],
        targets: &[Site],
        drift: &impl Drift,
        diagnostics: &mut ResidualWorkspace,
        options: WorkspaceOptions<'_>,
    ) -> Result<(&GaussianPlan, SolveResiduals)> {
        let cancel = options.cancellation;
        self.prepare(model, observations, targets, drift, options)?;
        let mut report = diagnostics.evaluate(self, model, model, observations, targets, cancel)?;
        report.original_drift_constraints =
            diagnostics.original_drift(self, observations, targets, drift, cancel)?;
        Ok((&self.plan, report))
    }
}
