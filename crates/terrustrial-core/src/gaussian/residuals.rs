//! Optional checks of the augmented kriging equations, independently accumulated.
use super::*;
use equilibration::{exponent, shift};

/// Indices in the supplied observation/drift-row and target order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResidualLocation {
    pub row: usize,
    pub target: usize,
}

/// Estimated componentwise backward error of one block of augmented equations.
/// Each equation uses `abs(sum(terms)) / sum(abs(terms))`, including its right
/// hand side among the terms. A zero denominator gives zero. This is not a
/// forward-error bound or a certificate of exact arithmetic, symmetry-preserving
/// perturbations, covariance-model validity, or prediction accuracy.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct EquationResiduals {
    pub equations: usize,
    pub max_componentwise_backward_error: f64,
    /// First maximum in row-major equation order; None for an empty block.
    pub worst: Option<ResidualLocation>,
    /// Nonzero products lost when scaled relative to an equation's largest
    /// product. This does not count ordinary rounding of representable products
    /// or compensated-sum rounding. Zero reported error is not an exactness claim.
    pub scaled_product_underflows: usize,
}

/// Checks `C W - F G = K` and `F^T W = Ft^T` in internal solve coordinates.
/// `G` is the stored drift correction (the negative Lagrange multiplier).
/// Covariance/noise and cross covariance are freshly evaluated, not reconstructed
/// from Cholesky factors. Both the prepared scaled drift constraints and freshly
/// evaluated original constraints are checked. Joint plans additionally check
/// their error covariance and sampling factor. No acceptance threshold is imposed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SolveResiduals {
    pub covariance_equations: EquationResiduals,
    pub drift_constraints: EquationResiduals,
    /// Fresh callback evaluations in original physical drift coordinates.
    pub original_drift_constraints: EquationResiduals,
    /// Stored conditional covariance versus the kriging error-covariance identity.
    /// Includes any declared roundoff repair; absent when uncertainty is disabled.
    pub conditional_covariance: Option<EquationResiduals>,
    /// Stored sampling factor versus the returned covariance, in physical units.
    pub sampling_factor: Option<EquationResiduals>,
    pub drift_qr_applied: bool,
}

/// Separate O(n * m + n) scratch for optional residual checks. Owned payloads are
/// budgeted and reused; checking performs no library allocations. User covariance
/// callbacks may allocate. Checks require O(n² + n*m) scalar covariance calls and
/// O(n*m*(n+p) + m*m*(n+p+m)) arithmetic, plus O(p*(n+m)) drift
/// callback evaluations. No new factorization or automatic refinement occurs.
pub struct ResidualWorkspace {
    maximum_observations: usize,
    maximum_targets: usize,
    row: Vec<f64>,
    weights: Vec<f64>,
    cross: Vec<f64>,
    drift_buffer: Vec<f64>,
    reservation: Reservation,
}
impl ResidualWorkspace {
    pub(crate) fn cached<'a>(
        slot: &'a mut Option<Self>,
        n: usize,
        m: usize,
        budget: &MemoryBudget,
    ) -> Result<&'a mut Self> {
        if !slot
            .as_ref()
            .is_some_and(|w| n <= w.maximum_observations && m <= w.maximum_targets)
        {
            let (n, m) = slot.as_ref().map_or((n, m), |w| {
                (n.max(w.maximum_observations), m.max(w.maximum_targets))
            });
            *slot = None;
            *slot = Some(Self::new(n, m, budget)?);
        }
        Ok(slot.as_mut().expect("initialized residual workspace"))
    }
    pub fn new(n: usize, m: usize, budget: &MemoryBudget) -> Result<Self> {
        if m == 0 {
            return Err(invalid("residual workspace needs target capacity"));
        }
        let elements = checked_len(&[n, m])?;
        let bytes = elements
            .checked_add(n)
            .and_then(|v| v.checked_mul(2))
            .and_then(|v| v.checked_mul(8))
            .ok_or(Error::SizeOverflow)?;
        if bytes > isize::MAX as usize {
            return Err(Error::SizeOverflow);
        }
        let reservation = budget.reserve(u64::try_from(bytes).map_err(|_| Error::SizeOverflow)?)?;
        fn buffer(n: usize) -> Result<Vec<f64>> {
            let mut values = Vec::new();
            values
                .try_reserve_exact(n)
                .map_err(|_| invalid("residual buffer allocation"))?;
            values.resize(n, 0.);
            Ok(values)
        }
        Ok(Self {
            maximum_observations: n,
            maximum_targets: m,
            row: buffer(n)?,
            weights: buffer(elements)?,
            cross: buffer(elements)?,
            drift_buffer: buffer(n)?,
            reservation,
        })
    }
    pub fn accounted_bytes(&self) -> u64 {
        self.reservation.bytes()
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn evaluate(
        &mut self,
        workspace: &GaussianWorkspace,
        data_model: &impl Covariance,
        target_model: &impl Covariance,
        observations: &[Observation],
        targets: &[Site],
        cancel: &Cancellation,
    ) -> Result<SolveResiduals> {
        cancel.check()?;
        let (n, m, p) = (observations.len(), targets.len(), workspace.fd.ncols());
        if n > self.maximum_observations || m > self.maximum_targets {
            return Err(invalid("residual workspace capacity exceeded"));
        }
        let scale = &workspace.equilibration;
        for j in 0..m {
            cancel.check()?;
            for i in 0..n {
                self.weights[i + j * n] =
                    scale.diagnostic_weight(workspace.plan.weights[(i, j)], i, j)?;
            }
        }
        let weights = MatRef::from_column_major_slice(&self.weights[..n * m], n, m);
        let mut covariance_equations = EquationResiduals::default();
        for i in 0..n {
            cancel.check()?;
            for r in 0..n {
                if r % 256 == 0 {
                    cancel.check()?;
                }
                // Preserve the preparation's symmetric-triangle orientation,
                // including block quadrature's floating-point reduction order.
                let value = data_model
                    .covariance(observations[i.max(r)].site(), observations[i.min(r)].site())?;
                check_finite(value)?;
                self.row[r] = scale.diagnostic_covariance(
                    value,
                    observations[i].noise_variance(),
                    i,
                    r,
                    workspace.covariance_units,
                )?;
            }
            for (j, target) in targets.iter().enumerate() {
                cancel.check()?;
                let value = target_model.covariance(observations[i].site(), target)?;
                check_finite(value)?;
                let rhs = scale.diagnostic_cross(value, i, j, workspace.covariance_units)?;
                self.cross[i + j * n] = rhs;
                let result = equation(
                    n + p + 1,
                    |r| {
                        if r < n {
                            (self.row[r], weights[(r, j)])
                        } else if r < n + p {
                            (-workspace.fd[(i, r - n)], workspace.correction[(r - n, j)])
                        } else {
                            (-rhs, 1.)
                        }
                    },
                    cancel,
                )?;
                covariance_equations.record(i, j, result)?;
            }
        }
        let mut drift_constraints = EquationResiduals::default();
        for i in 0..p {
            for j in 0..m {
                let result = equation(
                    n + 1,
                    |r| {
                        if r < n {
                            (workspace.fd[(r, i)], weights[(r, j)])
                        } else {
                            (-workspace.ft[(j, i)], 1.)
                        }
                    },
                    cancel,
                )?;
                drift_constraints.record(i, j, result)?;
            }
        }
        let (conditional_covariance, sampling_factor) = if workspace.plan.joint {
            let mut covariance = EquationResiduals::default();
            let mut factor = EquationResiduals::default();
            let cross = MatRef::from_column_major_slice(&self.cross[..n * m], n, m);
            for i in 0..m {
                for j in 0..m {
                    cancel.check()?;
                    let prior = scale.diagnostic_target_covariance(
                        target_model.covariance(&targets[i.max(j)], &targets[i.min(j)])?,
                        i,
                        j,
                        workspace.covariance_units,
                    )?;
                    let stored = scale.diagnostic_target_covariance(
                        workspace.plan.covariance[(i, j)],
                        i,
                        j,
                        workspace.covariance_units,
                    )?;
                    // V = Ctt - sym(K^T W) + sym(Ft G), independently reduced.
                    let terms = (n + p)
                        .checked_mul(2)
                        .and_then(|v| v.checked_add(4))
                        .ok_or(Error::SizeOverflow)?;
                    covariance.record(
                        i,
                        j,
                        equation(
                            terms,
                            |r| {
                                if r < n {
                                    (-cross[(r, i)], weights[(r, j)])
                                } else if r < 2 * n {
                                    (-cross[(r - n, j)], weights[(r - n, i)])
                                } else if r < 2 * n + p {
                                    (
                                        workspace.ft[(i, r - 2 * n)],
                                        workspace.correction[(r - 2 * n, j)],
                                    )
                                } else if r < 2 * (n + p) {
                                    (
                                        workspace.ft[(j, r - 2 * n - p)],
                                        workspace.correction[(r - 2 * n - p, i)],
                                    )
                                } else if r < 2 * (n + p) + 2 {
                                    (prior, 1.)
                                } else {
                                    (-stored, 1.)
                                }
                            },
                            cancel,
                        )?,
                    )?;
                    factor.record(
                        i,
                        j,
                        equation(
                            i.min(j) + 2,
                            |r| {
                                if r <= i.min(j) {
                                    (workspace.plan.factor[(i, r)], workspace.plan.factor[(j, r)])
                                } else {
                                    (-workspace.plan.covariance[(i, j)], 1.)
                                }
                            },
                            cancel,
                        )?,
                    )?;
                }
            }
            (Some(covariance), Some(factor))
        } else {
            (None, None)
        };
        cancel.check()?;
        Ok(SolveResiduals {
            covariance_equations,
            drift_constraints,
            original_drift_constraints: EquationResiduals::default(),
            conditional_covariance,
            sampling_factor,
            drift_qr_applied: workspace.drift_basis.diagnostic_r().is_some(),
        })
    }
    pub(super) fn original_drift(
        &mut self,
        workspace: &GaussianWorkspace,
        observations: &[Observation],
        targets: &[Site],
        drift: &impl Drift,
        cancel: &Cancellation,
    ) -> Result<EquationResiduals> {
        self.original_drift_with_target(workspace, observations, targets, drift, drift, cancel)
    }
    pub(super) fn original_drift_with_target(
        &mut self,
        workspace: &GaussianWorkspace,
        observations: &[Observation],
        targets: &[Site],
        drift: &impl Drift,
        target_drift: &impl Drift,
        cancel: &Cancellation,
    ) -> Result<EquationResiduals> {
        let (n, p) = (observations.len(), drift.columns());
        if n > self.maximum_observations
            || targets.len() > self.maximum_targets
            || p > n
            || target_drift.columns() != p
        {
            return Err(invalid(
                "original drift residual workspace capacity exceeded",
            ));
        }
        let buffer = &mut self.drift_buffer[..p];
        let mut report = EquationResiduals::default();
        for column in 0..p {
            for (i, observation) in observations.iter().enumerate() {
                cancel.check()?;
                evaluate_drift(drift, observation.site(), buffer)?;
                self.row[i] = buffer[column];
            }
            for (j, target) in targets.iter().enumerate() {
                cancel.check()?;
                evaluate_drift(target_drift, target, buffer)?;
                report.record(
                    column,
                    j,
                    equation(
                        n + 1,
                        |r| {
                            if r < n {
                                (self.row[r], workspace.plan.weights[(r, j)])
                            } else {
                                (-buffer[column], 1.)
                            }
                        },
                        cancel,
                    )?,
                )?;
            }
        }
        cancel.check()?;
        Ok(report)
    }
}
impl EquationResiduals {
    fn record(
        &mut self,
        row: usize,
        target: usize,
        (error, underflows): (f64, usize),
    ) -> Result<()> {
        self.equations = self.equations.checked_add(1).ok_or(Error::SizeOverflow)?;
        self.scaled_product_underflows = self
            .scaled_product_underflows
            .checked_add(underflows)
            .ok_or(Error::SizeOverflow)?;
        if self.worst.is_none() || error > self.max_componentwise_backward_error {
            self.max_componentwise_backward_error = error;
            self.worst = Some(ResidualLocation { row, target });
        }
        Ok(())
    }
}
fn check_finite(value: f64) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(Error::Numerical {
            context: "nonfinite residual diagnostic input",
        })
    }
}

#[derive(Default)]
struct Sum {
    high: f64,
    low: f64,
}
impl Sum {
    fn add(&mut self, value: f64) {
        let next = self.high + value;
        self.low += if self.high.abs() >= value.abs() {
            (self.high - next) + value
        } else {
            (value - next) + self.high
        };
        self.high = next;
    }
    fn value(&self) -> f64 {
        self.high + self.low
    }
}

// A diagnostic reduction, independent of the solver's matrix multiplication.
// Scale operands BEFORE multiplying to avoid overflowing or underflowing their
// product. FMA retains the rounded product's error; compensated addition limits
// cancellation loss. Both numerator and denominator share an equation-local
// binary exponent, which cancels in their ratio.
fn equation(
    count: usize,
    term: impl Fn(usize) -> (f64, f64),
    cancel: &Cancellation,
) -> Result<(f64, usize)> {
    cancel.check()?;
    let mut largest = i32::MIN;
    for r in 0..count {
        if r % 256 == 0 {
            cancel.check()?;
        }
        let (a, b) = term(r);
        check_finite(a)?;
        check_finite(b)?;
        if a != 0. && b != 0. {
            largest = largest.max(exponent(a.abs()) + exponent(b.abs()));
        }
    }
    if largest == i32::MIN {
        return Ok((0., 0));
    }
    let (mut numerator, mut denominator) = (Sum::default(), Sum::default());
    let mut underflows = 0;
    for r in 0..count {
        if r % 256 == 0 {
            cancel.check()?;
        }
        let (a, b) = term(r);
        if a == 0. || b == 0. {
            continue;
        }
        let (ea, eb) = (exponent(a.abs()), exponent(b.abs()));
        let (a, b) = (shift(a, -ea), shift(b, -eb));
        let product = a * b;
        let tail = a.mul_add(b, -product);
        let (value, tail) = (
            shift(product, ea + eb - largest),
            shift(tail, ea + eb - largest),
        );
        if value == 0. {
            underflows += 1;
        }
        numerator.add(value);
        numerator.add(tail);
        denominator.add(value.abs());
        denominator.add(if value < 0. { -tail } else { tail });
    }
    let ratio = numerator.value().abs() / denominator.value();
    check_finite(ratio)?;
    Ok((ratio, underflows))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::Point3;

    #[test]
    fn compensated_product_checks_cover_cancellation_overflow_and_subnormal_products() {
        let cancel = Cancellation::default();
        let check = |terms: &[(f64, f64)], expected: f64| {
            let (actual, underflows) = equation(terms.len(), |i| terms[i], &cancel).unwrap();
            assert_eq!(underflows, 0);
            assert!(
                (actual - expected).abs() <= 4. * f64::EPSILON * expected,
                "{actual:e} != {expected:e}"
            );
        };
        check(&[], 0.);
        check(&[(0., 1.), (-0., 2.)], 0.);
        let big = 2_f64.powi(1023);
        check(&[(big, big), (-big, big / 2.)], 1. / 3.);
        let tiny = f64::from_bits(1);
        check(&[(tiny, tiny), (-tiny, 2. * tiny)], 1. / 3.);
        check(
            &[(1., 1.), (2_f64.powi(-100), 1.), (-1., 1.)],
            2_f64.powi(-101),
        );
        let e = f64::EPSILON;
        check(
            &[(1. + e, 1. + e), (-(1. + 2. * e), 1.)],
            e * e / (2. + 4. * e),
        );
        // A nonzero scaled term below binary64's range is explicitly counted.
        assert_eq!(
            equation(3, |i| [(big, big), (-big, big), (tiny, tiny)][i], &cancel).unwrap(),
            (0., 1)
        );
        assert!(equation(1, |_| (f64::INFINITY, 0.), &cancel).is_err());
        assert!(equation(1, |_| (1., f64::NAN), &cancel).is_err());
        cancel.cancel();
        assert!(matches!(
            equation(1, |_| (1., 1.), &cancel),
            Err(Error::Cancelled)
        ));
    }

    struct Dense;
    impl Covariance for Dense {
        fn variables(&self) -> usize {
            3
        }
        fn point_covariance(&self, _: Point3, a: usize, _: Point3, b: usize) -> Result<f64> {
            Ok([[2., 0.5, 0.25], [0.5, 1., -0.125], [0.25, -0.125, 1.]][a][b])
        }
    }
    fn site(v: usize) -> Site {
        Site::point(Point3::new([0.; 3]).unwrap(), v)
    }
    #[test]
    fn independently_known_errors_detect_corrupted_weights_and_multipliers_without_allocating() {
        let budget = MemoryBudget::new(1 << 20);
        let cancel = Cancellation::default();
        let data = [
            Observation::new(0, site(0), 0., 0.).unwrap(),
            Observation::new(1, site(1), 0., 0.).unwrap(),
        ];
        let targets = [site(2)];
        let drift = ExternalDrift::new(1, |_: &Site, out: &mut [f64]| {
            out[0] = 1.;
            Ok(())
        })
        .unwrap();
        let options = || WorkspaceOptions {
            uncertainty: Uncertainty::Joint,
            tolerance: NumericalTolerance::default(),
            cancellation: &cancel,
        };
        let mut solver = GaussianWorkspace::new(2, 1, 1, &budget).unwrap();
        let mut diagnostic = ResidualWorkspace::new(2, 1, &budget).unwrap();
        solver
            .prepare(&Dense, &data, &targets, &drift, options())
            .unwrap();
        let good = diagnostic
            .evaluate(&solver, &Dense, &Dense, &data, &targets, &cancel)
            .unwrap();
        assert!(good.covariance_equations.max_componentwise_backward_error < 1e-15);
        assert!(good.drift_constraints.max_componentwise_backward_error < 1e-15);
        solver.plan.weights[(0, 0)] += 0.25;
        let bad = diagnostic
            .evaluate(&solver, &Dense, &Dense, &data, &targets, &cancel)
            .unwrap();
        assert!(
            (bad.covariance_equations.max_componentwise_backward_error - 8. / 45.).abs() < 1e-15
        );
        assert!((bad.drift_constraints.max_componentwise_backward_error - 1. / 9.).abs() < 1e-15);
        assert_eq!(
            bad.covariance_equations.worst,
            Some(ResidualLocation { row: 0, target: 0 })
        );
        solver
            .prepare(&Dense, &data, &targets, &drift, options())
            .unwrap();
        solver.correction[(0, 0)] += 0.5;
        let bad = diagnostic
            .evaluate(&solver, &Dense, &Dense, &data, &targets, &cancel)
            .unwrap();
        assert!(
            (bad.covariance_equations.max_componentwise_backward_error - 8. / 37.).abs() < 1e-15
        );
        assert!(bad.drift_constraints.max_componentwise_backward_error < 1e-15);
        assert_eq!(
            bad.covariance_equations.worst,
            Some(ResidualLocation { row: 1, target: 0 })
        );
        solver
            .prepare(&Dense, &data, &targets, &Simple, options())
            .unwrap();
        solver.plan.weights[(0, 0)] += 0.25;
        let bad = diagnostic
            .evaluate(&solver, &Dense, &Dense, &data, &targets, &cancel)
            .unwrap();
        assert!(
            (bad.covariance_equations.max_componentwise_backward_error - 7. / 17.).abs() < 1e-15
        );
        assert_eq!(bad.drift_constraints, EquationResiduals::default());
        let allocations = allocation_counter::measure(|| {
            for _ in 0..32 {
                assert_eq!(
                    diagnostic
                        .evaluate(&solver, &Dense, &Dense, &data, &targets, &cancel)
                        .unwrap(),
                    bad
                );
            }
        });
        assert_eq!(allocations.count_total, 0);
    }

    #[test]
    fn covariance_factor_and_original_drift_checks_detect_independent_corruption() {
        let budget = MemoryBudget::new(1 << 20);
        let cancel = Cancellation::default();
        let data = [
            Observation::new(0, site(0), 0., 0.).unwrap(),
            Observation::new(1, site(1), 0., 0.).unwrap(),
        ];
        // Duplicate targets and an exactly observed target exercise rank deficiency.
        let targets = [site(2), site(2), site(0)];
        let drift = ExternalDrift::new(1, |_: &Site, out: &mut [f64]| {
            out[0] = 1.;
            Ok(())
        })
        .unwrap();
        let options = || WorkspaceOptions {
            uncertainty: Uncertainty::Joint,
            tolerance: NumericalTolerance::default(),
            cancellation: &cancel,
        };
        let mut solver = GaussianWorkspace::new(2, 3, 1, &budget).unwrap();
        let mut diagnostic = ResidualWorkspace::new(2, 3, &budget).unwrap();
        solver
            .prepare(&Dense, &data, &targets, &drift, options())
            .unwrap();
        let report = diagnostic
            .evaluate(&solver, &Dense, &Dense, &data, &targets, &cancel)
            .unwrap();
        for block in [
            report.conditional_covariance.unwrap(),
            report.sampling_factor.unwrap(),
        ] {
            assert_eq!(block.equations, 9);
            assert!(block.max_componentwise_backward_error < 1e-14, "{block:?}");
        }
        let original = solver.plan.covariance[(0, 0)];
        solver.plan.covariance[(0, 0)] += 0.25;
        let report = diagnostic
            .evaluate(&solver, &Dense, &Dense, &data, &targets, &cancel)
            .unwrap();
        assert!(
            report
                .conditional_covariance
                .unwrap()
                .max_componentwise_backward_error
                > 0.01
        );
        assert!(
            report
                .sampling_factor
                .unwrap()
                .max_componentwise_backward_error
                > 0.01
        );
        solver.plan.covariance[(0, 0)] = original;
        solver.plan.factor[(1, 0)] += 0.25;
        let report = diagnostic
            .evaluate(&solver, &Dense, &Dense, &data, &targets, &cancel)
            .unwrap();
        assert!(
            report
                .conditional_covariance
                .unwrap()
                .max_componentwise_backward_error
                < 1e-14
        );
        assert!(
            report
                .sampling_factor
                .unwrap()
                .max_componentwise_backward_error
                > 0.01
        );
        solver
            .prepare(&Dense, &data, &targets, &drift, options())
            .unwrap();
        solver.plan.weights[(0, 0)] += 0.25;
        // A simultaneous bug in the stored drift design must not hide the error.
        solver.fd.fill(0.);
        solver.ft.fill(0.);
        let report = diagnostic
            .evaluate(&solver, &Dense, &Dense, &data, &targets, &cancel)
            .unwrap();
        assert_eq!(
            report.drift_constraints.max_componentwise_backward_error,
            0.
        );
        let original = diagnostic
            .original_drift(&solver, &data, &targets, &drift, &cancel)
            .unwrap();
        assert!((original.max_componentwise_backward_error - 1. / 9.).abs() < 1e-15);
        assert_eq!(original.worst, Some(ResidualLocation { row: 0, target: 0 }));
        let allocations = allocation_counter::measure(|| {
            for _ in 0..16 {
                diagnostic
                    .evaluate(&solver, &Dense, &Dense, &data, &targets, &cancel)
                    .unwrap();
                diagnostic
                    .original_drift(&solver, &data, &targets, &drift, &cancel)
                    .unwrap();
            }
        });
        assert_eq!(allocations.count_total, 0);
    }

    struct BrokenKernel {
        row: bool,
    }
    impl Covariance for BrokenKernel {
        fn variables(&self) -> usize {
            3
        }
        fn point_covariance(&self, a: Point3, va: usize, b: Point3, vb: usize) -> Result<f64> {
            Dense.point_covariance(a, va, b, vb)
        }
        fn covariance_row(&self, a: &Site, data: &[Observation], out: &mut [f64]) -> Result<()> {
            Dense.covariance_row(a, data, out)?;
            if self.row {
                for v in out {
                    *v += 0.125;
                }
            }
            Ok(())
        }
        fn covariance_column(&self, data: &[Observation], b: &Site, out: &mut [f64]) -> Result<()> {
            Dense.covariance_column(data, b, out)?;
            if !self.row {
                for v in out {
                    *v += 0.125;
                }
            }
            Ok(())
        }
    }
    #[test]
    fn fresh_scalar_covariances_reveal_faults_in_optimized_assembly() {
        let budget = MemoryBudget::new(1 << 20);
        let cancel = Cancellation::default();
        let data = [
            Observation::new(0, site(0), 0., 0.).unwrap(),
            Observation::new(1, site(1), 0., 0.).unwrap(),
        ];
        let targets = [site(2)];
        let mut solver = GaussianWorkspace::new(2, 1, 0, &budget).unwrap();
        let mut diagnostics = ResidualWorkspace::new(2, 1, &budget).unwrap();
        for row in [false, true] {
            let model = BrokenKernel { row };
            let mut conditioned = solver.condition(&model, &data, &Simple, &cancel).unwrap();
            let (_, report) = conditioned
                .prepare_with_residuals(
                    &targets,
                    &mut diagnostics,
                    WorkspaceOptions {
                        uncertainty: Uncertainty::None,
                        tolerance: NumericalTolerance::default(),
                        cancellation: &cancel,
                    },
                )
                .unwrap();
            assert!(report.covariance_equations.max_componentwise_backward_error > 1e-3);
        }
    }
    #[test]
    fn residual_checks_have_no_backend_allocations_at_large_shapes() {
        let budget = MemoryBudget::new(32 << 20);
        let cancel = Cancellation::default();
        let mut solver = GaussianWorkspace::new(129, 17, 0, &budget).unwrap();
        let mut diagnostics = ResidualWorkspace::new(129, 17, &budget).unwrap();
        let data: Vec<_> = (0..129)
            .map(|i| Observation::new(i, site(0), 0., 0.1).unwrap())
            .collect();
        let targets = [site(2); 17];
        solver
            .prepare(
                &Dense,
                &data,
                &targets,
                &Simple,
                WorkspaceOptions {
                    uncertainty: Uncertainty::None,
                    tolerance: NumericalTolerance::default(),
                    cancellation: &cancel,
                },
            )
            .unwrap();
        // Measure only optional checking, separating faer preparation packing.
        let allocations = allocation_counter::measure(|| {
            for _ in 0..3 {
                let report = diagnostics
                    .evaluate(&solver, &Dense, &Dense, &data, &targets, &cancel)
                    .unwrap();
                assert!(report.covariance_equations.max_componentwise_backward_error < 1e-12);
            }
        });
        assert_eq!(allocations.count_total, 0);
    }
}

/// Report for the numerical plan before the driver's exact-hard-value overwrite.
/// Target indices address the associated group’s targets. Original drift rows
/// address the caller's coefficient order; prepared drift rows may be QR changed.
/// This is a numerical diagnostic, not geological or statistical acceptance.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GroupResiduals {
    pub equations: SolveResiduals,
    pub observations: usize,
    /// Stable source ID at the worst covariance equation's observation row.
    pub covariance_observation_id: Option<u64>,
}
