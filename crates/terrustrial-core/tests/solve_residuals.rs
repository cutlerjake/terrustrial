use std::sync::atomic::{AtomicUsize, Ordering};
use terrustrial_core::{
    data::{Observation, Point3, Site, Support},
    execution::{Cancellation, MemoryBudget},
    gaussian::{
        Drift, GaussianPlan, GaussianWorkspace, NumericalTolerance, ResidualWorkspace, Simple,
        SolveResiduals, Uncertainty, WorkspaceOptions,
    },
    model::Covariance,
    Error, Result,
};

struct Model {
    variables: usize,
    units: Vec<f64>,
    component: f64,
}
impl Covariance for Model {
    fn variables(&self) -> usize {
        self.variables
    }
    fn point_covariance(&self, a: Point3, va: usize, b: Point3, vb: usize) -> Result<f64> {
        let d: f64 = a
            .coordinates()
            .iter()
            .zip(b.coordinates())
            .map(|(a, b)| (a - b).powi(2))
            .sum();
        let corr = if va == vb {
            1.
        } else if (va + vb).is_multiple_of(2) {
            0.2
        } else {
            -0.2
        };
        Ok(corr * (-d.sqrt() / 2.).exp() * self.units[va] * self.units[vb] * self.component)
    }
}
struct Trend {
    variables: usize,
    epsilon: Option<f64>,
}
impl Drift for Trend {
    fn columns(&self) -> usize {
        self.variables * if self.epsilon.is_some() { 2 } else { 1 }
    }
    fn evaluate(&self, s: &Site, out: &mut [f64]) -> Result<()> {
        out.fill(0.);
        if let Some(e) = self.epsilon {
            out[2 * s.variable] = 1.;
            out[2 * s.variable + 1] = 1. + e * s.support.center().coordinates()[0];
        } else {
            out[s.variable] = 1.;
        }
        Ok(())
    }
}
fn site(x: f64, v: usize, block: bool) -> Site {
    let center = Point3::new([x, 0.1 * x.sin(), 0.]).unwrap();
    Site {
        variable: v,
        support: if block {
            Support::block(center, [0.02, 0.03, 0.], [2, 2, 1]).unwrap()
        } else {
            Support::point(center)
        },
    }
}
fn options(cancel: &Cancellation, uncertainty: Uncertainty) -> WorkspaceOptions<'_> {
    WorkspaceOptions {
        uncertainty,
        tolerance: NumericalTolerance::default(),
        cancellation: cancel,
    }
}
fn bits(plan: &GaussianPlan) -> Vec<u64> {
    plan.weights()
        .col_iter()
        .flat_map(|c| c.iter().map(|v| v.to_bits()))
        .chain(
            plan.covariance()
                .into_iter()
                .flat_map(|a| a.col_iter().flat_map(|c| c.iter().map(|v| v.to_bits()))),
        )
        .collect()
}
fn check(report: SolveResiduals, n: usize, m: usize, p: usize) {
    assert_eq!(report.covariance_equations.equations, n * m);
    assert_eq!(report.drift_constraints.equations, p * m);
    assert_eq!(report.original_drift_constraints.equations, p * m);
    assert_eq!(
        report.conditional_covariance.is_some(),
        report.sampling_factor.is_some()
    );
    for block in [report.conditional_covariance, report.sampling_factor]
        .into_iter()
        .flatten()
    {
        assert_eq!(block.equations, m * m);
    }
    for block in [
        Some(report.covariance_equations),
        Some(report.drift_constraints),
        Some(report.original_drift_constraints),
        report.conditional_covariance,
        report.sampling_factor,
    ]
    .into_iter()
    .flatten()
    {
        assert!(
            block.max_componentwise_backward_error <= 1e-10,
            "{report:?}"
        );
        assert_eq!(block.scaled_product_underflows, 0);
        assert_eq!(block.worst.is_none(), block.equations == 0);
    }
}
#[test]
fn joint_point_block_component_and_qr_reports_preserve_plan_bits_across_extreme_units() {
    let budget = MemoryBudget::new(32 << 20);
    let cancel = Cancellation::default();
    let mut solver = GaussianWorkspace::new(24, 9, 4, &budget).unwrap();
    let mut diagnostics = ResidualWorkspace::new(24, 9, &budget).unwrap();
    for units in [
        [1., 1.],
        [2_f64.powi(-300); 2],
        [2_f64.powi(300); 2],
        [2_f64.powi(-500), 2_f64.powi(500)],
    ] {
        let model = Model {
            variables: 2,
            units: units.to_vec(),
            component: 1.,
        };
        let component = Model {
            variables: 2,
            units: units.to_vec(),
            component: 0.25,
        };
        let data: Vec<_> = (0..24)
            .map(|i| {
                Observation::new(
                    i as u64,
                    site(i as f64 / 8., i % 2, i == 7),
                    0.,
                    0.05 * units[i % 2] * units[i % 2],
                )
                .unwrap()
            })
            .collect();
        for epsilon in [None, Some(0.25), Some(2_f64.powi(-28))] {
            let drift = Trend {
                variables: 2,
                epsilon,
            };
            let mut conditioned = solver.condition(&model, &data, &drift, &cancel).unwrap();
            for m in [1, 7, 8, 9, 2] {
                let targets: Vec<_> = (0..m)
                    .map(|i| site(0.13 + i as f64 / 9., i % 2, i == 1))
                    .collect();
                for target in [&model, &component] {
                    for uncertainty in [Uncertainty::None, Uncertainty::Joint] {
                        let baseline = bits(
                            conditioned
                                .prepare_component(target, &targets, options(&cancel, uncertainty))
                                .unwrap(),
                        );
                        let (plan, report) = conditioned
                            .prepare_component_with_residuals(
                                target,
                                &targets,
                                &mut diagnostics,
                                options(&cancel, uncertainty),
                            )
                            .unwrap();
                        assert_eq!(bits(plan), baseline);
                        check(report, 24, m, drift.columns());
                        if epsilon == Some(2_f64.powi(-28)) {
                            assert!(report.drift_qr_applied);
                        }
                    }
                }
            }
        }
    }
    drop((solver, diagnostics));
    assert_eq!(budget.used(), 0);
}

#[test]
fn changing_dimensions_cover_joint_variables_and_backend_shape_boundaries() {
    let budget = MemoryBudget::new(128 << 20);
    let cancel = Cancellation::default();
    let mut solver = GaussianWorkspace::new(512, 16, 32, &budget).unwrap();
    let mut diagnostics = ResidualWorkspace::new(512, 16, &budget).unwrap();
    for (variables, n, m) in [
        (1, 8, 1),
        (2, 32, 7),
        (8, 64, 8),
        (32, 128, 9),
        (32, 512, 16),
        (1, 1, 1),
        (1, 0, 2),
    ] {
        let model = Model {
            variables,
            units: vec![1.; variables],
            component: 1.,
        };
        let data: Vec<_> = (0..n)
            .map(|i| {
                Observation::new(
                    i as u64,
                    site(i as f64 / 7., i % variables, false),
                    0.,
                    0.05,
                )
                .unwrap()
            })
            .collect();
        let targets: Vec<_> = (0..m)
            .map(|i| site(i as f64 / 9., i % variables, false))
            .collect();
        let mut conditioned = solver.condition(&model, &data, &Simple, &cancel).unwrap();
        let (_, report) = conditioned
            .prepare_with_residuals(
                &targets,
                &mut diagnostics,
                options(&cancel, Uncertainty::Joint),
            )
            .unwrap();
        check(report, n, m, 0);
        if n > 0 {
            let drift = Trend {
                variables,
                epsilon: None,
            };
            let mut conditioned = solver.condition(&model, &data, &drift, &cancel).unwrap();
            let (_, report) = conditioned
                .prepare_with_residuals(
                    &targets,
                    &mut diagnostics,
                    options(&cancel, Uncertainty::Joint),
                )
                .unwrap();
            check(report, n, m, variables);
        }
    }
}

struct Injected {
    calls: AtomicUsize,
    fail_at: AtomicUsize,
    mode: AtomicUsize,
    cancellation: Cancellation,
}
impl Covariance for Injected {
    fn variables(&self) -> usize {
        1
    }
    fn point_covariance(&self, a: Point3, _: usize, b: Point3, _: usize) -> Result<f64> {
        let call = self.calls.fetch_add(1, Ordering::Relaxed);
        if call == self.fail_at.load(Ordering::Relaxed) {
            match self.mode.load(Ordering::Relaxed) {
                0 => {
                    return Err(Error::Numerical {
                        context: "injected residual model failure",
                    })
                }
                1 => return Ok(f64::NAN),
                _ => self.cancellation.cancel(),
            }
        }
        Ok((-(a.coordinates()[0] - b.coordinates()[0]).abs()).exp())
    }
}
#[test]
fn callback_failure_nonfinite_cancellation_and_capacity_errors_allow_reuse_and_release_budget() {
    let budget = MemoryBudget::new(1 << 20);
    let cancel = Cancellation::default();
    let model = Injected {
        calls: AtomicUsize::new(0),
        fail_at: AtomicUsize::new(usize::MAX),
        mode: AtomicUsize::new(0),
        cancellation: cancel.clone(),
    };
    let data: Vec<_> = (0..4)
        .map(|i| Observation::new(i, site(i as f64, 0, false), 0., 0.01).unwrap())
        .collect();
    let targets = [site(0.5, 0, false), site(1.5, 0, false)];
    let mut solver = GaussianWorkspace::new(4, 2, 0, &budget).unwrap();
    let mut diagnostics = ResidualWorkspace::new(4, 2, &budget).unwrap();
    assert_eq!(diagnostics.accounted_bytes(), 16 * (4 * 2 + 4));
    let mut conditioned = solver.condition(&model, &data, &Simple, &cancel).unwrap();
    let baseline = bits(
        conditioned
            .prepare(&targets, options(&cancel, Uncertainty::None))
            .unwrap(),
    );
    let mut short = ResidualWorkspace::new(3, 2, &budget).unwrap();
    assert!(conditioned
        .prepare_with_residuals(&targets, &mut short, options(&cancel, Uncertainty::None))
        .is_err());
    for mode in [0, 1] {
        model.calls.store(0, Ordering::Relaxed);
        // Preparation: 2 target diagonals + 8 cross entries. Fail during fresh
        // diagnostic row assembly, after preparation has succeeded.
        model.fail_at.store(12, Ordering::Relaxed);
        model.mode.store(mode, Ordering::Relaxed);
        assert!(conditioned
            .prepare_with_residuals(
                &targets,
                &mut diagnostics,
                options(&cancel, Uncertainty::None)
            )
            .is_err());
        model.fail_at.store(usize::MAX, Ordering::Relaxed);
        let (plan, report) = conditioned
            .prepare_with_residuals(
                &targets,
                &mut diagnostics,
                options(&cancel, Uncertainty::None),
            )
            .unwrap();
        assert_eq!(bits(plan), baseline);
        check(report, 4, 2, 0);
    }
    model.calls.store(0, Ordering::Relaxed);
    model.fail_at.store(12, Ordering::Relaxed);
    model.mode.store(2, Ordering::Relaxed);
    assert!(matches!(
        conditioned.prepare_with_residuals(
            &targets,
            &mut diagnostics,
            options(&cancel, Uncertainty::None)
        ),
        Err(Error::Cancelled)
    ));
    model.fail_at.store(usize::MAX, Ordering::Relaxed);
    let fresh = Cancellation::default();
    let (plan, report) = conditioned
        .prepare_with_residuals(
            &targets,
            &mut diagnostics,
            options(&fresh, Uncertainty::None),
        )
        .unwrap();
    assert_eq!(bits(plan), baseline);
    check(report, 4, 2, 0);
    drop((solver, diagnostics, short));
    assert_eq!(budget.used(), 0);
    let short = MemoryBudget::new(95);
    assert!(matches!(
        ResidualWorkspace::new(4, 2, &short),
        Err(Error::MemoryBudget { .. })
    ));
    assert_eq!(short.used(), 0);
    assert!(matches!(
        ResidualWorkspace::new(usize::MAX, 2, &budget),
        Err(Error::SizeOverflow)
    ));
    assert!(ResidualWorkspace::new(4, 0, &budget).is_err());
}
