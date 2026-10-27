use terrustrial_core::{
    data::{Observation, Point3, Site},
    execution::{Cancellation, MemoryBudget},
    gaussian::{
        DiagnosticWorkspace, ExternalDrift, GaussianWorkspace, NumericalTolerance, Ordinary,
        Simple, Uncertainty, WorkspaceOptions,
    },
    model::Covariance,
    Error, Result,
};
struct Dense {
    n: usize,
    values: Vec<f64>,
}
impl Covariance for Dense {
    fn variables(&self) -> usize {
        self.n
    }
    fn point_covariance(&self, _: Point3, a: usize, _: Point3, b: usize) -> Result<f64> {
        Ok(self.values[a * self.n + b])
    }
}
fn observations(n: usize) -> Vec<Observation> {
    (0..n)
        .map(|i| {
            Observation::new(
                i as u64,
                Site::point(Point3::new([i as f64, 0., 0.]).unwrap(), i),
                i as f64,
                0.,
            )
            .unwrap()
        })
        .collect()
}
fn relative(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance * expected.abs(),
        "{actual:e} != {expected:e}"
    );
}
fn options(cancel: &Cancellation) -> WorkspaceOptions<'_> {
    WorkspaceOptions {
        uncertainty: Uncertainty::Joint,
        tolerance: NumericalTolerance::default(),
        cancellation: cancel,
    }
}

#[test]
fn covariance_condition_matches_analytic_eigenvalues_with_noise_and_extreme_units() {
    let budget = MemoryBudget::new(16 << 20);
    let cancel = Cancellation::default();
    let mut solver = GaussianWorkspace::new(2, 2, 0, &budget).unwrap();
    let mut diagnostic = DiagnosticWorkspace::new(2, &budget).unwrap();
    for rho in [0., 0.5, -0.75, 1. - 2_f64.powi(-40)] {
        for units in [2_f64.powi(-600), 1., 2_f64.powi(600)] {
            for noise in [0., 0.25] {
                let model = Dense {
                    n: 2,
                    values: vec![units, rho * units, rho * units, units],
                };
                let data: Vec<_> = observations(2)
                    .iter()
                    .map(|o| Observation::new(o.id(), *o.site(), o.value(), noise * units).unwrap())
                    .collect();
                let conditioned = solver.condition(&model, &data, &Simple, &cancel).unwrap();
                let report = conditioned.diagnostics(&mut diagnostic, &cancel).unwrap();
                let condition = report.covariance.unwrap();
                let expected = (1. + noise - rho.abs()) / (1. + noise + rho.abs());
                relative(condition.reciprocal_condition, expected, 1e-6);
                assert!((condition.log10_condition + expected.log10()).abs() < 1e-6);
                assert_eq!(condition.computed_zero_singular_values, 0);
                assert_eq!((condition.rows, condition.columns), (2, 2));
                assert!(!report.observation_equilibration);
                assert!(report.drift_design.is_none());
                assert!(report.drift_normal.is_none());
                assert!(!report.drift_qr_applied);
                if units != 1. {
                    assert_ne!(report.common_covariance_scale, 1.);
                }
            }
        }
    }
    let model = Dense {
        n: 2,
        values: vec![2_f64.powi(-600), 0.5, 0.5, 2_f64.powi(600)],
    };
    let data = observations(2);
    let conditioned = solver.condition(&model, &data, &Simple, &cancel).unwrap();
    let report = conditioned.diagnostics(&mut diagnostic, &cancel).unwrap();
    assert!(report.observation_equilibration);
    assert_eq!(report.common_covariance_scale, 1.);
    relative(
        report.covariance.unwrap().reciprocal_condition,
        1. / 3.,
        1e-12,
    );
    drop((solver, diagnostic));
    assert_eq!(budget.used(), 0);
}

#[test]
fn drift_reports_original_scaled_design_and_post_qr_normal_equations() {
    let budget = MemoryBudget::new(16 << 20);
    let cancel = Cancellation::default();
    let n = 4;
    let model = Dense {
        n,
        values: (0..n * n).map(|i| f64::from(i / n == i % n)).collect(),
    };
    let data = observations(n);
    let mut solver = GaussianWorkspace::new(n, 1, 2, &budget).unwrap();
    let mut diagnostic = DiagnosticWorkspace::new(n, &budget).unwrap();
    for epsilon in [0.25, 2_f64.powi(-28)] {
        let basis = [[1., 1.], [0., epsilon], [1., 1. + epsilon], [0., -epsilon]];
        let drift = ExternalDrift::new(2, |s: &Site, out: &mut [f64]| {
            out.copy_from_slice(&basis[s.variable]);
            Ok(())
        })
        .unwrap();
        let conditioned = solver.condition(&model, &data, &drift, &cancel).unwrap();
        let report = conditioned.diagnostics(&mut diagnostic, &cancel).unwrap();
        let f: Vec<_> = basis
            .iter()
            .map(|r| [r[0], r[1] / (1. + epsilon)])
            .collect();
        let a = f.iter().map(|r| r[0] * r[0]).sum::<f64>();
        let b = f.iter().map(|r| r[0] * r[1]).sum::<f64>();
        let d = f.iter().map(|r| r[1] * r[1]).sum::<f64>();
        let largest = 0.5 * (a + d + ((a - d).powi(2) + 4. * b * b).sqrt());
        // Cauchy-Binet determinant, avoiding cancellation in a*d-b*b.
        let determinant = (0..n)
            .flat_map(|i| (i + 1..n).map(move |j| (i, j)))
            .map(|(i, j)| (f[i][0] * f[j][1] - f[i][1] * f[j][0]).powi(2))
            .sum::<f64>();
        let expected = determinant.sqrt() / largest;
        let design = report.drift_design.unwrap();
        relative(design.reciprocal_condition, expected, 1e-6);
        assert_eq!((design.rows, design.columns), (4, 2));
        assert_eq!(report.drift_qr_applied, epsilon < 1e-6);
        if report.drift_qr_applied {
            relative(report.drift_normal.unwrap().reciprocal_condition, 1., 1e-6);
        } else {
            relative(
                report.drift_normal.unwrap().reciprocal_condition,
                expected * expected,
                1e-12,
            );
        }
    }
}

#[test]
fn diagnostics_do_not_change_plans_and_failures_leave_conditioning_usable() {
    let budget = MemoryBudget::new(16 << 20);
    let cancel = Cancellation::default();
    let model = Dense {
        n: 3,
        values: vec![1., 0.1, 0.2, 0.1, 1., 0.3, 0.2, 0.3, 1.],
    };
    let data = observations(3);
    let targets: Vec<_> = data.iter().map(|o| *o.site()).collect();
    let mut solver = GaussianWorkspace::new(3, 3, 3, &budget).unwrap();
    let drift = Ordinary { variables: 3 };
    let mut conditioned = solver.condition(&model, &data, &drift, &cancel).unwrap();
    let snapshot = |plan: &terrustrial_core::gaussian::GaussianPlan| {
        let mut result = Vec::new();
        let w = plan.weights();
        for j in 0..w.ncols() {
            for i in 0..w.nrows() {
                result.push(w[(i, j)].to_bits());
            }
        }
        let c = plan.covariance().unwrap();
        for j in 0..c.ncols() {
            for i in 0..c.nrows() {
                result.push(c[(i, j)].to_bits());
            }
        }
        result.extend(
            plan.estimate(&[0., 1., 2.])
                .unwrap()
                .into_iter()
                .map(f64::to_bits),
        );
        result
    };
    let expected = snapshot(conditioned.prepare(&targets, options(&cancel)).unwrap());
    let mut too_small = DiagnosticWorkspace::new(2, &budget).unwrap();
    assert!(conditioned.diagnostics(&mut too_small, &cancel).is_err());
    let mut diagnostic = DiagnosticWorkspace::new(3, &budget).unwrap();
    let report = conditioned.diagnostics(&mut diagnostic, &cancel).unwrap();
    let cancelled = Cancellation::default();
    cancelled.cancel();
    assert!(matches!(
        conditioned.diagnostics(&mut diagnostic, &cancelled),
        Err(Error::Cancelled)
    ));
    let bad = [Site::point(Point3::new([0.; 3]).unwrap(), 99)];
    assert!(conditioned.prepare(&bad, options(&cancel)).is_err());
    assert_eq!(
        report,
        conditioned.diagnostics(&mut diagnostic, &cancel).unwrap()
    );
    assert_eq!(
        expected,
        snapshot(conditioned.prepare(&targets, options(&cancel)).unwrap())
    );
    let measured = allocation_counter::measure(|| {
        for _ in 0..32 {
            assert_eq!(
                report,
                conditioned.diagnostics(&mut diagnostic, &cancel).unwrap()
            );
        }
    });
    assert_eq!(measured.count_total, 0);
    let bytes = diagnostic.accounted_bytes();
    let insufficient = MemoryBudget::new(bytes - 1);
    assert!(matches!(
        DiagnosticWorkspace::new(3, &insufficient),
        Err(Error::MemoryBudget { .. })
    ));
    assert_eq!(insufficient.used(), 0);
    assert!(matches!(
        DiagnosticWorkspace::new(usize::MAX, &budget),
        Err(Error::SizeOverflow)
    ));
    drop((diagnostic, too_small, solver));
    assert_eq!(budget.used(), 0);
}

#[test]
fn empty_scalar_and_backend_threshold_shapes_reuse_the_diagnostic_workspace() {
    let budget = MemoryBudget::new(128 << 20);
    let cancel = Cancellation::default();
    let mut diagnostic = DiagnosticWorkspace::new(129, &budget).unwrap();
    for n in [0, 1, 2, 7, 31, 127, 128, 129, 2, 0] {
        let model = Dense {
            n,
            values: (0..n * n)
                .map(|i| if i / n == i % n { 1. } else { 0. })
                .collect(),
        };
        let data = observations(n);
        let mut solver = GaussianWorkspace::new(n, 1, 0, &budget).unwrap();
        let conditioned = solver.condition(&model, &data, &Simple, &cancel).unwrap();
        let mut result = None;
        let measured = allocation_counter::measure(|| {
            result = Some(conditioned.diagnostics(&mut diagnostic, &cancel).unwrap());
        });
        let warmed = allocation_counter::measure(|| {
            result = Some(conditioned.diagnostics(&mut diagnostic, &cancel).unwrap());
        });
        eprintln!("n={n} initial={measured:?} warm={warmed:?}");
        assert_eq!(warmed.count_current, 0, "backend allocation leak, n={n}");
        assert_eq!(warmed.bytes_current, 0, "backend byte leak, n={n}");
        if n <= 32 {
            assert_eq!(warmed.count_total, 0, "n={n}");
        }
        // Faer packing is outside its reported SVD scratch. Bound the observed
        // transient allocation on these fixtures without claiming it is absent.
        assert!(warmed.count_total <= (4 * n) as u64, "n={n}, {warmed:?}");
        assert!(
            warmed.bytes_max <= (64 * n * n + 65536) as u64,
            "n={n}, {warmed:?}"
        );
        let report = result.unwrap();
        if n == 0 {
            assert!(report.covariance.is_none());
        } else {
            let c = report.covariance.unwrap();
            assert_eq!(c.reciprocal_condition, 1.);
            assert_eq!(c.log10_condition, 0.);
        }
    }
}

#[test]
fn nontrivial_large_covariance_and_rectangular_drift_match_analytic_spectra() {
    let budget = MemoryBudget::new(128 << 20);
    let cancel = Cancellation::default();
    let mut diagnostic = DiagnosticWorkspace::new(257, &budget).unwrap();
    for n in [7, 31, 129, 257] {
        let model = Dense {
            n,
            values: (0..n * n)
                .map(|index| {
                    let i = index / n;
                    let j = index % n;
                    if i == j {
                        1.
                    } else if i.abs_diff(j) == 1 {
                        -0.25
                    } else {
                        0.
                    }
                })
                .collect(),
        };
        let data = observations(n);
        let mut solver = GaussianWorkspace::new(n, 1, 0, &budget).unwrap();
        let conditioned = solver.condition(&model, &data, &Simple, &cancel).unwrap();
        let report = conditioned.diagnostics(&mut diagnostic, &cancel).unwrap();
        let c = (std::f64::consts::PI / (n + 1) as f64).cos();
        relative(
            report.covariance.unwrap().reciprocal_condition,
            (1. - 0.5 * c) / (1. + 0.5 * c),
            1e-12,
        );
    }
    for (n, p) in [(32, 17), (33, 18), (129, 17)] {
        let model = Dense {
            n,
            values: (0..n * n).map(|i| f64::from(i / n == i % n)).collect(),
        };
        let data = observations(n);
        let mut solver = GaussianWorkspace::new(n, 1, p, &budget).unwrap();
        let drift = ExternalDrift::new(p, |s: &Site, out: &mut [f64]| {
            for (j, v) in out.iter_mut().enumerate() {
                *v = (std::f64::consts::PI * (s.variable as f64 + 0.5) * (j + 1) as f64 / n as f64)
                    .cos();
            }
            Ok(())
        })
        .unwrap();
        let conditioned = solver.condition(&model, &data, &drift, &cancel).unwrap();
        let report = conditioned.diagnostics(&mut diagnostic, &cancel).unwrap();
        let maxima: Vec<f64> = (1..=p)
            .map(|j| {
                (0..n)
                    .map(|i| {
                        (std::f64::consts::PI * (i as f64 + 0.5) * j as f64 / n as f64)
                            .cos()
                            .abs()
                    })
                    .fold(0., f64::max)
            })
            .collect();
        let expected = maxima.iter().copied().fold(f64::INFINITY, f64::min)
            / maxima.iter().copied().fold(0., f64::max);
        relative(
            report.drift_design.unwrap().reciprocal_condition,
            expected,
            1e-12,
        );
        relative(
            report.drift_normal.unwrap().reciprocal_condition,
            expected * expected,
            1e-12,
        );
        assert!(!report.drift_qr_applied);
    }
}
