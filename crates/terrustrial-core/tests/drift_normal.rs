//! Independent orthogonal-design reference for dense drift normal equations.
use terrustrial_core::{
    data::{Observation, Point3, Site},
    execution::{Cancellation, MemoryBudget},
    gaussian::{
        ExternalDrift, GaussianWorkspace, NumericalTolerance, ResidualWorkspace, Uncertainty,
        WorkspaceOptions,
    },
    model::Covariance,
    Result,
};
struct White;
impl Covariance for White {
    fn variables(&self) -> usize {
        1
    }
    fn point_covariance(&self, a: Point3, _: usize, b: Point3, _: usize) -> Result<f64> {
        Ok(f64::from(a == b))
    }
}
fn basis(x: f64, j: usize, n: usize) -> f64 {
    if j == 0 {
        std::f64::consts::FRAC_1_SQRT_2
    } else {
        (std::f64::consts::PI * (x + 0.5) * j as f64 / n as f64).cos()
    }
}
fn site(x: f64) -> Site {
    Site::point(Point3::new([x, 0., 0.]).unwrap(), 0)
}
#[test]
fn dense_drift_weights_constraints_and_joint_uncertainty_match_orthogonal_design() {
    let budget = MemoryBudget::new(64 << 20);
    let cancel = Cancellation::default();
    for (n, p, m) in [(8, 1, 2), (32, 8, 7), (128, 32, 9), (129, 64, 17)] {
        let data: Vec<_> = (0..n)
            .map(|i| Observation::new(i as u64, site(i as f64), (i as f64).sin(), 0.25).unwrap())
            .collect();
        let targets: Vec<_> = (0..m).map(|j| site((n + j) as f64)).collect();
        let drift = ExternalDrift::new(p, |s: &Site, out: &mut [f64]| {
            for (j, value) in out.iter_mut().enumerate() {
                *value = basis(s.support.center().coordinates()[0], j, n);
            }
            Ok(())
        })
        .unwrap();
        let mut solver = GaussianWorkspace::new(n, m, p, &budget).unwrap();
        let mut diagnostic = ResidualWorkspace::new(n, m, &budget).unwrap();
        let mut conditioned = solver.condition(&White, &data, &drift, &cancel).unwrap();
        let (plan, report) = conditioned
            .prepare_with_residuals(
                &targets,
                &mut diagnostic,
                WorkspaceOptions {
                    uncertainty: Uncertainty::Joint,
                    tolerance: NumericalTolerance::default(),
                    cancellation: &cancel,
                },
            )
            .unwrap();
        assert!(!report.drift_qr_applied);
        assert!(report.covariance_equations.max_componentwise_backward_error < 1e-10);
        assert!(report.drift_constraints.max_componentwise_backward_error < 1e-10);
        // F^T F = n/2 I and C = 1.25 I. K is zero at the distinct targets.
        // Thus W = (2/n) F Ft^T, V = I + (2.5/n) Ft Ft^T.
        for j in 0..m {
            for i in 0..n {
                let expected = 2. / n as f64
                    * (0..p)
                        .map(|k| basis(i as f64, k, n) * basis((n + j) as f64, k, n))
                        .sum::<f64>();
                assert!((plan.weights()[(i, j)] - expected).abs() <= 1e-10 * (1. + expected.abs()));
            }
            for i in 0..m {
                let expected = f64::from(i == j)
                    + 2.5 / n as f64
                        * (0..p)
                            .map(|k| basis((n + i) as f64, k, n) * basis((n + j) as f64, k, n))
                            .sum::<f64>();
                assert!(
                    (plan.covariance().unwrap()[(i, j)] - expected).abs()
                        <= 1e-10 * (1. + expected.abs())
                );
            }
        }
    }
    assert_eq!(budget.used(), 0);
}
