use terrustrial_core::{
    data::Point3,
    execution::{Cancellation, MemoryBudget},
    fitting::{fit_lmc, CrossVariogramBin, LmcFitOptions},
    model::{Anisotropy, Correlation, Covariance, SpatialKernel},
};
fn kernel() -> SpatialKernel {
    SpatialKernel {
        correlation: Correlation::Exponential,
        anisotropy: Anisotropy::isotropic(2.0).unwrap(),
    }
}
fn bins(sill: [[f64; 2]; 2]) -> Vec<CrossVariogramBin> {
    let mut bins = Vec::new();
    for (a, row) in sill.iter().enumerate() {
        for (b, &value) in row.iter().enumerate().skip(a) {
            for i in 1..=12 {
                let h = i as f64 / 2.0;
                bins.push(CrossVariogramBin {
                    variables: [a, b],
                    lag: [h, 0.0, 0.0],
                    semivariance: value * (1.0 - (-h / 2.0).exp()),
                    pairs: 100,
                });
            }
        }
    }
    bins
}
#[test]
fn joint_fit_recovers_negative_cross_sill_and_preserves_covariance_psd() {
    let budget = MemoryBudget::new(1 << 20);
    let cancel = Cancellation::default();
    let fit = fit_lmc(
        2,
        &bins([[1.0, -0.6], [-0.6, 2.0]]),
        &[kernel()],
        LmcFitOptions {
            maximum_iterations: 5000,
            tolerance: 1e-11,
            budget: &budget,
            cancellation: &cancel,
        },
    )
    .unwrap();
    assert!(fit.converged);
    assert!(fit.weighted_sse < 1e-18);
    assert!(fit.unobserved_pairs.is_empty());
    assert_eq!(budget.used(), 0);
    let p = Point3::new([0.0; 3]).unwrap();
    assert!((fit.model.point_covariance(p, 0, p, 1).unwrap() + 0.6).abs() < 1e-9);
    let matrix = faer::Mat::from_fn(8, 8, |i, j| {
        fit.model
            .point_covariance(
                Point3::new([(i / 2) as f64, 0.0, 0.0]).unwrap(),
                i % 2,
                Point3::new([(j / 2) as f64, 0.0, 0.0]).unwrap(),
                j % 2,
            )
            .unwrap()
    });
    assert!(matrix
        .self_adjoint_eigenvalues(faer::Side::Lower)
        .unwrap()
        .iter()
        .all(|v| *v >= -1e-12));
}
#[test]
fn inconsistent_empirical_cross_variogram_is_projected_to_a_valid_joint_model() {
    let budget = MemoryBudget::new(1 << 20);
    let cancel = Cancellation::default();
    let fit = fit_lmc(
        2,
        &bins([[1.0, 2.0], [2.0, 1.0]]),
        &[kernel()],
        LmcFitOptions {
            maximum_iterations: 5000,
            tolerance: 1e-10,
            budget: &budget,
            cancellation: &cancel,
        },
    )
    .unwrap();
    assert!(fit.converged);
    assert!(fit.weighted_sse > 0.1);
    let sill = &fit.model.structures()[0].coregionalization;
    assert!(
        sill.covariance(0, 1).unwrap().powi(2)
            <= sill.covariance(0, 0).unwrap() * sill.covariance(1, 1).unwrap() + 1e-12
    );
    let limited = fit_lmc(
        2,
        &bins([[1.0, -0.6], [-0.6, 2.0]]),
        &[kernel()],
        LmcFitOptions {
            maximum_iterations: 1,
            tolerance: 1e-14,
            budget: &budget,
            cancellation: &cancel,
        },
    )
    .unwrap();
    assert!(!limited.converged);
}
