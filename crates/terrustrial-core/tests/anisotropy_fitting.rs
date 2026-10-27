use terrustrial_core::{
    anisotropy::{
        fitting::{fit_gaussian_tensor, DirectionalBin, TensorFitOptions},
        Tensor,
    },
    data::Point3,
    execution::{Cancellation, MemoryBudget},
    model::Anisotropy,
};
fn options<'a>(budget: &'a MemoryBudget, cancel: &'a Cancellation) -> TensorFitOptions<'a> {
    TensorFitOptions {
        structured_sill: 1.7,
        nugget: 0.1,
        range_bounds: [0.2, 10.0],
        maximum_iterations: 5000,
        tolerance: 1e-11,
        budget,
        cancellation: cancel,
    }
}
fn bins(anisotropy: &Anisotropy) -> Vec<DirectionalBin> {
    let origin = Point3::new([0.0; 3]).unwrap();
    let directions = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [1.0, -1.0, 1.0],
    ];
    directions
        .into_iter()
        .flat_map(|direction| {
            [0.3, 0.6, 0.9].map(|scale| {
                let lag = direction.map(|v| v * scale);
                let h = anisotropy.normalized_distance(origin, Point3::new(lag).unwrap());
                DirectionalBin {
                    lag,
                    semivariance: 0.1 + 1.7 * (1.0 - (-h * h).exp()),
                    pairs: 100,
                }
            })
        })
        .collect()
}
#[test]
fn fitted_local_tensor_recovers_known_rotated_anisotropy() {
    let c = std::f64::consts::FRAC_1_SQRT_2;
    let known = Anisotropy::new(
        [[c, c, 0.0], [-c, c, 0.0], [0.0, 0.0, 1.0]],
        [5.0, 2.0, 1.0],
    )
    .unwrap();
    let budget = MemoryBudget::new(1 << 20);
    let cancel = Cancellation::default();
    let fit = fit_gaussian_tensor(&bins(&known), options(&budget, &cancel)).unwrap();
    assert!(fit.converged);
    assert_eq!(fit.axes_at_bound, 0);
    assert!(fit.weighted_transformed_sse < 1e-20);
    assert!(fit.design_condition_number < 20.0);
    let actual = fit.tensor.matrix();
    let expected = Tensor::from_anisotropy(&known).unwrap().matrix();
    for i in 0..3 {
        for j in 0..3 {
            assert!((actual[i][j] - expected[i][j]).abs() < 1e-8);
        }
    }
    assert_eq!(budget.used(), 0);
}
#[test]
fn planar_evidence_is_rejected_and_saturated_bins_are_reported() {
    let known = Anisotropy::isotropic(2.0).unwrap();
    let data = bins(&known);
    let planar: Vec<_> = data
        .iter()
        .filter(|bin| bin.lag[2] == 0.0)
        .cloned()
        .collect();
    let budget = MemoryBudget::new(1 << 20);
    let cancel = Cancellation::default();
    assert!(fit_gaussian_tensor(&planar, options(&budget, &cancel)).is_err());
    let mut data = data;
    data.push(DirectionalBin {
        lag: [1.0, 0.0, 0.0],
        semivariance: 2.0,
        pairs: 100,
    });
    let fit = fit_gaussian_tensor(&data, options(&budget, &cancel)).unwrap();
    assert_eq!(fit.excluded_bins, 1);
    assert!(fit.converged);
    let mut limited = options(&budget, &cancel);
    limited.range_bounds = [0.2, 1.5];
    let fit = fit_gaussian_tensor(&data, limited).unwrap();
    assert_eq!(fit.axes_at_bound, 3);
    assert!(fit.weighted_transformed_sse > 0.0);
}
