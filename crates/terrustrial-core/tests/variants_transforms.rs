use terrustrial_core::{
    data::{Observation, Point3, Site},
    execution::{Cancellation, MemoryBudget},
    gaussian::{
        collocated,
        nonlinear::{categorical_indicators, threshold_indicators},
        GaussianPlan, NumericalTolerance, PrepareOptions, Simple, Uncertainty,
    },
    model::{
        Anisotropy, Coregionalization, Correlation, Covariance, Lmc, SpatialKernel, Structure,
    },
    transforms::{NormalScore, TailPolicy},
};
fn site(x: f64, v: usize) -> Site {
    Site::point(Point3::new([x, 0.0, 0.0]).unwrap(), v)
}
fn options<'a>(budget: &'a MemoryBudget, cancel: &'a Cancellation) -> PrepareOptions<'a> {
    PrepareOptions {
        uncertainty: Uncertainty::Joint,
        tolerance: NumericalTolerance::default(),
        budget,
        cancellation: cancel,
    }
}
#[test]
fn weighted_normal_scores_ties_tails_and_round_trips_are_explicit() {
    let budget = MemoryBudget::new(1 << 20);
    let transform = NormalScore::fit(
        &[(4.0, 1.0), (1.0, 1.0), (1.0, 1.0), (2.0, 2.0)],
        TailPolicy::Clamp,
        &budget,
    )
    .unwrap();
    let knots: Vec<_> = transform.knots().collect();
    assert_eq!(knots.len(), 3);
    assert!((knots[0].1 + 0.8416212335729143).abs() < 1e-12);
    for value in [1.0, 1.1, 1.9, 2.0, 3.0, 4.0] {
        assert!(
            (transform
                .inverse(transform.forward(value).unwrap())
                .unwrap()
                - value)
                .abs()
                < 1e-12
        );
    }
    assert_eq!(transform.inverse(-100.0).unwrap(), 1.0);
    assert_eq!(transform.inverse(100.0).unwrap(), 4.0);
    let (mean, var) = transform.conditional_moments(0.2, 0.0, 100).unwrap();
    assert_eq!(var, 0.0);
    assert_eq!(mean, transform.inverse(0.2).unwrap());
    let coarse = transform.conditional_moments(0.0, 1.0, 1024).unwrap();
    let fine = transform.conditional_moments(0.0, 1.0, 2048).unwrap();
    assert!((coarse.0 - fine.0).abs() < 1e-5);
    assert!((coarse.1 - fine.1).abs() < 1e-4);
    assert!(budget.used() > 0);
    drop(transform);
    assert_eq!(budget.used(), 0);
    let transform =
        NormalScore::fit(&[(0.0, 1.0), (1.0, 1.0)], TailPolicy::Reject, &budget).unwrap();
    assert!(transform.forward(2.0).is_err());
    assert!(NormalScore::fit(&[(0.0, -1.0)], TailPolicy::Clamp, &budget).is_err());
    assert!(NormalScore::fit(&[(0.0, 1e-300), (1.0, 1e300)], TailPolicy::Clamp, &budget).is_err());
}
#[test]
fn indicator_predictions_honor_categories_and_cumulative_thresholds() {
    let budget = MemoryBudget::new(1 << 20);
    let cancel = Cancellation::default();
    let model = Lmc::new(vec![Structure {
        correlation: Correlation::Exponential,
        anisotropy: Anisotropy::isotropic(1.0).unwrap(),
        coregionalization: Coregionalization::scalar(1.0).unwrap(),
    }])
    .unwrap();
    let observations = [Observation::new(0, site(0.0, 0), 0.0, 0.0).unwrap()];
    let plan = GaussianPlan::prepare(
        &model,
        &observations,
        &[site(0.0, 0), site(1.0, 0)],
        &Simple,
        options(&budget, &cancel),
    )
    .unwrap();
    let probabilities = categorical_indicators(&plan, &[1], &[0.3, 0.7]).unwrap();
    assert_eq!(probabilities[0].corrected, vec![0.0, 1.0]);
    let expected = 0.7 + 0.3 * (-1.0_f64).exp();
    assert!((probabilities[1].corrected[1] - expected).abs() < 1e-12);
    let cdf = threshold_indicators(&plan, &[2.0], &[1.0, 2.0, 3.0], &[0.1, 0.5, 0.9]).unwrap();
    assert_eq!(cdf[0].corrected, vec![0.0, 1.0, 1.0]);
    assert!(threshold_indicators(&plan, &[2.0], &[2.0, 1.0], &[0.1, 0.5]).is_err());
}
#[test]
fn collocated_markov_model_is_valid_and_approximation_reports_omitted_data() {
    let model = collocated::markov_one(
        SpatialKernel {
            correlation: Correlation::Exponential,
            anisotropy: Anisotropy::isotropic(2.0).unwrap(),
        },
        &[1.0, 0.6],
        Coregionalization::from_factor(2, 1, &[0.0, 0.8]).unwrap(),
    )
    .unwrap();
    let origin = site(0.0, 0);
    let target = site(1.0, 0);
    assert!(
        (model
            .point_covariance(origin.support.center(), 0, target.support.center(), 1)
            .unwrap()
            - 0.6 * (-0.5_f64).exp())
        .abs()
            < 1e-12
    );
    let observations = [
        Observation::new(0, origin, 1.0, 0.0).unwrap(),
        Observation::new(1, site(1.0, 1), 2.0, 0.0).unwrap(),
        Observation::new(2, site(2.0, 1), 3.0, 0.0).unwrap(),
    ];
    let selected = collocated::select(&observations, std::slice::from_ref(&target), &[0]).unwrap();
    assert_eq!(selected.indices, vec![0, 1]);
    assert_eq!(selected.omitted_secondary, 1);
    let budget = MemoryBudget::new(1 << 20);
    let cancel = Cancellation::default();
    let full = GaussianPlan::prepare(
        &model,
        &observations,
        std::slice::from_ref(&target),
        &Simple,
        options(&budget, &cancel),
    )
    .unwrap();
    let reduced = GaussianPlan::prepare(
        &model,
        &selected.observations,
        &[target],
        &Simple,
        options(&budget, &cancel),
    )
    .unwrap();
    assert!(reduced.covariance().unwrap()[(0, 0)] >= full.covariance().unwrap()[(0, 0)] - 1e-12);
    assert!(
        (reduced.estimate(&[1.0, 2.0]).unwrap()[0] - full.estimate(&[1.0, 2.0, 3.0]).unwrap()[0])
            .abs()
            > 1e-3
    );
}
