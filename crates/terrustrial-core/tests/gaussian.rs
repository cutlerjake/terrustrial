use terrustrial_core::{
    data::{Observation, Point3, Site, Support},
    execution::{Cancellation, MemoryBudget},
    gaussian::{
        GaussianPlan, LinearDrift, NumericalTolerance, Ordinary, PrepareOptions, Simple,
        Uncertainty,
    },
    model::{Anisotropy, Coregionalization, Correlation, Covariance, Lmc, Structure},
};

fn point(x: f64, y: f64, z: f64) -> Point3 {
    Point3::new([x, y, z]).unwrap()
}
fn site(x: f64, variable: usize) -> Site {
    Site::point(point(x, 0.0, 0.0), variable)
}
fn structure(variance: f64) -> Structure {
    Structure {
        correlation: Correlation::Exponential,
        anisotropy: Anisotropy::isotropic(2.0).unwrap(),
        coregionalization: Coregionalization::scalar(variance).unwrap(),
    }
}
fn scalar() -> Lmc {
    Lmc::new(vec![structure(1.0)]).unwrap()
}
fn options<'a>(budget: &'a MemoryBudget, cancellation: &'a Cancellation) -> PrepareOptions<'a> {
    PrepareOptions {
        budget,
        cancellation,
        uncertainty: Uncertainty::Joint,
        tolerance: NumericalTolerance::default(),
    }
}
fn close(a: f64, b: f64) {
    assert!((a - b).abs() < 1e-10, "{a} != {b}");
}

#[test]
fn cross_variable_conditioning_and_zero_variance_are_preserved() {
    let model = Lmc::new(vec![Structure {
        correlation: Correlation::Exponential,
        anisotropy: Anisotropy::isotropic(2.0).unwrap(),
        coregionalization: Coregionalization::from_factor(2, 2, &[1.0, 0.0, 0.6, 0.8]).unwrap(),
    }])
    .unwrap();
    let observations = [Observation::new(1, site(0.0, 0), 2.0, 0.0).unwrap()];
    let budget = MemoryBudget::new(1 << 20);
    let cancel = Cancellation::default();
    let plan = GaussianPlan::prepare(
        &model,
        &observations,
        &[site(0.0, 0), site(0.0, 1)],
        &Simple,
        options(&budget, &cancel),
    )
    .unwrap();
    let mean = plan.estimate(&[2.0]).unwrap();
    close(mean[0], 2.0);
    close(mean[1], 1.2);
    let cov = plan.covariance().unwrap();
    close(cov[(0, 0)], 0.0);
    close(cov[(0, 1)], 0.0);
    close(cov[(1, 1)], 0.64);
    for seed in 0..10 {
        let draw = plan.sample(&mean, seed, 0, 0).unwrap();
        assert_eq!(draw[0], 2.0);
        assert_eq!(draw, plan.sample(&mean, seed, 0, 0).unwrap());
    }
}

#[test]
fn ordinary_uncertainty_contains_drift_estimation_variance() {
    let budget = MemoryBudget::new(1 << 20);
    let cancel = Cancellation::default();
    let observation = [Observation::new(0, site(0.0, 0), 3.0, 0.0).unwrap()];
    let plan = GaussianPlan::prepare(
        &scalar(),
        &observation,
        &[site(1.0, 0)],
        &Ordinary { variables: 1 },
        options(&budget, &cancel),
    )
    .unwrap();
    close(plan.estimate(&[3.0]).unwrap()[0], 3.0);
    close(
        plan.covariance().unwrap()[(0, 0)],
        2.0 * (1.0 - (-0.5_f64).exp()),
    );
}

#[test]
fn linear_universal_kriging_reproduces_a_linear_field() {
    let points = [
        point(0.0, 0.0, 0.0),
        point(1.0, 0.0, 0.0),
        point(0.0, 1.0, 0.0),
        point(0.0, 0.0, 1.0),
    ];
    let values = [1.0, 3.0, 4.0, 5.0];
    let observations: Vec<_> = points
        .into_iter()
        .zip(values)
        .enumerate()
        .map(|(i, (p, v))| Observation::new(i as u64, Site::point(p, 0), v, 0.0).unwrap())
        .collect();
    let target = Site::point(point(0.2, 0.3, 0.4), 0);
    let budget = MemoryBudget::new(1 << 20);
    let cancel = Cancellation::default();
    let drift = LinearDrift::new(1, [0.0; 3], 1.0).unwrap();
    let plan = GaussianPlan::prepare(
        &scalar(),
        &observations,
        &[target],
        &drift,
        options(&budget, &cancel),
    )
    .unwrap();
    close(
        plan.estimate(&values).unwrap()[0],
        1.0 + 2.0 * 0.2 + 3.0 * 0.3 + 4.0 * 0.4,
    );
}

#[test]
fn factorial_component_estimates_sum_to_total_and_blocks_average_covariance() {
    let full = Lmc::new(vec![structure(0.4), structure(0.6)]).unwrap();
    let obs = [Observation::new(0, site(0.0, 0), 2.0, 0.0).unwrap()];
    let target = Site {
        support: Support::block(point(1.0, 0.0, 0.0), [0.5, 0.0, 0.0], [2, 1, 1]).unwrap(),
        variable: 0,
    };
    let expected = 0.5 * ((-0.75_f64 / 2.0).exp() + (-1.25_f64 / 2.0).exp());
    close(full.covariance(obs[0].site(), &target).unwrap(), expected);
    let budget = MemoryBudget::new(1 << 20);
    let cancel = Cancellation::default();
    let mut sum = 0.0;
    for i in 0..2 {
        let component = full.component(i).unwrap();
        let p = GaussianPlan::prepare_component(
            &full,
            &component,
            &obs,
            std::slice::from_ref(&target),
            &Simple,
            options(&budget, &cancel),
        )
        .unwrap();
        sum += p.estimate(&[2.0]).unwrap()[0];
    }
    close(sum, 2.0 * expected);
}

#[test]
fn budgets_and_degenerate_inputs_fail_explicitly() {
    let budget = MemoryBudget::new(1 << 20);
    let cancel = Cancellation::default();
    let p = GaussianPlan::prepare(
        &scalar(),
        &[],
        &[site(0.0, 0)],
        &Simple,
        options(&budget, &cancel),
    )
    .unwrap();
    assert!(budget.used() > 0);
    close(p.covariance().unwrap()[(0, 0)], 1.0);
    drop(p);
    assert_eq!(budget.used(), 0);
    assert!(GaussianPlan::prepare(
        &scalar(),
        &[],
        &[site(0.0, 0)],
        &Simple,
        options(&MemoryBudget::new(0), &cancel)
    )
    .is_err());
    let duplicate = [
        Observation::new(0, site(0.0, 0), 1.0, 0.0).unwrap(),
        Observation::new(1, site(0.0, 0), 2.0, 0.0).unwrap(),
    ];
    assert!(GaussianPlan::prepare(
        &scalar(),
        &duplicate,
        &[site(1.0, 0)],
        &Simple,
        options(&budget, &cancel)
    )
    .is_err());
    assert_eq!(budget.used(), 0);
    cancel.cancel();
    assert!(GaussianPlan::prepare(
        &scalar(),
        &[],
        &[site(0.0, 0)],
        &Simple,
        options(&budget, &cancel)
    )
    .is_err());
}

#[test]
fn continuous_models_do_not_jump_to_the_sill_at_the_scale() {
    for model in [
        Correlation::Exponential,
        Correlation::Gaussian,
        Correlation::Matern32,
        Correlation::Matern52,
    ] {
        assert!(model.evaluate(1.0) > 0.0);
        close(model.evaluate(0.0), 1.0);
        assert!((model.evaluate(1.0 - 1e-9) - model.evaluate(1.0 + 1e-9)).abs() < 1e-8);
        close(model.evaluate(f64::INFINITY), 0.0);
    }
    assert!(Point3::new([f64::NAN, 0.0, 0.0]).is_err());
    assert!(Anisotropy::isotropic(0.0).is_err());
    assert!(Coregionalization::from_factor(2, 2, &[1.0]).is_err());
}
