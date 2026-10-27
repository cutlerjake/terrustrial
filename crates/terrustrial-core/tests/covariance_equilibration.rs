//! Joint estimates must be invariant under representable changes of variable units.
use terrustrial_core::{
    data::{Observation, Point3, Site},
    execution::{Cancellation, MemoryBudget},
    gaussian::{
        Drift, GaussianWorkspace, NumericalTolerance, Ordinary, Simple, Uncertainty,
        WorkspaceOptions,
    },
    model::{Anisotropy, Coregionalization, Correlation, Lmc, Structure},
};
fn site(x: f64, variable: usize) -> Site {
    Site::point(Point3::new([x, 0., 0.]).unwrap(), variable)
}

struct Trend(usize);
impl Drift for Trend {
    fn columns(&self) -> usize {
        self.0
    }
    fn evaluate(&self, s: &Site, out: &mut [f64]) -> terrustrial_core::Result<()> {
        out.fill(0.);
        if self.0 == 2 {
            out[s.variable] = 1.;
        }
        if self.0 == 4 {
            out[2 * s.variable] = 1.;
            out[2 * s.variable + 1] = 1. + 2_f64.powi(-24) * s.support.center().coordinates()[0];
        }
        Ok(())
    }
}

#[test]
fn correlated_units_preserve_weights_means_covariance_and_draws() {
    let budget = MemoryBudget::new(1 << 25);
    let cancel = Cancellation::default();
    let targets = [site(0.13, 0), site(0.27, 1), site(0.63, 0), site(0.88, 1)];
    let options = |uncertainty| WorkspaceOptions {
        uncertainty,
        tolerance: NumericalTolerance::default(),
        cancellation: &cancel,
    };
    let mut workspace = GaussianWorkspace::new(24, 4, 4, &budget).unwrap();
    for columns in [0, 2, 4] {
        let drift = Trend(columns);
        for component in [false, true] {
            let mut reference: Option<[Vec<f64>; 4]> = None;
            for roots in [
                [1., 1.],
                [2_f64.powi(-500), 2_f64.powi(500)],
                [2_f64.powi(450), 2_f64.powi(-450)],
                [1e-154, 1e154],
            ] {
                let model = model(roots, Correlation::Exponential, 0.6);
                let component_model =
                    crate::model(roots.map(|r| r * 0.5), Correlation::Exponential, 0.6);
                let target_model = if component { &component_model } else { &model };
                let observations: Vec<_> = (0..24)
                    .map(|i| {
                        let v = (i % 2) as usize;
                        Observation::new(
                            i,
                            site(i as f64 / 16., v),
                            roots[v] * (i as f64).sin(),
                            0.05 * roots[v] * roots[v],
                        )
                        .unwrap()
                    })
                    .collect();
                let values: Vec<_> = observations.iter().map(Observation::value).collect();
                let data_mean: Vec<_> = observations
                    .iter()
                    .map(|o| 0.25 * roots[o.site().variable])
                    .collect();
                let target_mean: Vec<_> =
                    targets.iter().map(|s| 0.25 * roots[s.variable]).collect();
                let mut conditioned = workspace
                    .condition(&model, &observations, &drift, &cancel)
                    .unwrap();
                for uncertainty in [Uncertainty::Joint, Uncertainty::None] {
                    let plan = conditioned
                        .prepare_component(target_model, &targets, options(uncertainty))
                        .unwrap_or_else(|e| {
                            panic!("columns={columns},component={component},roots={roots:?}: {e:?}")
                        });
                    let mut weights = Vec::new();
                    for j in 0..4 {
                        for i in 0..24 {
                            weights.push(plan.weights()[(i, j)] * roots[i % 2] / roots[j % 2]);
                        }
                    }
                    let prediction: Vec<_> = plan
                        .estimate_with_mean(&values, &data_mean, &target_mean)
                        .unwrap()
                        .into_iter()
                        .enumerate()
                        .map(|(i, v)| v / roots[i % 2])
                        .collect();
                    let mut covariance = Vec::new();
                    let mut draw = Vec::new();
                    if let Some(cov) = plan.covariance() {
                        for j in 0..4 {
                            for i in 0..4 {
                                covariance.push(cov[(i, j)] / roots[i % 2] / roots[j % 2]);
                            }
                        }
                        draw = plan
                            .sample(&[0.; 4], 71, 0, 1)
                            .unwrap()
                            .into_iter()
                            .enumerate()
                            .map(|(i, v)| v / roots[i % 2])
                            .collect();
                    }
                    if let Some([w, p, c, d]) = &reference {
                        let tolerance = if columns == 4 { 2e-7 } else { 2e-12 };
                        for (actual, expected) in [
                            (&weights, w),
                            (&prediction, p),
                            (&covariance, c),
                            (&draw, d),
                        ] {
                            for (a, b) in actual.iter().zip(expected) {
                                assert!((a-b).abs()<tolerance,"columns={columns},component={component},roots={roots:?}: {a} vs {b}");
                            }
                        }
                    } else {
                        reference = Some([weights, prediction, covariance, draw]);
                    }
                }
            }
        }
    }
}

#[test]
fn target_only_scaling_preserves_small_variances_and_physical_absolute_tolerances() {
    let budget = MemoryBudget::new(1 << 20);
    let cancel = Cancellation::default();
    let roots = [1e-154, 1e154];
    let model = model(roots, Correlation::Nugget, 0.);
    let targets = [site(0.5, 0), site(0.5, 1)];
    let mut workspace = GaussianWorkspace::new(1, 2, 0, &budget).unwrap();
    for absolute in [0., 2e-308] {
        // No observations and a tiny-variance observation both require target
        // scaling even though conditioning has no heterogeneous diagonal.
        for data in [
            vec![],
            vec![Observation::new(0, site(0., 0), 0., 0.).unwrap()],
        ] {
            let plan = workspace
                .prepare(
                    &model,
                    &data,
                    &targets,
                    &Simple,
                    WorkspaceOptions {
                        uncertainty: Uncertainty::Joint,
                        tolerance: NumericalTolerance {
                            absolute,
                            relative: 1e-12,
                        },
                        cancellation: &cancel,
                    },
                )
                .unwrap();
            let covariance = plan.covariance().unwrap();
            assert!((covariance[(1, 1)] / 1e308 - 1.).abs() < 1e-14);
            if absolute == 0. {
                assert!((covariance[(0, 0)] / 1e-308 - 1.).abs() < 1e-14);
            } else {
                assert_eq!(covariance[(0, 0)], 0.);
            }
            assert_eq!(covariance[(0, 1)], 0.);
        }
    }
}

#[test]
fn equilibrated_preparation_reuses_storage_and_recovers_after_target_errors() {
    let budget = MemoryBudget::new(1 << 24);
    let cancel = Cancellation::default();
    let roots = [2_f64.powi(-500), 2_f64.powi(500)];
    let model = model(roots, Correlation::Exponential, 0.6);
    let data: Vec<_> = (0..24)
        .map(|i| Observation::new(i, site(i as f64 / 16., (i % 2) as usize), 0., 0.).unwrap())
        .collect();
    let targets = [site(0.13, 0), site(0.27, 1)];
    let drift = Trend(4);
    let options = || WorkspaceOptions {
        uncertainty: Uncertainty::Joint,
        tolerance: NumericalTolerance::default(),
        cancellation: &cancel,
    };
    let mut workspace = GaussianWorkspace::new(24, 2, 4, &budget).unwrap();
    let allocations = allocation_counter::measure(|| {
        workspace
            .prepare(&model, &data, &targets, &drift, options())
            .unwrap();
    });
    assert_eq!(allocations.count_total, 0);
    let mut conditioned = workspace.condition(&model, &data, &drift, &cancel).unwrap();
    let reference = conditioned
        .prepare(&targets, options())
        .unwrap()
        .to_owned_plan(&budget)
        .unwrap();
    // A component with covariance greater than its data covariance is invalid.
    let invalid_component = crate::model(roots.map(|v| v * 2.), Correlation::Exponential, 0.6);
    let allocations = allocation_counter::measure(|| {
        for _ in 0..4 {
            assert!(conditioned
                .prepare_component(&invalid_component, &targets, options())
                .is_err());
            let plan = conditioned.prepare(&targets, options()).unwrap();
            for j in 0..2 {
                for i in 0..24 {
                    assert_eq!(
                        plan.weights()[(i, j)].to_bits(),
                        reference.weights()[(i, j)].to_bits()
                    );
                }
                for i in 0..2 {
                    assert_eq!(
                        plan.covariance().unwrap()[(i, j)].to_bits(),
                        reference.covariance().unwrap()[(i, j)].to_bits()
                    );
                }
            }
        }
    });
    assert_eq!(allocations.count_total, 0);
    drop(workspace);
    drop(reference);
    assert_eq!(budget.used(), 0);
}
fn model(roots: [f64; 2], correlation: Correlation, rho: f64) -> Lmc {
    Lmc::new(vec![Structure {
        correlation,
        anisotropy: Anisotropy::isotropic(2.).unwrap(),
        coregionalization: Coregionalization::from_factor(
            2,
            2,
            &[
                roots[0],
                0.,
                rho * roots[1],
                (1. - rho * rho).sqrt() * roots[1],
            ],
        )
        .unwrap(),
    }])
    .unwrap()
}
#[test]
fn independent_joint_variables_retain_their_own_uncertainty_units() {
    let budget = MemoryBudget::new(1 << 24);
    let cancel = Cancellation::default();
    let targets = [site(0.5, 0), site(0.5, 1), site(1.5, 0), site(1.5, 1)];
    let mut workspace = GaussianWorkspace::new(8, 4, 2, &budget).unwrap();
    for roots in [
        [1., 1.],
        [2_f64.powi(-500), 2_f64.powi(500)],
        [1e-154, 1e154],
    ] {
        let model = model(roots, Correlation::Nugget, 0.);
        let observations: Vec<_> = (0..8)
            .map(|i| Observation::new(i, site((i / 2) as f64, (i % 2) as usize), 0., 0.).unwrap())
            .collect();
        let plan = workspace
            .prepare(
                &model,
                &observations,
                &targets,
                &Ordinary { variables: 2 },
                WorkspaceOptions {
                    uncertainty: Uncertainty::Joint,
                    tolerance: NumericalTolerance::default(),
                    cancellation: &cancel,
                },
            )
            .unwrap_or_else(|e| panic!("roots={roots:?}: {e:?}"));
        for j in 0..4 {
            for i in 0..8 {
                let expected = if i % 2 == j % 2 { 0.25 } else { 0. };
                assert!((plan.weights()[(i, j)] - expected).abs() < 1e-13);
            }
            for i in 0..4 {
                let expected = if i % 2 != j % 2 {
                    0.
                } else if i == j {
                    1.25
                } else {
                    0.25
                };
                let normalized = plan.covariance().unwrap()[(i, j)] / roots[i % 2] / roots[j % 2];
                assert!(
                    (normalized - expected).abs() < 1e-13,
                    "roots={roots:?}, ({i},{j}): {normalized} vs {expected}"
                );
            }
        }
    }
}

#[test]
fn heterogeneous_measurement_noise_preserves_representable_small_weights() {
    let budget = MemoryBudget::new(1 << 20);
    let cancel = Cancellation::default();
    let mut workspace = GaussianWorkspace::new(2, 1, 1, &budget).unwrap();
    let model = Lmc::new(vec![Structure {
        correlation: Correlation::Nugget,
        anisotropy: Anisotropy::isotropic(1.).unwrap(),
        coregionalization: Coregionalization::scalar(1.).unwrap(),
    }])
    .unwrap();
    // Independent observations have variances 1 and 1+noise. The ordinary
    // weights are (1+noise)/(2+noise) and 1/(2+noise). The second weighted
    // value remains important even when its weight is subnormal.
    for noise in [1e20, 1e300, f64::MAX] {
        let data = [
            Observation::new(0, site(0., 0), 2., 0.).unwrap(),
            Observation::new(1, site(1., 0), noise, noise).unwrap(),
        ];
        let plan = workspace
            .prepare(
                &model,
                &data,
                &[site(0.5, 0)],
                &Ordinary { variables: 1 },
                WorkspaceOptions {
                    uncertainty: Uncertainty::Joint,
                    tolerance: NumericalTolerance::default(),
                    cancellation: &cancel,
                },
            )
            .unwrap();
        assert!((plan.weights()[(0, 0)] - 1.).abs() < 1e-14);
        assert!((plan.weights()[(1, 0)] * noise - 1.).abs() < 1e-14);
        assert!((plan.estimate(&[2., noise]).unwrap()[0] - 3.).abs() < 1e-14);
        assert!((plan.covariance().unwrap()[(0, 0)] - 2.).abs() < 1e-14);
    }
}
