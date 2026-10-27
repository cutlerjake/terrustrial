//! Covariance units must not change weights or turn exact interpolation into NaN.
use terrustrial_core::{
    data::{Observation, Point3, Site},
    execution::{Cancellation, MemoryBudget},
    gaussian::{
        GaussianWorkspace, NumericalTolerance, Ordinary, Simple, Uncertainty, WorkspaceOptions,
    },
    model::{Anisotropy, Coregionalization, Correlation, Lmc, Structure},
};
fn site(x: f64) -> Site {
    Site::point(Point3::new([x, 0., 0.]).unwrap(), 0)
}
fn model(variance: f64) -> Lmc {
    Lmc::new(vec![Structure {
        correlation: Correlation::Nugget,
        anisotropy: Anisotropy::isotropic(1.).unwrap(),
        coregionalization: Coregionalization::scalar(variance).unwrap(),
    }])
    .unwrap()
}
#[test]
fn ordinary_weights_and_covariance_match_analytic_reference_at_extreme_units() {
    let observations: Vec<_> = (0..24)
        .map(|i| Observation::new(i, site(i as f64), 2., 0.).unwrap())
        .collect();
    let targets = [site(0.5), site(1.5)];
    let budget = MemoryBudget::new(1 << 24);
    let cancel = Cancellation::default();
    let mut workspace = GaussianWorkspace::new(24, 2, 1, &budget).unwrap();
    for variance in [
        1.,
        1e-308,
        1e308,
        f64::MIN_POSITIVE,
        1e-320,
        f64::from_bits(1),
    ] {
        let model = model(variance);
        let allocations = allocation_counter::measure(|| {
            let plan = workspace
                .prepare(
                    &model,
                    &observations,
                    &targets,
                    &Ordinary { variables: 1 },
                    WorkspaceOptions {
                        uncertainty: Uncertainty::Joint,
                        tolerance: NumericalTolerance::default(),
                        cancellation: &cancel,
                    },
                )
                .unwrap_or_else(|e| panic!("variance={variance}: {e:?}"));
            for j in 0..2 {
                for i in 0..24 {
                    assert!((plan.weights()[(i, j)] - 1. / 24.).abs() < 1e-13);
                }
                for i in 0..2 {
                    let expected = if i == j { 1. } else { 0. } + 1. / 24.;
                    assert!(
                        (plan.covariance().unwrap()[(i, j)] / variance - expected).abs()
                            < 1e-13 + 4. * f64::from_bits(1) / variance
                    );
                }
            }
        });
        assert_eq!(allocations.count_total, 0);
    }
}
#[test]
fn extreme_exact_interpolation_survives_symmetric_covariance_products() {
    let budget = MemoryBudget::new(1 << 26);
    let cancel = Cancellation::default();
    for (n, m) in [(1, 1), (64, 16)] {
        let observations: Vec<_> = (0..n)
            .map(|i| Observation::new(i, site(i as f64), 2., 0.).unwrap())
            .collect();
        let targets: Vec<_> = (0..m).map(|i| site(i as f64)).collect();
        let mut workspace = GaussianWorkspace::new(n as usize, m, 0, &budget).unwrap();
        for variance in [1e-308, 1e308] {
            let plan = workspace
                .prepare(
                    &model(variance),
                    &observations,
                    &targets,
                    &Simple,
                    WorkspaceOptions {
                        uncertainty: Uncertainty::Joint,
                        tolerance: NumericalTolerance::default(),
                        cancellation: &cancel,
                    },
                )
                .unwrap_or_else(|e| panic!("n={n},m={m},variance={variance}: {e:?}"));
            for j in 0..m {
                assert!((plan.weights()[(j, j)] - 1.).abs() < 1e-14);
                for i in 0..m {
                    assert_eq!(plan.covariance().unwrap()[(i, j)], 0.);
                }
            }
        }
    }
}

#[test]
fn overflowing_observation_variance_preserves_representable_posterior() {
    struct TestDrift(bool);
    impl terrustrial_core::gaussian::Drift for TestDrift {
        fn columns(&self) -> usize {
            usize::from(self.0)
        }
        fn evaluate(&self, _: &Site, out: &mut [f64]) -> terrustrial_core::Result<()> {
            out.fill(1.);
            Ok(())
        }
    }
    let budget = MemoryBudget::new(1 << 26);
    let cancel = Cancellation::default();
    // C = 2*v*I, K = v*[I;0]. Simple weights are I/2. Ordinary
    // weights add 1/(2*n) to every observation, with error covariance
    // v*(I/2 + 11^T/(2*n)). C itself need not fit in physical units.
    for (n, m) in [(1, 1), (24, 4), (64, 16)] {
        let mut workspace = GaussianWorkspace::new(n, m, 1, &budget).unwrap();
        for variance in [1., 1e308, f64::MAX] {
            let model = model(variance);
            let observations: Vec<_> = (0..n)
                .map(|i| Observation::new(i as u64, site(i as f64), 2., variance).unwrap())
                .collect();
            let targets: Vec<_> = (0..m).map(|i| site(i as f64)).collect();
            for ordinary in [false, true] {
                let drift = TestDrift(ordinary);
                let mut prepared = workspace
                    .condition(&model, &observations, &drift, &cancel)
                    .unwrap_or_else(|e| panic!("n={n},v={variance},ordinary={ordinary}: {e:?}"));
                for uncertainty in [Uncertainty::None, Uncertainty::Joint] {
                    let plan = prepared
                        .prepare(
                            &targets,
                            WorkspaceOptions {
                                uncertainty,
                                tolerance: NumericalTolerance::default(),
                                cancellation: &cancel,
                            },
                        )
                        .unwrap_or_else(|e| {
                            panic!("n={n},m={m},v={variance},ordinary={ordinary},uncertainty={uncertainty:?}: {e:?}")
                        });
                    let extra = if ordinary { 0.5 / n as f64 } else { 0. };
                    for j in 0..m {
                        for i in 0..n {
                            let expected = if i == j { 0.5 } else { 0. } + extra;
                            assert!((plan.weights()[(i, j)] - expected).abs() < 1e-13);
                        }
                        if let Some(covariance) = plan.covariance() {
                            for i in 0..m {
                                let expected = if i == j { 0.5 } else { 0. } + extra;
                                assert!((covariance[(i, j)] / variance - expected).abs() < 1e-13);
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn overflowing_noise_reuses_storage_and_rejects_unrepresentable_outputs() {
    let budget = MemoryBudget::new(1 << 20);
    let cancel = Cancellation::default();
    let variance = f64::MAX;
    let model = model(variance);
    let observations = [Observation::new(0, site(0.), 2., variance).unwrap()];
    let targets = [site(0.)];
    let mut workspace = GaussianWorkspace::new(1, 1, 1, &budget).unwrap();
    let options = || WorkspaceOptions {
        uncertainty: Uncertainty::Joint,
        tolerance: NumericalTolerance::default(),
        cancellation: &cancel,
    };
    let allocations = allocation_counter::measure(|| {
        workspace
            .prepare(
                &model,
                &observations,
                &targets,
                &Ordinary { variables: 1 },
                options(),
            )
            .unwrap();
    });
    assert_eq!(allocations.count_total, 0);
    let ordinary = Ordinary { variables: 1 };
    let mut conditioned = workspace
        .condition(&model, &observations, &ordinary, &cancel)
        .unwrap();
    let valid = conditioned.prepare(&targets, options()).unwrap();
    let covariance = valid.covariance().unwrap()[(0, 0)];
    let draw = valid.sample(&[0.], 71, 0, 1).unwrap()[0] / variance.sqrt();
    let mut unit_workspace = GaussianWorkspace::new(1, 1, 1, &budget).unwrap();
    let unit_observations = [Observation::new(0, site(0.), 2., 1.).unwrap()];
    let unit_model = Lmc::new(vec![Structure {
        correlation: Correlation::Nugget,
        anisotropy: Anisotropy::isotropic(1.).unwrap(),
        coregionalization: Coregionalization::scalar(1.).unwrap(),
    }])
    .unwrap();
    let unit_plan = unit_workspace
        .prepare(
            &unit_model,
            &unit_observations,
            &targets,
            &ordinary,
            options(),
        )
        .unwrap();
    assert!((draw - unit_plan.sample(&[0.], 71, 0, 1).unwrap()[0]).abs() < 1e-13);
    // At a new site ordinary kriging has variance v + (v + noise) = 3*v.
    // That overflow is material, so it must not be repaired into MAX.
    assert!(matches!(
        conditioned.prepare(&[site(1.)], options()),
        Err(terrustrial_core::Error::Numerical {
            context: "conditional variance exceeds representable physical units"
        })
    ));
    let recovered = conditioned.prepare(&targets, options()).unwrap();
    assert_eq!(
        recovered.covariance().unwrap()[(0, 0)].to_bits(),
        covariance.to_bits()
    );
}

#[test]
fn joint_predictions_and_draws_preserve_units_with_qr_and_measurement_noise() {
    use terrustrial_core::gaussian::ExternalDrift;
    let budget = MemoryBudget::new(1 << 26);
    let cancel = Cancellation::default();
    let mut workspace = GaussianWorkspace::new(24, 4, 3, &budget).unwrap();
    for columns in [2, 3] {
        let drift = ExternalDrift::new(columns, |s: &Site, out: &mut [f64]| {
            out[0] = (s.variable == 0) as u8 as f64;
            out[1] = (s.variable == 1) as u8 as f64;
            if columns == 3 {
                out[2] = out[0] + 2_f64.powi(-24) * s.support.center().coordinates()[0];
            }
            Ok(())
        })
        .unwrap();
        let mut reference: Option<(Vec<f64>, Vec<f64>, Vec<f64>)> = None;
        for variance in [1_f64, 1e-308, 1e308] {
            let root = variance.sqrt();
            let model = Lmc::new(vec![Structure {
                correlation: Correlation::Exponential,
                anisotropy: Anisotropy::isotropic(2.).unwrap(),
                coregionalization: Coregionalization::from_factor(
                    2,
                    2,
                    &[root, 0., 0.6 * root, 0.8 * root],
                )
                .unwrap(),
            }])
            .unwrap();
            let observations: Vec<_> = (0..24)
                .map(|i| {
                    let mut s = site((i / 2) as f64 / 8.);
                    s.variable = (i % 2) as usize;
                    Observation::new(i, s, 0., 0.05 * variance).unwrap()
                })
                .collect();
            let targets: Vec<_> = (0..4)
                .map(|i| {
                    let mut s = site(0.0625 + (i / 2) as f64 * 0.375);
                    s.variable = i % 2;
                    s
                })
                .collect();
            let mut conditioned = workspace
                .condition(&model, &observations, &drift, &cancel)
                .unwrap();
            let plan = conditioned
                .prepare(
                    &targets,
                    WorkspaceOptions {
                        uncertainty: Uncertainty::Joint,
                        tolerance: NumericalTolerance::default(),
                        cancellation: &cancel,
                    },
                )
                .unwrap();
            let weights: Vec<_> = plan
                .weights()
                .col_iter()
                .flat_map(|c| c.iter().copied())
                .collect();
            let covariance: Vec<_> = plan
                .covariance()
                .unwrap()
                .col_iter()
                .flat_map(|c| c.iter().map(|v| v / variance))
                .collect();
            let draws: Vec<_> = plan
                .sample(&[0.; 4], 71, 0, 1)
                .unwrap()
                .into_iter()
                .map(|v| v / root)
                .collect();
            if let Some((expected_w, expected_c, expected_d)) = &reference {
                let tolerance = if columns == 3 { 2e-8 } else { 2e-12 };
                for (actual, expected) in [
                    (&weights, expected_w),
                    (&covariance, expected_c),
                    (&draws, expected_d),
                ] {
                    for (a, b) in actual.iter().zip(expected) {
                        assert!(
                            (a - b).abs() < tolerance,
                            "variance={variance},columns={columns}: {a} vs {b}"
                        );
                    }
                }
            } else {
                reference = Some((weights, covariance, draws));
            }
        }
    }
}

#[test]
fn absolute_tolerance_and_failed_component_preparation_preserve_physical_contracts() {
    let observations: Vec<_> = (0..24)
        .map(|i| Observation::new(i, site(i as f64), 2., 0.).unwrap())
        .collect();
    let budget = MemoryBudget::new(1 << 24);
    let cancel = Cancellation::default();
    let mut workspace = GaussianWorkspace::new(24, 2, 1, &budget).unwrap();
    for variance in [1e-308, 1e308] {
        for absolute in [1.1 * variance, f64::MAX] {
            let plan = workspace
                .prepare(
                    &model(variance),
                    &observations,
                    &[site(0.5), site(1.5)],
                    &Ordinary { variables: 1 },
                    WorkspaceOptions {
                        uncertainty: Uncertainty::Joint,
                        tolerance: NumericalTolerance {
                            absolute,
                            relative: 0.,
                        },
                        cancellation: &cancel,
                    },
                )
                .unwrap();
            for col in plan.covariance().unwrap().col_iter() {
                assert!(col.iter().all(|v| *v == 0.));
            }
            assert_eq!(plan.sample(&[2., 2.], 0, 0, 0).unwrap(), [2., 2.]);
        }
    }
    let tiny = model(1e-308);
    let mut prepared = workspace
        .condition(&tiny, &observations, &Simple, &cancel)
        .unwrap();
    let options = || WorkspaceOptions {
        uncertainty: Uncertainty::Joint,
        tolerance: NumericalTolerance::default(),
        cancellation: &cancel,
    };
    assert!(prepared
        .prepare_component(&model(1e308), &[site(0.)], options())
        .is_err());
    let valid = prepared.prepare(&[site(0.5)], options()).unwrap();
    assert!((valid.covariance().unwrap()[(0, 0)] / 1e-308 - 1.).abs() < 1e-14);
}
