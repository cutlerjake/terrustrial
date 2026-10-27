//! Basis units must not change the represented unbiasedness constraints.
use terrustrial_core::{
    data::{Observation, Point3, Site},
    execution::{Cancellation, MemoryBudget},
    gaussian::{
        Drift, ExternalDrift, GaussianPlan, GaussianWorkspace, LinearDrift, NumericalTolerance,
        Ordinary, PrepareOptions, Uncertainty, WorkspaceOptions,
    },
    model::{Anisotropy, Coregionalization, Correlation, Lmc, Structure},
};

#[test]
fn near_dependent_drift_preserves_exact_joint_targets() {
    let model = Lmc::new(vec![Structure {
        correlation: Correlation::Exponential,
        anisotropy: Anisotropy::isotropic(2.).unwrap(),
        coregionalization: Coregionalization::from_factor(
            3,
            3,
            &[1., 0., 0., 0.6, 0.8, 0., -0.2, 0.3, 0.9],
        )
        .unwrap(),
    }])
    .unwrap();
    let site = |x, v| Site::point(Point3::new([x, 0., 0.]).unwrap(), v);
    let mut observations = Vec::new();
    for (v, xs) in [[-1.5, 1.75, 5.], [-1.2, 2.1, 4.7], [-0.9, 2.45, 4.4]]
        .into_iter()
        .enumerate()
    {
        for x in xs {
            observations
                .push(Observation::new(observations.len() as u64, site(x, v), x, 0.).unwrap());
        }
    }
    let mut targets: Vec<_> = observations.iter().map(|o| *o.site()).collect();
    targets.extend((0..3).map(|v| site(0.5, v)));
    let budget = MemoryBudget::new(1 << 24);
    let cancel = Cancellation::default();
    let prepare = |epsilon: f64| {
        let drift = ExternalDrift::new(6, move |s: &Site, out: &mut [f64]| {
            out.fill(0.);
            out[2 * s.variable] = 1.;
            out[2 * s.variable + 1] = 1. + epsilon * s.support.center().coordinates()[0];
            Ok(())
        })
        .unwrap();
        GaussianPlan::prepare(
            &model,
            &observations,
            &targets,
            &drift,
            PrepareOptions {
                uncertainty: Uncertainty::Joint,
                tolerance: NumericalTolerance::default(),
                budget: &budget,
                cancellation: &cancel,
            },
        )
        .unwrap()
    };
    let reference = prepare(1.);
    let actual = prepare(2_f64.powi(-20));
    for j in 0..targets.len() {
        for i in 0..observations.len() {
            assert!((actual.weights()[(i, j)] - reference.weights()[(i, j)]).abs() < 2e-8);
            if j < observations.len() {
                assert!((actual.weights()[(i, j)] - f64::from(i == j)).abs() < 2e-12);
            }
        }
        for i in 0..targets.len() {
            assert!(
                (actual.covariance().unwrap()[(i, j)] - reference.covariance().unwrap()[(i, j)])
                    .abs()
                    < 2e-8
            );
            if j < observations.len() {
                assert!(actual.covariance().unwrap()[(i, j)].abs() < 2e-12);
            }
        }
    }
}

#[test]
fn independent_drift_column_units_preserve_joint_ordinary_predictions() {
    let model = Lmc::new(vec![Structure {
        correlation: Correlation::Exponential,
        anisotropy: Anisotropy::isotropic(3.).unwrap(),
        coregionalization: Coregionalization::from_factor(2, 2, &[1., 0., 0.6, 0.8]).unwrap(),
    }])
    .unwrap();
    let site = |x, v| Site::point(Point3::new([x, 0., 0.]).unwrap(), v);
    let observations: Vec<_> = (0..24)
        .map(|i| {
            Observation::new(
                i,
                site((i / 2) as f64 * 0.2, (i % 2) as usize),
                (i as f64).sin(),
                0.03,
            )
            .unwrap()
        })
        .collect();
    let targets = [site(0.13, 0), site(0.13, 1), site(1.23, 0), site(1.23, 1)];
    let budget = MemoryBudget::new(1 << 24);
    let cancel = Cancellation::default();
    let options = || PrepareOptions {
        uncertainty: Uncertainty::Joint,
        tolerance: NumericalTolerance::default(),
        budget: &budget,
        cancellation: &cancel,
    };
    let expected = GaussianPlan::prepare(
        &model,
        &observations,
        &targets,
        &Ordinary { variables: 2 },
        options(),
    )
    .unwrap();
    let mut workspace = GaussianWorkspace::new(24, 4, 2, &budget).unwrap();
    let allocations = allocation_counter::measure(|| {
        for scales in [
            [1e200, 1e-200],
            [1e-200, 1e200],
            [-1e200, 1e-200],
            [f64::MIN_POSITIVE, f64::MAX],
            [f64::from_bits(1), 1.],
        ] {
            let drift = ExternalDrift::new(2, |s: &Site, out: &mut [f64]| {
                out.fill(0.);
                out[s.variable] = scales[s.variable];
                Ok(())
            })
            .unwrap();
            let actual = workspace
                .prepare(
                    &model,
                    &observations,
                    &targets,
                    &drift,
                    WorkspaceOptions {
                        uncertainty: Uncertainty::Joint,
                        tolerance: NumericalTolerance::default(),
                        cancellation: &cancel,
                    },
                )
                .unwrap();
            for j in 0..4 {
                for i in 0..24 {
                    assert!(
                        (actual.weights()[(i, j)] - expected.weights()[(i, j)]).abs() < 2e-12,
                        "scales={scales:?}"
                    );
                }
                for i in 0..4 {
                    assert!(
                        (actual.covariance().unwrap()[(i, j)]
                            - expected.covariance().unwrap()[(i, j)])
                            .abs()
                            < 2e-12,
                        "scales={scales:?}"
                    );
                }
            }
        }
    });
    assert_eq!(allocations.count_total, 0);
}

#[test]
fn linear_drift_avoids_overflow_before_representable_scaling() {
    let drift = LinearDrift::new(1, [-1e308; 3], 1e308).unwrap();
    let mut values = [0.; 4];
    drift
        .evaluate(
            &Site::point(Point3::new([1e308; 3]).unwrap(), 0),
            &mut values,
        )
        .unwrap();
    assert_eq!(values, [1., 2., 2., 2.]);
    let model = Lmc::new(vec![Structure {
        correlation: Correlation::Exponential,
        anisotropy: Anisotropy::isotropic(1e308).unwrap(),
        coregionalization: Coregionalization::scalar(1.).unwrap(),
    }])
    .unwrap();
    let observations: Vec<_> = (0..8)
        .map(|i| {
            let coordinates =
                std::array::from_fn(|axis| if i & (1 << axis) == 0 { -1e308 } else { 1e308 });
            let site = Site::point(Point3::new(coordinates).unwrap(), 0);
            drift.evaluate(&site, &mut values).unwrap();
            Observation::new(
                i,
                site,
                1. + 2. * values[1] + 3. * values[2] - values[3],
                0.,
            )
            .unwrap()
        })
        .collect();
    let budget = MemoryBudget::new(1 << 20);
    let cancel = Cancellation::default();
    let plan = GaussianPlan::prepare(
        &model,
        &observations,
        &[Site::point(Point3::new([0.; 3]).unwrap(), 0)],
        &drift,
        PrepareOptions {
            uncertainty: Uncertainty::Joint,
            tolerance: NumericalTolerance::default(),
            budget: &budget,
            cancellation: &cancel,
        },
    )
    .unwrap();
    let prediction = plan
        .estimate(
            &observations
                .iter()
                .map(Observation::value)
                .collect::<Vec<_>>(),
        )
        .unwrap();
    assert!((prediction[0] - 5.).abs() < 2e-12);
}

#[test]
fn nonconstant_external_drift_is_invariant_to_column_units() {
    let model = Lmc::new(vec![Structure {
        correlation: Correlation::Exponential,
        anisotropy: Anisotropy::isotropic(2.).unwrap(),
        coregionalization: Coregionalization::scalar(1.).unwrap(),
    }])
    .unwrap();
    let site = |x| Site::point(Point3::new([x, 0., 0.]).unwrap(), 0);
    let observations: Vec<_> = (0..16)
        .map(|i| {
            let x = i as f64 * 0.1;
            Observation::new(i, site(x), 2. + 3. * x - 0.7 * x * x, 0.01).unwrap()
        })
        .collect();
    let targets = [site(0.13), site(0.73), site(1.23)];
    let budget = MemoryBudget::new(1 << 24);
    let cancel = Cancellation::default();
    let mut reference: Option<Vec<f64>> = None;
    for scales in [[1.; 3], [1e200, -1e-200, 1e100], [1e-200, 1e200, -1e-100]] {
        let drift = ExternalDrift::new(3, |s: &Site, out: &mut [f64]| {
            let x = s.support.center().coordinates()[0];
            out.copy_from_slice(&[scales[0], scales[1] * x, scales[2] * x * x]);
            Ok(())
        })
        .unwrap();
        let plan = GaussianPlan::prepare(
            &model,
            &observations,
            &targets,
            &drift,
            PrepareOptions {
                uncertainty: Uncertainty::Joint,
                tolerance: NumericalTolerance::default(),
                budget: &budget,
                cancellation: &cancel,
            },
        )
        .unwrap();
        let values = plan
            .estimate(
                &observations
                    .iter()
                    .map(Observation::value)
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let mut result = Vec::new();
        for (s, &value) in targets.iter().zip(&values) {
            let x = s.support.center().coordinates()[0];
            assert!((value - (2. + 3. * x - 0.7 * x * x)).abs() < 2e-11);
        }
        for col in plan.weights().col_iter() {
            result.extend(col.iter().copied());
        }
        for col in plan.covariance().unwrap().col_iter() {
            result.extend(col.iter().copied());
        }
        if let Some(reference) = &reference {
            for (a, b) in result.iter().zip(reference) {
                assert!((a - b).abs() < 2e-11);
            }
        } else {
            reference = Some(result);
        }
    }
}

#[test]
fn normalization_errors_recover_without_losing_prepared_conditioning() {
    use terrustrial_core::Error;
    let model = Lmc::new(vec![Structure {
        correlation: Correlation::Exponential,
        anisotropy: Anisotropy::isotropic(2.).unwrap(),
        coregionalization: Coregionalization::scalar(1.).unwrap(),
    }])
    .unwrap();
    let site = |x| Site::point(Point3::new([x, 0., 0.]).unwrap(), 0);
    let observations = [
        Observation::new(0, site(0.), 2., 0.01).unwrap(),
        Observation::new(1, site(1.), 2., 0.01).unwrap(),
    ];
    let budget = MemoryBudget::new(1 << 20);
    let cancel = Cancellation::default();
    let options = || WorkspaceOptions {
        uncertainty: Uncertainty::Joint,
        tolerance: NumericalTolerance::default(),
        cancellation: &cancel,
    };
    let mut workspace = GaussianWorkspace::new(2, 1, 1, &budget).unwrap();
    for (data_scale, target_scale) in [(f64::MAX, f64::from_bits(1)), (f64::from_bits(1), f64::MAX)]
    {
        let drift = ExternalDrift::new(1, |s: &Site, out: &mut [f64]| {
            out[0] = if s.support.center().coordinates()[0] > 5. {
                target_scale
            } else {
                data_scale
            };
            Ok(())
        })
        .unwrap();
        let mut prepared = workspace
            .condition(&model, &observations, &drift, &cancel)
            .unwrap();
        assert!(matches!(
            prepared.prepare(&[site(9.)], options()),
            Err(Error::Numerical { .. })
        ));
        let plan = prepared.prepare(&[site(0.5)], options()).unwrap();
        assert!((plan.estimate(&[2., 2.]).unwrap()[0] - 2.).abs() < 1e-12);
    }
    let impossible = ExternalDrift::new(1, |s: &Site, out: &mut [f64]| {
        out[0] = if s.support.center().coordinates()[0] == 0. {
            f64::MAX
        } else {
            f64::from_bits(1)
        };
        Ok(())
    })
    .unwrap();
    assert!(matches!(
        workspace.condition(&model, &observations, &impossible, &cancel),
        Err(Error::Numerical { .. })
    ));
    let zero = ExternalDrift::new(1, |_: &Site, out: &mut [f64]| {
        out[0] = 0.;
        Ok(())
    })
    .unwrap();
    assert!(matches!(
        workspace.condition(&model, &observations, &zero, &cancel),
        Err(Error::Singular { .. })
    ));
    workspace
        .prepare(
            &model,
            &observations,
            &[site(0.5)],
            &Ordinary { variables: 1 },
            options(),
        )
        .unwrap();
    drop(workspace);
    assert_eq!(budget.used(), 0);
}

#[test]
fn near_collinear_drift_matches_analytic_linear_prediction() {
    // Independent noise makes the solution ordinary least squares. Symmetric
    // abscissae give closed-form weights and covariance without another solver.
    let model = Lmc::new(vec![Structure {
        correlation: Correlation::Nugget,
        anisotropy: Anisotropy::isotropic(1.).unwrap(),
        coregionalization: Coregionalization::scalar(1.).unwrap(),
    }])
    .unwrap();
    let site = |x| Site::point(Point3::new([x, 0., 0.]).unwrap(), 0);
    let observations: Vec<_> = (0..25)
        .map(|i| {
            let x = (i as f64 - 12.) / 8.;
            Observation::new(i, site(x), 3. + 2. * x, 0.).unwrap()
        })
        .collect();
    let targets = [site(1. / 16.), site(11. / 16.)];
    let xx: f64 = observations
        .iter()
        .map(|o| o.site().support.center().coordinates()[0].powi(2))
        .sum();
    let budget = MemoryBudget::new(1 << 24);
    let cancel = Cancellation::default();
    let mut workspace = GaussianWorkspace::new(25, 2, 2, &budget).unwrap();
    for exponent in [16, 24, 28] {
        let epsilon = 2_f64.powi(-exponent);
        let tolerance = 32. * f64::EPSILON / epsilon;
        for reverse in [false, true] {
            let drift = ExternalDrift::new(2, |s: &Site, out: &mut [f64]| {
                let x = s.support.center().coordinates()[0];
                out.copy_from_slice(&[1., 1. + epsilon * x]);
                if reverse {
                    out.swap(0, 1);
                }
                Ok(())
            })
            .unwrap();
            let allocations = allocation_counter::measure(|| {
                let plan = workspace
                    .prepare(
                        &model,
                        &observations,
                        &targets,
                        &drift,
                        WorkspaceOptions {
                            uncertainty: Uncertainty::Joint,
                            tolerance: NumericalTolerance::default(),
                            cancellation: &cancel,
                        },
                    )
                    .unwrap_or_else(|error| {
                        panic!("exponent={exponent}, reverse={reverse}: {error:?}")
                    });
                for (j, target) in targets.iter().enumerate() {
                    let tx = target.support.center().coordinates()[0];
                    for (i, observation) in observations.iter().enumerate() {
                        let x = observation.site().support.center().coordinates()[0];
                        let expected = 1. / 25. + x * tx / xx;
                        assert!((plan.weights()[(i, j)] - expected).abs() < tolerance,
                            "weight exponent={exponent}, reverse={reverse}, i={i}, j={j}: {} vs {expected}",plan.weights()[(i,j)]);
                    }
                    for (i, other) in targets.iter().enumerate() {
                        let ox = other.support.center().coordinates()[0];
                        let expected = if i == j { 1. } else { 0. } + 1. / 25. + tx * ox / xx;
                        assert!(
                            (plan.covariance().unwrap()[(i, j)] - expected).abs() < tolerance,
                            "covariance exponent={exponent}, reverse={reverse}"
                        );
                    }
                }
            });
            assert_eq!(allocations.count_total, 0);
        }
    }
}

#[test]
fn pivoted_drift_preserves_correlated_joint_predictions_and_workspace_reuse() {
    let model = Lmc::new(vec![Structure {
        correlation: Correlation::Exponential,
        anisotropy: Anisotropy::isotropic(2.).unwrap(),
        coregionalization: Coregionalization::from_factor(2, 2, &[1., 0., 0.6, 0.8]).unwrap(),
    }])
    .unwrap();
    let site = |x, variable| Site::point(Point3::new([x, 0., 0.]).unwrap(), variable);
    let observations: Vec<_> = (0..48)
        .map(|i| {
            Observation::new(
                i,
                site((i / 2) as f64 / 8. - 1.5, (i % 2) as usize),
                (i as f64).sin(),
                0.03,
            )
            .unwrap()
        })
        .collect();
    let stable = ExternalDrift::new(4, |s: &Site, out: &mut [f64]| {
        out.fill(0.);
        out[2 * s.variable] = 1.;
        out[2 * s.variable + 1] = s.support.center().coordinates()[0];
        Ok(())
    })
    .unwrap();
    let epsilon = 2_f64.powi(-28);
    let nearly_dependent = ExternalDrift::new(4, |s: &Site, out: &mut [f64]| {
        let mut values = [0.; 4];
        values[2 * s.variable] = 1.;
        values[2 * s.variable + 1] = 1. + epsilon * s.support.center().coordinates()[0];
        for (i, (column, scale)) in [3, 0, 2, 1]
            .into_iter()
            .zip([1e150, -1e-150, 1e100, -1e100])
            .enumerate()
        {
            out[i] = values[column] * scale;
        }
        Ok(())
    })
    .unwrap();
    let budget = MemoryBudget::new(1 << 26);
    let cancel = Cancellation::default();
    let options = |uncertainty| WorkspaceOptions {
        uncertainty,
        tolerance: NumericalTolerance::default(),
        cancellation: &cancel,
    };
    let mut reference = GaussianWorkspace::new(48, 4, 4, &budget).unwrap();
    let mut workspace = GaussianWorkspace::new(48, 4, 4, &budget).unwrap();
    {
        let mut prepared = workspace
            .condition(&model, &observations, &nearly_dependent, &cancel)
            .unwrap();
        assert!(prepared
            .prepare(&[site(0., 2)], options(Uncertainty::Joint))
            .is_err());
        for x in [0.125, 0.625] {
            let targets = [site(x, 0), site(x, 1), site(x + 0.25, 0), site(x + 0.25, 1)];
            for uncertainty in [Uncertainty::None, Uncertainty::Joint] {
                let expected = reference
                    .prepare(
                        &model,
                        &observations,
                        &targets,
                        &stable,
                        options(uncertainty),
                    )
                    .unwrap();
                let allocations = allocation_counter::measure(|| {
                    let actual = prepared.prepare(&targets, options(uncertainty)).unwrap();
                    for j in 0..4 {
                        for i in 0..48 {
                            assert!(
                                (actual.weights()[(i, j)] - expected.weights()[(i, j)]).abs()
                                    < 3e-6
                            );
                        }
                        if uncertainty == Uncertainty::Joint {
                            for i in 0..4 {
                                assert!(
                                    (actual.covariance().unwrap()[(i, j)]
                                        - expected.covariance().unwrap()[(i, j)])
                                        .abs()
                                        < 3e-6
                                );
                            }
                        }
                    }
                });
                assert_eq!(allocations.count_total, 0);
            }
        }
    }
    // Resetting conditioning must clear the transform even when the drift width
    // shrinks, including the zero-column simple-kriging case.
    for columns in [2, 0, 4] {
        let drift = ExternalDrift::new(columns.max(1), |s: &Site, out: &mut [f64]| {
            out.fill(0.);
            if columns == 4 {
                stable.evaluate(s, out)
            } else {
                out[s.variable] = 1.;
                Ok(())
            }
        })
        .unwrap();
        fn compare<'a, D: Drift>(
            model: &Lmc,
            observations: &[Observation],
            targets: &[Site],
            drift: &D,
            workspace: &mut GaussianWorkspace,
            reference: &mut GaussianWorkspace,
            options: impl Fn() -> WorkspaceOptions<'a>,
        ) {
            let expected = reference
                .prepare(model, observations, targets, drift, options())
                .unwrap();
            let actual = workspace
                .prepare(model, observations, targets, drift, options())
                .unwrap();
            for j in 0..targets.len() {
                for i in 0..observations.len() {
                    assert_eq!(
                        actual.weights()[(i, j)].to_bits(),
                        expected.weights()[(i, j)].to_bits()
                    );
                }
            }
        }
        if columns == 0 {
            compare(
                &model,
                &observations,
                &[site(0.2, 0)],
                &terrustrial_core::gaussian::Simple,
                &mut workspace,
                &mut reference,
                || options(Uncertainty::Joint),
            );
        } else {
            compare(
                &model,
                &observations,
                &[site(0.2, 0)],
                &drift,
                &mut workspace,
                &mut reference,
                || options(Uncertainty::Joint),
            );
        }
    }
    drop(workspace);
    drop(reference);
    assert_eq!(budget.used(), 0);
}

#[test]
fn numerically_rank_deficient_drift_is_rejected_without_regularization() {
    use terrustrial_core::Error;
    let model = Lmc::new(vec![Structure {
        correlation: Correlation::Nugget,
        anisotropy: Anisotropy::isotropic(1.).unwrap(),
        coregionalization: Coregionalization::scalar(1.).unwrap(),
    }])
    .unwrap();
    let site = |x| Site::point(Point3::new([x, 0., 0.]).unwrap(), 0);
    let observations: Vec<_> = (0..25)
        .map(|i| Observation::new(i, site((i as f64 - 12.) / 8.), 1., 0.).unwrap())
        .collect();
    let budget = MemoryBudget::new(1 << 24);
    let cancel = Cancellation::default();
    let mut workspace = GaussianWorkspace::new(25, 1, 3, &budget).unwrap();
    for epsilon in [0., 2_f64.powi(-50)] {
        let drift = ExternalDrift::new(2, |s: &Site, out: &mut [f64]| {
            out.copy_from_slice(&[1., 1. + epsilon * s.support.center().coordinates()[0]]);
            Ok(())
        })
        .unwrap();
        assert!(matches!(
            workspace.condition(&model, &observations, &drift, &cancel),
            Err(Error::Singular { .. })
        ));
    }
    let redundant = ExternalDrift::new(3, |s: &Site, out: &mut [f64]| {
        let x = s.support.center().coordinates()[0];
        out.copy_from_slice(&[1., x, 1. + 2. * x]);
        Ok(())
    })
    .unwrap();
    assert!(matches!(
        workspace.condition(&model, &observations, &redundant, &cancel),
        Err(Error::Singular { .. })
    ));
    workspace
        .prepare(
            &model,
            &observations,
            &[site(0.0625)],
            &Ordinary { variables: 1 },
            WorkspaceOptions {
                uncertainty: Uncertainty::Joint,
                tolerance: NumericalTolerance::default(),
                cancellation: &cancel,
            },
        )
        .unwrap();
}
