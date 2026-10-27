//! Independent augmented-system elimination and covariance-of-error references.
use rand::RngExt;
use terrustrial_core::{
    data::{Observation, Point3, Site},
    execution::{random_stream, Cancellation, MemoryBudget},
    gaussian::{
        Drift, ExternalDrift, GaussianPlan, GaussianWorkspace, NumericalTolerance, Ordinary,
        PrepareOptions, Simple, Uncertainty, WorkspaceOptions,
    },
    model::{Anisotropy, Coregionalization, Correlation, Covariance, Lmc, Structure},
};
fn solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Vec<f64> {
    let n = b.len();
    for j in 0..n {
        let pivot = (j..n)
            .max_by(|&i, &k| a[i][j].abs().total_cmp(&a[k][j].abs()))
            .unwrap();
        assert!(a[pivot][j].abs() > 1e-14);
        a.swap(j, pivot);
        b.swap(j, pivot);
        for i in j + 1..n {
            let ratio = a[i][j] / a[j][j];
            let (before, after) = a.split_at_mut(i);
            for (value, pivot) in after[0][j..].iter_mut().zip(&before[j][j..]) {
                *value -= ratio * pivot;
            }
            b[i] -= ratio * b[j];
        }
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        x[i] = (b[i] - (i + 1..n).map(|j| a[i][j] * x[j]).sum::<f64>()) / a[i][i];
    }
    x
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
fn joint_ordinary_weights_and_uncertainty_match_independent_augmented_systems() {
    let model = Lmc::new(vec![Structure {
        correlation: Correlation::Exponential,
        anisotropy: Anisotropy::isotropic(2.0).unwrap(),
        coregionalization: Coregionalization::from_factor(2, 2, &[1.4, 0.0, -0.3, 0.9]).unwrap(),
    }])
    .unwrap();
    let budget = MemoryBudget::new(1 << 20);
    let cancel = Cancellation::default();
    for case in 0..25 {
        let mut rng = random_stream(71, case, 0, 0);
        let observations: Vec<_> = (0..8)
            .map(|i| {
                Observation::new(
                    i,
                    Site::point(
                        Point3::new(std::array::from_fn(|_| rng.random_range(-2.0..2.0))).unwrap(),
                        i as usize % 2,
                    ),
                    rng.random_range(-3.0..3.0),
                    0.05,
                )
                .unwrap()
            })
            .collect();
        let targets: Vec<_> = (0..5)
            .map(|i| {
                Site::point(
                    Point3::new(std::array::from_fn(|_| rng.random_range(-2.0..2.0))).unwrap(),
                    i % 2,
                )
            })
            .collect();
        let plan = GaussianPlan::prepare(
            &model,
            &observations,
            &targets,
            &Ordinary { variables: 2 },
            options(&budget, &cancel),
        )
        .unwrap();
        let mut c = vec![vec![0.0; 8]; 8];
        let mut augmented = vec![vec![0.0; 10]; 10];
        for i in 0..8 {
            for j in 0..8 {
                let value = model
                    .covariance(observations[i].site(), observations[j].site())
                    .unwrap()
                    + if i == j { 0.05 } else { 0.0 };
                c[i][j] = value;
                augmented[i][j] = value;
            }
            let variable = observations[i].site().variable;
            augmented[i][8 + variable] = 1.0;
            augmented[8 + variable][i] = 1.0;
        }
        let k: Vec<Vec<f64>> = targets
            .iter()
            .map(|t| {
                observations
                    .iter()
                    .map(|o| model.covariance(o.site(), t).unwrap())
                    .collect()
            })
            .collect();
        let mut weights = Vec::new();
        for (j, target) in targets.iter().enumerate() {
            let mut rhs = k[j].clone();
            rhs.extend([
                if target.variable == 0 { 1.0 } else { 0.0 },
                if target.variable == 1 { 1.0 } else { 0.0 },
            ]);
            let solution = solve(augmented.clone(), rhs);
            for (i, value) in solution.iter().take(8).enumerate() {
                assert!((plan.weights()[(i, j)] - value).abs() < 1e-10);
            }
            weights.push(solution);
        }
        for i in 0..5 {
            for j in 0..5 {
                let mut reference = model.covariance(&targets[i], &targets[j]).unwrap();
                for a in 0..8 {
                    reference -= weights[i][a] * k[j][a] + weights[j][a] * k[i][a];
                    for b in 0..8 {
                        reference += weights[i][a] * c[a][b] * weights[j][b];
                    }
                }
                assert!((plan.covariance().unwrap()[(i, j)] - reference).abs() < 1e-10);
            }
        }
    }
}

#[test]
fn joint_external_drift_reuse_matches_augmented_systems_across_rhs_shapes() {
    let model = Lmc::new(vec![Structure {
        correlation: Correlation::Exponential,
        anisotropy: Anisotropy::isotropic(2.).unwrap(),
        coregionalization: Coregionalization::from_factor(2, 2, &[1.4, 0., -0.3, 0.9]).unwrap(),
    }])
    .unwrap();
    let mut rng = random_stream(914, 0, 0, 0);
    let observations: Vec<_> = (0..48)
        .map(|i| {
            Observation::new(
                i,
                Site::point(
                    Point3::new(std::array::from_fn(|_| rng.random_range(-2.0..2.0))).unwrap(),
                    i as usize % 2,
                ),
                0.,
                0.05,
            )
            .unwrap()
        })
        .collect();
    let targets: Vec<_> = (0..9)
        .map(|i| {
            Site::point(
                Point3::new(std::array::from_fn(|_| rng.random_range(-2.0..2.0))).unwrap(),
                i % 2,
            )
        })
        .collect();
    let budget = MemoryBudget::new(1 << 22);
    let cancel = Cancellation::default();
    let mut workspace = GaussianWorkspace::new(48, 9, 6, &budget).unwrap();
    // Includes more drift columns than targets, both tile boundaries, and
    // shrinking/regrowing regions after a previous, differently shaped solve.
    for (n, m, per_variable) in [
        (48, 9, 3),
        (16, 1, 3),
        (33, 5, 1),
        (32, 4, 2),
        (48, 2, 3),
        (17, 7, 1),
    ] {
        let p = 2 * per_variable;
        let drift = ExternalDrift::new(p, move |site: &Site, basis: &mut [f64]| {
            basis.fill(0.);
            let start = site.variable * per_variable;
            basis[start] = 1.;
            basis[start + 1..start + per_variable]
                .copy_from_slice(&site.support.center().coordinates()[..per_variable - 1]);
            Ok(())
        })
        .unwrap();
        let basis = |site: &Site| {
            let mut basis = vec![0.; p];
            drift.evaluate(site, &mut basis).unwrap();
            basis
        };
        let fd: Vec<_> = observations[..n].iter().map(|o| basis(o.site())).collect();
        let ft: Vec<_> = targets[..m].iter().map(basis).collect();
        let mut augmented = vec![vec![0.; n + p]; n + p];
        for i in 0..n {
            for j in 0..n {
                augmented[i][j] = model
                    .covariance(observations[i].site(), observations[j].site())
                    .unwrap()
                    + if i == j {
                        observations[i].noise_variance()
                    } else {
                        0.
                    };
            }
            for j in 0..p {
                augmented[i][n + j] = fd[i][j];
                augmented[n + j][i] = fd[i][j];
            }
        }
        let k: Vec<Vec<_>> = targets[..m]
            .iter()
            .map(|target| {
                observations[..n]
                    .iter()
                    .map(|o| model.covariance(o.site(), target).unwrap())
                    .collect()
            })
            .collect();
        let solutions: Vec<_> = (0..m)
            .map(|j| {
                let mut rhs = k[j].clone();
                rhs.extend_from_slice(&ft[j]);
                solve(augmented.clone(), rhs)
            })
            .collect();
        let plan = workspace
            .prepare(
                &model,
                &observations[..n],
                &targets[..m],
                &drift,
                WorkspaceOptions {
                    uncertainty: Uncertainty::Joint,
                    tolerance: NumericalTolerance::default(),
                    cancellation: &cancel,
                },
            )
            .unwrap();
        assert_eq!((plan.weights().nrows(), plan.weights().ncols()), (n, m));
        for i in 0..m {
            for (j, expected) in solutions[i][..n].iter().enumerate() {
                assert!((plan.weights()[(j, i)] - expected).abs() < 1e-10);
            }
            for j in 0..m {
                let error = model.covariance(&targets[i], &targets[j]).unwrap()
                    - (0..n).map(|r| k[j][r] * solutions[i][r]).sum::<f64>()
                    - (0..p).map(|r| ft[j][r] * solutions[i][n + r]).sum::<f64>();
                assert!((plan.covariance().unwrap()[(i, j)] - error).abs() < 1e-10);
            }
        }
        let trend = |basis: &Vec<f64>| {
            basis
                .iter()
                .enumerate()
                .map(|(j, x)| x * (0.2 + j as f64))
                .sum::<f64>()
        };
        let values: Vec<_> = fd.iter().map(trend).collect();
        let prediction = plan.estimate(&values).unwrap();
        for (observed, expected) in prediction.iter().zip(ft.iter().map(trend)) {
            assert!((observed - expected).abs() < 1e-10);
        }
    }
}
#[test]
fn gaussian_draw_ensemble_has_joint_means_and_cross_covariance() {
    let model = Lmc::new(vec![Structure {
        correlation: Correlation::Exponential,
        anisotropy: Anisotropy::isotropic(2.0).unwrap(),
        coregionalization: Coregionalization::from_factor(2, 2, &[1.0, 0.0, 0.6, 0.8]).unwrap(),
    }])
    .unwrap();
    let point = Point3::new([0.0; 3]).unwrap();
    let targets = [Site::point(point, 0), Site::point(point, 1)];
    let budget = MemoryBudget::new(1 << 20);
    let cancel = Cancellation::default();
    let plan =
        GaussianPlan::prepare(&model, &[], &targets, &Simple, options(&budget, &cancel)).unwrap();
    let expected_mean = [2.0, -1.0];
    let mut mean = [0.0; 2];
    let mut products = [[0.0; 2]; 2];
    let count = 40000;
    for realization in 0..count {
        let draw = plan.sample(&expected_mean, 918, realization, 0).unwrap();
        let delta = [draw[0] - mean[0], draw[1] - mean[1]];
        for i in 0..2 {
            mean[i] += delta[i] / (realization + 1) as f64;
        }
        for i in 0..2 {
            for j in 0..2 {
                products[i][j] += delta[i] * (draw[j] - mean[j]);
            }
        }
    }
    let covariance = plan.covariance().unwrap();
    for i in 0..2 {
        assert!(
            (mean[i] - expected_mean[i]).abs() < 7.0 * (covariance[(i, i)] / count as f64).sqrt()
        );
        for j in 0..2 {
            let observed = products[i][j] / (count - 1) as f64;
            let stderr = ((covariance[(i, i)] * covariance[(j, j)] + covariance[(i, j)].powi(2))
                / (count - 1) as f64)
                .sqrt();
            assert!((observed - covariance[(i, j)]).abs() < 7.0 * stderr);
        }
    }
}
#[test]
fn coincidence_roundoff_and_nugget_discontinuity_hold_at_extreme_scales() {
    let point = Point3::new([0.0; 3]).unwrap();
    let target = Site::point(point, 0);
    let budget = MemoryBudget::new(1 << 20);
    let cancel = Cancellation::default();
    for variance in [1e-250, 0.13, 1.7, 1e250] {
        let model = Lmc::new(vec![Structure {
            correlation: Correlation::Exponential,
            anisotropy: Anisotropy::isotropic(1.0).unwrap(),
            coregionalization: Coregionalization::scalar(variance).unwrap(),
        }])
        .unwrap();
        let obs = [Observation::new(0, target, 2.0, 0.0).unwrap()];
        let plan = GaussianPlan::prepare(
            &model,
            &obs,
            std::slice::from_ref(&target),
            &Simple,
            options(&budget, &cancel),
        )
        .unwrap();
        assert!((plan.estimate(&[2.0]).unwrap()[0] - 2.0).abs() < 1e-12);
        assert_eq!(plan.covariance().unwrap()[(0, 0)], 0.0);
    }
    let nugget = Lmc::new(vec![Structure {
        correlation: Correlation::Nugget,
        anisotropy: Anisotropy::isotropic(1e200).unwrap(),
        coregionalization: Coregionalization::scalar(1.0).unwrap(),
    }])
    .unwrap();
    assert_eq!(
        nugget
            .point_covariance(point, 0, Point3::new([1e-200, 0.0, 0.0]).unwrap(), 0)
            .unwrap(),
        0.0
    );
    assert_eq!(
        Anisotropy::isotropic(1e308).unwrap().normalized_distance(
            Point3::new([1e308, 0.0, 0.0]).unwrap(),
            Point3::new([-1e308, 0.0, 0.0]).unwrap()
        ),
        2.0
    );
}

#[test]
fn larger_joint_variable_systems_match_independent_augmented_elimination() {
    let budget = MemoryBudget::new(1 << 27);
    let cancel = Cancellation::default();
    let mut workspace = GaussianWorkspace::new(128, 33, 16, &budget).unwrap();
    // Cross drift and covariance product boundaries, then shrink and regrow the arena.
    for (n, m, v) in [
        (64, 7, 4),
        (64, 8, 4),
        (64, 9, 4),
        (63, 17, 1),
        (64, 15, 4),
        (64, 16, 4),
        (65, 17, 8),
        (128, 33, 16),
        (64, 16, 1),
    ] {
        let model = Lmc::new(
            (0..2)
                .map(|s| {
                    let f: Vec<_> = (0..v)
                        .flat_map(|i| {
                            (0..v).map(move |j| {
                                if i == j {
                                    1. + 0.2 * s as f64
                                } else {
                                    0.15 * ((i * 17 + j * 7 + s) as f64).sin()
                                }
                            })
                        })
                        .collect();
                    Structure {
                        correlation: if s == 0 {
                            Correlation::Exponential
                        } else {
                            Correlation::Spherical
                        },
                        anisotropy: Anisotropy::isotropic(2. + s as f64).unwrap(),
                        coregionalization: Coregionalization::from_factor(v, v, &f).unwrap(),
                    }
                })
                .collect(),
        )
        .unwrap();
        let mut rng = random_stream(3021, n as u64, v as u64, 0);
        let observations: Vec<_> = (0..n)
            .map(|i| {
                Observation::new(
                    i as u64,
                    Site::point(
                        Point3::new(std::array::from_fn(|_| rng.random_range(-2.0..2.0))).unwrap(),
                        i % v,
                    ),
                    (i % v) as f64 + 0.5,
                    0.02,
                )
                .unwrap()
            })
            .collect();
        let mut targets: Vec<_> = (0..m)
            .map(|i| {
                Site::point(
                    Point3::new(std::array::from_fn(|_| rng.random_range(-2.0..2.0))).unwrap(),
                    i % v,
                )
            })
            .collect();
        targets[m - 1] = targets[0]; // A singular conditional covariance is valid.
        let mut augmented = vec![vec![0.; n + v]; n + v];
        for i in 0..n {
            for j in 0..n {
                augmented[i][j] = model
                    .covariance(observations[i].site(), observations[j].site())
                    .unwrap()
                    + if i == j { 0.02 } else { 0. };
            }
            augmented[i][n + i % v] = 1.;
            augmented[n + i % v][i] = 1.;
        }
        let k: Vec<Vec<_>> = targets
            .iter()
            .map(|t| {
                observations
                    .iter()
                    .map(|o| model.covariance(o.site(), t).unwrap())
                    .collect()
            })
            .collect();
        let solutions: Vec<_> = targets
            .iter()
            .enumerate()
            .map(|(j, t)| {
                let mut rhs = k[j].clone();
                rhs.extend((0..v).map(|i| if t.variable == i { 1. } else { 0. }));
                solve(augmented.clone(), rhs)
            })
            .collect();
        let plan = workspace
            .prepare(
                &model,
                &observations,
                &targets,
                &Ordinary { variables: v },
                WorkspaceOptions {
                    uncertainty: Uncertainty::Joint,
                    tolerance: NumericalTolerance::default(),
                    cancellation: &cancel,
                },
            )
            .unwrap();
        for i in 0..m {
            for (j, expected) in solutions[i][..n].iter().enumerate() {
                assert!(
                    (plan.weights()[(j, i)] - expected).abs() < 2e-10,
                    "weights n={n} m={m} v={v}"
                );
            }
            for j in 0..m {
                let expected = model.covariance(&targets[i], &targets[j]).unwrap()
                    - (0..n).map(|r| k[j][r] * solutions[i][r]).sum::<f64>()
                    - solutions[i][n + targets[j].variable];
                assert!(
                    (plan.covariance().unwrap()[(i, j)] - expected).abs() < 2e-10,
                    "covariance n={n} m={m} v={v} i={i} j={j}"
                );
                assert_eq!(
                    plan.covariance().unwrap()[(i, j)].to_bits(),
                    plan.covariance().unwrap()[(j, i)].to_bits()
                );
            }
        }
        let prediction = plan
            .estimate(
                &observations
                    .iter()
                    .map(Observation::value)
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        for (t, value) in targets.iter().zip(&prediction) {
            assert!((value - (t.variable as f64 + 0.5)).abs() < 2e-10);
        }
        let draw = plan.sample(&prediction, 19, 2, 3).unwrap();
        assert!((draw[0] - draw[m - 1]).abs() < 2e-10);
    }
}

#[test]
fn dense_external_drift_joint_covariance_matches_augmented_system_reference() {
    let (n, m, p) = (128, 33, 6);
    let basis = |site: &Site| {
        let [x, y, z] = site.support.center().coordinates();
        [
            (site.variable == 0) as u8 as f64,
            (site.variable == 1) as u8 as f64,
            x,
            y,
            z,
            x * y,
        ]
    };
    let drift = ExternalDrift::new(p, |site: &Site, out: &mut [f64]| {
        out.copy_from_slice(&basis(site));
        Ok(())
    })
    .unwrap();
    let model = Lmc::new(vec![Structure {
        correlation: Correlation::Exponential,
        anisotropy: Anisotropy::isotropic(2.).unwrap(),
        coregionalization: Coregionalization::from_factor(2, 2, &[1., 0., 0.4, 0.9]).unwrap(),
    }])
    .unwrap();
    let mut rng = random_stream(913, 0, 0, 0);
    let observations: Vec<_> = (0..n)
        .map(|i| {
            Observation::new(
                i as u64,
                Site::point(
                    Point3::new(std::array::from_fn(|_| rng.random_range(-2.0..2.0))).unwrap(),
                    i % 2,
                ),
                (i as f64 * 0.31).sin(),
                0.02,
            )
            .unwrap()
        })
        .collect();
    let mut targets: Vec<_> = (0..m)
        .map(|i| {
            Site::point(
                Point3::new(std::array::from_fn(|_| rng.random_range(-2.0..2.0))).unwrap(),
                i % 2,
            )
        })
        .collect();
    targets[m - 1] = targets[0];
    let mut augmented = vec![vec![0.; n + p]; n + p];
    for i in 0..n {
        for j in 0..n {
            augmented[i][j] = model
                .covariance(observations[i].site(), observations[j].site())
                .unwrap()
                + if i == j { 0.02 } else { 0. };
        }
        for (j, value) in basis(observations[i].site()).iter().enumerate() {
            augmented[i][n + j] = *value;
            augmented[n + j][i] = *value;
        }
    }
    let k: Vec<Vec<_>> = targets
        .iter()
        .map(|t| {
            observations
                .iter()
                .map(|o| model.covariance(o.site(), t).unwrap())
                .collect()
        })
        .collect();
    let solutions: Vec<_> = targets
        .iter()
        .enumerate()
        .map(|(j, target)| {
            let mut rhs = k[j].clone();
            rhs.extend(basis(target));
            solve(augmented.clone(), rhs)
        })
        .collect();
    let budget = MemoryBudget::new(1 << 25);
    let cancel = Cancellation::default();
    let plan = GaussianPlan::prepare(
        &model,
        &observations,
        &targets,
        &drift,
        options(&budget, &cancel),
    )
    .unwrap();
    let close = |actual: f64, expected: f64| {
        assert!(
            (actual - expected).abs() <= 1e-10 * (1. + expected.abs()),
            "{actual} != {expected}"
        );
    };
    let values: Vec<_> = observations.iter().map(Observation::value).collect();
    let predicted = plan.estimate(&values).unwrap();
    for i in 0..m {
        for (j, &expected) in solutions[i][..n].iter().enumerate() {
            close(plan.weights()[(j, i)], expected);
        }
        close(
            predicted[i],
            (0..n).map(|j| solutions[i][j] * values[j]).sum(),
        );
        for j in 0..m {
            let expected = model.covariance(&targets[i], &targets[j]).unwrap()
                - (0..n).map(|r| k[j][r] * solutions[i][r]).sum::<f64>()
                - basis(&targets[j])
                    .iter()
                    .zip(&solutions[i][n..])
                    .map(|(f, lambda)| f * lambda)
                    .sum::<f64>();
            close(plan.covariance().unwrap()[(i, j)], expected);
            assert_eq!(
                plan.covariance().unwrap()[(i, j)].to_bits(),
                plan.covariance().unwrap()[(j, i)].to_bits()
            );
        }
    }
    let draw = plan.sample(&predicted, 913, 0, 0).unwrap();
    close(draw[0], draw[m - 1]);
}

#[test]
fn large_joint_covariances_match_pairwise_preparation_across_variance_scales() {
    let budget = MemoryBudget::new(1 << 27);
    let cancel = Cancellation::default();
    for scale in [1e-200, 1., 1e200] {
        let model = Lmc::new(vec![Structure {
            correlation: Correlation::Exponential,
            anisotropy: Anisotropy::isotropic(2.).unwrap(),
            coregionalization: Coregionalization::scalar(scale).unwrap(),
        }])
        .unwrap();
        let observations: Vec<_> = (0..64)
            .map(|i| {
                Observation::new(
                    i,
                    Site::point(
                        Point3::new([(i % 8) as f64, (i / 8) as f64, 0.]).unwrap(),
                        0,
                    ),
                    0.,
                    0.01 * scale,
                )
                .unwrap()
            })
            .collect();
        let targets: Vec<_> = (0..17)
            .map(|i| {
                Site::point(
                    Point3::new([0.2 + (i % 5) as f64, 0.3 + (i / 5) as f64, 0.]).unwrap(),
                    0,
                )
            })
            .collect();
        let large = GaussianPlan::prepare(
            &model,
            &observations,
            &targets,
            &Simple,
            options(&budget, &cancel),
        )
        .unwrap();
        for i in 0..targets.len() {
            for j in 0..=i {
                let pair = GaussianPlan::prepare(
                    &model,
                    &observations,
                    &[targets[i], targets[j]],
                    &Simple,
                    options(&budget, &cancel),
                )
                .unwrap();
                let error = (large.covariance().unwrap()[(i, j)] / scale
                    - pair.covariance().unwrap()[(0, 1)] / scale)
                    .abs();
                assert!(error < 2e-12, "scale={scale} i={i} j={j} error={error}");
            }
        }
    }
}
