use terrustrial_core::{
    data::{Group, Observation, Point3, Site, Support, Target},
    execution::{Cancellation, MemoryBudget},
    gaussian::{Drift, ExternalDrift, KnownMean, Ordinary, Simple},
    model::{Anisotropy, Coregionalization, Correlation, Lmc, Structure},
    simulation::{simulate_groups, SimulationOptions},
    spatial::{Neighborhood, SpatialIndex},
    Error, Result,
};

fn site(x: f64, variable: usize) -> Site {
    Site::point(Point3::new([x, 0.0, 0.0]).unwrap(), variable)
}
fn model() -> Lmc {
    Lmc::new(vec![Structure {
        correlation: Correlation::Exponential,
        anisotropy: Anisotropy::isotropic(2.0).unwrap(),
        coregionalization: Coregionalization::from_factor(2, 2, &[1.0, 0.0, 0.6, 0.8]).unwrap(),
    }])
    .unwrap()
}
fn trend(site: &Site) -> Result<f64> {
    // Mean of x^2 over supplied midpoint support. Deliberately nonlinear so
    // using only the center of a block would give the wrong result.
    Ok(site
        .support
        .nodes()
        .map(|p| {
            let x = p.coordinates()[0];
            if site.variable == 0 {
                2.0 + x * x
            } else {
                -3.0 + 0.5 * x
            }
        })
        .sum::<f64>()
        / site.support.node_count() as f64)
}
fn options<'a>(
    mean: Option<&'a dyn KnownMean>,
    realization: u64,
    budget: &'a MemoryBudget,
    cancel: &'a Cancellation,
) -> SimulationOptions<'a> {
    SimulationOptions {
        seed: 2026,
        realization,
        mean,
        hard_neighborhood: Neighborhood {
            maximum: 32,
            radius: 100.0,
        }
        .into(),
        simulated_neighborhood: Neighborhood {
            maximum: 32,
            radius: 100.0,
        }
        .into(),
        budget,
        cancellation: cancel,
    }
}
fn path() -> Vec<Group> {
    (0..3)
        .map(|i| Group {
            id: i,
            targets: (0..2)
                .map(|v| Target {
                    id: 100 + 2 * i + v as u64,
                    site: if i == 0 {
                        Site {
                            support: Support::block(
                                Point3::new([0.0; 3]).unwrap(),
                                [0.5, 0.0, 0.0],
                                [4, 1, 1],
                            )
                            .unwrap(),
                            variable: v,
                        }
                    } else {
                        site(i as f64, v)
                    },
                })
                .collect(),
        })
        .collect()
}

#[test]
fn support_aware_trends_equal_residual_simulation_for_simple_ordinary_and_external_drift() {
    fn compare(drift: &impl Drift) {
        let run = |shifted, workers| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(workers)
                .build()
                .unwrap()
                .install(|| {
                    let budget = MemoryBudget::new(1 << 20);
                    let cancel = Cancellation::default();
                    let mut hard = Vec::new();
                    for (i, x) in [-2.0, 4.0].into_iter().enumerate() {
                        for v in 0..2 {
                            let s = site(x, v);
                            let residual = 0.25 + i as f64 * 0.5 - v as f64 * 0.75;
                            hard.push(
                                Observation::new(
                                    (2 * i + v) as u64,
                                    s,
                                    residual + if shifted { trend(&s).unwrap() } else { 0.0 },
                                    0.0,
                                )
                                .unwrap(),
                            );
                        }
                    }
                    let mut store = SpatialIndex::new(vec![]).unwrap();
                    let mut out = Vec::new();
                    simulate_groups(
                        &model(),
                        hard.as_slice(),
                        &mut store,
                        path(),
                        drift,
                        options(
                            shifted.then_some(&trend as &dyn KnownMean),
                            3,
                            &budget,
                            &cancel,
                        ),
                        |group| {
                            out.extend_from_slice(group);
                            Ok(())
                        },
                    )
                    .unwrap();
                    assert_eq!(budget.used(), 0);
                    out
                })
        };
        let residual = run(false, 1);
        let shifted = run(true, 1);
        let parallel = run(true, 4);
        for ((a, b), c) in residual.iter().zip(&shifted).zip(&parallel) {
            assert!((a.value() + trend(a.site()).unwrap() - b.value()).abs() < 2e-12);
            assert_eq!(b.value().to_bits(), c.value().to_bits());
        }
    }
    compare(&Simple);
    compare(&Ordinary { variables: 2 });
    compare(
        &ExternalDrift::new(4, |site: &Site, out: &mut [f64]| {
            out.fill(0.0);
            out[site.variable * 2] = 1.0;
            out[site.variable * 2 + 1] = site.support.center().coordinates()[0];
            Ok(())
        })
        .unwrap(),
    );
}

#[test]
fn joint_sequential_trend_ensemble_matches_independent_gaussian_conditional_moments() {
    let model = model();
    let budget = MemoryBudget::new(1 << 20);
    let cancel = Cancellation::default();
    let hard = [Observation::new(0, site(0.0, 0), 2.25, 0.0).unwrap()];
    let path = || {
        (0..2).map(|i| Group {
            id: i,
            targets: (0..2)
                .map(|v| Target {
                    id: 10 + 2 * i + v as u64,
                    site: site(i as f64, v),
                })
                .collect(),
        })
    };
    // Conditional on residual R_0(0)=0.25. These moments come directly from
    // the specified separable exponential covariance, independently of the
    // implementation's prepared weights or conditional covariance.
    let r = (-0.5_f64).exp();
    let c = [0.6, r, 0.6 * r];
    let expected = [-3.0 + 0.25 * c[0], 3.0 + 0.25 * c[1], -2.5 + 0.25 * c[2]];
    let prior = [[1.0, 0.6 * r, r], [0.6 * r, 1.0, 0.6], [r, 0.6, 1.0]];
    let covariance: [[f64; 3]; 3] =
        std::array::from_fn(|i| std::array::from_fn(|j| prior[i][j] - c[i] * c[j]));
    let n = 8192;
    let mut sums = [0.0; 3];
    let mut products = [[0.0; 3]; 3];
    for realization in 0..n {
        let mut store = SpatialIndex::new(vec![]).unwrap();
        let mut all = Vec::new();
        simulate_groups(
            &model,
            &hard[..],
            &mut store,
            path(),
            &Simple,
            options(Some(&trend), realization, &budget, &cancel),
            |group| {
                all.extend(group.iter().map(Observation::value));
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(all[0].to_bits(), 2.25_f64.to_bits());
        for i in 0..3 {
            let centered = all[i + 1] - expected[i];
            sums[i] += centered;
            for j in 0..3 {
                products[i][j] += centered * (all[j + 1] - expected[j]);
            }
        }
    }
    for i in 0..3 {
        assert!((sums[i] / n as f64).abs() < 6.0 * (covariance[i][i] / n as f64).sqrt());
        for j in 0..3 {
            let error = (products[i][j] / n as f64 - covariance[i][j]).abs();
            let se = ((covariance[i][i] * covariance[j][j] + covariance[i][j].powi(2)) / n as f64)
                .sqrt();
            assert!(error < 6.0 * se, "covariance ({i},{j}), error {error}");
        }
    }
    assert_eq!(budget.used(), 0);
}

#[test]
fn mean_errors_overflow_and_cancellation_do_not_publish_a_group() {
    let budget = MemoryBudget::new(1 << 20);
    let model = model();
    let nan = |_: &Site| Ok(f64::NAN);
    let infinity = |_: &Site| Ok(f64::INFINITY);
    let missing = |_: &Site| Err(Error::Cancelled);
    let cancel = Cancellation::default();
    for mean in [&nan as &dyn KnownMean, &infinity, &missing] {
        let mut store = SpatialIndex::new(vec![]).unwrap();
        let mut published = 0;
        assert!(simulate_groups(
            &model,
            &[][..],
            &mut store,
            path(),
            &Simple,
            options(Some(mean), 0, &budget, &cancel),
            |_| {
                published += 1;
                Ok(())
            },
        )
        .is_err());
        assert_eq!(published, 0);
        assert_eq!(budget.used(), 0);
        assert!(terrustrial_core::spatial::ObservationSource::nearest(
            &store,
            Point3::new([0.0; 3]).unwrap(),
            Neighborhood {
                maximum: 100,
                radius: 100.0
            },
            &cancel,
        )
        .unwrap()
        .is_empty());
    }
    let hard = [Observation::new(0, site(-1.0, 0), f64::MAX, 0.0).unwrap()];
    let extreme = |_: &Site| Ok(-f64::MAX);
    let mut store = SpatialIndex::new(vec![]).unwrap();
    assert!(matches!(
        simulate_groups(
            &model,
            &hard[..],
            &mut store,
            path(),
            &Simple,
            options(Some(&extreme), 0, &budget, &cancel),
            |_| Ok(()),
        ),
        Err(Error::Numerical { .. })
    ));
    let stopping = |_: &Site| {
        cancel.cancel();
        Ok(0.0)
    };
    assert!(matches!(
        simulate_groups(
            &model,
            &[][..],
            &mut store,
            path(),
            &Simple,
            options(Some(&stopping), 0, &budget, &cancel),
            |_| Ok(()),
        ),
        Err(Error::Cancelled)
    ));
    assert_eq!(budget.used(), 0);
}

#[test]
fn zero_mean_preserves_draw_bits_and_hard_values_survive_large_trend_cancellation() {
    let model = model();
    let budget = MemoryBudget::new(1 << 20);
    let cancel = Cancellation::default();
    let hard = [Observation::new(0, site(0.0, 0), -0.0, 0.0).unwrap()];
    let run = |mean| {
        let mut store = SpatialIndex::new(vec![]).unwrap();
        let mut out = Vec::new();
        let groups = (0..2).map(|i| Group {
            id: i,
            targets: (0..2)
                .map(|v| Target {
                    id: 100 + 2 * i + v as u64,
                    site: site(i as f64, v),
                })
                .collect(),
        });
        simulate_groups(
            &model,
            &hard[..],
            &mut store,
            groups,
            &Simple,
            options(mean, 7, &budget, &cancel),
            |group| {
                out.extend(group.iter().map(|o| o.value().to_bits()));
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(out[0], (-0.0_f64).to_bits());
        assert_eq!(budget.used(), 0);
        out
    };
    assert_eq!(run(None), run(Some(&|_: &Site| Ok(0.0))));
    run(Some(&|_: &Site| Ok(1e16)));
    // An impossible additional mean buffer fails without even invoking the mean.
    let tiny_budget = MemoryBudget::new(1);
    let unreachable =
        |_: &Site| -> Result<f64> { panic!("budget must be checked before mean evaluation") };
    let mut store = SpatialIndex::new(vec![]).unwrap();
    assert!(matches!(
        simulate_groups(
            &model,
            &hard[..],
            &mut store,
            path(),
            &Simple,
            options(Some(&unreachable), 0, &tiny_budget, &cancel),
            |_| Ok(()),
        ),
        Err(Error::MemoryBudget { .. })
    ));
    assert_eq!(tiny_budget.used(), 0);
}
