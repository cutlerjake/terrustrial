//! Independent augmented-system checks of total-data/component-target semantics.
use terrustrial_core::{
    data::{Group, Observation, Point3, Site, Support, Target},
    estimation::{
        factorial::{self, Component},
        EstimationOptions,
    },
    execution::{Cancellation, ExecutionOptions, MemoryBudget},
    gaussian::{
        ExternalDrift, GaussianWorkspace, NumericalTolerance, Ordinary, Uncertainty,
        WorkspaceOptions,
    },
    model::{Anisotropy, Coregionalization, Correlation, Lmc, Structure},
    spatial::{Neighborhood, ObservationSource, SpatialIndex},
    Result,
};
fn site(x: f64, v: usize, block: bool) -> Site {
    let p = Point3::new([x, 0., 0.]).unwrap();
    Site {
        variable: v,
        support: if block {
            Support::block(p, [0.15, 0., 0.], [3, 1, 1]).unwrap()
        } else {
            Support::point(p)
        },
    }
}
fn model(v: usize, sill: f64, correlation: Correlation) -> Lmc {
    let f: Vec<_> = (0..v)
        .flat_map(|i| {
            (0..=v).map(move |j| {
                sill.sqrt()
                    * if j == v {
                        0.2_f64.sqrt()
                    } else if j == i {
                        0.8_f64.sqrt()
                    } else {
                        0.
                    }
            })
        })
        .collect();
    Lmc::new(vec![Structure {
        correlation,
        anisotropy: Anisotropy::isotropic(2.).unwrap(),
        coregionalization: Coregionalization::from_factor(v, v + 1, &f).unwrap(),
    }])
    .unwrap()
}
fn options<'a>(
    n: usize,
    b: &'a MemoryBudget,
    c: &'a Cancellation,
    mean: Option<&'a dyn terrustrial_core::gaussian::KnownMean>,
) -> EstimationOptions<'a> {
    EstimationOptions {
        neighborhood: Neighborhood {
            maximum: n,
            radius: 100.,
        }
        .into(),
        mean,
        uncertainty: Uncertainty::Joint,
        tolerance: NumericalTolerance::default(),
        budget: b,
        cancellation: c,
    }
}
fn execution(v: usize, batch: usize) -> ExecutionOptions {
    ExecutionOptions {
        batch_groups: batch,
        maximum_targets_per_group: v,
        bytes_per_group: 4 << 20,
    }
}
fn cov(a: &Site, b: &Site) -> f64 {
    let nodes = |s: &Site| {
        let x = s.support.center().coordinates()[0];
        let w = s.support.half_extents()[0];
        let n = s.support.divisions()[0];
        (0..n)
            .map(|i| x - w + (i as f64 + 0.5) * 2. * w / n as f64)
            .collect::<Vec<_>>()
    };
    let aa = nodes(a);
    let bb = nodes(b);
    aa.iter()
        .flat_map(|a| bb.iter().map(move |b| (-(a - b).abs() / 2.).exp()))
        .sum::<f64>()
        / (aa.len() * bb.len()) as f64
        * if a.variable == b.variable { 1. } else { 0.2 }
}
// Independent pivoted scalar elimination, intentionally separate from faer/GLS.
#[allow(clippy::needless_range_loop)]
fn solve(mut a: Vec<Vec<f64>>, mut b: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let n = a.len();
    let m = b[0].len();
    for k in 0..n {
        let pivot = (k..n)
            .max_by(|&i, &j| a[i][k].abs().total_cmp(&a[j][k].abs()))
            .unwrap();
        assert!(a[pivot][k].abs() > 1e-12);
        a.swap(k, pivot);
        b.swap(k, pivot);
        for i in k + 1..n {
            let f = a[i][k] / a[k][k];
            for j in k + 1..n {
                a[i][j] -= f * a[k][j];
            }
            for j in 0..m {
                b[i][j] -= f * b[k][j];
            }
        }
    }
    for i in (0..n).rev() {
        for j in 0..m {
            for k in i + 1..n {
                b[i][j] -= a[i][k] * b[k][j];
            }
            b[i][j] /= a[i][i];
        }
    }
    b
}
#[allow(clippy::needless_range_loop)]
fn reference(
    data: &[Observation],
    targets: &[Target],
    variables: usize,
    response: f64,
) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = data.len();
    let m = targets.len();
    let p = variables;
    let mut a = vec![vec![0.; n + p]; n + p];
    let mut b = vec![vec![0.; m]; n + p];
    for i in 0..n {
        for j in 0..n {
            a[i][j] = cov(data[i].site(), data[j].site())
                + if i == j { data[i].noise_variance() } else { 0. };
        }
        a[i][n + data[i].site().variable] = 1.;
        a[n + data[i].site().variable][i] = 1.;
        for j in 0..m {
            b[i][j] = 0.25 * cov(data[i].site(), &targets[j].site);
        }
    }
    for j in 0..m {
        b[n + targets[j].site.variable][j] = response;
    }
    let w = solve(a, b);
    let means = (0..m)
        .map(|j| {
            3. + (0..n)
                .map(|i| w[i][j] * (data[i].value() - 10.))
                .sum::<f64>()
        })
        .collect();
    let mut variance = vec![vec![0.; m]; m];
    for i in 0..m {
        for j in 0..m {
            let mut v = 0.25 * cov(&targets[i].site, &targets[j].site);
            for k in 0..n {
                v -= 0.25 * cov(data[k].site(), &targets[i].site) * w[k][j];
            }
            v -= response * w[n + targets[i].site.variable][j];
            variance[i][j] = v;
        }
    }
    (means, variance)
}

#[test]
fn centered_component_is_not_an_exact_total_observation_and_has_zero_sum_weights() -> Result<()> {
    let total = model(1, 4., Correlation::Nugget);
    let component = model(1, 1., Correlation::Nugget);
    let data = [
        Observation::new(0, site(0., 0, false), 10., 0.)?,
        Observation::new(1, site(2., 0, false), 14., 0.)?,
    ];
    let budget = MemoryBudget::new(32 << 20);
    let cancel = Cancellation::default();
    let zero = ExternalDrift::new(1, |_: &Site, out: &mut [f64]| {
        out.fill(0.);
        Ok(())
    })?;
    let drift = Ordinary { variables: 1 };
    let target = [site(0., 0, false)];
    let mut workspace = GaussianWorkspace::new(2, 1, 1, &budget)?;
    let mut conditioned = workspace.condition(&total, &data, &drift, &cancel)?;
    let plan = conditioned.prepare_component_with_drift(
        &component,
        &target,
        &zero,
        WorkspaceOptions {
            uncertainty: Uncertainty::Joint,
            tolerance: NumericalTolerance::default(),
            cancellation: &cancel,
        },
    )?;
    assert!((plan.weights()[(0, 0)] - 0.125).abs() < 1e-14);
    assert!((plan.weights()[(1, 0)] + 0.125).abs() < 1e-14);
    let hard = SpatialIndex::new(data.to_vec())?;
    let total_mean = |_: &Site| Ok(10.);
    let component_mean = |_: &Site| Ok(3.);
    factorial::estimate_groups(
        &total,
        &hard,
        [Group {
            id: 0,
            targets: vec![Target {
                id: 42,
                site: target[0],
            }],
        }],
        &drift,
        Component {
            covariance: &component,
            drift: &zero,
            mean: Some(&component_mean),
        },
        options(2, &budget, &cancel, Some(&total_mean)),
        execution(1, 1),
        |out| {
            assert!((out.values[0] - 2.5).abs() < 1e-14);
            assert!((out.covariance().unwrap()[(0, 0)] - 0.875).abs() < 1e-14);
            Ok(())
        },
    )?;
    Ok(())
}

#[test]
fn joint_component_drivers_match_independent_systems_across_support_sources_and_workers(
) -> Result<()> {
    let cancel = Cancellation::default();
    for variables in [1, 2, 8, 32] {
        for block in [false, true] {
            let total = model(variables, 1., Correlation::Exponential);
            let component = model(variables, 0.25, Correlation::Exponential);
            let data: Vec<_> = (0..variables * 3)
                .map(|i| {
                    Observation::new(
                        i as u64,
                        site(
                            (i / variables) as f64 * 1.7 + (i % variables) as f64 * 0.017,
                            i % variables,
                            block && i % 2 == 0,
                        ),
                        10. + (i as f64 * 0.37).sin(),
                        if i == 0 { 0. } else { 0.05 },
                    )
                    .unwrap()
                })
                .collect();
            let groups: Vec<_> = (0..4)
                .map(|g| Group {
                    id: g,
                    targets: (0..variables)
                        .map(|v| Target {
                            id: 10000 + g * variables as u64 + v as u64,
                            site: if g == 0 && v == 0 {
                                *data[0].site()
                            } else {
                                site(0.27 + g as f64 * 0.23 + v as f64 * 0.003, v, block)
                            },
                        })
                        .collect(),
                })
                .collect();
            let resident = SpatialIndex::new(data.clone())?;
            #[cfg(feature = "sqlite")]
            let disk_budget = MemoryBudget::new(32 << 20);
            #[cfg(feature = "sqlite")]
            let directory = tempfile::tempdir()?;
            #[cfg(feature = "sqlite")]
            let store = {
                use terrustrial_core::storage::packed::{Options, PackedBuilder};
                let mut builder = PackedBuilder::create(
                    directory.path().join("components.sqlite"),
                    Options {
                        leaf_records: 7,
                        cached_pages: 2,
                        directory_pages: 2,
                        readers: 4,
                        sqlite_cache_bytes: 4096,
                        ..Options::default()
                    },
                    &disk_budget,
                )?;
                for observation in data.iter().rev() {
                    builder.push(*observation, 0, &cancel)?;
                }
                builder.finish(&cancel)?
            };
            #[cfg(feature = "sqlite")]
            let snapshot = store.snapshot()?;
            let sources: Vec<&(dyn ObservationSource + Sync)> = vec![
                &resident,
                #[cfg(feature = "sqlite")]
                &snapshot,
            ];
            for response in [0., 0.25] {
                let expected: Vec<_> = groups
                    .iter()
                    .map(|g| reference(&data, &g.targets, variables, response))
                    .collect();
                let target_drift = ExternalDrift::new(variables, |s: &Site, out: &mut [f64]| {
                    out.fill(0.);
                    out[s.variable] = response;
                    Ok(())
                })?;
                let drift = Ordinary { variables };
                let mut baseline = None;
                for workers in [1, 4] {
                    let pool = rayon::ThreadPoolBuilder::new()
                        .num_threads(workers)
                        .build()
                        .unwrap();
                    for source in &sources {
                        for batch in [1, 3] {
                            let budget = MemoryBudget::new(128 << 20);
                            let mut actual = Vec::new();
                            let mut count = 0;
                            pool.install(|| {
                                factorial::estimate_groups(
                                    &total,
                                    *source,
                                    groups.iter().cloned(),
                                    &drift,
                                    Component {
                                        covariance: &component,
                                        drift: &target_drift,
                                        mean: Some(&|_: &Site| Ok(3.)),
                                    },
                                    options(
                                        data.len(),
                                        &budget,
                                        &cancel,
                                        Some(&|_: &Site| Ok(10.)),
                                    ),
                                    execution(variables, batch),
                                    |out| {
                                        let (mean, cov) = &expected[count];
                                        for (a, e) in out.values.iter().zip(mean) {
                                            assert!((a - e).abs() < 1e-10 * (1. + e.abs()));
                                            actual.push(a.to_bits());
                                        }
                                        let c = out.covariance().unwrap();
                                        for j in 0..variables {
                                            for i in 0..variables {
                                                assert!(
                                                    (c[(i, j)] - cov[i][j]).abs()
                                                        < 1e-10 * (1. + cov[i][j].abs())
                                                );
                                                actual.push(c[(i, j)].to_bits());
                                            }
                                        }
                                        count += 1;
                                        Ok(())
                                    },
                                )
                            })?;
                            assert_eq!(count, groups.len());
                            assert_eq!(budget.used(), 0);
                            if let Some(b) = &baseline {
                                assert_eq!(&actual, b);
                            } else {
                                baseline = Some(actual);
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

#[test]
fn component_errors_preserve_output_prefix_and_release_worker_budgets() -> Result<()> {
    let total = model(1, 1., Correlation::Exponential);
    let component = model(1, 0.25, Correlation::Exponential);
    let data = SpatialIndex::new(vec![
        Observation::new(0, site(0., 0, false), 10., 0.)?,
        Observation::new(1, site(2., 0, false), 14., 0.)?,
    ])?;
    let groups: Vec<_> = (0..3)
        .map(|i| Group {
            id: i,
            targets: vec![Target {
                id: 100 + i,
                site: site(i as f64, 0, false),
            }],
        })
        .collect();
    let total_drift = Ordinary { variables: 1 };
    let zero = ExternalDrift::new(1, |_: &Site, v: &mut [f64]| {
        v.fill(0.);
        Ok(())
    })?;
    let cancel = Cancellation::default();
    for workers in [1, 4] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .unwrap();
        for batch in [1, 3] {
            for mode in 0..4 {
                let budget = MemoryBudget::new(64 << 20);
                let mut emitted = 0;
                let target_mean = |s: &Site| {
                    if s.support.center().coordinates()[0] >= 1. && mode == 0 {
                        return Ok(f64::NAN);
                    }
                    Ok(3.)
                };
                let target_drift = ExternalDrift::new(
                    if mode == 2 { 2 } else { 1 },
                    |s: &Site, v: &mut [f64]| {
                        if s.support.center().coordinates()[0] >= 1. && mode == 1 {
                            return Err(terrustrial_core::Error::Numerical {
                                context: "injected target drift",
                            });
                        }
                        v.fill(0.);
                        Ok(())
                    },
                )?;
                let outcome = pool.install(|| {
                    factorial::estimate_groups(
                        &total,
                        &data,
                        groups.iter().cloned(),
                        &total_drift,
                        Component {
                            covariance: &component,
                            drift: &target_drift,
                            mean: Some(&target_mean),
                        },
                        options(2, &budget, &cancel, None),
                        execution(1, batch),
                        |out| {
                            assert_eq!(out.group.id, emitted);
                            emitted += 1;
                            if mode == 3 {
                                Err(terrustrial_core::Error::Numerical {
                                    context: "injected sink",
                                })
                            } else {
                                Ok(())
                            }
                        },
                    )
                });
                assert!(outcome.is_err());
                assert_eq!(emitted, if mode == 2 { 0 } else { 1 });
                assert_eq!(budget.used(), 0);
                let mut complete = 0;
                pool.install(|| {
                    factorial::estimate_groups(
                        &total,
                        &data,
                        groups.iter().cloned(),
                        &total_drift,
                        Component {
                            covariance: &component,
                            drift: &zero,
                            mean: None,
                        },
                        options(2, &budget, &cancel, None),
                        execution(1, batch),
                        |_| {
                            complete += 1;
                            Ok(())
                        },
                    )
                })?;
                assert_eq!(complete, 3);
                assert_eq!(budget.used(), 0);
            }
        }
    }
    let tiny = MemoryBudget::new(0);
    assert!(matches!(
        factorial::estimate_groups(
            &total,
            &data,
            groups.iter().cloned(),
            &total_drift,
            Component {
                covariance: &component,
                drift: &zero,
                mean: None
            },
            options(2, &tiny, &cancel, None),
            execution(1, 1),
            |_| panic!("no output")
        ),
        Err(terrustrial_core::Error::MemoryBudget { .. })
    ));
    assert_eq!(tiny.used(), 0);
    cancel.cancel();
    assert!(matches!(
        factorial::estimate_groups(
            &total,
            &data,
            groups,
            &total_drift,
            Component {
                covariance: &component,
                drift: &zero,
                mean: None
            },
            options(2, &tiny, &cancel, None),
            execution(1, 1),
            |_| panic!("no output")
        ),
        Err(terrustrial_core::Error::Cancelled)
    ));
    Ok(())
}

#[test]
fn joint_components_retain_negative_posterior_cross_covariance_and_total_identity() -> Result<()> {
    use terrustrial_core::{
        gaussian::{GaussianPlan, PrepareOptions, Simple},
        model::{components::Components, Covariance},
    };
    let cancel = Cancellation::default();
    let budget = MemoryBudget::new(64 << 20);
    for variables in [1, 2, 8, 32] {
        let a = model(variables, 0.25, Correlation::Exponential);
        let b = model(variables, 0.75, Correlation::Exponential);
        let total = Lmc::new(
            a.structures()
                .iter()
                .chain(b.structures())
                .cloned()
                .collect(),
        )?;
        assert!(Components::new(&total, &[0, 0]).is_err());
        assert!(Components::new(&total, &[2]).is_err());
        let joint = Components::new(&total, &[0, 1])?;
        let data: Vec<_> = (0..variables)
            .map(|v| Observation::new(v as u64, site(0., v, false), 2. + v as f64, 0.).unwrap())
            .collect();
        let targets: Vec<_> = (0..2)
            .flat_map(|c| (0..variables).map(move |v| (c, v)))
            .map(|(c, v)| joint.component_site(c, site(0., v, false)).unwrap())
            .collect();
        let plan = GaussianPlan::prepare(
            &joint,
            &data,
            &targets,
            &Simple,
            PrepareOptions {
                uncertainty: Uncertainty::Joint,
                tolerance: NumericalTolerance::default(),
                budget: &budget,
                cancellation: &cancel,
            },
        )?;
        let mean = plan.estimate(&data.iter().map(Observation::value).collect::<Vec<_>>())?;
        let covariance = plan.covariance().unwrap();
        for i in 0..2 * variables {
            let (ci, vi) = joint.decode_variable(targets[i].variable)?;
            let alpha = if ci == Some(0) { 0.25 } else { 0.75 };
            assert!((mean[i] - alpha * (2. + vi as f64)).abs() < 1e-12);
            for j in 0..2 * variables {
                let (cj, vj) = joint.decode_variable(targets[j].variable)?;
                let beta = if cj == Some(0) { 0.25 } else { 0.75 };
                let sill = if vi == vj { 1. } else { 0.2 };
                let expected = (if ci == cj { alpha } else { 0. } - alpha * beta) * sill;
                assert!(
                    (covariance[(i, j)] - expected).abs() < 1e-12,
                    "{i},{j}: {} != {expected}",
                    covariance[(i, j)]
                );
            }
        }
        // Exact total constraints persist in joint draws despite component uncertainty.
        for realization in 0..128 {
            let draw = plan.sample(&mean, 109, realization, 0)?;
            for v in 0..variables {
                assert!((draw[v] + draw[variables + v] - (2. + v as f64)).abs() < 1e-11);
            }
        }
        // Exercise the fallback when conditioning itself contains component variables.
        let component_data = [Observation::new(999, targets[0], 1., 0.)?];
        let mut row = [0.];
        let mut column = [0.];
        for target in &targets {
            joint.covariance_row(target, &component_data, &mut row)?;
            joint.covariance_column(&component_data, target, &mut column)?;
            assert_eq!(row[0], joint.covariance(target, component_data[0].site())?);
            assert_eq!(row, column);
        }
        assert!(joint.decode_variable(joint.variables()).is_err());
        assert!(joint.component_site(2, site(0., 0, false)).is_err());
        assert!(joint.component_site(0, site(0., variables, false)).is_err());
    }
    assert_eq!(budget.used(), 0);
    Ok(())
}
