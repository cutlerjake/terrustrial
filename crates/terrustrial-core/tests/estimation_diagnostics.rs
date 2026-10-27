use std::sync::atomic::{AtomicUsize, Ordering};
use terrustrial_core::{
    data::{Group, Observation, Point3, Site, Support, Target},
    estimation::{
        estimate_groups, estimate_groups_with_residuals, factorial, EstimationOptions,
        GroupEstimate, GroupResiduals,
    },
    execution::{Cancellation, ExecutionOptions, MemoryBudget},
    gaussian::{
        Drift, ExternalDrift, LinearDrift, NumericalTolerance, Ordinary, Simple, Uncertainty,
    },
    model::{Anisotropy, Coregionalization, Correlation, Covariance, Lmc, Structure},
    spatial::{Neighborhood, ObservationSource, SpatialIndex},
    Error, Result,
};
fn p(i: usize) -> Point3 {
    Point3::new([
        (i as f64 * 1.31).sin(),
        (i as f64 * 0.79).cos(),
        (i as f64 * 0.37 + 0.1).sin(),
    ])
    .unwrap()
}
fn site(i: usize) -> Site {
    Site {
        variable: i % 2,
        support: if i.is_multiple_of(3) {
            Support::point(p(i))
        } else {
            Support::block(p(i), [0.03, 0.04, 0.02], [2; 3]).unwrap()
        },
    }
}
fn model(variables: usize, scale: f64) -> Lmc {
    let factor = if variables == 1 {
        vec![scale.sqrt()]
    } else {
        vec![scale.sqrt(), 0., 0.3 * scale.sqrt(), 0.8 * scale.sqrt()]
    };
    Lmc::new(vec![Structure {
        correlation: Correlation::Exponential,
        anisotropy: Anisotropy::isotropic(2.).unwrap(),
        coregionalization: Coregionalization::from_factor(variables, variables, &factor).unwrap(),
    }])
    .unwrap()
}
fn observations() -> Vec<Observation> {
    (0..32)
        .map(|i| {
            Observation::new(
                u64::MAX - i as u64,
                site(i),
                if i == 0 { -0. } else { (i as f64).sin() },
                if i == 0 { 0. } else { 0.03 },
            )
            .unwrap()
        })
        .collect()
}
fn groups() -> Vec<Group> {
    [1, 2, 7, 9, 3, 1]
        .into_iter()
        .enumerate()
        .map(|(g, m)| Group {
            id: (g * 4) as u64,
            targets: (0..m)
                .map(|j| Target {
                    id: (1000 + g * 16 + j) as u64,
                    site: if j == 0 {
                        site(0)
                    } else if j == 2 {
                        site(41)
                    } else {
                        site(40 + j + g)
                    },
                })
                .collect(),
        })
        .collect()
}
fn execution(batch: usize) -> ExecutionOptions {
    ExecutionOptions {
        batch_groups: batch,
        maximum_targets_per_group: 9,
        bytes_per_group: 4 << 20,
    }
}
fn options<'a>(
    budget: &'a MemoryBudget,
    cancel: &'a Cancellation,
    uncertainty: Uncertainty,
) -> EstimationOptions<'a> {
    EstimationOptions {
        neighborhood: Neighborhood {
            maximum: 32,
            radius: 100.,
        }
        .into(),
        mean: None,
        uncertainty,
        tolerance: NumericalTolerance::default(),
        budget,
        cancellation: cancel,
    }
}
fn bits(out: &GroupEstimate) -> Vec<u64> {
    let mut bits = vec![out.group.id];
    bits.extend(out.group.targets.iter().map(|t| t.id));
    bits.extend(out.values.iter().map(|v| v.to_bits()));
    if let Some(c) = out.covariance() {
        for j in 0..c.ncols() {
            for i in 0..c.nrows() {
                bits.push(c[(i, j)].to_bits());
            }
        }
    }
    bits
}
fn check_report(report: &GroupResiduals, m: usize, p: usize, joint: bool) {
    let r = report.equations;
    assert_eq!(r.covariance_equations.equations, report.observations * m);
    assert_eq!(r.original_drift_constraints.equations, p * m);
    assert_eq!(r.drift_constraints.equations, p * m);
    assert_eq!(r.conditional_covariance.is_some(), joint);
    assert_eq!(r.sampling_factor.is_some(), joint);
    for block in [
        Some(r.covariance_equations),
        Some(r.drift_constraints),
        Some(r.original_drift_constraints),
        r.conditional_covariance,
        r.sampling_factor,
    ]
    .into_iter()
    .flatten()
    {
        assert!(block.max_componentwise_backward_error < 1e-10, "{block:?}");
        assert_eq!(block.scaled_product_underflows, 0);
    }
}
#[test]
fn ordered_reports_match_default_bits_across_storage_methods_workers_and_batches() {
    let data = observations();
    let resident = SpatialIndex::new(data.clone()).unwrap();
    #[cfg(feature = "sqlite")]
    let storage_budget = MemoryBudget::new(64 << 20);
    #[cfg(feature = "sqlite")]
    let directory = tempfile::tempdir().unwrap();
    #[cfg(feature = "sqlite")]
    let sqlite = {
        use terrustrial_core::storage::{SqliteReadPool, SqliteStore};
        let path = directory.path().join("ordinary.sqlite");
        let mut store = SqliteStore::create(&path, 4096, &storage_budget).unwrap();
        store
            .import(data.as_slice(), &Cancellation::default())
            .unwrap();
        drop(store);
        SqliteReadPool::open_snapshot(path, 4, 4096, &storage_budget, &Cancellation::default())
            .unwrap()
    };
    #[cfg(feature = "sqlite")]
    let packed = {
        use terrustrial_core::storage::packed::{Options, PackedBuilder};
        let mut builder = PackedBuilder::create(
            directory.path().join("packed.sqlite"),
            Options {
                leaf_records: 7,
                cached_pages: 2,
                readers: 4,
                directory_pages: 2,
                sqlite_cache_bytes: 4096,
                ..Options::default()
            },
            &storage_budget,
        )
        .unwrap();
        for o in data.iter().rev() {
            builder.push(*o, 0, &Cancellation::default()).unwrap();
        }
        builder.finish(&Cancellation::default()).unwrap()
    };
    #[cfg(feature = "sqlite")]
    let snapshot = packed.snapshot().unwrap();
    let sources: Vec<&(dyn ObservationSource + Sync)> = vec![
        &resident,
        #[cfg(feature = "sqlite")]
        &sqlite,
        #[cfg(feature = "sqlite")]
        &snapshot,
    ];
    let model = model(2, 1.);
    fn run(
        model: &Lmc,
        data: &[Observation],
        sources: &[&(dyn ObservationSource + Sync)],
        drift: &impl Drift,
    ) {
        for uncertainty in [Uncertainty::None, Uncertainty::Joint] {
            let mut reference = None;
            for workers in [1, 4] {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(workers)
                    .build()
                    .unwrap();
                for batch in [1, 3] {
                    for source in sources {
                        let budget = MemoryBudget::new(128 << 20);
                        let cancel = Cancellation::default();
                        let mut normal = Vec::new();
                        pool.install(|| {
                            estimate_groups(
                                model,
                                *source,
                                groups(),
                                drift,
                                options(&budget, &cancel, uncertainty),
                                execution(batch),
                                |out| {
                                    normal.push(bits(out));
                                    Ok(())
                                },
                            )
                        })
                        .unwrap();
                        let mut actual = Vec::new();
                        let mut reports = Vec::new();
                        pool.install(|| {
                            estimate_groups_with_residuals(
                                model,
                                *source,
                                groups(),
                                drift,
                                options(&budget, &cancel, uncertainty),
                                execution(batch),
                                |out, report| {
                                    check_report(
                                        report,
                                        out.values.len(),
                                        drift.columns(),
                                        uncertainty == Uncertainty::Joint,
                                    );
                                    let row =
                                        report.equations.covariance_equations.worst.unwrap().row;
                                    let mut ids: Vec<_> =
                                        data.iter().map(Observation::id).collect();
                                    ids.sort_unstable();
                                    assert_eq!(report.covariance_observation_id, Some(ids[row]));
                                    assert_eq!(out.values[0].to_bits(), (-0f64).to_bits());
                                    actual.push(bits(out));
                                    reports.push(*report);
                                    Ok(())
                                },
                            )
                        })
                        .unwrap();
                        assert_eq!(actual, normal);
                        if let Some(expected) = &reference {
                            assert_eq!(&(actual, reports), expected);
                        } else {
                            reference = Some((actual, reports));
                        }
                        assert_eq!(budget.used(), 0);
                    }
                }
            }
        }
    }
    run(&model, &data, &sources, &Simple);
    run(&model, &data, &sources, &Ordinary { variables: 2 });
    run(
        &model,
        &data,
        &sources,
        &LinearDrift::new(2, [0.; 3], 1.).unwrap(),
    );
    let external = ExternalDrift::new(4, |s: &Site, out: &mut [f64]| {
        out.fill(0.);
        out[2 * s.variable] = 1.;
        out[2 * s.variable + 1] = s
            .support
            .nodes()
            .map(|p| p.coordinates()[0].powi(2))
            .sum::<f64>()
            / s.support.node_count() as f64;
        Ok(())
    })
    .unwrap();
    run(&model, &data, &sources, &external);
    drop(sources);
    #[cfg(feature = "sqlite")]
    {
        drop((snapshot, packed, sqlite));
        assert_eq!(storage_budget.used(), 0);
    }
}

#[test]
fn factorial_report_uses_distinct_target_drift_and_matches_analytic_component_moments() {
    let site = |x| Site::point(Point3::new([x, 0., 0.]).unwrap(), 0);
    let data = SpatialIndex::new(vec![
        Observation::new(0, site(0.), 2., 0.).unwrap(),
        Observation::new(1, site(2.), 6., 0.).unwrap(),
    ])
    .unwrap();
    let total = model(1, 1.);
    let component = model(1, 0.25);
    let drift = Ordinary { variables: 1 };
    let zero = ExternalDrift::new(1, |_: &Site, out: &mut [f64]| {
        out[0] = 0.;
        Ok(())
    })
    .unwrap();
    let budget = MemoryBudget::new(64 << 20);
    let cancel = Cancellation::default();
    let group = Group {
        id: 0,
        targets: vec![
            Target {
                id: 10,
                site: site(0.),
            },
            Target {
                id: 11,
                site: site(2.),
            },
        ],
    };
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    pool.install(|| {
        factorial::estimate_groups_with_residuals(
            &total,
            &data,
            [group],
            &drift,
            factorial::Component {
                covariance: &component,
                drift: &zero,
                mean: None,
            },
            options(&budget, &cancel, Uncertainty::Joint),
            execution(1),
            |out, report| {
                check_report(report, 2, 1, true);
                assert!((out.values[0] + 0.5).abs() < 1e-12);
                assert!((out.values[1] - 0.5).abs() < 1e-12);
                let r = (-1f64).exp(); // Distance 2 divided by the configured scale 2.
                let reduction = 0.25f64.powi(2) * (1. - r) / 2.;
                let c = out.covariance().unwrap();
                assert!((c[(0, 0)] - (0.25 - reduction)).abs() < 1e-12);
                assert!((c[(0, 1)] - (0.25 * r + reduction)).abs() < 1e-12);
                Ok(())
            },
        )
    })
    .unwrap();
    assert_eq!(budget.used(), 0);
}

struct Injected {
    inner: Lmc,
    calls: AtomicUsize,
    fail_at: AtomicUsize,
    mode: AtomicUsize,
    cancel: Cancellation,
}
impl Covariance for Injected {
    fn variables(&self) -> usize {
        2
    }
    fn point_covariance(&self, a: Point3, va: usize, b: Point3, vb: usize) -> Result<f64> {
        if self.calls.fetch_add(1, Ordering::Relaxed) == self.fail_at.load(Ordering::Relaxed) {
            match self.mode.load(Ordering::Relaxed) {
                0 => {
                    return Err(Error::Numerical {
                        context: "injected diagnostic failure",
                    })
                }
                1 => return Ok(f64::NAN),
                _ => self.cancel.cancel(),
            }
        }
        self.inner.point_covariance(a, va, b, vb)
    }
}
#[test]
fn diagnostic_callback_failures_cancel_and_admission_errors_release_budgets_without_output() {
    let cancel = Cancellation::default();
    let model = Injected {
        inner: model(2, 1.),
        calls: AtomicUsize::new(0),
        fail_at: AtomicUsize::new(usize::MAX),
        mode: AtomicUsize::new(0),
        cancel: cancel.clone(),
    };
    let data = SpatialIndex::new(observations()).unwrap();
    let group = groups().remove(1);
    let budget = MemoryBudget::new(32 << 20);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let mut expected = Vec::new();
    pool.install(|| {
        estimate_groups(
            &model,
            &data,
            [group.clone()],
            &Simple,
            options(&budget, &cancel, Uncertainty::None),
            execution(1),
            |out| {
                expected = bits(out);
                Ok(())
            },
        )
    })
    .unwrap();
    let preparation_calls = model.calls.load(Ordering::Relaxed);
    for mode in 0..3 {
        model.calls.store(0, Ordering::Relaxed);
        model
            .fail_at
            .store(preparation_calls + 3, Ordering::Relaxed);
        model.mode.store(mode, Ordering::Relaxed);
        let result = pool.install(|| {
            estimate_groups_with_residuals(
                &model,
                &data,
                [group.clone()],
                &Simple,
                options(&budget, &cancel, Uncertainty::None),
                execution(1),
                |_, _| panic!("failed diagnostic emitted output"),
            )
        });
        assert!(result.is_err());
        if mode == 2 {
            assert!(matches!(result, Err(Error::Cancelled)));
        }
        assert_eq!(budget.used(), 0);
        model.fail_at.store(usize::MAX, Ordering::Relaxed);
        pool.install(|| {
            estimate_groups_with_residuals(
                &model,
                &data,
                [group.clone()],
                &Simple,
                options(&budget, &Cancellation::default(), Uncertainty::None),
                execution(1),
                |out, _| {
                    assert_eq!(bits(out), expected);
                    Ok(())
                },
            )
        })
        .unwrap();
        assert_eq!(budget.used(), 0);
    }
    let exec = execution(1);
    let exact = MemoryBudget::new(
        2 * exec.bytes_per_group + (exec.maximum_targets_per_group * 512 + 256) as u64,
    );
    pool.install(|| {
        estimate_groups(
            &model,
            &data,
            [group.clone()],
            &Simple,
            options(&exact, &Cancellation::default(), Uncertainty::None),
            exec,
            |_| Ok(()),
        )
    })
    .unwrap();
    assert!(matches!(
        pool.install(|| estimate_groups_with_residuals(
            &model,
            &data,
            [group],
            &Simple,
            options(&exact, &Cancellation::default(), Uncertainty::None),
            exec,
            |_, _| panic!("unadmitted report")
        )),
        Err(Error::MemoryBudget { .. })
    ));
    assert_eq!(exact.used(), 0);
}

#[test]
fn rejecting_an_ordered_report_preserves_only_the_accepted_prefix() {
    let budget = MemoryBudget::new(64 << 20);
    let cancel = Cancellation::default();
    let data = SpatialIndex::new(observations()).unwrap();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let mut emitted = Vec::new();
    let result = pool.install(|| {
        estimate_groups_with_residuals(
            &model(2, 1.),
            &data,
            groups(),
            &Simple,
            options(&budget, &cancel, Uncertainty::Joint),
            execution(3),
            |out, _| {
                if out.group.id == 8 {
                    return Err(Error::InvalidInput(
                        "caller rejected group 8 before publishing".into(),
                    ));
                }
                emitted.push(out.group.id);
                Ok(())
            },
        )
    });
    assert!(result.is_err());
    assert_eq!(emitted, vec![0, 4]);
    assert_eq!(budget.used(), 0);
}
