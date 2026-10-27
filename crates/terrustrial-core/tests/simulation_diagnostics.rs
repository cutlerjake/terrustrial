use terrustrial_core::{
    data::{Group, Observation, Point3, Site, Support, Target},
    execution::{Cancellation, ExecutionOptions, MemoryBudget},
    gaussian::{Drift, ExternalDrift, GroupResiduals, LinearDrift, Ordinary, Simple},
    identity,
    model::{Anisotropy, Coregionalization, Correlation, Covariance, Lmc, Structure},
    simulation::{self, parallel, CheckpointedOptions, SimulationOptions},
    spatial::{Neighborhood, ObservationSource, SpatialIndex},
    storage::{Checkpoint, CheckpointStore},
    Error, Result,
};
fn fail() -> Error {
    Error::Numerical {
        context: "rejected numerical simulation report",
    }
}
fn point(i: usize) -> Point3 {
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
            Support::point(point(i))
        } else {
            Support::block(point(i), [0.015, 0.02, 0.01], [2; 3]).unwrap()
        },
    }
}
fn model() -> Lmc {
    Lmc::new(vec![Structure {
        correlation: Correlation::Exponential,
        anisotropy: Anisotropy::isotropic(2.).unwrap(),
        coregionalization: Coregionalization::from_factor(2, 2, &[1., 0., 0.3, 0.8]).unwrap(),
    }])
    .unwrap()
}
fn hard() -> Vec<Observation> {
    (0..24)
        .map(|i| {
            Observation::new(
                u64::MAX - i as u64,
                site(i),
                if i == 0 { -0. } else { (i as f64).sin() },
                if i == 0 { 0. } else { 0.02 },
            )
            .unwrap()
        })
        .collect()
}
fn groups() -> Vec<Group> {
    [1, 2, 7, 8, 9, 16]
        .into_iter()
        .enumerate()
        .map(|(g, m)| Group {
            id: g as u64 * 3,
            targets: (0..m)
                .map(|j| Target {
                    id: 1000 + (g * 32 + j) as u64,
                    site: if j == 0 {
                        site(0)
                    } else {
                        site(30 + g * 32 + j)
                    },
                })
                .collect(),
        })
        .collect()
}
fn options<'a>(budget: &'a MemoryBudget, cancellation: &'a Cancellation) -> SimulationOptions<'a> {
    SimulationOptions {
        seed: 73,
        realization: 5,
        mean: None,
        hard_neighborhood: Neighborhood {
            maximum: 128,
            radius: 100.,
        }
        .into(),
        simulated_neighborhood: Neighborhood {
            maximum: 128,
            radius: 100.,
        }
        .into(),
        budget,
        cancellation,
    }
}
fn accept(_: &Group, report: &GroupResiduals) -> Result<()> {
    let r = report.equations;
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
        if block.max_componentwise_backward_error > 1e-10 || block.scaled_product_underflows != 0 {
            return Err(fail());
        }
    }
    Ok(())
}
fn values(s: &(impl ObservationSource + ?Sized)) -> Vec<(u64, u64)> {
    let mut out = Vec::new();
    s.scan(&mut |o| {
        out.push((o.id(), o.value().to_bits()));
        Ok(())
    })
    .unwrap();
    out.sort();
    out
}
#[test]
fn checked_paths_preserve_bits_and_order_across_methods_windows_workers_and_sources() {
    let data = hard();
    let resident = SpatialIndex::new(data.clone()).unwrap();
    #[cfg(feature = "sqlite")]
    let budget_storage = MemoryBudget::new(64 << 20);
    #[cfg(feature = "sqlite")]
    let dir = tempfile::tempdir().unwrap();
    #[cfg(feature = "sqlite")]
    let pool = {
        use terrustrial_core::storage::{SqliteReadPool, SqliteStore};
        let path = dir.path().join("hard.db");
        let mut s = SqliteStore::create(&path, 4096, &budget_storage).unwrap();
        s.import(data.as_slice(), &Cancellation::default()).unwrap();
        drop(s);
        SqliteReadPool::open_snapshot(path, 4, 4096, &budget_storage, &Cancellation::default())
            .unwrap()
    };
    #[cfg(feature = "sqlite")]
    let packed = {
        use terrustrial_core::storage::packed::{Options, PackedBuilder};
        let mut b = PackedBuilder::create(
            dir.path().join("packed.db"),
            Options {
                leaf_records: 7,
                cached_pages: 2,
                readers: 4,
                directory_pages: 2,
                sqlite_cache_bytes: 4096,
                ..Options::default()
            },
            &budget_storage,
        )
        .unwrap();
        for o in &data {
            b.push(*o, 0, &Cancellation::default()).unwrap();
        }
        b.finish(&Cancellation::default()).unwrap()
    };
    #[cfg(feature = "sqlite")]
    let snapshot = packed.snapshot().unwrap();
    let sources: Vec<&(dyn ObservationSource + Sync)> = vec![
        &resident,
        #[cfg(feature = "sqlite")]
        &pool,
        #[cfg(feature = "sqlite")]
        &snapshot,
    ];
    fn check(drift: &impl Drift, sources: &[&(dyn ObservationSource + Sync)]) {
        let model = model();
        let budget = MemoryBudget::new(256 << 20);
        let cancel = Cancellation::default();
        let mut expected = SpatialIndex::new(vec![]).unwrap();
        simulation::simulate_groups(
            &model,
            sources[0],
            &mut expected,
            groups(),
            drift,
            options(&budget, &cancel),
            |_| Ok(()),
        )
        .unwrap();
        let expected = values(&expected);
        let mut reports_reference = None;
        for source in sources {
            for workers in [1, 4] {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(workers)
                    .build()
                    .unwrap();
                for batch in [1, 4] {
                    for parallel in [false, true] {
                        let mut store = SpatialIndex::new(vec![]).unwrap();
                        let mut reports = Vec::new();
                        let mut notified = Vec::new();
                        let mut inspect = |g: &Group, r: &GroupResiduals| {
                            accept(g, r)?;
                            reports.push((g.id, *r));
                            Ok(())
                        };
                        pool.install(|| {
                            if parallel {
                                parallel::simulate_groups_with_residuals(
                                    &model,
                                    *source,
                                    &mut store,
                                    groups(),
                                    drift,
                                    options(&budget, &cancel),
                                    ExecutionOptions {
                                        batch_groups: batch,
                                        maximum_targets_per_group: 16,
                                        bytes_per_group: 16 << 20,
                                    },
                                    &mut inspect,
                                    |o| {
                                        notified.extend(
                                            o.iter().map(|o| (o.id(), o.value().to_bits())),
                                        );
                                        Ok(())
                                    },
                                )
                            } else {
                                simulation::simulate_groups_with_residuals(
                                    &model,
                                    *source,
                                    &mut store,
                                    groups(),
                                    drift,
                                    options(&budget, &cancel),
                                    &mut inspect,
                                    |o| {
                                        notified.extend(
                                            o.iter().map(|o| (o.id(), o.value().to_bits())),
                                        );
                                        Ok(())
                                    },
                                )
                            }
                        })
                        .unwrap();
                        assert_eq!(values(&store), expected);
                        notified.sort();
                        assert_eq!(notified, expected);
                        assert_eq!(
                            reports.iter().map(|r| r.0).collect::<Vec<_>>(),
                            vec![0, 3, 6, 9, 12, 15]
                        );
                        if let Some(r) = &reports_reference {
                            assert_eq!(r, &reports);
                        } else {
                            reports_reference = Some(reports);
                        }
                        assert_eq!(budget.used(), 0);
                    }
                }
            }
        }
    }
    check(&Simple, &sources);
    check(&Ordinary { variables: 2 }, &sources);
    check(&LinearDrift::new(2, [0.; 3], 1.).unwrap(), &sources);
    let ext = ExternalDrift::new(4, |s: &Site, out: &mut [f64]| {
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
    check(&ext, &sources);
}
#[derive(Default)]
struct MemoryStore {
    rows: Vec<Observation>,
    checkpoint: Option<Checkpoint>,
    commits: usize,
}
impl ObservationSource for MemoryStore {
    fn scan(&self, v: &mut dyn FnMut(&Observation) -> Result<()>) -> Result<()> {
        self.rows.as_slice().scan(v)
    }
}
impl CheckpointStore for MemoryStore {
    fn checkpoint(&self) -> Result<Option<Checkpoint>> {
        Ok(self.checkpoint.clone())
    }
    fn is_empty(&self) -> Result<bool> {
        Ok(self.rows.is_empty())
    }
    fn commit_group(
        &mut self,
        rows: &[Observation],
        expected: Option<&Checkpoint>,
        next: &Checkpoint,
        cancel: &Cancellation,
    ) -> Result<()> {
        cancel.check()?;
        assert_eq!(self.checkpoint.as_ref(), expected);
        self.rows.extend_from_slice(rows);
        self.checkpoint = Some(next.clone());
        self.commits += 1;
        Ok(())
    }
}
#[test]
fn rejection_and_cancellation_precede_every_write_and_checkpoint_and_resume_is_identified() {
    let model = model();
    let hard = hard();
    let budget = MemoryBudget::new(256 << 20);
    let policy = identity::opaque("residual policy", b"v1 tolerance 1e-10").unwrap();
    for stop in [0, 6, 15] {
        for cancel_inside in [false, true] {
            for engine in [0, 1, 2] {
                let cancel = Cancellation::default();
                let mut store = MemoryStore::default();
                let mut resident = SpatialIndex::new(vec![]).unwrap();
                let mut notified = 0;
                let inspect = |g: &Group, r: &GroupResiduals| {
                    accept(g, r)?;
                    if g.id == stop {
                        if cancel_inside {
                            cancel.cancel();
                            Ok(())
                        } else {
                            Err(fail())
                        }
                    } else {
                        Ok(())
                    }
                };
                let output = |_: &[Observation]| {
                    notified += 1;
                    Ok(())
                };
                let outcome = match engine {
                    0 => simulation::simulate_groups_with_residuals(
                        &model,
                        hard.as_slice(),
                        &mut resident,
                        groups(),
                        &Simple,
                        options(&budget, &cancel),
                        inspect,
                        output,
                    ),
                    1 => parallel::simulate_groups_with_residuals(
                        &model,
                        hard.as_slice(),
                        &mut resident,
                        groups(),
                        &Simple,
                        options(&budget, &cancel),
                        ExecutionOptions {
                            batch_groups: 4,
                            maximum_targets_per_group: 16,
                            bytes_per_group: 16 << 20,
                        },
                        inspect,
                        output,
                    ),
                    _ => simulation::simulate_checkpointed_with_residuals(
                        &model,
                        hard.as_slice(),
                        &mut store,
                        groups(),
                        &Simple,
                        CheckpointedOptions {
                            simulation: options(&budget, &cancel),
                            identity: [41; 32],
                        },
                        policy,
                        inspect,
                        output,
                    ),
                };
                assert!(outcome.is_err());
                assert_eq!(notified, stop as usize / 3);
                let accepted: usize = groups()
                    .iter()
                    .filter(|g| g.id < stop)
                    .map(|g| g.targets.len())
                    .sum();
                assert_eq!(
                    if engine == 2 {
                        values(&store)
                    } else {
                        values(&resident)
                    }
                    .len(),
                    accepted
                );
                assert_eq!(budget.used(), 0);
                if engine == 2 {
                    assert_eq!(store.commits, stop as usize / 3);
                    let cancel = Cancellation::default();
                    let before = values(&store);
                    if store.checkpoint.is_some() {
                        let changed = identity::opaque("residual policy", b"v2 changed").unwrap();
                        assert!(simulation::simulate_checkpointed_with_residuals(
                            &model,
                            hard.as_slice(),
                            &mut store,
                            groups(),
                            &Simple,
                            CheckpointedOptions {
                                simulation: options(&budget, &cancel),
                                identity: [41; 32]
                            },
                            changed,
                            accept,
                            |_| Ok(())
                        )
                        .is_err());
                        assert_eq!(values(&store), before);
                        assert!(simulation::simulate_checkpointed(
                            &model,
                            hard.as_slice(),
                            &mut store,
                            groups(),
                            &Simple,
                            CheckpointedOptions {
                                simulation: options(&budget, &cancel),
                                identity: [41; 32]
                            },
                            |_| Ok(())
                        )
                        .is_err());
                    }
                    let mut checked = Vec::new();
                    simulation::simulate_checkpointed_with_residuals(
                        &model,
                        hard.as_slice(),
                        &mut store,
                        groups(),
                        &Simple,
                        CheckpointedOptions {
                            simulation: options(&budget, &cancel),
                            identity: [41; 32],
                        },
                        policy,
                        |g, r| {
                            checked.push(g.id);
                            accept(g, r)
                        },
                        |_| Ok(()),
                    )
                    .unwrap();
                    assert_eq!(
                        checked,
                        groups()
                            .into_iter()
                            .filter(|g| g.id >= stop)
                            .map(|g| g.id)
                            .collect::<Vec<_>>()
                    );
                    let mut expected = SpatialIndex::new(vec![]).unwrap();
                    simulation::simulate_groups(
                        &model,
                        hard.as_slice(),
                        &mut expected,
                        groups(),
                        &Simple,
                        options(&budget, &cancel),
                        |_| Ok(()),
                    )
                    .unwrap();
                    assert_eq!(values(&store), values(&expected));
                }
            }
        }
    }
}
struct Broken(Lmc);
impl Covariance for Broken {
    fn variables(&self) -> usize {
        2
    }
    fn point_covariance(&self, a: Point3, va: usize, b: Point3, vb: usize) -> Result<f64> {
        self.0.point_covariance(a, va, b, vb)
    }
    fn covariance_column(&self, obs: &[Observation], b: &Site, out: &mut [f64]) -> Result<()> {
        self.0.covariance_column(obs, b, out)?;
        for v in out {
            *v *= 0.99;
        }
        Ok(())
    }
}
#[test]
fn independent_reports_detect_wrong_optimized_kernel_before_store_mutation() {
    let model = Broken(model());
    let data = hard();
    let budget = MemoryBudget::new(256 << 20);
    let cancel = Cancellation::default();
    let path = vec![Group {
        id: 0,
        targets: vec![Target {
            id: 100,
            site: site(87),
        }],
    }];
    for parallel in [false, true] {
        let mut store = SpatialIndex::new(vec![]).unwrap();
        let mut inspected = 0;
        let inspect = |_: &Group, r: &GroupResiduals| {
            inspected += 1;
            assert!(
                r.equations
                    .covariance_equations
                    .max_componentwise_backward_error
                    > 1e-4
            );
            Err(fail())
        };
        let result = if parallel {
            parallel::simulate_groups_with_residuals(
                &model,
                data.as_slice(),
                &mut store,
                path.clone(),
                &Simple,
                options(&budget, &cancel),
                ExecutionOptions {
                    batch_groups: 2,
                    maximum_targets_per_group: 1,
                    bytes_per_group: 16 << 20,
                },
                inspect,
                |_| panic!("rejected group emitted"),
            )
        } else {
            simulation::simulate_groups_with_residuals(
                &model,
                data.as_slice(),
                &mut store,
                path.clone(),
                &Simple,
                options(&budget, &cancel),
                inspect,
                |_| panic!("rejected group emitted"),
            )
        };
        assert!(result.is_err());
        assert_eq!(inspected, 1);
        assert!(values(&store).is_empty());
        assert_eq!(budget.used(), 0);
    }
}

#[cfg(feature = "sqlite")]
#[test]
fn durable_checked_resume_binds_policy_and_preserves_post_commit_notification_semantics() {
    use terrustrial_core::{simulation::identity::RunProvenance, storage::SqliteStore};
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("resume.db");
    let budget = MemoryBudget::new(128 << 20);
    let storage_budget = MemoryBudget::new(16 << 20);
    let cancel = Cancellation::default();
    let model = model();
    let hard = hard();
    let policy = identity::opaque("residual policy", b"physical scaled residual <= 1e-10").unwrap();
    let provenance = RunProvenance {
        implementation: identity::opaque("implementation", b"test").unwrap(),
        backend: identity::opaque("backend", b"f64").unwrap(),
        mean: None,
    };
    let mut store = SqliteStore::create(&path, 4096, &storage_budget).unwrap();
    // Inspection rejects the third group. Exactly two complete groups survive.
    assert!(simulation::simulate_checkpointed_identified_with_residuals(
        &model,
        hard.as_slice(),
        &mut store,
        || Ok(groups().into_iter().map(Ok)),
        &Simple,
        options(&budget, &cancel),
        provenance,
        policy,
        |g, r| {
            accept(g, r)?;
            if g.id == 6 {
                Err(fail())
            } else {
                Ok(())
            }
        },
        |_| Ok(())
    )
    .is_err());
    assert_eq!(store.checkpoint().unwrap().unwrap().completed_group, 3);
    assert_eq!(values(&store).len(), 3);
    drop(store);
    assert_eq!(storage_budget.used(), 0);
    let mut store = SqliteStore::open(&path, 4096, &storage_budget).unwrap();
    let changed = identity::opaque("residual policy", b"different tolerance").unwrap();
    assert!(simulation::simulate_checkpointed_identified_with_residuals(
        &model,
        hard.as_slice(),
        &mut store,
        || Ok(groups().into_iter().map(Ok)),
        &Simple,
        options(&budget, &cancel),
        provenance,
        changed,
        accept,
        |_| Ok(())
    )
    .is_err());
    assert_eq!(values(&store).len(), 3);
    // Notification failure is after durability, so the resumed third group stays.
    let mut inspected = Vec::new();
    assert!(simulation::simulate_checkpointed_identified_with_residuals(
        &model,
        hard.as_slice(),
        &mut store,
        || Ok(groups().into_iter().map(Ok)),
        &Simple,
        options(&budget, &cancel),
        provenance,
        policy,
        |g, r| {
            inspected.push(g.id);
            accept(g, r)
        },
        |_| Err(fail())
    )
    .is_err());
    assert_eq!(inspected, vec![6]);
    assert_eq!(store.checkpoint().unwrap().unwrap().completed_group, 6);
    drop(store);
    let mut store = SqliteStore::open(&path, 4096, &storage_budget).unwrap();
    let mut inspected = Vec::new();
    simulation::simulate_checkpointed_identified_with_residuals(
        &model,
        hard.as_slice(),
        &mut store,
        || Ok(groups().into_iter().map(Ok)),
        &Simple,
        options(&budget, &cancel),
        provenance,
        policy,
        |g, r| {
            inspected.push(g.id);
            accept(g, r)
        },
        |_| Ok(()),
    )
    .unwrap();
    assert_eq!(inspected, vec![9, 12, 15]);
    let mut expected = SpatialIndex::new(vec![]).unwrap();
    simulation::simulate_groups(
        &model,
        hard.as_slice(),
        &mut expected,
        groups(),
        &Simple,
        options(&budget, &cancel),
        |_| Ok(()),
    )
    .unwrap();
    assert_eq!(values(&store), values(&expected));
    assert_eq!(budget.used(), 0);
    drop(store);
    assert_eq!(storage_budget.used(), 0);
}

#[test]
fn diagnostic_evaluation_failures_and_tight_budgets_leave_no_group_and_allow_retry() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    struct Fault<'a> {
        model: Lmc,
        calls: AtomicUsize,
        after: usize,
        kind: usize,
        cancel: &'a Cancellation,
    }
    impl Covariance for Fault<'_> {
        fn variables(&self) -> usize {
            2
        }
        fn point_covariance(&self, a: Point3, va: usize, b: Point3, vb: usize) -> Result<f64> {
            if self.calls.fetch_add(1, Ordering::Relaxed) >= self.after {
                match self.kind {
                    0 => return Err(fail()),
                    1 => return Ok(f64::NAN),
                    _ => self.cancel.cancel(),
                }
            }
            self.model.point_covariance(a, va, b, vb)
        }
    }
    let hard = hard();
    let path = vec![Group {
        id: 0,
        targets: vec![Target {
            id: 100,
            site: site(87),
        }],
    }];
    let cancel = Cancellation::default();
    let budget = MemoryBudget::new(256 << 20);
    let baseline = Fault {
        model: model(),
        calls: AtomicUsize::new(0),
        after: usize::MAX,
        kind: 0,
        cancel: &cancel,
    };
    let mut expected = SpatialIndex::new(vec![]).unwrap();
    simulation::simulate_groups(
        &baseline,
        hard.as_slice(),
        &mut expected,
        path.clone(),
        &Simple,
        options(&budget, &cancel),
        |_| Ok(()),
    )
    .unwrap();
    let prepare_calls = baseline.calls.load(Ordering::Relaxed);
    let policy = identity::opaque("policy", b"reject failed diagnostics").unwrap();
    for kind in 0..3 {
        for engine in 0..3 {
            let cancel = Cancellation::default();
            let fault = Fault {
                model: model(),
                calls: AtomicUsize::new(0),
                after: prepare_calls,
                kind,
                cancel: &cancel,
            };
            let mut store = MemoryStore::default();
            let mut resident = SpatialIndex::new(vec![]).unwrap();
            let inspect = |_: &Group, _: &GroupResiduals| -> Result<()> {
                panic!("failed diagnostics must not be emitted")
            };
            let sink =
                |_: &[Observation]| -> Result<()> { panic!("failed group must not be emitted") };
            let r = match engine {
                0 => simulation::simulate_groups_with_residuals(
                    &fault,
                    hard.as_slice(),
                    &mut resident,
                    path.clone(),
                    &Simple,
                    options(&budget, &cancel),
                    inspect,
                    sink,
                ),
                1 => parallel::simulate_groups_with_residuals(
                    &fault,
                    hard.as_slice(),
                    &mut resident,
                    path.clone(),
                    &Simple,
                    options(&budget, &cancel),
                    ExecutionOptions {
                        batch_groups: 1,
                        maximum_targets_per_group: 1,
                        bytes_per_group: 16 << 20,
                    },
                    inspect,
                    sink,
                ),
                _ => simulation::simulate_checkpointed_with_residuals(
                    &fault,
                    hard.as_slice(),
                    &mut store,
                    path.clone(),
                    &Simple,
                    CheckpointedOptions {
                        simulation: options(&budget, &cancel),
                        identity: [1; 32],
                    },
                    policy,
                    inspect,
                    sink,
                ),
            };
            assert!(r.is_err());
            assert!(values(&resident).is_empty());
            assert!(store.rows.is_empty());
            assert_eq!(store.commits, 0);
            assert_eq!(budget.used(), 0);
            assert!(fault.calls.load(Ordering::Relaxed) > prepare_calls);
            let cancel = Cancellation::default();
            simulation::simulate_groups_with_residuals(
                &model(),
                hard.as_slice(),
                &mut resident,
                path.clone(),
                &Simple,
                options(&budget, &cancel),
                accept,
                |_| Ok(()),
            )
            .unwrap();
            assert_eq!(values(&resident), values(&expected));
        }
    }
    // Fixed budget ladder exercises failures at multiple admission/growth stages.
    let mut failures = 0;
    let mut successes = 0;
    for limit in [0, 64, 512, 4096, 16384, 65536, 262144, 1048576, 4194304] {
        let b = MemoryBudget::new(limit);
        let mut store = SpatialIndex::new(vec![]).unwrap();
        let r = simulation::simulate_groups_with_residuals(
            &model(),
            hard.as_slice(),
            &mut store,
            path.clone(),
            &Simple,
            options(&b, &cancel),
            accept,
            |_| Ok(()),
        );
        match r {
            Ok(()) => {
                successes += 1;
                assert_eq!(values(&store), values(&expected));
            }
            Err(Error::MemoryBudget { .. }) => {
                failures += 1;
                assert!(values(&store).is_empty());
            }
            other => panic!("unexpected result {other:?}"),
        }
        assert_eq!(b.used(), 0);
    }
    assert!(failures >= 3 && successes >= 1);
}

#[cfg(feature = "sqlite")]
#[test]
#[ignore = "subprocess entry point exercised by checked_process_exit_recovers_only_accepted_commits"]
fn checked_exit_child() {
    use terrustrial_core::storage::SqliteStore;
    let path = std::env::var("TERRUSTRIAL_CHECKED_EXIT_DATABASE").unwrap();
    let phase = std::env::var("TERRUSTRIAL_CHECKED_EXIT_PHASE").unwrap();
    let budget = MemoryBudget::new(128 << 20);
    let cancel = Cancellation::default();
    let mut store = SqliteStore::open(path, 4096, &budget).unwrap();
    let policy = identity::opaque("exit policy", b"v1").unwrap();
    let mut notifications = 0;
    simulation::simulate_checkpointed_with_residuals(
        &model(),
        hard().as_slice(),
        &mut store,
        groups(),
        &Simple,
        CheckpointedOptions {
            simulation: options(&budget, &cancel),
            identity: [49; 32],
        },
        policy,
        |g, r| {
            accept(g, r)?;
            if phase == "inspect" && g.id == 6 {
                std::process::exit(79);
            }
            Ok(())
        },
        |_| {
            notifications += 1;
            if phase == "notification" && notifications == 3 {
                std::process::exit(79);
            }
            Ok(())
        },
    )
    .unwrap();
    panic!("child did not exit at the requested point");
}
#[cfg(feature = "sqlite")]
#[test]
fn checked_process_exit_recovers_only_accepted_commits() {
    use terrustrial_core::storage::SqliteStore;
    for phase in ["inspect", "notification"] {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("exit.db");
        let budget = MemoryBudget::new(128 << 20);
        let cancel = Cancellation::default();
        drop(SqliteStore::create(&path, 4096, &budget).unwrap());
        let exit = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--ignored", "--exact", "checked_exit_child"])
            .env("TERRUSTRIAL_CHECKED_EXIT_DATABASE", &path)
            .env("TERRUSTRIAL_CHECKED_EXIT_PHASE", phase)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .unwrap();
        assert_eq!(exit.code(), Some(79));
        let mut store = SqliteStore::open(path, 4096, &budget).unwrap();
        let last = if phase == "inspect" { 3 } else { 6 };
        assert_eq!(store.checkpoint().unwrap().unwrap().completed_group, last);
        let prefix = groups()
            .iter()
            .filter(|g| g.id <= last)
            .map(|g| g.targets.len())
            .sum::<usize>();
        assert_eq!(values(&store).len(), prefix);
        let policy = identity::opaque("exit policy", b"v1").unwrap();
        simulation::simulate_checkpointed_with_residuals(
            &model(),
            hard().as_slice(),
            &mut store,
            groups(),
            &Simple,
            CheckpointedOptions {
                simulation: options(&budget, &cancel),
                identity: [49; 32],
            },
            policy,
            accept,
            |_| Ok(()),
        )
        .unwrap();
        let mut expected = SpatialIndex::new(vec![]).unwrap();
        simulation::simulate_groups(
            &model(),
            hard().as_slice(),
            &mut expected,
            groups(),
            &Simple,
            options(&budget, &cancel),
            |_| Ok(()),
        )
        .unwrap();
        assert_eq!(values(&store), values(&expected));
        drop(store);
        assert_eq!(budget.used(), 0);
    }
}

#[test]
fn negligible_far_field_products_are_reported_and_strict_policy_rejects_before_writing() {
    // Reproduces the original unbounded-radius resource fixture at group 10783.
    // A small residual alone must not hide nonzero diagnostic underflow counts.
    let model = Lmc::new(vec![Structure {
        correlation: Correlation::Exponential,
        anisotropy: Anisotropy::isotropic(2.).unwrap(),
        coregionalization: Coregionalization::from_factor(2, 2, &[1., 0., 0.4, 0.9]).unwrap(),
    }])
    .unwrap();
    let mut data = Vec::new();
    let make = |id, x, v| {
        Observation::new(
            id,
            Site::point(Point3::new([x, 0., 0.]).unwrap(), v),
            0.,
            0.,
        )
        .unwrap()
    };
    data.push(make(0, -1., 0));
    data.push(make(1, -1., 1));
    for i in 10767..10783 {
        for v in 0..2 {
            data.push(make(10 + i * 2 + v, i as f64 / 16., v as usize));
        }
    }
    let path = vec![Group {
        id: 0,
        targets: (0..2)
            .map(|v| Target {
                id: 100000 + v as u64,
                site: Site::point(Point3::new([10783. / 16., 0., 0.]).unwrap(), v),
            })
            .collect(),
    }];
    let budget = MemoryBudget::new(16 << 20);
    let cancel = Cancellation::default();
    let mut store = SpatialIndex::new(data[2..].to_vec()).unwrap();
    let before = values(&store);
    let mut options = options(&budget, &cancel);
    options.hard_neighborhood = Neighborhood {
        maximum: 2,
        radius: 1e12,
    }
    .into();
    options.simulated_neighborhood = Neighborhood {
        maximum: 32,
        radius: 1.1,
    }
    .into();
    let mut inspected = false;
    assert!(simulation::simulate_groups_with_residuals(
        &model,
        &data[..2],
        &mut store,
        path,
        &Simple,
        options,
        |_, r| {
            inspected = true;
            assert!(
                r.equations
                    .covariance_equations
                    .max_componentwise_backward_error
                    < 1e-12
            );
            assert!(r.equations.covariance_equations.scaled_product_underflows > 0);
            Err(fail())
        },
        |_| panic!("rejected group emitted")
    )
    .is_err());
    assert!(inspected);
    assert_eq!(values(&store), before);
    assert_eq!(budget.used(), 0);
}
