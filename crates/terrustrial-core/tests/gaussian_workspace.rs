use terrustrial_core::{
    data::{Observation, Point3, Site},
    execution::{Cancellation, MemoryBudget},
    gaussian::{
        Drift, ExternalDrift, GaussianPlan, GaussianWorkspace, NumericalTolerance, Ordinary,
        PrepareOptions, Simple, Uncertainty, WorkspaceOptions,
    },
    model::{Anisotropy, Coregionalization, Correlation, Lmc, Structure},
    Error,
};
fn site(x: f64, variable: usize) -> Site {
    Site::point(Point3::new([x, 0., 0.]).unwrap(), variable)
}
fn model() -> Lmc {
    Lmc::new(vec![Structure {
        correlation: Correlation::Exponential,
        anisotropy: Anisotropy::isotropic(3.).unwrap(),
        coregionalization: Coregionalization::from_factor(2, 2, &[1., 0., 0.6, 0.8]).unwrap(),
    }])
    .unwrap()
}
fn data() -> Vec<Observation> {
    (0..32)
        .map(|i| {
            Observation::new(
                i,
                site((i / 2) as f64 * 0.3, (i % 2) as usize),
                (i as f64 * 0.1).sin(),
                0.,
            )
            .unwrap()
        })
        .collect()
}
fn options(cancel: &Cancellation, uncertainty: Uncertainty) -> WorkspaceOptions<'_> {
    WorkspaceOptions {
        uncertainty,
        tolerance: NumericalTolerance::default(),
        cancellation: cancel,
    }
}

#[test]
fn mean_prediction_into_checks_dimensions_and_preserves_exact_residual_formula() {
    let budget = MemoryBudget::new(1 << 20);
    let cancel = Cancellation::default();
    let observations = [Observation::new(0, site(0., 0), 5., 0.).unwrap()];
    let mut workspace = GaussianWorkspace::new(1, 1, 0, &budget).unwrap();
    let plan = workspace
        .prepare(
            &model(),
            &observations,
            &[site(0., 0)],
            &Simple,
            options(&cancel, Uncertainty::None),
        )
        .unwrap();
    let mut out = [0.];
    let counts = allocation_counter::measure(|| {
        for _ in 0..100 {
            plan.estimate_with_mean_into(&[5.], &[2.], &[10.], &mut out)
                .unwrap();
            assert_eq!(out, [13.]);
        }
    });
    assert_eq!(counts.count_total, 0);
    for (values, means, targets) in [
        (vec![], vec![2.], vec![10.]),
        (vec![5.], vec![], vec![10.]),
        (vec![5.], vec![2.], vec![]),
        (vec![f64::NAN], vec![2.], vec![10.]),
        (vec![5.], vec![f64::INFINITY], vec![10.]),
        (vec![5.], vec![2.], vec![f64::NAN]),
    ] {
        assert!(plan
            .estimate_with_mean_into(&values, &means, &targets, &mut out)
            .is_err());
    }
    assert!(plan
        .estimate_with_mean_into(&[5.], &[2.], &[10.], &mut [])
        .is_err());
    assert!(plan
        .estimate_with_mean_into(&[f64::MAX], &[-f64::MAX], &[0.], &mut out)
        .is_err());
}

#[test]
fn reuse_across_shapes_and_drifts_matches_fresh_plans_without_allocating() {
    let budget = MemoryBudget::new(1 << 24);
    let cancel = Cancellation::default();
    let model = model();
    let observations = data();
    let targets = [
        site(0., 0),
        site(0.17, 1),
        site(1.37, 0),
        site(1.73, 1),
        site(5., 0),
    ];
    let mut workspace = GaussianWorkspace::new(32, 5, 3, &budget).unwrap();
    let drift = ExternalDrift::new(3, |site: &Site, b: &mut [f64]| {
        b[0] = (site.variable == 0) as u8 as f64;
        b[1] = (site.variable == 1) as u8 as f64;
        b[2] = site.support.center().coordinates()[0];
        Ok(())
    })
    .unwrap();
    let mut exercise = |n: usize, m: usize, drift: &dyn Drift, joint| {
        // Adapter keeps the public generic API monomorphized with a sized value.
        struct Borrowed<'a>(&'a dyn Drift);
        impl Drift for Borrowed<'_> {
            fn columns(&self) -> usize {
                self.0.columns()
            }
            fn evaluate(&self, s: &Site, b: &mut [f64]) -> terrustrial_core::Result<()> {
                self.0.evaluate(s, b)
            }
        }
        let drift = Borrowed(drift);
        let plan = workspace
            .prepare(
                &model,
                &observations[..n],
                &targets[..m],
                &drift,
                options(&cancel, joint),
            )
            .unwrap();
        let mut values = [0.; 32];
        for (v, o) in values.iter_mut().zip(&observations) {
            *v = o.value();
        }
        let mut prediction = [0.; 5];
        let mut noise = [0.; 5];
        let mut draw = [0.; 5];
        plan.estimate_into(&values[..n], &mut prediction[..m])
            .unwrap();
        if joint == Uncertainty::Joint {
            plan.sample_into(&prediction[..m], 77, 1, 4, &mut noise[..m], &mut draw[..m])
                .unwrap();
        }
        std::hint::black_box((prediction, draw));
    };
    // First pass includes shrinking/growing initialized regions, not just repeats
    // of identical dimensions. The measurement excludes only constructor/inputs.
    let allocations = allocation_counter::measure(|| {
        for _ in 0..100 {
            exercise(32, 5, &Simple, Uncertainty::Joint);
            exercise(16, 2, &Ordinary { variables: 2 }, Uncertainty::Joint);
            exercise(18, 3, &drift, Uncertainty::None);
            exercise(0, 4, &Simple, Uncertainty::Joint);
            exercise(1, 1, &Simple, Uncertainty::Joint);
        }
    });
    assert_eq!(allocations.count_total, 0, "{allocations:?}");
    assert_eq!(allocations.bytes_total, 0);
    let allocated = allocation_counter::measure(|| {
        for _ in 0..100 {
            let plan = GaussianPlan::prepare(
                &model,
                &observations,
                &targets,
                &Simple,
                PrepareOptions {
                    uncertainty: Uncertainty::Joint,
                    tolerance: NumericalTolerance::default(),
                    budget: &budget,
                    cancellation: &cancel,
                },
            )
            .unwrap();
            std::hint::black_box(plan.weights());
        }
    });
    assert!(allocated.count_total > 0);
    eprintln!("workspace: 500 preparations/predictions/draws, {} allocations, {} bytes; fresh: 100 preparations, {} allocations, {} bytes", allocations.count_total, allocations.bytes_total, allocated.count_total, allocated.bytes_total);

    // Compare reused buffers with clean preparations after the stress sequence.
    for n in [0, 1, 7, 32, 2, 18] {
        for m in [1, 5, 2] {
            let plan = workspace
                .prepare(
                    &model,
                    &observations[..n],
                    &targets[..m],
                    &Simple,
                    options(&cancel, Uncertainty::Joint),
                )
                .unwrap();
            let fresh = GaussianPlan::prepare(
                &model,
                &observations[..n],
                &targets[..m],
                &Simple,
                PrepareOptions {
                    uncertainty: Uncertainty::Joint,
                    tolerance: NumericalTolerance::default(),
                    budget: &budget,
                    cancellation: &cancel,
                },
            )
            .unwrap();
            for j in 0..m {
                for i in 0..n {
                    assert_eq!(
                        plan.weights()[(i, j)].to_bits(),
                        fresh.weights()[(i, j)].to_bits()
                    );
                }
            }
            for i in 0..m {
                for j in 0..m {
                    assert_eq!(
                        plan.covariance().unwrap()[(i, j)].to_bits(),
                        fresh.covariance().unwrap()[(i, j)].to_bits()
                    );
                }
            }
            let values = observations[..n]
                .iter()
                .map(Observation::value)
                .collect::<Vec<_>>();
            let mean = plan.estimate(&values).unwrap();
            let mut noise = vec![0.; m];
            let mut out = vec![0.; m];
            plan.sample_into(&mean, 21, 3, 1, &mut noise, &mut out)
                .unwrap();
            assert_eq!(out, fresh.sample(&mean, 21, 3, 1).unwrap());
        }
    }
    drop(workspace);
    assert_eq!(budget.used(), 0);
}
#[test]
fn workspace_recovers_after_failures_and_keeps_zero_variance_deterministic() {
    let budget = MemoryBudget::new(1 << 20);
    let cancel = Cancellation::default();
    let model = model();
    let mut workspace = GaussianWorkspace::new(32, 5, 3, &budget).unwrap();
    let observations = data();
    let targets = [site(0., 0), site(0.7, 1)];
    let mut bad = observations[..2].to_vec();
    bad[1] = bad[0];
    assert!(workspace
        .prepare(
            &model,
            &bad,
            &targets,
            &Simple,
            options(&cancel, Uncertainty::Joint)
        )
        .is_err());
    bad[1] = Observation::new(99, *bad[0].site(), 1., 0.).unwrap();
    assert!(matches!(
        workspace.prepare(
            &model,
            &bad,
            &targets,
            &Simple,
            options(&cancel, Uncertainty::Joint)
        ),
        Err(Error::Singular { .. })
    ));
    let invalid = ExternalDrift::new(1, |_: &Site, _: &mut [f64]| Ok(())).unwrap();
    assert!(workspace
        .prepare(
            &model,
            &observations,
            &targets,
            &invalid,
            options(&cancel, Uncertainty::Joint)
        )
        .is_err());
    let plan = workspace
        .prepare(
            &model,
            &observations,
            &targets,
            &Simple,
            options(&cancel, Uncertainty::Joint),
        )
        .unwrap();
    assert_eq!(plan.covariance().unwrap()[(0, 0)], 0.);
    let mean = plan
        .estimate(
            &observations
                .iter()
                .map(Observation::value)
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let mut noise = [0.; 2];
    let mut out = [0.; 2];
    plan.sample_into(&mean, 1, 1, 1, &mut noise, &mut out)
        .unwrap();
    assert_eq!(out[0].to_bits(), mean[0].to_bits());
    assert!(plan.sample_into(&mean, 1, 1, 1, &mut [], &mut out).is_err());
    let plan = workspace
        .prepare(
            &model,
            &observations,
            &targets,
            &Simple,
            options(&cancel, Uncertainty::None),
        )
        .unwrap();
    assert!(plan.covariance().is_none());
    assert!(plan
        .sample_into(&mean, 1, 1, 1, &mut noise, &mut out)
        .is_err());
    cancel.cancel();
    assert!(matches!(
        workspace.prepare(
            &model,
            &observations,
            &targets,
            &Simple,
            options(&cancel, Uncertainty::Joint)
        ),
        Err(Error::Cancelled)
    ));
    drop(workspace);
    assert_eq!(budget.used(), 0);
    assert!(matches!(
        GaussianWorkspace::new(32, 5, 3, &MemoryBudget::new(1)),
        Err(Error::MemoryBudget { .. })
    ));
    assert!(GaussianWorkspace::new(0, 1, 1, &budget).is_err());
    assert!(GaussianWorkspace::new(1, 0, 0, &budget).is_err());
    let mut small = GaussianWorkspace::new(1, 1, 0, &budget).unwrap();
    assert!(small
        .prepare(
            &model,
            &observations,
            &targets,
            &Simple,
            options(&Cancellation::default(), Uncertainty::Joint)
        )
        .is_err());
}

#[test]
fn retained_plan_survives_workspace_reuse_and_charges_only_final_payload() {
    let budget = MemoryBudget::new(1 << 20);
    let retained_budget = MemoryBudget::new(1 << 20);
    let cancel = Cancellation::default();
    let model = model();
    let observations = data();
    let targets = [site(0.17, 0), site(1.73, 1)];
    let mut workspace = GaussianWorkspace::new(32, 5, 3, &budget).unwrap();
    let plan = workspace
        .prepare(
            &model,
            &observations,
            &targets,
            &Ordinary { variables: 2 },
            options(&cancel, Uncertainty::Joint),
        )
        .unwrap();
    let values = observations
        .iter()
        .map(Observation::value)
        .collect::<Vec<_>>();
    let expected_mean = plan.estimate(&values).unwrap();
    let expected_draw = plan.sample(&expected_mean, 2, 7, 9).unwrap();
    let retained = plan.to_owned_plan(&retained_budget).unwrap();
    assert_eq!(retained.accounted_bytes(), (32 * 2 + 2 * 2 * 2) * 8);
    let tiny = MemoryBudget::new(retained.accounted_bytes() - 1);
    assert!(matches!(
        plan.to_owned_plan(&tiny),
        Err(Error::MemoryBudget { .. })
    ));
    assert_eq!(tiny.used(), 0);
    workspace
        .prepare(
            &model,
            &[],
            &targets[..1],
            &Simple,
            options(&cancel, Uncertainty::None),
        )
        .unwrap();
    drop(workspace);
    assert_eq!(budget.used(), 0);
    assert_eq!(retained.estimate(&values).unwrap(), expected_mean);
    assert_eq!(
        retained.sample(&expected_mean, 2, 7, 9).unwrap(),
        expected_draw
    );
    drop(retained);
    assert_eq!(retained_budget.used(), 0);
    assert!(matches!(
        GaussianWorkspace::new(usize::MAX, 1, 0, &budget),
        Err(Error::SizeOverflow)
    ));
}
