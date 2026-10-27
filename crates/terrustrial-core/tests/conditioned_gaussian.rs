use std::sync::atomic::{AtomicUsize, Ordering};
use terrustrial_core::{
    data::{Observation, Point3, Site, Support},
    execution::{Cancellation, MemoryBudget},
    gaussian::{
        Drift, ExternalDrift, GaussianPlan, GaussianWorkspace, NumericalTolerance, Ordinary,
        Simple, Uncertainty, WorkspaceOptions,
    },
    model::{Anisotropy, Coregionalization, Correlation, Covariance, Lmc, Structure},
    Error, Result,
};
fn site(x: f64, variable: usize) -> Site {
    Site::point(Point3::new([x, 0.2 * x.sin(), 0.]).unwrap(), variable)
}
fn model(scale: f64) -> Lmc {
    Lmc::new(vec![Structure {
        correlation: Correlation::Exponential,
        anisotropy: Anisotropy::isotropic(scale).unwrap(),
        coregionalization: Coregionalization::from_factor(2, 2, &[1., 0., 0.6, 0.8]).unwrap(),
    }])
    .unwrap()
}
fn observations() -> Vec<Observation> {
    (0..32)
        .map(|i| {
            Observation::new(
                i,
                site((i / 2) as f64 * 0.3, (i % 2) as usize),
                (i as f64 * 0.1).sin(),
                0.02,
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
fn equal(a: &GaussianPlan, b: &GaussianPlan, values: &[f64]) {
    assert_eq!(a.weights().ncols(), b.weights().ncols());
    assert_eq!(a.weights().nrows(), b.weights().nrows());
    for j in 0..a.weights().ncols() {
        for i in 0..a.weights().nrows() {
            assert_eq!(a.weights()[(i, j)].to_bits(), b.weights()[(i, j)].to_bits());
        }
    }
    assert_eq!(a.roundoff_corrections(), b.roundoff_corrections());
    let predicted = a.estimate(values).unwrap();
    assert_eq!(predicted, b.estimate(values).unwrap());
    match (a.covariance(), b.covariance()) {
        (Some(a_cov), Some(b_cov)) => {
            for i in 0..a_cov.nrows() {
                for j in 0..a_cov.ncols() {
                    assert_eq!(a_cov[(i, j)].to_bits(), b_cov[(i, j)].to_bits());
                }
            }
            assert_eq!(
                a.sample(&predicted, 32, 1, 8).unwrap(),
                b.sample(&predicted, 32, 1, 8).unwrap()
            );
        }
        (None, None) => (),
        _ => panic!("uncertainty mismatch"),
    }
}
#[test]
fn fixed_conditioning_matches_fresh_plans_for_changing_targets_drifts_and_components() {
    let budget = MemoryBudget::new(1 << 24);
    let cancel = Cancellation::default();
    let model = model(3.);
    let component = Lmc::new(vec![Structure {
        correlation: Correlation::Exponential,
        anisotropy: Anisotropy::isotropic(3.).unwrap(),
        coregionalization: Coregionalization::from_factor(2, 2, &[0.5, 0., 0.3, 0.4]).unwrap(),
    }])
    .unwrap();
    let mut observations = observations();
    observations[7] = Observation::new(
        7,
        Site {
            support: Support::block(
                Point3::new([0.9, 0.15, 0.]).unwrap(),
                [0.1, 0.1, 0.1],
                [2, 2, 2],
            )
            .unwrap(),
            variable: 1,
        },
        1.,
        0.02,
    )
    .unwrap();
    let targets = [
        site(0., 0),
        site(0.27, 1),
        site(1.17, 0),
        Site {
            support: Support::block(
                Point3::new([1.2, 0., 0.]).unwrap(),
                [0.1, 0.1, 0.1],
                [2, 2, 2],
            )
            .unwrap(),
            variable: 1,
        },
    ];
    let trend = ExternalDrift::new(3, |s: &Site, b: &mut [f64]| {
        b[0] = (s.variable == 0) as u8 as f64;
        b[1] = (s.variable == 1) as u8 as f64;
        b[2] = s.support.center().coordinates()[0];
        Ok(())
    })
    .unwrap();
    struct Borrowed<'a>(&'a dyn Drift);
    impl Drift for Borrowed<'_> {
        fn columns(&self) -> usize {
            self.0.columns()
        }
        fn evaluate(&self, s: &Site, b: &mut [f64]) -> Result<()> {
            self.0.evaluate(s, b)
        }
    }
    let ordinary = Ordinary { variables: 2 };
    let mut reusable = GaussianWorkspace::new(32, 4, 3, &budget).unwrap();
    let mut fresh = GaussianWorkspace::new(32, 4, 3, &budget).unwrap();
    for drift in [&Simple as &dyn Drift, &ordinary, &trend] {
        let drift = Borrowed(drift);
        for n in [32, 16] {
            let values: Vec<_> = observations[..n].iter().map(Observation::value).collect();
            let mut conditioned = reusable
                .condition(&model, &observations[..n], &drift, &cancel)
                .unwrap();
            for m in [4, 1, 3, 2, 4] {
                for uncertainty in [Uncertainty::Joint, Uncertainty::None] {
                    for target_model in [&model, &component] {
                        let a = conditioned
                            .prepare_component(
                                target_model,
                                &targets[..m],
                                options(&cancel, uncertainty),
                            )
                            .unwrap();
                        let b = fresh
                            .prepare_component(
                                &model,
                                target_model,
                                &observations[..n],
                                &targets[..m],
                                &drift,
                                options(&cancel, uncertainty),
                            )
                            .unwrap();
                        equal(a, b, &values);
                    }
                }
            }
        }
    }
}

#[test]
fn conditioning_callbacks_run_once_and_reused_target_preparation_allocates_nothing() {
    struct Counted {
        model: Lmc,
        rows: AtomicUsize,
    }
    impl Covariance for Counted {
        fn variables(&self) -> usize {
            2
        }
        fn point_covariance(&self, a: Point3, av: usize, b: Point3, bv: usize) -> Result<f64> {
            self.model.point_covariance(a, av, b, bv)
        }
        fn covariance_row(&self, a: &Site, o: &[Observation], out: &mut [f64]) -> Result<()> {
            self.rows.fetch_add(1, Ordering::Relaxed);
            self.model.covariance_row(a, o, out)
        }
        fn covariance_column(&self, o: &[Observation], b: &Site, out: &mut [f64]) -> Result<()> {
            self.model.covariance_column(o, b, out)
        }
    }
    struct CountedDrift(AtomicUsize);
    impl Drift for CountedDrift {
        fn columns(&self) -> usize {
            2
        }
        fn evaluate(&self, s: &Site, b: &mut [f64]) -> Result<()> {
            self.0.fetch_add(1, Ordering::Relaxed);
            Ordinary { variables: 2 }.evaluate(s, b)
        }
    }
    let model = Counted {
        model: model(3.),
        rows: AtomicUsize::new(0),
    };
    let drift = CountedDrift(AtomicUsize::new(0));
    let observations = observations();
    let targets = [site(0.17, 0), site(0.33, 1), site(1.21, 0), site(2.31, 1)];
    let budget = MemoryBudget::new(1 << 22);
    let cancel = Cancellation::default();
    let mut workspace = GaussianWorkspace::new(32, 4, 2, &budget).unwrap();
    {
        let mut conditioned = workspace
            .condition(&model, &observations, &drift, &cancel)
            .unwrap();
        assert_eq!(model.rows.load(Ordering::Relaxed), 32);
        assert_eq!(drift.0.load(Ordering::Relaxed), 32);
        let allocations = allocation_counter::measure(|| {
            for _ in 0..100 {
                for m in [4, 1, 3, 2] {
                    for uncertainty in [Uncertainty::None, Uncertainty::Joint] {
                        std::hint::black_box(
                            conditioned
                                .prepare(&targets[..m], options(&cancel, uncertainty))
                                .unwrap()
                                .weights(),
                        );
                    }
                }
            }
        });
        assert_eq!(allocations.count_total, 0, "{allocations:?}");
        assert_eq!(model.rows.load(Ordering::Relaxed), 32);
        assert_eq!(drift.0.load(Ordering::Relaxed), 32 + 2000);
    }
    drop(workspace);
    assert_eq!(budget.used(), 0);
}

#[test]
fn failures_preserve_conditioning_and_new_sessions_rebuild_it() {
    let observations = observations();
    let targets = [site(0.3, 0), site(0.7, 1)];
    let budget = MemoryBudget::new(1 << 22);
    let cancel = Cancellation::default();
    let mut workspace = GaussianWorkspace::new(32, 2, 2, &budget).unwrap();
    let mut fresh = GaussianWorkspace::new(32, 2, 2, &budget).unwrap();
    let ordinary = Ordinary { variables: 2 };
    for model in [model(3.), model(0.5)] {
        {
            let mut conditioned = workspace
                .condition(&model, &observations, &ordinary, &cancel)
                .unwrap();
            let stopped = Cancellation::default();
            stopped.cancel();
            assert!(matches!(
                conditioned.prepare(&targets, options(&stopped, Uncertainty::Joint)),
                Err(Error::Cancelled)
            ));
            assert!(conditioned
                .prepare(&[], options(&cancel, Uncertainty::Joint))
                .is_err());
            assert!(conditioned
                .prepare(&[site(1., 2)], options(&cancel, Uncertainty::Joint))
                .is_err());
            assert!(conditioned
                .prepare(
                    &[site(1., 0), site(2., 1), site(3., 0)],
                    options(&cancel, Uncertainty::Joint)
                )
                .is_err());
            struct Bad;
            impl Covariance for Bad {
                fn variables(&self) -> usize {
                    2
                }
                fn point_covariance(
                    &self,
                    _: Point3,
                    _: usize,
                    _: Point3,
                    _: usize,
                ) -> Result<f64> {
                    Ok(f64::NAN)
                }
            }
            assert!(conditioned
                .prepare_component(&Bad, &targets, options(&cancel, Uncertainty::Joint))
                .is_err());
            let a = conditioned
                .prepare(&targets, options(&cancel, Uncertainty::Joint))
                .unwrap();
            let b = fresh
                .prepare(
                    &model,
                    &observations,
                    &targets,
                    &ordinary,
                    options(&cancel, Uncertainty::Joint),
                )
                .unwrap();
            equal(
                a,
                b,
                &observations
                    .iter()
                    .map(Observation::value)
                    .collect::<Vec<_>>(),
            );
        }
        assert!(workspace
            .condition(
                &model,
                &[observations[0], observations[0]],
                &Simple,
                &cancel
            )
            .is_err());
        let exact = [
            Observation::new(0, site(0., 0), 1., 0.).unwrap(),
            Observation::new(1, site(0., 0), 1., 0.).unwrap(),
        ];
        assert!(matches!(
            workspace.condition(&model, &exact, &Simple, &cancel),
            Err(Error::Singular { .. })
        ));
        // Unconditional simulation is the empty fixed-conditioning system.
        let mut empty = workspace.condition(&model, &[], &Simple, &cancel).unwrap();
        equal(
            empty
                .prepare(&targets, options(&cancel, Uncertainty::Joint))
                .unwrap(),
            fresh
                .prepare(
                    &model,
                    &[],
                    &targets,
                    &Simple,
                    options(&cancel, Uncertainty::Joint),
                )
                .unwrap(),
            &[],
        );
    }
}

#[test]
fn retained_exact_plans_and_callback_dimension_guards_survive_reuse() {
    struct Dimensions {
        inner: Lmc,
        variables: AtomicUsize,
    }
    impl Covariance for Dimensions {
        fn variables(&self) -> usize {
            self.variables.load(Ordering::Relaxed)
        }
        fn point_covariance(&self, a: Point3, av: usize, b: Point3, bv: usize) -> Result<f64> {
            self.inner.point_covariance(a, av, b, bv)
        }
    }
    let model = Dimensions {
        inner: model(3.),
        variables: AtomicUsize::new(2),
    };
    let data = [Observation::new(0, site(0., 0), 2., 0.).unwrap()];
    let targets = [site(0., 0), site(0.7, 1)];
    let budget = MemoryBudget::new(1 << 20);
    let retained_budget = MemoryBudget::new(1 << 20);
    let cancel = Cancellation::default();
    let mut workspace = GaussianWorkspace::new(1, 2, 0, &budget).unwrap();
    let retained;
    {
        let mut conditioned = workspace
            .condition(&model, &data, &Simple, &cancel)
            .unwrap();
        model.variables.store(3, Ordering::Relaxed);
        assert!(conditioned
            .prepare(&targets, options(&cancel, Uncertainty::Joint))
            .is_err());
        model.variables.store(2, Ordering::Relaxed);
        let plan = conditioned
            .prepare(&targets, options(&cancel, Uncertainty::Joint))
            .unwrap();
        assert_eq!(plan.covariance().unwrap()[(0, 0)], 0.);
        for value in [2., -3., 17.] {
            let prediction = plan.estimate(&[value]).unwrap();
            assert_eq!(prediction[0], value);
            assert_eq!(plan.sample(&prediction, 19, 1, 4).unwrap()[0], value);
        }
        retained = plan.to_owned_plan(&retained_budget).unwrap();
    }
    let replacement = crate::model(0.5);
    let mut conditioned = workspace
        .condition(&replacement, &[], &Simple, &cancel)
        .unwrap();
    assert!(conditioned
        .prepare(&targets, options(&cancel, Uncertainty::Joint))
        .is_ok());
    assert_eq!(retained.covariance().unwrap()[(0, 0)], 0.);
    assert_eq!(retained.estimate(&[2.]).unwrap()[0], 2.);
    drop(workspace);
    assert_eq!(budget.used(), 0);
    drop(retained);
    assert_eq!(retained_budget.used(), 0);
}
