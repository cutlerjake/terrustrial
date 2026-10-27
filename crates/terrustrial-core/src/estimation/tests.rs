use super::*;
use crate::{
    gaussian::Simple,
    model::{Anisotropy, Coregionalization, Correlation, Lmc, Structure},
};
#[test]
fn centered_prediction_reuses_values_handles_extremes_and_recovers() {
    let site = |x| Site::point(Point3::new([x, 0., 0.]).unwrap(), 0);
    let model = Lmc::new(vec![Structure {
        correlation: Correlation::Exponential,
        anisotropy: Anisotropy::isotropic(2.).unwrap(),
        coregionalization: Coregionalization::scalar(1.).unwrap(),
    }])
    .unwrap();
    let observations = [
        Observation::new(0, site(0.), f64::MAX, 0.).unwrap(),
        Observation::new(1, site(1.), f64::MAX, 0.).unwrap(),
    ];
    let group = Group {
        id: 0,
        targets: vec![Target {
            id: 2,
            site: site(2.),
        }],
    };
    let plan_budget = MemoryBudget::new(1 << 20);
    let cancel = Cancellation::default();
    let mut workspace = GaussianWorkspace::new(2, 1, 0, &plan_budget).unwrap();
    let plan = workspace
        .prepare(
            &model,
            &observations,
            &[site(2.)],
            &Simple,
            WorkspaceOptions {
                uncertainty: Uncertainty::None,
                tolerance: NumericalTolerance::default(),
                cancellation: &cancel,
            },
        )
        .unwrap();
    // One values vector and target-mean vector, with no observation-mean vector.
    let required = (2 * 8 + std::mem::size_of::<Site>() + 8) as u64;
    let budget = MemoryBudget::new(required);
    let mut slot = None;
    let buffers = EstimationBuffers::cached(&mut slot, 2, 1, true, &budget).unwrap();
    assert_eq!(budget.used(), required);
    let mean = |s: &Site| {
        Ok(if s.support.center().coordinates()[0] == 2. {
            3.
        } else {
            f64::MAX
        })
    };
    let mut output = [0.];
    let allocations = allocation_counter::measure(|| {
        for _ in 0..100 {
            buffers.fill(&observations, &group);
            buffers
                .predict(plan, &observations, Some(&mean), &mut output)
                .unwrap();
            assert_eq!(output, [3.]);
        }
    });
    assert_eq!(allocations.count_total, 0);
    let nonfinite = |_: &Site| Ok(f64::NAN);
    let residual_overflow = |_: &Site| Ok(-f64::MAX);
    let target_overflow = |s: &Site| {
        Ok(if s.support.center().coordinates()[0] == 2. {
            f64::MAX
        } else {
            0.
        })
    };
    let target_nonfinite = |s: &Site| {
        Ok(if s.support.center().coordinates()[0] == 2. {
            f64::INFINITY
        } else {
            f64::MAX
        })
    };
    let cancelled = |_: &Site| Err(Error::Cancelled);
    for bad in [
        &nonfinite as &dyn KnownMean,
        &residual_overflow,
        &target_overflow,
        &target_nonfinite,
        &cancelled,
    ] {
        buffers.fill(&observations, &group);
        assert!(buffers
            .predict(plan, &observations, Some(bad), &mut output)
            .is_err());
        buffers.fill(&observations, &group);
        buffers
            .predict(plan, &observations, Some(&mean), &mut output)
            .unwrap();
        assert_eq!(output, [3.]);
    }
    drop(slot);
    assert_eq!(budget.used(), 0);
}
#[test]
fn selection_growth_is_budgeted_and_failure_preserves_prior_contents() {
    let budget = MemoryBudget::new(2 * std::mem::size_of::<Observation>() as u64);
    let observation =
        Observation::new(0, Site::point(Point3::new([0.; 3]).unwrap(), 0), 1., 0.).unwrap();
    let mut selected = Selection::new(&budget).unwrap();
    selected.extend(std::slice::from_ref(&observation)).unwrap();
    let reserved = budget.used();
    assert!(selected.extend(&[observation, observation]).is_err());
    assert_eq!(selected.observations.len(), 1);
    assert_eq!(selected.observations[0].value(), 1.);
    assert_eq!(budget.used(), reserved);
    selected.extend(std::slice::from_ref(&observation)).unwrap();
    canonicalize_observations(&mut selected.observations, true).unwrap();
    assert_eq!(selected.observations.len(), 1);
    drop(selected);
    assert_eq!(budget.used(), 0);
}
#[test]
fn complete_driver_allocations_do_not_scale_with_group_count() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let model = Lmc::new(vec![Structure {
        correlation: Correlation::Exponential,
        anisotropy: Anisotropy::isotropic(3.).unwrap(),
        coregionalization: Coregionalization::scalar(1.).unwrap(),
    }])
    .unwrap();
    let hard: Vec<_> = (0..32)
        .map(|i| {
            Observation::new(
                i,
                Site::point(Point3::new([i as f64 * 0.2, 0., 0.]).unwrap(), 0),
                i as f64,
                0.01,
            )
            .unwrap()
        })
        .collect();
    let groups = |n| {
        (0..n)
            .map(|i| Group {
                id: i,
                targets: (0..1 + i as usize % 4)
                    .map(|j| Target {
                        id: 100 + i * 4 + j as u64,
                        site: Site::point(
                            Point3::new([0.13 + (i % 8) as f64 * 0.13 + j as f64 * 0.17, 0.1, 0.1])
                                .unwrap(),
                            0,
                        ),
                    })
                    .collect(),
            })
            .collect::<Vec<_>>()
    };
    let budget = MemoryBudget::new(1 << 26);
    let cancel = Cancellation::default();
    let mean = |s: &Site| Ok(s.support.center().coordinates()[0]);
    for uncertainty in [Uncertainty::None, Uncertainty::Joint] {
        for use_mean in [false, true] {
            let run = |groups| {
                estimate_groups(
                    &model,
                    hard.as_slice(),
                    groups,
                    &Simple,
                    EstimationOptions {
                        neighborhood: Neighborhood {
                            maximum: 32,
                            radius: 100.,
                        }
                        .into(),
                        mean: if use_mean { Some(&mean) } else { None },
                        uncertainty,
                        tolerance: NumericalTolerance::default(),
                        budget: &budget,
                        cancellation: &cancel,
                    },
                    ExecutionOptions {
                        batch_groups: 4,
                        maximum_targets_per_group: 4,
                        bytes_per_group: 1 << 20,
                    },
                    |_| Ok(()),
                )
            };
            pool.install(|| run(groups(16))).unwrap();
            let small = groups(16);
            let large = groups(256);
            let a = pool.install(|| allocation_counter::measure(|| run(small).unwrap()));
            let b = pool.install(|| allocation_counter::measure(|| run(large).unwrap()));
            assert_eq!(
                a.count_total, b.count_total,
                "{uncertainty:?}, mean={use_mean}"
            );
            assert_eq!(a.bytes_total, b.bytes_total);
            assert_eq!(budget.used(), 0);
        }
    }
}
#[test]
fn warmed_estimation_preparation_allocates_nothing_including_output() {
    warmed_preparation::<Unchecked>();
    warmed_preparation::<diagnostics::Checked>();
}
fn warmed_preparation<M: Mode>() {
    let model = Lmc::new(vec![Structure {
        correlation: Correlation::Exponential,
        anisotropy: Anisotropy::isotropic(3.).unwrap(),
        coregionalization: Coregionalization::scalar(1.).unwrap(),
    }])
    .unwrap();
    let hard: Vec<_> = (0..32)
        .map(|i| {
            Observation::new(
                i,
                Site::point(Point3::new([i as f64 * 0.2, 0., 0.]).unwrap(), 0),
                i as f64,
                0.01,
            )
            .unwrap()
        })
        .collect();
    let groups: Vec<_> = (0..8)
        .map(|i| Group {
            id: i,
            targets: (0..1 + i as usize % 4)
                .map(|j| Target {
                    id: 100 + i * 4 + j as u64,
                    site: Site::point(
                        Point3::new([0.13 + i as f64 * 0.13 + j as f64 * 0.17, 0.1, 0.1]).unwrap(),
                        0,
                    ),
                })
                .collect(),
        })
        .collect();
    let query: NeighborhoodQuery<'_> = Neighborhood {
        maximum: 32,
        radius: 100.,
    }
    .into();
    let cancel = Cancellation::default();
    let parent = MemoryBudget::new(1 << 24);
    let trend = |s: &Site| Ok(s.support.center().coordinates()[0].powi(2));
    for honor_exact in [true, false] {
        for uncertainty in [Uncertainty::None, Uncertainty::Joint] {
            for use_mean in [false, true] {
                let mut worker = Worker::new(query, 1 << 20).unwrap();
                let mut slot = Estimated::<M>::new(MemoryBudget::new(1 << 20)).unwrap();
                let mut diagnostic = M::default();
                let options = EstimationOptions {
                    neighborhood: query,
                    mean: if use_mean { Some(&trend) } else { None },
                    uncertainty,
                    tolerance: NumericalTolerance::default(),
                    budget: &parent,
                    cancellation: &cancel,
                };
                let mut conditioning =
                    NeighborhoodConditioning::new(&model, &Simple, &worker.budget).unwrap();
                // Warm the largest target and conditioning capacities first.
                prepare_estimate(
                    hard.as_slice(),
                    &groups[3],
                    &options,
                    &factorial::Component {
                        covariance: &model,
                        drift: &Simple,
                        mean: options.mean,
                    },
                    honor_exact,
                    &mut worker,
                    &mut conditioning,
                    &mut slot,
                    &mut diagnostic,
                )
                .unwrap();
                let count = allocation_counter::measure(|| {
                    for _ in 0..100 {
                        for group in &groups {
                            prepare_estimate(
                                hard.as_slice(),
                                group,
                                &options,
                                &factorial::Component {
                                    covariance: &model,
                                    drift: &Simple,
                                    mean: options.mean,
                                },
                                honor_exact,
                                &mut worker,
                                &mut conditioning,
                                &mut slot,
                                &mut diagnostic,
                            )
                            .unwrap();
                            assert_eq!(slot.values.len(), group.targets.len());
                            assert_eq!(
                                slot.uncertainty.covariance.as_ref().map(|c| c.ncols()),
                                if uncertainty == Uncertainty::Joint {
                                    Some(group.targets.len())
                                } else {
                                    None
                                }
                            );
                        }
                    }
                });
                assert_eq!(
                    count.count_total, 0,
                    "{uncertainty:?}, mean={use_mean}: {count:?}"
                );
                assert_eq!(count.bytes_total, 0);
            }
        }
    }
}
#[test]
fn retained_uncertainty_needs_no_weights_or_sampling_factor() {
    let model = Lmc::new(vec![Structure {
        correlation: Correlation::Exponential,
        anisotropy: Anisotropy::isotropic(3.).unwrap(),
        coregionalization: Coregionalization::scalar(1.).unwrap(),
    }])
    .unwrap();
    let observations: Vec<_> = (0..8)
        .map(|i| {
            Observation::new(
                i,
                Site::point(Point3::new([i as f64, 0., 0.]).unwrap(), 0),
                0.,
                0.01,
            )
            .unwrap()
        })
        .collect();
    let targets = [
        Site::point(Point3::new([0.3, 0., 0.]).unwrap(), 0),
        Site::point(Point3::new([0.7, 0., 0.]).unwrap(), 0),
    ];
    let budget = MemoryBudget::new(1 << 20);
    let cancel = Cancellation::default();
    let mut workspace = GaussianWorkspace::new(8, 2, 0, &budget).unwrap();
    let options = |uncertainty| WorkspaceOptions {
        uncertainty,
        tolerance: NumericalTolerance::default(),
        cancellation: &cancel,
    };
    let plan = workspace
        .prepare(
            &model,
            &observations,
            &targets,
            &Simple,
            options(Uncertainty::None),
        )
        .unwrap();
    let empty = MemoryBudget::new(0);
    let allocation = allocation_counter::measure(|| {
        let retained = RetainedUncertainty::from_plan(plan, &empty).unwrap();
        assert!(retained.covariance.is_none());
    });
    assert_eq!(allocation.count_total, 0);
    let plan = workspace
        .prepare(
            &model,
            &observations,
            &targets,
            &Simple,
            options(Uncertainty::Joint),
        )
        .unwrap();
    let covariance_budget = MemoryBudget::new(2 * 2 * 8);
    assert!(plan.to_owned_plan(&covariance_budget).is_err());
    assert!(RetainedUncertainty::from_plan(plan, &MemoryBudget::new(31)).is_err());
    let retained = RetainedUncertainty::from_plan(plan, &covariance_budget).unwrap();
    assert_eq!(covariance_budget.used(), 32);
    for i in 0..2 {
        for j in 0..2 {
            assert_eq!(
                retained.covariance.as_ref().unwrap()[(i, j)].to_bits(),
                plan.covariance().unwrap()[(i, j)].to_bits()
            );
        }
    }
    let value = retained.covariance.as_ref().unwrap()[(0, 0)].to_bits();
    workspace
        .prepare(
            &model,
            &[],
            &targets[..1],
            &Simple,
            options(Uncertainty::None),
        )
        .unwrap();
    drop(workspace);
    assert_eq!(
        retained.covariance.as_ref().unwrap()[(0, 0)].to_bits(),
        value
    );
    assert_eq!(budget.used(), 0);
    drop(retained);
    assert_eq!(covariance_budget.used(), 0);
}
#[test]
fn input_arena_reuses_capacity_and_recovers_from_failed_growth() {
    let budget = MemoryBudget::new(1 << 20);
    let mut slot = None;
    EstimationBuffers::cached(&mut slot, 32, 8, true, &budget).unwrap();
    let reserved = budget.used();
    let allocation = allocation_counter::measure(|| {
        for (n, m, mean) in [(0, 1, false), (32, 8, true), (4, 3, false), (16, 5, true)] {
            EstimationBuffers::cached(&mut slot, n, m, mean, &budget).unwrap();
        }
    });
    assert_eq!(allocation.count_total, 0);
    assert_eq!(budget.used(), reserved);
    assert!(EstimationBuffers::cached(&mut slot, usize::MAX, 8, true, &budget).is_err());
    assert_eq!(budget.used(), 0);
    assert!(slot.is_none());
    EstimationBuffers::cached(&mut slot, 4, 2, true, &budget).unwrap();
    drop(slot);
    assert_eq!(budget.used(), 0);
}
