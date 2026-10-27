use std::sync::atomic::{AtomicUsize, Ordering};
use terrustrial_core::{
    data::{Group, Observation, Point3, Site, Support, Target},
    estimation::EstimationOptions,
    execution::{Cancellation, ExecutionOptions, MemoryBudget},
    gaussian::{
        Drift, ExternalDrift, GaussianPlan, NumericalTolerance, Ordinary, PrepareOptions, Simple,
        Uncertainty,
    },
    model::{Anisotropy, Coregionalization, Correlation, Covariance, Lmc, Structure},
    simulation::{self, parallel, SimulationOptions},
    spatial::{Neighborhood, NeighborhoodQuery, SearchNeighborhood, SpatialIndex, VariableLimit},
    Error, Result,
};
fn point(p: [f64; 3]) -> Point3 {
    Point3::new(p).unwrap()
}
fn site(p: [f64; 3], v: usize) -> Site {
    Site::point(point(p), v)
}
// Keep the independent fused-residual numerical check while comparing cached
// worker results bit-for-bit with a fresh plan using the same faer reduction.
fn centered_reference(
    plan: &GaussianPlan,
    values: &[f64],
    means: &[f64],
    target_means: &[f64],
) -> Result<Vec<f64>> {
    let fused = plan.estimate_with_mean(values, means, target_means)?;
    let residuals: Vec<_> = values.iter().zip(means).map(|(v, m)| v - m).collect();
    let mut prediction = plan.estimate(&residuals)?;
    for ((value, mean), reference) in prediction.iter_mut().zip(target_means).zip(fused) {
        *value += mean;
        assert!((*value - reference).abs() <= 1e-10 * (1. + reference.abs()));
    }
    Ok(prediction)
}
fn model() -> Lmc {
    Lmc::new(vec![Structure {
        correlation: Correlation::Exponential,
        anisotropy: Anisotropy::isotropic(3.0).unwrap(),
        coregionalization: Coregionalization::from_factor(2, 2, &[1., 0., 0.6, 0.8]).unwrap(),
    }])
    .unwrap()
}
fn hard() -> Vec<Observation> {
    (0..24)
        .map(|i| {
            Observation::new(
                i,
                site(
                    [(i / 2 % 3) as f64 - 1., (i / 6 % 2) as f64, (i / 12) as f64],
                    (i % 2) as usize,
                ),
                if i == 0 { -0.0 } else { (i as f64).sin() },
                0.,
            )
            .unwrap()
        })
        .collect()
}
fn path() -> Vec<Group> {
    (0..35)
        .map(|i| Group {
            id: i * 7 + 3,
            targets: (0..2)
                .map(|v| Target {
                    id: 1000 + i * 2 + v as u64,
                    site: if i == 1 {
                        Site {
                            support: Support::block(
                                point([1.7, 0.3, 0.7]),
                                [0.1, 0.1, 0.],
                                [2, 2, 1],
                            )
                            .unwrap(),
                            variable: v,
                        }
                    } else {
                        site(
                            [
                                if i == 0 {
                                    -1.
                                } else {
                                    (i * 13 % 37) as f64 * 0.12
                                },
                                if i == 0 {
                                    0.
                                } else {
                                    (i * 7 % 19) as f64 * 0.1
                                },
                                0.,
                            ],
                            v,
                        )
                    },
                })
                .collect(),
        })
        .collect()
}
fn execution(batch_groups: usize) -> ExecutionOptions {
    ExecutionOptions {
        batch_groups,
        maximum_targets_per_group: 2,
        bytes_per_group: 1 << 20,
    }
}

#[test]
fn rolling_simulation_preserves_sequential_bits_and_checked_rejection() {
    fn compare(drift: &impl Drift) {
        let model = model();
        let hard = SpatialIndex::new(hard()).unwrap();
        let mut groups = path();
        // Interleaved low/high IDs make projection ranges overlap. Membership
        // must still be checked exactly, including near the u64 boundary.
        for group in &mut groups {
            for target in &mut group.targets {
                if target.id % 2 == 0 {
                    target.id = u64::MAX - target.id;
                }
            }
        }
        // Repeated supports cross both small and large admission windows.
        groups[19].targets[0].site = groups[3].targets[0].site;
        let query = Neighborhood {
            maximum: 12,
            radius: 20.,
        };
        let budget = MemoryBudget::new(1 << 28);
        let cancel = Cancellation::default();
        let mut expected = Vec::new();
        simulation::simulate_groups(
            &model,
            &hard,
            &mut SpatialIndex::new(vec![]).unwrap(),
            groups.clone(),
            drift,
            options(query.into(), &budget, &cancel),
            |out| {
                expected.extend(out.iter().map(|o| (o.id(), o.value().to_bits())));
                Ok(())
            },
        )
        .unwrap();
        for workers in [1, 4] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(workers)
                .build()
                .unwrap();
            for capacity in [1, 3, 16] {
                let mut actual = Vec::new();
                let mut store = SpatialIndex::with_budget(vec![], &budget).unwrap();
                pool.install(|| {
                    parallel::simulate_pipelined(
                        &model,
                        &hard,
                        &mut store,
                        groups.clone(),
                        drift,
                        options(query.into(), &budget, &cancel),
                        execution(capacity),
                        |out| {
                            actual.extend(out.iter().map(|o| (o.id(), o.value().to_bits())));
                            Ok(())
                        },
                    )
                })
                .unwrap();
                assert_eq!(actual, expected, "workers={workers}, capacity={capacity}");
                drop(store);
                assert_eq!(budget.used(), 0);
            }
        }
        let mut store = SpatialIndex::with_budget(vec![], &budget).unwrap();
        let mut outputs = 0;
        let result = parallel::simulate_pipelined_with_residuals(
            &model,
            &hard,
            &mut store,
            groups.clone(),
            drift,
            options(query.into(), &budget, &cancel),
            execution(8),
            |group, _| {
                if group.id == groups[4].id {
                    Err(Error::Cancelled)
                } else {
                    Ok(())
                }
            },
            |_| {
                outputs += 1;
                Ok(())
            },
        );
        assert!(matches!(result, Err(Error::Cancelled)));
        assert_eq!(outputs, 4);
        assert_eq!(store.len(), 8);
        drop(store);
        assert_eq!(budget.used(), 0);
    }
    compare(&Simple);
    compare(&Ordinary { variables: 2 });
}

#[cfg(feature = "sqlite")]
#[test]
fn topological_simulation_with_async_publication_matches_resident_and_rejects_before_commit() {
    use terrustrial_core::storage::packed::{
        Options, PackedBuilder, ScheduledOptions, ScheduledWriter,
    };
    let dir = tempfile::tempdir().unwrap();
    let budget = MemoryBudget::new(1 << 28);
    let cancel = Cancellation::default();
    let hard = SpatialIndex::new(hard()).unwrap();
    let model = model();
    let groups = path();
    let query = Neighborhood {
        maximum: 12,
        radius: 20.,
    };
    let mut expected = Vec::new();
    simulation::simulate_groups(
        &model,
        &hard,
        &mut SpatialIndex::new(vec![]).unwrap(),
        groups.clone(),
        &Simple,
        options(query.into(), &budget, &cancel),
        |out| {
            expected.extend(out.iter().map(|o| o.value().to_bits()));
            Ok(())
        },
    )
    .unwrap();
    for reject in [false, true] {
        let file = dir
            .path()
            .join(if reject { "rejected" } else { "accepted" });
        let mut builder = PackedBuilder::create(file, Options::default(), &budget).unwrap();
        for group in &groups {
            for target in &group.targets {
                builder
                    .push(
                        Observation::new(target.id, target.site, 0., 0.).unwrap(),
                        group.id + 1,
                        &cancel,
                    )
                    .unwrap();
            }
        }
        let store = builder.finish(&cancel).unwrap();
        let mut writer = ScheduledWriter::new(
            store.clone(),
            ScheduledOptions {
                maximum_group_observations: 2,
                queued_groups: 8,
                commit_groups: 4,
                retained_groups: 40,
            },
            &budget,
            &cancel,
        )
        .unwrap();
        let mut actual = Vec::new();
        let result = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap()
            .install(|| {
                parallel::simulate_pipelined_with_residuals(
                    &model,
                    &hard,
                    &mut writer,
                    groups.clone(),
                    &Simple,
                    options(query.into(), &budget, &cancel),
                    execution(16),
                    |group, _| {
                        if reject && group.id == groups[4].id {
                            Err(Error::Cancelled)
                        } else {
                            Ok(())
                        }
                    },
                    |out| {
                        actual.extend(out.iter().map(|o| o.value().to_bits()));
                        Ok(())
                    },
                )
            });
        let count = if reject { 4 } else { groups.len() };
        if reject {
            assert!(matches!(result, Err(Error::Cancelled)));
        } else {
            result.unwrap();
        }
        assert_eq!(actual, expected[..count * 2]);
        assert_eq!(
            store.snapshot().unwrap().completed_group(),
            groups[count - 1].id + 1
        );
        drop((writer, store));
        assert_eq!(budget.used(), 0);
    }
}
fn options<'a>(
    query: NeighborhoodQuery<'a>,
    budget: &'a MemoryBudget,
    cancel: &'a Cancellation,
) -> SimulationOptions<'a> {
    SimulationOptions {
        seed: 76,
        realization: 3,
        mean: Some(&trend),
        hard_neighborhood: Neighborhood {
            maximum: 24,
            radius: 20.,
        }
        .into(),
        simulated_neighborhood: query,
        budget,
        cancellation: cancel,
    }
}
fn trend(s: &Site) -> Result<f64> {
    Ok(s.support
        .nodes()
        .map(|p| p.coordinates()[0].powi(2) + s.variable as f64 * 3.)
        .sum::<f64>()
        / s.support.node_count() as f64)
}

#[test]
fn large_joint_groups_preserve_estimation_and_simulation_bits_across_workers() {
    let variables = 8;
    let factors: Vec<_> = (0..variables)
        .flat_map(|i| {
            (0..variables).map(move |j| {
                if i == j {
                    1.
                } else {
                    0.1 * ((i * 13 + j) as f64).sin()
                }
            })
        })
        .collect();
    let model = Lmc::new(vec![Structure {
        correlation: Correlation::Exponential,
        anisotropy: Anisotropy::isotropic(3.).unwrap(),
        coregionalization: Coregionalization::from_factor(variables, variables, &factors).unwrap(),
    }])
    .unwrap();
    let hard = SpatialIndex::new(
        (0..128)
            .map(|i| {
                Observation::new(
                    i,
                    site(
                        [((i / 8) % 4) as f64, (i / 32) as f64, 0.],
                        (i % 8) as usize,
                    ),
                    (i as f64 * 0.3).sin(),
                    0.01,
                )
                .unwrap()
            })
            .collect(),
    )
    .unwrap();
    let groups: Vec<_> = (0..9)
        .map(|i| Group {
            id: i,
            targets: (0..32)
                .map(|j| Target {
                    id: 10000 + i * 32 + j,
                    site: site(
                        [0.13 + i as f64 * 0.17, 0.23 + (j / 8) as f64 * 0.3, 0.19],
                        (j % 8) as usize,
                    ),
                })
                .collect(),
        })
        .collect();
    let budget = MemoryBudget::new(1 << 29);
    let cancel = Cancellation::default();
    let hard_query = Neighborhood {
        maximum: 96,
        radius: 100.,
    };
    let drift = Ordinary { variables };
    let options = || SimulationOptions {
        seed: 811,
        realization: 2,
        mean: Some(&trend),
        hard_neighborhood: hard_query.into(),
        simulated_neighborhood: Neighborhood {
            maximum: 32,
            radius: 100.,
        }
        .into(),
        budget: &budget,
        cancellation: &cancel,
    };
    let mut simulation_reference = Vec::new();
    simulation::simulate_groups(
        &model,
        &hard,
        &mut SpatialIndex::new(vec![]).unwrap(),
        groups.clone(),
        &drift,
        options(),
        |g| {
            simulation_reference.extend(g.iter().map(|o| o.value().to_bits()));
            Ok(())
        },
    )
    .unwrap();
    let mut estimation_reference = None;
    for workers in [1, 2, 3, 4, 8] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .unwrap();
        for batch in [1, 4, 7, 12] {
            let execution = ExecutionOptions {
                batch_groups: batch,
                maximum_targets_per_group: 32,
                bytes_per_group: 1 << 24,
            };
            let mut simulated = Vec::new();
            let mut estimated = Vec::new();
            pool.install(|| {
                parallel::simulate_groups(
                    &model,
                    &hard,
                    &mut SpatialIndex::new(vec![]).unwrap(),
                    groups.clone(),
                    &drift,
                    options(),
                    execution,
                    |g| {
                        simulated.extend(g.iter().map(|o| o.value().to_bits()));
                        Ok(())
                    },
                )
                .unwrap();
                terrustrial_core::estimation::estimate_groups(
                    &model,
                    &hard,
                    groups.clone(),
                    &drift,
                    EstimationOptions {
                        neighborhood: hard_query.into(),
                        mean: Some(&trend),
                        uncertainty: Uncertainty::Joint,
                        tolerance: NumericalTolerance::default(),
                        budget: &budget,
                        cancellation: &cancel,
                    },
                    execution,
                    |out| {
                        estimated.extend(out.values.iter().map(|v| v.to_bits()));
                        for col in out.covariance().unwrap().col_iter() {
                            estimated.extend(col.iter().map(|v| v.to_bits()));
                        }
                        Ok(())
                    },
                )
                .unwrap();
            });
            assert_eq!(
                simulated, simulation_reference,
                "workers={workers} batch={batch}"
            );
            if let Some(reference) = &estimation_reference {
                assert_eq!(&estimated, reference);
            } else {
                estimation_reference = Some(estimated);
            }
            assert_eq!(budget.used(), 0);
        }
    }
}
#[test]
fn serial_and_rolling_match_bits_across_workers_windows_drifts_and_search_caps() {
    let spherical: NeighborhoodQuery<'_> = Neighborhood {
        maximum: 16,
        radius: 5.,
    }
    .into();
    let constrained = SearchNeighborhood::new(
        15,
        Anisotropy::isotropic(5.).unwrap(),
        Some(5),
        vec![
            VariableLimit {
                variable: 0,
                maximum: 8,
            },
            VariableLimit {
                variable: 1,
                maximum: 8,
            },
        ],
    )
    .unwrap();
    fn compare(drift: &impl Drift, query: NeighborhoodQuery<'_>) {
        let budget = MemoryBudget::new(1 << 28);
        let cancel = Cancellation::default();
        let hard = SpatialIndex::new(hard()).unwrap();
        let mut reference = Vec::new();
        simulation::simulate_groups(
            &model(),
            &hard,
            &mut SpatialIndex::new(vec![]).unwrap(),
            path(),
            drift,
            options(query, &budget, &cancel),
            |g| {
                reference.extend(g.iter().map(|o| o.value().to_bits()));
                Ok(())
            },
        )
        .unwrap();
        for workers in [1, 2, 3, 4, 8] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(workers)
                .build()
                .unwrap();
            for batch in [1, 7, 17, 32, 40] {
                let mut actual = Vec::new();
                pool.install(|| {
                    parallel::simulate_pipelined(
                        &model(),
                        &hard,
                        &mut SpatialIndex::new(vec![]).unwrap(),
                        path(),
                        drift,
                        options(query, &budget, &cancel),
                        execution(batch),
                        |g| {
                            actual.extend(g.iter().map(|o| o.value().to_bits()));
                            Ok(())
                        },
                    )
                })
                .unwrap();
                assert_eq!(reference, actual, "workers={workers}, batch={batch}");
                assert_eq!(budget.used(), 0);
            }
        }
        assert_eq!(reference[0], (-0.0_f64).to_bits());
    }
    for query in [spherical, NeighborhoodQuery::Constrained(&constrained)] {
        compare(&Simple, query);
        compare(&Ordinary { variables: 2 }, query);
        compare(
            &ExternalDrift::new(3, |s: &Site, b: &mut [f64]| {
                b[0] = (s.variable == 0) as u8 as f64;
                b[1] = (s.variable == 1) as u8 as f64;
                b[2] = b[0] + 2_f64.powi(-24) * s.support.center().coordinates()[0];
                Ok(())
            })
            .unwrap(),
            query,
        );
        compare(
            &ExternalDrift::new(3, |s: &Site, b: &mut [f64]| {
                b[0] = (s.variable == 0) as u8 as f64;
                b[1] = (s.variable == 1) as u8 as f64;
                b[2] = s.support.center().coordinates()[0];
                Ok(())
            })
            .unwrap(),
            query,
        );
    }
}
#[test]
fn estimation_matches_direct_joint_plans_and_emits_in_order() {
    let budget = MemoryBudget::new(1 << 28);
    let cancel = Cancellation::default();
    let hard = hard();
    let model = model();
    let groups = path();
    let sites: Vec<Vec<_>> = groups
        .iter()
        .map(|g| g.targets.iter().map(|t| t.site).collect())
        .collect();
    let values = hard.iter().map(Observation::value).collect::<Vec<_>>();
    let means = hard
        .iter()
        .map(|o| trend(o.site()).unwrap())
        .collect::<Vec<_>>();
    let mut reference_bits = None;
    for workers in [1, 2, 4, 8] {
        let mut actual_bits = Vec::new();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .unwrap();
        let mut position = 0;
        pool.install(|| {
            terrustrial_core::estimation::estimate_groups(
                &model,
                hard.as_slice(),
                groups.clone(),
                &Ordinary { variables: 2 },
                EstimationOptions {
                    neighborhood: Neighborhood {
                        maximum: 100,
                        radius: 100.,
                    }
                    .into(),
                    mean: Some(&trend),
                    uncertainty: Uncertainty::Joint,
                    tolerance: NumericalTolerance::default(),
                    budget: &budget,
                    cancellation: &cancel,
                },
                execution(8),
                |result| {
                    let plan = GaussianPlan::prepare(
                        &model,
                        &hard,
                        &sites[position],
                        &Ordinary { variables: 2 },
                        PrepareOptions {
                            uncertainty: Uncertainty::Joint,
                            tolerance: NumericalTolerance::default(),
                            budget: &MemoryBudget::new(1 << 20),
                            cancellation: &cancel,
                        },
                    )?;
                    let expected = plan.estimate_with_mean(
                        &values,
                        &means,
                        &sites[position]
                            .iter()
                            .map(|s| trend(s).unwrap())
                            .collect::<Vec<_>>(),
                    )?;
                    assert_eq!(result.group.id, groups[position].id);
                    for (j, expected_value) in expected.iter().enumerate() {
                        // Direct plan uses source order, whereas the driver sorts by distance.
                        assert!((*expected_value - result.values[j]).abs() < 2e-11);
                        for k in 0..2 {
                            assert!(
                                (plan.covariance().unwrap()[(j, k)]
                                    - result.covariance().unwrap()[(j, k)])
                                    .abs()
                                    < 2e-11
                            );
                        }
                    }
                    actual_bits.extend(result.values.iter().map(|v| v.to_bits()));
                    for j in 0..2 {
                        for k in 0..2 {
                            actual_bits.push(result.covariance().unwrap()[(j, k)].to_bits());
                        }
                    }
                    position += 1;
                    Ok(())
                },
            )
        })
        .unwrap();
        if let Some(reference) = &reference_bits {
            assert_eq!(&actual_bits, reference);
        } else {
            reference_bits = Some(actual_bits);
        }
        assert_eq!(position, groups.len());
        assert_eq!(budget.used(), 0);
    }
}

#[test]
fn estimation_reuses_changing_shapes_with_optional_covariance_and_means() {
    use terrustrial_core::spatial::ObservationSource;
    let records = hard();
    let source = SpatialIndex::new(records).unwrap();
    let model = model();
    let budget = MemoryBudget::new(1 << 28);
    let cancel = Cancellation::default();
    let groups: Vec<_> = (0..29)
        .map(|i| Group {
            id: i,
            targets: (0..1 + i as usize % 4)
                .map(|j| Target {
                    id: 1000 + i * 4 + j as u64,
                    site: site(
                        [
                            [0.13, 0.53, 1.43, 2.73, 8.13][i as usize % 5] + (j / 2) as f64 * 0.1,
                            0.17,
                            0.23,
                        ],
                        j % 2,
                    ),
                })
                .collect(),
        })
        .collect();
    let query = Neighborhood {
        maximum: 24,
        radius: 1.5,
    };
    for uncertainty in [Uncertainty::None, Uncertainty::Joint] {
        for use_mean in [false, true] {
            let mut reference = Vec::new();
            for group in &groups {
                let count = group.targets.len() as f64;
                let mut center = [0.; 3];
                for target in &group.targets {
                    for (j, x) in center.iter_mut().enumerate() {
                        *x += target.site.support.center().coordinates()[j] / count;
                    }
                }
                let mut observations = source.nearest(point(center), query, &cancel).unwrap();
                let sites: Vec<_> = group.targets.iter().map(|t| t.site).collect();
                let solve = |observations: &[Observation]| {
                    let plan = GaussianPlan::prepare(
                        &model,
                        observations,
                        &sites,
                        &Simple,
                        PrepareOptions {
                            uncertainty,
                            tolerance: NumericalTolerance::default(),
                            budget: &budget,
                            cancellation: &cancel,
                        },
                    )
                    .unwrap();
                    let values: Vec<_> = observations.iter().map(Observation::value).collect();
                    let values = if use_mean {
                        centered_reference(
                            &plan,
                            &values,
                            &observations
                                .iter()
                                .map(|o| trend(o.site()).unwrap())
                                .collect::<Vec<_>>(),
                            &sites.iter().map(|s| trend(s).unwrap()).collect::<Vec<_>>(),
                        )
                        .unwrap()
                    } else {
                        plan.estimate(&values).unwrap()
                    };
                    let covariance = plan.covariance().map(|c| {
                        (0..sites.len())
                            .flat_map(|j| (0..sites.len()).map(move |i| c[(i, j)]))
                            .collect::<Vec<_>>()
                    });
                    (values, covariance, plan.roundoff_corrections())
                };
                let distance_ordered = solve(&observations);
                observations.sort_unstable_by_key(Observation::id);
                let (values, covariance, corrections) = solve(&observations);
                // Row permutation changes rounding, not neighborhood membership
                // or the represented system. Check the previous distance order
                // numerically, and the driver ID order bit-for-bit below.
                for (a, b) in values.iter().zip(&distance_ordered.0) {
                    assert!((a - b).abs() < 2e-11);
                }
                if let (Some(a), Some(b)) = (&covariance, &distance_ordered.1) {
                    for (a, b) in a.iter().zip(b) {
                        assert!((a - b).abs() < 2e-11);
                    }
                }
                reference.push((
                    values,
                    covariance.map(|c| c.into_iter().map(f64::to_bits).collect::<Vec<_>>()),
                    corrections,
                ));
            }
            for workers in [1, 2, 4, 8] {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(workers)
                    .build()
                    .unwrap();
                let mut position = 0;
                pool.install(|| {
                    terrustrial_core::estimation::estimate_groups(
                        &model,
                        &source,
                        groups.clone(),
                        &Simple,
                        EstimationOptions {
                            neighborhood: query.into(),
                            mean: if use_mean { Some(&trend) } else { None },
                            uncertainty,
                            tolerance: NumericalTolerance::default(),
                            budget: &budget,
                            cancellation: &cancel,
                        },
                        ExecutionOptions {
                            batch_groups: 7,
                            maximum_targets_per_group: 4,
                            bytes_per_group: 1 << 20,
                        },
                        |result| {
                            assert_eq!(result.group.id, groups[position].id);
                            assert_eq!(result.group.targets.len(), groups[position].targets.len());
                            for (a, b) in result.group.targets.iter().zip(&groups[position].targets)
                            {
                                assert_eq!(a.id, b.id);
                                assert_eq!(a.site, b.site);
                            }
                            assert_eq!(
                                result
                                    .values
                                    .iter()
                                    .map(|v| v.to_bits())
                                    .collect::<Vec<_>>(),
                                reference[position]
                                    .0
                                    .iter()
                                    .map(|v| v.to_bits())
                                    .collect::<Vec<_>>()
                            );
                            let covariance = result.covariance().map(|c| {
                                (0..c.ncols())
                                    .flat_map(|j| (0..c.nrows()).map(move |i| c[(i, j)].to_bits()))
                                    .collect::<Vec<_>>()
                            });
                            assert_eq!(covariance, reference[position].1);
                            assert_eq!(result.roundoff_corrections(), reference[position].2);
                            position += 1;
                            Ok(())
                        },
                    )
                })
                .unwrap();
                assert_eq!(position, groups.len());
                assert_eq!(budget.used(), 0);
            }
        }
    }
}

#[test]
fn estimation_releases_retained_results_on_callback_and_mean_errors() {
    for joint in [false, true] {
        let budget = MemoryBudget::new(1 << 28);
        let cancel = Cancellation::default();
        let mut emitted = 0;
        let options = |mean| EstimationOptions {
            neighborhood: Neighborhood {
                maximum: 24,
                radius: 100.,
            }
            .into(),
            mean,
            uncertainty: if joint {
                Uncertainty::Joint
            } else {
                Uncertainty::None
            },
            tolerance: NumericalTolerance::default(),
            budget: &budget,
            cancellation: &cancel,
        };
        let error = terrustrial_core::estimation::estimate_groups(
            &model(),
            hard().as_slice(),
            path(),
            &Simple,
            options(Some(&trend)),
            execution(8),
            |_| {
                emitted += 1;
                if emitted == 3 {
                    Err(Error::InvalidInput("callback failure".into()))
                } else {
                    Ok(())
                }
            },
        )
        .unwrap_err();
        assert!(matches!(error, Error::InvalidInput(_)));
        assert_eq!(emitted, 3);
        assert_eq!(budget.used(), 0);
        let invalid_mean = |_: &Site| Ok(f64::NAN);
        emitted = 0;
        assert!(terrustrial_core::estimation::estimate_groups(
            &model(),
            hard().as_slice(),
            path(),
            &Simple,
            options(Some(&invalid_mean)),
            execution(8),
            |_| {
                emitted += 1;
                Ok(())
            }
        )
        .is_err());
        assert_eq!(emitted, 0);
        assert_eq!(budget.used(), 0);
    }
}

#[test]
fn reused_estimation_selection_keeps_exact_data_outside_group_neighborhood() {
    let hard = vec![
        Observation::new(1, site([-1., 0., 0.], 0), -0., 0.).unwrap(),
        Observation::new(2, site([-1., 0., 0.], 0), 0., 0.).unwrap(),
    ];
    let groups: Vec<_> = (0..19)
        .map(|i| Group {
            id: i,
            targets: vec![
                Target {
                    id: 100 + 2 * i,
                    site: site([-1., 0., 0.], 0),
                },
                Target {
                    id: 101 + 2 * i,
                    site: site([8. + i as f64 * 0.1, 0., 0.], 0),
                },
            ],
        })
        .collect();
    let budget = MemoryBudget::new(1 << 28);
    let cancel = Cancellation::default();
    let mut emitted = 0;
    terrustrial_core::estimation::estimate_groups(
        &model(),
        hard.as_slice(),
        groups.clone(),
        &Simple,
        EstimationOptions {
            neighborhood: Neighborhood {
                maximum: 1,
                radius: 0.1,
            }
            .into(),
            mean: None,
            uncertainty: Uncertainty::Joint,
            tolerance: NumericalTolerance::default(),
            budget: &budget,
            cancellation: &cancel,
        },
        execution(3),
        |result| {
            assert_eq!(result.values[0].to_bits(), (-0.0_f64).to_bits());
            assert_eq!(result.covariance().unwrap()[(0, 0)], 0.);
            emitted += 1;
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(emitted, groups.len());
    assert_eq!(budget.used(), 0);
    let conflicting = vec![
        hard[0],
        Observation::new(2, site([-1., 0., 0.], 0), 1., 0.).unwrap(),
    ];
    assert!(terrustrial_core::estimation::estimate_groups(
        &model(),
        conflicting.as_slice(),
        groups,
        &Simple,
        EstimationOptions {
            neighborhood: Neighborhood {
                maximum: 1,
                radius: 0.1
            }
            .into(),
            mean: None,
            uncertainty: Uncertainty::None,
            tolerance: NumericalTolerance::default(),
            budget: &budget,
            cancellation: &cancel,
        },
        execution(3),
        |_| panic!("conflicting exact data must not be emitted")
    )
    .is_err());
    assert_eq!(budget.used(), 0);
}
#[test]
fn prepared_unique_and_aliased_selections_preserve_values_errors_and_slot_reuse() {
    // Separate neighborhoods make the same retained slot alternate between
    // unique data, aliases and unique data again. Test both compaction backends.
    for count in [8, 48, 64] {
        for variant in 0..5 {
            let mut hard: Vec<_> = (0..3)
                .flat_map(|cluster| {
                    (0..count).map(move |i| {
                        Observation::new(
                            (cluster * count + i) as u64,
                            site([cluster as f64 * 10. + i as f64 * 0.02, 0., 0.], 0),
                            if i == 0 { -0. } else { (i as f64).sin() },
                            if i == 0 { 0. } else { 0.01 },
                        )
                        .unwrap()
                    })
                })
                .collect();
            if variant != 0 {
                let first = hard[count];
                hard.push(
                    Observation::new(
                        if variant % 2 == 1 { first.id() } else { 10_000 },
                        *first.site(),
                        if variant >= 3 { 1. } else { 0. },
                        0.,
                    )
                    .unwrap(),
                );
            }
            let groups: Vec<_> = (0..3)
                .map(|i| Group {
                    id: i,
                    targets: vec![Target {
                        id: 100_000 + i,
                        site: site([i as f64 * 10. + 0.013, 0.01, 0.], 0),
                    }],
                })
                .collect();
            let budget = MemoryBudget::new(1 << 26);
            let cancel = Cancellation::default();
            let options = || SimulationOptions {
                seed: 912,
                realization: 3,
                mean: None,
                hard_neighborhood: Neighborhood {
                    maximum: count + 2,
                    radius: 2.,
                }
                .into(),
                simulated_neighborhood: Neighborhood {
                    maximum: 4,
                    radius: 1.,
                }
                .into(),
                budget: &budget,
                cancellation: &cancel,
            };
            let mut expected = Vec::new();
            let reference = simulation::simulate_groups(
                &model(),
                hard.as_slice(),
                &mut SpatialIndex::new(vec![]).unwrap(),
                groups.clone(),
                &Simple,
                options(),
                |group| {
                    expected.extend(group.iter().map(|o| o.value().to_bits()));
                    Ok(())
                },
            );
            assert_eq!(reference.is_err(), variant >= 3);
            assert_eq!(expected.len(), if variant >= 3 { 1 } else { 3 });
            let expected_error = reference.err().map(|e| e.to_string());
            for workers in [1, 2, 4, 8] {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(workers)
                    .build()
                    .unwrap();
                for batch in [1, 2, 4] {
                    let mut actual = Vec::new();
                    let result = pool.install(|| {
                        parallel::simulate_groups(
                            &model(),
                            hard.as_slice(),
                            &mut SpatialIndex::new(vec![]).unwrap(),
                            groups.clone(),
                            &Simple,
                            options(),
                            execution(batch),
                            |group| {
                                actual.extend(group.iter().map(|o| o.value().to_bits()));
                                Ok(())
                            },
                        )
                    });
                    assert_eq!(
                        result.err().map(|e| e.to_string()),
                        expected_error,
                        "count={count} variant={variant} workers={workers} batch={batch}"
                    );
                    assert_eq!(actual, expected);
                    assert_eq!(budget.used(), 0);
                }
            }
        }
    }
}

#[test]
fn repeated_sites_bind_actual_predecessors_and_checked_ids_reject_collisions() {
    let budget = MemoryBudget::new(1 << 26);
    let cancel = Cancellation::default();
    let query = Neighborhood {
        maximum: 10,
        radius: 10.,
    }
    .into();
    let mut groups = path();
    groups.truncate(3);
    groups[2].targets[0].site = groups[1].targets[0].site;
    let mut expected = Vec::new();
    let mut actual = Vec::new();
    simulation::simulate_groups(
        &model(),
        hard().as_slice(),
        &mut SpatialIndex::new(vec![]).unwrap(),
        groups.clone(),
        &Simple,
        options(query, &budget, &cancel),
        |g| {
            expected.extend(g.iter().map(|o| o.value().to_bits()));
            Ok(())
        },
    )
    .unwrap();
    parallel::simulate_groups(
        &model(),
        hard().as_slice(),
        &mut SpatialIndex::new(vec![]).unwrap(),
        groups.clone(),
        &Simple,
        options(query, &budget, &cancel),
        execution(3),
        |g| {
            actual.extend(g.iter().map(|o| o.value().to_bits()));
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(expected, actual);
    assert_eq!(actual[2], actual[4]);
    groups[0].targets[0].id = 0;
    assert!(parallel::simulate_groups(
        &model(),
        hard().as_slice(),
        &mut SpatialIndex::new(vec![]).unwrap(),
        groups,
        &Simple,
        options(query, &budget, &cancel),
        execution(3),
        |_| Ok(())
    )
    .is_err());
    assert_eq!(budget.used(), 0);
}
#[test]
fn budgets_cancellation_invalid_windows_and_callback_failures_release_reservations() {
    let cancel = Cancellation::default();
    let query = Neighborhood {
        maximum: 10,
        radius: 10.,
    }
    .into();
    for (limit, allowance) in [(1000, 1 << 20), (1 << 26, 100)] {
        let budget = MemoryBudget::new(limit);
        let mut emitted = 0;
        let mut exec = execution(4);
        exec.bytes_per_group = allowance;
        let err = parallel::simulate_groups(
            &model(),
            hard().as_slice(),
            &mut SpatialIndex::new(vec![]).unwrap(),
            path(),
            &Simple,
            options(query, &budget, &cancel),
            exec,
            |_| {
                emitted += 1;
                Ok(())
            },
        )
        .unwrap_err();
        assert!(matches!(err, Error::MemoryBudget { .. }));
        assert_eq!(emitted, 0);
        assert_eq!(budget.used(), 0);
    }
    let budget = MemoryBudget::new(1 << 26);
    let mut emitted = 0;
    let mut store = SpatialIndex::new(vec![]).unwrap();
    let err = parallel::simulate_groups(
        &model(),
        hard().as_slice(),
        &mut store,
        path(),
        &Simple,
        options(query, &budget, &cancel),
        execution(4),
        |_| {
            emitted += 1;
            Err(Error::InvalidInput("sink failure".into()))
        },
    )
    .unwrap_err();
    assert!(matches!(err, Error::InvalidInput(_)));
    assert_eq!(emitted, 1);
    assert_eq!(store.len(), 2);
    assert_eq!(budget.used(), 0);
    cancel.cancel();
    assert!(matches!(
        parallel::simulate_groups(
            &model(),
            hard().as_slice(),
            &mut SpatialIndex::new(vec![]).unwrap(),
            path(),
            &Simple,
            options(query, &budget, &cancel),
            execution(4),
            |_| Ok(())
        ),
        Err(Error::Cancelled)
    ));
    assert_eq!(budget.used(), 0);
}
#[test]
fn covariance_preparation_actually_overlaps_between_workers() {
    struct Counting {
        active: AtomicUsize,
        peak: AtomicUsize,
    }
    impl Covariance for Counting {
        fn variables(&self) -> usize {
            2
        }
        fn point_covariance(&self, a: Point3, av: usize, b: Point3, bv: usize) -> Result<f64> {
            let count = self.active.fetch_add(1, Ordering::SeqCst) + 1;
            self.peak.fetch_max(count, Ordering::SeqCst);
            std::thread::sleep(std::time::Duration::from_micros(100));
            self.active.fetch_sub(1, Ordering::SeqCst);
            Ok(if av == bv { (-a.distance(b)).exp() } else { 0. })
        }
    }
    let counter = Counting {
        active: AtomicUsize::new(0),
        peak: AtomicUsize::new(0),
    };
    let budget = MemoryBudget::new(1 << 26);
    let cancel = Cancellation::default();
    rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap()
        .install(|| {
            terrustrial_core::estimation::estimate_groups(
                &counter,
                &[][..],
                path().into_iter().take(8),
                &Simple,
                EstimationOptions {
                    neighborhood: Neighborhood {
                        maximum: 2,
                        radius: 1.,
                    }
                    .into(),
                    mean: None,
                    uncertainty: Uncertainty::Joint,
                    tolerance: NumericalTolerance::default(),
                    budget: &budget,
                    cancellation: &cancel,
                },
                execution(8),
                |_| Ok(()),
            )
        })
        .unwrap();
    assert!(counter.peak.load(Ordering::SeqCst) > 1);
    assert_eq!(budget.used(), 0);
}

#[test]
fn lattice_distance_ties_match_the_sequential_path() {
    let make_path = || {
        (0..400u64).map(|i| Group {
            id: i,
            targets: (0..2)
                .map(|v| Target {
                    id: 100 + i * 2 + v as u64,
                    site: site([(i % 20) as f64, (i / 20) as f64, 0.], v),
                })
                .collect(),
        })
    };
    let budget = MemoryBudget::new(1 << 27);
    let cancel = Cancellation::default();
    let hard: Vec<_> = (0..4)
        .map(|i| {
            Observation::new(
                i,
                site([-1., (i / 2) as f64, 0.], (i % 2) as usize),
                0.25 * i as f64,
                0.,
            )
            .unwrap()
        })
        .collect();
    let make_options = || SimulationOptions {
        seed: 2026,
        realization: 1,
        mean: None,
        hard_neighborhood: Neighborhood {
            maximum: 4,
            radius: 8.,
        }
        .into(),
        simulated_neighborhood: Neighborhood {
            maximum: 32,
            radius: 8.,
        }
        .into(),
        budget: &budget,
        cancellation: &cancel,
    };
    let mut expected = Vec::new();
    simulation::simulate_groups(
        &model(),
        hard.as_slice(),
        &mut SpatialIndex::new(vec![]).unwrap(),
        make_path(),
        &Simple,
        make_options(),
        |g| {
            expected.extend(g.iter().map(|o| o.value().to_bits()));
            Ok(())
        },
    )
    .unwrap();
    let mut actual = Vec::new();
    parallel::simulate_groups(
        &model(),
        hard.as_slice(),
        &mut SpatialIndex::new(vec![]).unwrap(),
        make_path(),
        &Simple,
        make_options(),
        execution(32),
        |g| {
            actual.extend(g.iter().map(|o| o.value().to_bits()));
            Ok(())
        },
    )
    .unwrap();
    if let Some(i) = expected.iter().zip(&actual).position(|(a, b)| a != b) {
        panic!("first mismatch at group {}, variable {}", i / 2, i % 2);
    }
}

#[test]
fn nearest_lattice_ties_match_full_scan() {
    use terrustrial_core::spatial::{ObservationSink, ObservationSource};
    let observations: Vec<_> = (0..600)
        .map(|i| {
            Observation::new(
                i,
                site([(i / 2 % 20) as f64, (i / 40) as f64, 0.], (i % 2) as usize),
                0.,
                0.,
            )
            .unwrap()
        })
        .collect();
    let mut incremental = SpatialIndex::new(vec![]).unwrap();
    for o in &observations {
        incremental.append(*o).unwrap();
    }
    let bulk = SpatialIndex::new(observations.clone()).unwrap();
    for center in [[2., 2., 0.], [5., 3., 0.], [6., 12., 0.]] {
        for maximum in [2, 7, 16, 32, 67] {
            let query = Neighborhood {
                maximum,
                radius: 8.,
            };
            let cancel = Cancellation::default();
            let mut candidates: Vec<_> = observations
                .iter()
                .map(|o| (point(center).distance(o.site().support.center()), o.id()))
                .filter(|(d, _)| *d <= query.radius)
                .collect();
            candidates.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            let expected: Vec<_> = candidates
                .into_iter()
                .take(maximum)
                .map(|(_, id)| id)
                .collect();
            for index in [&bulk, &incremental] {
                let actual: Vec<_> = index
                    .nearest(point(center), query, &cancel)
                    .unwrap()
                    .iter()
                    .map(Observation::id)
                    .collect();
                assert_eq!(actual, expected, "center={center:?}, k={maximum}");
            }
        }
    }
}

#[test]
fn estimation_reuses_conditioning_across_batches_without_changing_parallel_results() {
    use terrustrial_core::spatial::ObservationSource;
    struct Counted {
        model: Lmc,
        rows: AtomicUsize,
    }
    impl Covariance for Counted {
        fn variables(&self) -> usize {
            self.model.variables()
        }
        fn point_covariance(&self, a: Point3, va: usize, b: Point3, vb: usize) -> Result<f64> {
            self.model.point_covariance(a, va, b, vb)
        }
        fn covariance_row(&self, a: &Site, data: &[Observation], out: &mut [f64]) -> Result<()> {
            self.rows.fetch_add(1, Ordering::Relaxed);
            self.model.covariance_row(a, data, out)
        }
        fn covariance_column(&self, data: &[Observation], b: &Site, out: &mut [f64]) -> Result<()> {
            self.model.covariance_column(data, b, out)
        }
    }
    let model = Counted {
        model: model(),
        rows: AtomicUsize::new(0),
    };
    let source = SpatialIndex::new(hard()).unwrap();
    let cancel = Cancellation::default();
    let budget = MemoryBudget::new(1 << 28);
    let query = Neighborhood {
        maximum: 24,
        radius: 100.,
    };
    let mut selected = source.nearest(point([0.25; 3]), query, &cancel).unwrap();
    selected.sort_unstable_by_key(Observation::id);
    let values: Vec<_> = selected.iter().map(Observation::value).collect();
    let means: Vec<_> = selected.iter().map(|o| trend(o.site()).unwrap()).collect();
    let groups: Vec<_> = (0..65)
        .map(|i| {
            let offset = (1 + i % 8) as f64 / 256.;
            let center = (i * 13 % 37) as f64 * 0.07;
            Group {
                id: i,
                targets: vec![
                    Target {
                        id: 1000 + i * 2,
                        site: site([center - offset, 0.25, 0.25], 0),
                    },
                    Target {
                        id: 1001 + i * 2,
                        site: site([center + offset, 0.25, 0.25], 1),
                    },
                ],
            }
        })
        .collect();
    let drift = ExternalDrift::new(3, |s: &Site, out: &mut [f64]| {
        out[0] = if s.variable == 0 { 1. } else { 0. };
        out[1] = if s.variable == 1 { 1. } else { 0. };
        out[2] = s.support.center().coordinates()[0];
        Ok(())
    })
    .unwrap();
    for uncertainty in [Uncertainty::None, Uncertainty::Joint] {
        let mut reference = Vec::new();
        for group in &groups {
            let sites: Vec<_> = group.targets.iter().map(|t| t.site).collect();
            let plan = GaussianPlan::prepare(
                &model.model,
                &selected,
                &sites,
                &drift,
                PrepareOptions {
                    uncertainty,
                    tolerance: NumericalTolerance::default(),
                    budget: &budget,
                    cancellation: &cancel,
                },
            )
            .unwrap();
            let predicted = centered_reference(
                &plan,
                &values,
                &means,
                &sites.iter().map(|s| trend(s).unwrap()).collect::<Vec<_>>(),
            )
            .unwrap();
            let covariance = plan.covariance().map(|c| {
                (0..2)
                    .flat_map(|j| (0..2).map(move |i| c[(i, j)].to_bits()))
                    .collect::<Vec<_>>()
            });
            reference.push((
                predicted.into_iter().map(f64::to_bits).collect::<Vec<_>>(),
                covariance,
            ));
        }
        for (workers, batch) in [(1, 1), (1, 16), (4, 16), (4, 7)] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(workers)
                .build()
                .unwrap();
            let mut actual = Vec::new();
            model.rows.store(0, Ordering::Relaxed);
            pool.install(|| {
                terrustrial_core::estimation::estimate_groups(
                    &model,
                    &source,
                    groups.clone(),
                    &drift,
                    EstimationOptions {
                        neighborhood: query.into(),
                        mean: Some(&trend),
                        uncertainty,
                        tolerance: NumericalTolerance::default(),
                        budget: &budget,
                        cancellation: &cancel,
                    },
                    execution(batch),
                    |result| {
                        let covariance = result.covariance().map(|c| {
                            (0..2)
                                .flat_map(|j| (0..2).map(move |i| c[(i, j)].to_bits()))
                                .collect::<Vec<_>>()
                        });
                        actual.push((
                            result
                                .values
                                .iter()
                                .map(|v| v.to_bits())
                                .collect::<Vec<_>>(),
                            covariance,
                        ));
                        Ok(())
                    },
                )
            })
            .unwrap();
            let rows = model.rows.load(Ordering::Relaxed);
            assert!(rows >= selected.len() && rows <= workers.min(batch) * selected.len(),
                "at most one conditioning build per participating worker, including across admission windows");
            assert_eq!(actual, reference);
            assert_eq!(budget.used(), 0);
        }
    }
}

#[test]
fn large_neighborhood_validation_preserves_exact_aliases_and_joint_results() {
    use terrustrial_core::spatial::ObservationSource;
    let mut records: Vec<_> = (0..160)
        .map(|i| {
            Observation::new(
                i,
                site(
                    [(i % 7) as f64, ((i / 7) % 7) as f64, (i / 49) as f64],
                    (i % 2) as usize,
                ),
                (i as f64 * 0.3).sin(),
                if i % 3 == 0 { 0. } else { 0.01 },
            )
            .unwrap()
        })
        .collect();
    for i in (0..30).step_by(3) {
        let o = &records[i];
        records.push(Observation::new(200 + i as u64, *o.site(), o.value(), 0.).unwrap());
    }
    let source = SpatialIndex::new(records).unwrap();
    let model = model();
    let budget = MemoryBudget::new(1 << 29);
    let cancel = Cancellation::default();
    let query = Neighborhood {
        maximum: 200,
        radius: 100.,
    };
    let groups: Vec<_> = (0..13)
        .map(|i| Group {
            id: i,
            targets: (0..2)
                .map(|v| Target {
                    id: 1000 + 2 * i + v as u64,
                    site: site([0.13 + i as f64 * 0.1, 0.17, 0.23], v),
                })
                .collect(),
        })
        .collect();
    let mut expected = Vec::new();
    for group in &groups {
        let mut selected: Vec<Observation> = Vec::new();
        let center = group.targets[0].site.support.center();
        // Independent source selection and stable exact-support coalescing.
        for observation in source.nearest(center, query, &cancel).unwrap() {
            if observation.noise_variance() != 0.
                || !selected
                    .iter()
                    .any(|o| o.noise_variance() == 0. && o.site() == observation.site())
            {
                selected.push(observation);
            }
        }
        assert_eq!(selected.len(), 160);
        selected.sort_unstable_by_key(Observation::id);
        let sites: Vec<_> = group.targets.iter().map(|t| t.site).collect();
        let plan = GaussianPlan::prepare(
            &model,
            &selected,
            &sites,
            &Ordinary { variables: 2 },
            PrepareOptions {
                uncertainty: Uncertainty::Joint,
                tolerance: NumericalTolerance::default(),
                budget: &budget,
                cancellation: &cancel,
            },
        )
        .unwrap();
        let mut bits: Vec<_> = plan
            .estimate(&selected.iter().map(Observation::value).collect::<Vec<_>>())
            .unwrap()
            .into_iter()
            .map(f64::to_bits)
            .collect();
        let covariance = plan.covariance().unwrap();
        for j in 0..2 {
            for i in 0..2 {
                bits.push(covariance[(i, j)].to_bits());
            }
        }
        expected.push(bits);
    }
    for workers in [1, 2, 4, 8] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .unwrap();
        let mut actual = Vec::new();
        let limits = ExecutionOptions {
            batch_groups: 8,
            maximum_targets_per_group: 2,
            bytes_per_group: 1 << 23,
        };
        pool.install(|| {
            terrustrial_core::estimation::estimate_groups(
                &model,
                &source,
                groups.clone(),
                &Ordinary { variables: 2 },
                EstimationOptions {
                    neighborhood: query.into(),
                    mean: None,
                    uncertainty: Uncertainty::Joint,
                    tolerance: NumericalTolerance::default(),
                    budget: &budget,
                    cancellation: &cancel,
                },
                limits,
                |result| {
                    let mut bits: Vec<_> = result.values.iter().map(|v| v.to_bits()).collect();
                    let covariance = result.covariance().unwrap();
                    for j in 0..2 {
                        for i in 0..2 {
                            bits.push(covariance[(i, j)].to_bits());
                        }
                    }
                    actual.push(bits);
                    Ok(())
                },
            )
        })
        .unwrap();
        assert_eq!(expected, actual);
        assert_eq!(budget.used(), 0);
    }
}

#[test]
fn near_collinear_estimation_preserves_bits_across_workers_and_batches() {
    for (variances, noise) in [
        ([1_f64; 2], 0.),
        ([1e-308; 2], 0.),
        ([1e300; 2], 0.),
        ([1e308; 2], 1e308),
        ([1e-300, 1e300], 0.),
        ([1e300, 1e-300], 0.),
    ] {
        let roots = variances.map(f64::sqrt);
        let model = Lmc::new(vec![Structure {
            correlation: Correlation::Exponential,
            anisotropy: Anisotropy::isotropic(3.).unwrap(),
            coregionalization: Coregionalization::from_factor(
                2,
                2,
                &[roots[0], 0., 0.6 * roots[1], 0.8 * roots[1]],
            )
            .unwrap(),
        }])
        .unwrap();
        let observations: Vec<_> = hard()
            .into_iter()
            .map(|o| Observation::new(o.id(), *o.site(), o.value(), noise).unwrap())
            .collect();
        let mut groups = path();
        if noise > 0. {
            // Keep target variances representable while the observation
            // covariance plus noise overflows in physical units.
            for group in &mut groups {
                for target in &mut group.targets {
                    target.site = site(
                        target.site.support.center().coordinates().map(|v| v * 0.2),
                        target.site.variable,
                    );
                }
            }
        }
        let drift = ExternalDrift::new(4, |s: &Site, out: &mut [f64]| {
            out.fill(0.);
            out[2 * s.variable] = 1.;
            out[2 * s.variable + 1] = 1. + 2_f64.powi(-24) * s.support.center().coordinates()[0];
            Ok(())
        })
        .unwrap();
        let budget = MemoryBudget::new(1 << 28);
        let cancel = Cancellation::default();
        let mut reference = None;
        for workers in [1, 2, 4, 8] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(workers)
                .build()
                .unwrap();
            for batch in [1, 8, 64] {
                let mut actual = Vec::new();
                pool.install(|| {
                    terrustrial_core::estimation::estimate_groups(
                        &model,
                        observations.as_slice(),
                        groups.clone(),
                        &drift,
                        EstimationOptions {
                            neighborhood: Neighborhood {
                                maximum: 100,
                                radius: 100.,
                            }
                            .into(),
                            mean: Some(&trend),
                            uncertainty: Uncertainty::Joint,
                            tolerance: NumericalTolerance::default(),
                            budget: &budget,
                            cancellation: &cancel,
                        },
                        execution(batch),
                        |result| {
                            actual.push(result.group.id);
                            actual.extend(result.values.iter().map(|v| v.to_bits()));
                            for column in result.covariance().unwrap().col_iter() {
                                actual.extend(column.iter().map(|v| v.to_bits()));
                            }
                            Ok(())
                        },
                    )
                })
                .unwrap();
                if let Some(reference) = &reference {
                    assert_eq!(reference, &actual, "workers={workers}, batch={batch}");
                } else {
                    reference = Some(actual);
                }
                assert_eq!(budget.used(), 0);
            }
        }
    }
}
