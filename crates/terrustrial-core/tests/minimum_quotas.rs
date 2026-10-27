use terrustrial_core::{
    data::{Observation, Point3, Site},
    execution::{Cancellation, MemoryBudget},
    model::Anisotropy,
    spatial::{
        MinimumQuotas, ObservationSource, SearchNeighborhood, SearchWorkspace, SpatialIndex,
        VariableLimit, VariableMinimum,
    },
    Error,
};
fn row(id: u64, variable: usize, octant: usize, distance: f64) -> Observation {
    Observation::new(
        id,
        Site::point(
            Point3::new([if octant == 0 { distance } else { -distance }, 0., 0.]).unwrap(),
            variable,
        ),
        id as f64,
        0.,
    )
    .unwrap()
}
fn query(
    maximum: usize,
    varmax: usize,
    octmax: usize,
    minimums: MinimumQuotas,
) -> SearchNeighborhood {
    SearchNeighborhood::new(
        maximum,
        Anisotropy::isotropic(10.).unwrap(),
        Some(octmax),
        (0..3)
            .map(|variable| VariableLimit {
                variable,
                maximum: varmax,
            })
            .collect(),
    )
    .unwrap()
    .with_minimums(minimums)
    .unwrap()
}
fn ids(rows: &[Observation]) -> Vec<u64> {
    rows.iter().map(Observation::id).collect()
}
// Enumerate every subset of the complete input. No production bucket pruning,
// quota feasibility or flow code is used. Prefer size, then lexicographic ranks.
fn oracle(
    rows: &[Observation],
    maximum: usize,
    varmax: usize,
    octmax: usize,
    minimums: &MinimumQuotas,
) -> Option<Vec<u64>> {
    let mut ordered: Vec<_> = rows
        .iter()
        .filter(|o| o.site().support.center().coordinates()[0].abs() <= 10.)
        .collect();
    ordered.sort_unstable_by(|a, b| {
        a.site().support.center().coordinates()[0]
            .abs()
            .total_cmp(&b.site().support.center().coordinates()[0].abs())
            .then(a.id().cmp(&b.id()))
    });
    let mut best: Option<Vec<usize>> = None;
    for mask in 0..1_usize << ordered.len() {
        let count = mask.count_ones() as usize;
        if count < minimums.total || count > maximum {
            continue;
        }
        let mut variables = [0; 3];
        let mut sectors = [0; 8];
        let mut ranks = Vec::new();
        for (i, o) in ordered.iter().enumerate() {
            if mask & (1 << i) != 0 {
                variables[o.site().variable] += 1;
                sectors[usize::from(o.site().support.center().coordinates()[0] < 0.)] += 1;
                ranks.push(i);
            }
        }
        if variables.iter().any(|&n| n > varmax)
            || sectors.iter().any(|&n| n > octmax)
            || minimums
                .variables
                .iter()
                .any(|m| variables[m.variable] < m.minimum)
            || sectors
                .iter()
                .zip(minimums.octants)
                .any(|(&n, min)| n < min)
        {
            continue;
        }
        if best
            .as_ref()
            .is_none_or(|b| ranks.len() > b.len() || (ranks.len() == b.len() && ranks < *b))
        {
            best = Some(ranks);
        }
    }
    best.map(|ranks| ranks.into_iter().map(|i| ordered[i].id()).collect())
}
#[test]
fn exhaustive_crossing_quota_graphs_match_global_subset_oracle() {
    let budget = MemoryBudget::new(1 << 20);
    let cancel = Cancellation::default();
    let point = Point3::new([0.; 3]).unwrap();
    for pattern in 0..64_u64 {
        let rows: Vec<_> = (0..6)
            .filter(|i| pattern & (1 << i) != 0)
            .map(|i| {
                row(
                    u64::MAX - i,
                    (i / 2) as usize,
                    (i % 2) as usize,
                    0.25 * (i + 1) as f64,
                )
            })
            .collect();
        for var_mask in 0..8 {
            for oct_mask in 0..4 {
                let minimums = MinimumQuotas {
                    total: 1,
                    variables: (0..3)
                        .map(|variable| VariableMinimum {
                            variable,
                            minimum: usize::from(var_mask & (1 << variable) != 0),
                        })
                        .collect(),
                    octants: [oct_mask & 1, (oct_mask >> 1) & 1, 0, 0, 0, 0, 0, 0],
                };
                let expected = oracle(&rows, 3, 1, 1, &minimums);
                let query = query(3, 1, 1, minimums);
                let mut workspace = SearchWorkspace::new(&query, &budget).unwrap();
                for reverse in [false, true] {
                    let mut input = rows.clone();
                    if reverse {
                        input.reverse();
                    }
                    let result = input.as_slice().search(point, &mut workspace, &cancel);
                    match &expected {
                        Some(expected) => assert_eq!(
                            &ids(result.unwrap()),
                            expected,
                            "pattern={pattern},var={var_mask},oct={oct_mask}"
                        ),
                        None => assert!(
                            matches!(result, Err(Error::InsufficientNeighborhood)),
                            "pattern={pattern},var={var_mask},oct={oct_mask}"
                        ),
                    }
                    let report = workspace.diagnostics();
                    assert!(report.complete);
                    assert_eq!(report.minimums_satisfied, expected.is_some());
                    assert_eq!(report.selected, expected.as_ref().map_or(0, Vec::len));
                }
            }
        }
        assert_eq!(budget.used(), 0);
    }
}
#[test]
fn nearest_greedy_trap_and_bucket_pruning_preserve_feasible_lexicographic_result() {
    let budget = MemoryBudget::new(1 << 20);
    let cancel = Cancellation::default();
    let point = Point3::new([0.; 3]).unwrap();
    // Closest A+ exhausts both A and + caps. B+ and A- together are feasible.
    let rows = [row(0, 0, 0, 0.1), row(1, 1, 0, 0.2), row(2, 0, 1, 0.3)];
    let base = SearchNeighborhood::new(
        2,
        Anisotropy::isotropic(10.).unwrap(),
        Some(1),
        vec![
            VariableLimit {
                variable: 0,
                maximum: 1,
            },
            VariableLimit {
                variable: 1,
                maximum: 1,
            },
        ],
    )
    .unwrap();
    let mut greedy = SearchWorkspace::new(&base, &budget).unwrap();
    assert_eq!(
        ids(rows.as_slice().search(point, &mut greedy, &cancel).unwrap()),
        [0]
    );
    let constrained = base
        .clone()
        .with_minimums(MinimumQuotas {
            total: 2,
            ..Default::default()
        })
        .unwrap();
    let mut workspace = SearchWorkspace::new(&constrained, &budget).unwrap();
    assert_eq!(
        ids(rows
            .as_slice()
            .search(point, &mut workspace, &cancel)
            .unwrap()),
        [1, 2]
    );
    let data: Vec<_> = (0..15)
        .map(|i| {
            row(
                if i == 0 { u64::MAX } else { i },
                (i % 3) as usize,
                ((i / 3) % 2) as usize,
                if i < 2 { 0.2 } else { i as f64 * 0.25 },
            )
        })
        .collect();
    let minimums = MinimumQuotas {
        total: 3,
        variables: vec![VariableMinimum {
            variable: 2,
            minimum: 1,
        }],
        octants: [1, 1, 0, 0, 0, 0, 0, 0],
    };
    let expected = oracle(&data, 4, 2, 2, &minimums).unwrap();
    let constrained = query(4, 2, 2, minimums);
    let mut workspace = SearchWorkspace::new(&constrained, &budget).unwrap();
    assert_eq!(
        ids(data
            .as_slice()
            .search(point, &mut workspace, &cancel)
            .unwrap()),
        expected
    );
    assert!(workspace.diagnostics().retained_candidates < data.len());
    let index = SpatialIndex::new(data.clone()).unwrap();
    let mut result = Ok(());
    let allocations = allocation_counter::measure(|| {
        for _ in 0..32 {
            result = index
                .search(point, &mut workspace, &cancel)
                .map(|actual| assert!(ids_without_allocation(actual, &expected)));
            if result.is_err() {
                break;
            }
        }
    });
    result.unwrap();
    assert_eq!(allocations.count_total, 0);
    fn ids_without_allocation(actual: &[Observation], expected: &[u64]) -> bool {
        actual.len() == expected.len() && actual.iter().zip(expected).all(|(a, &b)| a.id() == b)
    }
}
#[test]
fn invalid_quotas_resource_failures_cancellation_and_reuse_are_explicit() {
    let base = || {
        SearchNeighborhood::new(
            3,
            Anisotropy::isotropic(2.).unwrap(),
            Some(1),
            vec![VariableLimit {
                variable: 0,
                maximum: 2,
            }],
        )
        .unwrap()
    };
    for minimums in [
        MinimumQuotas {
            total: 4,
            ..Default::default()
        },
        MinimumQuotas {
            variables: vec![VariableMinimum {
                variable: 0,
                minimum: 3,
            }],
            ..Default::default()
        },
        MinimumQuotas {
            variables: vec![VariableMinimum {
                variable: 1,
                minimum: 1,
            }],
            ..Default::default()
        },
        MinimumQuotas {
            variables: vec![
                VariableMinimum {
                    variable: 0,
                    minimum: 1
                };
                2
            ],
            ..Default::default()
        },
        MinimumQuotas {
            octants: [2, 0, 0, 0, 0, 0, 0, 0],
            ..Default::default()
        },
        MinimumQuotas {
            octants: [1; 8],
            ..Default::default()
        },
    ] {
        assert!(base().with_minimums(minimums).is_err());
    }
    let q = base()
        .with_minimums(MinimumQuotas {
            total: 1,
            ..Default::default()
        })
        .unwrap();
    let budget = MemoryBudget::new(1 << 20);
    let cancel = Cancellation::default();
    let point = Point3::new([0.; 3]).unwrap();
    let mut workspace = SearchWorkspace::new(&q, &budget).unwrap();
    let bytes = workspace.accounted_bytes();
    assert_eq!(budget.used(), bytes);
    let small = MemoryBudget::new(bytes - 1);
    assert!(matches!(
        SearchWorkspace::new(&q, &small),
        Err(Error::MemoryBudget { .. })
    ));
    assert_eq!(small.used(), 0);
    assert!(matches!(
        [].as_slice().search(point, &mut workspace, &cancel),
        Err(Error::InsufficientNeighborhood)
    ));
    let data = [row(1, 0, 0, 0.5)];
    let stopped = Cancellation::default();
    stopped.cancel();
    assert!(matches!(
        data.as_slice().search(point, &mut workspace, &stopped),
        Err(Error::Cancelled)
    ));
    assert!(!workspace.diagnostics().complete);
    assert_eq!(
        ids(data
            .as_slice()
            .search(point, &mut workspace, &cancel)
            .unwrap()),
        [1]
    );
    drop(workspace);
    assert_eq!(budget.used(), 0);
}

#[test]
fn multi_capacity_completion_matches_full_subset_reference() {
    let budget = MemoryBudget::new(1 << 20);
    let cancel = Cancellation::default();
    let point = Point3::new([0.; 3]).unwrap();
    // Deterministic varied capacities, non-unit lower bounds, ties and pruning.
    // Every result is compared with all subsets of the original input.
    for case in 0..256_u64 {
        let maximum = 2 + case as usize % 5;
        let varmax = 1 + (case / 5) as usize % 3;
        let octmax = 1 + (case / 15) as usize % 4;
        let rows: Vec<_> = (0..10_u64)
            .map(|i| {
                row(
                    u64::MAX - i,
                    ((i * 7 + case) % 3) as usize,
                    ((i * i + case * i / 3) % 2) as usize,
                    0.25 * (1 + (i * 11 + case * 3) % 17) as f64,
                )
            })
            .collect();
        let minimums = MinimumQuotas {
            total: 1 + (case / 3) as usize % maximum,
            variables: vec![VariableMinimum {
                variable: (case % 3) as usize,
                minimum: (case / 7) as usize % (varmax.min(maximum) + 1),
            }],
            octants: [
                (case / 11) as usize % (octmax.min(maximum) + 1),
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
        };
        let expected = oracle(&rows, maximum, varmax, octmax, &minimums);
        let query = query(maximum, varmax, octmax, minimums);
        let mut workspace = SearchWorkspace::new(&query, &budget).unwrap();
        match expected {
            Some(expected) => assert_eq!(
                ids(rows.search(point, &mut workspace, &cancel).unwrap()),
                expected,
                "case={case}"
            ),
            None => assert!(
                matches!(
                    rows.search(point, &mut workspace, &cancel),
                    Err(Error::InsufficientNeighborhood)
                ),
                "case={case}"
            ),
        }
    }
    assert_eq!(budget.used(), 0);
}

#[test]
fn octant_minimums_without_maxima_use_all_frame_signs_and_reset_cleanly() {
    let cancel = Cancellation::default();
    let budget = MemoryBudget::new(1 << 20);
    let point = Point3::new([0.; 3]).unwrap();
    // A signed permutation makes world octants different from frame octants.
    let geometry = Anisotropy::new([[0., 1., 0.], [0., 0., -1.], [-1., 0., 0.]], [10.; 3]).unwrap();
    let rows: Vec<_> = (0..8)
        .map(|octant| {
            let local: [f64; 3] =
                std::array::from_fn(|axis| if octant & (1 << axis) != 0 { -1. } else { 1. });
            Observation::new(
                octant,
                Site::point(
                    Point3::new([-local[2], local[0], -local[1]]).unwrap(),
                    usize::MAX,
                ),
                octant as f64,
                0.,
            )
            .unwrap()
        })
        .collect();
    let base = SearchNeighborhood::new(8, geometry, None, vec![]).unwrap();
    for sector in 0..8 {
        let mut octants = [0; 8];
        octants[sector] = 1;
        let query = base
            .clone()
            .with_minimums(MinimumQuotas {
                octants,
                ..Default::default()
            })
            .unwrap();
        assert_eq!(query.octant_maximum(), None);
        let mut workspace = SearchWorkspace::new(&query, &budget).unwrap();
        assert_eq!(
            ids(rows.search(point, &mut workspace, &cancel).unwrap()),
            (0..8).collect::<Vec<_>>()
        );
        let absent: Vec<_> = rows
            .iter()
            .copied()
            .filter(|o| o.id() != sector as u64)
            .collect();
        assert!(matches!(
            absent.search(point, &mut workspace, &cancel),
            Err(Error::InsufficientNeighborhood)
        ));
        let cleared = query
            .clone()
            .with_minimums(MinimumQuotas::default())
            .unwrap();
        assert!(cleared.minimums().is_none());
        assert_eq!(cleared.octant_maximum(), None);
        let mut cleared = SearchWorkspace::new(&cleared, &budget).unwrap();
        assert_eq!(
            absent.search(point, &mut cleared, &cancel).unwrap().len(),
            7
        );
    }
    let query = base
        .with_minimums(MinimumQuotas {
            total: 8,
            ..Default::default()
        })
        .unwrap();
    let mut workspace = SearchWorkspace::new(&query, &budget).unwrap();
    assert_eq!(
        rows.search(point, &mut workspace, &cancel).unwrap().len(),
        8
    );
}

#[cfg(feature = "sqlite")]
#[test]
fn point_and_block_payloads_match_across_sqlite_readers_and_resident_search() {
    use terrustrial_core::{
        data::Support,
        storage::{SqliteReadPool, SqliteStore},
    };
    let budget = MemoryBudget::new(32 << 20);
    let cancel = Cancellation::default();
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("quota.sqlite");
    let mut store = SqliteStore::create(&path, 4096, &budget).unwrap();
    let mut data: Vec<_> = (0..15)
        .map(|i| {
            row(
                i,
                (i % 3) as usize,
                ((i / 3) % 2) as usize,
                0.25 * (i + 1) as f64,
            )
        })
        .collect();
    for (i, observation) in data.iter_mut().enumerate().filter(|(i, _)| i % 2 == 1) {
        let mut site = *observation.site();
        site.support = Support::block(site.support.center(), [0.2, 0.1, 0.], [2, 3, 1]).unwrap();
        *observation = Observation::new(
            observation.id(),
            site,
            if i == 1 { -0. } else { i as f64 },
            0.01,
        )
        .unwrap();
    }
    store.import(data.as_slice(), &cancel).unwrap();
    let pool = SqliteReadPool::open(&path, 2, 4096, &budget).unwrap();
    let snapshot = SqliteReadPool::open_snapshot(&path, 2, 4096, &budget, &cancel).unwrap();
    let index = SpatialIndex::new(data.clone()).unwrap();
    let minimums = MinimumQuotas {
        total: 4,
        variables: vec![VariableMinimum {
            variable: 2,
            minimum: 1,
        }],
        octants: [1, 1, 0, 0, 0, 0, 0, 0],
    };
    let expected = oracle(&data, 4, 2, 2, &minimums).unwrap();
    let query = query(4, 2, 2, minimums);
    let mut workspace = SearchWorkspace::new(&query, &budget).unwrap();
    for source in [&store as &dyn ObservationSource, &pool, &snapshot, &index] {
        let actual = source
            .search(Point3::new([0.; 3]).unwrap(), &mut workspace, &cancel)
            .unwrap();
        assert_eq!(ids(actual), expected);
        for actual in actual {
            let original = data.iter().find(|o| o.id() == actual.id()).unwrap();
            assert_eq!(actual.site(), original.site());
            assert_eq!(actual.value().to_bits(), original.value().to_bits());
            assert_eq!(
                actual.noise_variance().to_bits(),
                original.noise_variance().to_bits()
            );
        }
        assert!(matches!(
            source.search(Point3::new([100.; 3]).unwrap(), &mut workspace, &cancel),
            Err(Error::InsufficientNeighborhood)
        ));
    }
}

#[test]
fn gaussian_drivers_preserve_quota_selection_and_propagate_infeasibility() {
    gaussian_quota_drivers(false);
}

#[test]
fn adaptive_gaussian_drivers_preserve_bits_and_propagate_infeasibility() {
    gaussian_quota_drivers(true);
}

fn gaussian_quota_drivers(adaptive: bool) {
    use terrustrial_core::{
        data::{Group, Target},
        estimation::EstimationOptions,
        execution::ExecutionOptions,
        gaussian::{NumericalTolerance, Simple, Uncertainty},
        model::{Coregionalization, Correlation, Lmc, Structure},
        simulation::{self, parallel, SimulationOptions},
        spatial::NeighborhoodQuery,
    };
    let budget = MemoryBudget::new(128 << 20);
    let cancel = Cancellation::default();
    let model = Lmc::new(vec![Structure {
        correlation: Correlation::Exponential,
        anisotropy: Anisotropy::isotropic(3.).unwrap(),
        coregionalization: Coregionalization::from_factor(2, 2, &[1., 0., -0.3, 0.8]).unwrap(),
    }])
    .unwrap();
    let hard: Vec<_> = (0..8)
        .map(|i| {
            row(
                i,
                (i % 2) as usize,
                ((i / 2) % 2) as usize,
                (i + 1) as f64 * 0.1,
            )
        })
        .collect();
    let groups: Vec<_> = (0..6)
        .map(|i| Group {
            id: i,
            targets: (0..2)
                .map(|v| Target {
                    id: 100 + 2 * i + v as u64,
                    site: Site::point(Point3::new([0.03 * i as f64, 0.1, 0.]).unwrap(), v),
                })
                .collect(),
        })
        .collect();
    let make = |total| {
        let query = SearchNeighborhood::new(
            4,
            Anisotropy::isotropic(if adaptive { 0.05 } else { 10. }).unwrap(),
            None,
            (0..2)
                .map(|variable| VariableLimit {
                    variable,
                    maximum: 2,
                })
                .collect(),
        )
        .unwrap()
        .with_minimums(MinimumQuotas {
            total,
            variables: vec![
                VariableMinimum {
                    variable: 0,
                    minimum: 1,
                },
                VariableMinimum {
                    variable: 1,
                    minimum: 1,
                },
            ],
            ..Default::default()
        })
        .unwrap();
        if adaptive {
            query
                .with_expansion(vec![[0.2; 3], [0.4; 3], [1.; 3], [2.; 3], [10.; 3]])
                .unwrap()
        } else {
            query
        }
    };
    let hard_query = make(4);
    let sim_query = make(2);
    let execution = |batch_groups| ExecutionOptions {
        batch_groups,
        maximum_targets_per_group: 2,
        bytes_per_group: 1 << 20,
    };
    let options = || SimulationOptions {
        seed: 123,
        realization: 4,
        mean: None,
        hard_neighborhood: NeighborhoodQuery::Constrained(&hard_query),
        simulated_neighborhood: NeighborhoodQuery::Constrained(&sim_query),
        budget: &budget,
        cancellation: &cancel,
    };
    // Existing conditioning seeds satisfy simulated-source minima for group zero.
    let seeds = vec![row(9000, 0, 1, 0.95), row(9001, 1, 0, 0.95)];
    let mut serial = Vec::new();
    simulation::simulate_groups(
        &model,
        hard.as_slice(),
        &mut SpatialIndex::new(seeds.clone()).unwrap(),
        groups.clone(),
        &Simple,
        options(),
        |g| {
            serial.extend(g.iter().map(|o| o.value().to_bits()));
            Ok(())
        },
    )
    .unwrap();
    let mut estimate_reference = None;
    for workers in [1, 4] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .unwrap();
        for batch in [1, 4] {
            let mut simulated = Vec::new();
            pool.install(|| {
                parallel::simulate_groups(
                    &model,
                    hard.as_slice(),
                    &mut SpatialIndex::new(seeds.clone()).unwrap(),
                    groups.clone(),
                    &Simple,
                    options(),
                    execution(batch),
                    |g| {
                        simulated.extend(g.iter().map(|o| o.value().to_bits()));
                        Ok(())
                    },
                )
            })
            .unwrap();
            assert_eq!(simulated, serial);
            let mut estimated = Vec::new();
            pool.install(|| {
                terrustrial_core::estimation::estimate_groups(
                    &model,
                    hard.as_slice(),
                    groups.clone(),
                    &Simple,
                    EstimationOptions {
                        neighborhood: NeighborhoodQuery::Constrained(&hard_query),
                        mean: None,
                        uncertainty: Uncertainty::Joint,
                        tolerance: NumericalTolerance::default(),
                        budget: &budget,
                        cancellation: &cancel,
                    },
                    execution(batch),
                    |g| {
                        estimated.extend(g.values.iter().map(|v| v.to_bits()));
                        let c = g.covariance().unwrap();
                        for j in 0..c.ncols() {
                            for i in 0..c.nrows() {
                                estimated.push(c[(i, j)].to_bits());
                            }
                        }
                        Ok(())
                    },
                )
            })
            .unwrap();
            if let Some(expected) = &estimate_reference {
                assert_eq!(&estimated, expected);
            } else {
                estimate_reference = Some(estimated);
            }
        }
    }
    // Minimums apply to each source independently. An empty simulation store
    // cannot satisfy positive simulated-source minima, even with enough hard data.
    for parallel in [false, true] {
        let mut output = SpatialIndex::new(vec![]).unwrap();
        let mut calls = 0;
        let sink = |_: &[Observation]| {
            calls += 1;
            Ok(())
        };
        let result = if parallel {
            parallel::simulate_groups(
                &model,
                hard.as_slice(),
                &mut output,
                groups.clone(),
                &Simple,
                options(),
                execution(4),
                sink,
            )
        } else {
            simulation::simulate_groups(
                &model,
                hard.as_slice(),
                &mut output,
                groups.clone(),
                &Simple,
                options(),
                sink,
            )
        };
        assert!(matches!(result, Err(Error::InsufficientNeighborhood)));
        assert_eq!(calls, 0);
        let mut count = 0;
        output
            .scan(&mut |_| {
                count += 1;
                Ok(())
            })
            .unwrap();
        assert_eq!(count, 0);
    }
    let mut calls = 0;
    let result = terrustrial_core::estimation::estimate_groups(
        &model,
        &hard[..3],
        groups,
        &Simple,
        EstimationOptions {
            neighborhood: NeighborhoodQuery::Constrained(&hard_query),
            mean: None,
            uncertainty: Uncertainty::Joint,
            tolerance: NumericalTolerance::default(),
            budget: &budget,
            cancellation: &cancel,
        },
        execution(4),
        |_| {
            calls += 1;
            Ok(())
        },
    );
    assert!(matches!(result, Err(Error::InsufficientNeighborhood)));
    assert_eq!(calls, 0);
    assert_eq!(budget.used(), 0);
}
