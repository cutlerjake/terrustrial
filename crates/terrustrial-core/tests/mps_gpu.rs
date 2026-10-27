#![cfg(feature = "gpu")]
use std::sync::Arc;
use terrustrial_core::{
    compute::gpu::Context,
    execution::{Cancellation, MemoryBudget},
    grid::{Boundary, Event, Grid, Shape, Variable},
    mps::{
        self,
        gpu::{Matcher, Query},
        MatchOptions,
    },
};

// Opt-in hardware test: CI without a device still compiles the shader host path.
// Execute explicitly with --ignored on each backend; never count a skip as evidence.
#[test]
#[ignore = "requires a native compute adapter"]
fn portable_gpu_matches_all_cpu_scores_and_strict_validity() {
    let context = Arc::new(Context::new().expect("native GPU adapter required"));
    eprintln!("adapter: {:?}", context.adapter_info());
    let budget = MemoryBudget::new(1 << 27);
    let cancel = Cancellation::default();
    let shape = Shape::new([7, 5, 3]).unwrap();
    let training = Grid::new(
        shape,
        vec![
            Variable::Continuous { scale: 0.125 },
            Variable::Categorical { classes: u32::MAX },
        ],
        (0..shape.cells())
            .flat_map(|i| {
                [
                    if i % 17 == 3 {
                        None
                    } else {
                        Some(1e14 + (i as f64 * 0.25).sin() * 4.0)
                    },
                    if i % 13 == 2 {
                        None
                    } else {
                        Some(if i % 3 == 0 { 4_000_000_000.0 } else { 5.0 })
                    },
                ]
            })
            .collect(),
        &budget,
    )
    .unwrap();
    let mut matcher = Matcher::new(&training, context, &budget, &cancel).unwrap();
    let events = [
        Event {
            offset: [-1, 0, 0],
            variable: 0,
            value: 1e14 + 0.375,
            weight: 1.0,
        },
        Event {
            offset: [0, 1, 0],
            variable: 1,
            value: 4_000_000_000.0,
            weight: 0.5,
        },
        Event {
            offset: [0, 0, 1],
            variable: 0,
            value: 1e14 - 0.125,
            weight: 2.0,
        },
    ];
    let extreme = [Event {
        offset: [i64::MIN, i64::MAX, 0],
        variable: 1,
        value: 5.0,
        weight: 1.0,
    }];
    let queries = [
        Query {
            events: &events,
            requested: &[0, 1],
            boundary: Boundary::Bounded,
        },
        Query {
            events: &events,
            requested: &[0, 1],
            boundary: Boundary::Periodic,
        },
        Query {
            events: &extreme,
            requested: &[0, 1],
            boundary: Boundary::Bounded,
        },
        Query {
            events: &extreme,
            requested: &[0, 1],
            boundary: Boundary::Periodic,
        },
        Query {
            events: &[],
            requested: &[0, 1],
            boundary: Boundary::Bounded,
        },
    ];
    let first = matcher.score_batch(&queries, &budget, &cancel).unwrap();
    let second = matcher.score_batch(&queries, &budget, &cancel).unwrap();
    for (q, query) in queries.iter().enumerate() {
        let config = MatchOptions {
            neighbors: shape.cells(),
            boundary: query.boundary,
            seed: 0,
            realization: 0,
            query_id: 0,
            budget: &budget,
            cancellation: &cancel,
        };
        let reference = mps::best_matches(&training, query.events, query.requested, &config);
        let mut expected = vec![None; shape.cells()];
        if let Ok(matches) = reference {
            for entry in matches {
                expected[entry.index] = Some(entry.mismatch);
            }
        }
        let config = MatchOptions {
            neighbors: 4,
            ..config
        };
        let ranked = mps::best_matches(&training, query.events, query.requested, &config);
        let accelerated =
            matcher.best_matches_approximate(query.events, query.requested, &config, 1e-3);
        match (ranked, accelerated) {
            (Ok(a), Ok(b)) => assert_eq!(
                a.iter()
                    .map(|m| (m.index, m.mismatch.to_bits()))
                    .collect::<Vec<_>>(),
                b.iter()
                    .map(|m| (m.index, m.mismatch.to_bits()))
                    .collect::<Vec<_>>()
            ),
            (Err(_), Err(_)) => {}
            (a, b) => panic!("CPU={a:?}, GPU={b:?}"),
        }
        let actual = first.query(q).unwrap();
        assert_eq!(
            actual,
            second.query(q).unwrap(),
            "same-device repeatability"
        );
        for (i, (cpu, gpu)) in expected.iter().zip(actual.iter()).enumerate() {
            assert_eq!(cpu.is_some(), gpu.is_some(), "validity query={q}, cell={i}");
            if let (Some(cpu), Some(gpu)) = (cpu, gpu) {
                assert!(
                    (cpu - gpu).abs() <= 2e-5 * cpu.abs().max(1.0),
                    "query={q}, cell={i}: cpu={cpu}, gpu={gpu}"
                );
            }
        }
    }
    assert!(matcher
        .score_batch(&queries, &MemoryBudget::new(1), &cancel)
        .is_err());
    cancel.cancel();
    assert!(matcher.score_batch(&queries, &budget, &cancel).is_err());
}

#[test]
fn portable_shader_validates_without_a_device() {
    let module =
        wgpu::naga::front::wgsl::parse_str(include_str!("../src/mps/pattern.wgsl")).unwrap();
    wgpu::naga::valid::Validator::new(
        wgpu::naga::valid::ValidationFlags::all(),
        wgpu::naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .unwrap();
}

#[test]
#[ignore = "requires a native compute adapter"]
fn gpu_joint_simulation_preserves_partial_hard_data_and_repeats() {
    let context = Arc::new(Context::new().unwrap());
    let budget = MemoryBudget::new(1 << 25);
    let cancel = Cancellation::default();
    let variables = vec![
        Variable::Continuous { scale: 1.0 },
        Variable::Continuous { scale: 2.0 },
        Variable::Categorical { classes: 3 },
    ];
    let training = Grid::new(
        Shape::new([13, 7, 1]).unwrap(),
        variables.clone(),
        (0..91)
            .flat_map(|i| {
                [
                    Some(i as f64 / 10.0),
                    Some(i as f64 / 5.0),
                    Some((i % 3) as f64),
                ]
            })
            .collect(),
        &budget,
    )
    .unwrap();
    let mut matcher = Matcher::new(&training, context, &budget, &cancel).unwrap();
    let mut results = Vec::new();
    for gpu in [false, true, true] {
        let mut values = vec![None; 16 * 3];
        values[0] = Some(0.123);
        values[7 * 3 + 2] = Some(2.0);
        let mut state = Grid::new(
            Shape::new([4, 4, 1]).unwrap(),
            variables.clone(),
            values,
            &budget,
        )
        .unwrap();
        let config = terrustrial_core::simulation::mps::SimulationOptions {
            template: &[[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0]],
            maximum_neighbors: 4,
            matching_neighbors: 3,
            training_boundary: Boundary::Periodic,
            seed: 19,
            realization: 4,
            budget: &budget,
            cancellation: &cancel,
        };
        if gpu {
            terrustrial_core::simulation::mps::simulate_gpu_approximate(
                &mut matcher,
                &mut state,
                0..16,
                config,
                1e-3,
                |_, _| Ok(()),
            )
            .unwrap();
        } else {
            terrustrial_core::simulation::mps::simulate(
                &training,
                &mut state,
                0..16,
                config,
                |_, _| Ok(()),
            )
            .unwrap();
        }
        assert_eq!(state.get(0, 0).unwrap(), Some(0.123));
        assert_eq!(state.get(7, 2).unwrap(), Some(2.0));
        for cell in 1..16 {
            assert_eq!(
                state.get(cell, 1).unwrap().unwrap(),
                2.0 * state.get(cell, 0).unwrap().unwrap()
            );
        }
        results.push(
            state
                .values()
                .iter()
                .map(|v| v.unwrap().to_bits())
                .collect::<Vec<_>>(),
        );
    }
    assert_eq!(results[0], results[1]);
    assert_eq!(results[1], results[2]);
}

#[test]
#[ignore = "requires a native compute adapter"]
fn integer_categorical_scores_and_seeded_ties_match_scalar_bit_for_bit() {
    let context = Arc::new(Context::new().unwrap());
    let budget = MemoryBudget::new(1 << 26);
    let cancel = Cancellation::default();
    let shape = Shape::new([11, 7, 3]).unwrap();
    let training = Grid::new(
        shape,
        vec![
            Variable::Categorical { classes: 5 },
            Variable::Categorical { classes: u32::MAX },
            Variable::Continuous { scale: 1.0 },
        ],
        (0..shape.cells())
            .flat_map(|i| {
                [
                    if i % 73 == 3 {
                        None
                    } else {
                        Some((i % 5) as f64)
                    },
                    if i % 31 == 7 {
                        None
                    } else {
                        Some(if i % 2 == 0 { 4_000_000_000.0 } else { 5.0 })
                    },
                    if i % 61 == 1 {
                        None
                    } else {
                        Some(i as f64 / 37.0)
                    },
                ]
            })
            .collect(),
        &budget,
    )
    .unwrap();
    let mut matcher = Matcher::new(&training, context, &budget, &cancel).unwrap();
    for count in [1, 7, 31, 47] {
        for weight in [0.001, 3.0, f64::MAX] {
            let mut events: Vec<_> = (0..count)
                .map(|i| Event {
                    offset: [
                        (i % 5) as i64 - 2,
                        ((i / 5) % 5) as i64 - 2,
                        (i / 25) as i64 - 1,
                    ],
                    variable: i % 2,
                    value: if i % 2 == 0 {
                        (i % 5) as f64
                    } else {
                        4_000_000_000.0
                    },
                    weight,
                })
                .collect();
            for boundary in [Boundary::Bounded, Boundary::Periodic] {
                let scores = matcher
                    .score_batch(
                        &[Query {
                            events: &events,
                            requested: &[0, 1, 2],
                            boundary,
                        }],
                        &budget,
                        &cancel,
                    )
                    .unwrap();
                let view = scores.query(0).unwrap();
                assert!(view.is_exact());
                let config = MatchOptions {
                    neighbors: shape.cells(),
                    boundary,
                    seed: 71,
                    realization: 5,
                    query_id: count as u64,
                    budget: &budget,
                    cancellation: &cancel,
                };
                let reference = mps::best_matches(&training, &events, &[0, 1, 2], &config);
                let mut expected = vec![None; shape.cells()];
                if let Ok(matches) = reference {
                    for entry in matches {
                        expected[entry.index] = Some(entry.mismatch.to_bits());
                    }
                }
                assert_eq!(
                    view.iter()
                        .map(|score| score.map(f64::to_bits))
                        .collect::<Vec<_>>(),
                    expected
                );
                for neighbors in [1, 8, shape.cells() + 1] {
                    let config = MatchOptions {
                        neighbors,
                        ..config
                    };
                    let a = mps::best_matches(&training, &events, &[0, 1, 2], &config);
                    let b = matcher.best_matches_categorical(&events, &[0, 1, 2], &config);
                    match (a, b) {
                        (Ok(a), Ok(b)) => assert_eq!(
                            a.iter()
                                .map(|m| (m.index, m.mismatch.to_bits()))
                                .collect::<Vec<_>>(),
                            b.iter()
                                .map(|m| (m.index, m.mismatch.to_bits()))
                                .collect::<Vec<_>>()
                        ),
                        (Err(_), Err(_)) => {}
                        (a, b) => panic!("reference={a:?}, gpu={b:?}"),
                    }
                }
            }
            if events.len() > 1 {
                events[0].weight = 2.0;
                let config = MatchOptions {
                    neighbors: 1,
                    boundary: Boundary::Periodic,
                    seed: 1,
                    realization: 0,
                    query_id: 0,
                    budget: &budget,
                    cancellation: &cancel,
                };
                assert!(matcher
                    .best_matches_categorical(&events, &[0], &config)
                    .is_err());
            }
        }
    }
}

#[test]
#[ignore = "requires a native compute adapter and about 250 MiB of logical storage"]
fn gpu_dispatch_crosses_the_workgroup_row_boundary_without_aliasing() {
    let context = Arc::new(Context::new().unwrap());
    let budget = MemoryBudget::new(1 << 29);
    let cancel = Cancellation::default();
    let shape = Shape::new([2049, 2049, 1]).unwrap();
    assert!(shape.cells() > context.limits().max_compute_workgroups_per_dimension as usize * 64);
    let training = Grid::new(
        shape,
        vec![Variable::Categorical { classes: 7 }],
        (0..shape.cells()).map(|i| Some((i % 7) as f64)).collect(),
        &budget,
    )
    .unwrap();
    let retained = budget.used();
    {
        let mut matcher = Matcher::new(&training, context, &budget, &cancel).unwrap();
        let events = [Event {
            offset: [-1, 0, 0],
            variable: 0,
            value: 3.0,
            weight: 1.0,
        }];
        let scores = matcher
            .score_batch(
                &[Query {
                    events: &events,
                    requested: &[0],
                    boundary: Boundary::Bounded,
                }],
                &budget,
                &cancel,
            )
            .unwrap();
        let view = scores.query(0).unwrap();
        assert!(view.is_exact());
        assert_eq!(view.len(), shape.cells());
        for (index, value) in view.iter().enumerate() {
            let expected = if index % 2049 == 0 {
                None
            } else {
                Some(f64::from((index - 1) % 7 != 3))
            };
            assert_eq!(value, expected, "cell {index}");
        }
    }
    assert_eq!(budget.used(), retained);
}
