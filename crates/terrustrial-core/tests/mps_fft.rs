#![cfg(feature = "fft")]
use rand::RngExt;
use terrustrial_core::{
    execution::{random_stream, Cancellation, MemoryBudget},
    grid::{Boundary, Event, Grid, Shape, Variable},
    mps::{
        self,
        fft::{Matcher, Refinement},
        MatchOptions,
    },
};

#[test]
fn fft_agrees_with_independent_scalar_ranking_on_mixed_prime_3d_grids() {
    let budget = MemoryBudget::new(1 << 27);
    let cancel = Cancellation::default();
    for dimensions in [[7, 5, 3], [8, 4, 1], [1, 1, 1]] {
        let shape = Shape::new(dimensions).unwrap();
        for missing in [false, true] {
            let variables = vec![
                Variable::Continuous { scale: 2.0 },
                Variable::Categorical { classes: u32::MAX },
                Variable::Continuous { scale: 0.1 },
            ];
            let values = (0..shape.cells())
                .flat_map(|i| {
                    [
                        if missing && i % 13 == 2 {
                            None
                        } else {
                            Some((i as f64 * 0.23).sin() * 5.0)
                        },
                        if missing && i % 11 == 3 {
                            None
                        } else {
                            Some(if i % 3 == 0 { 4_000_000_000.0 } else { 5.0 })
                        },
                        Some((i as f64 * 0.7).cos()),
                    ]
                })
                .collect();
            let training = Grid::new(shape, variables, values, &budget).unwrap();
            let mut matcher = Matcher::new(&training, &budget, &cancel).unwrap();
            let mut rng = random_stream(123, 0, 0, 0);
            for query in 0..30 {
                let events = [
                    Event {
                        offset: [0; 3],
                        variable: 0,
                        value: rng.random_range(-3.0..3.0),
                        weight: 1.0,
                    },
                    Event {
                        offset: [-1, 0, 0],
                        variable: 1,
                        value: if query % 2 == 0 { 4_000_000_000.0 } else { 5.0 },
                        weight: 3.0,
                    },
                    Event {
                        offset: [0, 1, 0],
                        variable: 2,
                        value: rng.random_range(-1.0..1.0),
                        weight: 0.25,
                    },
                    Event {
                        offset: [0, 0, 1],
                        variable: 0,
                        value: rng.random_range(-3.0..3.0),
                        weight: 2.0,
                    },
                ];
                for boundary in [Boundary::Bounded, Boundary::Periodic] {
                    let config = MatchOptions {
                        neighbors: 3,
                        boundary,
                        seed: 7,
                        realization: 1,
                        query_id: query,
                        budget: &budget,
                        cancellation: &cancel,
                    };
                    let reference = mps::best_matches(&training, &events, &[0, 1, 2], &config);
                    let accelerated = matcher.best_matches_approximate(
                        &events,
                        &[0, 1, 2],
                        &config,
                        Refinement::default(),
                    );
                    match (reference, accelerated) {
                        (Ok(a), Ok(b)) => {
                            assert_eq!(a.len(), b.matches.len());
                            for (a, b) in a.iter().zip(&b.matches) {
                                assert_eq!(
                                    a.index, b.index,
                                    "{dimensions:?} missing={missing} query={query}"
                                );
                                assert_eq!(a.mismatch.to_bits(), b.mismatch.to_bits());
                            }
                        }
                        (Err(_), Err(_)) => {}
                        (a, b) => panic!("reference={a:?}, fft={b:?}"),
                    }
                }
            }
        }
    }
}

#[test]
fn tied_categories_absent_categories_aliases_and_extreme_offsets_are_supported() {
    let budget = MemoryBudget::new(1 << 22);
    let cancel = Cancellation::default();
    let training = Grid::new(
        Shape::new([7, 5, 1]).unwrap(),
        vec![Variable::Categorical { classes: 4 }],
        (0..35).map(|i| Some((i % 3) as f64)).collect(),
        &budget,
    )
    .unwrap();
    let mut matcher = Matcher::new(&training, &budget, &cancel).unwrap();
    for boundary in [Boundary::Periodic, Boundary::Bounded] {
        for value in [1.0, 3.0] {
            for offset in [[1, 0, 0], [i64::MIN, i64::MAX, 0]] {
                let events = [
                    Event {
                        offset,
                        variable: 0,
                        value,
                        weight: 1.0,
                    },
                    Event {
                        offset: [0, 0, 1],
                        variable: 0,
                        value,
                        weight: 0.5,
                    },
                ];
                let config = MatchOptions {
                    neighbors: 4,
                    boundary,
                    seed: 3,
                    realization: 5,
                    query_id: 0,
                    budget: &budget,
                    cancellation: &cancel,
                };
                let reference = mps::best_matches(&training, &events, &[0], &config);
                let accelerated =
                    matcher.best_matches_approximate(&events, &[0], &config, Refinement::default());
                match (reference, accelerated) {
                    (Ok(a), Ok(b)) => assert_eq!(
                        a.iter().map(|m| m.index).collect::<Vec<_>>(),
                        b.matches.iter().map(|m| m.index).collect::<Vec<_>>()
                    ),
                    (Err(_), Err(_)) => {}
                    (a, b) => panic!("reference={a:?}, fft={b:?}"),
                }
            }
        }
    }
}

#[test]
fn large_baselines_do_not_lose_pattern_contrast_and_errors_leave_cache_reusable() {
    let budget = MemoryBudget::new(1 << 22);
    let cancel = Cancellation::default();
    let training = Grid::new(
        Shape::new([32, 1, 1]).unwrap(),
        vec![Variable::Continuous { scale: 0.125 }],
        (0..32).map(|i| Some(1e14 + i as f64 * 0.125)).collect(),
        &budget,
    )
    .unwrap();
    let retained = budget.used();
    {
        let mut matcher = Matcher::new(&training, &budget, &cancel).unwrap();
        let config = MatchOptions {
            neighbors: 3,
            boundary: Boundary::Periodic,
            seed: 3,
            realization: 5,
            query_id: 0,
            budget: &budget,
            cancellation: &cancel,
        };
        let events = [Event {
            offset: [-1, 0, 0],
            variable: 0,
            value: 1e14 + 7.0 * 0.125,
            weight: 1.0,
        }];
        assert!(matcher
            .best_matches_approximate(&events, &[1], &config, Refinement::default())
            .is_err());
        assert!(matcher
            .best_matches_approximate(
                &events,
                &[0],
                &config,
                Refinement {
                    absolute: f64::NAN,
                    relative: 0.0
                }
            )
            .is_err());
        for _ in 0..2 {
            let exact = mps::best_matches(&training, &events, &[0], &config).unwrap();
            let result = matcher
                .best_matches_approximate(&events, &[0], &config, Refinement::default())
                .unwrap();
            assert_eq!(
                result.matches.iter().map(|m| m.index).collect::<Vec<_>>(),
                exact.iter().map(|m| m.index).collect::<Vec<_>>()
            );
            assert_eq!(result.matches[0].index, 8);
            assert_eq!(result.matches[0].mismatch, 0.0);
        }
    }
    assert_eq!(budget.used(), retained);
    assert!(Matcher::new(&training, &MemoryBudget::new(1), &cancel).is_err());
    cancel.cancel();
    assert!(Matcher::new(&training, &budget, &cancel).is_err());
}

#[test]
fn cached_simulation_matches_scalar_and_preserves_joint_pairs_and_partial_hard_data() {
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
    let mut matcher = Matcher::new(&training, &budget, &cancel).unwrap();
    let mut results = Vec::new();
    for accelerated in [false, true, true] {
        let mut values = vec![None; 64 * 3];
        values[0] = Some(0.123);
        values[7 * 3 + 2] = Some(2.0);
        let mut state = Grid::new(
            Shape::new([8, 8, 1]).unwrap(),
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
        if accelerated {
            terrustrial_core::simulation::mps::simulate_fft_approximate(
                &mut matcher,
                &mut state,
                0..64,
                config,
                Refinement::default(),
                |_, _| Ok(()),
            )
            .unwrap();
        } else {
            terrustrial_core::simulation::mps::simulate(
                &training,
                &mut state,
                0..64,
                config,
                |_, _| Ok(()),
            )
            .unwrap();
        }
        assert_eq!(state.get(0, 0).unwrap(), Some(0.123));
        assert_eq!(state.get(7, 2).unwrap(), Some(2.0));
        for cell in 1..64 {
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
