use terrustrial_core::{
    execution::{Cancellation, MemoryBudget},
    grid::{Boundary, Event, Grid, Shape, Variable},
    mps::{self, MatchOptions},
};
fn options<'a>(budget: &'a MemoryBudget, cancel: &'a Cancellation) -> MatchOptions<'a> {
    MatchOptions {
        neighbors: 1,
        boundary: Boundary::Bounded,
        seed: 7,
        realization: 0,
        query_id: 0,
        budget,
        cancellation: cancel,
    }
}
#[test]
fn mixed_pattern_scores_match_hand_calculation_and_boundaries_do_not_wrap() {
    let budget = MemoryBudget::new(1 << 20);
    let cancel = Cancellation::default();
    let training = Grid::new(
        Shape::new([4, 1, 1]).unwrap(),
        vec![
            Variable::Continuous { scale: 1.0 },
            Variable::Categorical { classes: 2 },
        ],
        vec![
            Some(0.0),
            Some(0.0),
            Some(1.0),
            Some(1.0),
            Some(2.0),
            Some(0.0),
            Some(3.0),
            Some(1.0),
        ],
        &budget,
    )
    .unwrap();
    let events = [Event {
        offset: [-1, 0, 0],
        variable: 0,
        value: 1.0,
        weight: 1.0,
    }];
    let draw = mps::draw(&training, &events, &[0, 1], &options(&budget, &cancel)).unwrap();
    assert_eq!(draw.source_index, 2);
    assert_eq!(draw.mismatch, 0.0);
    assert_eq!(draw.values, vec![(0, 2.0), (1, 0.0)]);
    let mut events = events.to_vec();
    events.push(Event {
        offset: [0; 3],
        variable: 1,
        value: 1.0,
        weight: 4.0,
    });
    let mut configuration = options(&budget, &cancel);
    configuration.neighbors = 2;
    let matches = mps::best_matches(&training, &events, &[0, 1], &configuration).unwrap();
    assert!(matches
        .iter()
        .all(|m| [1, 3].contains(&m.index) && (m.mismatch - 0.2).abs() < 1e-12));
    let off_grid = [Event {
        offset: [4, 0, 0],
        variable: 0,
        value: 2.0,
        weight: 1.0,
    }];
    assert!(mps::draw(&training, &off_grid, &[0], &options(&budget, &cancel)).is_err());
    configuration.boundary = Boundary::Periodic;
    configuration.neighbors = 1;
    assert_eq!(
        mps::draw(&training, &off_grid, &[0], &configuration)
            .unwrap()
            .source_index,
        2
    );
}
#[test]
fn joint_simulation_preserves_partial_hard_data_and_cross_attribute_pairs() {
    let run = || {
        let budget = MemoryBudget::new(1 << 20);
        let cancel = Cancellation::default();
        let variables = vec![
            Variable::Categorical { classes: 2 },
            Variable::Continuous { scale: 1.0 },
            Variable::Continuous { scale: 2.0 },
        ];
        let training = Grid::new(
            Shape::new([4, 2, 1]).unwrap(),
            variables.clone(),
            (0..8)
                .flat_map(|i| {
                    [
                        Some((i % 2) as f64),
                        Some(i as f64 / 5.0),
                        Some(2.0 * i as f64 / 5.0),
                    ]
                })
                .collect(),
            &budget,
        )
        .unwrap();
        let mut values = vec![None; 16 * 3];
        values[0] = Some(1.0);
        values[1] = Some(0.123);
        let mut state =
            Grid::new(Shape::new([4, 4, 1]).unwrap(), variables, values, &budget).unwrap();
        terrustrial_core::simulation::mps::simulate(
            &training,
            &mut state,
            0..16,
            terrustrial_core::simulation::mps::SimulationOptions {
                template: &[[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0]],
                maximum_neighbors: 4,
                matching_neighbors: 2,
                training_boundary: Boundary::Periodic,
                seed: 9,
                realization: 3,
                budget: &budget,
                cancellation: &cancel,
            },
            |_, _| Ok(()),
        )
        .unwrap();
        assert_eq!(state.get(0, 0).unwrap(), Some(1.0));
        assert_eq!(state.get(0, 1).unwrap(), Some(0.123));
        assert!(state.values().iter().all(Option::is_some));
        for cell in 1..16 {
            assert_eq!(
                state.get(cell, 2).unwrap().unwrap(),
                2.0 * state.get(cell, 1).unwrap().unwrap()
            );
        }
        state
            .values()
            .iter()
            .map(|v| v.unwrap().to_bits())
            .collect::<Vec<_>>()
    };
    assert_eq!(run(), run());
}
#[test]
fn missing_training_values_cannot_be_selected_and_unconditional_ties_are_randomized() {
    let budget = MemoryBudget::new(1 << 20);
    let cancel = Cancellation::default();
    let training = Grid::new(
        Shape::new([5, 1, 1]).unwrap(),
        vec![Variable::Categorical { classes: 4 }],
        vec![Some(0.0), Some(1.0), Some(2.0), Some(3.0), None],
        &budget,
    )
    .unwrap();
    let mut counts = [0; 4];
    for realization in 0..2048 {
        let mut config = options(&budget, &cancel);
        config.realization = realization;
        let draw = mps::draw(&training, &[], &[0], &config).unwrap();
        counts[draw.source_index] += 1;
    }
    assert!(counts.iter().all(|count| (*count as i64 - 512).abs() < 150));
    let unknown = [Event {
        offset: [0; 3],
        variable: 0,
        value: 4.0,
        weight: 1.0,
    }];
    assert!(mps::draw(&training, &unknown, &[0], &options(&budget, &cancel)).is_err());
    assert!(Shape::new([usize::MAX, 2, 2]).is_err());
}
