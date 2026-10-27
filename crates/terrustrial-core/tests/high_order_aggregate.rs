use terrustrial_core::{
    execution::{Cancellation, MemoryBudget},
    high_order::{
        aggregate::{Options, PrefixSource, Unsupported},
        Axis, FitOptions, Marginal, Plan, Prototype, ReplicateVisitor,
    },
    Error, Result,
};
fn options<'a>(budget: &'a MemoryBudget, cancellation: &'a Cancellation) -> Options<'a> {
    Options {
        minimum_replicates: 1,
        unsupported: Unsupported::Reject,
        fit: FitOptions {
            maximum_iterations: 100_000,
            tolerance: 1e-12,
            minimum_normalizer_fraction: 1e-10,
            budget,
            cancellation,
        },
    }
}
fn plan(budget: &MemoryBudget, cancel: &Cancellation) -> Result<Plan> {
    Plan::new(
        &[Axis::Categorical { classes: 2 }],
        vec![
            Prototype(vec![Marginal::Categorical(vec![1.0, 0.0])]),
            Prototype(vec![Marginal::Categorical(vec![0.0, 1.0])]),
        ],
        0.0,
        budget,
        cancel,
    )
}
// Independent explicit Legendre coefficient expansion. A coefficient uses all
// rows covering its highest nonzero polynomial index. No production kernel,
// log weighting, prefix telescoping or optimizer is used in this reference.
fn reference(rows: &[(Vec<f64>, Vec<f64>)], event: [f64; 2]) -> [f64; 2] {
    let p = |w, x: f64| match w {
        0 => 1.0,
        1 => x,
        2 => (3.0 * x * x - 1.0) / 2.0,
        _ => unreachable!(),
    };
    let mut values = [0.0; 2];
    for w0 in 0..=2 {
        for w1 in 0..=2 {
            let required = if w1 > 0 {
                2
            } else if w0 > 0 {
                1
            } else {
                0
            };
            let mut counts = 0;
            let mut moment = [0.0; 2];
            for (target, row) in rows {
                if row.len() < required {
                    continue;
                }
                let factor = (if w0 == 0 { 1.0 } else { p(w0, row[0]) })
                    * (if w1 == 0 { 1.0 } else { p(w1, row[1]) });
                moment[target[0] as usize] += factor;
                counts += 1;
            }
            assert!(counts > 0);
            let factor =
                (2 * w0 + 1) as f64 * (2 * w1 + 1) as f64 * p(w0, event[0]) * p(w1, event[1]);
            for c in 0..2 {
                values[c] += factor * moment[c] / counts as f64;
            }
        }
    }
    values
}
#[test]
fn prefix_aggregation_matches_independent_moment_coefficients_and_preserves_the_base() -> Result<()>
{
    let budget = MemoryBudget::new(1 << 24);
    let cancel = Cancellation::default();
    let plan = plan(&budget, &cancel)?;
    let a = (1.0f64 / 3.0).sqrt();
    let mut rows = vec![];
    for i in 0..4usize {
        rows.push((
            vec![(i.count_ones() % 2) as f64],
            vec![
                if i & 1 == 0 { -a } else { a },
                if i & 2 == 0 { -a } else { a },
            ],
        ));
    }
    for x in [-a, a, -a, a] {
        rows.push((vec![0.0], vec![x]));
    }
    for _ in 0..3 {
        rows.push((vec![1.0], vec![]));
    }
    for x in [-0.5, -0.2, 0.0, 0.2, 0.5] {
        for y in [-0.4, 0.0, 0.4] {
            let expected = reference(&rows, [x, y]);
            let total = expected.iter().sum::<f64>();
            assert!(total > 0.0);
            let fit = plan.condition_prefixes(
                rows.as_slice(),
                &[
                    (Axis::Continuous { degree: 2 }, x),
                    (Axis::Continuous { degree: 2 }, y),
                ],
                options(&budget, &cancel),
            )?;
            assert_eq!(fit.evidence.support, [11, 8, 4]);
            assert_eq!(fit.evidence.used_prefix, 2);
            let probability = (expected[0] / total).clamp(0.0, 1.0);
            assert!(
                (fit.distribution.weights()[0] - probability).abs() < 1e-11,
                "{x}, {y}: {:?} vs {probability}",
                fit.distribution.weights()
            );
            assert!((fit.distribution.diagnostics.log_normalizer.exp() - total).abs() < 1e-11);
        }
    }
    Ok(())
}
#[test]
fn complete_rows_and_empty_events_reduce_to_the_original_estimator() -> Result<()> {
    let budget = MemoryBudget::new(1 << 24);
    let cancel = Cancellation::default();
    let plan = plan(&budget, &cancel)?;
    let rows = vec![
        (vec![0.0], vec![0.1, 1.0]),
        (vec![1.0], vec![0.3, 1.0]),
        (vec![0.0], vec![-0.4, 0.0]),
    ];
    let event = [
        (Axis::Continuous { degree: 4 }, 0.2),
        (Axis::Categorical { classes: 2 }, 1.0),
    ];
    let ordinary = plan.condition(rows.as_slice(), &event, options(&budget, &cancel).fit)?;
    let aggregated = plan.condition_prefixes(rows.as_slice(), &event, options(&budget, &cancel))?;
    for (a, b) in ordinary
        .weights()
        .iter()
        .zip(aggregated.distribution.weights())
    {
        assert!((a - b).abs() < 1e-12);
    }
    assert_eq!(aggregated.evidence.support, [3, 3, 3]);
    assert_eq!(aggregated.evidence.minimum_row_fraction, 1.0);
    let rows: Vec<_> = rows
        .into_iter()
        .map(|(target, _)| (target, vec![]))
        .collect();
    let fit = plan.condition_prefixes(rows.as_slice(), &[], options(&budget, &cancel))?;
    assert_eq!(fit.evidence.support, [3]);
    assert_eq!(fit.evidence.used_prefix, 0);
    assert!((fit.distribution.weights()[0] - 2.0 / 3.0).abs() < 1e-12);
    assert!(fit.distribution.diagnostics.log_normalizer.abs() < 1e-12);
    Ok(())
}
#[test]
fn categorical_uniform_measure_scaling_and_impossible_normalizers_are_explicit() -> Result<()> {
    let budget = MemoryBudget::new(1 << 24);
    let cancel = Cancellation::default();
    let plan = plan(&budget, &cancel)?;
    let rows = vec![
        (vec![0.0], vec![0.0, 0.0]),
        (vec![1.0], vec![1.0, 1.0]),
        (vec![0.0], vec![0.0]),
        (vec![1.0], vec![]),
    ];
    let cat = Axis::Categorical { classes: 2 };
    let fit = plan.condition_prefixes(
        rows.as_slice(),
        &[(cat, 0.0), (cat, 0.0)],
        options(&budget, &cancel),
    )?;
    assert_eq!(fit.evidence.support, [4, 3, 2]);
    assert!((fit.distribution.weights()[0] - 13.0 / 14.0).abs() < 1e-12);
    assert!((fit.distribution.diagnostics.log_normalizer.exp() - 7.0 / 3.0).abs() < 1e-12);
    assert!(fit.distribution.diagnostics.negative_weight_fraction > 0.0);
    assert!(matches!(
        plan.condition_prefixes(
            rows.as_slice(),
            &[(cat, 1.0), (cat, 0.0)],
            options(&budget, &cancel)
        ),
        Err(Error::Numerical { .. })
    ));
    Ok(())
}
#[test]
fn sparse_grid_prefixes_retain_available_evidence_and_report_unsupported_interactions() -> Result<()>
{
    use terrustrial_core::{
        grid::{Boundary, Grid, Shape, Variable},
        high_order::grid::{GridReplicates, Location},
    };
    let budget = MemoryBudget::new(1 << 24);
    let cancel = Cancellation::default();
    let plan = plan(&budget, &cancel)?;
    let grid = Grid::new(
        Shape::new([5, 1, 1])?,
        vec![Variable::Categorical { classes: 2 }],
        vec![Some(0.0), Some(1.0), None, Some(0.0), Some(1.0)],
        &budget,
    )?;
    let locations = [
        Location {
            offset: [-1, 0, 0],
            variable: 0,
        },
        Location {
            offset: [-2, 0, 0],
            variable: 0,
        },
    ];
    let source = GridReplicates::new(&grid, &[0], &locations, Boundary::Bounded, &budget, &cancel)?;
    let mut rows = vec![];
    source.scan_prefixes(&mut |target, condition| {
        rows.push((target.to_vec(), condition.to_vec()));
        Ok(())
    })?;
    assert_eq!(
        rows,
        vec![
            (vec![0.0], vec![]),
            (vec![1.0], vec![0.0]),
            (vec![0.0], vec![]),
            (vec![1.0], vec![0.0])
        ]
    );
    let cat = Axis::Categorical { classes: 2 };
    let event = [(cat, 0.0), (cat, 1.0)];
    assert!(plan
        .condition_prefixes(&source, &event, options(&budget, &cancel))
        .is_err());
    let mut settings = options(&budget, &cancel);
    settings.unsupported = Unsupported::ZeroHigherInteractions;
    let fit = plan.condition_prefixes(&source, &event, settings)?;
    assert_eq!(fit.evidence.support, [4, 2, 0]);
    assert_eq!(fit.evidence.used_prefix, 1);
    assert!((fit.distribution.weights()[1] - 0.75).abs() < 1e-12);
    // Raising the explicit count floor to three admits only the target marginal.
    settings.minimum_replicates = 3;
    let fit = plan.condition_prefixes(&source, &event, settings)?;
    assert_eq!(fit.evidence.used_prefix, 0);
    assert_eq!(fit.distribution.weights(), [0.5, 0.5]);
    Ok(())
}
#[test]
fn replay_counts_resources_and_failure_paths_are_checked() -> Result<()> {
    use std::cell::Cell;
    let budget = MemoryBudget::new(1 << 24);
    let cancel = Cancellation::default();
    let plan = plan(&budget, &cancel)?;
    struct Changing(Cell<u32>);
    impl PrefixSource for Changing {
        fn scan_prefixes(&self, v: &mut ReplicateVisitor<'_>) -> Result<()> {
            let pass = self.0.get();
            self.0.set(pass + 1);
            v(&[0.0], if pass == 0 { &[0.0] } else { &[] })
        }
    }
    let before = budget.used();
    assert!(plan
        .condition_prefixes(
            &Changing(Cell::new(0)),
            &[(Axis::Categorical { classes: 2 }, 0.0)],
            options(&budget, &cancel)
        )
        .is_err());
    assert_eq!(budget.used(), before);
    struct Stream;
    impl PrefixSource for Stream {
        fn scan_prefixes(&self, v: &mut ReplicateVisitor<'_>) -> Result<()> {
            for i in 0..100_000 {
                let n = i % 3;
                v(&[(i % 2) as f64], &[0.0, 0.0][..n])?;
            }
            Ok(())
        }
    }
    let event = [(Axis::Continuous { degree: 0 }, 0.0); 2];
    {
        let fit = plan.condition_prefixes(&Stream, &event, options(&budget, &cancel))?;
        assert_eq!(fit.evidence.support, [100_000, 66_666, 33_333]);
        assert!(budget.used() - before <= 40);
        assert!((fit.distribution.weights()[0] - 0.5).abs() < 1e-12);
    }
    assert_eq!(budget.used(), before);
    let tiny = MemoryBudget::new(1);
    assert!(matches!(
        plan.condition_prefixes(&Stream, &event, options(&tiny, &cancel)),
        Err(Error::MemoryBudget { .. })
    ));
    assert_eq!(tiny.used(), 0);
    let invalid = vec![(vec![0.0], vec![0.0, 0.0, 0.0])];
    assert!(plan
        .condition_prefixes(invalid.as_slice(), &event, options(&budget, &cancel))
        .is_err());
    cancel.cancel();
    assert!(matches!(
        plan.condition_prefixes(&Stream, &event, options(&budget, &cancel)),
        Err(Error::Cancelled)
    ));
    Ok(())
}

#[test]
fn partial_evidence_fits_a_normalized_mixed_joint_density() -> Result<()> {
    let budget = MemoryBudget::new(1 << 24);
    let cancel = Cancellation::default();
    let plan = Plan::new(
        &[
            Axis::Continuous { degree: 6 },
            Axis::Categorical { classes: 2 },
        ],
        vec![
            Prototype(vec![
                Marginal::TruncatedNormal {
                    mean: -0.7,
                    deviation: 0.07,
                },
                Marginal::Categorical(vec![1.0, 0.0]),
            ]),
            Prototype(vec![
                Marginal::TruncatedNormal {
                    mean: 0.7,
                    deviation: 0.07,
                },
                Marginal::Categorical(vec![0.0, 1.0]),
            ]),
        ],
        0.0,
        &budget,
        &cancel,
    )?;
    let rows = vec![
        (vec![-0.7, 0.0], vec![0.2]),
        (vec![0.7, 1.0], vec![-0.2]),
        (vec![-0.7, 0.0], vec![]),
    ];
    let fit = plan.condition_prefixes(
        rows.as_slice(),
        &[(Axis::Continuous { degree: 2 }, 0.1)],
        options(&budget, &cancel),
    )?;
    assert_eq!(fit.evidence.support, [3, 2]);
    let mut integral = 0.0;
    for i in 0..10_000 {
        let x = -1.0 + (i as f64 + 0.5) / 5000.0;
        for c in 0..2 {
            integral += fit.distribution.log_probability(&[x, c as f64])?.exp() / 5000.0;
        }
    }
    assert!((integral - 1.0).abs() < 1e-9);
    for item in 0..10_000 {
        let mut a = [0.0; 2];
        let mut b = [0.0; 2];
        fit.distribution.draw_into(9, 0, item, &mut a, &cancel)?;
        fit.distribution.draw_into(9, 0, item, &mut b, &cancel)?;
        assert_eq!(a, b);
        assert_eq!(a[0] > 0.0, a[1] == 1.0);
    }
    Ok(())
}
#[test]
fn sparse_sample_simulation_needs_no_exhaustive_training_image_and_preserves_hard_data(
) -> Result<()> {
    use terrustrial_core::{
        grid::{Boundary, Grid, Shape, Variable},
        simulation::high_order::{simulate_prefixes, AggregationPolicy, SimulationOptions},
    };
    let budget = MemoryBudget::new(1 << 24);
    let cancel = Cancellation::default();
    let plan = plan(&budget, &cancel)?;
    let hard = vec![
        Some(-0.0),
        Some(1.0),
        Some(0.0),
        None,
        Some(1.0),
        Some(0.0),
        Some(1.0),
        None,
    ];
    let samples = Grid::new(
        Shape::new([8, 1, 1])?,
        vec![Variable::Categorical { classes: 2 }],
        hard.clone(),
        &budget,
    )?;
    let mut initial = hard.clone();
    initial.resize(32, None);
    let mut state = Grid::new(
        Shape::new([32, 1, 1])?,
        samples.variables().to_vec(),
        initial.clone(),
        &budget,
    )?;
    let mut repeat = Grid::new(
        state.shape(),
        samples.variables().to_vec(),
        initial,
        &budget,
    )?;
    for grid in [&mut state, &mut repeat] {
        let mut outputs = 0;
        simulate_prefixes(
            &plan,
            &samples,
            grid,
            0..32,
            SimulationOptions {
                template: &[[-1, 0, 0], [-2, 0, 0]],
                maximum_neighbors: 2,
                training_boundary: Boundary::Bounded,
                seed: 27,
                realization: 3,
                fit: options(&budget, &cancel).fit,
            },
            AggregationPolicy {
                minimum_replicates: 1,
                unsupported: Unsupported::Reject,
            },
            |_, draw| {
                let evidence = draw.aggregation.as_ref().unwrap();
                assert_eq!(evidence.support, [6, 4, 2]);
                assert_eq!(evidence.used_prefix, 2);
                outputs += 1;
                Ok(())
            },
        )?;
        assert_eq!(outputs, 26);
        for (i, value) in hard.iter().enumerate() {
            if let Some(value) = value {
                assert_eq!(grid.values()[i].unwrap().to_bits(), value.to_bits());
            }
        }
        assert_eq!(grid.values()[3], Some(1.0));
        assert_eq!(grid.values()[7], Some(0.0));
        for i in 8..32 {
            assert_eq!(grid.values()[i], Some(if i % 2 == 0 { 1.0 } else { 0.0 }));
        }
    }
    assert_eq!(state.values(), repeat.values());
    Ok(())
}

#[test]
fn row_cancellation_and_overflow_scale_prefix_products_are_reported() -> Result<()> {
    let budget = MemoryBudget::new(1 << 24);
    let cancel = Cancellation::default();
    let plan = plan(&budget, &cancel)?;
    // C0=2, C1=1, R1=1+3*(-1/6)=1/2, so the full row's
    // aggregate weight is 1/2 - 1 + 1/2 = 0 exactly.
    let rows = vec![(vec![0.0], vec![-1.0 / 6.0]), (vec![1.0], vec![])];
    let fit = plan.condition_prefixes(
        rows.as_slice(),
        &[(Axis::Continuous { degree: 1 }, 1.0)],
        options(&budget, &cancel),
    )?;
    assert_eq!(fit.evidence.minimum_row_fraction, 0.0);
    assert_eq!(fit.distribution.weights(), [0.0, 1.0]);
    assert_eq!(fit.distribution.diagnostics.nonzero_kernel_rows, 1);
    let event = vec![(Axis::Continuous { degree: 1 }, 1.0); 2048];
    let rows = vec![(vec![0.0], vec![1.0; 2048]), (vec![1.0], vec![1.0; 2048])];
    let fit = plan.condition_prefixes(rows.as_slice(), &event, options(&budget, &cancel))?;
    assert_eq!(fit.distribution.weights(), [0.5, 0.5]);
    assert_eq!(fit.evidence.minimum_row_fraction, 1.0);
    assert!((fit.distribution.diagnostics.log_normalizer - 2048.0 * 4.0f64.ln()).abs() < 1e-9);
    Ok(())
}
