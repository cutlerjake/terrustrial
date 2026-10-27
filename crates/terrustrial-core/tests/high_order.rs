use terrustrial_core::{
    execution::{Cancellation, MemoryBudget},
    high_order::{Axis, FitOptions, Marginal, Plan, Prototype, ReplicateSource, ReplicateVisitor},
    Error, Result,
};

fn options<'a>(budget: &'a MemoryBudget, cancellation: &'a Cancellation) -> FitOptions<'a> {
    FitOptions {
        maximum_iterations: 100_000,
        tolerance: 1e-12,
        minimum_normalizer_fraction: 1e-10,
        budget,
        cancellation,
    }
}
fn category(value: usize) -> Marginal {
    let mut p = vec![0.0; 2];
    p[value] = 1.0;
    Marginal::Categorical(p)
}

#[test]
fn smoothed_categorical_kernel_matches_an_independent_ridge_solution() -> Result<()> {
    let budget = MemoryBudget::new(1 << 24);
    let cancel = Cancellation::default();
    let plan = Plan::new(
        &[Axis::SmoothedCategorical {
            classes: 2,
            uniform_mass: 0.4,
        }],
        vec![Prototype(vec![category(0)]), Prototype(vec![category(1)])],
        0.3,
        &budget,
        &cancel,
    )?;
    let rows = vec![
        (vec![0.0], vec![0.0]),
        (vec![1.0], vec![1.0]),
        (vec![1.0], vec![1.0]),
    ];
    let fit = plan.condition(
        rows.as_slice(),
        &[(
            Axis::SmoothedCategorical {
                classes: 2,
                uniform_mass: 0.2,
            },
            0.0,
        )],
        options(&budget, &cancel),
    )?;
    // Conditioning masses 0.9,0.1,0.1 give p0=9/11. Target Gram
    // = 0.6 I + 0.2 J, plus ridge 0.3 I. On the simplex the constant
    // component cancels, leaving alpha0=(0.6*p0 + 0.3/2)/(0.6+0.3).
    let expected = (0.6 * 9.0 / 11.0 + 0.15) / 0.9;
    assert!((fit.weights()[0] - expected).abs() < 1e-12);
    assert!((fit.log_probability(&[0.0])?.exp() - expected).abs() < 1e-12);
    assert_eq!(fit.diagnostics.nonzero_kernel_rows, 3);
    assert_eq!(fit.diagnostics.negative_weight_fraction, 0.0);
    assert!((fit.diagnostics.log_normalizer - 1.1_f64.ln()).abs() < 1e-12);
    Ok(())
}

#[test]
fn unseen_categorical_events_require_explicit_smoothing_and_survive_log_underflow() -> Result<()> {
    let budget = MemoryBudget::new(1 << 24);
    let cancel = Cancellation::default();
    let plan = Plan::new(
        &[Axis::Categorical { classes: 2 }],
        vec![Prototype(vec![category(0)]), Prototype(vec![category(1)])],
        0.0,
        &budget,
        &cancel,
    )?;
    let rows = vec![
        (vec![0.0], vec![0.0, 0.0]),
        (vec![0.0], vec![0.0, 0.0]),
        (vec![1.0], vec![1.0, 1.0]),
    ];
    let strict = Axis::Categorical { classes: 2 };
    let smooth = Axis::SmoothedCategorical {
        classes: 2,
        uniform_mass: 0.2,
    };
    assert!(plan
        .condition(
            rows.as_slice(),
            &[(strict, 0.0), (strict, 1.0)],
            options(&budget, &cancel)
        )
        .is_err());
    let fit = plan.condition(
        rows.as_slice(),
        &[(smooth, 0.0), (smooth, 1.0)],
        options(&budget, &cancel),
    )?;
    assert!((fit.weights()[0] - 2.0 / 3.0).abs() < 1e-12);
    assert_eq!(fit.diagnostics.normalizer_fraction, 1.0);
    let large = vec![(vec![0.0], vec![1.0; 2048]), (vec![1.0], vec![1.0; 2048])];
    let fit = plan.condition(
        large.as_slice(),
        &vec![(smooth, 0.0); 2048],
        options(&budget, &cancel),
    )?;
    assert_eq!(fit.weights(), [0.5, 0.5]);
    assert!((fit.diagnostics.log_normalizer - (2048.0 * 0.1_f64.ln() + 2.0_f64.ln())).abs() < 1e-9);
    for (classes, uniform_mass) in [
        (0, 0.1),
        (2, -0.1),
        (2, 1.0),
        (2, f64::NAN),
        (2, f64::INFINITY),
        (2, f64::from_bits(1)),
    ] {
        let axis = Axis::SmoothedCategorical {
            classes,
            uniform_mass,
        };
        assert!(plan
            .condition(
                rows.as_slice(),
                &[(axis, 0.0); 2],
                options(&budget, &cancel)
            )
            .is_err());
        assert!(Plan::new(
            &[axis],
            vec![Prototype(vec![category(0)])],
            0.0,
            &budget,
            &cancel
        )
        .is_err());
    }
    let zero = Axis::SmoothedCategorical {
        classes: 2,
        uniform_mass: 0.0,
    };
    let a = plan.condition(
        rows.as_slice(),
        &[(strict, 0.0); 2],
        options(&budget, &cancel),
    )?;
    let b = plan.condition(
        rows.as_slice(),
        &[(zero, 0.0); 2],
        options(&budget, &cancel),
    )?;
    assert_eq!(a.weights(), b.weights());
    Ok(())
}

#[test]
fn smoothed_prefix_kernels_preserve_uniform_measure_scaling() -> Result<()> {
    use terrustrial_core::high_order::aggregate;
    let budget = MemoryBudget::new(1 << 24);
    let cancel = Cancellation::default();
    let plan = Plan::new(
        &[Axis::Categorical { classes: 2 }],
        vec![Prototype(vec![category(0)]), Prototype(vec![category(1)])],
        0.0,
        &budget,
        &cancel,
    )?;
    let rows = vec![
        (vec![0.0], vec![]),
        (vec![0.0], vec![0.0]),
        (vec![1.0], vec![1.0]),
    ];
    let event = [(
        Axis::SmoothedCategorical {
            classes: 2,
            uniform_mass: 0.2,
        },
        1.0,
    )];
    let fit = plan.condition_prefixes(
        rows.as_slice(),
        &event,
        aggregate::Options {
            minimum_replicates: 1,
            unsupported: aggregate::Unsupported::Reject,
            fit: options(&budget, &cancel),
        },
    )?;
    // Normalized kernels R are 0.2 and 1.8, giving row weights
    // 1/3, 1/3-1/2+0.2/2=-1/15, 1/3-1/2+1.8/2=11/15.
    assert!((fit.distribution.weights()[0] - 4.0 / 15.0).abs() < 1e-12);
    assert!((fit.distribution.diagnostics.negative_weight_fraction - 1.0 / 17.0).abs() < 1e-12);
    assert_eq!(fit.evidence.support, [3, 2]);
    let complete = &rows[1..];
    let a = plan.condition(complete, &event, options(&budget, &cancel))?;
    let b = plan.condition_prefixes(
        complete,
        &event,
        aggregate::Options {
            minimum_replicates: 1,
            unsupported: aggregate::Unsupported::Reject,
            fit: options(&budget, &cancel),
        },
    )?;
    for (a, b) in a.weights().iter().zip(b.distribution.weights()) {
        assert!((a - b).abs() < 1e-12);
    }
    Ok(())
}
fn normal(mean: f64, deviation: f64) -> Marginal {
    Marginal::TruncatedNormal { mean, deviation }
}

#[test]
fn known_continuous_targets_use_bayes_weights_and_exact_hard_bits() -> Result<()> {
    let budget = MemoryBudget::new(1 << 24);
    let cancel = Cancellation::default();
    let plan = Plan::new(
        &[
            Axis::Continuous { degree: 6 },
            Axis::Categorical { classes: 2 },
        ],
        vec![
            Prototype(vec![normal(-0.5, 0.2), category(0)]),
            Prototype(vec![normal(0.5, 0.2), category(1)]),
        ],
        0.0,
        &budget,
        &cancel,
    )?;
    let rows = vec![(vec![-0.5, 0.0], vec![]), (vec![0.5, 1.0], vec![])];
    let fit = plan.condition(rows.as_slice(), &[], options(&budget, &cancel))?;
    let used = budget.used();
    {
        let posterior = fit.condition_targets(&[Some(0.1), None], &budget, &cancel)?;
        // Equal priors and symmetric truncation constants cancel. Log density
        // ratio is ((0.1+0.5)^2-(0.1-0.5)^2)/(2*0.2^2) = 2.5.
        let expected = 1.0 / (1.0 + (-2.5_f64).exp());
        assert!((posterior.weights()[1] - expected).abs() < 1e-12);
        let mut count = 0;
        for item in 0..10_000 {
            let mut a = [0.0; 2];
            let mut b = [0.0; 2];
            posterior.draw_into(9, 0, item, &mut a, &cancel)?;
            posterior.draw_into(9, 0, item, &mut b, &cancel)?;
            assert_eq!(a, b);
            assert_eq!(a[0].to_bits(), 0.1_f64.to_bits());
            count += a[1] as usize;
        }
        assert!((count as f64 / 10_000.0 - expected).abs() < 0.015);
        let zero = fit.condition_targets(&[Some(-0.0), None], &budget, &cancel)?;
        let mut draw = [0.0; 2];
        zero.draw_into(4, 0, 0, &mut draw, &cancel)?;
        assert_eq!(draw[0].to_bits(), (-0.0_f64).to_bits());
        let unconditioned = fit.condition_targets(&[None, None], &budget, &cancel)?;
        assert_eq!(fit.weights(), unconditioned.weights());
        let mut a = [0.0; 2];
        let mut b = [0.0; 2];
        fit.draw_into(42, 0, 0, &mut a, &cancel)?;
        unconditioned.draw_into(42, 0, 0, &mut b, &cancel)?;
        assert_eq!(a, b);
    }
    assert_eq!(budget.used(), used);
    for known in [
        vec![None],
        vec![Some(f64::NAN), None],
        vec![None, Some(2.0)],
    ] {
        assert!(fit.condition_targets(&known, &budget, &cancel).is_err());
    }
    let tiny = MemoryBudget::new(1);
    assert!(matches!(
        fit.condition_targets(&[None, Some(1.0)], &tiny, &cancel),
        Err(Error::MemoryBudget { .. })
    ));
    assert_eq!(tiny.used(), 0);
    cancel.cancel();
    assert!(matches!(
        fit.condition_targets(&[None, Some(1.0)], &budget, &cancel),
        Err(Error::Cancelled)
    ));
    Ok(())
}

#[test]
fn partial_hard_categories_condition_the_joint_component_instead_of_overwriting_a_draw(
) -> Result<()> {
    use terrustrial_core::grid::{Boundary, Grid, Shape, Variable};
    let budget = MemoryBudget::new(1 << 24);
    let cancel = Cancellation::default();
    let axis = Axis::SmoothedCategorical {
        classes: 2,
        uniform_mass: 0.2,
    };
    let plan = Plan::new(
        &[axis; 2],
        vec![
            Prototype(vec![category(0), category(0)]),
            Prototype(vec![category(1), category(1)]),
        ],
        0.0,
        &budget,
        &cancel,
    )?;
    let training = Grid::new(
        Shape::new([2, 1, 1])?,
        vec![Variable::Categorical { classes: 2 }; 2],
        vec![Some(0.0), Some(0.0), Some(1.0), Some(1.0)],
        &budget,
    )?;
    for realization in 0..100 {
        let mut state = Grid::new(
            training.shape(),
            training.variables().to_vec(),
            vec![Some(-0.0), None, Some(1.0), None],
            &budget,
        )?;
        terrustrial_core::simulation::high_order::simulate(
            &plan,
            &training,
            &mut state,
            [0, 1],
            terrustrial_core::simulation::high_order::SimulationOptions {
                template: &[[-1, 0, 0]],
                maximum_neighbors: 1,
                training_boundary: Boundary::Bounded,
                seed: 3,
                realization,
                fit: options(&budget, &cancel),
            },
            |_, draw| {
                assert!(draw.hard_target_log_evidence.unwrap().is_finite());
                Ok(())
            },
        )?;
        assert_eq!(
            state.values(),
            &[Some(-0.0), Some(0.0), Some(1.0), Some(1.0)]
        );
        assert_eq!(state.values()[0].unwrap().to_bits(), (-0.0_f64).to_bits());
    }
    let fit = plan.condition(
        &[(vec![0.0, 0.0], vec![]), (vec![1.0, 1.0], vec![])][..],
        &[],
        options(&budget, &cancel),
    )?;
    assert!(fit
        .condition_targets(&[Some(0.0), Some(1.0)], &budget, &cancel)
        .is_err());
    Ok(())
}

#[test]
fn categorical_parity_retains_a_joint_relationship_absent_from_every_pair() -> Result<()> {
    let budget = MemoryBudget::new(1 << 24);
    let cancel = Cancellation::default();
    let axes = [Axis::Categorical { classes: 2 }; 3];
    let prototypes = (0..8)
        .map(|bits| Prototype((0..3).map(|i| category((bits >> i) & 1)).collect()))
        .collect();
    let plan = Plan::new(&axes, prototypes, 0.0, &budget, &cancel)?;
    let rows: Vec<_> = (0..8usize)
        .filter(|bits| bits.count_ones() % 2 == 0)
        .map(|bits| ((0..3).map(|i| ((bits >> i) & 1) as f64).collect(), vec![]))
        .collect();
    let fit = plan.condition(rows.as_slice(), &[], options(&budget, &cancel))?;
    for bits in 0..8usize {
        let values: Vec<_> = (0..3).map(|i| ((bits >> i) & 1) as f64).collect();
        let probability = fit.log_probability(&values)?.exp();
        assert!(
            (probability
                - if bits.count_ones() % 2 == 0 {
                    0.25
                } else {
                    0.0
                })
            .abs()
                < 1e-11
        );
    }
    for a in 0..3 {
        for b in a + 1..3 {
            let mut frequencies = [0.0; 4];
            for (bits, &weight) in fit.weights().iter().enumerate() {
                frequencies[((bits >> a) & 1) * 2 + ((bits >> b) & 1)] += weight;
            }
            assert!(frequencies.iter().all(|x| (x - 0.25).abs() < 1e-11));
        }
    }
    let mut counts = [0; 8];
    for item in 0..20_000 {
        let mut value = [0.0; 3];
        let mut repeat = [0.0; 3];
        fit.draw_into(10, 2, item, &mut value, &cancel)?;
        fit.draw_into(10, 2, item, &mut repeat, &cancel)?;
        assert_eq!(value, repeat);
        let bits = value[0] as usize + 2 * value[1] as usize + 4 * value[2] as usize;
        assert_eq!(bits.count_ones() % 2, 0);
        counts[bits] += 1;
    }
    assert!(counts
        .iter()
        .enumerate()
        .all(|(bits, n)| bits.count_ones() % 2 != 0 || (4700..5300).contains(n)));
    Ok(())
}

#[test]
fn mixed_joint_density_normalizes_and_draws_share_the_component() -> Result<()> {
    let budget = MemoryBudget::new(1 << 24);
    let cancel = Cancellation::default();
    let plan = Plan::new(
        &[
            Axis::Continuous { degree: 6 },
            Axis::Categorical { classes: 2 },
        ],
        vec![
            Prototype(vec![normal(-0.65, 0.07), category(0)]),
            Prototype(vec![normal(0.65, 0.07), category(1)]),
        ],
        0.0,
        &budget,
        &cancel,
    )?;
    let rows = vec![(vec![-0.65, 0.0], vec![0.0]), (vec![0.65, 1.0], vec![1.0])];
    let unconditional_rows: Vec<_> = rows.iter().map(|(v, _)| (v.clone(), vec![])).collect();
    let fit = plan.condition(
        unconditional_rows.as_slice(),
        &[],
        options(&budget, &cancel),
    )?;
    assert!((fit.weights()[0] - 0.5).abs() < 1e-10);
    let mut mass = 0.0;
    for i in 0..20_000 {
        for category in 0..2 {
            let x = -1.0 + (i as f64 + 0.5) / 10_000.0;
            mass += fit.log_probability(&[x, category as f64])?.exp() / 10_000.0;
        }
    }
    assert!((mass - 1.0).abs() < 1e-10, "{mass}");
    for i in 0..10_000 {
        let mut v = [0.0; 2];
        fit.draw_into(42, 0, i, &mut v, &cancel)?;
        assert_eq!(v[0] > 0.0, v[1] == 1.0);
    }
    let conditional = plan.condition(
        rows.as_slice(),
        &[(Axis::Categorical { classes: 2 }, 1.0)],
        options(&budget, &cancel),
    )?;
    assert!(conditional.weights()[1] > 1.0 - 1e-12);
    assert_eq!(conditional.diagnostics.nonzero_kernel_rows, 1);
    assert_eq!(
        conditional.log_probability(&[1.01, 1.0])?,
        f64::NEG_INFINITY
    );
    assert!(conditional.log_probability(&[0.0, 2.0]).is_err());
    Ok(())
}

#[test]
fn signed_embedding_is_projected_to_a_positive_simplex_without_clipping_weights() -> Result<()> {
    let budget = MemoryBudget::new(1 << 24);
    let cancel = Cancellation::default();
    let plan = Plan::new(
        &[Axis::Categorical { classes: 2 }],
        vec![Prototype(vec![category(0)]), Prototype(vec![category(1)])],
        0.0,
        &budget,
        &cancel,
    )?;
    // K_1(1,1)=2 and K_1(-1,1)=-1. The empirical conditional
    // category masses are [2,-1]. The constrained optimum is exactly [1,0].
    let rows = vec![(vec![0.0], vec![1.0]), (vec![1.0], vec![-1.0])];
    let fit = plan.condition(
        rows.as_slice(),
        &[(Axis::Continuous { degree: 1 }, 1.0)],
        options(&budget, &cancel),
    )?;
    assert_eq!(fit.weights(), [1.0, 0.0]);
    assert!((fit.diagnostics.negative_weight_fraction - 1.0 / 3.0).abs() < 1e-14);
    assert!((fit.diagnostics.normalizer_fraction - 1.0 / 3.0).abs() < 1e-14);
    let cancelled_rows = vec![
        (vec![0.0], vec![1.0]),
        (vec![1.0], vec![-1.0]),
        (vec![1.0], vec![-1.0]),
    ];
    assert!(matches!(
        plan.condition(
            cancelled_rows.as_slice(),
            &[(Axis::Continuous { degree: 1 }, 1.0)],
            options(&budget, &cancel)
        ),
        Err(Error::Numerical { .. })
    ));
    let negative_rows = vec![(vec![1.0], vec![-1.0])];
    assert!(plan
        .condition(
            negative_rows.as_slice(),
            &[(Axis::Continuous { degree: 1 }, 1.0)],
            options(&budget, &cancel)
        )
        .is_err());
    Ok(())
}

#[test]
fn log_scaled_conditioning_handles_products_beyond_float_range() -> Result<()> {
    let budget = MemoryBudget::new(1 << 24);
    let cancel = Cancellation::default();
    let plan = Plan::new(
        &[Axis::Categorical { classes: 2 }],
        vec![Prototype(vec![category(0)]), Prototype(vec![category(1)])],
        0.0,
        &budget,
        &cancel,
    )?;
    for (degree, value, dimensions) in [(1, 1.0, 2048), (0, 0.0, 2048)] {
        let rows = vec![
            (vec![0.0], vec![value; dimensions]),
            (vec![1.0], vec![value; dimensions]),
        ];
        let event = vec![(Axis::Continuous { degree }, value); dimensions];
        let fit = plan.condition(rows.as_slice(), &event, options(&budget, &cancel))?;
        assert_eq!(fit.weights(), [0.5, 0.5]);
        let expected = (dimensions as f64)
            * if degree == 0 {
                0.5f64.ln()
            } else {
                2.0f64.ln()
            }
            + 2.0f64.ln();
        assert!((fit.diagnostics.log_normalizer - expected).abs() < 1e-9);
    }
    Ok(())
}

#[test]
fn duplicate_prototypes_ridge_and_fractional_categorical_probabilities() -> Result<()> {
    let budget = MemoryBudget::new(1 << 24);
    let cancel = Cancellation::default();
    let rows = vec![(vec![0.0], vec![])];
    for ridge in [0.0, 0.25] {
        let plan = Plan::new(
            &[Axis::Categorical { classes: 2 }],
            vec![Prototype(vec![category(0)]), Prototype(vec![category(0)])],
            ridge,
            &budget,
            &cancel,
        )?;
        let fit = plan.condition(rows.as_slice(), &[], options(&budget, &cancel))?;
        assert!((fit.log_probability(&[0.0])?.exp() - 1.0).abs() < 1e-14);
        if ridge > 0.0 {
            assert!((fit.weights()[0] - 0.5).abs() < 1e-12);
        }
    }
    // Q=I and q=[1,0]; adding lambda=2 gives analytic optimum [2/3,1/3].
    let plan = Plan::new(
        &[Axis::Categorical { classes: 2 }],
        vec![Prototype(vec![category(0)]), Prototype(vec![category(1)])],
        2.0,
        &budget,
        &cancel,
    )?;
    let fit = plan.condition(rows.as_slice(), &[], options(&budget, &cancel))?;
    assert!((fit.weights()[0] - 2.0 / 3.0).abs() < 1e-12);
    let plan = Plan::new(
        &[Axis::Categorical { classes: 2 }],
        vec![Prototype(vec![Marginal::Categorical(vec![0.2, 0.8])])],
        0.0,
        &budget,
        &cancel,
    )?;
    let fit = plan.condition(rows.as_slice(), &[], options(&budget, &cancel))?;
    assert!((fit.log_probability(&[1.0])?.exp() - 0.8).abs() < 1e-14);
    Ok(())
}

#[test]
fn extreme_valid_normal_scales_have_finite_normalized_densities() -> Result<()> {
    let budget = MemoryBudget::new(1 << 24);
    let cancel = Cancellation::default();
    for (mean, deviation) in [
        (0.0, f64::MAX),
        (0.0, 1e-200),
        (1.0, 1e-5),
        (-1.0, 0.03),
        (0.0, 1.0),
    ] {
        let plan = Plan::new(
            &[Axis::Continuous { degree: 20 }],
            vec![Prototype(vec![normal(mean, deviation)])],
            0.0,
            &budget,
            &cancel,
        )?;
        let rows = vec![(vec![mean], vec![])];
        let fit = plan.condition(rows.as_slice(), &[], options(&budget, &cancel))?;
        let density = fit.log_probability(&[mean])?;
        assert!(density.is_finite(), "{mean}, {deviation}: {density}");
        if deviation == f64::MAX {
            assert!((density + 2.0f64.ln()).abs() < 1e-12, "{density}");
        }
        for item in 0..100 {
            let mut v = [0.0];
            fit.draw_into(93, 1, item, &mut v, &cancel)?;
            assert!((-1.0..=1.0).contains(&v[0]));
        }
    }
    Ok(())
}

#[test]
fn streaming_source_resources_cancellation_validation_and_nonconvergence() -> Result<()> {
    let budget = MemoryBudget::new(1 << 24);
    let cancel = Cancellation::default();
    struct Stream;
    impl ReplicateSource for Stream {
        fn scan(&self, visit: &mut ReplicateVisitor<'_>) -> Result<()> {
            for i in 0..100_000 {
                visit(&[(i % 2) as f64], &[])?;
            }
            Ok(())
        }
    }
    let baseline = budget.used();
    {
        let plan = Plan::new(
            &[Axis::Categorical { classes: 2 }],
            vec![Prototype(vec![category(0)]), Prototype(vec![category(1)])],
            0.0,
            &budget,
            &cancel,
        )?;
        let after_plan = budget.used();
        {
            let fit = plan.condition(&Stream, &[], options(&budget, &cancel))?;
            assert_eq!(fit.diagnostics.rows, 100_000);
            assert_eq!(fit.weights(), [0.5, 0.5]);
            assert!(budget.used() - after_plan <= 16);
        }
        assert_eq!(budget.used(), after_plan);
        let tiny = MemoryBudget::new(1);
        assert!(matches!(
            plan.condition(&Stream, &[], options(&tiny, &cancel)),
            Err(Error::MemoryBudget { .. })
        ));
        assert_eq!(tiny.used(), 0);
        for rows in [
            vec![],
            vec![(vec![2.0], vec![])],
            vec![(vec![0.0], vec![0.0])],
            vec![(vec![f64::NAN], vec![])],
        ] {
            assert!(plan
                .condition(rows.as_slice(), &[], options(&budget, &cancel))
                .is_err());
            assert_eq!(budget.used(), after_plan);
        }
        cancel.cancel();
        assert!(matches!(
            plan.condition(&Stream, &[], options(&budget, &cancel)),
            Err(Error::Cancelled)
        ));
    }
    assert_eq!(budget.used(), baseline);
    let cancel = Cancellation::default();
    let invalid_axes = [Axis::Continuous { degree: usize::MAX }];
    assert!(matches!(
        Plan::new(
            &invalid_axes,
            vec![Prototype(vec![normal(0.0, 0.1)])],
            0.0,
            &budget,
            &cancel
        ),
        Err(Error::SizeOverflow)
    ));
    assert!(Plan::new(
        &[Axis::Continuous { degree: 1 }],
        vec![Prototype(vec![normal(1.0, 1e-100)])],
        0.0,
        &budget,
        &cancel
    )
    .is_err());
    let plan = Plan::new(
        &[Axis::Categorical { classes: 3 }],
        (0..3)
            .map(|i| {
                let mut p = vec![0.0; 3];
                p[i] = 1.0;
                Prototype(vec![Marginal::Categorical(p)])
            })
            .collect(),
        0.0,
        &budget,
        &cancel,
    )?;
    let rows = vec![
        (vec![0.0], vec![]),
        (vec![1.0], vec![]),
        (vec![2.0], vec![]),
    ];
    let mut limited = options(&budget, &cancel);
    limited.maximum_iterations = 1;
    assert!(matches!(
        plan.condition(rows.as_slice(), &[], limited),
        Err(Error::Convergence { .. })
    ));
    Ok(())
}

#[test]
fn grid_replicates_respect_geometry_missing_values_and_boundaries() -> Result<()> {
    use terrustrial_core::{
        grid::{Boundary, Grid, Shape, Variable},
        high_order::grid::{GridReplicates, Location},
    };
    let budget = MemoryBudget::new(1 << 24);
    let cancel = Cancellation::default();
    let training = Grid::new(
        Shape::new([5, 1, 1])?,
        vec![Variable::Categorical { classes: 2 }],
        vec![Some(0.0), Some(1.0), None, Some(0.0), Some(1.0)],
        &budget,
    )?;
    let targets = [0];
    let locations = [Location {
        offset: [-1, 0, 0],
        variable: 0,
    }];
    let source = GridReplicates::new(
        &training,
        &targets,
        &locations,
        Boundary::Bounded,
        &budget,
        &cancel,
    )?;
    let mut rows = vec![];
    source.scan(&mut |targets, condition| {
        rows.push((targets.to_vec(), condition.to_vec()));
        Ok(())
    })?;
    assert_eq!(rows, vec![(vec![1.0], vec![0.0]), (vec![1.0], vec![0.0])]);
    let source = GridReplicates::new(
        &training,
        &targets,
        &locations,
        Boundary::Periodic,
        &budget,
        &cancel,
    )?;
    let mut periodic = vec![];
    source.scan(&mut |targets, condition| {
        periodic.push((targets.to_vec(), condition.to_vec()));
        Ok(())
    })?;
    assert_eq!(
        periodic,
        vec![
            (vec![0.0], vec![1.0]),
            (vec![1.0], vec![0.0]),
            (vec![1.0], vec![0.0])
        ]
    );
    let extreme = [Location {
        offset: [i64::MIN, i64::MAX, i64::MIN],
        variable: 0,
    }];
    let source = GridReplicates::new(
        &training,
        &targets,
        &extreme,
        Boundary::Bounded,
        &budget,
        &cancel,
    )?;
    let mut count = 0;
    source.scan(&mut |_, _| {
        count += 1;
        Ok(())
    })?;
    assert_eq!(count, 0);
    let source = GridReplicates::new(
        &training,
        &targets,
        &extreme,
        Boundary::Periodic,
        &budget,
        &cancel,
    )?;
    source.scan(&mut |_, _| {
        count += 1;
        Ok(())
    })?;
    assert!(count > 0);
    assert!(
        GridReplicates::new(&training, &[0, 0], &[], Boundary::Bounded, &budget, &cancel).is_err()
    );
    assert!(GridReplicates::new(
        &training,
        &targets,
        &[locations[0], locations[0]],
        Boundary::Bounded,
        &budget,
        &cancel
    )
    .is_err());
    Ok(())
}

#[test]
fn joint_high_order_simulation_conditions_spatially_preserves_partial_hard_data_and_repeats(
) -> Result<()> {
    use terrustrial_core::{
        grid::{Boundary, Grid, Shape, Variable},
        simulation::high_order::{simulate, SimulationOptions},
    };
    let budget = MemoryBudget::new(1 << 24);
    let cancel = Cancellation::default();
    let axes = [Axis::Categorical { classes: 2 }; 3];
    let prototypes = (0..8)
        .map(|bits| Prototype((0..3).map(|i| category((bits >> i) & 1)).collect()))
        .collect();
    let plan = Plan::new(&axes, prototypes, 0.0, &budget, &cancel)?;
    let cycle = [0usize, 3, 5, 6];
    let training = Grid::new(
        Shape::new([4, 1, 1])?,
        vec![Variable::Categorical { classes: 2 }; 3],
        cycle
            .iter()
            .flat_map(|bits| (0..3).map(move |i| Some(((bits >> i) & 1) as f64)))
            .collect(),
        &budget,
    )?;
    let mut hard = vec![None; 96];
    hard[2] = Some(-0.0);
    let mut state = Grid::new(
        Shape::new([32, 1, 1])?,
        training.variables().to_vec(),
        hard.clone(),
        &budget,
    )?;
    let mut repeat = Grid::new(
        state.shape(),
        training.variables().to_vec(),
        hard.clone(),
        &budget,
    )?;
    for grid in [&mut state, &mut repeat] {
        let mut outputs = 0;
        simulate(
            &plan,
            &training,
            grid,
            0..32,
            SimulationOptions {
                template: &[[-1, 0, 0]],
                maximum_neighbors: 1,
                training_boundary: Boundary::Periodic,
                seed: 17,
                realization: 8,
                fit: options(&budget, &cancel),
            },
            |_, draw| {
                outputs += 1;
                assert!(draw.diagnostics.rows > 0);
                Ok(())
            },
        )?;
        assert_eq!(outputs, 32);
    }
    assert_eq!(state.values(), repeat.values());
    assert_eq!(state.values()[2].unwrap().to_bits(), (-0.0f64).to_bits());
    let bits: Vec<_> = state
        .values()
        .as_chunks::<3>()
        .0
        .iter()
        .map(|v| v[0].unwrap() as usize + 2 * v[1].unwrap() as usize + 4 * v[2].unwrap() as usize)
        .collect();
    for pair in bits.windows(2) {
        let previous = cycle.iter().position(|v| *v == pair[0]).unwrap();
        assert_eq!(pair[1], cycle[(previous + 1) % 4]);
    }
    // A contradictory fully observed previous node and partially hard current
    // node must fail; no event is silently removed to make progress.
    let mut conflict = Grid::new(
        Shape::new([2, 1, 1])?,
        training.variables().to_vec(),
        vec![Some(0.0), Some(0.0), Some(0.0), Some(0.0), None, None],
        &budget,
    )?;
    let before = conflict.values().to_vec();
    assert!(simulate(
        &plan,
        &training,
        &mut conflict,
        [1],
        SimulationOptions {
            template: &[[-1, 0, 0]],
            maximum_neighbors: 1,
            training_boundary: Boundary::Periodic,
            seed: 17,
            realization: 8,
            fit: options(&budget, &cancel),
        },
        |_, _| Ok(())
    )
    .is_err());
    assert_eq!(conflict.values(), before);
    Ok(())
}
