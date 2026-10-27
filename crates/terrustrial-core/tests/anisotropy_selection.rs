use terrustrial_core::{
    anisotropy::{
        field::{Edge, Options, Window},
        fitting::DirectionalBin,
        selection::select_regularization,
    },
    data::Point3,
    execution::{Cancellation, MemoryBudget},
    Error,
};
fn point(x: f64) -> Point3 {
    Point3::new([x, 0., 0.]).unwrap()
}
fn options<'a>(budget: &'a MemoryBudget, cancellation: &'a Cancellation) -> Options<'a> {
    Options {
        range_bounds: [0.2, 10.],
        regularization: 0.,
        maximum_iterations: 20000,
        tolerance: 1e-11,
        maximum_cg_iterations: 2000,
        cg_tolerance: 1e-12,
        rank_tolerance: 1e-10,
        interpolation_neighbors: 4,
        interpolation_power: 2.,
        budget,
        cancellation,
    }
}
fn bins() -> Vec<DirectionalBin> {
    [
        [1., 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.],
        [1., 1., 0.],
        [1., 0., 1.],
        [0., 1., 1.],
        [1., -1., 1.],
    ]
    .into_iter()
    .flat_map(|d| {
        [0.3_f64, 0.6, 0.9].map(move |h| {
            let lag = d.map(|x| x * h);
            let q: f64 = (0..3).map(|i| (lag[i] / [5., 2., 1.][i]).powi(2)).sum();
            DirectionalBin {
                lag,
                semivariance: 0.1 + 1.7 * -(-q).exp_m1(),
                pairs: 100,
            }
        })
    })
    .collect()
}
#[test]
fn automatic_smoothing_recovers_constant_anisotropy_without_held_out_bins() {
    let bins = bins();
    let windows: Vec<_> = (0..6)
        .map(|i| Window {
            position: point(i as f64),
            bins: &bins,
            structured_sill: 1.7,
            nugget: 0.1,
        })
        .collect();
    let edges: Vec<_> = (0..5)
        .map(|i| Edge {
            nodes: [i, i + 1],
            weight: 1.,
        })
        .collect();
    let budget = MemoryBudget::new(1 << 24);
    let cancel = Cancellation::default();
    let run = || {
        select_regularization(
            &windows,
            &edges,
            &[0, 0, 1, 1, 2, 2],
            &[0.001, 0.1, 1.],
            options(&budget, &cancel),
        )
        .unwrap()
    };
    let result = run();
    assert!(result.fit.converged);
    for c in &result.candidates {
        assert!(c.failures.iter().all(Option::is_none));
        assert!(c.validation_loss.unwrap() < 1e-18, "{:?}", c);
    }
    for (_, tensor) in result.fit.field.anchors() {
        for (i, row) in tensor.matrix().iter().enumerate() {
            for (j, &value) in row.iter().enumerate() {
                assert!((value - if i == j { [25., 4., 1.][i] } else { 0. }).abs() < 1e-7);
            }
        }
    }
    let repeated = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap()
        .install(run);
    assert_eq!(result.selected, repeated.selected);
    assert_eq!(
        result
            .candidates
            .iter()
            .map(|c| c.validation_loss.unwrap().to_bits())
            .collect::<Vec<_>>(),
        repeated
            .candidates
            .iter()
            .map(|c| c.validation_loss.unwrap().to_bits())
            .collect::<Vec<_>>()
    );
    assert_eq!(budget.used(), 0);
}
#[test]
fn validation_only_direction_cannot_rescue_training_rank() {
    let full = bins();
    let planar: Vec<_> = full.iter().filter(|b| b.lag[2] == 0.).cloned().collect();
    let windows = [
        Window {
            position: point(0.),
            bins: &full,
            structured_sill: 1.7,
            nugget: 0.1,
        },
        Window {
            position: point(1.),
            bins: &planar,
            structured_sill: 1.7,
            nugget: 0.1,
        },
    ];
    let budget = MemoryBudget::new(1 << 20);
    let cancel = Cancellation::default();
    let error = select_regularization(
        &windows,
        &[Edge {
            nodes: [0, 1],
            weight: 1.,
        }],
        &[0, 1],
        &[0.1],
        options(&budget, &cancel),
    );
    match error {
        Err(Error::InvalidInput(s)) => assert!(s.contains("rank"), "{s}"),
        _ => panic!("held-out bins must not enter training"),
    }
    assert_eq!(budget.used(), 0);
}
#[test]
fn selection_rejects_invalid_folds_and_propagates_resource_cancellation() {
    let b = bins();
    let w = [
        Window {
            position: point(0.),
            bins: &b,
            structured_sill: 1.7,
            nugget: 0.1,
        },
        Window {
            position: point(1.),
            bins: &b,
            structured_sill: 1.7,
            nugget: 0.1,
        },
    ];
    let e = [Edge {
        nodes: [0, 1],
        weight: 1.,
    }];
    let budget = MemoryBudget::new(1 << 20);
    let cancel = Cancellation::default();
    for folds in [&[0, 0][..], &[0, 2][..], &[0][..]] {
        assert!(select_regularization(&w, &e, folds, &[0.1], options(&budget, &cancel)).is_err());
    }
    let tiny = MemoryBudget::new(1);
    assert!(matches!(
        select_regularization(&w, &e, &[0, 1], &[0.1], options(&tiny, &cancel)),
        Err(Error::MemoryBudget { .. })
    ));
    cancel.cancel();
    assert!(matches!(
        select_regularization(&w, &e, &[0, 1], &[0.1], options(&budget, &cancel)),
        Err(Error::Cancelled)
    ));
    assert_eq!(budget.used(), 0);
}

#[test]
fn prediction_scorer_sees_only_training_fit_and_negative_losses_are_valid() {
    use terrustrial_core::anisotropy::{selection::select_families_by, Family};
    let bins = bins();
    let windows: Vec<_> = (0..3)
        .map(|i| Window {
            position: point(i as f64),
            bins: &bins,
            structured_sill: 1.7,
            nugget: 0.1,
        })
        .collect();
    let edges = [
        Edge {
            nodes: [0, 1],
            weight: 1.,
        },
        Edge {
            nodes: [1, 2],
            weight: 1.,
        },
    ];
    let budget = MemoryBudget::new(1 << 20);
    let cancel = Cancellation::default();
    let mut calls = 0;
    let selected = select_families_by(
        &windows,
        &edges,
        &[0, 1, 2],
        &[Family::Gaussian],
        &[0.1, 1.],
        options(&budget, &cancel),
        |fit, fold| {
            calls += 1;
            assert_eq!(fit.nodes[fold].used_bins, 0);
            assert_eq!(fit.nodes[fold].design_rank, 0);
            assert!(fit
                .nodes
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != fold)
                .all(|(_, n)| n.used_bins > 0));
            Ok(-1. - fold as f64)
        },
    )
    .unwrap();
    assert_eq!(calls, 6);
    assert_eq!(selected.selected, 0);
    assert_eq!(selected.candidates[0].validation_loss, Some(-2.));
    assert!(selected.fit.nodes.iter().all(|n| n.used_bins > 0));
    assert!(select_families_by(
        &windows,
        &edges,
        &[0, 1, 2],
        &[Family::Gaussian],
        &[0.1],
        options(&budget, &cancel),
        |_, _| Ok(f64::NAN)
    )
    .is_err());
    assert!(matches!(
        select_families_by(
            &windows,
            &edges,
            &[0, 1, 2],
            &[Family::Gaussian],
            &[0.1],
            options(&budget, &cancel),
            |_, _| Err(Error::Cancelled)
        ),
        Err(Error::Cancelled)
    ));
    assert_eq!(budget.used(), 0);
}
