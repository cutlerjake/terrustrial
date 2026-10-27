use terrustrial_core::{
    anisotropy::evidence::{self, Marginal, Options},
    data::Point3,
    execution::{Cancellation, MemoryBudget},
    variography::{self, Direction, Sample},
    Error,
};
fn samples() -> Vec<Sample> {
    (0..12)
        .map(|i| Sample {
            point: Point3::new([i as f64, (i % 3) as f64, (i % 2) as f64]).unwrap(),
            values: vec![
                Some((i * i) as f64),
                if i % 3 == 0 {
                    None
                } else {
                    Some(2. * i as f64)
                },
            ],
        })
        .collect()
}
fn options<'a>(
    directions: &'a [Direction],
    edges: &'a [f64],
    budget: &'a MemoryBudget,
    cancel: &'a Cancellation,
) -> Options<'a> {
    Options {
        maximum_groups: 2,
        maximum_samples: 8,
        link_neighbors: 2,
        lag_edges: edges,
        directions,
        workspace_blocks: 2,
        budget,
        cancellation: cancel,
    }
}
#[test]
fn automatic_windows_preserve_groups_and_match_independent_pair_oracle() {
    let samples = samples();
    let groups: Vec<_> = (0..12).map(|i| i / 2).collect();
    let directions = [Direction::new([1., 0., 0.], std::f64::consts::FRAC_PI_2, 100.).unwrap()];
    let edges = [2., 5., 100.];
    let budget = MemoryBudget::new(1 << 24);
    let cancel = Cancellation::default();
    let marginals = [
        Marginal {
            variable: 0,
            structured_sill: 2.,
            nugget: 0.,
        },
        Marginal {
            variable: 1,
            structured_sill: 4.,
            nugget: 0.,
        },
    ];
    let result = evidence::build(
        &samples,
        &groups,
        &marginals,
        options(&directions, &edges, &budget, &cancel),
    )
    .unwrap();
    let mut seen = Vec::new();
    for w in result.windows() {
        assert!(w.groups.len() <= 2 && w.samples <= 8);
        seen.extend_from_slice(&w.groups);
        let local: Vec<_> = samples
            .iter()
            .zip(&groups)
            .filter(|(_, g)| w.groups.contains(g))
            .map(|(s, _)| s.clone())
            .collect();
        assert_eq!(local.len(), w.samples);
        let mut expected = Vec::new();
        for m in marginals {
            for b in variography::experimental(
                &local,
                m.variable,
                m.variable,
                &edges,
                Some(&directions[0]),
                &cancel,
            )
            .unwrap()
            {
                if let (Some(h), Some(g)) = (b.mean_distance, b.semivariance) {
                    expected.push((h, g / m.structured_sill, b.pairs));
                }
            }
        }
        assert_eq!(w.bins.len(), expected.len());
        for (actual, (h, g, pairs)) in w.bins.iter().zip(expected) {
            assert_eq!(actual.pairs, pairs);
            assert!((actual.lag[0] - h).abs() < 1e-12);
            assert!((actual.semivariance - g).abs() < 1e-10);
        }
    }
    seen.sort_unstable();
    assert_eq!(seen, (0..6).collect::<Vec<_>>());
    assert_eq!(budget.used(), result.accounted_bytes());
    assert!(result.edges().iter().all(|e| e.nodes[0] < e.nodes[1]));
    drop(result);
    assert_eq!(budget.used(), 0);
}
#[test]
fn values_do_not_change_geometry_and_excluded_evidence_is_reported() {
    let mut samples = samples();
    let groups: Vec<_> = (0..12).map(|i| i / 2).collect();
    let d = [Direction::new([1., 0., 0.], std::f64::consts::FRAC_PI_2, 100.).unwrap()];
    let edges = [100.];
    let b = MemoryBudget::new(1 << 24);
    let c = Cancellation::default();
    let m = [Marginal {
        variable: 0,
        structured_sill: 1.,
        nugget: 1.,
    }];
    let first = evidence::build(&samples, &groups, &m, options(&d, &edges, &b, &c)).unwrap();
    for s in &mut samples {
        s.values[0] = Some(0.);
    }
    let second = evidence::build(&samples, &groups, &m, options(&d, &edges, &b, &c)).unwrap();
    for (a, z) in first.windows().iter().zip(second.windows()) {
        assert_eq!(a.position, z.position);
        assert_eq!(a.groups, z.groups);
        assert_eq!(z.below_nugget_bins, 1);
        assert!(z.bins.is_empty());
    }
}
#[test]
fn oversized_groups_invalid_priors_and_resource_errors_do_not_split_or_leak() {
    let samples = samples();
    let d = [Direction::new([1., 0., 0.], 0.5, 100.).unwrap()];
    let edges = [100.];
    let b = MemoryBudget::new(1 << 24);
    let c = Cancellation::default();
    let m = [Marginal {
        variable: 0,
        structured_sill: 1.,
        nugget: 0.,
    }];
    assert!(evidence::build(&samples, &[0; 12], &m, options(&d, &edges, &b, &c)).is_err());
    assert_eq!(b.used(), 0);
    assert!(evidence::build(
        &samples,
        &[0; 12],
        &[m[0], m[0]],
        options(&d, &edges, &b, &c)
    )
    .is_err());
    let tiny = MemoryBudget::new(1);
    assert!(matches!(
        evidence::build(&samples, &[0; 12], &m, options(&d, &edges, &tiny, &c)),
        Err(Error::MemoryBudget { .. })
    ));
    c.cancel();
    assert!(matches!(
        evidence::build(&samples, &[0; 12], &m, options(&d, &edges, &b, &c)),
        Err(Error::Cancelled)
    ));
    assert_eq!(b.used(), 0);
}
