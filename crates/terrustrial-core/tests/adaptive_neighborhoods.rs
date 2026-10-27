use terrustrial_core::storage::PointFile;
use terrustrial_core::{
    data::{Observation, Point3, Site},
    execution::{Cancellation, MemoryBudget},
    model::Anisotropy,
    spatial::{
        MinimumQuotas, ObservationSource, SearchNeighborhood, SearchStop, SearchWorkspace,
        SpatialIndex, VariableLimit, VariableMinimum,
    },
    Error, Result,
};
fn point(p: [f64; 3]) -> Point3 {
    Point3::new(p).unwrap()
}
fn row(id: u64, p: [f64; 3], variable: usize) -> Observation {
    Observation::new(id, Site::point(point(p), variable), (id % 100) as f64, 0.).unwrap()
}
fn ids(rows: &[Observation]) -> Vec<u64> {
    rows.iter().map(Observation::id).collect()
}
fn configuration(minimum: usize, scales: Vec<[f64; 3]>) -> SearchNeighborhood {
    SearchNeighborhood::new(
        4,
        Anisotropy::new([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]], [0.25; 3]).unwrap(),
        Some(2),
        (0..3)
            .map(|variable| VariableLimit {
                variable,
                maximum: 2,
            })
            .collect(),
    )
    .unwrap()
    .with_minimums(MinimumQuotas {
        total: minimum,
        variables: vec![VariableMinimum {
            variable: 2,
            minimum: 1,
        }],
        octants: [0; 8],
    })
    .unwrap()
    .with_expansion(scales)
    .unwrap()
}
// Full materialization at each geometry, followed by exhaustive subset selection.
// Uses explicit frame coordinates, no production search/retention/flow routines.
fn oracle(
    rows: &[Observation],
    center: [f64; 3],
    levels: &[[f64; 3]],
    minimum: usize,
) -> Option<(usize, Vec<u64>)> {
    for (level, radii) in levels.iter().enumerate() {
        let mut candidates = Vec::new();
        for row in rows {
            if row.site().variable >= 3 {
                continue;
            }
            let p = row.site().support.center().coordinates();
            let local = [p[1] - center[1], center[0] - p[0], p[2] - center[2]];
            let distance = local
                .iter()
                .zip(radii)
                .map(|(x, r)| (x / r).powi(2))
                .sum::<f64>()
                .sqrt();
            if distance <= 1. {
                let sector = (local[0] < 0.) as usize
                    | (((local[1] < 0.) as usize) << 1)
                    | (((local[2] < 0.) as usize) << 2);
                candidates.push((distance, row.id(), row.site().variable, sector));
            }
        }
        candidates.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        let mut best: Option<Vec<usize>> = None;
        for mask in 0..1_usize << candidates.len() {
            let n = mask.count_ones() as usize;
            if n < minimum || n > 4 {
                continue;
            }
            let mut variables = [0; 3];
            let mut sectors = [0; 8];
            let mut ranks = Vec::new();
            for (i, &(_, _, variable, sector)) in candidates.iter().enumerate() {
                if mask & (1 << i) != 0 {
                    variables[variable] += 1;
                    sectors[sector] += 1;
                    ranks.push(i);
                }
            }
            if variables[2] == 0
                || variables.iter().any(|&n| n > 2)
                || sectors.iter().any(|&n| n > 2)
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
        if let Some(best) = best {
            return Some((level, best.into_iter().map(|i| candidates[i].1).collect()));
        }
    }
    None
}
#[test]
fn first_feasible_geometry_matches_independent_subset_reference_and_disk_scan() {
    let budget = MemoryBudget::new(16 << 20);
    let cancel = Cancellation::default();
    let temp = tempfile::tempdir().unwrap();
    let levels = [[0.25; 3], [0.8, 0.4, 0.25], [0.8, 1.8, 0.6], [3.; 3]];
    let mut data: Vec<_> = (0..10_u64)
        .map(|i| {
            row(
                if i == 0 { u64::MAX } else { i },
                [
                    0.071 + (i * 7 % 13) as f64 * 0.11,
                    -0.057 + (i * 5 % 11) as f64 * 0.087,
                    (i % 3) as f64 * 0.051,
                ],
                (i % 3) as usize,
            )
        })
        .collect();
    data.push(row(22, [0.071, -0.057, 0.], 0));
    data.push(row(23, [0.01; 3], 99));
    let index = SpatialIndex::new(data.clone()).unwrap();
    let file = PointFile::create(
        temp.path().join("adaptive.points"),
        data.as_slice(),
        &cancel,
    )
    .unwrap();
    for minimum in 1..=4 {
        let query = configuration(minimum, levels[1..].to_vec());
        let mut workspace = SearchWorkspace::new(&query, &budget).unwrap();
        let bytes = workspace.accounted_bytes();
        for center in [[0.; 3], [0.2, 0.1, 0.], [1.7, 0.5, 0.], [4.; 3], [0.; 3]] {
            let expected = oracle(&data, center, &levels, minimum);
            for source in [&index as &dyn ObservationSource, &file] {
                let actual = source.search(point(center), &mut workspace, &cancel);
                match &expected {
                    Some((_, expected)) => assert_eq!(ids(actual.unwrap()), *expected),
                    None => assert!(matches!(actual, Err(Error::InsufficientNeighborhood))),
                }
                let report = workspace.diagnostics();
                assert_eq!(
                    report.level,
                    expected.as_ref().map_or(levels.len() - 1, |e| e.0)
                );
                assert_eq!(report.attempts, report.level + 1);
                assert_eq!(
                    report.stop,
                    Some(if expected.is_some() {
                        SearchStop::Satisfied
                    } else {
                        SearchStop::Exhausted
                    })
                );
                assert!(report.complete);
                assert_eq!(report.minimums_satisfied, expected.is_some());
                assert_eq!(budget.used(), bytes);
            }
            let mut reverse = data.clone();
            reverse.reverse();
            let result = reverse.search(point(center), &mut workspace, &cancel);
            if let Some((_, expected)) = expected {
                assert_eq!(ids(result.unwrap()), expected);
            } else {
                assert!(matches!(result, Err(Error::InsufficientNeighborhood)));
            }
        }
        let allocations = allocation_counter::measure(|| {
            for _ in 0..32 {
                assert!(index
                    .search(point([0.; 3]), &mut workspace, &cancel)
                    .is_ok());
            }
        });
        assert_eq!(allocations.count_total, 0);
    }
    assert_eq!(budget.used(), 0);
}

#[test]
fn explicit_limits_reject_bad_geometry_and_keep_memory_independent_of_attempts() {
    let base =
        SearchNeighborhood::new(3, Anisotropy::isotropic(1.).unwrap(), None, vec![]).unwrap();
    assert!(base.clone().with_expansion(vec![[2.; 3]]).is_err());
    let base = base
        .with_minimums(MinimumQuotas {
            total: 2,
            ..Default::default()
        })
        .unwrap();
    for levels in [
        vec![[1.; 3]],
        vec![[2., 0.5, 1.]],
        vec![[f64::INFINITY; 3]],
        vec![[f64::NAN; 3]],
        vec![[2.; 3], [1.5; 3]],
        vec![[0.; 3]],
        vec![[2.; 3], [2.; 3]],
    ] {
        assert!(base.clone().with_expansion(levels).is_err());
    }
    let expanded = base
        .clone()
        .with_expansion((2..258).map(|r| [r as f64; 3]).collect())
        .unwrap();
    assert_eq!(expanded.expansion_scales().len(), 256);
    assert!(expanded
        .clone()
        .with_minimums(MinimumQuotas::default())
        .is_err());
    let cleared = expanded
        .clone()
        .with_expansion(vec![])
        .unwrap()
        .with_minimums(MinimumQuotas::default())
        .unwrap();
    assert!(cleared.minimums().is_none());
    let budget = MemoryBudget::new(1 << 20);
    let a = SearchWorkspace::new(&base, &budget).unwrap();
    let b = SearchWorkspace::new(&expanded, &budget).unwrap();
    assert_eq!(a.accounted_bytes(), b.accounted_bytes());
    let small = MemoryBudget::new(b.accounted_bytes() - 1);
    assert!(matches!(
        SearchWorkspace::new(&expanded, &small),
        Err(Error::MemoryBudget { .. })
    ));
    assert_eq!(small.used(), 0);
    drop((a, b));
    assert_eq!(budget.used(), 0);
}

#[test]
fn expansion_stops_on_feasibility_or_exhaustion_and_never_masks_source_errors() {
    use std::cell::Cell;
    struct Source {
        attempts: Cell<usize>,
        fail_at: usize,
        cancel_at: usize,
    }
    impl ObservationSource for Source {
        fn scan(&self, _: &mut dyn FnMut(&Observation) -> Result<()>) -> Result<()> {
            panic!("indexed visit required")
        }
        fn visit_bounds(
            &self,
            _: [f64; 3],
            _: [f64; 3],
            cancel: &Cancellation,
            visitor: &mut dyn FnMut(&Observation) -> Result<()>,
        ) -> Result<()> {
            let attempt = self.attempts.get() + 1;
            self.attempts.set(attempt);
            visitor(&row(1, [1.5, 0., 0.], 0))?;
            visitor(&row(2, [-1.5, 0., 0.], 0))?;
            if attempt == self.fail_at {
                return Err(Error::Numerical {
                    context: "injected source failure",
                });
            }
            if attempt == self.cancel_at {
                cancel.cancel();
            }
            Ok(())
        }
    }
    let query = SearchNeighborhood::new(3, Anisotropy::isotropic(1.).unwrap(), None, vec![])
        .unwrap()
        .with_minimums(MinimumQuotas {
            total: 2,
            ..Default::default()
        })
        .unwrap()
        .with_expansion(vec![[2.; 3], [4.; 3]])
        .unwrap();
    let budget = MemoryBudget::new(1 << 20);
    let mut workspace = SearchWorkspace::new(&query, &budget).unwrap();
    for (fail, cancel) in [(0, 0), (2, 0), (0, 2), (1, 0), (0, 1), (0, 0)] {
        let source = Source {
            attempts: Cell::new(0),
            fail_at: fail,
            cancel_at: cancel,
        };
        let token = Cancellation::default();
        let result = source.search(point([0.; 3]), &mut workspace, &token);
        if fail != 0 {
            assert!(matches!(result, Err(Error::Numerical { .. })));
        } else if cancel != 0 {
            assert!(matches!(result, Err(Error::Cancelled)));
        } else {
            assert_eq!(ids(result.unwrap()), [1, 2]);
        }
        if fail != 0 || cancel != 0 {
            assert_eq!(source.attempts.get(), fail.max(cancel));
        } else {
            // Spatial density probes may revisit a geometry. Expansion still
            // stops at the first feasible configured geometry.
            assert!(source.attempts.get() >= 2);
            assert_eq!(workspace.diagnostics().attempts, 2);
        }
        assert_eq!(workspace.diagnostics().complete, fail == 0 && cancel == 0);
        assert_eq!(
            workspace.diagnostics().stop,
            if fail == 0 && cancel == 0 {
                Some(SearchStop::Satisfied)
            } else {
                None
            }
        );
    }
    let token = Cancellation::default();
    token.cancel();
    assert!(matches!(
        [].as_slice().search(point([0.; 3]), &mut workspace, &token),
        Err(Error::Cancelled)
    ));
    assert_eq!(workspace.diagnostics().attempts, 0);
    assert!(matches!(
        [].as_slice()
            .search(point([0.; 3]), &mut workspace, &Cancellation::default()),
        Err(Error::InsufficientNeighborhood)
    ));
    assert_eq!(workspace.diagnostics().attempts, 3);
    assert_eq!(workspace.diagnostics().stop, Some(SearchStop::Exhausted));
}

#[cfg(feature = "sqlite")]
#[test]
fn sqlite_readers_match_resident_stages_and_complete_block_payloads() {
    use terrustrial_core::{
        data::Support,
        identity,
        storage::{SqliteReadPool, SqliteStore},
    };
    let budget = MemoryBudget::new(16 << 20);
    let cancel = Cancellation::default();
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("adaptive.sqlite");
    let mut data: Vec<_> = (0..9)
        .map(|i| row(i, [0.12 + i as f64 * 0.21, 0.01, 0.], (i % 3) as usize))
        .collect();
    for (i, o) in data.iter_mut().enumerate().filter(|(i, _)| i % 2 != 0) {
        let site = Site {
            support: Support::block(o.site().support.center(), [0.1, 0.02, 0.], [2, 2, 1]).unwrap(),
            variable: o.site().variable,
        };
        *o = Observation::new(o.id(), site, if i == 1 { -0. } else { o.value() }, 0.02).unwrap();
    }
    let mut store = SqliteStore::create(&path, 4096, &budget).unwrap();
    store.import(data.as_slice(), &cancel).unwrap();
    let pool = SqliteReadPool::open(&path, 2, 4096, &budget).unwrap();
    let snapshot = SqliteReadPool::open_snapshot(&path, 2, 4096, &budget, &cancel).unwrap();
    let query = configuration(3, vec![[0.5; 3], [1.; 3], [3.; 3]]);
    let mut workspace = SearchWorkspace::new(&query, &budget).unwrap();
    for center in [[0.; 3], [1.5, 0., 0.], [100.; 3], [0.; 3]] {
        let expected = data
            .search(point(center), &mut workspace, &cancel)
            .map(|found| identity::observations(found, &cancel).unwrap());
        let report = workspace.diagnostics();
        for source in [&store as &dyn ObservationSource, &pool, &snapshot] {
            let found = source.search(point(center), &mut workspace, &cancel);
            match &expected {
                Ok(expected) => assert_eq!(
                    identity::observations(found.unwrap(), &cancel).unwrap(),
                    *expected
                ),
                Err(_) => assert!(matches!(found, Err(Error::InsufficientNeighborhood))),
            }
            assert_eq!(workspace.diagnostics(), report);
        }
    }
    drop((workspace, store, pool, snapshot));
    assert_eq!(budget.used(), 0);
}

#[test]
fn extreme_expansions_keep_finite_scoring_and_include_closed_boundaries() {
    let budget = MemoryBudget::new(1 << 20);
    let cancel = Cancellation::default();
    for (center, base, expanded, data, expected) in [
        (
            [0.; 3],
            1e-200,
            2e-200,
            vec![
                row(u64::MAX, [2e-200, 0., 0.], 0),
                row(1, [-2e-200, 0., 0.], 0),
                row(2, [2e-200_f64.next_up(), 0., 0.], 0),
            ],
            vec![1, u64::MAX],
        ),
        (
            [-1.2e308, 0., 0.],
            1.,
            1.7e308,
            vec![
                row(1, [-1.2e308, 0., 0.], 0),
                row(2, [0., 0., 0.], 0),
                row(3, [1.2e308, 0., 0.], 0),
            ],
            vec![1, 2],
        ),
    ] {
        let query = SearchNeighborhood::new(3, Anisotropy::isotropic(base).unwrap(), None, vec![])
            .unwrap()
            .with_minimums(MinimumQuotas {
                total: 2,
                ..Default::default()
            })
            .unwrap()
            .with_expansion(vec![[expanded; 3]])
            .unwrap();
        let mut workspace = SearchWorkspace::new(&query, &budget).unwrap();
        let index = SpatialIndex::new(data.clone()).unwrap();
        assert_eq!(
            ids(data.search(point(center), &mut workspace, &cancel).unwrap()),
            expected
        );
        assert_eq!(workspace.diagnostics().level, 1);
        assert_eq!(
            ids(index
                .search(point(center), &mut workspace, &cancel)
                .unwrap()),
            expected
        );
    }
}
