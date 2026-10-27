use terrustrial_core::storage::PointFile;
use terrustrial_core::{
    data::{Observation, Point3, Site},
    execution::{Cancellation, MemoryBudget},
    model::Anisotropy,
    spatial::{
        ObservationSource, SearchNeighborhood, SearchWorkspace, SpatialIndex, VariableLimit,
    },
    Error,
};
struct Scan<'a>(&'a [Observation]);
impl ObservationSource for Scan<'_> {
    fn scan(
        &self,
        visitor: &mut dyn FnMut(&Observation) -> terrustrial_core::Result<()>,
    ) -> terrustrial_core::Result<()> {
        self.0.scan(visitor)
    }
}
fn point(p: [f64; 3]) -> Point3 {
    Point3::new(p).unwrap()
}
fn obs(id: u64, p: [f64; 3], variable: usize) -> Observation {
    Observation::new(
        id,
        Site::point(point(p), variable),
        0.01 * (id % 100) as f64,
        0.0,
    )
    .unwrap()
}
fn ids(data: &[Observation]) -> Vec<u64> {
    data.iter().map(Observation::id).collect()
}
fn geometry() -> Anisotropy {
    let angle = 0.37_f64;
    let (s, c) = angle.sin_cos();
    Anisotropy::new(
        [[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]],
        [3.5, 1.25, 2.0],
    )
    .unwrap()
}
fn observations() -> Vec<Observation> {
    (0..2400)
        .map(|i| {
            obs(
                if i == 0 { u64::MAX } else { i },
                [
                    ((i * 37) % 101) as f64 / 12.0 - 4.0,
                    ((i * 19) % 73) as f64 / 11.0 - 3.0,
                    ((i * 13) % 59) as f64 / 13.0 - 2.0,
                ],
                (i % 4) as usize,
            )
        })
        .collect()
}
// Independent materialize/sort/greedy oracle. It deliberately retains all
// eligible observations and does not use SearchWorkspace or its geometry code.
fn oracle(
    data: &[Observation],
    center: [f64; 3],
    maximum: usize,
    cap: Option<usize>,
    limits: &[VariableLimit],
) -> Vec<u64> {
    let (s, c) = 0.37_f64.sin_cos();
    let mut candidates = Vec::new();
    for o in data {
        if !limits.is_empty() && !limits.iter().any(|l| l.variable == o.site().variable) {
            continue;
        }
        let p = o.site().support.center().coordinates();
        let [x, y, z] = std::array::from_fn(|i| p[i] - center[i]);
        let local = [c * x + s * y, -s * x + c * y, z];
        let d = ((local[0] / 3.5).powi(2) + (local[1] / 1.25).powi(2) + (local[2] / 2.0).powi(2))
            .sqrt();
        if d > 1.0 {
            continue;
        }
        let sector = if cap.is_some() {
            (local[0] < 0.0) as usize
                | (((local[1] < 0.0) as usize) << 1)
                | (((local[2] < 0.0) as usize) << 2)
        } else {
            0
        };
        candidates.push((d, o.id(), o.site().variable, sector));
    }
    candidates.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    let mut counts = std::collections::HashMap::<usize, usize>::new();
    let mut sectors = [0; 8];
    let mut out = Vec::new();
    for (_, id, v, sector) in candidates {
        let limit = limits
            .iter()
            .find(|l| l.variable == v)
            .map_or(maximum, |l| l.maximum);
        if *counts.get(&v).unwrap_or(&0) >= limit || sectors[sector] >= cap.unwrap_or(maximum) {
            continue;
        }
        out.push(id);
        *counts.entry(v).or_default() += 1;
        sectors[sector] += 1;
        if out.len() == maximum {
            break;
        }
    }
    out
}

#[test]
fn bounded_selection_matches_independent_global_greedy_oracle_and_reuses_buffers() {
    let data = observations();
    let mut reverse = data.clone();
    reverse.reverse();
    let index = SpatialIndex::new(reverse.clone()).unwrap();
    let temp = tempfile::tempdir().unwrap();
    let cancel = Cancellation::default();
    let file =
        PointFile::create(temp.path().join("input.points"), data.as_slice(), &cancel).unwrap();
    let budget = MemoryBudget::new(1 << 24);
    for maximum in [1, 7, 32, 71] {
        for cap in [None, Some(0), Some(1), Some(4), Some(100)] {
            for limits in [
                vec![],
                vec![
                    VariableLimit {
                        variable: 0,
                        maximum: 2,
                    },
                    VariableLimit {
                        variable: 1,
                        maximum: 15,
                    },
                    VariableLimit {
                        variable: 3,
                        maximum: 0,
                    },
                ],
            ] {
                let query =
                    SearchNeighborhood::new(maximum, geometry(), cap, limits.clone()).unwrap();
                let mut workspace = SearchWorkspace::new(&query, &budget).unwrap();
                let reserved = workspace.accounted_bytes();
                for center in [[0.0; 3], [1.7, -0.1, 1.0], [100.0; 3], [0.0; 3]] {
                    let expected = oracle(&data, center, maximum, cap, &limits);
                    for source in [
                        &Scan(&data) as &dyn ObservationSource,
                        &Scan(&reverse),
                        &index,
                        &file,
                    ] {
                        assert_eq!(
                            ids(source
                                .search(point(center), &mut workspace, &cancel)
                                .unwrap()),
                            expected
                        );
                        assert_eq!(budget.used(), reserved);
                    }
                }
            }
        }
    }
    assert_eq!(budget.used(), 0);
}

#[test]
fn crossing_caps_ties_planes_and_tiny_or_huge_coordinates_are_deterministic() {
    let cancel = Cancellation::default();
    let budget = MemoryBudget::new(1 << 20);
    let query = SearchNeighborhood::new(
        5,
        Anisotropy::isotropic(10.0).unwrap(),
        Some(2),
        vec![
            VariableLimit {
                variable: 0,
                maximum: 3,
            },
            VariableLimit {
                variable: 1,
                maximum: 2,
            },
        ],
    )
    .unwrap();
    let data = [
        obs(u64::MAX, [1.0, 0.0, 0.0], 0),
        obs(1, [1.0, -0.0, 0.0], 0),
        obs(2, [0.5, 0.0, 0.0], 0),
        obs(3, [-2.0, 0.0, 0.0], 0),
        obs(4, [-3.0, 0.0, 0.0], 1),
        obs(5, [0.0, -4.0, 0.0], 1),
        obs(6, [0.01, 0.0, 0.0], 99),
    ];
    let mut workspace = SearchWorkspace::new(&query, &budget).unwrap();
    assert_eq!(
        ids(data[..]
            .search(point([0.0; 3]), &mut workspace, &cancel)
            .unwrap()),
        [2, 1, 3, 4, 5]
    );
    drop(workspace);
    for (radius, center, data, expected) in [
        (
            1e-200,
            [0.0; 3],
            vec![
                obs(1, [1e-200, 0.0, 0.0], 0),
                obs(2, [1e-200_f64.next_up(), 0.0, 0.0], 0),
                obs(3, [0.0; 3], 0),
            ],
            vec![3, 1],
        ),
        (
            1.7e308,
            [0.0; 3],
            vec![
                obs(1, [1.2e308, 1.2e308, 0.0], 0),
                obs(2, [-1.2e308, -1.2e308, 0.0], 0),
                obs(3, [1.3e308, 1.3e308, 0.0], 0),
            ],
            vec![1, 2],
        ),
        (
            1.7e308,
            [-1.2e308, 0.0, 0.0],
            vec![
                obs(1, [1.2e308, 0.0, 0.0], 0),
                obs(2, [-1.2e308, 0.0, 0.0], 0),
            ],
            vec![2],
        ),
    ] {
        let r = 0.5_f64.sqrt();
        let query = SearchNeighborhood::new(
            10,
            Anisotropy::new([[r, r, 0.0], [-r, r, 0.0], [0.0, 0.0, 1.0]], [radius; 3]).unwrap(),
            None,
            vec![],
        )
        .unwrap();
        let mut workspace = SearchWorkspace::new(&query, &budget).unwrap();
        let index = SpatialIndex::new(data.clone()).unwrap();
        for source in [&Scan(&data) as &dyn ObservationSource, &index] {
            assert_eq!(
                ids(source
                    .search(point(center), &mut workspace, &cancel)
                    .unwrap()),
                expected
            );
        }
    }
    assert_eq!(budget.used(), 0);
}

#[test]
fn invalid_configurations_budgets_cancellation_and_failed_sources_are_explicit() {
    let g = geometry();
    assert!(SearchNeighborhood::new(0, g.clone(), None, vec![]).is_err());
    assert!(SearchNeighborhood::new(
        3,
        g.clone(),
        None,
        vec![
            VariableLimit {
                variable: 1,
                maximum: 1
            };
            2
        ]
    )
    .is_err());
    let impossible =
        SearchNeighborhood::new(usize::MAX, g.clone(), Some(usize::MAX), vec![]).unwrap();
    assert!(matches!(
        SearchWorkspace::new(&impossible, &MemoryBudget::new(u64::MAX)),
        Err(Error::SizeOverflow)
    ));
    let query = SearchNeighborhood::new(8, g, None, vec![]).unwrap();
    let tiny = MemoryBudget::new(1);
    assert!(matches!(
        SearchWorkspace::new(&query, &tiny),
        Err(Error::MemoryBudget { .. })
    ));
    assert_eq!(tiny.used(), 0);
    let budget = MemoryBudget::new(1 << 20);
    let mut workspace = SearchWorkspace::new(&query, &budget).unwrap();
    let cancel = Cancellation::default();
    cancel.cancel();
    assert!(matches!(
        observations()
            .as_slice()
            .search(point([0.0; 3]), &mut workspace, &cancel),
        Err(Error::Cancelled)
    ));
    struct Failed;
    impl ObservationSource for Failed {
        fn scan(
            &self,
            visitor: &mut dyn FnMut(&Observation) -> terrustrial_core::Result<()>,
        ) -> terrustrial_core::Result<()> {
            visitor(&obs(0, [0.0; 3], 0))?;
            Err(Error::Io(std::io::Error::other("injected source failure")))
        }
    }
    assert!(Failed
        .search(point([0.0; 3]), &mut workspace, &Cancellation::default())
        .is_err());
    assert!([]
        .as_slice()
        .search(point([0.0; 3]), &mut workspace, &Cancellation::default())
        .unwrap()
        .is_empty());
    drop(workspace);
    assert_eq!(budget.used(), 0);
}

#[cfg(feature = "sqlite")]
#[test]
fn sqlite_constrained_selection_matches_oracle_including_unindexed_extremes() {
    let temp = tempfile::tempdir().unwrap();
    let budget = MemoryBudget::new(1 << 24);
    let cancel = Cancellation::default();
    let mut data = observations();
    data.extend([
        obs(10000, [1e100, 0.0, 0.0], 1),
        obs(10001, [1e-200, 0.0, 0.0], 0),
    ]);
    let mut store = terrustrial_core::storage::SqliteStore::create(
        temp.path().join("data.sqlite"),
        4096,
        &budget,
    )
    .unwrap();
    store.import(data.as_slice(), &cancel).unwrap();
    for (center, g) in [
        ([0.0; 3], geometry()),
        ([1e100, 0.0, 0.0], Anisotropy::isotropic(1.0).unwrap()),
        ([0.0; 3], Anisotropy::isotropic(1e-200).unwrap()),
        ([0.0; 3], Anisotropy::isotropic(f64::MAX).unwrap()),
    ] {
        let query = SearchNeighborhood::new(
            32,
            g,
            Some(4),
            vec![
                VariableLimit {
                    variable: 0,
                    maximum: 12,
                },
                VariableLimit {
                    variable: 1,
                    maximum: 17,
                },
            ],
        )
        .unwrap();
        let mut workspace = SearchWorkspace::new(&query, &budget).unwrap();
        let expected = ids(data
            .as_slice()
            .search(point(center), &mut workspace, &cancel)
            .unwrap());
        assert_eq!(
            ids(store
                .search(point(center), &mut workspace, &cancel)
                .unwrap()),
            expected
        );
    }
    store.integrity_check().unwrap();
    drop(store);
    assert_eq!(budget.used(), 0);
}

#[test]
fn index_enclosure_covers_the_accepted_frame_tolerance_for_thin_ellipsoids() {
    let query = SearchNeighborhood::new(
        4,
        Anisotropy::new(
            [[1.0, 0.0, 0.0], [5e-11, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [100.0, 1e-12, 1.0],
        )
        .unwrap(),
        None,
        vec![],
    )
    .unwrap();
    // The accepted frame is not exactly orthogonal. F^T alone would give a
    // y extent of 1e-12 and incorrectly prune this valid interior observation.
    let data = vec![obs(0, [50.0, -2.5e-9, 0.0], 0)];
    let index = SpatialIndex::new(data.clone()).unwrap();
    let budget = MemoryBudget::new(1 << 20);
    let cancel = Cancellation::default();
    let mut workspace = SearchWorkspace::new(&query, &budget).unwrap();
    assert_eq!(
        ids(data
            .as_slice()
            .search(point([0.0; 3]), &mut workspace, &cancel)
            .unwrap()),
        [0]
    );
    assert_eq!(
        ids(index
            .search(point([0.0; 3]), &mut workspace, &cancel)
            .unwrap()),
        [0]
    );
    #[cfg(feature = "sqlite")]
    {
        let temp = tempfile::tempdir().unwrap();
        let mut store = terrustrial_core::storage::SqliteStore::create(
            temp.path().join("thin.sqlite"),
            4096,
            &budget,
        )
        .unwrap();
        store.import(data.as_slice(), &cancel).unwrap();
        assert_eq!(
            ids(store
                .search(point([0.0; 3]), &mut workspace, &cancel)
                .unwrap()),
            [0]
        );
    }
    drop(workspace);
    assert_eq!(budget.used(), 0);
}

#[test]
fn dense_probe_matches_exhaustive_scan_across_density_changes_and_missing_classes() {
    let budget = MemoryBudget::new(4 << 20);
    let cancel = Cancellation::default();
    let dense: Vec<_> = (0..8000u64)
        .flat_map(|i| {
            (0..4).map(move |v| {
                obs(
                    i * 4 + v,
                    [
                        (i % 20) as f64 * 0.02,
                        ((i / 20) % 20) as f64 * 0.02,
                        (i / 400) as f64 * 0.02,
                    ],
                    v as usize,
                )
            })
        })
        .collect();
    let sparse = vec![
        obs(1, [2., 0., 0.], 0),
        obs(2, [-2., 0., 0.], 1),
        obs(3, [0., 2., 0.], 2),
        obs(4, [0., 0., 2.], 3),
    ];
    for octants in [None, Some(4)] {
        let query = SearchNeighborhood::new(
            32,
            geometry(),
            octants,
            (0..4)
                .map(|variable| VariableLimit {
                    variable,
                    maximum: 8,
                })
                .collect(),
        )
        .unwrap();
        let mut indexed = SearchWorkspace::new(&query, &budget).unwrap();
        let mut exhaustive = SearchWorkspace::new(&query, &budget).unwrap();
        for rows in [&dense, &sparse, &dense] {
            let index = SpatialIndex::new(rows.clone()).unwrap();
            for p in [
                [0.21, 0.19, 0.20],
                [2., 1., 0.],
                [0., 0., 0.],
                [100., 100., 100.],
                [0.2, 0.2, 0.2],
            ] {
                let got = ids(index.search(point(p), &mut indexed, &cancel).unwrap());
                let expected = ids(rows
                    .as_slice()
                    .search(point(p), &mut exhaustive, &cancel)
                    .unwrap());
                assert_eq!(got, expected, "point={p:?}, octants={octants:?}");
            }
        }
    }
}

#[test]
fn dense_constrained_queries_prune_work_without_changing_selected_ids() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    struct Counted {
        index: SpatialIndex,
        visited: AtomicUsize,
    }
    impl ObservationSource for Counted {
        fn scan(
            &self,
            v: &mut dyn FnMut(&Observation) -> terrustrial_core::Result<()>,
        ) -> terrustrial_core::Result<()> {
            self.index.scan(v)
        }
        fn visit_bounds(
            &self,
            lo: [f64; 3],
            hi: [f64; 3],
            c: &Cancellation,
            v: &mut dyn FnMut(&Observation) -> terrustrial_core::Result<()>,
        ) -> terrustrial_core::Result<()> {
            self.index.visit_bounds(lo, hi, c, &mut |o| {
                self.visited.fetch_add(1, Ordering::Relaxed);
                v(o)
            })
        }
    }
    let rows: Vec<_> = (0..20000)
        .map(|i| {
            obs(
                i,
                [(i % 100) as f64, (i / 100) as f64, 0.],
                (i % 4) as usize,
            )
        })
        .collect();
    let source = Counted {
        index: SpatialIndex::new(rows.clone()).unwrap(),
        visited: AtomicUsize::new(0),
    };
    let query = SearchNeighborhood::new(
        16,
        Anisotropy::isotropic(1000.).unwrap(),
        None,
        (0..4)
            .map(|variable| VariableLimit {
                variable,
                maximum: 4,
            })
            .collect(),
    )
    .unwrap();
    let budget = MemoryBudget::new(1 << 20);
    let cancel = Cancellation::default();
    let mut a = SearchWorkspace::new(&query, &budget).unwrap();
    let mut b = SearchWorkspace::new(&query, &budget).unwrap();
    source
        .search(point([50., 100., 0.]), &mut a, &cancel)
        .unwrap();
    source.visited.store(0, Ordering::Relaxed);
    assert_eq!(
        ids(source
            .search(point([51., 100., 0.]), &mut a, &cancel)
            .unwrap()),
        ids(rows
            .as_slice()
            .search(point([51., 100., 0.]), &mut b, &cancel)
            .unwrap())
    );
    assert!(source.visited.load(Ordering::Relaxed) < rows.len() / 20);
}
