#![cfg(feature = "sqlite")]
use std::sync::atomic::{AtomicU64, Ordering};
use terrustrial_core::{
    data::{Observation, Point3, Site},
    execution::{Cancellation, MemoryBudget},
    spatial::{NearestWorkspace, Neighborhood, ObservationSource},
    storage::{Checkpoint, SqliteStore},
};
static SEQUENCE: AtomicU64 = AtomicU64::new(0);
struct Temp(std::path::PathBuf);
impl Temp {
    fn new() -> Self {
        Self(std::env::temp_dir().join(format!(
            "terrustrial-store-{}-{}.sqlite",
            std::process::id(),
            SEQUENCE.fetch_add(1, Ordering::Relaxed)
        )))
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}
fn obs(id: u64, p: [f64; 3]) -> Observation {
    Observation::new(id, Site::point(Point3::new(p).unwrap(), 0), id as f64, 0.0).unwrap()
}
fn ids(observations: &[Observation]) -> Vec<u64> {
    observations.iter().map(Observation::id).collect()
}

#[test]
fn disk_rtree_matches_oracle_at_float_boundaries_and_unsigned_ids() {
    let path = Temp::new();
    let budget = MemoryBudget::new(1 << 20);
    let cancel = Cancellation::default();
    let mut store = SqliteStore::create(&path.0, 64 * 1024, &budget).unwrap();
    let mut data = vec![
        obs(u64::MAX, [1.0, 0.0, 0.0]),
        obs(1, [-1.0, 0.0, 0.0]),
        obs(2, [1e100, 0.0, 0.0]),
        obs(3, [1e-200, 0.0, 0.0]),
        obs(4, [0.0, 0.0, 0.0]),
        obs(5, [1.0_f64.next_up(), 0.0, 0.0]),
    ];
    for i in 10..2000 {
        data.push(obs(
            i,
            [
                (i % 17) as f64 / 3.0,
                ((i / 17) % 11) as f64 / 7.0,
                (i % 31) as f64 / 13.0,
            ],
        ));
    }
    store.import(data.as_slice(), &cancel).unwrap();
    for (point, radius) in [
        ([0.0; 3], 1.0),
        ([1e100, 0.0, 0.0], 1.0),
        ([1e-200, 0.0, 0.0], 1e-201),
        ([2.333, 0.777, 1.3], 1.23),
    ] {
        for maximum in [1, 3, 32, 3000] {
            let query = Neighborhood { maximum, radius };
            let point = Point3::new(point).unwrap();
            assert_eq!(
                ids(&store.nearest(point, query, &cancel).unwrap()),
                ids(&data.as_slice().nearest(point, query, &cancel).unwrap())
            );
            let mut workspace = NearestWorkspace::new(query, &budget).unwrap();
            assert_eq!(
                ids(store.nearest_into(point, &mut workspace, &cancel).unwrap()),
                ids(&data.as_slice().nearest(point, query, &cancel).unwrap())
            );
        }
    }
    store.integrity_check().unwrap();
    drop(store);
    assert_eq!(budget.used(), 0);
    let store = SqliteStore::open(&path.0, 4096, &budget).unwrap();
    assert_eq!(store.len().unwrap(), data.len() as u64);
    assert!(SqliteStore::create(&path.0, 4096, &budget).is_err());
}

#[test]
fn bounded_disk_nearest_matches_full_sort_across_scales_ties_and_reuse() {
    let cancel = Cancellation::default();
    for scale in [
        1e-200,
        f32::MIN_POSITIVE as f64,
        1e-20,
        1.,
        1e20,
        f32::MAX as f64 / 8.,
        1e200,
    ] {
        let path = Temp::new();
        let budget = MemoryBudget::new(8 << 20);
        let mut data: Vec<_> = (0..729)
            .map(|i| {
                obs(
                    u64::MAX - i,
                    [
                        ((i % 9) as f64 - 4.) * scale,
                        (((i / 9) % 9) as f64 - 4.) * scale,
                        ((i / 81) as f64 - 4.) * scale,
                    ],
                )
            })
            .collect();
        // Equal-distance candidates have both signed-SQL and unsigned-Rust IDs.
        // Neighboring binary64 coordinates share outward-rounded index boxes.
        for (id, x) in [
            (1, scale),
            (2, -scale),
            (3, scale.next_up()),
            (4, scale.next_down()),
            (5, -0.),
        ] {
            data.push(obs(id, [x, 0., 0.]));
        }
        let mut store = SqliteStore::create(&path.0, 65536, &budget).unwrap();
        store.import(data.as_slice(), &cancel).unwrap();
        for maximum in [1, 7, 32, 127, 1024] {
            for radius in [scale.next_down(), scale, scale.next_up(), 3.25 * scale] {
                let query = Neighborhood { maximum, radius };
                let mut workspace = NearestWorkspace::new(query, &budget).unwrap();
                for center in [[0.; 3], [0.125, -0.25, 0.5], [3.9; 3], [20.; 3], [0.; 3]]
                    .into_iter()
                    .cycle()
                    .take(25)
                {
                    let point = Point3::new(center.map(|v| v * scale)).unwrap();
                    let mut expected: Vec<_> = data
                        .iter()
                        .filter_map(|o| {
                            let distance = point.distance(o.site().support.center());
                            (distance <= radius).then_some((distance, o.id()))
                        })
                        .collect();
                    expected.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
                    expected.truncate(maximum);
                    let actual = store.nearest_into(point, &mut workspace, &cancel).unwrap();
                    assert_eq!(
                        ids(actual),
                        expected.iter().map(|v| v.1).collect::<Vec<_>>(),
                        "scale={scale}, radius={radius}, maximum={maximum}, center={center:?}"
                    );
                }
                let cancelled = Cancellation::default();
                cancelled.cancel();
                let point = Point3::new([0.; 3]).unwrap();
                assert!(store
                    .nearest_into(point, &mut workspace, &cancelled)
                    .is_err());
                assert_eq!(
                    ids(store.nearest_into(point, &mut workspace, &cancel).unwrap()),
                    ids(&store.nearest(point, query, &cancel).unwrap())
                );
            }
        }
        drop(store);
        assert_eq!(budget.used(), 0);
    }
}

#[test]
fn checkpoints_roll_back_output_and_reject_stale_or_mismatched_writers() {
    let path = Temp::new();
    let budget = MemoryBudget::new(1 << 20);
    let cancel = Cancellation::default();
    let mut store = SqliteStore::create(&path.0, 4096, &budget).unwrap();
    let first = Checkpoint {
        identity: [42; 32],
        completed_group: 0,
    };
    store
        .commit_group(&[obs(1, [0.0; 3])], None, &first, &cancel)
        .unwrap();
    let next = Checkpoint {
        identity: first.identity,
        completed_group: 1,
    };
    assert!(store
        .commit_group(
            &[obs(2, [1.0; 3]), obs(1, [2.0; 3])],
            Some(&first),
            &next,
            &cancel
        )
        .is_err());
    assert_eq!(store.len().unwrap(), 1);
    assert_eq!(store.checkpoint().unwrap(), Some(first.clone()));
    assert!(store
        .commit_group(&[obs(2, [1.0; 3])], None, &next, &cancel)
        .is_err());
    assert!(store
        .commit_group(
            &[],
            Some(&first),
            &Checkpoint {
                identity: [0; 32],
                completed_group: 2
            },
            &cancel
        )
        .is_err());
    drop(store);
    let mut store = SqliteStore::open(&path.0, 4096, &budget).unwrap();
    store
        .commit_group(&[obs(2, [1.0; 3])], Some(&first), &next, &cancel)
        .unwrap();
    assert_eq!(store.len().unwrap(), 2);
    assert_eq!(store.checkpoint().unwrap(), Some(next.clone()));
    cancel.cancel();
    assert!(store
        .commit_group(
            &[],
            Some(&next),
            &Checkpoint {
                identity: next.identity,
                completed_group: 3
            },
            &cancel
        )
        .is_err());
    assert_eq!(store.checkpoint().unwrap(), Some(next));
}

#[test]
fn failed_stream_import_leaves_no_partial_records() {
    let path = Temp::new();
    let budget = MemoryBudget::new(1 << 20);
    let mut store = SqliteStore::create(&path.0, 4096, &budget).unwrap();
    assert!(store
        .import(
            &[obs(0, [0.0; 3]), obs(0, [1.0; 3])][..],
            &Cancellation::default()
        )
        .is_err());
    assert!(store.is_empty().unwrap());
    store.integrity_check().unwrap();
}

#[test]
fn interrupted_joint_simulation_resumes_bit_for_bit_and_preserves_excluded_hard_target() {
    use terrustrial_core::spatial::{NeighborhoodQuery, SearchNeighborhood, VariableLimit};
    use terrustrial_core::{
        data::{Group, Target},
        gaussian::Simple,
        model::{Anisotropy, Coregionalization, Correlation, Lmc, Structure},
        simulation::{simulate_checkpointed, CheckpointedOptions, SimulationOptions},
    };
    fn run(
        mean: Option<&dyn terrustrial_core::gaussian::KnownMean>,
        query: Option<&SearchNeighborhood>,
    ) {
        let path_a = Temp::new();
        let path_b = Temp::new();
        let budget = MemoryBudget::new(1 << 24);
        let model = Lmc::new(vec![Structure {
            correlation: Correlation::Exponential,
            anisotropy: Anisotropy::isotropic(3.0).unwrap(),
            coregionalization: Coregionalization::from_factor(2, 2, &[1.0, 0.0, 0.6, 0.8]).unwrap(),
        }])
        .unwrap();
        let hard = [Observation::new(
            0,
            Site::point(Point3::new([0.0; 3]).unwrap(), 0),
            0.153,
            0.0,
        )
        .unwrap()];
        let groups: Vec<_> = (0..8)
            .map(|i| Group {
                id: i,
                targets: (0..2)
                    .map(|v| Target {
                        id: 100 + 2 * i + v as u64,
                        site: Site::point(Point3::new([i as f64, 0.0, 0.0]).unwrap(), v),
                    })
                    .collect(),
            })
            .collect();
        fn options<'a>(
            budget: &'a MemoryBudget,
            cancel: &'a Cancellation,
            mean: Option<&'a dyn terrustrial_core::gaussian::KnownMean>,
            query: Option<&'a SearchNeighborhood>,
        ) -> SimulationOptions<'a> {
            SimulationOptions {
                seed: 153,
                realization: 4,
                mean,
                hard_neighborhood: query.map_or_else(
                    || {
                        Neighborhood {
                            maximum: 1,
                            radius: 0.1,
                        }
                        .into()
                    },
                    NeighborhoodQuery::Constrained,
                ),
                simulated_neighborhood: query.map_or_else(
                    || {
                        Neighborhood {
                            maximum: 8,
                            radius: 10.0,
                        }
                        .into()
                    },
                    NeighborhoodQuery::Constrained,
                ),
                budget,
                cancellation: cancel,
            }
        }
        let cancel = Cancellation::default();
        let mut uninterrupted = SqliteStore::create(&path_a.0, 4096, &budget).unwrap();
        simulate_checkpointed(
            &model,
            &hard[..],
            &mut uninterrupted,
            groups.clone(),
            &Simple,
            CheckpointedOptions {
                simulation: options(&budget, &cancel, mean, query),
                identity: [7; 32],
            },
            |_| Ok(()),
        )
        .unwrap();
        let mut interrupted = SqliteStore::create(&path_b.0, 4096, &budget).unwrap();
        let stop = Cancellation::default();
        let mut count = 0;
        assert!(simulate_checkpointed(
            &model,
            &hard[..],
            &mut interrupted,
            groups.clone(),
            &Simple,
            CheckpointedOptions {
                simulation: options(&budget, &stop, mean, query),
                identity: [7; 32]
            },
            |_| {
                count += 1;
                if count == 3 {
                    stop.cancel();
                }
                Ok(())
            }
        )
        .is_err());
        assert_eq!(interrupted.len().unwrap(), 6);
        drop(interrupted);
        let mut resumed = SqliteStore::open(&path_b.0, 4096, &budget).unwrap();
        simulate_checkpointed(
            &model,
            &hard[..],
            &mut resumed,
            groups,
            &Simple,
            CheckpointedOptions {
                simulation: options(&budget, &cancel, mean, query),
                identity: [7; 32],
            },
            |_| Ok(()),
        )
        .unwrap();
        fn contents(store: &SqliteStore) -> Vec<(u64, u64)> {
            let mut out = Vec::new();
            store
                .scan(&mut |o| {
                    out.push((o.id(), o.value().to_bits()));
                    Ok(())
                })
                .unwrap();
            out.sort_unstable();
            out
        }
        assert_eq!(contents(&uninterrupted), contents(&resumed));
        assert_eq!(contents(&resumed)[0], (100, 0.153_f64.to_bits()));
        // A group center far from its first target must still honor that exact target.
        let mut memory = terrustrial_core::spatial::SpatialIndex::new(vec![]).unwrap();
        let group = Group {
            id: 0,
            targets: vec![
                Target {
                    id: 1000,
                    site: *hard[0].site(),
                },
                Target {
                    id: 1001,
                    site: Site::point(Point3::new([10.0, 0.0, 0.0]).unwrap(), 0),
                },
            ],
        };
        terrustrial_core::simulation::simulate_groups(
            &model,
            &hard[..],
            &mut memory,
            [group],
            &Simple,
            options(&budget, &cancel, mean, query),
            |values| {
                assert_eq!(values[0].value().to_bits(), 0.153_f64.to_bits());
                Ok(())
            },
        )
        .unwrap();
    }
    let mean = |site: &Site| {
        let x = site.support.center().coordinates()[0];
        Ok(if site.variable == 0 {
            2.0 + 0.25 * x * x
        } else {
            -1.0 + 0.5 * x
        })
    };
    let query = SearchNeighborhood::new(
        4,
        Anisotropy::new(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [3.0, 1.0, 1.0],
        )
        .unwrap(),
        Some(2),
        vec![VariableLimit {
            variable: 1,
            maximum: 3,
        }],
    )
    .unwrap();
    for search in [None, Some(&query)] {
        run(None, search);
        run(Some(&mean), search);
    }
}
