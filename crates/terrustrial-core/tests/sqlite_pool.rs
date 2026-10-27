#![cfg(feature = "sqlite")]
use rayon::prelude::*;
use terrustrial_core::{
    data::{Group, Observation, Point3, Site, Target},
    estimation::EstimationOptions,
    execution::{Cancellation, ExecutionOptions, MemoryBudget},
    gaussian::{NumericalTolerance, Ordinary, Uncertainty},
    model::{Anisotropy, Coregionalization, Correlation, Lmc, Structure},
    simulation::{parallel, SimulationOptions},
    spatial::{
        NearestWorkspace, Neighborhood, ObservationSource, SearchNeighborhood, SearchWorkspace,
        SpatialIndex,
    },
    storage::{SqliteReadPool, SqliteStore},
};
fn site(x: f64, v: usize) -> Site {
    Site::point(Point3::new([x, 0., 0.]).unwrap(), v)
}
fn fixture() -> Vec<Observation> {
    (0..96)
        .map(|i| {
            Observation::new(
                u64::MAX - i,
                site((i / 2) as f64 * 0.2, (i % 2) as usize),
                1. + (i as f64).sin(),
                0.01,
            )
            .unwrap()
        })
        .collect()
}
#[test]
fn concurrent_queries_match_scans_including_unindexed_extremes_and_exact_values() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("points.sqlite");
    let budget = MemoryBudget::new(1 << 24);
    let cancel = Cancellation::default();
    let mut data = fixture();
    for (id, x) in [(1, 1e200), (2, 1e-200), (3, 1.0_f64.next_up())] {
        data.push(Observation::new(id, site(x, 0), id as f64, 0.).unwrap());
    }
    let mut store = SqliteStore::create(&path, 65536, &budget).unwrap();
    store.import(data.as_slice(), &cancel).unwrap();
    let pool = SqliteReadPool::open(&path, 2, 65536, &budget).unwrap();
    let query = Neighborhood {
        maximum: 16,
        radius: 1.1,
    };
    let constrained =
        SearchNeighborhood::new(16, Anisotropy::isotropic(1.1).unwrap(), Some(5), vec![]).unwrap();
    rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap()
        .install(|| {
            (0..100).into_par_iter().for_each(|i| {
                let p = site(
                    if i % 3 == 0 {
                        1e200
                    } else if i % 3 == 1 {
                        1e-200
                    } else {
                        i as f64 * 0.07
                    },
                    0,
                )
                .support
                .center();
                let expected = data.as_slice().nearest(p, query, &cancel).unwrap();
                let mut workspace = NearestWorkspace::new(query, &budget).unwrap();
                let found = pool.nearest_into(p, &mut workspace, &cancel).unwrap();
                assert_eq!(
                    found.iter().map(Observation::id).collect::<Vec<_>>(),
                    expected.iter().map(Observation::id).collect::<Vec<_>>()
                );
                let mut a = SearchWorkspace::new(&constrained, &budget).unwrap();
                let mut b = SearchWorkspace::new(&constrained, &budget).unwrap();
                assert_eq!(
                    pool.search(p, &mut a, &cancel)
                        .unwrap()
                        .iter()
                        .map(Observation::id)
                        .collect::<Vec<_>>(),
                    data.as_slice()
                        .search(p, &mut b, &cancel)
                        .unwrap()
                        .iter()
                        .map(Observation::id)
                        .collect::<Vec<_>>()
                );
                for o in pool.at_site(&Site::point(p, 0)).unwrap() {
                    let original = data.iter().find(|v| v.id() == o.id()).unwrap();
                    assert_eq!(original.value().to_bits(), o.value().to_bits());
                }
            });
        });
    let mut seen = Vec::new();
    ObservationSource::visit_bounds(&store, [0.; 3], [0.; 3], &cancel, &mut |o| {
        seen.push(o.id());
        Ok(())
    })
    .unwrap();
    // Indexed rows are pruned, while extreme rows remain conservative candidates.
    assert_eq!(seen.len(), 4);
    drop(pool);
    drop(store);
    assert_eq!(budget.used(), 0);
}
#[test]
fn pooled_disk_conditioning_matches_memory_for_parallel_joint_estimation_and_simulation() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("points.sqlite");
    let data = fixture();
    let memory = SpatialIndex::new(data.clone()).unwrap();
    let budget = MemoryBudget::new(1 << 27);
    let cancel = Cancellation::default();
    let mut store = SqliteStore::create(&path, 65536, &budget).unwrap();
    store.import(data.as_slice(), &cancel).unwrap();
    drop(store);
    let model = Lmc::new(vec![Structure {
        correlation: Correlation::Exponential,
        anisotropy: Anisotropy::isotropic(3.).unwrap(),
        coregionalization: Coregionalization::from_factor(2, 2, &[1., 0., 0.4, 0.9]).unwrap(),
    }])
    .unwrap();
    let groups: Vec<_> = (0..31)
        .map(|i| Group {
            id: i,
            targets: (0..2)
                .map(|v| Target {
                    id: 1000 + i * 2 + v,
                    site: site(0.13 + i as f64 * 0.13, v as usize),
                })
                .collect(),
        })
        .collect();
    let query = Neighborhood {
        maximum: 24,
        radius: 3.,
    };
    let execution = ExecutionOptions {
        batch_groups: 8,
        maximum_targets_per_group: 2,
        bytes_per_group: 1 << 20,
    };
    let trend = |s: &Site| Ok(s.support.center().coordinates()[0] * 0.2 + s.variable as f64);
    let mut reference = None;
    for readers in [1, 4] {
        let pool = SqliteReadPool::open(&path, readers, 65536, &budget).unwrap();
        for workers in [1, 4] {
            let threads = rayon::ThreadPoolBuilder::new()
                .num_threads(workers)
                .build()
                .unwrap();
            for hard in [
                &memory as &dyn ObservationSourceSync,
                &pool as &dyn ObservationSourceSync,
            ] {
                let bits = threads.install(|| {
                    let mut bits = Vec::new();
                    terrustrial_core::estimation::estimate_groups(
                        &model,
                        hard,
                        groups.clone(),
                        &Ordinary { variables: 2 },
                        EstimationOptions {
                            neighborhood: query.into(),
                            mean: Some(&trend),
                            uncertainty: Uncertainty::Joint,
                            tolerance: NumericalTolerance::default(),
                            budget: &budget,
                            cancellation: &cancel,
                        },
                        execution,
                        |out| {
                            bits.extend(out.values.iter().map(|v| v.to_bits()));
                            let cov = out.covariance().unwrap();
                            for j in 0..cov.ncols() {
                                for i in 0..cov.nrows() {
                                    bits.push(cov[(i, j)].to_bits());
                                }
                            }
                            Ok(())
                        },
                    )
                    .unwrap();
                    let mut output = SpatialIndex::new(vec![]).unwrap();
                    parallel::simulate_groups(
                        &model,
                        hard,
                        &mut output,
                        groups.clone(),
                        &Ordinary { variables: 2 },
                        SimulationOptions {
                            seed: 52,
                            realization: 3,
                            mean: Some(&trend),
                            hard_neighborhood: query.into(),
                            simulated_neighborhood: query.into(),
                            budget: &budget,
                            cancellation: &cancel,
                        },
                        execution,
                        |out| {
                            bits.extend(out.iter().map(|o| o.value().to_bits()));
                            Ok(())
                        },
                    )
                    .unwrap();
                    bits
                });
                if let Some(expected) = &reference {
                    assert_eq!(&bits, expected);
                } else {
                    reference = Some(bits);
                }
            }
        }
    }
    assert_eq!(budget.used(), 0);
}
trait ObservationSourceSync: ObservationSource + Sync {}
impl<T: ObservationSource + Sync> ObservationSourceSync for T {}
