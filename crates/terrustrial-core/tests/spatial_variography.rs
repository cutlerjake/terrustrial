use terrustrial_core::storage::PointFile;
use terrustrial_core::{
    anisotropy::{NonstationaryGaussian, Tensor},
    data::{Group, Observation, Point3, Site, Target},
    execution::{Cancellation, MemoryBudget},
    gaussian::Simple,
    model::{Anisotropy, Coregionalization, Correlation, Covariance, Lmc, Structure},
    simulation::{simulate_groups, SimulationOptions},
    spatial::{Neighborhood, ObservationSource, SpatialIndex},
    variography::{self, Bin, Sample},
};
fn point(x: f64, y: f64, z: f64) -> Point3 {
    Point3::new([x, y, z]).unwrap()
}
fn obs(id: u64, x: f64) -> Observation {
    Observation::new(id, Site::point(point(x, 0.0, 0.0), 0), x, 0.0).unwrap()
}

#[test]
fn spatial_ties_and_disk_scan_agree_independent_of_input_order() {
    let input = vec![obs(4, 1.0), obs(3, -1.0), obs(2, 1.0), obs(1, -1.0)];
    let cancel = Cancellation::default();
    let query = Neighborhood {
        maximum: 2,
        radius: 2.0,
    };
    let index = SpatialIndex::new(input.clone()).unwrap();
    let ids = |values: Vec<Observation>| values.iter().map(Observation::id).collect::<Vec<_>>();
    assert_eq!(
        ids(index.nearest(point(0.0, 0.0, 0.0), query, &cancel).unwrap()),
        vec![1, 2]
    );
    let path = std::env::temp_dir().join(format!(
        "terrustrial-points-{}-{}.bin",
        std::process::id(),
        rand::random::<u64>()
    ));
    let file = PointFile::create(&path, input.as_slice(), &cancel).unwrap();
    assert_eq!(
        ids(file.nearest(point(0.0, 0.0, 0.0), query, &cancel).unwrap()),
        vec![1, 2]
    );
    assert!(PointFile::open(&path).is_ok());
    std::fs::OpenOptions::new()
        .write(true)
        .open(&path)
        .unwrap()
        .set_len(10)
        .unwrap();
    assert!(PointFile::open(&path).is_err());
    std::fs::remove_file(path).unwrap();
}

#[test]
fn pair_counts_cross_variograms_and_empty_bins_are_explicit() {
    let data: Vec<_> = (0..3)
        .map(|i| Sample {
            point: point(i as f64, 0.0, 0.0),
            values: vec![Some(i as f64), Some(2.0 * i as f64)],
        })
        .collect();
    let bins = variography::experimental(
        &data,
        0,
        1,
        &[1.0, 2.0, 3.0],
        None,
        &Cancellation::default(),
    )
    .unwrap();
    assert_eq!(bins[0].pairs, 2);
    assert_eq!(bins[0].semivariance, Some(1.0));
    assert_eq!(bins[1].pairs, 1);
    assert_eq!(bins[1].semivariance, Some(4.0));
    assert_eq!(bins[2].pairs, 0);
    assert_eq!(bins[2].semivariance, None);
}

#[test]
fn automatic_fit_recovers_known_nonnegative_parameters() {
    let bins: Vec<_> = (1..=24)
        .map(|i| {
            let h = i as f64 * 0.5;
            Bin {
                pairs: 100,
                mean_distance: Some(h),
                semivariance: Some(0.2 + 1.4 * (1.0 - (-h / 3.0).exp())),
            }
        })
        .collect();
    let fit = variography::fit_single(
        &bins,
        &[Correlation::Exponential],
        [0.2, 20.0],
        &Cancellation::default(),
    )
    .unwrap();
    assert!((fit.scale - 3.0).abs() < 1e-5);
    assert!((fit.nugget - 0.2).abs() < 1e-6);
    assert!((fit.structured_sill - 1.4).abs() < 1e-6);
}

#[test]
fn nonstationary_covariance_is_symmetric_and_psd_on_varying_tensors() {
    let model = NonstationaryGaussian {
        field: |p: Point3| {
            Tensor::new([
                [1.0 + p.coordinates()[0].powi(2), 0.2, 0.0],
                [0.2, 2.0, 0.0],
                [0.0, 0.0, 0.5],
            ])
        },
        coregionalization: Coregionalization::scalar(1.0).unwrap(),
    };
    let points: Vec<_> = (0..15)
        .map(|i| point(i as f64 * 0.2, 0.1 * (i % 3) as f64, 0.0))
        .collect();
    let matrix = faer::Mat::from_fn(points.len(), points.len(), |i, j| {
        model.point_covariance(points[i], 0, points[j], 0).unwrap()
    });
    for i in 0..points.len() {
        assert!((matrix[(i, i)] - 1.0).abs() < 1e-12);
        for j in 0..points.len() {
            assert!((matrix[(i, j)] - matrix[(j, i)]).abs() < 1e-12);
        }
    }
    assert!(matrix
        .self_adjoint_eigenvalues(faer::Side::Lower)
        .unwrap()
        .iter()
        .all(|x| *x >= -1e-10));
}

#[test]
fn joint_sequential_path_repeats_across_rayon_pool_sizes_and_honors_data() {
    let model = Lmc::new(vec![Structure {
        correlation: Correlation::Exponential,
        anisotropy: Anisotropy::isotropic(4.0).unwrap(),
        coregionalization: Coregionalization::from_factor(2, 2, &[1.0, 0.0, 0.5, 0.5_f64.sqrt()])
            .unwrap(),
    }])
    .unwrap();
    let hard = [Observation::new(0, Site::point(point(0.0, 0.0, 0.0), 0), 1.25, 0.0).unwrap()];
    use terrustrial_core::spatial::{NeighborhoodQuery, SearchNeighborhood, VariableLimit};
    let query = SearchNeighborhood::new(
        6,
        Anisotropy::new(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [4.0, 1.0, 1.0],
        )
        .unwrap(),
        Some(3),
        vec![VariableLimit {
            variable: 1,
            maximum: 3,
        }],
    )
    .unwrap();
    let run = |workers, constrained| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .unwrap()
            .install(|| {
                let mut simulated = SpatialIndex::new(vec![]).unwrap();
                let mut values = Vec::new();
                let groups = (0..4).map(|i| Group {
                    id: i,
                    targets: (0..2)
                        .map(|v| Target {
                            id: 10 + i * 2 + v,
                            site: Site::point(point(i as f64, 0.0, 0.0), v as usize),
                        })
                        .collect(),
                });
                simulate_groups(
                    &model,
                    hard.as_slice(),
                    &mut simulated,
                    groups,
                    &Simple,
                    SimulationOptions {
                        seed: 42,
                        realization: 0,
                        mean: None,
                        hard_neighborhood: if constrained {
                            NeighborhoodQuery::Constrained(&query)
                        } else {
                            Neighborhood {
                                maximum: 8,
                                radius: 10.0,
                            }
                            .into()
                        },
                        simulated_neighborhood: if constrained {
                            NeighborhoodQuery::Constrained(&query)
                        } else {
                            Neighborhood {
                                maximum: 8,
                                radius: 10.0,
                            }
                            .into()
                        },
                        budget: &MemoryBudget::new(1 << 22),
                        cancellation: &Cancellation::default(),
                    },
                    |group| {
                        values.extend(group.iter().map(Observation::value));
                        Ok(())
                    },
                )
                .unwrap();
                values
            })
    };
    for constrained in [false, true] {
        let a = run(1, constrained);
        let b = run(4, constrained);
        assert_eq!(a, b);
        assert_eq!(a[0], 1.25);
        assert_eq!(a.len(), 8);
    }
}
