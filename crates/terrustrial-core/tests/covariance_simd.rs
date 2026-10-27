use terrustrial_core::{
    data::{Observation, Point3, Site, Support},
    execution::{Cancellation, MemoryBudget},
    gaussian::{GaussianWorkspace, NumericalTolerance, Ordinary, Uncertainty, WorkspaceOptions},
    model::{Anisotropy, Coregionalization, Correlation, Covariance, Lmc, Structure},
    Error, Result,
};
struct Scalar<'a>(&'a Lmc);
impl Covariance for Scalar<'_> {
    fn variables(&self) -> usize {
        self.0.variables()
    }
    fn point_covariance(&self, a: Point3, av: usize, b: Point3, bv: usize) -> Result<f64> {
        self.0.point_covariance(a, av, b, bv)
    }
}
fn point(p: [f64; 3]) -> Point3 {
    Point3::new(p).unwrap()
}
fn make_model(kernel: Correlation, scale: f64, rotated: bool) -> Lmc {
    let (s, c) = 0.37_f64.sin_cos();
    let anisotropy = if rotated {
        Anisotropy::new(
            [[c, s, 0.], [-s, c, 0.], [0., 0., 1.]],
            [scale, scale * 0.5, scale * 1.5],
        )
        .unwrap()
    } else {
        Anisotropy::isotropic(scale).unwrap()
    };
    Lmc::new(vec![
        Structure {
            correlation: kernel,
            anisotropy,
            coregionalization: Coregionalization::from_factor(
                3,
                3,
                &[1., 0., 0., -0.3, 0.7, 0., 0.1, -0.2, 0.6],
            )
            .unwrap(),
        },
        Structure {
            correlation: Correlation::Nugget,
            anisotropy: Anisotropy::isotropic(1.).unwrap(),
            coregionalization: Coregionalization::from_factor(
                3,
                3,
                &[0.1, 0., 0., 0., 0.1, 0., 0., 0., 0.1],
            )
            .unwrap(),
        },
    ])
    .unwrap()
}
fn rows(scale: f64) -> Vec<Observation> {
    (0..67)
        .map(|i| {
            let coordinates = if i % 11 == 0 {
                [0.; 3]
            } else {
                [
                    (i % 7) as f64 * 0.13 * scale,
                    (i % 11) as f64 * -0.17 * scale,
                    (i % 5) as f64 * 0.19 * scale,
                ]
            };
            Observation::new(i, Site::point(point(coordinates), (i % 3) as usize), 0., 0.).unwrap()
        })
        .collect()
}
fn compare(model: &Lmc, target: &Site, observations: &[Observation]) {
    let mut scalar = vec![0.; observations.len()];
    let mut simd = vec![0.; observations.len()];
    let sites: Vec<_> = observations.iter().map(|o| *o.site()).collect();
    for kind in 0..3 {
        let (a, b) = if kind == 2 {
            (
                Scalar(model).covariance_sites_row(target, &sites, &mut scalar),
                model.covariance_sites_row(target, &sites, &mut simd),
            )
        } else if kind == 1 {
            (
                Scalar(model).covariance_column(observations, target, &mut scalar),
                model.covariance_column(observations, target, &mut simd),
            )
        } else {
            (
                Scalar(model).covariance_row(target, observations, &mut scalar),
                model.covariance_row(target, observations, &mut simd),
            )
        };
        match (a, b) {
            (Ok(()), Ok(())) => assert_eq!(
                scalar.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                simd.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
            ),
            (Err(Error::Numerical { .. }), Err(Error::Numerical { .. })) => (),
            result => panic!("scalar/SIMD disagreement (kind={kind}): {result:?}"),
        }
    }
}
#[test]
fn batched_colocated_rows_and_ragged_triangles_preserve_scalar_bits_without_allocations() {
    for kernel in [
        Correlation::Nugget,
        Correlation::Spherical,
        Correlation::Exponential,
        Correlation::Gaussian,
        Correlation::Matern32,
        Correlation::Matern52,
    ] {
        for scale in [1e-200, 1.0, 1e200] {
            for rotated in [false, true] {
                let model = make_model(kernel, scale, rotated);
                let mut observations = rows(scale);
                // Mixed point/block lanes must retain scalar quadrature order.
                observations[19] = Observation::new(
                    19,
                    Site {
                        support: Support::block(point([0.; 3]), [scale * 0.1; 3], [2, 1, 2])
                            .unwrap(),
                        variable: 1,
                    },
                    0.,
                    0.,
                )
                .unwrap();
                let sites: Vec<_> = (0..19)
                    .map(|i| {
                        Site::point(
                            point([if i < 17 { 0.0 } else { scale * 0.1 }, -0.0, 0.0]),
                            i % 3,
                        )
                    })
                    .collect();
                for ragged in [false, true] {
                    for left in [false, true] {
                        let mut expected: Vec<Vec<f64>> = (0..sites.len())
                            .map(|i| {
                                vec![
                                    0.;
                                    if ragged {
                                        observations.len() - i
                                    } else {
                                        observations.len()
                                    }
                                ]
                            })
                            .collect();
                        let mut actual = expected.clone();
                        let mut expected_rows: Vec<_> =
                            expected.iter_mut().map(Vec::as_mut_slice).collect();
                        let mut actual_rows: Vec<_> =
                            actual.iter_mut().map(Vec::as_mut_slice).collect();
                        let reference = if left {
                            Scalar(&model).covariance_columns(
                                &observations,
                                &sites,
                                &mut expected_rows,
                            )
                        } else {
                            Scalar(&model).covariance_rows(
                                &sites,
                                &observations,
                                &mut expected_rows,
                            )
                        };
                        let mut result = Ok(());
                        let allocations = allocation_counter::measure(|| {
                            result = if left {
                                model.covariance_columns(&observations, &sites, &mut actual_rows)
                            } else {
                                model.covariance_rows(&sites, &observations, &mut actual_rows)
                            };
                        });
                        assert_eq!(allocations.count_total, 0);
                        match (reference, result) {
                            (Ok(()), Ok(())) => {
                                for (a, b) in actual.iter().flatten().zip(expected.iter().flatten())
                                {
                                    assert_eq!(a.to_bits(), b.to_bits());
                                }
                            }
                            (Err(Error::Numerical { .. }), Err(Error::Numerical { .. })) => (),
                            results => panic!("batched/scalar mismatch: {results:?}"),
                        }
                    }
                }
            }
        }
    }
    let model = make_model(Correlation::Exponential, 1., false);
    let data = rows(1.);
    let sites = [*data[0].site(); 3];
    let mut empty = [];
    let mut one = [0.];
    let mut too_long = [0.; 68];
    model
        .covariance_rows(&sites[..2], &data, &mut [&mut empty, &mut one])
        .unwrap();
    assert!(model
        .covariance_columns(&data, &sites, &mut [&mut one])
        .is_err());
    assert!(model
        .covariance_rows(&sites[..1], &data, &mut [&mut too_long])
        .is_err());
    let invalid = [Site::point(point([0.; 3]), usize::MAX)];
    assert!(model
        .covariance_rows(&invalid, &data, &mut [&mut one])
        .is_err());
}

#[test]
fn all_kernels_joint_variables_rotations_extreme_scales_and_tail_lengths_match_scalar_bits() {
    for scale in [1e-200, 1e-100, 1., 1e100, 1e200] {
        let observations = rows(scale);
        for kernel in [
            Correlation::Nugget,
            Correlation::Spherical,
            Correlation::Exponential,
            Correlation::Gaussian,
            Correlation::Matern32,
            Correlation::Matern52,
        ] {
            for rotated in [false, true] {
                let model = make_model(kernel, scale, rotated);
                for variable in 0..3 {
                    for p in [[0.; 3], [0.37 * scale, -0.12 * scale, 0.17 * scale]] {
                        let target = Site::point(point(p), variable);
                        for n in (0..=17).chain([31, 32, 33, 63, 64, 65, 67]) {
                            compare(&model, &target, &observations[..n]);
                        }
                    }
                }
            }
        }
    }
}
#[test]
fn colocated_lanes_and_adjacent_distinct_distances_match_scalar_bits() {
    for kernel in [
        Correlation::Exponential,
        Correlation::Gaussian,
        Correlation::Matern32,
        Correlation::Matern52,
    ] {
        for rotated in [false, true] {
            let model = make_model(kernel, 1., rotated);
            let observations: Vec<_> = (0..67)
                .map(|i| {
                    let x = match (i / 7) % 3 {
                        0 => 1.,
                        1 => 1.0_f64.next_up(),
                        _ => 0.,
                    };
                    Observation::new(i as u64, Site::point(point([x, 0., 0.]), i % 3), 0., 0.)
                        .unwrap()
                })
                .collect();
            for variable in 0..3 {
                for n in 1..=observations.len() {
                    compare(
                        &model,
                        &Site::point(point([0.; 3]), variable),
                        &observations[..n],
                    );
                }
            }
        }
    }
}
#[test]
fn block_quadrature_and_overflow_underflow_boundaries_keep_scalar_semantics() {
    let mut observations = rows(1.);
    observations[3] = Observation::new(
        3,
        Site {
            support: Support::block(point([0.3, 0., 0.]), [0.1, 0.2, 0.3], [2, 3, 4]).unwrap(),
            variable: 1,
        },
        0.,
        0.,
    )
    .unwrap();
    observations[5] = Observation::new(
        5,
        Site {
            support: Support::block(point([0.4, 0., 0.]), [0.1, 0.2, 0.3], [1, 1, 1]).unwrap(),
            variable: 2,
        },
        0.,
        0.,
    )
    .unwrap();
    for kernel in [
        Correlation::Nugget,
        Correlation::Spherical,
        Correlation::Exponential,
        Correlation::Gaussian,
        Correlation::Matern32,
        Correlation::Matern52,
    ] {
        let model = make_model(kernel, 1., true);
        compare(&model, &Site::point(point([0.; 3]), 0), &observations);
        let target = Site {
            support: Support::block(point([0.; 3]), [0.1, 0.1, 0.1], [2, 3, 2]).unwrap(),
            variable: 1,
        };
        compare(&model, &target, &observations);
        let extremes = [
            [f64::MAX, 0., 0.],
            [-f64::MAX, 0., 0.],
            [f64::MAX, f64::MAX, 0.],
            [f64::MIN_POSITIVE, 0., 0.],
            [f64::from_bits(1), 0., 0.],
            [0.; 3],
            [-0., 0., 0.],
            [1., 0., 0.],
            [1.0_f64.next_down(), 0., 0.],
            [1.0_f64.next_up(), 0., 0.],
        ];
        let observations: Vec<_> = extremes
            .into_iter()
            .enumerate()
            .map(|(i, p)| Observation::new(i as u64, Site::point(point(p), i % 3), 0., 0.).unwrap())
            .collect();
        for scale in [1e-200, 1., 1e200] {
            for rotated in [false, true] {
                let model = make_model(kernel, scale, rotated);
                for target in [
                    Site::point(point([0.; 3]), 0),
                    Site::point(point([f64::MAX, 0., 0.]), 1),
                ] {
                    compare(&model, &target, &observations);
                }
            }
        }
    }
}
#[test]
fn complete_joint_plans_match_scalar_reference_and_simd_lines_allocate_nothing() {
    let model = make_model(Correlation::Exponential, 3., true);
    let observations: Vec<_> = (0..32)
        .map(|i| {
            Observation::new(
                i,
                Site::point(point([(i / 3) as f64 * 0.37, 0., 0.]), (i % 3) as usize),
                0.,
                0.01,
            )
            .unwrap()
        })
        .collect();
    let targets: Vec<_> = (0..12)
        .map(|i| Site::point(point([0.11 + (i / 3) as f64 * 0.21, 0., 0.]), i % 3))
        .collect();
    let budget = MemoryBudget::new(1 << 22);
    let cancel = Cancellation::default();
    let mut scalar = GaussianWorkspace::new(32, targets.len(), 3, &budget).unwrap();
    let mut simd = GaussianWorkspace::new(32, targets.len(), 3, &budget).unwrap();
    let options = || WorkspaceOptions {
        uncertainty: Uncertainty::Joint,
        tolerance: NumericalTolerance::default(),
        cancellation: &cancel,
    };
    let a = scalar
        .prepare(
            &Scalar(&model),
            &observations,
            &targets,
            &Ordinary { variables: 3 },
            options(),
        )
        .unwrap();
    let b = simd
        .prepare(
            &model,
            &observations,
            &targets,
            &Ordinary { variables: 3 },
            options(),
        )
        .unwrap();
    for j in 0..targets.len() {
        for i in 0..32 {
            assert_eq!(a.weights()[(i, j)].to_bits(), b.weights()[(i, j)].to_bits());
        }
    }
    for i in 0..targets.len() {
        for j in 0..targets.len() {
            assert_eq!(
                a.covariance().unwrap()[(i, j)].to_bits(),
                b.covariance().unwrap()[(i, j)].to_bits()
            );
        }
    }
    let mut out = [0.; 32];
    let sites: Vec<_> = observations.iter().map(|o| *o.site()).collect();
    let allocations = allocation_counter::measure(|| {
        for _ in 0..1000 {
            model
                .covariance_row(&targets[0], &observations, &mut out)
                .unwrap();
            std::hint::black_box(&out);
            model
                .covariance_column(&observations, &targets[0], &mut out)
                .unwrap();
            std::hint::black_box(&out);
            model
                .covariance_sites_row(&targets[0], &sites, &mut out)
                .unwrap();
            std::hint::black_box(&out);
        }
    });
    assert_eq!(allocations.count_total, 0);
    assert!(model
        .covariance_sites_row(&targets[0], &sites, &mut out[..31])
        .is_err());
    assert!(model
        .covariance_row(&targets[0], &observations, &mut out[..31])
        .is_err());
    assert!(model
        .covariance_column(&observations, &targets[0], &mut out[..31])
        .is_err());
    let invalid = Observation::new(99, Site::point(point([0.; 3]), usize::MAX), 0., 0.).unwrap();
    assert!(model
        .covariance_sites_row(&targets[0], &[*invalid.site()], &mut out[..1])
        .is_err());
    assert!(model
        .covariance_row(&targets[0], std::slice::from_ref(&invalid), &mut out[..1])
        .is_err());
    assert!(model
        .covariance_column(&[invalid], &targets[0], &mut out[..1])
        .is_err());
    assert!(model
        .covariance_row(&Site::point(point([0.; 3]), usize::MAX), &[], &mut [])
        .is_err());
    assert!(model
        .covariance_column(&[], &Site::point(point([0.; 3]), usize::MAX), &mut [])
        .is_err());
}

#[test]
fn cross_covariance_preserves_argument_order_for_custom_models() {
    // An intentionally directional diagnostic function detects accidental
    // transposition. It is not a physically valid standalone covariance model.
    struct Directed;
    impl Covariance for Directed {
        fn variables(&self) -> usize {
            1
        }
        fn point_covariance(&self, a: Point3, _: usize, b: Point3, _: usize) -> Result<f64> {
            Ok(10. * a.coordinates()[0] + b.coordinates()[0])
        }
    }
    let observations: Vec<_> = (0..2)
        .map(|i| Observation::new(i, Site::point(point([i as f64, 0., 0.]), 0), 0., 0.).unwrap())
        .collect();
    let target = Site::point(point([2., 0., 0.]), 0);
    let mut row = [0.; 2];
    let mut column = [0.; 2];
    Directed
        .covariance_row(&target, &observations, &mut row)
        .unwrap();
    Directed
        .covariance_column(&observations, &target, &mut column)
        .unwrap();
    assert_eq!(row, [20., 21.]);
    assert_eq!(column, [2., 12.]);
    assert!(Directed
        .covariance_column(&observations, &target, &mut [])
        .is_err());
    let data_model = Lmc::new(vec![Structure {
        correlation: Correlation::Nugget,
        anisotropy: Anisotropy::isotropic(1.).unwrap(),
        coregionalization: Coregionalization::scalar(1.).unwrap(),
    }])
    .unwrap();
    let budget = MemoryBudget::new(1 << 16);
    let cancel = Cancellation::default();
    let mut workspace = GaussianWorkspace::new(2, 1, 0, &budget).unwrap();
    let plan = workspace
        .prepare_component(
            &data_model,
            &Directed,
            &observations,
            &[target],
            &terrustrial_core::gaussian::Simple,
            WorkspaceOptions {
                uncertainty: Uncertainty::None,
                tolerance: NumericalTolerance::default(),
                cancellation: &cancel,
            },
        )
        .unwrap();
    assert_eq!(plan.weights()[(0, 0)], 2.);
    assert_eq!(plan.weights()[(1, 0)], 12.);
}
