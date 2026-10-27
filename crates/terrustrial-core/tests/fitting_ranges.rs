use terrustrial_core::{
    execution::{Cancellation, MemoryBudget},
    fitting::{
        ranges::{self, Bound, Kernel, Parameter, Scale, Stop},
        CrossVariogramBin, LmcFitOptions,
    },
    model::{Anisotropy, Correlation, SpatialKernel},
    Error,
};
fn options<'a>(budget: &'a MemoryBudget, cancel: &'a Cancellation) -> ranges::Options<'a> {
    ranges::Options {
        maximum_evaluations: 1000,
        initial_log_step: 0.75,
        log_tolerance: 1e-5,
        improvement_tolerance: 1e-14,
        diagnostic_log_step: 0.01,
        sensitivity_relative_tolerance: 1e-4,
        sill: LmcFitOptions {
            maximum_iterations: 10000,
            tolerance: 1e-11,
            budget,
            cancellation: cancel,
        },
    }
}
fn bins(
    scales: [f64; 3],
    angle: f64,
    nugget: bool,
    all_directions: bool,
    magnitude: f64,
) -> Vec<CrossVariogramBin> {
    let (s, c) = angle.sin_cos();
    let sill = [[1.0, -0.6], [-0.6, 2.0]];
    let noise = [[0.2, 0.1], [0.1, 0.3]];
    let directions = if all_directions {
        vec![
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ]
    } else {
        vec![[1.0, 0.0, 0.0]]
    };
    let mut bins = Vec::new();
    for a in 0..2 {
        for b in a..2 {
            for direction in &directions {
                for i in 1..=18 {
                    let lag = direction.map(|x| x * (i as f64).powi(2) / 36.0);
                    let [x, y, z] = lag;
                    let h = (((c * x + s * y) / scales[0]).powi(2)
                        + ((-s * x + c * y) / scales[1]).powi(2)
                        + (z / scales[2]).powi(2))
                    .sqrt();
                    bins.push(CrossVariogramBin {
                        variables: [a, b],
                        lag,
                        semivariance: magnitude
                            * (sill[a][b] * (1.0 - (-h).exp())
                                + if nugget { noise[a][b] } else { 0.0 }),
                        pairs: 100 + (i as u64) * 7,
                    });
                }
            }
        }
    }
    bins
}
#[test]
fn shared_isotropic_range_and_negative_cross_sills_are_recovered_across_value_scales() {
    let budget = MemoryBudget::new(1 << 24);
    let cancel = Cancellation::default();
    for magnitude in [1.0, 1e-200, 1e150] {
        let fit = ranges::fit(
            2,
            &bins([2.3; 3], 0.0, false, false, magnitude),
            &[Kernel::isotropic(Correlation::Exponential, 0)],
            &[Parameter {
                initial: 0.6,
                bounds: [0.1, 10.0],
            }],
            options(&budget, &cancel),
        )
        .unwrap();
        assert_eq!(fit.stop, Stop::LogStepTolerance);
        assert!(
            (fit.parameters[0] / 2.3 - 1.0).abs() < 5e-5,
            "{:?}",
            fit.parameters
        );
        assert!(fit.normalized_weighted_sse < 1e-9);
        assert_eq!(fit.sensitivity_rank, 1);
        assert_eq!(fit.bounds, [Bound::Interior]);
        assert!(fit.objective_history.windows(2).all(|x| x[1] < x[0]));
        assert!(fit.evaluations <= 1000);
        let sill = &fit.fit.model.structures()[0].coregionalization;
        assert!((sill.covariance(0, 1).unwrap() / magnitude + 0.6).abs() < 2e-5);
        assert!(sill.covariance(0, 1).unwrap() / magnitude < 0.0);
        assert_eq!(budget.used(), 0);
    }
}
#[test]
fn anisotropic_axes_recover_and_missing_directions_lose_sensitivity_rank() {
    let budget = MemoryBudget::new(1 << 24);
    let cancel = Cancellation::default();
    let (s, c) = 0.37_f64.sin_cos();
    let template = Kernel {
        correlation: Correlation::Exponential,
        frame: [[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]],
        scales: std::array::from_fn(|index| Scale::Parameter {
            index,
            multiplier: 1.0,
        }),
    };
    let parameters = [
        Parameter {
            initial: 1.0,
            bounds: [0.2, 8.0],
        },
        Parameter {
            initial: 3.0,
            bounds: [0.5, 12.0],
        },
        Parameter {
            initial: 2.0,
            bounds: [0.2, 5.0],
        },
    ];
    let fit = ranges::fit(
        2,
        &bins([2.0, 5.0, 1.3], 0.37, false, true, 1.0),
        &[template],
        &parameters,
        options(&budget, &cancel),
    )
    .unwrap();
    assert_eq!(fit.stop, Stop::LogStepTolerance);
    for (&actual, expected) in fit.parameters.iter().zip([2.0, 5.0, 1.3]) {
        assert!(
            (actual / expected - 1.0).abs() < 2e-4,
            "{:?}",
            fit.parameters
        );
    }
    assert_eq!(fit.sensitivity_rank, 3);
    assert!(fit.normalized_weighted_sse < 1e-8);
    let aligned = Kernel {
        correlation: Correlation::Exponential,
        frame: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        scales: std::array::from_fn(|index| Scale::Parameter {
            index,
            multiplier: 1.0,
        }),
    };
    let fit = ranges::fit(
        2,
        &bins([2.0, 5.0, 1.3], 0.0, false, false, 1.0),
        &[aligned],
        &parameters,
        options(&budget, &cancel),
    )
    .unwrap();
    assert_eq!(fit.sensitivity_rank, 1);
    assert_eq!(&fit.parameter_sensitivity[1..], &[0.0, 0.0]);
    assert!((fit.parameters[1] - 3.0).abs() < 1e-14);
    assert_eq!(budget.used(), 0);
}
#[test]
fn nested_nugget_and_range_fit_remain_jointly_psd() {
    let budget = MemoryBudget::new(1 << 24);
    let cancel = Cancellation::default();
    let templates = [
        Kernel::fixed(SpatialKernel {
            correlation: Correlation::Nugget,
            anisotropy: Anisotropy::isotropic(1.0).unwrap(),
        }),
        Kernel::isotropic(Correlation::Exponential, 0),
    ];
    let fit = ranges::fit(
        2,
        &bins([2.3; 3], 0.0, true, false, 1.0),
        &templates,
        &[Parameter {
            initial: 1.0,
            bounds: [0.3, 8.0],
        }],
        options(&budget, &cancel),
    )
    .unwrap();
    assert_eq!(fit.stop, Stop::LogStepTolerance);
    assert!((fit.parameters[0] / 2.3 - 1.0).abs() < 1e-4);
    assert!(fit.normalized_weighted_sse < 1e-9);
    for structure in fit.fit.model.structures() {
        let b = &structure.coregionalization;
        assert!(
            b.covariance(0, 1).unwrap().powi(2)
                <= b.covariance(0, 0).unwrap() * b.covariance(1, 1).unwrap() + 1e-14
        );
    }
    assert_eq!(budget.used(), 0);
}
#[test]
fn bounds_flat_profiles_evaluation_limits_and_fixed_models_are_reported() {
    let budget = MemoryBudget::new(1 << 24);
    let cancel = Cancellation::default();
    let fit = ranges::fit(
        2,
        &bins([100.0; 3], 0.0, false, false, 1.0),
        &[Kernel::isotropic(Correlation::Exponential, 0)],
        &[Parameter {
            initial: 1.0,
            bounds: [0.2, 4.0],
        }],
        options(&budget, &cancel),
    )
    .unwrap();
    assert_eq!(fit.bounds, [Bound::Upper]);
    assert_eq!(fit.parameters, [4.0]);
    let mut flat = bins([2.0; 3], 0.0, false, false, 1.0);
    for bin in &mut flat {
        bin.lag = [100.0, 0.0, 0.0];
        bin.semivariance = if bin.variables == [0, 1] { -0.6 } else { 1.0 };
    }
    let fit = ranges::fit(
        2,
        &flat,
        &[Kernel::isotropic(Correlation::Spherical, 0)],
        &[Parameter {
            initial: 1.0,
            bounds: [0.2, 4.0],
        }],
        options(&budget, &cancel),
    )
    .unwrap();
    assert_eq!(fit.sensitivity_rank, 0);
    assert_eq!(fit.parameters, [1.0]);
    let mut limited = options(&budget, &cancel);
    limited.maximum_evaluations = 3;
    let fit = ranges::fit(
        2,
        &bins([2.3; 3], 0.0, false, false, 1.0),
        &[Kernel::isotropic(Correlation::Exponential, 0)],
        &[Parameter {
            initial: 0.6,
            bounds: [0.1, 10.0],
        }],
        limited,
    )
    .unwrap();
    assert_eq!(fit.stop, Stop::EvaluationLimit);
    assert_eq!(fit.evaluations, 3);
    assert_eq!(fit.objective_history.len(), 1);
    let fixed = Kernel::fixed(SpatialKernel {
        correlation: Correlation::Exponential,
        anisotropy: Anisotropy::isotropic(2.3).unwrap(),
    });
    let fit = ranges::fit(
        2,
        &bins([2.3; 3], 0.0, false, false, 1.0),
        &[fixed],
        &[],
        options(&budget, &cancel),
    )
    .unwrap();
    assert_eq!(fit.stop, Stop::NoFreeParameters);
    assert_eq!(fit.evaluations, 1);
    assert!(fit.parameters.is_empty());
    assert_eq!(budget.used(), 0);
}
#[test]
fn invalid_ranges_resources_cancellation_and_incomplete_inner_solves_fail_explicitly() {
    let budget = MemoryBudget::new(1 << 24);
    let cancel = Cancellation::default();
    let data = bins([2.3; 3], 0.0, false, false, 1.0);
    let kernel = Kernel::isotropic(Correlation::Exponential, 0);
    let parameter = Parameter {
        initial: 1.0,
        bounds: [0.1, 10.0],
    };
    for p in [
        Parameter {
            initial: 0.0,
            ..parameter
        },
        Parameter {
            bounds: [0.0, 10.0],
            ..parameter
        },
        Parameter {
            bounds: [1.0, 1.0],
            ..parameter
        },
        Parameter {
            initial: f64::NAN,
            ..parameter
        },
    ] {
        assert!(ranges::fit(
            2,
            &data,
            std::slice::from_ref(&kernel),
            &[p],
            options(&budget, &cancel)
        )
        .is_err());
    }
    assert!(ranges::fit(
        2,
        &data,
        &[Kernel::isotropic(Correlation::Nugget, 0)],
        &[parameter],
        options(&budget, &cancel)
    )
    .is_err());
    assert!(ranges::fit(
        2,
        &data,
        std::slice::from_ref(&kernel),
        &[parameter, parameter],
        options(&budget, &cancel)
    )
    .is_err());
    let mut opts = options(&budget, &cancel);
    opts.sill.maximum_iterations = 1;
    assert!(matches!(
        ranges::fit(2, &data, std::slice::from_ref(&kernel), &[parameter], opts),
        Err(Error::Convergence { .. })
    ));
    let tiny = MemoryBudget::new(1);
    assert!(matches!(
        ranges::fit(
            2,
            &data,
            std::slice::from_ref(&kernel),
            &[parameter],
            options(&tiny, &cancel)
        ),
        Err(Error::MemoryBudget { .. })
    ));
    assert_eq!(tiny.used(), 0);
    assert_eq!(budget.used(), 0);
    let mut too_small = options(&budget, &cancel);
    too_small.maximum_evaluations = 3;
    too_small.diagnostic_log_step = 1e-300;
    assert!(ranges::fit(
        2,
        &data,
        std::slice::from_ref(&kernel),
        &[parameter],
        too_small
    )
    .is_err());
    cancel.cancel();
    assert!(matches!(
        ranges::fit(2, &data, &[kernel], &[parameter], options(&budget, &cancel)),
        Err(Error::Cancelled)
    ));
}

#[test]
fn zero_design_and_zero_data_report_unidentified_ranges_without_invalid_models() {
    let budget = MemoryBudget::new(1 << 24);
    let cancel = Cancellation::default();
    let mut data = bins([2.3; 3], 0.0, false, false, 1.0);
    // At these enormous scales every represented exponential feature is zero.
    let fit = ranges::fit(
        2,
        &data,
        &[Kernel::isotropic(Correlation::Exponential, 0)],
        &[Parameter {
            initial: 1e100,
            bounds: [1e90, 1e110],
        }],
        options(&budget, &cancel),
    )
    .unwrap();
    assert_eq!(fit.fit.zero_feature_kernels, [0]);
    assert_eq!(fit.sensitivity_rank, 0);
    assert!(fit.normalized_weighted_sse > 0.0);
    for structure in fit.fit.model.structures() {
        assert_eq!(structure.coregionalization.covariance(0, 0).unwrap(), 0.0);
    }
    for bin in &mut data {
        bin.semivariance = 0.0;
    }
    let fit = ranges::fit(
        2,
        &data,
        &[Kernel::isotropic(Correlation::Exponential, 0)],
        &[Parameter {
            initial: 1.0,
            bounds: [0.1, 10.0],
        }],
        options(&budget, &cancel),
    )
    .unwrap();
    assert_eq!(fit.sensitivity_rank, 0);
    assert_eq!(fit.normalized_weighted_sse, 0.0);
    assert!(fit.fit.zero_feature_kernels.is_empty());
    assert_eq!(budget.used(), 0);
}

#[test]
fn three_variable_matern_fit_recovers_shared_range_and_reports_an_unobserved_pair() {
    let budget = MemoryBudget::new(1 << 24);
    let cancel = Cancellation::default();
    // A A^T for rows [1,0,0], [.4,.8,0], [-.2,.1,.7]. The missing (1,2)
    // cross-variogram is deliberately not replaced by invented zero bins.
    let sill = [[1.0, 0.4, -0.2], [0.4, 0.8, 0.0], [-0.2, 0.0, 0.54]];
    let mut data = Vec::new();
    for (a, row) in sill.iter().enumerate() {
        for (b, &sill) in row.iter().enumerate().skip(a) {
            if [a, b] == [1, 2] {
                continue;
            }
            for i in 1..=20 {
                let h = i as f64 * 0.4;
                let x = 3.0_f64.sqrt() * h / 3.1;
                data.push(CrossVariogramBin {
                    variables: [a, b],
                    lag: [h, 0.0, 0.0],
                    semivariance: sill * (1.0 - (1.0 + x) * (-x).exp()),
                    pairs: 200,
                });
            }
        }
    }
    let fit = ranges::fit(
        3,
        &data,
        &[Kernel::isotropic(Correlation::Matern32, 0)],
        &[Parameter {
            initial: 1.5,
            bounds: [0.5, 8.0],
        }],
        options(&budget, &cancel),
    )
    .unwrap();
    assert_eq!(fit.stop, Stop::LogStepTolerance);
    assert!((fit.parameters[0] / 3.1 - 1.0).abs() < 1e-4);
    assert_eq!(fit.fit.unobserved_pairs, [[1, 2]]);
    assert_eq!(fit.sensitivity_rank, 1);
    let b = &fit.fit.model.structures()[0].coregionalization;
    let matrix = faer::Mat::from_fn(3, 3, |i, j| b.covariance(i, j).unwrap());
    assert!(matrix
        .self_adjoint_eigenvalues(faer::Side::Lower)
        .unwrap()
        .iter()
        .all(|v| *v >= -1e-12));
    assert_eq!(budget.used(), 0);
}
