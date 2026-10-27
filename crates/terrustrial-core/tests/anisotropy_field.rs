use terrustrial_core::{
    anisotropy::{
        field::{fit, Edge, Options, Window},
        fitting::DirectionalBin,
        AnchorField, Tensor, TensorField,
    },
    data::Point3,
    execution::{Cancellation, MemoryBudget},
    model::Anisotropy,
    Error,
};
fn point(x: [f64; 3]) -> Point3 {
    Point3::new(x).unwrap()
}
fn options<'a>(budget: &'a MemoryBudget, cancel: &'a Cancellation) -> Options<'a> {
    Options {
        range_bounds: [0.2, 10.0],
        regularization: 0.01,
        maximum_iterations: 20000,
        tolerance: 1e-11,
        maximum_cg_iterations: 2000,
        cg_tolerance: 1e-12,
        rank_tolerance: 1e-10,
        interpolation_neighbors: 4,
        interpolation_power: 2.0,
        budget,
        cancellation: cancel,
    }
}
fn bins(scales: [f64; 3]) -> Vec<DirectionalBin> {
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
        [0.3, 0.6, 0.9].map(|s| {
            let lag = d.map(|v| s * v);
            let h2 = (0..3).map(|j| (lag[j] / scales[j]).powi(2)).sum::<f64>();
            DirectionalBin {
                lag,
                semivariance: 0.1 + 1.7 * (-(-h2).exp_m1()),
                pairs: 100,
            }
        })
    })
    .collect()
}
fn window(i: usize, bins: &[DirectionalBin]) -> Window<'_> {
    Window {
        position: point([i as f64, 0., 0.]),
        bins,
        structured_sill: 1.7,
        nugget: 0.1,
    }
}
fn metric_error(a: [f64; 6], b: [f64; 6]) -> f64 {
    (0..6).map(|j| (a[j] - b[j]).abs()).fold(0.0, f64::max)
}
#[test]
fn constant_field_and_smoothing_only_window() {
    let b = bins([5., 2., 1.]);
    let windows = [window(0, &b), window(1, &[]), window(2, &b)];
    let edges = [
        Edge {
            nodes: [0, 1],
            weight: 1.0,
        },
        Edge {
            nodes: [1, 2],
            weight: 1.0,
        },
    ];
    let budget = MemoryBudget::new(1 << 20);
    let cancel = Cancellation::default();
    let result = fit(&windows, &edges, options(&budget, &cancel)).unwrap();
    assert!(result.converged);
    assert!(result.cg_relative_residual < 1e-12);
    assert!(result.weighted_transformed_sse < 1e-20);
    assert_eq!(
        result
            .nodes
            .iter()
            .map(|n| n.design_rank)
            .collect::<Vec<_>>(),
        [6, 0, 6]
    );
    for (_, tensor) in result.field.anchors() {
        let m = tensor.matrix();
        for (i, s) in [5., 2., 1.].iter().enumerate() {
            for (j, value) in m[i].iter().enumerate() {
                assert!((value - if i == j { s * s } else { 0. }).abs() < 1e-7);
            }
        }
    }
    assert_eq!(budget.used(), 0);
    let mut opts = options(&budget, &cancel);
    opts.regularization = 0.;
    assert!(fit(&windows, &edges, opts)
        .unwrap_err_string()
        .contains("rank 0"));
}
// Avoid a Debug bound on successful field output when checking error diagnostics.
trait ErrorText {
    fn unwrap_err_string(self) -> String;
}
impl<T> ErrorText for Result<T, Error> {
    fn unwrap_err_string(self) -> String {
        match self {
            Err(e) => e.to_string(),
            Ok(_) => panic!("expected error"),
        }
    }
}
#[test]
fn graph_solution_matches_independent_dense_normal_equations() {
    use faer::{Mat, Side};
    let a = bins([5., 2., 1.]);
    let b = bins([2., 3., 1.5]);
    let windows = [window(0, &a), window(1, &[]), window(2, &b)];
    let edges = [
        Edge {
            nodes: [0, 1],
            weight: 1.0,
        },
        Edge {
            nodes: [1, 2],
            weight: 1.0,
        },
    ];
    let budget = MemoryBudget::new(1 << 20);
    let cancel = Cancellation::default();
    let mut opts = options(&budget, &cancel);
    opts.regularization = 0.07;
    let fit = fit(&windows, &edges, opts).unwrap();
    assert!(fit.converged);
    let mut h = Mat::<f64>::zeros(18, 18);
    let mut rhs = Mat::<f64>::zeros(18, 1);
    for (i, w) in windows.iter().enumerate() {
        for b in w.bins {
            let [x, y, z] = b.lag.map(|v| v / fit.coordinate_scale);
            let p = [
                x * x,
                y * y,
                z * z,
                2_f64.sqrt() * x * y,
                2_f64.sqrt() * x * z,
                2_f64.sqrt() * y * z,
            ];
            let target = -(-((b.semivariance - 0.1) / 1.7)).ln_1p();
            for j in 0..6 {
                rhs[(6 * i + j, 0)] += p[j] * target;
                for k in 0..6 {
                    h[(6 * i + j, 6 * i + k)] += p[j] * p[k];
                }
            }
        }
    }
    for e in &edges {
        for j in 0..6 {
            let a = 6 * e.nodes[0] + j;
            let b = 6 * e.nodes[1] + j;
            h[(a, a)] += 0.07;
            h[(b, b)] += 0.07;
            h[(a, b)] -= 0.07;
            h[(b, a)] -= 0.07;
        }
    }
    use faer::prelude::Solve;
    let answer = h.llt(Side::Lower).unwrap().solve(&rhs);
    for i in 0..3 {
        for j in 0..6 {
            assert!((fit.nodes[i].normalized_metric[j] - answer[(6 * i + j, 0)]).abs() < 1e-8);
        }
    }
    let average = std::array::from_fn(|j| {
        0.5 * (fit.nodes[0].normalized_metric[j] + fit.nodes[2].normalized_metric[j])
    });
    assert!(metric_error(fit.nodes[1].normalized_metric, average) < 1e-9);
    let independent = fit::independent(&windows, &edges, &budget, &cancel);
    assert!(fit.roughness < independent);
    assert!(
        (fit.objective - 0.5 * (fit.weighted_transformed_sse + 0.07 * fit.roughness)).abs() < 1e-15
    );
}
mod fit {
    use super::*;
    pub fn independent(
        windows: &[Window<'_>],
        edges: &[Edge],
        budget: &MemoryBudget,
        cancel: &Cancellation,
    ) -> f64 {
        let w = [window(0, windows[0].bins), window(2, windows[2].bins)];
        let mut o = options(budget, cancel);
        o.regularization = 0.;
        let f = super::fit(&w, &[], o).unwrap();
        let x = [
            f.nodes[0].normalized_metric,
            std::array::from_fn(|j| {
                0.5 * (f.nodes[0].normalized_metric[j] + f.nodes[1].normalized_metric[j])
            }),
            f.nodes[1].normalized_metric,
        ];
        edges
            .iter()
            .map(|e| {
                e.weight
                    * (0..6)
                        .map(|j| (x[e.nodes[0]][j] - x[e.nodes[1]][j]).powi(2))
                        .sum::<f64>()
            })
            .sum()
    }
}
#[test]
fn pooled_rank_constraints_and_validation() {
    let b = bins([2., 2., 2.]);
    let a: Vec<_> = b.iter().filter(|b| b.lag[2] == 0.).cloned().collect();
    let c: Vec<_> = b.iter().filter(|b| b.lag[2] != 0.).cloned().collect();
    let windows = [window(0, &a), window(1, &c)];
    let edge = [Edge {
        nodes: [0, 1],
        weight: 1.,
    }];
    let budget = MemoryBudget::new(1 << 20);
    let cancel = Cancellation::default();
    let result = fit(&windows, &edge, options(&budget, &cancel)).unwrap();
    assert!(result.converged);
    assert!(result.nodes.iter().all(|n| n.design_rank < 6));
    assert!(fit(&windows, &[], options(&budget, &cancel)).is_err());
    let all = [window(0, &b)];
    let mut o = options(&budget, &cancel);
    o.range_bounds = [0.2, 1.5];
    let result = fit(&all, &[], o).unwrap();
    assert!(result.converged);
    assert_eq!(result.nodes[0].axes_at_bound, 3);
    assert!(result.weighted_transformed_sse > 0.);
    let mut o = options(&budget, &cancel);
    o.range_bounds = [0.2, 1.5];
    o.maximum_cg_iterations = 1;
    o.maximum_iterations = 1;
    o.tolerance = 1e-16;
    let result = fit(&all, &[], o).unwrap();
    assert!(!result.converged);
    assert!(result.stationarity > 1e-16);
    for e in [
        [Edge {
            nodes: [0, 0],
            weight: 1.,
        }],
        [Edge {
            nodes: [0, 2],
            weight: 1.,
        }],
        [Edge {
            nodes: [0, 1],
            weight: f64::NAN,
        }],
    ] {
        assert!(fit(&windows, &e, options(&budget, &cancel)).is_err());
    }
    assert!(fit(
        &windows,
        &[
            edge[0],
            Edge {
                nodes: [1, 0],
                weight: 2.
            }
        ],
        options(&budget, &cancel)
    )
    .is_err());
    assert!(fit(
        &[window(0, &b), window(0, &b)],
        &[],
        options(&budget, &cancel)
    )
    .is_err());
    let small = MemoryBudget::new(1);
    assert!(matches!(
        fit(&all, &[], options(&small, &cancel)),
        Err(Error::MemoryBudget { .. })
    ));
    cancel.cancel();
    assert!(matches!(
        fit(&all, &[], options(&budget, &cancel)),
        Err(Error::Cancelled)
    ));
    assert_eq!(budget.used(), 0);
}
#[test]
fn indexed_interpolation_matches_full_sort_including_extreme_distances() {
    let values: Vec<_> = (0..120)
        .map(|i| {
            let p = [(i % 7) as f64, (i / 7 % 5) as f64, (i / 35) as f64];
            (
                p,
                Tensor::from_anisotropy(&Anisotropy::isotropic(1. + i as f64 / 120.).unwrap())
                    .unwrap(),
            )
        })
        .collect();
    for scale in [1., 1e-200, 1e200] {
        let anchors: Vec<_> = values
            .iter()
            .map(|(p, t)| (point(p.map(|v| v * scale)), t.clone()))
            .collect();
        for k in [1, 4, 120, 200] {
            let field = AnchorField::new(anchors.clone(), k, 2.).unwrap();
            for query in [
                [0.5, 0.5, 0.5],
                [2.4, 1.7, 3.1],
                [10., 2., -1.],
                [0., 0., 0.],
            ] {
                let query = point(query.map(|v| v * scale));
                let actual = field.at(query).unwrap().matrix();
                let mut ds: Vec<_> = anchors
                    .iter()
                    .enumerate()
                    .map(|(i, (p, _))| (query.distance(*p), i))
                    .collect();
                ds.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
                ds.truncate(k.min(anchors.len()));
                let expected = if ds[0].0 == 0. {
                    anchors[ds[0].1].1.matrix()
                } else {
                    let closest = ds[0].0;
                    let total = ds.iter().map(|d| (closest / d.0).powi(2)).sum::<f64>();
                    std::array::from_fn(|i| {
                        std::array::from_fn(|j| {
                            ds.iter()
                                .map(|d| {
                                    (closest / d.0).powi(2) / total * anchors[d.1].1.matrix()[i][j]
                                })
                                .sum::<f64>()
                        })
                    })
                };
                for i in 0..3 {
                    for j in 0..3 {
                        assert!(
                            (actual[i][j] - expected[i][j]).abs() < 1e-12,
                            "scale {scale} k {k}"
                        );
                    }
                }
            }
            for (p, t) in &anchors {
                assert_eq!(field.at(*p).unwrap().matrix(), t.matrix());
            }
        }
    }
    let tensor = values[0].1.clone();
    assert!(AnchorField::new(
        vec![(point([0.; 3]), tensor.clone()), (point([-0.; 3]), tensor)],
        2,
        2.
    )
    .is_err());
}

#[test]
fn rotated_fitted_field_is_repeatable_and_produces_joint_psd_covariance() {
    use terrustrial_core::{
        anisotropy::NonstationaryGaussian,
        model::{Coregionalization, Covariance},
    };
    let angle = 0.37_f64;
    let (s, c) = angle.sin_cos();
    let models = [
        Anisotropy::new([[c, s, 0.], [-s, c, 0.], [0., 0., 1.]], [5., 2., 1.]).unwrap(),
        Anisotropy::new([[c, 0., s], [0., 1., 0.], [-s, 0., c]], [2., 3., 1.5]).unwrap(),
    ];
    let evidence: Vec<Vec<_>> = models
        .iter()
        .map(|a| {
            bins([1.; 3])
                .into_iter()
                .map(|mut b| {
                    let h = a.normalized_distance(point([0.; 3]), point(b.lag));
                    b.semivariance = 0.1 + 1.7 * (-(-h * h).exp_m1());
                    b
                })
                .collect()
        })
        .collect();
    let windows = [window(0, &evidence[0]), window(1, &evidence[1])];
    let edges = [Edge {
        nodes: [0, 1],
        weight: 1.,
    }];
    let budget = MemoryBudget::new(1 << 20);
    let cancel = Cancellation::default();
    let a = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap()
        .install(|| fit(&windows, &edges, options(&budget, &cancel)).unwrap());
    let b = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap()
        .install(|| fit(&windows, &edges, options(&budget, &cancel)).unwrap());
    assert!(a.converged && b.converged);
    assert_eq!(
        a.nodes
            .iter()
            .map(|n| n.normalized_metric.map(f64::to_bits))
            .collect::<Vec<_>>(),
        b.nodes
            .iter()
            .map(|n| n.normalized_metric.map(f64::to_bits))
            .collect::<Vec<_>>()
    );
    let model = NonstationaryGaussian {
        field: a.field,
        coregionalization: Coregionalization::from_factor(2, 2, &[1., 0., -0.5, 1.]).unwrap(),
    };
    let points: Vec<_> = (0..12)
        .map(|i| point([i as f64 / 7., (i % 3) as f64 / 11., (i % 5) as f64 / 13.]))
        .collect();
    let covariance = faer::Mat::from_fn(24, 24, |i, j| {
        model
            .point_covariance(points[i / 2], i % 2, points[j / 2], j % 2)
            .unwrap()
    });
    for i in 0..24 {
        for j in 0..24 {
            assert!((covariance[(i, j)] - covariance[(j, i)]).abs() < 1e-12);
        }
    }
    assert!(covariance
        .self_adjoint_eigenvalues(faer::Side::Lower)
        .unwrap()
        .iter()
        .all(|v| *v >= -1e-10));
}

#[test]
fn excluded_bins_zero_target_and_invalid_numeric_parameters() {
    let mut b = bins([2.; 3]);
    b.extend([
        DirectionalBin {
            lag: [1.; 3],
            semivariance: 1.8,
            pairs: 100,
        },
        DirectionalBin {
            lag: [0.; 3],
            semivariance: 0.2,
            pairs: 100,
        },
        DirectionalBin {
            lag: [1.; 3],
            semivariance: 0.2,
            pairs: 0,
        },
        DirectionalBin {
            lag: [1.; 3],
            semivariance: 0.01,
            pairs: 100,
        },
    ]);
    let budget = MemoryBudget::new(1 << 20);
    let cancel = Cancellation::default();
    let f = fit(&[window(0, &b)], &[], options(&budget, &cancel)).unwrap();
    assert_eq!(f.nodes[0].excluded_bins, 4);
    let mut zero = bins([2.; 3]);
    for bin in &mut zero {
        bin.semivariance = 0.1;
    }
    let f = fit(&[window(0, &zero)], &[], options(&budget, &cancel)).unwrap();
    assert!(f.converged);
    assert_eq!(f.cg_iterations, 0);
    assert_eq!(f.nodes[0].axes_at_bound, 3);
    for bin in [
        DirectionalBin {
            lag: [f64::NAN, 0., 0.],
            semivariance: 0.2,
            pairs: 100,
        },
        DirectionalBin {
            lag: [1.; 3],
            semivariance: f64::INFINITY,
            pairs: 100,
        },
    ] {
        assert!(fit(&[window(0, &[bin])], &[], options(&budget, &cancel)).is_err());
    }
    for bad in [0., -1., f64::NAN, f64::INFINITY] {
        let mut o = options(&budget, &cancel);
        o.tolerance = bad;
        assert!(fit(&[window(0, &b)], &[], o).is_err());
    }
    let mut o = options(&budget, &cancel);
    o.range_bounds = [1e-300, 1e-299];
    assert!(fit(&[window(0, &b)], &[], o).is_err());
    assert_eq!(budget.used(), 0);
}
