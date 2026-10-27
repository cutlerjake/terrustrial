use terrustrial_core::{
    anisotropy::{
        field::{self, Edge, Options, Window},
        fitting::DirectionalBin,
        selection::select_families,
        Family, Nonstationary, Tensor,
    },
    data::Point3,
    execution::{Cancellation, MemoryBudget},
    model::{Anisotropy, Coregionalization, Correlation, Covariance},
};
const FAMILIES: [(Family, Correlation); 4] = [
    (Family::Gaussian, Correlation::Gaussian),
    (Family::Exponential, Correlation::Exponential),
    (Family::Matern32, Correlation::Matern32),
    (Family::Matern52, Correlation::Matern52),
];
fn p(v: [f64; 3]) -> Point3 {
    Point3::new(v).unwrap()
}
fn o<'a>(budget: &'a MemoryBudget, cancel: &'a Cancellation) -> Options<'a> {
    Options {
        range_bounds: [0.1, 20.],
        regularization: 0.1,
        maximum_iterations: 20000,
        tolerance: 1e-10,
        maximum_cg_iterations: 2000,
        cg_tolerance: 1e-12,
        rank_tolerance: 1e-10,
        interpolation_neighbors: 4,
        interpolation_power: 2.,
        budget,
        cancellation: cancel,
    }
}
#[test]
fn all_supported_nonstationary_families_reduce_to_stationary_and_are_joint_psd() {
    let (s, c) = 0.71_f64.sin_cos();
    let geometry = Anisotropy::new([[c, s, 0.], [-s, c, 0.], [0., 0., 1.]], [3., 1., 0.5]).unwrap();
    let tensor = Tensor::from_anisotropy(&geometry).unwrap();
    let points: Vec<_> = (0..20)
        .map(|i| {
            p([
                (i % 5) as f64 * 0.31,
                ((i / 5) % 4) as f64 * 0.29,
                (i % 3) as f64 * 0.21,
            ])
        })
        .collect();
    let sill = Coregionalization::from_factor(2, 2, &[1., 0., -0.5, 0.8]).unwrap();
    for (family, kernel) in FAMILIES {
        let stationary = Nonstationary {
            family,
            field: |_: Point3| Ok(tensor.clone()),
            coregionalization: sill.clone(),
        };
        for &a in &points {
            for &b in &points {
                for va in 0..2 {
                    for vb in 0..2 {
                        let expected = kernel.evaluate(geometry.normalized_distance(a, b))
                            * sill.covariance(va, vb).unwrap();
                        let actual = stationary.point_covariance(a, va, b, vb).unwrap();
                        assert!(
                            (actual - expected).abs() < 1e-12,
                            "{family:?} {actual} {expected}"
                        );
                    }
                }
            }
        }
        let model = Nonstationary {
            family,
            coregionalization: sill.clone(),
            field: |p: Point3| {
                let [x, y, z] = p.coordinates();
                let (s, c) = (0.8 * x + 0.2 * y).sin_cos();
                Tensor::from_anisotropy(&Anisotropy::new(
                    [[c, s, 0.], [-s, c, 0.], [0., 0., 1.]],
                    [1. + x, 0.5 + y, 0.3 + z],
                )?)
            },
        };
        let matrix = faer::Mat::from_fn(40, 40, |i, j| {
            model
                .point_covariance(points[i / 2], i % 2, points[j / 2], j % 2)
                .unwrap()
        });
        for i in 0..40 {
            for j in 0..40 {
                assert!((matrix[(i, j)] - matrix[(j, i)]).abs() < 1e-14);
            }
        }
        assert!(
            matrix
                .self_adjoint_eigenvalues(faer::Side::Lower)
                .unwrap()
                .iter()
                .all(|x| *x >= -1e-12),
            "{family:?}"
        );
    }
}
fn bins(kernel: Correlation) -> Vec<DirectionalBin> {
    [
        [1_f64, 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.],
        [1., 1., 0.],
        [1., 0., 1.],
        [0., 1., 1.],
        [1., -1., 1.],
    ]
    .into_iter()
    .flat_map(|d| {
        [0.1, 0.3, 0.6, 1., 1.8].map(move |scale| {
            let lag = d.map(|x| x * scale);
            let distance = (0..3)
                .map(|i| (lag[i] / [3., 1.5, 0.8][i]).powi(2))
                .sum::<f64>()
                .sqrt();
            DirectionalBin {
                lag,
                semivariance: 0.1 + 1.4 * (1. - kernel.evaluate(distance)),
                pairs: 100,
            }
        })
    })
    .collect()
}
#[test]
fn fitting_recovers_each_radial_family_and_validation_selects_matern() {
    let budget = MemoryBudget::new(1 << 24);
    let cancel = Cancellation::default();
    for (family, kernel) in FAMILIES {
        let bins = bins(kernel);
        let windows: Vec<_> = (0..6)
            .map(|i| Window {
                position: p([i as f64, 0., 0.]),
                bins: &bins,
                structured_sill: 1.4,
                nugget: 0.1,
            })
            .collect();
        let edges: Vec<_> = (0..5)
            .map(|i| Edge {
                nodes: [i, i + 1],
                weight: 1.,
            })
            .collect();
        let result = field::fit_family(&windows, &edges, family, o(&budget, &cancel)).unwrap();
        assert!(result.converged);
        for (_, tensor) in result.field.anchors() {
            for (i, row) in tensor.matrix().iter().enumerate() {
                for (j, &x) in row.iter().enumerate() {
                    assert!(
                        (x - if i == j { [9., 2.25, 0.64][i] } else { 0. }).abs() < 1e-7,
                        "{family:?} {i} {j} {x}"
                    );
                }
            }
        }
        if family == Family::Matern32 {
            let selection = select_families(
                &windows,
                &edges,
                &[0, 0, 1, 1, 2, 2],
                &FAMILIES.map(|p| p.0),
                &[0.01, 0.1],
                o(&budget, &cancel),
            )
            .unwrap();
            assert_eq!(selection.fit.family, Family::Matern32);
            assert!(
                selection.candidates[selection.selected]
                    .validation_loss
                    .unwrap()
                    < 1e-18
            );
            for c in &selection.candidates {
                if c.family != Family::Matern32 {
                    assert!(c.validation_loss.is_none_or(|v| v > 1e-8), "{c:?}");
                }
            }
        }
    }
    assert_eq!(budget.used(), 0);
}
