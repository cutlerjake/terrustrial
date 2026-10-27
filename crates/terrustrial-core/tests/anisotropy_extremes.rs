//! Independent scale invariance and analytic references, not just scalar/SIMD parity.
use terrustrial_core::{
    data::{Observation, Point3, Site},
    execution::{Cancellation, MemoryBudget},
    gaussian::{GaussianWorkspace, NumericalTolerance, Simple, Uncertainty, WorkspaceOptions},
    model::{Anisotropy, Coregionalization, Correlation, Covariance, Lmc, Structure},
};
const IDENTITY: [[f64; 3]; 3] = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]];
fn point(p: [f64; 3]) -> Point3 {
    Point3::new(p).unwrap()
}
fn rotation() -> [[f64; 3]; 3] {
    let c = std::f64::consts::FRAC_1_SQRT_2;
    [[c, c, 0.], [-c, c, 0.], [0., 0., 1.]]
}
fn large_unit() -> f64 {
    // Power-of-two scaling preserves the normalized fixture coordinates exactly.
    f64::from_bits(2046_u64 << 52)
}
fn close(actual: f64, expected: f64) {
    assert!(
        actual.is_finite() && (actual - expected).abs() <= 2e-13 * (1. + expected.abs()),
        "actual={actual:?}, expected={expected:?}"
    );
}
fn model(kernel: Correlation, frame: [[f64; 3]; 3], scales: [f64; 3]) -> Lmc {
    Lmc::new(vec![Structure {
        correlation: kernel,
        anisotropy: Anisotropy::new(frame, scales).unwrap(),
        coregionalization: Coregionalization::scalar(1.).unwrap(),
    }])
    .unwrap()
}
#[test]
fn overflowing_differences_rotations_and_norms_preserve_finite_scaled_distances() {
    let u = large_unit();
    for frame in [IDENTITY, rotation()] {
        let geometry = Anisotropy::new(frame, [1.7 * u, 0.9 * u, 1.3 * u]).unwrap();
        for (a, b) in [
            ([1.5, 0., 0.], [-1.5, 0., 0.]),
            ([1.5, 1.5, 0.], [0., 0., 0.]),
            ([1.5, 1.5, 0.], [-1.5, -1.5, 0.]),
            ([1.5, 1.5, 0.], [1.5, -1.5, 0.]),
            ([1.5, 1.5, 1.5], [1.5, 1.5, 1.5]),
        ] {
            // Explicit normalized-coordinate calculation never forms huge differences.
            let delta: [f64; 3] = std::array::from_fn(|i| a[i] - b[i]);
            let projected: [f64; 3] = std::array::from_fn(|i| {
                (frame[i][0] * delta[0] + frame[i][1] * delta[1] + frame[i][2] * delta[2])
                    / [1.7, 0.9, 1.3][i]
            });
            let expected = projected[0].hypot(projected[1]).hypot(projected[2]);
            let a = point(a.map(|x| x * u));
            let b = point(b.map(|x| x * u));
            close(geometry.normalized_distance(a, b), expected);
            close(geometry.normalized_distance(b, a), expected);
        }
    }
    // Preserve the tiny transverse component even when an unrelated difference overflows.
    let mixed = Anisotropy::new(IDENTITY, [1.7 * u, 1e-308, 1.]).unwrap();
    close(
        mixed.normalized_distance(point([1.5 * u, 1e-308, 0.]), point([-1.5 * u, 0., 0.])),
        (3. / 1.7_f64).hypot(1.),
    );
    let rotated = Anisotropy::new(rotation(), [1.7 * u, 0.9 * u, 1e-308]).unwrap();
    close(
        rotated.normalized_distance(
            point([1.5 * u, 1.5 * u, 1e-308]),
            point([-1.5 * u, -1.5 * u, 0.]),
        ),
        (3. * 2_f64.sqrt() / 1.7).hypot(1.),
    );
    // A true infinite scaled distance must not become NaN through inf - inf
    // on a different, exactly coincident coordinate.
    let isotropic = Anisotropy::isotropic(1e-200).unwrap();
    assert_eq!(
        isotropic.normalized_distance(
            point([1.5 * u, 1.5 * u, 0.]),
            point([-1.5 * u, 1.5 * u, 0.]),
        ),
        f64::INFINITY
    );
}
#[test]
fn scalar_and_native_width_covariance_lines_match_normalized_references() {
    let u = large_unit();
    for kernel in [
        Correlation::Nugget,
        Correlation::Spherical,
        Correlation::Exponential,
        Correlation::Gaussian,
        Correlation::Matern32,
        Correlation::Matern52,
    ] {
        for frame in [IDENTITY, rotation()] {
            let large = model(kernel, frame, [1.7 * u, 0.9 * u, 1.3 * u]);
            let reference = model(kernel, frame, [1.7, 0.9, 1.3]);
            let target = Site::point(point([1.5 * u, 1.5 * u, 0.]), 0);
            let small_target = Site::point(point([1.5, 1.5, 0.]), 0);
            let coords = [
                [-1.5, -1.5, 0.],
                [0., 0., 0.],
                [1.5, 1.5, 0.],
                [-1.5, 1.5, 0.],
                [0.5, 1.5, 0.],
            ];
            let observations: Vec<_> = (0..33)
                .map(|i| {
                    Observation::new(
                        i,
                        Site::point(point(coords[i as usize % coords.len()].map(|x| x * u)), 0),
                        0.,
                        0.,
                    )
                    .unwrap()
                })
                .collect();
            for length in (0..=17).chain([31, 32, 33]) {
                let mut row = vec![0.; length];
                let mut column = vec![0.; length];
                let allocations = allocation_counter::measure(|| {
                    large
                        .covariance_row(&target, &observations[..length], &mut row)
                        .unwrap();
                    large
                        .covariance_column(&observations[..length], &target, &mut column)
                        .unwrap();
                });
                assert_eq!(allocations.count_total, 0);
                for i in 0..length {
                    let expected = reference
                        .covariance(
                            &small_target,
                            &Site::point(point(coords[i % coords.len()]), 0),
                        )
                        .unwrap();
                    close(row[i], expected);
                    close(column[i], expected);
                    let scalar = large.covariance(&target, observations[i].site()).unwrap();
                    assert_eq!(row[i].to_bits(), scalar.to_bits());
                    assert_eq!(column[i].to_bits(), scalar.to_bits());
                }
            }
        }
    }
}
#[test]
fn complete_gaussian_plan_matches_the_normalized_coordinate_system() {
    let u = large_unit();
    let budget = MemoryBudget::new(1 << 22);
    let cancel = Cancellation::default();
    let mut workspace = GaussianWorkspace::new(8, 9, 0, &budget).unwrap();
    let options = || WorkspaceOptions {
        uncertainty: Uncertainty::Joint,
        tolerance: NumericalTolerance::default(),
        cancellation: &cancel,
    };
    let mut reference_weights = Vec::new();
    let mut reference_covariance = Vec::new();
    for scale in [1., u, 1., u] {
        let model = model(
            Correlation::Exponential,
            rotation(),
            [1.7 * scale, 0.9 * scale, 1.3 * scale],
        );
        let observations: Vec<_> = (0..8)
            .map(|i| {
                Observation::new(
                    i,
                    Site::point(
                        point([
                            if i & 1 == 0 {
                                -1.5 * scale
                            } else {
                                1.5 * scale
                            },
                            if i & 2 == 0 {
                                -1.5 * scale
                            } else {
                                1.5 * scale
                            },
                            if i & 4 == 0 {
                                -0.5 * scale
                            } else {
                                0.5 * scale
                            },
                        ]),
                        0,
                    ),
                    i as f64,
                    0.01,
                )
                .unwrap()
            })
            .collect();
        let targets: Vec<_> = (0..9)
            .map(|i| {
                Site::point(
                    point([
                        ((i % 3) as f64 - 1.) * scale,
                        ((i / 3) as f64 - 1.) * scale,
                        0.,
                    ]),
                    0,
                )
            })
            .collect();
        let plan = workspace
            .prepare(&model, &observations, &targets, &Simple, options())
            .unwrap();
        let weights: Vec<_> = (0..9)
            .flat_map(|j| (0..8).map(move |i| (i, j)))
            .map(|(i, j)| plan.weights()[(i, j)])
            .collect();
        let covariance: Vec<_> = (0..9)
            .flat_map(|j| (0..9).map(move |i| (i, j)))
            .map(|(i, j)| plan.covariance().unwrap()[(i, j)])
            .collect();
        if reference_weights.is_empty() {
            reference_weights = weights;
            reference_covariance = covariance;
        } else {
            for (actual, expected) in weights
                .into_iter()
                .zip(&reference_weights)
                .chain(covariance.into_iter().zip(&reference_covariance))
            {
                close(actual, *expected);
            }
        }
    }
    drop(workspace);
    assert_eq!(budget.used(), 0);
}
