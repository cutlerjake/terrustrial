//! Independent diagonal-tensor formula, dense Schur reference and joint draws.
use faer::prelude::Solve;
use terrustrial_core::{
    anisotropy::{Family, Nonstationary, Tensor},
    data::{Observation, Point3, Site},
    execution::{Cancellation, MemoryBudget},
    gaussian::{GaussianPlan, NumericalTolerance, PrepareOptions, Simple, Uncertainty},
    model::{Coregionalization, Correlation},
};
fn scales(p: Point3) -> [f64; 3] {
    let [x, y, z] = p.coordinates();
    [1. + 0.2 * x, 1.5 + 0.1 * y, 0.7 + 0.1 * z]
}
fn point(i: usize) -> Point3 {
    Point3::new([
        (i % 4) as f64 * 0.41,
        (i / 4) as f64 * 0.33,
        (i % 3) as f64 * 0.27,
    ])
    .unwrap()
}
fn covariance(a: &Site, b: &Site, kernel: Correlation) -> f64 {
    let pa = a.support.center();
    let pb = b.support.center();
    let sa = scales(pa);
    let sb = scales(pb);
    let delta: [f64; 3] = std::array::from_fn(|j| pa.coordinates()[j] - pb.coordinates()[j]);
    let q = (0..3)
        .map(|j| 2. * delta[j] * delta[j] / (sa[j] * sa[j] + sb[j] * sb[j]))
        .sum::<f64>();
    let normalization = (0..3)
        .map(|j| (2. * sa[j] * sb[j] / (sa[j] * sa[j] + sb[j] * sb[j])).sqrt())
        .product::<f64>();
    [[1., 0.4], [0.4, 0.8]][a.variable][b.variable] * normalization * kernel.evaluate(q.sqrt())
}
#[test]
fn nonstationary_joint_conditioning_and_8192_draws_match_independent_moments() {
    const N: usize = 8192;
    const M: usize = 8;
    let observations: Vec<_> = (0..7)
        .map(|i| {
            Observation::new(
                i as u64,
                Site::point(point(i / 2), i % 2),
                0.3 * (i as f64).sin(),
                0.05,
            )
            .unwrap()
        })
        .collect();
    let targets: Vec<_> = (0..M)
        .map(|i| Site::point(point(5 + i / 2), i % 2))
        .collect();
    let values: Vec<_> = observations.iter().map(Observation::value).collect();
    let budget = MemoryBudget::new(1 << 20);
    let cancel = Cancellation::default();
    for (family, kernel) in [
        (Family::Gaussian, Correlation::Gaussian),
        (Family::Exponential, Correlation::Exponential),
        (Family::Matern32, Correlation::Matern32),
        (Family::Matern52, Correlation::Matern52),
    ] {
        let model = Nonstationary {
            family,
            coregionalization: Coregionalization::from_factor(2, 2, &[1., 0., 0.4, 0.8]).unwrap(),
            field: |p: Point3| {
                let s = scales(p);
                Tensor::new(std::array::from_fn(|i| {
                    std::array::from_fn(|j| if i == j { s[i] * s[i] } else { 0. })
                }))
            },
        };
        let k = faer::Mat::from_fn(7, 7, |i, j| {
            covariance(observations[i].site(), observations[j].site(), kernel)
                + if i == j { 0.05 } else { 0. }
        });
        let cross = faer::Mat::from_fn(7, M, |i, j| {
            covariance(observations[i].site(), &targets[j], kernel)
        });
        let weights = k.llt(faer::Side::Lower).unwrap().solve(&cross);
        let mean: [f64; M] =
            std::array::from_fn(|j| (0..7).map(|i| weights[(i, j)] * values[i]).sum());
        let conditional = faer::Mat::from_fn(M, M, |i, j| {
            covariance(&targets[i], &targets[j], kernel)
                - (0..7).map(|k| cross[(k, i)] * weights[(k, j)]).sum::<f64>()
        });
        let plan = GaussianPlan::prepare(
            &model,
            &observations,
            &targets,
            &Simple,
            PrepareOptions {
                uncertainty: Uncertainty::Joint,
                tolerance: NumericalTolerance::default(),
                budget: &budget,
                cancellation: &cancel,
            },
        )
        .unwrap();
        let actual_mean = plan.estimate(&values).unwrap();
        let actual_covariance = plan.covariance().unwrap();
        for i in 0..M {
            assert!((actual_mean[i] - mean[i]).abs() < 1e-10);
            for j in 0..M {
                assert!((actual_covariance[(i, j)] - conditional[(i, j)]).abs() < 1e-10);
            }
        }
        let mut sum = [0.; M];
        let mut products = [[0.; M]; M];
        let mut noise = [0.; M];
        let mut draw = [0.; M];
        let allocations = allocation_counter::measure(|| {
            for realization in 0..N {
                plan.sample_into(
                    &actual_mean,
                    9317,
                    realization as u64,
                    73,
                    &mut noise,
                    &mut draw,
                )
                .unwrap();
                let residual: [f64; M] = std::array::from_fn(|i| draw[i] - mean[i]);
                for i in 0..M {
                    sum[i] += residual[i];
                    for j in 0..M {
                        products[i][j] += residual[i] * residual[j];
                    }
                }
            }
        });
        assert_eq!(allocations.count_total, 0);
        for i in 0..M {
            assert!(
                (sum[i] / N as f64).abs() <= 6. * (conditional[(i, i)] / N as f64).sqrt(),
                "{family:?} mean {i}"
            );
            for j in 0..M {
                let se = ((conditional[(i, i)] * conditional[(j, j)]
                    + conditional[(i, j)].powi(2))
                    / N as f64)
                    .sqrt();
                assert!(
                    (products[i][j] / N as f64 - conditional[(i, j)]).abs() <= 6. * se,
                    "{family:?} covariance {i},{j}"
                );
            }
        }
    }
    assert_eq!(budget.used(), 0);
}
