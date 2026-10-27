#![allow(clippy::needless_range_loop)]
//! Independent compound-symmetric posterior and 8192-draw ensemble gates.
//! This checks large multivariate sampling, separately from sequential-driver
//! conditioning tests. It is not a geological predictive-calibration claim.
use terrustrial_core::{
    data::{Observation, Point3, Site, Support},
    execution::{Cancellation, MemoryBudget},
    gaussian::{GaussianPlan, NumericalTolerance, PrepareOptions, Simple, Uncertainty},
    model::{Anisotropy, Coregionalization, Correlation, Lmc, Structure},
};
const SAMPLES: usize = 8192;
fn correlation(a: &Site, b: &Site) -> f64 {
    a.support
        .nodes()
        .map(|a| {
            b.support
                .nodes()
                .map(|b| {
                    let a = a.coordinates();
                    let b = b.coordinates();
                    (-((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2))
                        .sqrt()
                        / 2.)
                        .exp()
                })
                .sum::<f64>()
        })
        .sum::<f64>()
        / (a.support.node_count() * b.support.node_count()) as f64
}
#[test]
fn analytic_joint_ensembles_cover_one_to_32_variables_and_256_targets() {
    // Predeclared seed and six-standard-error limits. No sample retries or trimming.
    for (variables, targets, noise, blocks, rank_one) in [
        (1, 1, 0., false, false),
        (2, 7, 0.03, true, false),
        (2, 8, 0., false, false),
        (8, 9, 0.03, true, false),
        (8, 16, 0., false, false),
        (8, 64, 0.03, true, false),
        (32, 128, 0.03, false, false),
        (32, 256, 0., true, false),
        (8, 64, 0.03, true, true),
    ] {
        let a: f64 = if rank_one { 0. } else { 0.7 };
        let b: f64 = 0.3;
        let mut factor = vec![0.; variables * (variables + 1)];
        for i in 0..variables {
            factor[i * (variables + 1) + i] = a.sqrt();
            factor[i * (variables + 1) + variables] = b.sqrt();
        }
        let model = Lmc::new(vec![Structure {
            correlation: Correlation::Exponential,
            anisotropy: Anisotropy::isotropic(2.).unwrap(),
            coregionalization: Coregionalization::from_factor(variables, variables + 1, &factor)
                .unwrap(),
        }])
        .unwrap();
        let origin = Point3::new([0.; 3]).unwrap();
        let observed: Vec<_> = (0..variables)
            .map(|v| {
                Observation::new(
                    v as u64,
                    Site::point(origin, v),
                    (v as f64 * 0.6).sin(),
                    noise,
                )
                .unwrap()
            })
            .collect();
        let sites: Vec<_> = (0..targets)
            .map(|i| {
                let center = Point3::new([0.25 + (i / variables) as f64 * 0.3, 0.1, 0.]).unwrap();
                Site {
                    variable: i % variables,
                    support: if blocks {
                        Support::block(center, [0.04, 0.03, 0.02], [2; 3]).unwrap()
                    } else {
                        Support::point(center)
                    },
                }
            })
            .collect();
        let budget = MemoryBudget::new(64 << 20);
        let cancel = Cancellation::default();
        let plan = GaussianPlan::prepare(
            &model,
            &observed,
            &sites,
            &Simple,
            PrepareOptions {
                budget: &budget,
                cancellation: &cancel,
                uncertainty: Uncertainty::Joint,
                tolerance: NumericalTolerance::default(),
            },
        )
        .unwrap();
        let mean = plan
            .estimate(&observed.iter().map(Observation::value).collect::<Vec<_>>())
            .unwrap();
        // For C=aI+b11^T, C(C+noise I)^-1 and C(C+noise I)^-1 C
        // have known eigenvalues in the common and orthogonal subspaces.
        let common = a + variables as f64 * b;
        let mean_diag = a / (a + noise);
        let mean_common = common / (common + noise);
        let reduction_diag = a * a / (a + noise);
        let reduction_common = common * common / (common + noise);
        let sum_y = observed.iter().map(Observation::value).sum::<f64>();
        let rho: Vec<_> = sites
            .iter()
            .map(|s| correlation(s, observed[0].site()))
            .collect();
        let expected: Vec<_> = sites
            .iter()
            .enumerate()
            .map(|(i, s)| {
                rho[i]
                    * (mean_diag * observed[s.variable].value()
                        + (mean_common - mean_diag) / variables as f64 * sum_y)
            })
            .collect();
        let mut covariance = vec![vec![0.; targets]; targets];
        for i in 0..targets {
            assert!((mean[i] - expected[i]).abs() < 1e-10);
            for j in 0..targets {
                let equal = sites[i].variable == sites[j].variable;
                covariance[i][j] = correlation(&sites[i], &sites[j])
                    * (if equal { a + b } else { b })
                    - rho[i]
                        * rho[j]
                        * (if equal { reduction_diag } else { 0. }
                            + (reduction_common - reduction_diag) / variables as f64);
                assert!((plan.covariance().unwrap()[(i, j)] - covariance[i][j]).abs() < 1e-10);
            }
        }
        // Rank-one fields have one stochastic degree per spatial support.
        let active: Vec<_> = (0..targets)
            .filter(|i| !rank_one || i % variables == 0)
            .collect();
        let rank = active.len();
        let mut lower = vec![vec![0.; rank]; rank];
        for i in 0..rank {
            for j in 0..=i {
                let v = covariance[active[i]][active[j]]
                    - (0..j).map(|k| lower[i][k] * lower[j][k]).sum::<f64>();
                lower[i][j] = if i == j {
                    assert!(v > 0.);
                    v.sqrt()
                } else {
                    v / lower[j][j]
                };
            }
        }
        let mut sums = vec![0.; targets];
        let mut products = vec![vec![0.; targets]; targets];
        let mut zsum = vec![0.; rank];
        let mut zproducts = vec![vec![0.; rank]; rank];
        let mut third = vec![0.; rank];
        let mut fourth = vec![0.; rank];
        let mut standard = vec![0.; targets];
        let mut output = vec![0.; targets];
        let mut error = vec![0.; targets];
        let mut z = vec![0.; rank];
        for realization in 0..SAMPLES {
            plan.sample_into(
                &mean,
                928173,
                realization as u64,
                0,
                &mut standard,
                &mut output,
            )
            .unwrap();
            for i in 0..targets {
                error[i] = output[i] - expected[i];
                sums[i] += error[i];
                for j in 0..=i {
                    products[i][j] += error[i] * error[j];
                }
            }
            for i in 0..rank {
                z[i] = (error[active[i]] - (0..i).map(|j| lower[i][j] * z[j]).sum::<f64>())
                    / lower[i][i];
                zsum[i] += z[i];
                third[i] += z[i].powi(3);
                fourth[i] += z[i].powi(4);
                for j in 0..=i {
                    zproducts[i][j] += z[i] * z[j];
                }
            }
        }
        let n = SAMPLES as f64;
        let mut maximum = 0_f64;
        for i in 0..targets {
            maximum = maximum.max((sums[i] / n).abs() / (covariance[i][i] / n).sqrt());
            for j in 0..=i {
                let se =
                    ((covariance[i][i] * covariance[j][j] + covariance[i][j].powi(2)) / n).sqrt();
                maximum = maximum.max((products[i][j] / n - covariance[i][j]).abs() / se);
            }
        }
        for i in 0..rank {
            maximum = maximum
                .max(zsum[i].abs() / n.sqrt())
                .max((third[i] / n).abs() / (15. / n).sqrt())
                .max((fourth[i] / n - 3.).abs() / (96. / n).sqrt());
            for j in 0..=i {
                let same = i == j;
                maximum = maximum.max(
                    (zproducts[i][j] / n - (if same { 1. } else { 0. })).abs()
                        / ((if same { 2. } else { 1. }) / n).sqrt(),
                );
            }
        }
        println!("variables={variables} targets={targets} blocks={blocks} rank_one={rank_one} realizations={SAMPLES} maximum_standard_errors={maximum}");
        assert!(
            maximum <= 6.,
            "variables={variables},targets={targets},maximum={maximum}"
        );
        drop(plan);
        assert_eq!(budget.used(), 0);
    }
}
