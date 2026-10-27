//! Independent augmented systems across joint size, support and physical units.
//! This is a finite acceptance matrix, not a claim of arbitrary-size acceptance.
use faer::{prelude::Solve, Mat};
use terrustrial_core::{
    data::{Observation, Point3, Site, Support},
    execution::{Cancellation, MemoryBudget},
    gaussian::{
        Drift, ExternalDrift, GaussianWorkspace, LinearDrift, NumericalTolerance, Ordinary, Simple,
        Uncertainty, WorkspaceOptions,
    },
    model::{Anisotropy, Coregionalization, Correlation, Lmc, Structure},
};
fn unit(v: usize) -> f64 {
    10f64.powi((v % 7) as i32 - 3)
}
fn shared(v: usize) -> f64 {
    if v.is_multiple_of(2) {
        0.3
    } else {
        -0.2
    }
}
fn diagonal(v: usize) -> f64 {
    1. + 0.01 * v as f64
}
fn point(i: usize) -> Point3 {
    Point3::new([
        1.7 * (i as f64 * 1.271 + 0.2).sin(),
        1.3 * (i as f64 * 2.137 + 0.7).sin(),
        0.9 * (i as f64 * 0.917 + 0.4).sin(),
    ])
    .unwrap()
}
fn site(i: usize, variables: usize, block: bool) -> Site {
    Site {
        variable: i % variables,
        support: if block {
            Support::block(point(i), [0.06, 0.02, 0.04], [2; 3]).unwrap()
        } else {
            Support::point(point(i))
        },
    }
}
// Independent construction of the configured discrete supports, not the
// production support iterator. No continuous integration accuracy is claimed.
fn nodes(s: &Site) -> Vec<[f64; 3]> {
    let center = s.support.center().coordinates();
    let half = s.support.half_extents();
    let divisions = s.support.divisions();
    let mut out = Vec::new();
    for z in 0..divisions[2] {
        for y in 0..divisions[1] {
            for x in 0..divisions[0] {
                let index = [x, y, z];
                out.push(std::array::from_fn(|a| {
                    center[a] + half[a] * (2. * (index[a] as f64 + 0.5) / divisions[a] as f64 - 1.)
                }));
            }
        }
    }
    out
}
fn covariance(a: &Site, b: &Site) -> f64 {
    let an = nodes(a);
    let bn = nodes(b);
    let mut total = 0.;
    // The same configured rotation/ranges, evaluated independently.
    let (s, c) = 0.43f64.sin_cos();
    for a in &an {
        for b in &bn {
            let d: [f64; 3] = std::array::from_fn(|i| a[i] - b[i]);
            let h = (((c * d[0] - s * d[1]) / 1.3).powi(2)
                + ((s * d[0] + c * d[1]) / 0.7).powi(2)
                + (d[2] / 2.1).powi(2))
            .sqrt();
            total += (-h).exp();
        }
    }
    let sill = shared(a.variable) * shared(b.variable)
        + if a.variable == b.variable {
            diagonal(a.variable).powi(2)
        } else {
            0.
        };
    sill * total / (an.len() * bn.len()) as f64
}
fn model(variables: usize) -> Lmc {
    let mut factor = vec![0.; variables * (variables + 1)];
    for v in 0..variables {
        factor[v * (variables + 1) + v] = unit(v) * diagonal(v);
        factor[v * (variables + 1) + variables] = unit(v) * shared(v);
    }
    let (s, c) = 0.43f64.sin_cos();
    Lmc::new(vec![Structure {
        correlation: Correlation::Exponential,
        anisotropy: Anisotropy::new([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]], [1.3, 0.7, 2.1])
            .unwrap(),
        coregionalization: Coregionalization::from_factor(variables, variables + 1, &factor)
            .unwrap(),
    }])
    .unwrap()
}
fn basis(s: &Site, per_variable: usize) -> Vec<f64> {
    let p = s.support.center().coordinates();
    match per_variable {
        0 => vec![],
        1 => vec![1.],
        4 => vec![1., p[0], p[1], p[2]],
        2 => {
            // A nonlinear external covariate must be averaged over support.
            let ns = nodes(s);
            vec![
                1.,
                ns.iter().map(|p| p[0] * p[0] + 0.3 * p[1]).sum::<f64>() / ns.len() as f64,
            ]
        }
        _ => unreachable!(),
    }
}
fn mean(s: &Site) -> f64 {
    let ns = nodes(s);
    (1. + 0.1 * s.variable as f64
        + ns.iter().map(|p| p[0] * p[0] - 0.2 * p[2]).sum::<f64>() / ns.len() as f64)
        * unit(s.variable)
}
fn close(actual: f64, expected: f64, label: &str) {
    assert!(
        (actual - expected).abs() <= 1e-10 * (1. + expected.abs()),
        "{label}: {actual} != {expected}"
    );
}
#[allow(clippy::too_many_arguments)]
fn check(
    variables: usize,
    n: usize,
    m: usize,
    block: bool,
    noisy: bool,
    per_variable: usize,
    drift: &impl Drift,
    workspace: &mut GaussianWorkspace,
) {
    let model = model(variables);
    let observations: Vec<_> = (0..n)
        .map(|i| {
            let s = site(i, variables, block && !i.is_multiple_of(3));
            let value = mean(&s) + unit(s.variable) * (i as f64 * 0.31).cos();
            let noise = if noisy {
                0.03 * unit(s.variable).powi(2)
            } else {
                0.
            };
            Observation::new(i as u64, s, value, noise).unwrap()
        })
        .collect();
    let mut targets: Vec<_> = (0..m).map(|j| site(n + j + 13, variables, block)).collect();
    // Exact conditioning and a rank-deficient joint target distribution.
    if m > 1 {
        targets[0] = *observations[0].site();
    }
    if m > 2 {
        targets[1] = targets[0];
    }
    if m > 3 {
        targets[2] = *observations[1].site();
    }
    let columns = variables * per_variable;
    let c = Mat::from_fn(n, n, |i, j| {
        covariance(observations[i].site(), observations[j].site())
            + if i == j && noisy { 0.03 } else { 0. }
    });
    let k = Mat::from_fn(n, m, |i, j| covariance(observations[i].site(), &targets[j]));
    let f: Vec<_> = observations
        .iter()
        .map(|o| basis(o.site(), per_variable))
        .collect();
    let ft: Vec<_> = targets.iter().map(|t| basis(t, per_variable)).collect();
    let augmented = Mat::from_fn(n + columns, n + columns, |i, j| {
        if i < n && j < n {
            c[(i, j)]
        } else if i < n {
            let col = j - n;
            if col / per_variable == observations[i].site().variable {
                f[i][col % per_variable]
            } else {
                0.
            }
        } else if j < n {
            let col = i - n;
            if col / per_variable == observations[j].site().variable {
                f[j][col % per_variable]
            } else {
                0.
            }
        } else {
            0.
        }
    });
    let rhs = Mat::from_fn(n + columns, m, |i, j| {
        if i < n {
            k[(i, j)]
        } else {
            let col = i - n;
            if col / per_variable == targets[j].variable {
                ft[j][col % per_variable]
            } else {
                0.
            }
        }
    });
    // Indefinite augmented LU, independent of the production LLT/QR reduction.
    let solution = augmented.partial_piv_lu().solve(&rhs);
    let residual = &augmented * &solution - &rhs;
    for j in 0..m {
        for i in 0..n + columns {
            assert!(
                residual[(i, j)].abs() < 1e-9,
                "reference system is insufficiently resolved"
            );
        }
    }
    let plan = workspace
        .prepare(
            &model,
            &observations,
            &targets,
            drift,
            WorkspaceOptions {
                uncertainty: Uncertainty::Joint,
                tolerance: NumericalTolerance::default(),
                cancellation: &Cancellation::default(),
            },
        )
        .unwrap();
    let values: Vec<_> = observations.iter().map(Observation::value).collect();
    let data_mean: Vec<_> = observations.iter().map(|o| mean(o.site())).collect();
    let target_mean: Vec<_> = targets.iter().map(mean).collect();
    let actual = plan
        .estimate_with_mean(&values, &data_mean, &target_mean)
        .unwrap();
    for j in 0..m {
        let expected = target_mean[j] / unit(targets[j].variable)
            + (0..n)
                .map(|i| {
                    solution[(i, j)] * (values[i] - data_mean[i])
                        / unit(observations[i].site().variable)
                })
                .sum::<f64>();
        close(actual[j] / unit(targets[j].variable), expected, "mean");
        for i in 0..n {
            close(
                plan.weights()[(i, j)] * unit(observations[i].site().variable)
                    / unit(targets[j].variable),
                solution[(i, j)],
                "weight",
            );
        }
        for col in 0..columns {
            let constraint: f64 = (0..n)
                .filter(|&i| observations[i].site().variable == col / per_variable)
                .map(|i| solution[(i, j)] * f[i][col % per_variable])
                .sum();
            close(constraint, rhs[(n + col, j)], "reference drift");
        }
    }
    let w = solution.as_ref().subrows(0, n);
    let cw = &c * w;
    for i in 0..m {
        for j in 0..m {
            let expected = covariance(&targets[i], &targets[j])
                - (0..n)
                    .map(|a| w[(a, i)] * k[(a, j)] + w[(a, j)] * k[(a, i)] - w[(a, i)] * cw[(a, j)])
                    .sum::<f64>();
            close(
                plan.covariance().unwrap()[(i, j)]
                    / (unit(targets[i].variable) * unit(targets[j].variable)),
                expected,
                "joint covariance",
            );
        }
    }
    if !noisy && m > 1 {
        close(
            actual[0] / unit(targets[0].variable),
            values[0] / unit(targets[0].variable),
            "hard-site mean in numerical plan",
        );
    }
}
#[test]
fn independent_joint_matrix_covers_32_variables_512_observations_64_targets_and_workspace_reuse() {
    let budget = MemoryBudget::new(64 << 20);
    let mut workspace = GaussianWorkspace::new(512, 64, 128, &budget).unwrap();
    // Deliberate growth/shrinkage crosses SIMD/packing boundaries. Heterotopic
    // samples have only one measured attribute each; unmeasured values are absent.
    for (variables, n, m) in [
        (1, 8, 1),
        (8, 64, 9),
        (32, 512, 64),
        (2, 32, 7),
        (32, 128, 16),
        (8, 128, 2),
        (2, 17, 8),
    ] {
        for (block, noisy) in [(false, false), (false, true), (true, false), (true, true)] {
            check(variables, n, m, block, noisy, 0, &Simple, &mut workspace);
            check(
                variables,
                n,
                m,
                block,
                noisy,
                1,
                &Ordinary { variables },
                &mut workspace,
            );
            check(
                variables,
                n,
                m,
                block,
                noisy,
                4,
                &LinearDrift::new(variables, [0.; 3], 1.).unwrap(),
                &mut workspace,
            );
            let external = ExternalDrift::new(2 * variables, |s: &Site, out: &mut [f64]| {
                out.fill(0.);
                out[2 * s.variable..2 * s.variable + 2].copy_from_slice(&basis(s, 2));
                Ok(())
            })
            .unwrap();
            check(variables, n, m, block, noisy, 2, &external, &mut workspace);
        }
    }
}
