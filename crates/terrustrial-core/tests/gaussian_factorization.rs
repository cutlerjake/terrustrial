//! Exponential Markov conditioning gives a closed-form large-system reference.
use terrustrial_core::{
    data::{Observation, Point3, Site},
    execution::{Cancellation, MemoryBudget},
    gaussian::{
        ExternalDrift, GaussianWorkspace, NumericalTolerance, Simple, Uncertainty, WorkspaceOptions,
    },
    model::{Anisotropy, Coregionalization, Correlation, Lmc, Structure},
    Error,
};

fn site(x: f64) -> Site {
    Site::point(Point3::new([x, 0., 0.]).unwrap(), 0)
}
fn model(correlation: Correlation) -> Lmc {
    Lmc::new(vec![Structure {
        correlation,
        anisotropy: Anisotropy::isotropic(2.).unwrap(),
        coregionalization: Coregionalization::scalar(1.).unwrap(),
    }])
    .unwrap()
}

#[test]
fn gaussian_markov_weights_and_uncertainty_hold_across_factor_block_boundaries() {
    let model = model(Correlation::Exponential);
    let budget = MemoryBudget::new(1 << 29);
    let cancel = Cancellation::default();
    let mut workspace = GaussianWorkspace::new(1025, 3, 0, &budget).unwrap();
    for n in [
        31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 511, 512, 513, 1023, 1024, 1025,
    ] {
        let data: Vec<_> = (0..n)
            .map(|i| Observation::new(i as u64, site(i as f64), (i as f64).sin(), 0.).unwrap())
            .collect();
        let coordinates = [0.25, 0.75, n as f64 - 1.5];
        let targets = coordinates.map(site);
        let plan = workspace
            .prepare(
                &model,
                &data,
                &targets,
                &Simple,
                WorkspaceOptions {
                    uncertainty: Uncertainty::Joint,
                    tolerance: NumericalTolerance::default(),
                    cancellation: &cancel,
                },
            )
            .unwrap();
        for (j, &x) in coordinates.iter().enumerate() {
            let left = x.floor() as usize;
            let fraction = x - x.floor();
            let wl = ((1. - fraction) / 2.).sinh() / 0.5_f64.sinh();
            let wr = (fraction / 2.).sinh() / 0.5_f64.sinh();
            for i in 0..n {
                let expected = if i == left {
                    wl
                } else if i == left + 1 {
                    wr
                } else {
                    0.
                };
                assert!(
                    (plan.weights()[(i, j)] - expected).abs() < 2e-12,
                    "n={n},weight({i},{j})"
                );
            }
            for (i, &other) in coordinates.iter().enumerate() {
                let expected = (-(x - other).abs() / 2.).exp()
                    - wl * (-(left as f64 - other).abs() / 2.).exp()
                    - wr * (-((left + 1) as f64 - other).abs() / 2.).exp();
                assert!(
                    (plan.covariance().unwrap()[(i, j)] - expected).abs() < 2e-12,
                    "n={n},covariance({i},{j})"
                );
            }
        }
    }
    drop(workspace);
    assert_eq!(budget.used(), 0);
}

#[test]
fn singular_large_conditioning_still_errors_and_workspace_recovers() {
    let model = model(Correlation::Nugget);
    let budget = MemoryBudget::new(1 << 27);
    let cancel = Cancellation::default();
    let mut workspace = GaussianWorkspace::new(257, 1, 0, &budget).unwrap();
    for n in [33, 64, 65, 128, 129, 256, 257] {
        let mut data: Vec<_> = (0..n)
            .map(|i| Observation::new(i as u64, site(i as f64), 0., 0.).unwrap())
            .collect();
        data[n - 1] = Observation::new((n - 1) as u64, site(0.), 0., 0.).unwrap();
        let options = || WorkspaceOptions {
            uncertainty: Uncertainty::Joint,
            tolerance: NumericalTolerance::default(),
            cancellation: &cancel,
        };
        assert!(
            matches!(
                workspace.prepare(&model, &data, &[site(0.5)], &Simple, options()),
                Err(Error::Singular { .. })
            ),
            "n={n}"
        );
        data[n - 1] = Observation::new((n - 1) as u64, site((n - 1) as f64), 0., 0.).unwrap();
        let plan = workspace
            .prepare(&model, &data, &[site(0.5)], &Simple, options())
            .unwrap();
        assert_eq!(plan.covariance().unwrap()[(0, 0)], 1.);
        assert!(plan.weights().col(0).iter().all(|w| *w == 0.));
    }
}

#[test]
fn large_equilibrated_qr_constraints_match_independent_least_squares() {
    let roots = [2_f64.powi(-450), 2_f64.powi(450)];
    let model = Lmc::new(vec![Structure {
        correlation: Correlation::Nugget,
        anisotropy: Anisotropy::isotropic(2.).unwrap(),
        coregionalization: Coregionalization::from_factor(2, 2, &[roots[0], 0., 0., roots[1]])
            .unwrap(),
    }])
    .unwrap();
    let drift = ExternalDrift::new(4, |s: &Site, out: &mut [f64]| {
        out.fill(0.);
        out[2 * s.variable] = 1.;
        out[2 * s.variable + 1] = 1. + 2_f64.powi(-24) * s.support.center().coordinates()[0];
        Ok(())
    })
    .unwrap();
    let budget = MemoryBudget::new(1 << 27);
    let cancel = Cancellation::default();
    let mut workspace = GaussianWorkspace::new(258, 4, 4, &budget).unwrap();
    for count in [33, 65, 129] {
        let make_site = |x, v| Site {
            variable: v,
            ..site(x)
        };
        let data: Vec<_> = (0..2 * count)
            .map(|i| {
                let x = ((i / 2) as f64 - (count / 2) as f64) / 8.;
                Observation::new(i as u64, make_site(x, i % 2), 0., 0.).unwrap()
            })
            .collect();
        let xx: f64 = data
            .iter()
            .step_by(2)
            .map(|o| o.site().support.center().coordinates()[0].powi(2))
            .sum();
        let targets = [
            make_site(0.0625, 0),
            make_site(0.0625, 1),
            make_site(0.4375, 0),
            make_site(0.4375, 1),
        ];
        let plan = workspace
            .prepare(
                &model,
                &data,
                &targets,
                &drift,
                WorkspaceOptions {
                    uncertainty: Uncertainty::Joint,
                    tolerance: NumericalTolerance::default(),
                    cancellation: &cancel,
                },
            )
            .unwrap();
        for (j, target) in targets.iter().enumerate() {
            let x = target.support.center().coordinates()[0];
            for (i, observation) in data.iter().enumerate() {
                let expected = if i % 2 == j % 2 {
                    1. / count as f64
                        + x * observation.site().support.center().coordinates()[0] / xx
                } else {
                    0.
                };
                let actual = plan.weights()[(i, j)] * roots[i % 2] / roots[j % 2];
                assert!(
                    (actual - expected).abs() < 2e-7,
                    "count={count},weight({i},{j})"
                );
            }
            for (i, other) in targets.iter().enumerate() {
                let expected = if i % 2 == j % 2 {
                    f64::from(i == j)
                        + 1. / count as f64
                        + x * other.support.center().coordinates()[0] / xx
                } else {
                    0.
                };
                let actual = plan.covariance().unwrap()[(i, j)] / roots[i % 2] / roots[j % 2];
                assert!(
                    (actual - expected).abs() < 2e-7,
                    "count={count},covariance({i},{j})"
                );
            }
        }
    }
}
