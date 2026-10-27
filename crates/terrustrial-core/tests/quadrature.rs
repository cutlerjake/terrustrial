use statrs::function::erf::erf;
use std::sync::atomic::{AtomicUsize, Ordering};
use terrustrial_core::{
    data::Observation,
    data::{Point3, Site, Support},
    execution::{Cancellation, MemoryBudget},
    gaussian::{GaussianWorkspace, NumericalTolerance, Simple, Uncertainty, WorkspaceOptions},
    model::{
        quadrature::{refine_covariance, RefinementOptions, RefinementStop},
        Anisotropy, Coregionalization, Correlation, Covariance, Lmc, Structure,
    },
    Error, Result,
};
fn site(center: [f64; 3], half: [f64; 3], n: usize, variable: usize) -> Site {
    Site {
        variable,
        support: Support::block(
            Point3::new(center).unwrap(),
            half,
            half.map(|h| if h == 0. { 1 } else { n }),
        )
        .unwrap(),
    }
}
fn model(kernel: Correlation) -> Lmc {
    Lmc::new(vec![Structure {
        correlation: kernel,
        anisotropy: Anisotropy::new([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], [1.3, 0.9, 1.7])
            .unwrap(),
        coregionalization: Coregionalization::from_factor(2, 2, &[1., 0., -0.4, 1.2]).unwrap(),
    }])
    .unwrap()
}
// Closed-form antiderivatives of exp(-(x-y)^2/r^2), independent of midpoint nodes.
fn gaussian_axis(a: f64, ha: f64, b: f64, hb: f64, r: f64) -> f64 {
    if ha == 0. && hb == 0. {
        return (-((a - b) / r).powi(2)).exp();
    }
    if ha == 0. {
        return gaussian_axis(b, hb, a, ha, r);
    }
    let root_pi = std::f64::consts::PI.sqrt();
    if hb == 0. {
        return root_pi * r / (4. * ha) * (erf((a + ha - b) / r) - erf((a - ha - b) / r));
    }
    let h = |x: f64| 0.5 * r * r * (-(x / r).powi(2)).exp() + 0.5 * root_pi * r * x * erf(x / r);
    (h(a + ha - b + hb) - h(a - ha - b + hb) - h(a + ha - b - hb) + h(a - ha - b - hb))
        / (4. * ha * hb)
}
fn exact(a: &Site, b: &Site) -> f64 {
    let ac = a.support.center().coordinates();
    let bc = b.support.center().coordinates();
    let ah = a.support.half_extents();
    let bh = b.support.half_extents();
    let spatial: f64 = (0..3)
        .map(|i| gaussian_axis(ac[i], ah[i], bc[i], bh[i], [1.3, 0.9, 1.7][i]))
        .product();
    spatial * [[1., -0.4], [-0.4, 1.6]][a.variable][b.variable]
}
fn close(a: f64, b: f64, tolerance: f64) {
    assert!(
        (a - b).abs() <= tolerance,
        "{a} != {b} (tolerance {tolerance})"
    );
}

#[test]
fn midpoint_convergence_matches_closed_form_point_line_area_and_volume_integrals() {
    let model = model(Correlation::Gaussian);
    for half in [[0.4, 0., 0.], [0.4, 0.3, 0.], [0.4, 0.3, 0.2]] {
        for other_half in [[0.; 3], half] {
            for coincident in [false, true] {
                let mut errors = Vec::new();
                for n in [1, 2, 4, 8, 16] {
                    let a = site([0.; 3], half, n, 0);
                    let b = site(
                        if coincident {
                            [0.; 3]
                        } else {
                            [0.7, -0.2, 0.1]
                        },
                        other_half,
                        n,
                        1,
                    );
                    let value = model.covariance(&a, &b).unwrap();
                    let expected = exact(&a, &b);
                    errors.push((value - expected).abs());
                    close(value, model.covariance(&b, &a).unwrap(), 1e-12);
                }
                // Predeclared second-order convergence gate for these smooth fixtures.
                for w in errors.windows(2) {
                    assert!(w[1] < 0.3 * w[0], "errors={errors:?}");
                }
                assert!(errors[4] < 5e-4, "errors={errors:?}");
            }
        }
    }
    // Nonsmooth exponential covariance averaged over an identical 1-D block.
    // For width w and range r: 2r/w - 2r^2/w^2 * (1-exp(-w/r)).
    let model = self::model(Correlation::Exponential);
    let (width, range) = (0.8_f64, 1.3_f64);
    let expected =
        2. * range / width + 2. * range * range / (width * width) * (-width / range).exp_m1();
    let mut previous = f64::INFINITY;
    for n in [2, 4, 8, 16, 32, 64] {
        let a = site([0.; 3], [width / 2., 0., 0.], n, 0);
        let error = (model.covariance(&a, &a).unwrap() - expected).abs();
        assert!(error < previous);
        previous = error;
    }
    assert!(previous < 5e-5, "error={previous}");
}

#[test]
fn nugget_references_retain_discrete_coincidence_and_reveal_grid_aliasing() {
    let model = model(Correlation::Nugget);
    fn gcd(mut a: usize, mut b: usize) -> usize {
        while b != 0 {
            let t = b;
            b = a % b;
            a = t;
        }
        a
    }
    for n in 1..=9 {
        let a = site([0.; 3], [0.5, 0., 0.], n, 0);
        close(model.covariance(&a, &a).unwrap(), 1. / n as f64, 1e-15);
        let center = site([0.; 3], [0.; 3], 1, 0);
        close(
            model.covariance(&a, &center).unwrap(),
            if n % 2 == 1 { 1. / n as f64 } else { 0. },
            1e-15,
        );
        // Only dyadic midpoint coordinates have identical binary64 representations
        // guaranteed by integer grid arithmetic; use dyadic pairs for cross grids.
        for m in [1, 2, 4, 8] {
            if !n.is_power_of_two() {
                continue;
            }
            let b = site([0.; 3], [0.5, 0., 0.], m, 0);
            let g = gcd(n, m);
            let common = if (n / g) % 2 == 1 && (m / g) % 2 == 1 {
                g
            } else {
                0
            };
            close(
                model.covariance(&a, &b).unwrap(),
                common as f64 / (n * m) as f64,
                1e-15,
            );
        }
    }
}

#[test]
fn common_refinement_matches_analytic_gaussian_weights_and_uncertainty() {
    let model = model(Correlation::Gaussian);
    let sites = [
        site([-0.7, 0., 0.], [0.25, 0., 0.], 2, 0),
        site([0.8, 0., 0.], [0.35, 0., 0.], 2, 1),
        site([0.1, 0., 0.], [0.2, 0., 0.], 2, 0),
        site([0.3, 0., 0.], [0.; 3], 1, 1),
    ];
    let cancel = Cancellation::default();
    let budget = MemoryBudget::new(1 << 20);
    let refined = refine_covariance(
        &model,
        &sites,
        RefinementOptions {
            relative_tolerance: 1e-5,
            maximum_refinements: 8,
            ..Default::default()
        },
        &budget,
        &cancel,
    )
    .unwrap();
    assert_eq!(refined.report().stop, RefinementStop::Stable);
    assert_eq!(refined.report().stable_comparisons, 2);
    assert_eq!(budget.used(), refined.accounted_bytes());
    let nodes = refined.sites()[0].support.node_count();
    assert!(nodes > 2);
    assert!(refined.sites()[..3]
        .iter()
        .all(|s| s.support.node_count() == nodes));
    assert_eq!(refined.sites()[3], sites[3]);
    let observations = [
        Observation::new(0, refined.sites()[0], 1.2, 0.1).unwrap(),
        Observation::new(1, refined.sites()[1], -0.4, 0.1).unwrap(),
    ];
    let mut workspace = GaussianWorkspace::new(2, 2, 0, &budget).unwrap();
    let plan = workspace
        .prepare(
            &model,
            &observations,
            &refined.sites()[2..],
            &Simple,
            WorkspaceOptions {
                uncertainty: Uncertainty::Joint,
                tolerance: NumericalTolerance::default(),
                cancellation: &cancel,
            },
        )
        .unwrap();
    let a = exact(&sites[0], &sites[0]) + 0.1;
    let b = exact(&sites[0], &sites[1]);
    let d = exact(&sites[1], &sites[1]) + 0.1;
    let determinant = a * d - b * b;
    let cross: [[f64; 2]; 2] = std::array::from_fn(|j| {
        [
            exact(&sites[0], &sites[j + 2]),
            exact(&sites[1], &sites[j + 2]),
        ]
    });
    let weights: [[f64; 2]; 2] = std::array::from_fn(|j| {
        [
            (d * cross[j][0] - b * cross[j][1]) / determinant,
            (a * cross[j][1] - b * cross[j][0]) / determinant,
        ]
    });
    for j in 0..2 {
        for i in 0..2 {
            close(plan.weights()[(i, j)], weights[j][i], 2e-5);
            let expected = exact(&sites[i + 2], &sites[j + 2])
                - cross[i][0] * weights[j][0]
                - cross[i][1] * weights[j][1];
            close(plan.covariance().unwrap()[(i, j)], expected, 2e-5);
        }
    }
    let prediction = plan.estimate(&[1.2, -0.4]).unwrap();
    for j in 0..2 {
        close(
            prediction[j],
            1.2 * weights[j][0] - 0.4 * weights[j][1],
            2e-5,
        );
    }
    drop(workspace);
    drop(refined);
    assert_eq!(budget.used(), 0);
}

struct Controlled<'a> {
    calls: AtomicUsize,
    cancel: Option<&'a Cancellation>,
    nan: bool,
}
impl Covariance for Controlled<'_> {
    fn variables(&self) -> usize {
        1
    }
    fn point_covariance(&self, _: Point3, _: usize, _: Point3, _: usize) -> Result<f64> {
        panic!("override must be honored")
    }
    fn covariance(&self, a: &Site, b: &Site) -> Result<f64> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        if let Some(cancel) = self.cancel {
            cancel.cancel();
        }
        if self.nan {
            return Ok(f64::NAN);
        }
        Ok(1. / a.support.node_count() as f64 + 1. / b.support.node_count() as f64)
    }
}
#[test]
fn work_memory_dimension_cancellation_and_failure_limits_leave_no_partial_level() {
    let budget = MemoryBudget::new(1 << 20);
    let cancel = Cancellation::default();
    let model = Controlled {
        calls: AtomicUsize::new(0),
        cancel: None,
        nan: false,
    };
    let sites = [
        site([0.; 3], [0.5, 0., 0.], 1, 0),
        site([2., 0., 0.], [0.; 3], 1, 0),
    ];
    // Two blocks/point matrices require 3+8 = 11 node pairs for the first level.
    let options = RefinementOptions {
        maximum_point_pairs: 10,
        ..Default::default()
    };
    let result = refine_covariance(&model, &sites, options, &budget, &cancel).unwrap();
    assert_eq!(result.report().stop, RefinementStop::WorkLimit);
    assert_eq!(result.sites(), sites);
    assert_eq!(model.calls.load(Ordering::Relaxed), 0);
    assert_eq!(result.report().maximum_absolute_change, None);
    let retained = result.accounted_bytes();
    drop(result);
    assert_eq!(budget.used(), 0);
    let result = refine_covariance(
        &model,
        &sites,
        RefinementOptions {
            maximum_point_pairs: 11,
            ..options
        },
        &budget,
        &cancel,
    )
    .unwrap();
    assert_eq!(result.report().stop, RefinementStop::WorkLimit);
    assert_eq!(result.report().completed_refinements, 1);
    assert_eq!(result.report().estimated_point_pairs, 11);
    assert_eq!(result.report().covariance_evaluations, 6);
    assert_eq!(result.sites()[0].support.node_count(), 2);
    drop(result);
    let small = MemoryBudget::new(retained * 2 - 1);
    assert!(matches!(
        refine_covariance(&model, &sites, options, &small, &cancel),
        Err(Error::MemoryBudget { .. })
    ));
    assert_eq!(small.used(), 0);
    let huge = [site([0.; 3], [0.5, 0., 0.], usize::MAX, 0)];
    let result = refine_covariance(
        &model,
        &huge,
        RefinementOptions::default(),
        &budget,
        &cancel,
    )
    .unwrap();
    assert_eq!(result.report().stop, RefinementStop::DimensionLimit);
    assert_eq!(result.sites(), huge);
    drop(result);
    let collapsed = [site([1e20, 0., 0.], [1., 0., 0.], 1, 0)];
    let result = refine_covariance(
        &model,
        &collapsed,
        RefinementOptions::default(),
        &budget,
        &cancel,
    )
    .unwrap();
    assert_eq!(result.report().stop, RefinementStop::ResolutionLimit);
    assert_eq!(result.report().completed_refinements, 0);
    assert_eq!(result.sites(), collapsed);
    drop(result);
    let nan = Controlled {
        calls: AtomicUsize::new(0),
        cancel: None,
        nan: true,
    };
    assert!(matches!(
        refine_covariance(&nan, &sites, RefinementOptions::default(), &budget, &cancel),
        Err(Error::Numerical { .. })
    ));
    let cancelling = Controlled {
        calls: AtomicUsize::new(0),
        cancel: Some(&cancel),
        nan: false,
    };
    assert!(matches!(
        refine_covariance(
            &cancelling,
            &sites,
            RefinementOptions::default(),
            &budget,
            &cancel
        ),
        Err(Error::Cancelled)
    ));
    assert_eq!(cancelling.calls.load(Ordering::Relaxed), 1);
    assert_eq!(budget.used(), 0);
    assert!(matches!(
        refine_covariance(&model, &sites, options, &budget, &cancel),
        Err(Error::Cancelled)
    ));
    let cancel = Cancellation::default();
    for options in [
        RefinementOptions {
            absolute_tolerance: f64::NAN,
            ..options
        },
        RefinementOptions {
            relative_tolerance: -1.,
            ..options
        },
        RefinementOptions {
            stable_comparisons: 0,
            ..options
        },
    ] {
        assert!(refine_covariance(&model, &sites, options, &budget, &cancel).is_err());
        assert_eq!(budget.used(), 0);
    }
    let points = [site([0.; 3], [0.; 3], 1, 0)];
    let result = refine_covariance(&model, &points, options, &budget, &cancel).unwrap();
    assert_eq!(result.report().stop, RefinementStop::PointsOnly);
    drop(result);
    let mut counts = Vec::new();
    for levels in [2, 6] {
        let measured = allocation_counter::measure(|| {
            let result = refine_covariance(
                &model,
                &sites,
                RefinementOptions {
                    maximum_refinements: levels,
                    absolute_tolerance: 0.,
                    relative_tolerance: 0.,
                    maximum_point_pairs: u64::MAX,
                    ..options
                },
                &budget,
                &cancel,
            )
            .unwrap();
            assert_eq!(result.report().stop, RefinementStop::RefinementLimit);
        });
        counts.push(measured.count_total);
    }
    assert_eq!(counts, [2, 2]);
    assert_eq!(budget.used(), 0);
}

struct Oscillatory;
impl Covariance for Oscillatory {
    fn variables(&self) -> usize {
        1
    }
    fn point_covariance(&self, a: Point3, _: usize, b: Point3, _: usize) -> Result<f64> {
        let f = |p: Point3| (8. * std::f64::consts::PI * p.coordinates()[0]).cos();
        Ok(f(a) * f(b))
    }
}
#[test]
fn successive_agreement_is_explicitly_not_an_error_certificate() {
    let budget = MemoryBudget::new(4096);
    let cancel = Cancellation::default();
    let sites = [site([0.; 3], [1., 0., 0.], 1, 0)];
    let result = refine_covariance(
        &Oscillatory,
        &sites,
        RefinementOptions::default(),
        &budget,
        &cancel,
    )
    .unwrap();
    assert_eq!(result.report().stop, RefinementStop::Stable);
    // The exact average of cos(8 pi x) over [-1,1] is zero, hence variance zero.
    // Rules with 1, 2 and 4 midpoints all alias to variance one.
    close(
        Oscillatory
            .covariance(&result.sites()[0], &result.sites()[0])
            .unwrap(),
        1.,
        1e-14,
    );
    drop(result);
    assert_eq!(budget.used(), 0);
}

struct ExtremeRule;
impl Covariance for ExtremeRule {
    fn variables(&self) -> usize {
        1
    }
    fn point_covariance(&self, _: Point3, _: usize, _: Point3, _: usize) -> Result<f64> {
        unreachable!()
    }
    fn covariance(&self, a: &Site, b: &Site) -> Result<f64> {
        Ok(
            if a.support.center() == b.support.center()
                || a.support.node_count() > 1
                || b.support.node_count() > 1
            {
                f64::MAX
            } else {
                -f64::MAX
            },
        )
    }
}
#[test]
fn tolerance_comparison_handles_overflow_without_false_stability() {
    let sites = [
        site([0.; 3], [0.5, 0., 0.], 1, 0),
        site([2., 0., 0.], [0.5, 0., 0.], 1, 0),
    ];
    let budget = MemoryBudget::new(4096);
    let cancel = Cancellation::default();
    for (absolute_tolerance, relative_tolerance, expected) in [
        (0., 0.5, RefinementStop::RefinementLimit),
        (f64::MAX, 1., RefinementStop::Stable),
    ] {
        let result = refine_covariance(
            &ExtremeRule,
            &sites,
            RefinementOptions {
                absolute_tolerance,
                relative_tolerance,
                stable_comparisons: 1,
                maximum_refinements: 1,
                ..Default::default()
            },
            &budget,
            &cancel,
        )
        .unwrap();
        assert_eq!(result.report().stop, expected);
        assert_eq!(result.report().maximum_relative_change, Some(2.));
        assert_eq!(result.report().maximum_absolute_change, Some(f64::INFINITY));
    }
    assert_eq!(budget.used(), 0);
}
struct Fails(bool);
impl Covariance for Fails {
    fn variables(&self) -> usize {
        1
    }
    fn point_covariance(&self, _: Point3, _: usize, _: Point3, _: usize) -> Result<f64> {
        assert!(!self.0, "injected covariance panic");
        Err(Error::Numerical {
            context: "injected covariance failure",
        })
    }
}
#[test]
fn callback_errors_panics_empty_input_and_zero_levels_release_reservations() {
    use std::panic::{catch_unwind, AssertUnwindSafe};
    let budget = MemoryBudget::new(4096);
    let cancel = Cancellation::default();
    let sites = [site([0.; 3], [0.5, 0., 0.], 1, 0)];
    assert!(matches!(
        refine_covariance(
            &Fails(false),
            &sites,
            RefinementOptions::default(),
            &budget,
            &cancel
        ),
        Err(Error::Numerical {
            context: "injected covariance failure"
        })
    ));
    assert_eq!(budget.used(), 0);
    assert!(catch_unwind(AssertUnwindSafe(|| refine_covariance(
        &Fails(true),
        &sites,
        RefinementOptions::default(),
        &budget,
        &cancel
    )))
    .is_err());
    assert_eq!(budget.used(), 0);
    let empty = refine_covariance(
        &Fails(true),
        &[],
        RefinementOptions::default(),
        &budget,
        &cancel,
    )
    .unwrap();
    assert_eq!(empty.report().stop, RefinementStop::PointsOnly);
    assert_eq!(empty.accounted_bytes(), 0);
    drop(empty);
    let zero = refine_covariance(
        &Fails(true),
        &sites,
        RefinementOptions {
            maximum_refinements: 0,
            ..Default::default()
        },
        &budget,
        &cancel,
    )
    .unwrap();
    assert_eq!(zero.report().stop, RefinementStop::RefinementLimit);
    assert_eq!(zero.sites(), sites);
    drop(zero);
    let invalid = [site([0.; 3], [0.5, 0., 0.], 1, 1)];
    assert!(refine_covariance(
        &Fails(true),
        &invalid,
        RefinementOptions::default(),
        &budget,
        &cancel
    )
    .is_err());
    assert_eq!(budget.used(), 0);
}

#[test]
fn sparse_block_refinement_visits_only_affected_entries_with_fixed_buffers() {
    let count = 100_000_usize;
    let mut sites = vec![site([0.; 3], [0.; 3], 1, 0); count];
    sites[count / 2] = site([0.; 3], [0.5, 0., 0.], 1, 0);
    let model = Controlled {
        calls: AtomicUsize::new(0),
        cancel: None,
        nan: false,
    };
    let budget = MemoryBudget::new(32 << 20);
    let cancel = Cancellation::default();
    let allocations = allocation_counter::measure(|| {
        let result = refine_covariance(
            &model,
            &sites,
            RefinementOptions {
                maximum_refinements: 1,
                maximum_point_pairs: (6 * count - 1) as u64,
                ..Default::default()
            },
            &budget,
            &cancel,
        )
        .unwrap();
        assert_eq!(
            result.report().covariance_evaluations,
            (4 * count - 2) as u64
        );
        assert_eq!(
            result.report().estimated_point_pairs,
            (6 * count - 1) as u64
        );
        assert_eq!(result.report().stop, RefinementStop::RefinementLimit);
        assert_eq!(budget.used(), result.accounted_bytes());
    });
    assert_eq!(allocations.count_total, 2);
    assert_eq!(model.calls.load(Ordering::Relaxed), 4 * count - 2);
    assert_eq!(budget.used(), 0);
}
