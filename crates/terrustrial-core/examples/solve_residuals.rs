//! Cost of reused target preparation with optional residual checks enabled.
use std::{hint::black_box, time::Instant};
use terrustrial_core::{
    data::{Observation, Point3, Site},
    execution::{Cancellation, MemoryBudget},
    gaussian::{
        GaussianWorkspace, NumericalTolerance, Ordinary, ResidualWorkspace, Uncertainty,
        WorkspaceOptions,
    },
    model::{Anisotropy, Coregionalization, Correlation, Lmc, Structure},
};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 4 {
        return Err("usage: solve_residuals OBSERVATIONS TARGETS ITERATIONS".into());
    }
    let n: usize = args[1].parse()?;
    let m: usize = args[2].parse()?;
    let iterations: usize = args[3].parse()?;
    if !(4..=512).contains(&n) || !(1..=256).contains(&m) || !(1..=1000000).contains(&iterations) {
        return Err("require observations 4..512, targets 1..256, iterations 1..1000000".into());
    }
    let budget = MemoryBudget::new(256 << 20);
    let cancel = Cancellation::default();
    let start = Instant::now();
    let model = Lmc::new(vec![Structure {
        correlation: Correlation::Exponential,
        anisotropy: Anisotropy::isotropic(3.)?,
        coregionalization: Coregionalization::from_factor(2, 2, &[1., 0., -0.3, 0.8])?,
    }])?;
    let data: Vec<_> = (0..n)
        .map(|i| {
            Observation::new(
                i as u64,
                Site::point(
                    Point3::new([
                        (i / 2 % 13) as f64 * 0.2,
                        (i / 26 % 11) as f64 * 0.2,
                        (i / 286) as f64 * 0.3,
                    ])
                    .unwrap(),
                    i % 2,
                ),
                i as f64,
                0.01,
            )
        })
        .collect::<Result<_, _>>()?;
    let targets: Vec<_> = (0..m)
        .map(|i| {
            Site::point(
                Point3::new([
                    0.13 + (i % 13) as f64 * 0.2,
                    0.17 + (i / 13) as f64 * 0.1,
                    0.07,
                ])
                .unwrap(),
                i % 2,
            )
        })
        .collect();
    let mut solver = GaussianWorkspace::new(n, m, 2, &budget)?;
    let drift = Ordinary { variables: 2 };
    let mut conditioned = solver.condition(&model, &data, &drift, &cancel)?;
    let mut workspace = ResidualWorkspace::new(n, m, &budget)?;
    let options = || WorkspaceOptions {
        uncertainty: Uncertainty::Joint,
        tolerance: NumericalTolerance::default(),
        cancellation: &cancel,
    };
    let (plan, expected) =
        conditioned.prepare_with_residuals(&targets, &mut workspace, options())?;
    let weights: Vec<_> = plan
        .weights()
        .col_iter()
        .flat_map(|c| c.iter().map(|v| v.to_bits()))
        .collect();
    let covariance: Vec<_> = plan
        .covariance()
        .unwrap()
        .col_iter()
        .flat_map(|c| c.iter().map(|v| v.to_bits()))
        .collect();
    assert!(
        expected
            .covariance_equations
            .max_componentwise_backward_error
            < 1e-10
    );
    assert!(expected.drift_constraints.max_componentwise_backward_error < 1e-10);
    let preparation_seconds = start.elapsed().as_secs_f64();
    let mut run = || {
        let (plan, report) = conditioned
            .prepare_with_residuals(black_box(&targets), &mut workspace, options())
            .unwrap();
        assert_eq!(report, expected);
        assert!(plan
            .weights()
            .col_iter()
            .flat_map(|c| c.iter().map(|v| v.to_bits()))
            .eq(weights.iter().copied()));
        assert!(plan
            .covariance()
            .unwrap()
            .col_iter()
            .flat_map(|c| c.iter().map(|v| v.to_bits()))
            .eq(covariance.iter().copied()));
        black_box(report);
    };
    let start = Instant::now();
    for _ in 0..iterations {
        run();
    }
    let seconds = start.elapsed().as_secs_f64();
    let allocations = allocation_counter::measure(run);
    let mut out = csv::Writer::from_writer(std::io::stdout().lock());
    out.write_record(["metric", "value"])?;
    for (key, value) in [
        ("observations", n.to_string()),
        ("targets", m.to_string()),
        ("iterations", iterations.to_string()),
        (
            "residual_workspace_bytes",
            workspace.accounted_bytes().to_string(),
        ),
        ("total_reserved_bytes", budget.used().to_string()),
        (
            "covariance_backward_error",
            expected
                .covariance_equations
                .max_componentwise_backward_error
                .to_string(),
        ),
        (
            "drift_backward_error",
            expected
                .drift_constraints
                .max_componentwise_backward_error
                .to_string(),
        ),
        (
            "scaled_product_underflows",
            (expected.covariance_equations.scaled_product_underflows
                + expected.drift_constraints.scaled_product_underflows)
                .to_string(),
        ),
        (
            "prepare_check_allocations",
            allocations.count_total.to_string(),
        ),
        (
            "prepare_check_allocated_bytes",
            allocations.bytes_total.to_string(),
        ),
        (
            "prepare_check_peak_bytes",
            allocations.bytes_max.to_string(),
        ),
        (
            "prepare_check_retained_bytes",
            allocations.bytes_current.to_string(),
        ),
        ("preparation_seconds", preparation_seconds.to_string()),
        ("seconds", seconds.to_string()),
    ] {
        out.write_record([key, &value])?;
    }
    out.flush()?;
    Ok(())
}
