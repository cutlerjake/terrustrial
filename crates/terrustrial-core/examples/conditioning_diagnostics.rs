//! Opt-in diagnostics of a prepared joint model, plus their isolated cost.
use std::{hint::black_box, time::Instant};
use terrustrial_core::{
    data::{Observation, Point3, Site},
    execution::{Cancellation, MemoryBudget},
    gaussian::{DiagnosticWorkspace, GaussianWorkspace, Ordinary},
    model::{Anisotropy, Coregionalization, Correlation, Lmc, Structure},
};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 3 {
        return Err("usage: conditioning_diagnostics OBSERVATIONS ITERATIONS".into());
    }
    let n: usize = args[1].parse()?;
    let iterations: usize = args[2].parse()?;
    if !(4..=512).contains(&n) || !(1..=1000000).contains(&iterations) {
        return Err("require 4<=observations<=512 and 1<=iterations<=1000000".into());
    }
    let budget = MemoryBudget::new(256 << 20);
    let cancel = Cancellation::default();
    let preparation = Instant::now();
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
    let mut solver = GaussianWorkspace::new(n, 1, 2, &budget)?;
    let drift = Ordinary { variables: 2 };
    let conditioned = solver.condition(&model, &data, &drift, &cancel)?;
    let mut workspace = DiagnosticWorkspace::new(n, &budget)?;
    let expected = conditioned.diagnostics(&mut workspace, &cancel)?;
    let preparation_seconds = preparation.elapsed().as_secs_f64();
    let start = Instant::now();
    for _ in 0..iterations {
        let result = conditioned.diagnostics(black_box(&mut workspace), &cancel)?;
        assert_eq!(result, expected);
        black_box(result);
    }
    let seconds = start.elapsed().as_secs_f64();
    let allocations = allocation_counter::measure(|| {
        assert_eq!(
            conditioned.diagnostics(&mut workspace, &cancel).unwrap(),
            expected
        );
    });
    let mut output = csv::Writer::from_writer(std::io::stdout().lock());
    output.write_record(["metric", "value"])?;
    for (key, value) in [
        ("observations", n.to_string()),
        ("iterations", iterations.to_string()),
        (
            "diagnostic_workspace_bytes",
            workspace.accounted_bytes().to_string(),
        ),
        ("total_reserved_bytes", budget.used().to_string()),
        (
            "covariance_rcond",
            expected
                .covariance
                .unwrap()
                .reciprocal_condition
                .to_string(),
        ),
        (
            "covariance_log10_condition",
            expected.covariance.unwrap().log10_condition.to_string(),
        ),
        (
            "drift_design_rcond",
            expected
                .drift_design
                .unwrap()
                .reciprocal_condition
                .to_string(),
        ),
        (
            "drift_normal_rcond",
            expected
                .drift_normal
                .unwrap()
                .reciprocal_condition
                .to_string(),
        ),
        ("drift_qr_applied", expected.drift_qr_applied.to_string()),
        (
            "diagnostic_allocations",
            allocations.count_total.to_string(),
        ),
        (
            "diagnostic_allocated_bytes",
            allocations.bytes_total.to_string(),
        ),
        (
            "diagnostic_peak_allocated_bytes",
            allocations.bytes_max.to_string(),
        ),
        (
            "diagnostic_retained_allocated_bytes",
            allocations.bytes_current.to_string(),
        ),
        ("preparation_seconds", preparation_seconds.to_string()),
        ("seconds", seconds.to_string()),
    ] {
        output.write_record([key, &value])?;
    }
    output.flush()?;
    Ok(())
}
