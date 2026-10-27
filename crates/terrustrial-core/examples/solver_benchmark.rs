use std::{hint::black_box, time::Instant};
use terrustrial_core::{
    data::{Observation, Point3, Site},
    execution::{Cancellation, MemoryBudget},
    gaussian::{
        GaussianPlan, GaussianWorkspace, NumericalTolerance, PrepareOptions, Simple, Uncertainty,
        WorkspaceOptions,
    },
    model::{Anisotropy, Coregionalization, Correlation, Lmc, Structure},
};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mode = std::env::args().nth(1).unwrap_or_default();
    let reuse = mode.ends_with("-workspace");
    let uncertainty = match mode.as_str() {
        "core-estimation" | "core-estimation-workspace" => Uncertainty::None,
        "core-joint" | "core-joint-workspace" => Uncertainty::Joint,
        _ => return Err("usage: solver_benchmark core-estimation|core-joint ITERATIONS".into()),
    };
    let iterations: usize = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "10000".into())
        .parse()?;
    if iterations == 0 {
        return Err("positive iterations required".into());
    }
    let observations: Vec<_> = (0..32)
        .map(|i| {
            Observation::new(
                i,
                Site::point(
                    Point3::new([(i % 4) as f64, ((i / 4) % 4) as f64, (i / 16) as f64]).unwrap(),
                    0,
                ),
                i as f64 / 32.0,
                0.0,
            )
            .unwrap()
        })
        .collect();
    let targets: Vec<_> = (0..8)
        .map(|i| {
            Site::point(
                Point3::new([
                    0.25 + (i % 2) as f64,
                    0.25 + ((i / 2) % 2) as f64,
                    0.25 + (i / 4) as f64,
                ])
                .unwrap(),
                0,
            )
        })
        .collect();
    let values: Vec<_> = observations.iter().map(Observation::value).collect();
    let model = Lmc::new(vec![Structure {
        correlation: Correlation::Spherical,
        anisotropy: Anisotropy::isotropic(8.0)?,
        coregionalization: Coregionalization::scalar(1.7)?,
    }])?;
    let budget = MemoryBudget::new(1 << 24);
    let cancel = Cancellation::default();
    {
        let mut workspace = if reuse {
            Some(GaussianWorkspace::new(32, 8, 0, &budget)?)
        } else {
            None
        };
        let mut run = || -> terrustrial_core::Result<[f64; 8]> {
            let mut out = [0.0; 8];
            if let Some(workspace) = &mut workspace {
                let plan = workspace.prepare(
                    &model,
                    &observations,
                    &targets,
                    &Simple,
                    WorkspaceOptions {
                        uncertainty,
                        tolerance: NumericalTolerance::default(),
                        cancellation: &cancel,
                    },
                )?;
                plan.estimate_into(black_box(&values), &mut out)?;
            } else {
                let plan = GaussianPlan::prepare(
                    &model,
                    &observations,
                    &targets,
                    &Simple,
                    PrepareOptions {
                        uncertainty,
                        tolerance: NumericalTolerance::default(),
                        budget: &budget,
                        cancellation: &cancel,
                    },
                )?;
                plan.estimate_into(black_box(&values), &mut out)?;
            }
            Ok(out)
        };
        for _ in 0..10 {
            black_box(run()?);
        }
        let start = Instant::now();
        let mut checksum = 0.0;
        for _ in 0..iterations {
            checksum += black_box(run()?).iter().sum::<f64>();
        }
        let seconds = start.elapsed().as_secs_f64();
        let mut output = csv::Writer::from_writer(std::io::stdout().lock());
        output.write_record(["metric", "value"])?;
        for (key, value) in [
            ("mode", format!("{uncertainty:?}")),
            ("reuse_workspace", reuse.to_string()),
            ("iterations", iterations.to_string()),
            ("conditioning_points", "32".into()),
            ("targets_per_system", "8".into()),
            ("seconds", seconds.to_string()),
            (
                "systems_per_second",
                (iterations as f64 / seconds).to_string(),
            ),
            ("checksum", checksum.to_string()),
        ] {
            output.write_record([key, &value])?;
        }
        output.flush()?;
    }
    Ok(())
}
