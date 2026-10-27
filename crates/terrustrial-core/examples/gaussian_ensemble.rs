//! Full-field SGS validation driver for scripts/validate_gaussian.py.
use rand::seq::SliceRandom;
use std::{
    fs::{self, OpenOptions},
    path::Path,
    time::Instant,
};
use terrustrial_core::{
    data::{Group, Observation, Point3, Site, Support, Target},
    execution::{random_stream, Cancellation, MemoryBudget},
    gaussian::Drift,
    model::{Anisotropy, Coregionalization, Correlation, Lmc, Structure},
    simulation::{self, parallel, SimulationOptions},
    spatial::{Neighborhood, SpatialIndex},
    Result,
};
type AuditResult<T> = std::result::Result<T, Box<dyn std::error::Error>>;
const FIRST_TARGET: u64 = 1000;

struct Basis(usize);
impl Drift for Basis {
    fn columns(&self) -> usize {
        self.0
    }
    fn evaluate(&self, site: &Site, output: &mut [f64]) -> Result<()> {
        output.fill(0.);
        if self.0 == 3 {
            output[site.variable] = 1.;
        }
        if self.0 == 6 {
            output[2 * site.variable] = 1.;
            // Deliberately nearly dependent. The independent reference uses
            // the equivalent stable basis [1,x], not this transformation.
            output[2 * site.variable + 1] =
                1. + 2_f64.powi(-20) * site.support.center().coordinates()[0];
        }
        Ok(())
    }
}
fn mean(site: &Site) -> f64 {
    site.support
        .nodes()
        .map(|p| {
            let x = p.coordinates()[0];
            (site.variable as f64 - 1.) * 0.7 + 0.05 * x * x + 0.1 * (site.variable + 1) as f64 * x
        })
        .sum::<f64>()
        / site.support.node_count() as f64
}
fn file(path: &Path) -> std::io::Result<std::fs::File> {
    OpenOptions::new().write(true).create_new(true).open(path)
}
fn main() -> AuditResult<()> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 11 {
        return Err("usage: gaussian_ensemble FIXTURE.csv METHOD ORDER UNITS ENGINE WORKERS BATCH REALIZATIONS OUTPUT_DIRECTORY SEED".into());
    }
    let method = args[2].as_str();
    let basis = Basis(match method {
        "simple" | "unconditional" => 0,
        "ordinary" => 3,
        "external" => 6,
        _ => return Err("unknown method".into()),
    });
    let roots = match args[4].as_str() {
        "unit" => [1.; 3],
        "scaled" => [2_f64.powi(-400), 1., 2_f64.powi(400)],
        _ => return Err("unknown units".into()),
    };
    let engine = args[5].as_str();
    if !["serial", "parallel", "serial-checked", "parallel-checked"].contains(&engine) {
        return Err("unknown engine".into());
    }
    let workers: usize = args[6].parse()?;
    let batch: usize = args[7].parse()?;
    let count: u64 = args[8].parse()?;
    let seed: u64 = args[10].parse()?;
    if !(1..=32).contains(&workers) || !(1..=128).contains(&batch) || !(1..=65536).contains(&count)
    {
        return Err("workers/batch/realizations out of audit limits".into());
    }
    let directory = Path::new(&args[9]);
    fs::create_dir(directory)?;
    let mut reader = csv::Reader::from_path(&args[1])?;
    let header = [
        "kind",
        "index",
        "group",
        "variable",
        "x",
        "half_width",
        "nodes",
        "value",
        "noise",
    ];
    if reader.headers()?.iter().ne(header) {
        return Err("fixture header mismatch".into());
    }
    let mut hard = Vec::new();
    let mut targets = Vec::new();
    let mut groups: Vec<Group> = Vec::new();
    for record in reader.records() {
        let row = record?;
        let index: usize = row[1].parse()?;
        let group: usize = row[2].parse()?;
        let variable: usize = row[3].parse()?;
        if variable >= 3 {
            return Err("fixture variable out of range".into());
        }
        let x: f64 = row[4].parse()?;
        let width: f64 = row[5].parse()?;
        let nodes: usize = row[6].parse()?;
        let center = Point3::new([x, 0., 0.])?;
        let site = Site {
            variable,
            support: if width == 0. {
                Support::point(center)
            } else {
                Support::block(center, [width, 0., 0.], [nodes, 1, 1])?
            },
        };
        if row[0] == *"hard" {
            if index != hard.len() || index >= FIRST_TARGET as usize {
                return Err("hard IDs must be consecutive and below target IDs".into());
            }
            hard.push(Observation::new(
                index as u64,
                site,
                row[7].parse::<f64>()? * roots[variable],
                row[8].parse::<f64>()? * roots[variable] * roots[variable],
            )?);
        } else if row[0] == *"target" {
            if index != targets.len() || group > groups.len() {
                return Err("target IDs/groups must be consecutive".into());
            }
            if group == groups.len() {
                groups.push(Group {
                    id: group as u64,
                    targets: Vec::new(),
                });
            }
            targets.push(site);
            groups[group].targets.push(Target {
                id: FIRST_TARGET + index as u64,
                site,
            });
        } else {
            return Err("unknown fixture row".into());
        }
    }
    if method == "unconditional" {
        hard.clear();
    }
    match args[3].as_str() {
        "forward" => (),
        "reverse" => groups.reverse(),
        "permuted" => groups.shuffle(&mut random_stream(7319, 0, 0, 91)),
        _ => return Err("unknown path order".into()),
    }
    for (rank, group) in groups.iter_mut().enumerate() {
        group.id = rank as u64;
    }
    let model = Lmc::new(vec![Structure {
        correlation: Correlation::Exponential,
        anisotropy: Anisotropy::isotropic(2.)?,
        coregionalization: Coregionalization::from_factor(
            3,
            3,
            &[
                roots[0],
                0.,
                0.,
                0.6 * roots[1],
                0.8 * roots[1],
                0.,
                -0.2 * roots[2],
                0.3 * roots[2],
                0.9 * roots[2],
            ],
        )?,
    }])?;
    let budget = MemoryBudget::new(2 << 30);
    let cancel = Cancellation::default();
    let known_mean = |s: &Site| Ok(mean(s) * roots[s.variable]);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .build()?;
    let mut values = csv::Writer::from_writer(file(&directory.join("values.csv"))?);
    values.write_record(
        std::iter::once("realization".to_string())
            .chain((0..targets.len()).map(|i| format!("target_{i}"))),
    )?;
    let maximum_targets = groups
        .iter()
        .map(|g| g.targets.len())
        .max()
        .ok_or("empty path")?;
    let mut maximum_residual = 0_f64;
    let mut checked_groups = 0_u64;
    let mut inspect = |_: &Group,
                       report: &terrustrial_core::gaussian::GroupResiduals|
     -> Result<()> {
        let r = report.equations;
        for block in [
            Some(r.covariance_equations),
            Some(r.drift_constraints),
            Some(r.original_drift_constraints),
            r.conditional_covariance,
            r.sampling_factor,
        ]
        .into_iter()
        .flatten()
        {
            maximum_residual = maximum_residual.max(block.max_componentwise_backward_error);
            if block.max_componentwise_backward_error > 1e-8 || block.scaled_product_underflows != 0
            {
                return Err(terrustrial_core::Error::Numerical {
                    context: "ensemble residual acceptance",
                });
            }
        }
        checked_groups += 1;
        Ok(())
    };
    let mut total_seconds = 0.;
    let mut checksum = 0_u64;
    for realization in 0..count {
        let mut output = vec![f64::NAN; targets.len()];
        let mut simulated = SpatialIndex::new(vec![])?;
        let options = SimulationOptions {
            seed,
            realization,
            mean: Some(&known_mean),
            hard_neighborhood: Neighborhood {
                maximum: hard.len().max(1),
                radius: 100.,
            }
            .into(),
            simulated_neighborhood: Neighborhood {
                maximum: targets.len(),
                radius: 100.,
            }
            .into(),
            budget: &budget,
            cancellation: &cancel,
        };
        let mut sink = |observations: &[Observation]| {
            for o in observations {
                output[(o.id() - FIRST_TARGET) as usize] = o.value();
            }
            Ok(())
        };
        let start = Instant::now();
        pool.install(|| {
            if engine == "serial-checked" {
                simulation::simulate_groups_with_residuals(
                    &model,
                    hard.as_slice(),
                    &mut simulated,
                    groups.clone(),
                    &basis,
                    options,
                    &mut inspect,
                    &mut sink,
                )
            } else if engine == "parallel-checked" {
                parallel::simulate_groups_with_residuals(
                    &model,
                    hard.as_slice(),
                    &mut simulated,
                    groups.clone(),
                    &basis,
                    options,
                    terrustrial_core::execution::ExecutionOptions {
                        batch_groups: batch,
                        maximum_targets_per_group: maximum_targets,
                        bytes_per_group: 16 << 20,
                    },
                    &mut inspect,
                    &mut sink,
                )
            } else if engine == "serial" {
                simulation::simulate_groups(
                    &model,
                    hard.as_slice(),
                    &mut simulated,
                    groups.clone(),
                    &basis,
                    options,
                    &mut sink,
                )
            } else {
                parallel::simulate_groups(
                    &model,
                    hard.as_slice(),
                    &mut simulated,
                    groups.clone(),
                    &basis,
                    options,
                    terrustrial_core::execution::ExecutionOptions {
                        batch_groups: batch,
                        maximum_targets_per_group: maximum_targets,
                        bytes_per_group: 16 << 20,
                    },
                    &mut sink,
                )
            }
        })?;
        total_seconds += start.elapsed().as_secs_f64();
        if budget.used() != 0 || output.iter().any(|v| !v.is_finite()) {
            return Err("incomplete output or leaked budget reservation".into());
        }
        for (i, site) in targets.iter().enumerate() {
            for observation in &hard {
                if observation.noise_variance() == 0.
                    && observation.site() == site
                    && observation.value().to_bits() != output[i].to_bits()
                {
                    return Err("exact hard-data value changed".into());
                }
            }
            for j in 0..i {
                if targets[j] == *site && output[i].to_bits() != output[j].to_bits() {
                    return Err("repeated-target value changed".into());
                }
            }
        }
        for &value in &output {
            checksum = checksum.rotate_left(7) ^ value.to_bits();
        }
        values.write_record(
            std::iter::once(realization.to_string())
                .chain(output.iter().map(|v| format!("{v:.17e}"))),
        )?;
    }
    values.flush()?;
    let mut diagnostics = csv::Writer::from_writer(file(&directory.join("diagnostics.csv"))?);
    diagnostics.write_record(["checked_groups", "maximum_residual"])?;
    diagnostics.write_record([checked_groups.to_string(), maximum_residual.to_string()])?;
    diagnostics.flush()?;
    let mut metrics = csv::Writer::from_writer(file(&directory.join("metrics.csv"))?);
    metrics.write_record(["metric", "value"])?;
    for (key, value) in [
        ("realizations", count.to_string()),
        ("targets", targets.len().to_string()),
        ("groups", groups.len().to_string()),
        ("seconds", total_seconds.to_string()),
        ("checksum", checksum.to_string()),
    ] {
        metrics.write_record([key, value.as_str()])?;
    }
    metrics.flush()?;
    Ok(())
}
