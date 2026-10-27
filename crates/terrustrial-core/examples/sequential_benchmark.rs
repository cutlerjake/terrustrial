//! End-to-end joint SGS, including neighborhood queries and evolving store writes.
use std::time::Instant;
use terrustrial_core::{
    data::{Group, Observation, Point3, Site, Target},
    execution::{Cancellation, MemoryBudget},
    gaussian::{Drift, KnownMean, Ordinary, Simple},
    model::{Anisotropy, Coregionalization, Correlation, Lmc, Structure},
    simulation::{simulate_groups, SimulationOptions},
    spatial::{Neighborhood, SpatialIndex},
    Result,
};
enum BenchmarkDrift {
    Simple,
    Ordinary,
}
impl Drift for BenchmarkDrift {
    fn columns(&self) -> usize {
        match self {
            Self::Simple => Simple.columns(),
            Self::Ordinary => Ordinary { variables: 2 }.columns(),
        }
    }
    fn evaluate(&self, site: &Site, values: &mut [f64]) -> Result<()> {
        match self {
            Self::Simple => Simple.evaluate(site, values),
            Self::Ordinary => Ordinary { variables: 2 }.evaluate(site, values),
        }
    }
}
fn trend(site: &Site) -> Result<f64> {
    let [x, y, z] = site.support.center().coordinates();
    Ok(if site.variable == 0 {
        1.0 + 0.002 * x * x + 0.1 * y.sin() + 0.05 * z
    } else {
        -2.0 + 0.01 * x - 0.003 * y * y - 0.1 * z
    })
}
fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    let zero = |_: &Site| Ok(0.0);
    let mean: Option<&dyn KnownMean> = match args.get(1).map(String::as_str) {
        Some("core-sequential" | "core-sequential-parallel") => None,
        Some("core-sequential-zero") => Some(&zero),
        Some("core-sequential-trend") => Some(&trend),
        _ => return Err("usage: sequential_benchmark core-sequential|core-sequential-parallel|core-sequential-zero|core-sequential-trend GROUPS [BATCH_GROUPS] [simple|ordinary]".into()),
    };
    let count: u64 = args.get(2).ok_or("group count required")?.parse()?;
    if count == 0 || count > 1 << 40 {
        return Err("group count must be in 1..=2^40".into());
    }
    let batch_groups = args
        .get(3)
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(32);
    if batch_groups == 0 {
        return Err("batch groups must be positive".into());
    }
    let drift_name = args.get(4).map_or("simple", String::as_str);
    let drift = match drift_name {
        "simple" => BenchmarkDrift::Simple,
        "ordinary" => BenchmarkDrift::Ordinary,
        _ => return Err("drift must be simple or ordinary".into()),
    };
    let dump = match args.get(5).map(String::as_str) {
        None => false,
        Some("dump") if args.len() == 6 && count <= 1_000_000 => true,
        _ => return Err("optional final mode is dump, limited to 1000000 groups".into()),
    };
    // Full output comparison is a separate untimed preflight. Ordinary timing
    // mode retains no extra draws and does not write inside the output callback.
    let mut draws = if dump {
        Vec::with_capacity(2 * count as usize)
    } else {
        Vec::new()
    };
    let model = Lmc::new(vec![Structure {
        correlation: Correlation::Exponential,
        anisotropy: Anisotropy::isotropic(3.0)?,
        coregionalization: Coregionalization::from_factor(2, 2, &[1.0, 0.0, 0.6, 0.8])?,
    }])?;
    let mut hard = Vec::new();
    for i in 0..4 {
        let site = Site::point(Point3::new([-1.0, (i / 2) as f64, 0.0])?, i % 2);
        let shift = mean.map(|m| m.evaluate(&site)).transpose()?.unwrap_or(0.0);
        hard.push(Observation::new(
            i as u64,
            site,
            shift + 0.25 * i as f64,
            0.0,
        )?);
    }
    let budget = MemoryBudget::new(1 << 24);
    let cancel = Cancellation::default();
    let mut simulated = SpatialIndex::new(vec![])?;
    let groups = (0..count).map(|i| Group {
        id: i,
        targets: (0..2)
            .map(|variable| Target {
                id: 100 + 2 * i + variable as u64,
                site: Site::point(
                    Point3::new([
                        (i % 100) as f64,
                        ((i / 100) % 100) as f64,
                        (i / 10000) as f64,
                    ])
                    .unwrap(),
                    variable,
                ),
            })
            .collect(),
    });
    let mut checksum = 0_u64;
    let start = Instant::now();
    let options = SimulationOptions {
        seed: 2026,
        realization: 1,
        mean,
        hard_neighborhood: Neighborhood {
            maximum: 4,
            radius: 8.0,
        }
        .into(),
        simulated_neighborhood: Neighborhood {
            maximum: 32,
            radius: 8.0,
        }
        .into(),
        budget: &budget,
        cancellation: &cancel,
    };
    let mut collect = |group: &[Observation]| {
        for observation in group {
            checksum = checksum.rotate_left(7) ^ observation.value().to_bits();
            if dump {
                draws.push(observation.value());
            }
        }
        Ok(())
    };
    if args[1] == "core-sequential-parallel" {
        terrustrial_core::simulation::parallel::simulate_groups(
            &model,
            hard.as_slice(),
            &mut simulated,
            groups,
            &drift,
            options,
            terrustrial_core::execution::ExecutionOptions {
                batch_groups,
                maximum_targets_per_group: 2,
                bytes_per_group: 1 << 18,
            },
            &mut collect,
        )?;
    } else {
        simulate_groups(
            &model,
            hard.as_slice(),
            &mut simulated,
            groups,
            &drift,
            options,
            &mut collect,
        )?;
    }
    let seconds = start.elapsed().as_secs_f64();
    assert_eq!(budget.used(), 0);
    let mut output = csv::Writer::from_writer(std::io::stdout().lock());
    output.write_record(["metric", "value"])?;
    if dump {
        output.write_record(["groups", &count.to_string()])?;
        output.write_record(["drift", drift_name])?;
        for (index, value) in draws.iter().enumerate() {
            output.write_record([format!("draw_{index}"), value.to_string()])?;
        }
        output.flush()?;
        return Ok(());
    }
    for (key, value) in [
        ("groups", count.to_string()),
        ("drift", drift_name.into()),
        (
            "batch_groups",
            if args[1] == "core-sequential-parallel" {
                batch_groups
            } else {
                1
            }
            .to_string(),
        ),
        ("variables_per_group", "2".into()),
        ("hard_observations", "4".into()),
        ("maximum_simulated_neighbors", "32".into()),
        ("checksum", checksum.to_string()),
        ("seconds", seconds.to_string()),
        ("groups_per_second", (count as f64 / seconds).to_string()),
    ] {
        output.write_record([key, &value])?;
    }
    output.flush()?;
    Ok(())
}
