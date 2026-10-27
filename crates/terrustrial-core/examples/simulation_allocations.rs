//! Complete serial and parallel simulation allocation traffic on one Rayon worker.
//! Inputs and pool construction are excluded; the evolving store and driver are included.
use rand::seq::SliceRandom;
use std::time::Instant;
use terrustrial_core::{
    data::{Group, Observation, Point3, Site, Target},
    execution::{random_stream, Cancellation, ExecutionOptions, MemoryBudget},
    gaussian::{KnownMean, Simple},
    model::{Anisotropy, Coregionalization, Correlation, Lmc, Structure},
    simulation::{self, SimulationOptions},
    spatial::{Neighborhood, SpatialIndex},
    Result,
};
#[derive(Clone, Copy)]
enum Layout {
    Grid,
    Line,
    Shuffled,
}
impl Layout {
    fn name(self) -> &'static str {
        match self {
            Self::Grid => "grid",
            Self::Line => "line",
            Self::Shuffled => "shuffled",
        }
    }
}
fn groups(n: usize, layout: Layout) -> Vec<Group> {
    let mut positions: Vec<_> = (0..n).collect();
    if matches!(layout, Layout::Shuffled) {
        positions.shuffle(&mut random_stream(2026, 0, 0, 19));
    }
    positions
        .into_iter()
        .enumerate()
        .map(|(i, position)| Group {
            id: i as u64,
            targets: (0..2)
                .map(|v| Target {
                    id: 100 + 2 * i as u64 + v as u64,
                    site: Site::point(
                        Point3::new(match layout {
                            Layout::Line => [position as f64 * 0.3, 0., 0.],
                            Layout::Grid | Layout::Shuffled => [
                                (position % 32) as f64 * 0.3,
                                ((position / 32) % 32) as f64 * 0.3,
                                (position / 1024) as f64 * 0.3,
                            ],
                        })
                        .unwrap(),
                        v,
                    ),
                })
                .collect(),
        })
        .collect()
}
fn trend(site: &Site) -> Result<f64> {
    Ok(site.support.center().coordinates()[0] * 0.1 + site.variable as f64)
}
fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let n = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "2048".into())
        .parse::<usize>()?;
    if n == 0 || n > 1 << 20 {
        return Err("group count must be in 1..=2^20".into());
    }
    let batch_groups = std::env::args()
        .nth(3)
        .unwrap_or_else(|| "32".into())
        .parse::<usize>()?;
    if batch_groups == 0 || batch_groups > 256 {
        return Err("batch groups must be in 1..=256".into());
    }
    let layout = match std::env::args().nth(4).as_deref().unwrap_or("grid") {
        "grid" => Layout::Grid,
        "line" => Layout::Line,
        "shuffled" => Layout::Shuffled,
        _ => return Err("layout must be grid, line, or shuffled".into()),
    };
    let pool = rayon::ThreadPoolBuilder::new().num_threads(1).build()?;
    let model = Lmc::new(vec![Structure {
        correlation: Correlation::Exponential,
        anisotropy: Anisotropy::isotropic(3.)?,
        coregionalization: Coregionalization::from_factor(2, 2, &[1., 0., 0.6, 0.8])?,
    }])?;
    let hard = SpatialIndex::new(
        (0..4)
            .map(|i| {
                Observation::new(
                    i,
                    Site::point(
                        Point3::new([-1., (i / 2) as f64, 0.]).unwrap(),
                        i as usize % 2,
                    ),
                    i as f64 * 0.2,
                    0.,
                )
                .unwrap()
            })
            .collect(),
    )?;
    let budget = MemoryBudget::new(1 << 29);
    let cancel = Cancellation::default();
    let mut out = csv::Writer::from_writer(std::io::stdout().lock());
    out.write_record(["metric", "value"])?;
    out.write_record(["layout", layout.name()])?;
    out.write_record(["batch_groups", &batch_groups.to_string()])?;
    for with_mean in [false, true] {
        let mut reference = None;
        for parallel in [false, true] {
            let mean: Option<&dyn KnownMean> = with_mean.then_some(&trend);
            let run = |groups: Vec<Group>| -> Result<u64> {
                let mut simulated = SpatialIndex::new(vec![])?;
                let mut checksum = 0_u64;
                let options = SimulationOptions {
                    seed: 2026,
                    realization: 1,
                    mean,
                    hard_neighborhood: Neighborhood {
                        maximum: 4,
                        radius: 3.,
                    }
                    .into(),
                    simulated_neighborhood: Neighborhood {
                        maximum: 16,
                        radius: 3.,
                    }
                    .into(),
                    budget: &budget,
                    cancellation: &cancel,
                };
                let mut output = |group: &[Observation]| {
                    for o in group {
                        checksum = checksum.rotate_left(7) ^ o.value().to_bits();
                    }
                    Ok(())
                };
                if parallel {
                    simulation::parallel::simulate_groups(
                        &model,
                        &hard,
                        &mut simulated,
                        groups,
                        &Simple,
                        options,
                        ExecutionOptions {
                            batch_groups,
                            maximum_targets_per_group: 2,
                            bytes_per_group: 1 << 18,
                        },
                        &mut output,
                    )?;
                } else {
                    simulation::simulate_groups(
                        &model,
                        &hard,
                        &mut simulated,
                        groups,
                        &Simple,
                        options,
                        &mut output,
                    )?;
                }
                assert_eq!(budget.used(), 0);
                Ok(checksum)
            };
            pool.install(|| run(groups(64, layout)))?;
            let input = groups(n, layout);
            let (seconds, checksum) = pool.install(|| {
                let start = Instant::now();
                let result = run(input);
                (start.elapsed().as_secs_f64(), result)
            });
            let checksum = checksum?;
            if let Some(expected) = reference {
                assert_eq!(checksum, expected);
            } else {
                reference = Some(checksum);
            }
            let input = groups(n, layout);
            let (allocations, check) = pool.install(|| {
                let mut result = Ok(0);
                let measured = allocation_counter::measure(|| {
                    result = run(input);
                });
                (measured, result)
            });
            assert_eq!(checksum, check?);
            let prefix = format!("parallel_{parallel}_mean_{with_mean}");
            for (key, value) in [
                ("groups", n.to_string()),
                ("seconds", seconds.to_string()),
                ("checksum", checksum.to_string()),
                ("allocations", allocations.count_total.to_string()),
                ("allocated_bytes", allocations.bytes_total.to_string()),
            ] {
                out.write_record([format!("{prefix}_{key}"), value])?;
            }
        }
    }
    out.flush()?;
    Ok(())
}
