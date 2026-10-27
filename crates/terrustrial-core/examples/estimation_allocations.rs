//! Complete estimation-call allocation counts on one warmed Rayon worker.
//! Input construction is excluded; driver arenas, scheduling and outputs included.
use std::time::Instant;
use terrustrial_core::{
    data::{Group, Observation, Point3, Site, Target},
    estimation::{estimate_groups, EstimationOptions},
    execution::{Cancellation, ExecutionOptions, MemoryBudget},
    gaussian::{KnownMean, NumericalTolerance, Ordinary, Uncertainty},
    model::{Anisotropy, Coregionalization, Correlation, Lmc, Structure},
    spatial::{Neighborhood, SpatialIndex},
    Result,
};
fn groups(count: usize) -> Vec<Group> {
    (0..count)
        .map(|i| Group {
            id: i as u64,
            targets: (0..4)
                .map(|j| Target {
                    id: (100 + i * 4 + j) as u64,
                    site: Site::point(
                        Point3::new([
                            0.13 + (i % 13) as f64 * 0.1 + (j / 2) as f64 * 0.2,
                            0.17 + ((i / 13) % 7) as f64 * 0.1,
                            0.31,
                        ])
                        .unwrap(),
                        j % 2,
                    ),
                })
                .collect(),
        })
        .collect()
}
fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let count: usize = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "8192".into())
        .parse()?;
    if count == 0 || count > 1 << 24 {
        return Err("group count must be in 1..=2^24".into());
    }
    let pool = rayon::ThreadPoolBuilder::new().num_threads(1).build()?;
    let model = Lmc::new(vec![Structure {
        correlation: Correlation::Exponential,
        anisotropy: Anisotropy::isotropic(3.)?,
        coregionalization: Coregionalization::from_factor(2, 2, &[1., 0., 0.6, 0.8])?,
    }])?;
    let hard = SpatialIndex::new(
        (0..64)
            .map(|i| {
                Observation::new(
                    i,
                    Site::point(
                        Point3::new([(i / 2 % 4) as f64, (i / 8 % 4) as f64, (i / 32) as f64])
                            .unwrap(),
                        (i % 2) as usize,
                    ),
                    (i as f64 * 0.3).sin(),
                    0.01,
                )
                .unwrap()
            })
            .collect(),
    )?;
    let budget = MemoryBudget::new(1 << 28);
    let cancel = Cancellation::default();
    let trend =
        |site: &Site| Ok(site.support.center().coordinates()[0].powi(2) + site.variable as f64);
    let mut output = csv::Writer::from_writer(std::io::stdout().lock());
    output.write_record(["metric", "value"])?;
    for uncertainty in [Uncertainty::None, Uncertainty::Joint] {
        for known_mean in [false, true] {
            let mean: Option<&dyn KnownMean> = known_mean.then_some(&trend);
            let run = |groups: Vec<Group>| -> Result<u64> {
                let mut checksum = 0_u64;
                estimate_groups(
                    &model,
                    &hard,
                    groups,
                    &Ordinary { variables: 2 },
                    EstimationOptions {
                        neighborhood: Neighborhood {
                            maximum: 64,
                            radius: 100.,
                        }
                        .into(),
                        mean,
                        uncertainty,
                        tolerance: NumericalTolerance::default(),
                        budget: &budget,
                        cancellation: &cancel,
                    },
                    ExecutionOptions {
                        batch_groups: 64,
                        maximum_targets_per_group: 4,
                        bytes_per_group: 1 << 20,
                    },
                    |result| {
                        for value in &result.values {
                            checksum = checksum.rotate_left(7) ^ value.to_bits();
                        }
                        if let Some(cov) = result.covariance() {
                            for j in 0..cov.ncols() {
                                for i in 0..cov.nrows() {
                                    checksum = checksum.rotate_left(7) ^ cov[(i, j)].to_bits();
                                }
                            }
                        }
                        checksum ^= result.roundoff_corrections() as u64;
                        Ok(())
                    },
                )?;
                assert_eq!(budget.used(), 0);
                Ok(checksum)
            };
            pool.install(|| run(groups(16)))?;
            let timed = groups(count);
            let (seconds, checksum) = pool.install(|| {
                let start = Instant::now();
                let result = run(timed);
                (start.elapsed().as_secs_f64(), result)
            });
            let checksum = checksum?;
            let measured = groups(count);
            let (allocations, measured_checksum) = pool.install(|| {
                let mut result = Ok(0);
                let allocations = allocation_counter::measure(|| result = run(measured));
                (allocations, result)
            });
            assert_eq!(checksum, measured_checksum?);
            let prefix = format!("{uncertainty:?}_mean_{known_mean}");
            for (key, value) in [
                ("groups", count.to_string()),
                ("seconds", seconds.to_string()),
                ("checksum", checksum.to_string()),
                ("allocations", allocations.count_total.to_string()),
                ("allocated_bytes", allocations.bytes_total.to_string()),
            ] {
                output.write_record([format!("{prefix}_{key}"), value])?;
            }
        }
    }
    output.flush()?;
    Ok(())
}
