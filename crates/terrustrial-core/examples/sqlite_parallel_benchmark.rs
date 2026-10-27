//! Joint estimation through a bounded read pool, with fixed total page-cache budget.
use std::time::Instant;
use terrustrial_core::{
    data::{Group, Point3, Site, Target},
    estimation::{estimate_groups, EstimationOptions},
    execution::{Cancellation, ExecutionOptions, MemoryBudget},
    gaussian::{NumericalTolerance, Ordinary, Uncertainty},
    model::{Anisotropy, Coregionalization, Correlation, Lmc, Structure},
    spatial::Neighborhood,
    storage::SqliteReadPool,
};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if !(3..=7).contains(&args.len()) {
        return Err("usage: sqlite_parallel_benchmark DATABASE GROUPS [ORDER_OFFSET [RADIUS [MAXIMUM [regular|mixed]]]]".into());
    }
    let n: usize = args[2].parse()?;
    if n == 0 || n > 1 << 20 {
        return Err("groups must be in 1..=2^20".into());
    }
    let offset: usize = args.get(3).map(|v| v.parse()).transpose()?.unwrap_or(0);
    let radius: f64 = args.get(4).map(|v| v.parse()).transpose()?.unwrap_or(3.);
    let maximum: usize = args.get(5).map(|v| v.parse()).transpose()?.unwrap_or(32);
    let mixed = match args.get(6).map(String::as_str).unwrap_or("regular") {
        "regular" => false,
        "mixed" => true,
        _ => return Err("layout must be regular or mixed".into()),
    };
    let neighborhood = Neighborhood { maximum, radius };
    neighborhood.validate()?;
    let mut cases = [(1, 1), (4, 1), (4, 4)];
    cases.rotate_left(offset % 3);
    let budget = MemoryBudget::new(512 << 20);
    let cancel = Cancellation::default();
    let store = SqliteReadPool::open(&args[1], 1, 4096, &budget)?;
    let rows = store.len()?;
    drop(store);
    let model = Lmc::new(vec![Structure {
        correlation: Correlation::Exponential,
        anisotropy: Anisotropy::isotropic(6.)?,
        coregionalization: Coregionalization::scalar(1.)?,
    }])?;
    let groups: Vec<_> = (0..n)
        .map(|i| Group {
            id: i as u64,
            targets: (0..2)
                .map(|j| Target {
                    id: (i * 2 + j) as u64,
                    site: Site::point(
                        Point3::new([
                            (if mixed && i % 2 == 1 {
                                -2.6
                            } else {
                                ((i * 71) % 150) as f64
                            }) + 0.25
                                + j as f64 * 0.2,
                            ((i * 113) % 200) as f64 + 0.5,
                            ((i * 137) % 200) as f64 + 0.125,
                        ])
                        .unwrap(),
                        0,
                    ),
                })
                .collect(),
        })
        .collect();
    let mut csv = csv::Writer::from_writer(std::io::stdout().lock());
    csv.write_record(["metric", "value"])?;
    csv.write_record(["rows", &rows.to_string()])?;
    csv.write_record(["groups", &n.to_string()])?;
    csv.write_record(["order_offset", &(offset % 3).to_string()])?;
    csv.write_record(["radius", &radius.to_string()])?;
    csv.write_record(["maximum", &maximum.to_string()])?;
    csv.write_record(["mixed", &mixed.to_string()])?;
    let mut expected = None;
    for (workers, readers) in cases {
        let threads = rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()?;
        let opening = Instant::now();
        let pool = SqliteReadPool::open(&args[1], readers, (32 << 20) / readers as u64, &budget)?;
        let open_seconds = opening.elapsed().as_secs_f64();
        let options = || EstimationOptions {
            neighborhood: neighborhood.into(),
            mean: None,
            uncertainty: Uncertainty::Joint,
            tolerance: NumericalTolerance::default(),
            budget: &budget,
            cancellation: &cancel,
        };
        let execution = ExecutionOptions {
            batch_groups: 64,
            maximum_targets_per_group: 2,
            bytes_per_group: 4 << 20,
        };
        let run = |groups| {
            let mut checksum = 0u64;
            estimate_groups(
                &model,
                &pool,
                groups,
                &Ordinary { variables: 1 },
                options(),
                execution,
                |out| {
                    for &value in &out.values {
                        checksum = checksum.rotate_left(7) ^ value.to_bits();
                    }
                    let c = out.covariance().unwrap();
                    for j in 0..c.ncols() {
                        for i in 0..c.nrows() {
                            checksum = checksum.rotate_left(7) ^ c[(i, j)].to_bits();
                        }
                    }
                    Ok(())
                },
            )?;
            terrustrial_core::Result::Ok(checksum)
        };
        threads.install(|| run(groups[..n.min(64)].to_vec()))?;
        for repeat in 0..3 {
            let input = groups.clone(); // Exclude common input construction from timing.
            let start = Instant::now();
            let checksum = threads.install(|| run(input))?;
            let seconds = start.elapsed().as_secs_f64();
            if let Some(reference) = expected {
                assert_eq!(checksum, reference);
            } else {
                expected = Some(checksum);
            }
            let prefix = format!("workers_{workers}_readers_{readers}_run_{repeat}");
            csv.write_record([format!("{prefix}_seconds"), seconds.to_string()])?;
            csv.write_record([format!("{prefix}_checksum"), checksum.to_string()])?;
        }
        csv.write_record([
            format!("workers_{workers}_readers_{readers}_open_seconds"),
            open_seconds.to_string(),
        ])?;
        csv.write_record([
            format!("workers_{workers}_readers_{readers}_cache_and_slots_bytes"),
            pool.accounted_bytes().to_string(),
        ])?;
        assert_eq!(budget.used(), pool.accounted_bytes());
        drop(pool);
        assert_eq!(budget.used(), 0);
    }
    csv.flush()?;
    Ok(())
}
