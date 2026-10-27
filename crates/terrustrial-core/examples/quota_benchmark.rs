//! Fixed synthetic quota selection workloads. Setup and full ID checks are untimed.
use std::{hint::black_box, time::Instant};
use terrustrial_core::{
    data::{Observation, Point3, Site},
    execution::{Cancellation, MemoryBudget},
    model::Anisotropy,
    spatial::{
        MinimumQuotas, ObservationSource, SearchNeighborhood, SearchWorkspace, VariableLimit,
    },
};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() < 3 || args.len() > 4 || !["full", "trap"].contains(&args[1].as_str()) {
        return Err("usage: quota_benchmark full|trap ITERATIONS [dump]".into());
    }
    let iterations: usize = args[2].parse()?;
    if iterations == 0 || args.get(3).is_some_and(|s| s != "dump") {
        return Err("invalid arguments".into());
    }
    let full = args[1] == "full";
    let rows: Vec<_> = (0..96)
        .map(|i| {
            let (variable, x) = if full {
                ((i % 2) as usize, (i as f64 + 1.) * 0.01)
            } else {
                (
                    usize::from(i / 32 == 1),
                    (i as f64 + 1.) * 0.01 * if i / 32 == 2 { -1. } else { 1. },
                )
            };
            Observation::new(
                i,
                Site::point(Point3::new([x, 0., 0.]).unwrap(), variable),
                i as f64,
                0.,
            )
            .unwrap()
        })
        .collect();
    let maximum = if full { 32 } else { 2 };
    let query = SearchNeighborhood::new(
        maximum,
        Anisotropy::isotropic(2.)?,
        if full { None } else { Some(1) },
        (0..2)
            .map(|variable| VariableLimit {
                variable,
                maximum: maximum / 2,
            })
            .collect(),
    )?
    .with_minimums(MinimumQuotas {
        total: maximum,
        ..Default::default()
    })?;
    let expected: Vec<u64> = if full {
        (0..32).collect()
    } else {
        vec![32, 64]
    };
    let budget = MemoryBudget::new(1 << 20);
    let cancel = Cancellation::default();
    let point = Point3::new([0.; 3])?;
    let mut workspace = SearchWorkspace::new(&query, &budget)?;
    let found = rows.search(point, &mut workspace, &cancel)?;
    assert_eq!(
        found.iter().map(Observation::id).collect::<Vec<_>>(),
        expected
    );
    let mut writer = csv::Writer::from_writer(std::io::stdout().lock());
    writer.write_record(["metric", "value"])?;
    if args.len() == 4 {
        for (i, row) in found.iter().enumerate() {
            writer.write_record([format!("selected_{i}"), row.id().to_string()])?;
        }
    } else {
        let start = Instant::now();
        let mut checksum = 0_u64;
        for _ in 0..iterations {
            for row in black_box(rows.as_slice()).search(point, &mut workspace, &cancel)? {
                checksum = checksum.rotate_left(7) ^ black_box(row.id());
            }
        }
        let seconds = start.elapsed().as_secs_f64();
        let allocations = allocation_counter::measure(|| {
            for _ in 0..32 {
                black_box(rows.search(point, &mut workspace, &cancel).unwrap());
            }
        });
        assert_eq!(allocations.count_total, 0);
        for (key, value) in [
            ("iterations", iterations.to_string()),
            ("checksum", checksum.to_string()),
            ("seconds", seconds.to_string()),
            ("workspace_bytes", workspace.accounted_bytes().to_string()),
            ("query_allocations", allocations.count_total.to_string()),
        ] {
            writer.write_record([key, &value])?;
        }
    }
    writer.flush()?;
    Ok(())
}
