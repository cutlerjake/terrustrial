//! Same-build allocated best-first versus reusable near-first spatial queries.
use rand::seq::SliceRandom;
use std::{hint::black_box, time::Instant};
use terrustrial_core::{
    data::{Observation, Point3, Site},
    execution::{random_stream, Cancellation, MemoryBudget},
    spatial::{NearestWorkspace, Neighborhood, ObservationSink, ObservationSource, SpatialIndex},
    Result,
};
fn checksum(found: &[Observation]) -> u64 {
    found
        .iter()
        .fold(0_u64, |sum, o| sum.rotate_left(7) ^ o.id())
}
fn run(
    points: &[Point3],
    iterations: usize,
    mut query: impl FnMut(Point3) -> Result<u64>,
) -> Result<(f64, u64)> {
    for &p in points.iter().take(10) {
        black_box(query(p)?);
    }
    let start = Instant::now();
    let mut sum = 0_u64;
    for i in 0..iterations {
        sum = sum.wrapping_add(black_box(query(black_box(points[i % points.len()]))?));
    }
    Ok((start.elapsed().as_secs_f64(), sum))
}
fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let iterations: usize = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "10000".into())
        .parse()?;
    if iterations == 0 {
        return Err("positive iterations required".into());
    }
    let layout = std::env::args().nth(3).unwrap_or_else(|| "grid".into());
    if !matches!(layout.as_str(), "grid" | "shuffled") {
        return Err("layout must be grid or shuffled".into());
    }
    // Optional filters isolate phases for profiling; ordinary comparisons retain
    // both construction paths and both query APIs.
    let construction_filter = std::env::args().nth(4).unwrap_or_else(|| "all".into());
    let backend_filter = std::env::args().nth(5).unwrap_or_else(|| "all".into());
    if !matches!(construction_filter.as_str(), "all" | "bulk" | "incremental")
        || !matches!(backend_filter.as_str(), "all" | "owned" | "reused")
    {
        return Err("invalid construction or query filter".into());
    }
    let mut out = csv::Writer::from_writer(std::io::stdout().lock());
    out.write_record(["metric", "value"])?;
    out.write_record(["insertion_order", &layout])?;
    for construction in ["bulk", "incremental"] {
        if construction_filter != "all" && construction_filter != construction {
            continue;
        }
        let mut records: Vec<_> = (0..100000)
            .map(|i| {
                Observation::new(
                    i,
                    Site::point(
                        Point3::new([
                            (i % 73) as f64 * 0.3,
                            ((i / 73) % 61) as f64 * 0.3,
                            (i / 4453) as f64 * 0.3,
                        ])
                        .unwrap(),
                        0,
                    ),
                    0.,
                    0.,
                )
                .unwrap()
            })
            .collect();
        if layout == "shuffled" {
            records.shuffle(&mut random_stream(2026, 0, 0, 81));
        }
        let construction_start = Instant::now();
        let index = if construction == "bulk" {
            SpatialIndex::new(records)?
        } else {
            let mut index = SpatialIndex::new(vec![])?;
            for observation in records {
                index.append(observation)?;
            }
            index
        };
        let construction_seconds = construction_start.elapsed().as_secs_f64();
        let points: Vec<_> = (0..1000)
            .map(|i| {
                Point3::new([
                    ((i * 17) % 73) as f64 * 0.3 + 0.13,
                    ((i * 23) % 61) as f64 * 0.3 + 0.17,
                    ((i * 7) % 23) as f64 * 0.3 + 0.19,
                ])
                .unwrap()
            })
            .collect();
        let query = Neighborhood {
            maximum: 32,
            radius: 8.,
        };
        let budget = MemoryBudget::new(1 << 20);
        let mut workspace = NearestWorkspace::new(query, &budget)?;
        let cancel = Cancellation::default();
        let mut elapsed = [0.; 2];
        let mut sum = 0;
        for pair in 0..4 {
            let mut sums = [0; 2];
            for backend in if pair % 2 == 0 { [0, 1] } else { [1, 0] } {
                if (backend_filter == "owned" && backend == 1)
                    || (backend_filter == "reused" && backend == 0)
                {
                    continue;
                }
                let (seconds, value) = if backend == 0 {
                    run(&points, iterations, |p| {
                        Ok(checksum(&index.nearest(p, query, &cancel)?))
                    })?
                } else {
                    run(&points, iterations, |p| {
                        Ok(checksum(index.nearest_into(p, &mut workspace, &cancel)?))
                    })?
                };
                elapsed[backend] += seconds;
                sums[backend] = value;
            }
            if backend_filter == "all" {
                assert_eq!(sums[0], sums[1]);
            }
            sum = sums[usize::from(backend_filter == "reused")];
        }
        let old = allocation_counter::measure(|| {
            for &p in &points {
                black_box(index.nearest(p, query, &cancel).unwrap());
            }
        });
        let new = allocation_counter::measure(|| {
            for &p in &points {
                black_box(index.nearest_into(p, &mut workspace, &cancel).unwrap());
            }
        });
        assert_eq!(new.count_total, 0);
        for (key, value) in [
            ("observations", "100000".into()),
            ("construction_seconds", construction_seconds.to_string()),
            ("iterations_per_pair", iterations.to_string()),
            ("pairs", "4".into()),
            ("allocated_seconds", elapsed[0].to_string()),
            ("reused_seconds", elapsed[1].to_string()),
            ("checksum", sum.to_string()),
            ("allocation_measurement_queries", points.len().to_string()),
            ("allocated_count", old.count_total.to_string()),
            ("allocated_bytes", old.bytes_total.to_string()),
            ("reused_count", new.count_total.to_string()),
            ("reused_bytes", new.bytes_total.to_string()),
        ] {
            out.write_record([format!("{construction}_{key}"), value])?;
        }
    }
    out.flush()?;
    Ok(())
}
