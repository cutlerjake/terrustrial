//! Compares durable packed transactions with non-durable resident insertion.
//! The semantics and build cost differ; output is diagnostic, not a speed gate.
#[cfg(feature = "sqlite")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::time::Instant;
    use terrustrial_core::{
        data::{Observation, Point3, Site},
        execution::{Cancellation, MemoryBudget},
        spatial::{
            NearestWorkspace, Neighborhood, ObservationSink, ObservationSource, SpatialIndex,
        },
        storage::packed::{Options, PackedBuilder},
    };
    let args: Vec<_> = std::env::args().collect();
    let backend = args.get(1).map(String::as_str).unwrap_or("packed");
    let mode = args.get(2).map(String::as_str).unwrap_or("publish");
    let n: usize = args.get(3).map(String::as_str).unwrap_or("10000").parse()?;
    let groups: usize = args.get(4).map(String::as_str).unwrap_or("200").parse()?;
    if !["packed", "resident"].contains(&backend)
        || !["publish", "append"].contains(&mode)
        || groups == 0
    {
        return Err("invalid arguments".into());
    }
    fn observation(i: usize) -> Observation {
        let j = i / 4;
        Observation::new(
            i as u64,
            Site::point(
                Point3::new([(j % 101) as f64, ((j / 101) % 97) as f64, (j / 9797) as f64])
                    .unwrap(),
                i % 4,
            ),
            (i as f64).sin(),
            0.01,
        )
        .unwrap()
    }
    let dir = tempfile::tempdir()?;
    let budget = MemoryBudget::new(128 << 20);
    let cancel = Cancellation::default();
    let options = Options {
        cached_pages: 4096,
        readers: 1,
        ..Options::default()
    };
    let start = Instant::now();
    let store = if backend == "packed" {
        let mut b = PackedBuilder::create(dir.path().join("updates.sqlite"), options, &budget)?;
        for i in 0..n {
            b.push(observation(i), 0, &cancel)?;
        }
        if mode == "publish" {
            for i in n..n + groups * 4 {
                b.push(observation(i), ((i - n) / 4 + 1) as u64, &cancel)?;
            }
        }
        Some(b.finish(&cancel)?)
    } else {
        None
    };
    let mut resident = if backend == "resident" {
        Some(SpatialIndex::new((0..n).map(observation).collect())?)
    } else {
        None
    };
    let build = start.elapsed().as_secs_f64();
    let mut latencies = Vec::with_capacity(groups);
    let mut checksum = 0u64;
    let mut w = NearestWorkspace::new(
        Neighborhood {
            maximum: 32,
            radius: 8.,
        },
        &budget,
    )?;
    let start = Instant::now();
    for g in 0..groups {
        let records: [Observation; 4] = std::array::from_fn(|v| observation(n + g * 4 + v));
        let start = Instant::now();
        let snapshot = if let Some(store) = &store {
            Some(if mode == "publish" {
                store.publish_group(
                    (g + 1) as u64,
                    &records.map(|o| (o.id(), o.value())),
                    &cancel,
                )?
            } else {
                store.append_batch(records.map(Ok), &cancel)?
            })
        } else {
            for o in records {
                resident.as_mut().unwrap().append(o)?;
            }
            None
        };
        latencies.push(start.elapsed().as_secs_f64());
        let source: &dyn ObservationSource = if let Some(s) = &snapshot {
            s
        } else {
            resident.as_ref().unwrap()
        };
        for o in source.nearest_into(records[0].site().support.center(), &mut w, &cancel)? {
            checksum = checksum.wrapping_add(o.id().wrapping_mul(g as u64 + 1));
        }
    }
    let elapsed = start.elapsed().as_secs_f64();
    latencies.sort_unstable_by(f64::total_cmp);
    let mut csv = csv::Writer::from_writer(std::io::stdout());
    csv.write_record(["metric", "value"])?;
    for (key, value) in [
        ("build_seconds", build),
        ("seconds", elapsed),
        ("update_p50_seconds", latencies[groups / 2]),
        (
            "update_p95_seconds",
            latencies[(groups * 95 / 100).min(groups - 1)],
        ),
        ("update_max_seconds", latencies[groups - 1]),
    ] {
        csv.write_record([key, &value.to_string()])?;
    }
    csv.write_record(["checksum", &checksum.to_string()])?;
    if let Some(store) = store {
        let start = Instant::now();
        store.compact(&cancel)?;
        csv.write_record([
            "compact_seconds",
            &start.elapsed().as_secs_f64().to_string(),
        ])?;
        let start = Instant::now();
        let exported =
            store.export_compacted(dir.path().join("export.sqlite"), options, &budget, &cancel)?;
        let export_seconds = start.elapsed().as_secs_f64();
        let mut count = 0;
        exported.snapshot()?.scan(&mut |_| {
            count += 1;
            Ok(())
        })?;
        assert_eq!(count, n + groups * 4);
        csv.write_record(["export_seconds", &export_seconds.to_string()])?;
        csv.write_record(["reserved_bytes", &budget.used().to_string()])?;
    }
    csv.flush()?;
    Ok(())
}
#[cfg(not(feature = "sqlite"))]
fn main() {
    eprintln!("Requires the sqlite feature.");
}
