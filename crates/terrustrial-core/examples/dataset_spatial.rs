//! Local dataset diagnostic. Caller-owned input stays resident for random target
//! generation; this is not a beyond-RAM benchmark. No data leaves this process.
use rayon::prelude::*;
use std::time::Instant;
use terrustrial_core::{
    data::{Observation, Point3, Site},
    execution::{Cancellation, MemoryBudget},
    spatial::{NearestWorkspace, Neighborhood, ObservationSource, SpatialIndex},
    Result,
};
fn mix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}
fn center(input: &[Observation], i: usize) -> Point3 {
    let index = (mix(i as u64) as usize % (input.len() / 2)) * 2;
    let p = input[index].site().support.center().coordinates();
    Point3::new(std::array::from_fn(|a| {
        p[a] + (mix(i as u64 * 3 + a as u64) >> 11) as f64 / (1u64 << 53) as f64 * 0.8 - 0.4
    }))
    .unwrap()
}
fn measure(
    source: &(impl ObservationSource + Sync),
    input: &[Observation],
    queries: usize,
    workers: usize,
    budget: &MemoryBudget,
) -> Result<(f64, u64)> {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .build()
        .unwrap();
    let query = Neighborhood {
        maximum: 32,
        radius: 8.,
    };
    let mut workspaces = (0..workers)
        .map(|_| NearestWorkspace::new(query, budget))
        .collect::<Result<Vec<_>>>()?;
    let cancel = Cancellation::default();
    for i in 0..16 {
        let p = center(input, i);
        let mut expected: Vec<_> = input
            .iter()
            .filter_map(|o| {
                let d = p.distance(o.site().support.center());
                (d <= query.radius).then_some((d, o.id()))
            })
            .collect();
        expected.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        expected.truncate(query.maximum);
        let found = source.nearest_into(p, &mut workspaces[0], &cancel)?;
        assert_eq!(
            found.iter().map(Observation::id).collect::<Vec<_>>(),
            expected.iter().map(|v| v.1).collect::<Vec<_>>()
        );
    }
    let run = |workspaces: &mut [NearestWorkspace], count: usize, offset: usize| -> Result<u64> {
        pool.install(|| {
            workspaces
                .par_iter_mut()
                .enumerate()
                .map(|(worker, w)| {
                    let mut checksum = 0u64;
                    for i in (worker..count).step_by(workers) {
                        let p = center(input, i + offset);
                        for o in source.nearest_into(p, w, &cancel)? {
                            checksum = checksum.wrapping_add(mix(o.id().wrapping_add(i as u64)));
                        }
                    }
                    Ok(checksum)
                })
                .try_reduce(|| 0u64, |a, b| Ok(a.wrapping_add(b)))
        })
    };
    run(&mut workspaces, 1000, 0)?;
    let start = Instant::now();
    let checksum = run(&mut workspaces, queries, 1000)?;
    Ok((start.elapsed().as_secs_f64(), checksum))
}
fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 6 {
        return Err("usage: dataset_spatial resident|packed INPUT.csv ROWS QUERIES WORKERS".into());
    }
    let rows: usize = args[3].parse()?;
    let queries: usize = args[4].parse()?;
    let workers: usize = args[5].parse()?;
    if rows == 0 || queries == 0 || workers == 0 {
        return Err("positive dimensions required".into());
    }
    let mut reader = csv::Reader::from_path(&args[2])?;
    let headers = reader.headers()?.clone();
    let columns = ["x", "y", "z", "porosity.dat", "acoustic_impedance.dat"]
        .map(|name| {
            headers
                .iter()
                .position(|h| h == name)
                .ok_or("missing column")
        })
        .into_iter()
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let mut input = Vec::new();
    for (i, row) in reader.records().take(rows).enumerate() {
        let row = row?;
        let mut values = [0.; 5];
        for (j, &col) in columns.iter().enumerate() {
            values[j] = row[col].parse()?;
        }
        let point = Point3::new([values[0], values[1], values[2]])?;
        for variable in 0..2 {
            input.push(Observation::new(
                (i * 2 + variable) as u64,
                Site::point(point, variable),
                values[3 + variable],
                0.01,
            )?);
        }
    }
    if input.len() != rows * 2 {
        return Err("dataset shorter than requested row count".into());
    }
    let budget = MemoryBudget::new(512 << 20);
    let start = Instant::now();
    let mut out = csv::Writer::from_writer(std::io::stdout());
    out.write_record(["metric", "value"])?;
    let result = if args[1] == "resident" {
        let source = SpatialIndex::new(input.clone())?;
        out.write_record(["build_seconds", &start.elapsed().as_secs_f64().to_string()])?;
        measure(&source, &input, queries, workers, &budget)?
    } else if args[1] == "packed" {
        #[cfg(feature = "sqlite")]
        {
            use terrustrial_core::storage::packed::{Options, PackedBuilder};
            let temp = tempfile::tempdir()?;
            let mut builder = PackedBuilder::create(
                temp.path().join("dataset.sqlite"),
                Options {
                    cached_pages: 16384,
                    readers: workers,
                    ..Options::default()
                },
                &budget,
            )?;
            let cancel = Cancellation::default();
            for &o in &input {
                builder.push(o, 0, &cancel)?;
            }
            let store = builder.finish(&cancel)?;
            let source = store.snapshot()?;
            out.write_record(["build_seconds", &start.elapsed().as_secs_f64().to_string()])?;
            measure(&source, &input, queries, workers, &budget)?
        }
        #[cfg(not(feature = "sqlite"))]
        {
            return Err("packed requires sqlite feature".into());
        }
    } else {
        return Err("unknown backend".into());
    };
    out.write_record(["seconds", &result.0.to_string()])?;
    out.write_record(["checksum", &result.1.to_string()])?;
    out.write_record(["observations", &input.len().to_string()])?;
    out.flush()?;
    Ok(())
}
