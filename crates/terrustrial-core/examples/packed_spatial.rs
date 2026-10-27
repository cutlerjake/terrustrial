//! Local comparison driver. Build/index setup and warming are reported separately
//! from queries. This is not a beyond-RAM or production acceptance certificate.
#[cfg(feature = "sqlite")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use rayon::prelude::*;
    use std::{hint::black_box, time::Instant};
    use terrustrial_core::{
        data::{Observation, Point3, Site},
        execution::{Cancellation, MemoryBudget},
        spatial::{NearestWorkspace, Neighborhood, ObservationSource, SpatialIndex},
        storage::packed::{Options, PackedBuilder},
        Result,
    };
    let args: Vec<_> = std::env::args().collect();
    let backend = args.get(1).map(String::as_str).unwrap_or("packed");
    let n: usize = args
        .get(2)
        .map(String::as_str)
        .unwrap_or("100000")
        .parse()?;
    let queries: usize = args.get(3).map(String::as_str).unwrap_or("10000").parse()?;
    let workers: usize = args.get(4).map(String::as_str).unwrap_or("1").parse()?;
    let cache: usize = args.get(5).map(String::as_str).unwrap_or("256").parse()?;
    let leaf: usize = args.get(6).map(String::as_str).unwrap_or("64").parse()?;
    let shape = args.get(7).map(String::as_str).unwrap_or("grid");
    let workload = args.get(8).map(String::as_str).unwrap_or("nearest");
    let memory_mib: u64 = args.get(9).map(String::as_str).unwrap_or("256").parse()?;
    if !["grid", "cloud", "cluster", "line"].contains(&shape)
        || !["nearest", "estimate"].contains(&workload)
    {
        return Err("invalid workload".into());
    }
    if !["packed", "resident"].contains(&backend) || n == 0 || queries == 0 || workers == 0 {
        return Err("invalid benchmark arguments".into());
    }
    fn mix(mut x: u64) -> f64 {
        x = x.wrapping_add(0x9e3779b97f4a7c15);
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
        ((x ^ (x >> 31)) >> 11) as f64 / ((1u64 << 53) as f64)
    }
    fn observation(i: usize, shape: &str) -> Observation {
        let j = i / 4;
        Observation::new(
            i as u64,
            Site::point(
                Point3::new(match shape {
                    "cloud" => std::array::from_fn(|a| mix(j as u64 * 3 + a as u64) * 100.),
                    "cluster" => std::array::from_fn(|a| {
                        (j % 8) as f64 * 12. + mix(j as u64 * 3 + a as u64) * 3.
                    }),
                    "line" => [j as f64 * 0.01, 0., 0.],
                    _ => [(j % 101) as f64, ((j / 101) % 97) as f64, (j / 9797) as f64],
                })
                .unwrap(),
                i % 4,
            ),
            (i as f64).sin(),
            0.01,
        )
        .unwrap()
    }
    fn run(
        source: &(impl ObservationSource + Sync),
        queries: usize,
        workspaces: &mut [NearestWorkspace],
        budget: &MemoryBudget,
        n: usize,
        shape: &str,
        workload: &str,
    ) -> Result<u64> {
        let workers = workspaces.len();
        let center = |i: usize| {
            let j = i % 997;
            if shape == "grid" {
                Point3::new([
                    ((j * 17) % 101) as f64 + 0.1,
                    ((j * 31) % 97) as f64 + 0.2,
                    0.3,
                ])
                .unwrap()
            } else {
                let p = observation((j * 997) % n, shape)
                    .site()
                    .support
                    .center()
                    .coordinates();
                Point3::new([p[0] + 0.1, p[1] + 0.2, p[2] + 0.3]).unwrap()
            }
        };
        if workload == "estimate" {
            use terrustrial_core::{
                data::{Group, Target},
                estimation::{estimate_groups, EstimationOptions},
                execution::ExecutionOptions,
                gaussian::{NumericalTolerance, Simple, Uncertainty},
                model::{Anisotropy, Coregionalization, Correlation, Lmc, Structure},
            };
            let model = Lmc::new(vec![Structure {
                correlation: Correlation::Exponential,
                anisotropy: Anisotropy::isotropic(5.)?,
                coregionalization: Coregionalization::from_factor(
                    4,
                    4,
                    &[
                        1., 0., 0., 0., 0.3, 1., 0., 0., -0.2, 0.4, 1., 0., 0.1, 0.2, 0.3, 1.,
                    ],
                )?,
            }])?;
            let groups = (0..queries).map(|i| Group {
                id: i as u64,
                targets: (0..4)
                    .map(|v| Target {
                        id: (i * 4 + v) as u64,
                        site: Site::point(center(i), v),
                    })
                    .collect(),
            });
            let mut checksum = 0u64;
            estimate_groups(
                &model,
                source,
                groups,
                &Simple,
                EstimationOptions {
                    neighborhood: Neighborhood {
                        maximum: 32,
                        radius: 8.,
                    }
                    .into(),
                    mean: None,
                    uncertainty: Uncertainty::Joint,
                    tolerance: NumericalTolerance::default(),
                    budget,
                    cancellation: &Cancellation::default(),
                },
                ExecutionOptions {
                    batch_groups: workers * 8,
                    maximum_targets_per_group: 4,
                    bytes_per_group: 1 << 20,
                },
                |out| {
                    for v in &out.values {
                        checksum = checksum.wrapping_add(v.to_bits());
                    }
                    let c = out.covariance().unwrap();
                    for j in 0..c.ncols() {
                        for i in 0..c.nrows() {
                            checksum = checksum.wrapping_add(c[(i, j)].to_bits());
                        }
                    }
                    Ok(())
                },
            )?;
            return Ok(black_box(checksum));
        }
        workspaces
            .par_iter_mut()
            .enumerate()
            .map(|(worker, workspace)| {
                let cancel = Cancellation::default();
                let mut checksum = 0u64;
                for i in (worker..queries).step_by(workers) {
                    let p = center(i);
                    for o in source.nearest_into(p, workspace, &cancel)? {
                        checksum = checksum.wrapping_add(o.id().wrapping_mul(i as u64 + 1));
                    }
                }
                Ok(black_box(checksum))
            })
            .try_reduce(|| 0u64, |a, b| Ok(a.wrapping_add(b)))
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .build()?;
    let dir = tempfile::tempdir()?;
    let budget = MemoryBudget::new(
        memory_mib
            .checked_mul(1 << 20)
            .ok_or("memory limit overflow")?,
    );
    // Reserve all worker scratch before the cache is populated, and retain it
    // across warmup and measurement. Query scratch is not a cache eviction victim.
    let mut workspaces = (0..workers)
        .map(|_| {
            NearestWorkspace::new(
                Neighborhood {
                    maximum: 32,
                    radius: 8.,
                },
                &budget,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let start = Instant::now();
    let mut csv = csv::Writer::from_writer(std::io::stdout());
    csv.write_record(["metric", "value"])?;
    csv.write_record(["backend", backend])?;
    csv.write_record(["shape", shape])?;
    csv.write_record(["workload", workload])?;
    csv.write_record(["logical_budget_bytes", &budget.limit().to_string()])?;
    if backend == "packed" {
        let options = Options {
            leaf_records: leaf,
            cached_pages: cache,
            readers: workers,
            sqlite_cache_bytes: 1 << 20,
            ..Options::default()
        };
        let mut builder =
            PackedBuilder::create(dir.path().join("packed.sqlite"), options, &budget)?;
        for i in 0..n {
            builder.push(observation(i, shape), 0, &Cancellation::default())?;
        }
        let store = builder.finish(&Cancellation::default())?;
        csv.write_record(["build_seconds", &start.elapsed().as_secs_f64().to_string()])?;
        let snapshot = store.snapshot()?;
        pool.install(|| {
            run(
                &snapshot,
                queries.min(1000),
                &mut workspaces,
                &budget,
                n,
                shape,
                workload,
            )
        })?;
        let warm = store.statistics();
        let start = Instant::now();
        let checksum = pool.install(|| {
            run(
                &snapshot,
                queries,
                &mut workspaces,
                &budget,
                n,
                shape,
                workload,
            )
        })?;
        csv.write_record(["seconds", &start.elapsed().as_secs_f64().to_string()])?;
        csv.write_record(["checksum", &checksum.to_string()])?;
        csv.write_record(["statistics", &format!("{:?}", store.statistics())])?;
        let measured = store.statistics();
        csv.write_record([
            "query_page_reads",
            &(measured.page_reads - warm.page_reads).to_string(),
        ])?;
        csv.write_record([
            "query_page_bytes",
            &(measured.bytes_read - warm.bytes_read).to_string(),
        ])?;
        csv.write_record(["reserved_bytes", &budget.used().to_string()])?;
    } else {
        let index =
            SpatialIndex::with_budget((0..n).map(|i| observation(i, shape)).collect(), &budget)?;
        csv.write_record(["build_seconds", &start.elapsed().as_secs_f64().to_string()])?;
        pool.install(|| {
            run(
                &index,
                queries.min(1000),
                &mut workspaces,
                &budget,
                n,
                shape,
                workload,
            )
        })?;
        let start = Instant::now();
        let checksum = pool.install(|| {
            run(
                &index,
                queries,
                &mut workspaces,
                &budget,
                n,
                shape,
                workload,
            )
        })?;
        csv.write_record(["seconds", &start.elapsed().as_secs_f64().to_string()])?;
        csv.write_record(["checksum", &checksum.to_string()])?;
    }
    csv.flush()?;
    Ok(())
}
#[cfg(not(feature = "sqlite"))]
fn main() {
    eprintln!("The packed spatial benchmark requires the sqlite feature.");
}
