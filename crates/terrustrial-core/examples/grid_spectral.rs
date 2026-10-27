//! Local full-map Fourier/direct reference benchmark. No data are published.
use std::{hint::black_box, time::Instant};
use terrustrial_core::{
    execution::{Cancellation, MemoryBudget},
    grid::{Layout, Source},
    variography::grid::spectral::{Accuracy, Method, Workspace},
    Result,
};
struct Fixture {
    layout: Layout,
    values: [Vec<Option<f64>>; 2],
}
impl Source for Fixture {
    fn layout(&self) -> Layout {
        self.layout
    }
    fn variables(&self) -> usize {
        2
    }
    fn read(&mut self, v: usize, first: u64, out: &mut [Option<f64>]) -> Result<()> {
        out.copy_from_slice(&self.values[v][first as usize..first as usize + out.len()]);
        Ok(())
    }
}
#[derive(Clone, Copy)]
struct Point {
    coordinates: [i64; 3],
    values: [f64; 2],
}
#[derive(Clone, Copy, Default, Debug, PartialEq)]
struct Lag {
    pairs: u64,
    gamma: f64,
}
fn direct(points: &[Point], dimensions: [u64; 3], output: &mut [Lag]) {
    output.fill(Lag::default());
    let lengths = dimensions.map(|n| 2 * n as i64 - 1);
    for a in points {
        for b in points {
            let lag: [i64; 3] = std::array::from_fn(|i| {
                b.coordinates[i] - a.coordinates[i] + dimensions[i] as i64 - 1
            });
            let index = (lag[0] + lengths[0] * (lag[1] + lengths[1] * lag[2])) as usize;
            output[index].pairs += 1;
            output[index].gamma += 0.5 * (a.values[0] - b.values[0]) * (a.values[1] - b.values[1]);
        }
    }
    for lag in output {
        if lag.pairs > 0 {
            lag.gamma /= lag.pairs as f64;
        }
    }
}
fn transformed<S: Source + ?Sized>(
    workspace: &mut Workspace,
    source: &mut S,
    variables: [usize; 2],
    output: &mut [Lag],
    cancel: &Cancellation,
) -> Result<([usize; 3], u128)> {
    let [nx, ny, nz] = source.layout().dimensions().map(|n| n as i64);
    let mut map = workspace.compute(source, variables, [1.; 3], Accuracy::default(), cancel)?;
    let mut i = 0;
    let mut methods = [0; 3];
    for z in 1 - nz..nz {
        for y in 1 - ny..ny {
            for x in 1 - nx..nx {
                let estimate = map.lag([x, y, z])?;
                methods[match estimate.method {
                    Method::Analytic => 0,
                    Method::Fourier => 1,
                    Method::Direct => 2,
                }] += 1;
                output[i] = Lag {
                    pairs: estimate.statistics.pairs,
                    gamma: estimate.statistics.semivariance.unwrap_or(0.0),
                };
                i += 1;
            }
        }
    }
    Ok((methods, map.source_values_read))
}
fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if !(7..=8).contains(&args.len()) {
        return Err(
            "usage: grid_spectral NX NY NZ WORKERS ITERATIONS fft|direct auto|cross|localized|slab|thin|fragmented [resident|file]"
                .into(),
        );
    }
    let dimensions = [args[0].parse()?, args[1].parse()?, args[2].parse()?];
    let layout = Layout::new(dimensions)?;
    let workers: usize = args[3].parse()?;
    let iterations: usize = args[4].parse()?;
    if layout.cells() > 32768
        || !(1..=8).contains(&workers)
        || !(1..=1000000).contains(&iterations)
        || !["fft", "direct"].contains(&args[5].as_str())
        || !["auto", "cross", "localized", "slab", "thin", "fragmented"].contains(&args[6].as_str())
    {
        return Err("invalid fixture size, workers, repetitions or method".into());
    }
    let storage = args.get(7).map_or("resident", String::as_str);
    if !["resident", "file"].contains(&storage) {
        return Err("invalid storage mode".into());
    }
    let variables = if args[6] == "auto" { [0, 0] } else { [0, 1] };
    let values = std::array::from_fn(|v| {
        (0..layout.cells())
            .map(|i| {
                let point = [
                    i % dimensions[0],
                    i / dimensions[0] % dimensions[1],
                    i / dimensions[0] / dimensions[1],
                ];
                if args[6] == "thin" && point[0] == 0 {
                    return (v == 0).then_some(f64::MAX);
                }
                if args[6] == "fragmented" {
                    let low = (0..3).all(|a| point[a] < (dimensions[a] / 3).max(1));
                    let high =
                        (0..3).all(|a| point[a] >= dimensions[a] - (dimensions[a] / 3).max(1));
                    if !low && !high {
                        return (v == 0).then_some(f64::MAX);
                    }
                }
                let localized = ["localized", "slab"].contains(&args[6].as_str());
                let origin = [
                    if args[6] == "slab" {
                        0
                    } else {
                        dimensions[0] / 2
                    },
                    dimensions[1] / 4,
                    dimensions[2] / 4,
                ];
                let extent = [
                    if args[6] == "slab" {
                        dimensions[0]
                    } else {
                        (dimensions[0] / 8).max(1)
                    },
                    (dimensions[1] / 4).max(1),
                    (dimensions[2] / 2).max(1),
                ];
                if localized
                    && (0..3).any(|a| point[a] < origin[a] || point[a] - origin[a] >= extent[a])
                {
                    return (v == 0).then_some(f64::MAX);
                }
                let present = args[6] == "auto"
                    || args[6] == "slab"
                    || !(i * 7 + v as u64 * 3).is_multiple_of(5 + v as u64);
                present.then_some(((i * 123 + v as u64 * 321 + i * i * 17) % 97) as f64 - 48.)
            })
            .collect::<Vec<_>>()
    });
    let mut source = Fixture { layout, values };
    let points: Vec<_> = (0..layout.cells())
        .filter_map(|i| {
            Some(Point {
                coordinates: [
                    (i % dimensions[0]) as i64,
                    (i / dimensions[0] % dimensions[1]) as i64,
                    (i / dimensions[0] / dimensions[1]) as i64,
                ],
                values: [
                    source.values[variables[0]][i as usize]?,
                    source.values[variables[1]][i as usize]?,
                ],
            })
        })
        .collect();
    let budget = MemoryBudget::new(256 << 20);
    let cancel = Cancellation::default();
    let benchmark = Benchmark {
        dimensions,
        workers,
        iterations,
        variables,
        method: &args[5],
        points: &points,
    };
    if storage == "file" {
        let mut file = tempfile::tempfile()?;
        terrustrial_core::storage::grid::write(&mut source, &mut file, 1024, &budget, &cancel)?;
        drop(source);
        let mut source = terrustrial_core::storage::grid::FileSource::open(file, &budget)?;
        benchmark.measure(&mut source, &budget, &cancel)?;
    } else {
        benchmark.measure(&mut source, &budget, &cancel)?;
    }
    Ok(())
}
struct Benchmark<'a> {
    dimensions: [u64; 3],
    workers: usize,
    iterations: usize,
    variables: [usize; 2],
    method: &'a str,
    points: &'a [Point],
}
impl Benchmark<'_> {
    fn measure(
        &self,
        source: &mut (impl Source + Send),
        budget: &MemoryBudget,
        cancel: &Cancellation,
    ) -> std::result::Result<(), Box<dyn std::error::Error>> {
        let (dimensions, workers, iterations, variables) = (
            self.dimensions,
            self.workers,
            self.iterations,
            self.variables,
        );
        let layout = source.layout();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()?;
        let lags = dimensions
            .into_iter()
            .map(|n| (2 * n - 1) as usize)
            .product();
        let mut output = vec![Lag::default(); lags];
        let mut reference = output.clone();
        direct(self.points, dimensions, &mut reference);
        pool.install(|| -> Result<()> {
            let mut workspace = Workspace::new(1024, budget)?;
            let methods = if self.method == "fft" {
                transformed(&mut workspace, source, variables, &mut output, cancel)?
            } else {
                direct(self.points, dimensions, &mut output);
                ([0, 0, lags], 0)
            };
            let mut maximum_scaled_error = 0_f64;
            for (a, b) in output.iter().zip(&reference) {
                assert_eq!(a.pairs, b.pairs);
                let scaled_error = (a.gamma - b.gamma).abs() / (1. + b.gamma.abs());
                assert!(scaled_error <= 1e-10, "{a:?} vs {b:?}");
                maximum_scaled_error = maximum_scaled_error.max(scaled_error);
            }
            let expected = output.clone();
            let start = Instant::now();
            for _ in 0..iterations {
                let mut fresh = vec![Lag::default(); lags];
                if self.method == "fft" {
                    let mut work = Workspace::new(1024, budget)?;
                    assert_eq!(
                        transformed(&mut work, source, variables, &mut fresh, cancel)?,
                        methods
                    );
                } else {
                    direct(self.points, dimensions, &mut fresh);
                }
                assert_eq!(fresh, expected);
                black_box(&fresh);
            }
            let fresh_seconds = start.elapsed().as_secs_f64();
            let start = Instant::now();
            for _ in 0..iterations {
                if self.method == "fft" {
                    assert_eq!(
                        transformed(&mut workspace, source, variables, &mut output, cancel)?,
                        methods
                    );
                } else {
                    direct(self.points, dimensions, &mut output);
                }
                assert_eq!(output, expected);
                black_box(&output);
            }
            let reused_seconds = start.elapsed().as_secs_f64();
            let mut counts = blake3::Hasher::new();
            let mut values = blake3::Hasher::new();
            for lag in &output {
                counts.update(&lag.pairs.to_le_bytes());
                values.update(&lag.gamma.to_bits().to_le_bytes());
            }
            let total: u64 = output.iter().map(|l| l.pairs).sum();
            assert_eq!(total, self.points.len() as u64 * self.points.len() as u64);
            println!("metric,value");
            println!("cells,{}", layout.cells());
            println!("workers,{workers}");
            println!("iterations,{iterations}");
            println!("lags,{lags}");
            println!("eligible,{}", self.points.len());
            println!("ordered_pairs,{total}");
            println!("counts_hash,{}", counts.finalize().to_hex());
            println!("values_hash,{}", values.finalize().to_hex());
            println!("maximum_scaled_error,{maximum_scaled_error}");
            println!("analytic_lags,{}", methods.0[0]);
            println!("fourier_lags,{}", methods.0[1]);
            println!("direct_lags,{}", methods.0[2]);
            println!("source_values_read,{}", methods.1);
            println!("reserved_bytes,{}", budget.used());
            println!("fresh_seconds,{fresh_seconds}");
            println!("reused_seconds,{reused_seconds}");
            Ok(())
        })?;
        Ok(())
    }
}
