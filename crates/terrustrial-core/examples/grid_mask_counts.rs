//! Exact all-lag mask counts against independent resident pair enumeration.
use std::{hint::black_box, time::Instant};
use terrustrial_core::{
    execution::{Cancellation, MemoryBudget},
    grid::{Layout, Source},
    variography::grid::counts::Workspace,
    Result,
};
struct Fixture {
    layout: Layout,
}
fn present(i: u64, v: usize) -> bool {
    !(i * 7 + v as u64 * 3).is_multiple_of(5 + v as u64 % 4)
}
impl Source for Fixture {
    fn layout(&self) -> Layout {
        self.layout
    }
    fn variables(&self) -> usize {
        2
    }
    fn read(&mut self, v: usize, first: u64, out: &mut [Option<f64>]) -> Result<()> {
        for (i, value) in out.iter_mut().enumerate() {
            *value = present(first + i as u64, v).then_some(1.);
        }
        Ok(())
    }
}
fn direct(points: &[[i64; 3]], dimensions: [u64; 3], output: &mut [u64]) {
    output.fill(0);
    let lengths = dimensions.map(|n| 2 * n as i64 - 1);
    for a in points {
        for b in points {
            let lag: [i64; 3] = std::array::from_fn(|i| b[i] - a[i] + dimensions[i] as i64 - 1);
            let index = (lag[0] + lengths[0] * (lag[1] + lengths[1] * lag[2])) as usize;
            output[index] += 1;
        }
    }
}
fn transformed(
    workspace: &mut Workspace,
    source: &mut Fixture,
    output: &mut [u64],
    cancel: &Cancellation,
) -> Result<()> {
    let counts = workspace.compute(source, [0, 1], cancel)?;
    let [nx, ny, nz] = source.layout.dimensions().map(|n| n as i64);
    let mut i = 0;
    for z in 1 - nz..nz {
        for y in 1 - ny..ny {
            for x in 1 - nx..nx {
                output[i] = counts.pairs([x, y, z]);
                i += 1;
            }
        }
    }
    Ok(())
}
fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 6 {
        return Err("usage: grid_mask_counts NX NY NZ WORKERS ITERATIONS ntt|direct".into());
    }
    let dimensions = [args[0].parse()?, args[1].parse()?, args[2].parse()?];
    let layout = Layout::new(dimensions)?;
    let workers: usize = args[3].parse()?;
    let iterations: usize = args[4].parse()?;
    if layout.cells() > 65536
        || !(1..=8).contains(&workers)
        || !(1..=1000000).contains(&iterations)
        || !["ntt", "direct"].contains(&args[5].as_str())
    {
        return Err("invalid fixture size, workers, repetitions or method".into());
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .build()?;
    let points: Vec<_> = (0..layout.cells())
        .filter(|&i| present(i, 0) && present(i, 1))
        .map(|i| {
            [
                (i % dimensions[0]) as i64,
                (i / dimensions[0] % dimensions[1]) as i64,
                (i / dimensions[0] / dimensions[1]) as i64,
            ]
        })
        .collect();
    let lags = dimensions
        .into_iter()
        .map(|n| (2 * n - 1) as usize)
        .product();
    let mut output = vec![0; lags];
    let budget = MemoryBudget::new(256 << 20);
    let cancel = Cancellation::default();
    let mut source = Fixture { layout };
    pool.install(|| -> Result<()> {
        let mut workspace = Workspace::new(1024, &budget)?;
        if args[5] == "ntt" {
            transformed(&mut workspace, &mut source, &mut output, &cancel)?;
        } else {
            direct(&points, dimensions, &mut output);
        }
        let expected = output.clone();
        let start = Instant::now();
        for _ in 0..iterations {
            let mut fresh = vec![0; lags];
            if args[5] == "ntt" {
                let mut work = Workspace::new(1024, &budget)?;
                transformed(&mut work, &mut source, &mut fresh, &cancel)?;
            } else {
                direct(&points, dimensions, &mut fresh);
            }
            assert_eq!(fresh, expected);
            black_box(&fresh);
        }
        let fresh_seconds = start.elapsed().as_secs_f64();
        let start = Instant::now();
        for _ in 0..iterations {
            if args[5] == "ntt" {
                transformed(&mut workspace, &mut source, &mut output, &cancel)?;
            } else {
                direct(&points, dimensions, &mut output);
            }
            assert_eq!(output, expected);
            black_box(&output);
        }
        let reused_seconds = start.elapsed().as_secs_f64();
        let mut hash = blake3::Hasher::new();
        for &count in &output {
            hash.update(&count.to_le_bytes());
        }
        let total: u64 = output.iter().sum();
        assert_eq!(total, points.len() as u64 * points.len() as u64);
        println!("metric,value");
        println!("cells,{}", layout.cells());
        println!("workers,{workers}");
        println!("iterations,{iterations}");
        println!("lags,{lags}");
        println!("eligible,{}", points.len());
        println!("ordered_pairs,{total}");
        println!("counts_hash,{}", hash.finalize().to_hex());
        println!("reserved_bytes,{}", budget.used());
        println!("fresh_seconds,{fresh_seconds}");
        println!("reused_seconds,{reused_seconds}");
        Ok(())
    })?;
    Ok(())
}
