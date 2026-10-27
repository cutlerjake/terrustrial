//! Exact irregular variogram reference, indexed whole-call and reuse measurements.
use std::{hint::black_box, time::Instant};
use terrustrial_core::{
    data::Point3,
    execution::{Cancellation, MemoryBudget},
    variography::{
        self,
        indexed::{SampleIndex, Workspace},
        Bin, Sample,
    },
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 6 {
        return Err(
            "usage: indexed_variography sparse|dense N WORKERS ITERATIONS oracle|indexed".into(),
        );
    }
    let geometry = &args[1];
    let n: usize = args[2].parse()?;
    let workers: usize = args[3].parse()?;
    let iterations: usize = args[4].parse()?;
    let mode = &args[5];
    if !["sparse", "dense"].contains(&geometry.as_str())
        || !["oracle", "indexed"].contains(&mode.as_str())
        || !(2..=2000000).contains(&n)
        || !(1..=32).contains(&workers)
        || !(1..=1000000).contains(&iterations)
    {
        return Err("invalid geometry, mode, size, workers or iterations".into());
    }
    let samples: Vec<_> = (0..n)
        .map(|i| {
            let point = if geometry == "sparse" {
                // A line with bounded transverse irregularity has an analytic pair
                // count below radius 3.1: precisely the three nearest index offsets.
                [i as f64, (i % 7) as f64 * 0.03, (i % 11) as f64 * 0.02]
            } else {
                [
                    (i % 31) as f64 / 31.,
                    (i / 31 % 29) as f64 / 29.,
                    (i / 899 % 23) as f64 / 23.,
                ]
            };
            Sample {
                point: Point3::new(point).unwrap(),
                values: vec![Some((i % 17) as f64 - 8.), Some(4. - (i % 17) as f64 * 0.5)],
            }
        })
        .collect();
    let edges = [0.5, 1.1, 2.1, 3.1];
    let cancel = Cancellation::default();
    let budget = MemoryBudget::new(512 << 20);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .build()?;
    let fingerprint = |bins: &[Bin]| -> Vec<(u64, Option<u64>, Option<u64>)> {
        bins.iter()
            .map(|b| {
                (
                    b.pairs,
                    b.mean_distance.map(f64::to_bits),
                    b.semivariance.map(f64::to_bits),
                )
            })
            .collect()
    };
    let (bins, candidates, build_seconds, fresh_seconds, reused_seconds, allocations) = pool
        .install(|| -> terrustrial_core::Result<_> {
            if mode == "oracle" {
                let expected = variography::experimental(&samples, 0, 1, &edges, None, &cancel)?;
                let start = Instant::now();
                for _ in 0..iterations {
                    black_box(variography::experimental(
                        black_box(&samples),
                        0,
                        1,
                        &edges,
                        None,
                        &cancel,
                    )?);
                }
                let seconds = start.elapsed().as_secs_f64();
                return Ok((
                    expected,
                    n as u128 * (n - 1) as u128 / 2,
                    0.,
                    seconds,
                    seconds,
                    None,
                ));
            }
            let start = Instant::now();
            let index = SampleIndex::new(&samples, &budget, &cancel)?;
            let mut workspace =
                Workspace::new(edges.len(), if workers == 1 { 1 } else { 32 }, &budget)?;
            let build_seconds = start.elapsed().as_secs_f64();
            let result = index.experimental([0, 1], &edges, None, &mut workspace, &cancel)?;
            let candidates = result.candidate_pairs;
            let bins = result.bins.to_vec();
            let expected = fingerprint(&bins);
            let start = Instant::now();
            for _ in 0..iterations {
                black_box(index.experimental([0, 1], &edges, None, &mut workspace, &cancel)?);
            }
            let reused_seconds = start.elapsed().as_secs_f64();
            assert_eq!(
                fingerprint(
                    index
                        .experimental([0, 1], &edges, None, &mut workspace, &cancel)?
                        .bins
                ),
                expected
            );
            let start = Instant::now();
            for _ in 0..iterations {
                let index = SampleIndex::new(black_box(&samples), &budget, &cancel)?;
                let mut workspace =
                    Workspace::new(edges.len(), if workers == 1 { 1 } else { 32 }, &budget)?;
                black_box(index.experimental([0, 1], &edges, None, &mut workspace, &cancel)?);
            }
            let fresh_seconds = start.elapsed().as_secs_f64();
            let allocations = (workers == 1).then(|| {
                allocation_counter::measure(|| {
                    black_box(
                        index
                            .experimental([0, 1], &edges, None, &mut workspace, &cancel)
                            .unwrap(),
                    );
                })
            });
            Ok((
                bins,
                candidates,
                build_seconds,
                fresh_seconds,
                reused_seconds,
                allocations,
            ))
        })?;
    if geometry == "sparse" {
        let expected: u64 = (1..=3).map(|h| n.saturating_sub(h) as u64).sum();
        assert_eq!(bins.iter().map(|b| b.pairs).sum::<u64>(), expected);
    } else {
        assert_eq!(
            bins.iter().map(|b| b.pairs as u128).sum::<u128>(),
            n as u128 * (n - 1) as u128 / 2
        );
    }
    println!("metric,value\nsamples,{n}\nworkers,{workers}\niterations,{iterations}\ncandidate_pairs,{candidates}\nbuild_seconds,{build_seconds}\nfresh_seconds,{fresh_seconds}\nreused_seconds,{reused_seconds}");
    for (i, bin) in bins.iter().enumerate() {
        println!(
            "bin_{i}_pairs,{}\nbin_{i}_distance,{}\nbin_{i}_gamma,{}",
            bin.pairs,
            bin.mean_distance.unwrap_or(0.),
            bin.semivariance.unwrap_or(0.)
        );
    }
    if let Some(a) = allocations {
        println!(
            "warm_allocations,{}\nwarm_allocated_bytes,{}",
            a.count_total, a.bytes_total
        );
        assert_eq!(a.count_total, 0);
    }
    Ok(())
}
