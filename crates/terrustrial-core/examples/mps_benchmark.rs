//! Repeatable categorical pattern queries on SGEMS or the reservoir CSV fixture.
use rand::RngExt;
use std::{env, path::Path, time::Instant};
use terrustrial_core::{
    execution::{random_stream, Cancellation, MemoryBudget},
    grid::{Boundary, Event, Grid, Variable},
    mps::{
        self,
        fft::{Matcher, Refinement},
        MatchOptions,
    },
};

#[path = "support/training.rs"]
mod training;
use training::{read_training, BenchmarkResult};

fn run() -> BenchmarkResult<()> {
    let args: Vec<_> = env::args().collect();
    if args.len() != 4 || !["mps-scalar", "mps-fft", "mps-gpu"].contains(&args[1].as_str()) {
        return Err(
            "usage: mps_benchmark <mps-scalar|mps-fft|mps-gpu> TI.sgems|merged_str.csv QUERIES"
                .into(),
        );
    }
    let queries = args[3].parse::<usize>()?;
    if queries == 0 {
        return Err("positive query count required".into());
    }
    let budget = MemoryBudget::new(2 << 30);
    let cancel = Cancellation::default();
    let (shape, values) = read_training(Path::new(&args[2]))?;
    let classes = values
        .iter()
        .flatten()
        .fold(0.0_f64, |maximum, value| maximum.max(*value))
        + 1.0;
    if classes > u32::MAX as f64 {
        return Err("too many classes".into());
    }
    let training = Grid::new(
        shape,
        vec![Variable::Categorical {
            classes: classes as u32,
        }],
        values,
        &budget,
    )?;
    let start = Instant::now();
    let mut matcher = if args[1] == "mps-fft" {
        Some(Matcher::new(&training, &budget, &cancel)?)
    } else {
        None
    };
    #[cfg(feature = "gpu")]
    let mut gpu = if args[1] == "mps-gpu" {
        let context = std::sync::Arc::new(terrustrial_core::compute::gpu::Context::new()?);
        eprintln!("adapter: {:?}", context.adapter_info());
        Some(mps::gpu::Matcher::new(
            &training, context, &budget, &cancel,
        )?)
    } else {
        None
    };
    #[cfg(not(feature = "gpu"))]
    if args[1] == "mps-gpu" {
        return Err("enable the gpu feature".into());
    }
    let preparation = start.elapsed().as_secs_f64();
    let template: Vec<_> = if shape.dimensions()[2] == 1 {
        (-3..=3)
            .flat_map(|y| {
                (-3..=3).filter_map(move |x| {
                    if x == 0 && y == 0 {
                        None
                    } else {
                        Some([x, y, 0])
                    }
                })
            })
            .collect()
    } else {
        (-1..=1)
            .flat_map(|z| {
                (-1..=1).flat_map(move |y| {
                    (-1..=1).filter_map(move |x| {
                        if x == 0 && y == 0 && z == 0 {
                            None
                        } else {
                            Some([x, y, z])
                        }
                    })
                })
            })
            .collect()
    };
    let mut rng = random_stream(173, 0, 0, 0);
    let start = Instant::now();
    let mut checksum = 0_u64;
    let mut mismatch = 0.0;
    let mut refined = 0_usize;
    for query in 0..queries {
        let center = shape.coordinates(rng.random_range(0..shape.cells()))?;
        let events: Vec<_> = template
            .iter()
            .map(|offset| {
                let position = std::array::from_fn(|axis| {
                    (center[axis] as i64 + offset[axis]).rem_euclid(shape.dimensions()[axis] as i64)
                        as usize
                });
                Event {
                    offset: *offset,
                    variable: 0,
                    value: training
                        .get(shape.index(position).unwrap(), 0)
                        .unwrap()
                        .unwrap(),
                    weight: 1.0,
                }
            })
            .collect();
        let options = MatchOptions {
            neighbors: 8,
            boundary: Boundary::Periodic,
            seed: 29,
            realization: 0,
            query_id: query as u64,
            budget: &budget,
            cancellation: &cancel,
        };
        #[cfg(feature = "gpu")]
        let gpu_matches = if let Some(matcher) = &mut gpu {
            Some(matcher.best_matches_categorical(&events, &[0], &options)?)
        } else {
            None
        };
        #[cfg(not(feature = "gpu"))]
        let gpu_matches: Option<Vec<mps::Match>> = None;
        let matches = if let Some(matches) = gpu_matches {
            refined += matches.len();
            matches
        } else if let Some(matcher) = &mut matcher {
            let report =
                matcher.best_matches_approximate(&events, &[0], &options, Refinement::default())?;
            refined += report.refined_candidates;
            report.matches
        } else {
            refined += shape.cells();
            mps::best_matches(&training, &events, &[0], &options)?
        };
        for result in matches {
            checksum = checksum
                .wrapping_mul(6364136223846793005)
                .wrapping_add(result.index as u64);
            mismatch += result.mismatch;
        }
    }
    let seconds = start.elapsed().as_secs_f64();
    println!("metric,value\ncells,{}\nqueries,{queries}\nevents,{}\nmatching_neighbors,8\npreparation_seconds,{preparation}\nseconds,{seconds}\nqueries_per_second,{}\nrefined_candidates,{refined}\nmatch_checksum,{checksum}\nmismatch_sum,{mismatch}\nreserved_bytes,{}", shape.cells(), template.len(), queries as f64 / seconds, budget.used());
    Ok(())
}
fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}
