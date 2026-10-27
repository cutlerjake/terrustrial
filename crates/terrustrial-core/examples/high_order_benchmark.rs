//! Joint SLM density-query baseline, distinct from a top-k pattern-query benchmark.
#[path = "support/training.rs"]
mod training;
use rand::RngExt;
use std::{env, fs::File, path::Path, time::Instant};
use terrustrial_core::{
    execution::{random_stream, Cancellation, MemoryBudget},
    grid::{Boundary, Grid, Shape, Variable},
    high_order::{
        aggregate,
        grid::{GridReplicates, Location},
        Axis, FitOptions, Marginal, Plan, Prototype,
    },
};
use training::BenchmarkResult;

fn run() -> BenchmarkResult<()> {
    let args: Vec<_> = env::args().collect();
    if args.len() != 4 || !["high-order", "high-order-prefix"].contains(&args[1].as_str()) {
        return Err(
            "usage: high_order_benchmark <high-order|high-order-prefix> TI.sgems|merged_str.csv QUERIES".into(),
        );
    }
    let prefix = args[1] == "high-order-prefix";
    let queries: usize = args[3].parse()?;
    if queries == 0 {
        return Err("positive query count required".into());
    }
    let start = Instant::now();
    let budget = MemoryBudget::new(2 << 30);
    let cancel = Cancellation::default();
    let mixed = Path::new(&args[2]).extension().is_some_and(|e| e == "csv");
    let (shape, mut data) = if mixed {
        let shape = Shape::new([150, 200, 200])?;
        (
            shape,
            training::read_reservoir(
                File::open(&args[2])?,
                shape,
                &["porosity.dat", "facies.dat"],
            )?,
        )
    } else {
        training::read_training(Path::new(&args[2]))?
    };
    let variables = if mixed { 2 } else { 1 };
    if prefix {
        let mut mask = random_stream(1049, 0, 0, 0);
        for row in data.chunks_exact_mut(variables) {
            if mask.random::<f64>() >= 0.1 {
                row.fill(None);
            }
        }
    }
    let retained = data
        .chunks_exact(variables)
        .filter(|r| r[0].is_some())
        .count();
    if retained == 0 {
        return Err("no retained source cells".into());
    }
    let classes = data
        .chunks_exact(variables)
        .filter_map(|v| v[variables - 1])
        .fold(0.0, f64::max)
        + 1.0;
    if classes > 32.0 {
        return Err("this benchmark limits the prototype category count to 32".into());
    }
    let classes = classes as u32;
    let mut axes = vec![];
    let mut definitions = vec![];
    let mut minimum = 0.0;
    let mut maximum = 0.0;
    if mixed {
        minimum = data
            .as_chunks::<2>()
            .0
            .iter()
            .filter_map(|v| v[0])
            .fold(f64::INFINITY, f64::min);
        maximum = data
            .as_chunks::<2>()
            .0
            .iter()
            .filter_map(|v| v[0])
            .fold(f64::NEG_INFINITY, f64::max);
        if minimum >= maximum {
            return Err("nonconstant porosity required".into());
        }
        for row in data.as_chunks_mut::<2>().0 {
            row[0] = row[0].map(|value| {
                (2.0 * ((value - minimum) / (maximum - minimum)) - 1.0).clamp(-1.0, 1.0)
            });
        }
        axes.push(Axis::Continuous { degree: 6 });
        definitions.push(Variable::Continuous { scale: 1.0 });
    }
    axes.push(Axis::Categorical { classes });
    definitions.push(Variable::Categorical { classes });
    let grid = Grid::new(shape, definitions, data, &budget)?;
    let loading = start.elapsed().as_secs_f64();
    let start = Instant::now();
    let mut prototypes = vec![];
    for category in 0..classes {
        for i in 0..if mixed { 9 } else { 1 } {
            let mut p = vec![0.0; classes as usize];
            p[category as usize] = 1.0;
            let mut marginal = vec![];
            if mixed {
                marginal.push(Marginal::TruncatedNormal {
                    mean: -1.0 + 0.25 * i as f64,
                    deviation: 0.15,
                });
            }
            marginal.push(Marginal::Categorical(p));
            prototypes.push(Prototype(marginal));
        }
    }
    let plan = Plan::new(&axes, prototypes, 1e-6, &budget, &cancel)?;
    let preparation = start.elapsed().as_secs_f64();
    let targets: Vec<_> = (0..variables).collect();
    let locations: Vec<_> = (0..variables)
        .map(|variable| Location {
            offset: [-1, 0, 0],
            variable,
        })
        .collect();
    let source = GridReplicates::new(
        &grid,
        &targets,
        &locations,
        Boundary::Periodic,
        &budget,
        &cancel,
    )?;
    let mut rng = random_stream(173, 0, 0, 0);
    let mut checksum = 0u64;
    let mut contributing = 0u64;
    let mut iterations = 0usize;
    let mut maximum_negative = 0.0_f64;
    let mut minimum_normalizer = 1.0_f64;
    let mut support_full = shape.cells() as u64;
    let mut minimum_row_fraction = 1.0_f64;
    let mut draw = vec![0.0; variables];
    let start = Instant::now();
    for query in 0..queries {
        let mut neighbor = None;
        for _ in 0..100_000 {
            let mut point = shape.coordinates(rng.random_range(0..shape.cells()))?;
            point[0] = (point[0] + shape.dimensions()[0] - 1) % shape.dimensions()[0];
            let candidate = shape.index(point)?;
            if grid.get(candidate, 0)?.is_some() {
                neighbor = Some(candidate);
                break;
            }
        }
        let neighbor = neighbor.ok_or("could not select a retained query event")?;
        let event: Vec<_> = axes
            .iter()
            .enumerate()
            .map(|(i, &axis)| Ok((axis, grid.get(neighbor, i)?.unwrap())))
            .collect::<terrustrial_core::Result<_>>()?;
        let settings = FitOptions {
            maximum_iterations: 100_000,
            tolerance: 1e-9,
            minimum_normalizer_fraction: 1e-10,
            budget: &budget,
            cancellation: &cancel,
        };
        let fit = if prefix {
            let result = plan.condition_prefixes(
                &source,
                &event,
                aggregate::Options {
                    minimum_replicates: 1,
                    unsupported: aggregate::Unsupported::Reject,
                    fit: settings,
                },
            )?;
            support_full = *result.evidence.support.last().unwrap();
            minimum_row_fraction = minimum_row_fraction.min(result.evidence.minimum_row_fraction);
            result.distribution
        } else {
            plan.condition(&source, &event, settings)?
        };
        fit.draw_into(29, 0, query as u64, &mut draw, &cancel)?;
        for value in fit.weights().iter().chain(&draw) {
            checksum = checksum
                .wrapping_mul(6364136223846793005)
                .wrapping_add(value.to_bits());
        }
        contributing += fit.diagnostics.nonzero_kernel_rows;
        iterations += fit.diagnostics.iterations;
        maximum_negative = maximum_negative.max(fit.diagnostics.negative_weight_fraction);
        minimum_normalizer = minimum_normalizer.min(fit.diagnostics.normalizer_fraction);
    }
    let seconds = start.elapsed().as_secs_f64();
    println!("metric,value\nprefix_aggregation,{}\nretained_source_cells,{retained}\nfull_prefix_support,{support_full}\nminimum_row_fraction,{minimum_row_fraction}\ncells,{}\nqueries,{queries}\nvariables,{variables}\nevents,{variables}\nprototypes,{}\ncontinuous_degree,{}\nporosity_min,{minimum}\nporosity_max,{maximum}\nloading_seconds,{loading}\npreparation_seconds,{preparation}\nseconds,{seconds}\nqueries_per_second,{}\nnonzero_kernel_rows,{contributing}\nfit_iterations,{iterations}\nmaximum_negative_weight_fraction,{maximum_negative}\nminimum_normalizer_fraction,{minimum_normalizer}\nmixture_draw_checksum,{checksum}\nreserved_bytes,{}",prefix as u8,shape.cells(),plan.prototypes().len(),if mixed {6} else {0},queries as f64/seconds,budget.used());
    Ok(())
}
fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}
