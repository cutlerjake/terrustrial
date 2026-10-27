//! Full binary-facies simulations for scripts/validate_ensembles.py. This is a
//! validation fixture, not a general simulator CLI or a public SIS grid API.
#[path = "support/training.rs"]
mod training;
use rand::{seq::SliceRandom, RngExt};
use std::{env, fs::OpenOptions, path::Path, time::Instant};
use terrustrial_core::{
    data::{Observation, Point3, Site},
    execution::{random_stream, Cancellation, MemoryBudget},
    gaussian::{
        nonlinear::categorical_indicators, GaussianPlan, NumericalTolerance, PrepareOptions,
        Simple, Uncertainty,
    },
    grid::{Boundary, Grid, Shape, Variable},
    high_order::{Axis, FitOptions, Marginal, Plan, Prototype},
    model::{Anisotropy, Coregionalization, Correlation, Lmc, Structure},
    mps, Result,
};
use training::BenchmarkResult;

const SEED: u64 = 294_631;
const RADIUS: i64 = 12;

fn create(path: &Path) -> std::io::Result<std::fs::File> {
    OpenOptions::new().create_new(true).write(true).open(path)
}
fn site(shape: Shape, cell: usize) -> Result<Site> {
    Ok(Site::point(
        Point3::new(shape.coordinates(cell)?.map(|v| v as f64))?,
        0,
    ))
}
struct Sis<'a> {
    model: &'a Lmc,
    prior: f64,
    template: &'a [[i64; 3]],
    neighbors: usize,
    budget: &'a MemoryBudget,
    cancel: &'a Cancellation,
}
impl Sis<'_> {
    // Stationary binary SK indicator simulation, with the same ordered bounded
    // template as the production MPS and SLM drivers. For two complementary
    // indicators a shared covariance is exact for the indicator kriging step.
    // Simplex correction is explicit. No image cleaning or morphology is applied.
    fn simulate(
        &self,
        shape: Shape,
        values: &mut [Option<f64>],
        path: &[usize],
        realization: u64,
        maximum_correction: &mut f64,
    ) -> Result<()> {
        for &cell in path {
            self.cancel.check()?;
            if values[cell].is_some() {
                continue;
            }
            let center = shape.coordinates(cell)?.map(|v| v as i64);
            let dimensions = shape.dimensions();
            let mut observations = vec![];
            let mut labels = vec![];
            for offset in self.template {
                let position: [i64; 3] = std::array::from_fn(|a| center[a] + offset[a]);
                if (0..3).any(|a| position[a] < 0 || position[a] >= dimensions[a] as i64) {
                    continue;
                }
                let neighbor = shape.index(position.map(|v| v as usize))?;
                if let Some(value) = values[neighbor] {
                    observations.push(Observation::new(
                        neighbor as u64,
                        site(shape, neighbor)?,
                        value,
                        0.0,
                    )?);
                    labels.push(value as usize);
                    if labels.len() == self.neighbors {
                        break;
                    }
                }
            }
            let plan = GaussianPlan::prepare(
                self.model,
                &observations,
                &[site(shape, cell)?],
                &Simple,
                PrepareOptions {
                    uncertainty: Uncertainty::None,
                    tolerance: NumericalTolerance::default(),
                    budget: self.budget,
                    cancellation: self.cancel,
                },
            )?;
            let estimate = categorical_indicators(&plan, &labels, &[1.0 - self.prior, self.prior])?;
            *maximum_correction = maximum_correction.max(estimate[0].maximum_correction);
            let mut rng = random_stream(SEED, realization, cell as u64, 62);
            values[cell] = Some(f64::from(rng.random::<f64>() >= estimate[0].corrected[0]));
        }
        Ok(())
    }
}

fn run() -> BenchmarkResult<()> {
    let args: Vec<_> = env::args().collect();
    if args.len() != 11 {
        return Err("usage: ensemble_validation METHOD TRAINING.sgems HARD.csv SIDE REALIZATIONS NEIGHBORS PRIOR RANGE_X RANGE_Y OUTPUT_DIRECTORY".into());
    }
    let method = args[1].as_str();
    if ![
        "mps-scalar",
        "mps-gpu",
        "high-order",
        "high-order-smoothed",
        "sis",
    ]
    .contains(&method)
    {
        return Err("unknown validation method".into());
    }
    let side: usize = args[4].parse()?;
    let realizations: usize = args[5].parse()?;
    let neighbors: usize = args[6].parse()?;
    let prior: f64 = args[7].parse()?;
    let scales = [args[8].parse()?, args[9].parse()?, 1.0];
    if realizations == 0 || neighbors == 0 || !(0.0..1.0).contains(&prior) {
        return Err("positive realizations/neighbors and a nondegenerate prior required".into());
    }
    let output = Path::new(&args[10]);
    let start = Instant::now();
    let budget = MemoryBudget::new(1 << 30);
    let cancel = Cancellation::default();
    let shape = Shape::new([side, side, 1])?;
    let (training_shape, values) = training::read_training(Path::new(&args[2]))?;
    let variables = vec![Variable::Categorical { classes: 2 }];
    let training = Grid::new(training_shape, variables.clone(), values, &budget)?;
    let mut hard = vec![None; shape.cells()];
    let mut reader = csv::Reader::from_path(&args[3])?;
    if reader.headers()?.iter().collect::<Vec<_>>() != ["cell", "value"] {
        return Err("hard data columns must be cell,value".into());
    }
    for row in reader.records() {
        let row = row?;
        let cell: usize = row[0].parse()?;
        let value: f64 = row[1].parse()?;
        if cell >= shape.cells() || hard[cell].is_some() {
            return Err("invalid or repeated hard data cell".into());
        }
        hard[cell] = Some(value);
    }
    // Grid validates all hard labels before any output is created.
    let _hard = Grid::new(shape, variables.clone(), hard.clone(), &budget)?;
    let mut template: Vec<_> = (-RADIUS..=RADIUS)
        .flat_map(|y| {
            (-RADIUS..=RADIUS).filter_map(move |x| {
                (x * x + y * y > 0 && x * x + y * y <= RADIUS * RADIUS).then_some([x, y, 0])
            })
        })
        .collect();
    template.sort_by_key(|v| (v[0] * v[0] + v[1] * v[1], v[1], v[0]));
    let plan = Plan::new(
        &[if method == "high-order-smoothed" {
            Axis::SmoothedCategorical {
                classes: 2,
                uniform_mass: 0.01,
            }
        } else {
            Axis::Categorical { classes: 2 }
        }],
        vec![
            Prototype(vec![Marginal::Categorical(vec![1.0, 0.0])]),
            Prototype(vec![Marginal::Categorical(vec![0.0, 1.0])]),
        ],
        1e-6,
        &budget,
        &cancel,
    )?;
    let model = Lmc::new(vec![Structure {
        correlation: Correlation::Exponential,
        anisotropy: Anisotropy::new([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], scales)?,
        coregionalization: Coregionalization::scalar(prior * (1.0 - prior))?,
    }])?;
    #[cfg(feature = "gpu")]
    let mut gpu = if method == "mps-gpu" {
        let context = std::sync::Arc::new(terrustrial_core::compute::gpu::Context::new()?);
        eprintln!("adapter: {:?}", context.adapter_info());
        Some(mps::gpu::Matcher::new(
            &training, context, &budget, &cancel,
        )?)
    } else {
        None
    };
    #[cfg(not(feature = "gpu"))]
    if method == "mps-gpu" {
        return Err("enable the gpu feature".into());
    }
    let preparation_seconds = start.elapsed().as_secs_f64();
    let mut runs = csv::Writer::from_writer(create(&output.join("runs.csv"))?);
    runs.write_record([
        "realization",
        "status",
        "completed",
        "hard_mismatches",
        "seconds",
        "maximum_correction",
        "nonzero_mismatch_draws",
        "maximum_mismatch",
        "error",
    ])?;
    for realization in 0..realizations {
        let mut path: Vec<_> = (0..shape.cells()).collect();
        path.shuffle(&mut random_stream(SEED, realization as u64, 0, 61));
        let mut state = Grid::new(shape, variables.clone(), hard.clone(), &budget)?;
        let mut sis_values = hard.clone();
        let mut correction = 0.0;
        let mut nonzero_mismatch = 0;
        let mut maximum_mismatch = 0.0_f64;
        let mut record = |_: usize, draw: &mps::JointDraw| {
            nonzero_mismatch += usize::from(draw.mismatch > 0.0);
            maximum_mismatch = maximum_mismatch.max(draw.mismatch);
            Ok(())
        };
        let options = terrustrial_core::simulation::mps::SimulationOptions {
            template: &template,
            maximum_neighbors: neighbors,
            matching_neighbors: 8,
            training_boundary: Boundary::Bounded,
            seed: SEED,
            realization: realization as u64,
            budget: &budget,
            cancellation: &cancel,
        };
        let start = Instant::now();
        let result = match method {
            "mps-scalar" => terrustrial_core::simulation::mps::simulate(
                &training,
                &mut state,
                path,
                options,
                &mut record,
            ),
            #[cfg(feature = "gpu")]
            "mps-gpu" => terrustrial_core::simulation::mps::simulate_gpu_approximate(
                gpu.as_mut().unwrap(),
                &mut state,
                path,
                options,
                1e-5,
                &mut record,
            ),
            "high-order" | "high-order-smoothed" => {
                terrustrial_core::simulation::high_order::simulate(
                    &plan,
                    &training,
                    &mut state,
                    path,
                    terrustrial_core::simulation::high_order::SimulationOptions {
                        template: &template,
                        maximum_neighbors: neighbors,
                        training_boundary: Boundary::Bounded,
                        seed: SEED,
                        realization: realization as u64,
                        fit: FitOptions {
                            maximum_iterations: 100_000,
                            tolerance: 1e-9,
                            minimum_normalizer_fraction: 1e-10,
                            budget: &budget,
                            cancellation: &cancel,
                        },
                    },
                    |_, _| Ok(()),
                )
            }
            "sis" => Sis {
                model: &model,
                prior,
                template: &template,
                neighbors,
                budget: &budget,
                cancel: &cancel,
            }
            .simulate(
                shape,
                &mut sis_values,
                &path,
                realization as u64,
                &mut correction,
            ),
            _ => unreachable!(),
        };
        let seconds = start.elapsed().as_secs_f64();
        let values = if method == "sis" {
            &sis_values
        } else {
            state.values()
        };
        let completed = values.iter().filter(|v| v.is_some()).count();
        let hard_mismatches = hard
            .iter()
            .zip(values)
            .filter(|(a, b)| a.is_some() && a.map(f64::to_bits) != b.map(f64::to_bits))
            .count();
        let mut csv = csv::Writer::from_writer(create(
            &output.join(format!("realization-{realization:05}.csv")),
        )?);
        csv.write_record(["cell", "value"])?;
        for (cell, value) in values.iter().enumerate() {
            csv.write_record([
                cell.to_string(),
                value.map(|v| v.to_string()).unwrap_or_default(),
            ])?;
        }
        csv.flush()?;
        let error = result
            .as_ref()
            .err()
            .map(ToString::to_string)
            .unwrap_or_default();
        runs.write_record([
            realization.to_string(),
            if result.is_ok() { "complete" } else { "failed" }.into(),
            completed.to_string(),
            hard_mismatches.to_string(),
            seconds.to_string(),
            correction.to_string(),
            nonzero_mismatch.to_string(),
            maximum_mismatch.to_string(),
            error.clone(),
        ])?;
        runs.flush()?;
        eprintln!(
            "{method} realization {realization}: {completed}/{} cells, {seconds:.3}s {error}",
            shape.cells()
        );
    }
    println!("preparation_seconds,{preparation_seconds}");
    Ok(())
}
fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}
