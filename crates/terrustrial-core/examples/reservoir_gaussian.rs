//! Local-only geological holdout validation, with training-only joint LMC fitting.
#[path = "support/gaussian_holdout.rs"]
mod holdout;
#[allow(dead_code, reason = "Shared loaders support other benchmark formats.")]
#[path = "support/training.rs"]
mod training;
use holdout::{Split, Training};
use std::{
    collections::BinaryHeap,
    fs::{self, File},
    path::Path,
    time::Instant,
};
use terrustrial_core::{
    data::{Group, Observation, Point3, Site, Target},
    estimation::{estimate_groups, EstimationOptions},
    execution::{Cancellation, ExecutionOptions, MemoryBudget},
    fitting::{
        ranges::{self, Kernel, Parameter, Scale},
        CrossVariogramBin, LmcFitOptions,
    },
    gaussian::{NumericalTolerance, Ordinary, Uncertainty},
    grid::{Grid, Shape, Source, Variable},
    model::{Anisotropy, Correlation},
    spatial::{
        MinimumQuotas, ObservationSource, SearchNeighborhood, SearchWorkspace, SpatialIndex,
        VariableLimit, VariableMinimum,
    },
    variography::grid::{experimental, Options},
};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
fn writer(path: impl AsRef<Path>, header: &[&str]) -> Result<csv::Writer<File>> {
    let mut out = csv::Writer::from_writer(File::create_new(path)?);
    out.write_record(header)?;
    Ok(out)
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 3 {
        return Err("usage: reservoir_gaussian INPUT.csv NEW_OUTPUT_DIRECTORY".into());
    }
    let root = Path::new(&args[2]);
    fs::create_dir(root)?;
    let total = Instant::now();
    let budget = MemoryBudget::new(1 << 30);
    let cancel = Cancellation::default();
    let shape = Shape::new([150, 200, 200])?;
    let values = training::read_reservoir(
        File::open(&args[1])?,
        shape,
        &["porosity.dat", "acoustic_impedance.dat"],
    )?;
    let mut grid = Grid::new(
        shape,
        vec![Variable::Continuous { scale: 1. }; 2],
        values,
        &budget,
    )?;
    let loading_seconds = total.elapsed().as_secs_f64();
    let layout = grid.layout();
    let split = Split {
        training_end: 100,
        evaluation_start: 120,
    };
    let stats = holdout::moments(&mut grid, split)?;
    let scales = [stats[0].scale()?, stats[1].scale()?];
    let mut moment_out = writer(
        root.join("normalization.csv"),
        &["variable", "training_count", "mean", "stddev"],
    )?;
    for v in 0..2 {
        moment_out.write_record([
            v.to_string(),
            stats[v].count.to_string(),
            stats[v].mean.to_string(),
            scales[v].to_string(),
        ])?;
    }
    moment_out.flush()?;
    let fit_start = Instant::now();
    let offsets: Vec<_> = (0..3)
        .flat_map(|axis| {
            [1, 2, 4, 8, 16, 32].map(move |h| {
                let mut lag = [0; 3];
                lag[axis] = h;
                lag
            })
        })
        .collect();
    let mut bins = Vec::new();
    let mut training = Training {
        source: &mut grid,
        split,
        moments: stats,
    };
    let mut bin_out = writer(
        root.join("variograms.csv"),
        &["a", "b", "dx", "dy", "dz", "pairs", "semivariance"],
    )?;
    for pair in [[0, 0], [0, 1], [1, 1]] {
        let variogram = experimental(
            &mut training,
            Options {
                variables: pair,
                offsets: &offsets,
                spacing: [1.; 3],
                tile_cells: 8192,
                budget: &budget,
                cancellation: &cancel,
            },
        )?;
        for lag in variogram.lags {
            let gamma = lag.statistics.semivariance.ok_or("empty training lag")?;
            // Independent rectangular training-domain pair count; no held-out endpoint.
            let dims = [100_u64, 200, 200];
            let expected: u64 = (0..3)
                .map(|a| dims[a] - lag.offset[a].unsigned_abs())
                .product();
            if lag.statistics.pairs != expected {
                return Err("training variogram count disagrees with split geometry".into());
            }
            bin_out.write_record([
                pair[0].to_string(),
                pair[1].to_string(),
                lag.offset[0].to_string(),
                lag.offset[1].to_string(),
                lag.offset[2].to_string(),
                expected.to_string(),
                gamma.to_string(),
            ])?;
            bins.push(CrossVariogramBin {
                variables: pair,
                lag: lag.offset.map(|x| x as f64),
                semivariance: gamma,
                pairs: expected,
            });
        }
    }
    bin_out.flush()?;
    let variogram_seconds = fit_start.elapsed().as_secs_f64();
    eprintln!("training variograms complete");
    // One predeclared anisotropic exponential family; only these ranges and PSD
    // joint sills are fitted. No holdout tuning or automatic family-selection claim.
    let template = Kernel {
        correlation: Correlation::Exponential,
        frame: [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
        scales: std::array::from_fn(|index| Scale::Parameter {
            index,
            multiplier: 1.,
        }),
    };
    let fit_start = Instant::now();
    let fit = ranges::fit(
        2,
        &bins,
        &[template],
        &[Parameter {
            initial: 5.,
            bounds: [0.25, 100.],
        }; 3],
        ranges::Options {
            maximum_evaluations: 1000,
            initial_log_step: 1.,
            log_tolerance: 1e-3,
            improvement_tolerance: 1e-10,
            diagnostic_log_step: 0.02,
            sensitivity_relative_tolerance: 1e-3,
            sill: LmcFitOptions {
                maximum_iterations: 10000,
                tolerance: 1e-9,
                budget: &budget,
                cancellation: &cancel,
            },
        },
    )?;
    if !fit.fit.converged || fit.stop != ranges::Stop::LogStepTolerance {
        return Err(format!("training fit incomplete: {:?}", fit.stop).into());
    }
    let fit_seconds = fit_start.elapsed().as_secs_f64();
    let model = &fit.fit.model;
    let mut model_out = writer(
        root.join("model.csv"),
        &["range_x", "range_y", "range_z", "c00", "c01", "c11"],
    )?;
    let sill = &model.structures()[0].coregionalization;
    model_out.write_record([
        fit.parameters[0].to_string(),
        fit.parameters[1].to_string(),
        fit.parameters[2].to_string(),
        sill.covariance(0, 0)?.to_string(),
        sill.covariance(0, 1)?.to_string(),
        sill.covariance(1, 1)?.to_string(),
    ])?;
    model_out.flush()?;
    eprintln!("training fit complete: {:?}", fit.parameters);
    let selection_start = Instant::now();
    let mut observations = Vec::new();
    let mut selected = BinaryHeap::new();
    let mut eligible = 0_u64;
    for cell in 0..layout.cells() {
        if !split.evaluation(layout, cell) {
            continue;
        }
        let point = Point3::new(holdout::coordinates(layout, cell))?;
        for v in 0..2 {
            if split.known(layout, cell, v) {
                let value = (grid.values()[cell as usize * 2 + v]
                    .ok_or("missing condition value")?
                    - stats[v].mean)
                    / scales[v];
                observations.push(Observation::new(
                    cell * 2 + v as u64,
                    Site::point(point, v),
                    value,
                    0.,
                )?);
            }
        }
        if split.target(layout, cell) {
            eligible += 1;
            let key = (holdout::hash(cell ^ 0x4242_4747_9898_2121), cell);
            if selected.len() < 4096 {
                selected.push(key);
            } else if key < *selected.peek().unwrap() {
                selected.pop();
                selected.push(key);
            }
        }
    }
    if selected.len() != 4096 {
        return Err("not enough evaluation targets".into());
    }
    let mut cells: Vec<_> = selected.into_iter().map(|(_, cell)| cell).collect();
    cells.sort_unstable();
    let mut hard_out = writer(
        root.join("conditioning.csv"),
        &["id", "x", "y", "z", "variable", "value"],
    )?;
    for o in &observations {
        let p = o.site().support.center().coordinates();
        hard_out.write_record([
            o.id().to_string(),
            p[0].to_string(),
            p[1].to_string(),
            p[2].to_string(),
            o.site().variable.to_string(),
            o.value().to_string(),
        ])?;
    }
    hard_out.flush()?;
    let hard_count = observations.len();
    let index = SpatialIndex::new(observations)?;
    let selection_seconds = selection_start.elapsed().as_secs_f64();
    let neighborhood = SearchNeighborhood::new(
        32,
        Anisotropy::isotropic(20.)?,
        None,
        (0..2)
            .map(|variable| VariableLimit {
                variable,
                maximum: 16,
            })
            .collect(),
    )?
    .with_minimums(MinimumQuotas {
        total: 8,
        variables: (0..2)
            .map(|variable| VariableMinimum {
                variable,
                minimum: 4,
            })
            .collect(),
        octants: [0; 8],
    })?;
    let mut reference = writer(
        root.join("reference_neighbors.csv"),
        &["cell", "rank", "observation_id"],
    )?;
    let mut search = SearchWorkspace::new(&neighborhood, &budget)?;
    for &cell in cells.iter().step_by(cells.len() / 32) {
        for (rank, o) in index
            .search(
                Point3::new(holdout::coordinates(layout, cell))?,
                &mut search,
                &cancel,
            )?
            .iter()
            .enumerate()
        {
            reference.write_record([cell.to_string(), rank.to_string(), o.id().to_string()])?;
        }
    }
    reference.flush()?;
    let mut metrics = writer(root.join("metrics.csv"), &["metric", "value"])?;
    for (k, v) in [
        ("input_cells", layout.cells().to_string()),
        ("training_cells", stats[0].count.to_string()),
        ("evaluation_cells", "1200000".into()),
        ("eligible_targets", eligible.to_string()),
        ("selected_targets", cells.len().to_string()),
        ("conditioning_observations", hard_count.to_string()),
        ("loading_seconds", loading_seconds.to_string()),
        ("variogram_seconds", variogram_seconds.to_string()),
        ("fit_seconds", fit_seconds.to_string()),
        ("selection_and_index_seconds", selection_seconds.to_string()),
        ("fit_weighted_sse", fit.fit.weighted_sse.to_string()),
        ("fit_evaluations", fit.evaluations.to_string()),
        ("fit_sensitivity_rank", fit.sensitivity_rank.to_string()),
        ("fit_bound_status", format!("{:?}", fit.bounds)),
    ] {
        metrics.write_record([k, &v])?;
    }
    for (workers, batch) in [(1, 32), (2, 32), (4, 32), (8, 32), (4, 1), (4, 8)] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()?;
        let groups: Vec<_> = cells
            .iter()
            .map(|&cell| Group {
                id: cell,
                targets: (0..2)
                    .map(|v| Target {
                        id: cell * 2 + v as u64,
                        site: Site::point(
                            Point3::new(holdout::coordinates(layout, cell)).unwrap(),
                            v,
                        ),
                    })
                    .collect(),
            })
            .collect();
        let mut out = writer(
            root.join(format!("estimates-w{workers}-b{batch}.csv")),
            &[
                "cell",
                "x",
                "y",
                "z",
                "actual0",
                "actual1",
                "prediction0",
                "prediction1",
                "c00",
                "c01",
                "c11",
                "roundoff",
            ],
        )?;
        let start = Instant::now();
        let mut count = 0;
        pool.install(|| {
            estimate_groups(
                model,
                &index,
                groups,
                &Ordinary { variables: 2 },
                EstimationOptions {
                    neighborhood: terrustrial_core::spatial::NeighborhoodQuery::Constrained(
                        &neighborhood,
                    ),
                    mean: None,
                    uncertainty: Uncertainty::Joint,
                    tolerance: NumericalTolerance::default(),
                    budget: &budget,
                    cancellation: &cancel,
                },
                ExecutionOptions {
                    batch_groups: batch,
                    maximum_targets_per_group: 2,
                    bytes_per_group: 1 << 20,
                },
                |result| {
                    let cell = *cells.get(count).ok_or_else(|| {
                        terrustrial_core::Error::InvalidInput("extra estimation output".into())
                    })?;
                    if result.group.id != cell {
                        return Err(terrustrial_core::Error::InvalidInput(
                            "estimation output order".into(),
                        ));
                    }
                    count += 1;
                    let p = holdout::coordinates(layout, cell);
                    let c = result.covariance().unwrap();
                    let y = std::array::from_fn::<_, 2, _>(|v| {
                        (grid.values()[cell as usize * 2 + v].unwrap() - stats[v].mean) / scales[v]
                    });
                    out.write_record([
                        cell.to_string(),
                        p[0].to_string(),
                        p[1].to_string(),
                        p[2].to_string(),
                        y[0].to_string(),
                        y[1].to_string(),
                        result.values[0].to_string(),
                        result.values[1].to_string(),
                        c[(0, 0)].to_string(),
                        c[(0, 1)].to_string(),
                        c[(1, 1)].to_string(),
                        result.roundoff_corrections().to_string(),
                    ])
                    .map_err(std::io::Error::other)?;
                    Ok(())
                },
            )
        })?;
        out.flush()?;
        if count != cells.len() {
            return Err("incomplete estimation output".into());
        }
        metrics.write_record([
            format!("estimate_w{workers}_b{batch}_seconds"),
            start.elapsed().as_secs_f64().to_string(),
        ])?;
        eprintln!("estimation complete: workers={workers}, batch={batch}");
    }
    metrics.write_record(["total_seconds", &total.elapsed().as_secs_f64().to_string()])?;
    metrics.flush()?;
    print!("{}", fs::read_to_string(root.join("metrics.csv"))?);
    Ok(())
}
