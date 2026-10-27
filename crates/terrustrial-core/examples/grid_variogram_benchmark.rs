//! Full-reservoir exact auto/cross variograms, in memory and through bounded I/O.
#[path = "support/training.rs"]
#[allow(
    dead_code,
    reason = "The shared loader also has SGEMS support, unused by this reservoir-only fixture."
)]
mod training;
use std::{env, fs::File, path::Path, time::Instant};
use terrustrial_core::{
    execution::{Cancellation, MemoryBudget},
    grid::{self, Grid, Shape, Source, Variable},
    storage::grid::ingest::{self, Rows},
    variography::grid::{experimental, Options},
};
use training::BenchmarkResult;

struct IndicatorRows(training::ReservoirRows<File>);
impl Rows for IndicatorRows {
    fn next(&mut self, values: &mut [Option<f64>]) -> terrustrial_core::Result<Option<u64>> {
        let cell = self.0.next(values)?;
        if cell.is_some() {
            values[1] = values[1].map(|v| f64::from(v == 1.0));
        }
        Ok(cell)
    }
}

fn reservoir(path: &str, budget: &MemoryBudget) -> BenchmarkResult<Grid> {
    let shape = Shape::new([150, 200, 200])?;
    let mut values =
        training::read_reservoir(File::open(path)?, shape, &["porosity.dat", "facies.dat"])?;
    for row in values.as_chunks_mut::<2>().0 {
        // Facies are categories, not quantities. Use a named indicator, never
        // differences of arbitrary facies codes, as the second variable.
        row[1] = row[1].map(|v| f64::from(v == 1.0));
    }
    Ok(Grid::new(
        shape,
        vec![Variable::Continuous { scale: 1.0 }; 2],
        values,
        budget,
    )?)
}
fn run() -> BenchmarkResult<()> {
    let args: Vec<_> = env::args().collect();
    if args.len() != 4
        || ![
            "grid-import",
            "grid-import-stream",
            "grid-memory",
            "grid-file",
        ]
        .contains(&args[1].as_str())
    {
        return Err("usage: grid_variogram_benchmark <grid-import|grid-import-stream CSV NEW_FILE | grid-memory CSV LAGS_PER_AXIS | grid-file FILE LAGS_PER_AXIS>".into());
    }
    let budget = MemoryBudget::new(1 << 30);
    let cancel = Cancellation::default();
    let start = Instant::now();
    if args[1] == "grid-import-stream" {
        let budget = MemoryBudget::new(16 << 20);
        let shape = Shape::new([150, 200, 200])?;
        let mut input = IndicatorRows(training::ReservoirRows::new(
            File::open(&args[2])?,
            shape,
            &["porosity.dat", "facies.dat"],
        )?);
        let report = ingest::import(
            &mut input,
            Path::new(&args[3]),
            ingest::Options {
                layout: grid::Layout::new([150, 200, 200])?,
                variables: 2,
                rows_per_run: 65_536,
                merge_fan_in: 32,
                io_buffer_bytes: 65_536,
                disk_limit_bytes: 1 << 30,
                budget: &budget,
                cancellation: &cancel,
            },
        )?;
        let seconds = start.elapsed().as_secs_f64();
        println!("metric,value\ncells,{}\nvariables,2\nseconds,{seconds}\nfile_bytes,{}\ninitial_runs,{}\nmerge_passes,{}\npeak_accounted_disk_bytes,{}\ntotal_logical_bytes_written,{}\nmaximum_working_reservation_bytes,{}\nrows_per_second,{}",report.rows,report.output_bytes,report.initial_runs,report.merge_passes,report.peak_accounted_disk_bytes,report.total_logical_bytes_written,report.maximum_working_reservation_bytes,report.rows as f64/seconds);
        return Ok(());
    }
    if args[1] == "grid-import" {
        let mut source = reservoir(&args[2], &budget)?;
        let loading = start.elapsed().as_secs_f64();
        let start = Instant::now();
        let file = File::create_new(&args[3])?;
        terrustrial_core::storage::grid::write(&mut source, &file, 8192, &budget, &cancel)?;
        file.sync_all()?;
        println!("metric,value\ncells,{}\nvariables,2\nloading_seconds,{loading}\nseconds,{}\nfile_bytes,{}",source.layout().cells(),start.elapsed().as_secs_f64(),file.metadata()?.len());
        return Ok(());
    }
    let count: i64 = args[3].parse()?;
    if !(1..=128).contains(&count) {
        return Err("this fixture requires 1..128 lags per axis".into());
    }
    let mut source: Box<dyn Source> = if args[1] == "grid-memory" {
        Box::new(reservoir(&args[2], &budget)?)
    } else {
        Box::new(terrustrial_core::storage::grid::FileSource::open(
            File::open(&args[2])?,
            &budget,
        )?)
    };
    let loading = start.elapsed().as_secs_f64();
    let offsets: Vec<_> = (0..3)
        .flat_map(|axis| {
            (1..=count).map(move |lag| {
                let mut h = [0; 3];
                h[axis] = lag;
                h
            })
        })
        .collect();
    let start = Instant::now();
    let mut checksum = 0_u64;
    let mut pairs = 0_u128;
    let mut reads = 0_u128;
    let mut result_rows = vec![];
    for variables in [[0, 0], [0, 1], [1, 1]] {
        let result = experimental(
            &mut *source,
            Options {
                variables,
                offsets: &offsets,
                spacing: [1.0; 3],
                tile_cells: 8192,
                budget: &budget,
                cancellation: &cancel,
            },
        )?;
        reads += result.source_values_read;
        for (i, lag) in result.lags.iter().enumerate() {
            let gamma = lag.statistics.semivariance.ok_or("no eligible pairs")?;
            pairs += lag.statistics.pairs as u128;
            checksum = checksum
                .wrapping_mul(6364136223846793005)
                .wrapping_add(gamma.to_bits())
                .wrapping_add(lag.statistics.pairs);
            result_rows.push((
                format!(
                    "gamma_{}_{}_axis{}_lag{}",
                    variables[0],
                    variables[1],
                    i / count as usize,
                    i % count as usize + 1
                ),
                gamma,
            ));
        }
    }
    let seconds = start.elapsed().as_secs_f64();
    println!("metric,value\ncells,{}\nvariables,2\nlag_queries,{}\nvariable_pairs,3\ninput_values_read,{reads}\neligible_pairs,{pairs}\nresult_checksum,{checksum}\nloading_seconds,{loading}\nseconds,{seconds}\npairs_per_second,{}\nreserved_bytes,{}",source.layout().cells(),offsets.len(),pairs as f64/seconds,budget.used());
    for (key, value) in result_rows {
        println!("{key},{value}");
    }
    Ok(())
}
fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}
