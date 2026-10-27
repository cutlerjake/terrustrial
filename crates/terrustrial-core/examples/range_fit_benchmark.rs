//! Joint range/sill fitting to exact directional reservoir auto/cross variograms.
use std::{fs::File, time::Instant};
use terrustrial_core::{
    data::Point3,
    execution::{Cancellation, MemoryBudget},
    fitting::{
        ranges::{self, Kernel, Parameter, Scale},
        CrossVariogramBin, LmcFitOptions,
    },
    grid::Source,
    model::{Correlation, Covariance},
    storage::grid::FileSource,
    variography::grid::{experimental, Options},
};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 4 || args[1] != "lmc-ranges" {
        return Err(
            "usage: range_fit_benchmark lmc-ranges RESERVOIR.trgrid FIT_REPETITIONS".into(),
        );
    }
    let repetitions: usize = args[3].parse()?;
    if !(1..=1000).contains(&repetitions) {
        return Err("fit repetitions must be in 1..=1000".into());
    }
    let budget = MemoryBudget::new(16 << 20);
    let cancel = Cancellation::default();
    let preparation = Instant::now();
    let mut source = FileSource::open(File::open(&args[2])?, &budget)?;
    if source.variables() != 2 || source.layout().dimensions() != [150, 200, 200] {
        return Err("expected two-variable reservoir column file".into());
    }
    let mut stddev = [0.0; 2];
    let _read_reservation = budget.reserve(8192 * std::mem::size_of::<Option<f64>>() as u64)?;
    let mut buffer = vec![None; 8192];
    for (variable, stddev) in stddev.iter_mut().enumerate() {
        let mut count = 0_u64;
        let mut mean = 0.0;
        let mut m2 = 0.0;
        let mut first = 0;
        while first < source.layout().cells() {
            let n = (source.layout().cells() - first).min(buffer.len() as u64) as usize;
            source.read(variable, first, &mut buffer[..n])?;
            for value in &buffer[..n] {
                let value = value.ok_or("this fixture expects complete reservoir columns")?;
                count += 1;
                let d = value - mean;
                mean += d / count as f64;
                m2 += d * (value - mean);
            }
            first += n as u64;
        }
        *stddev = (m2 / count as f64).sqrt();
        if !stddev.is_finite() || *stddev <= 0.0 {
            return Err("nonpositive or invalid marginal variance".into());
        }
    }
    let offsets: Vec<_> = (0..3)
        .flat_map(|axis| {
            [1, 2, 4, 8, 16, 32].into_iter().map(move |h| {
                let mut lag = [0; 3];
                lag[axis] = h;
                lag
            })
        })
        .collect();
    let mut bins = Vec::new();
    let mut pairs = 0_u128;
    let mut vario_checksum = 0_u64;
    for variables in [[0, 0], [0, 1], [1, 1]] {
        let variogram = experimental(
            &mut source,
            Options {
                variables,
                offsets: &offsets,
                spacing: [1.0; 3],
                tile_cells: 8192,
                budget: &budget,
                cancellation: &cancel,
            },
        )?;
        for (lag, result) in offsets.iter().zip(&variogram.lags) {
            let gamma = result
                .statistics
                .semivariance
                .ok_or("empty variogram bin")?;
            pairs += result.statistics.pairs as u128;
            vario_checksum =
                vario_checksum.rotate_left(7) ^ gamma.to_bits() ^ result.statistics.pairs;
            bins.push(CrossVariogramBin {
                variables,
                lag: lag.map(|x| x as f64),
                semivariance: gamma / (stddev[variables[0]] * stddev[variables[1]]),
                pairs: result.statistics.pairs,
            });
        }
    }
    let preparation_seconds = preparation.elapsed().as_secs_f64();
    let templates = [Kernel {
        correlation: Correlation::Exponential,
        frame: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        scales: std::array::from_fn(|index| Scale::Parameter {
            index,
            multiplier: 1.0,
        }),
    }];
    let parameters = [Parameter {
        initial: 5.0,
        bounds: [0.25, 100.0],
    }; 3];
    let start = Instant::now();
    let mut last = None;
    let mut checksum = None;
    for _ in 0..repetitions {
        let fit = ranges::fit(
            2,
            &bins,
            &templates,
            &parameters,
            ranges::Options {
                maximum_evaluations: 1000,
                initial_log_step: 1.0,
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
        let mut hash = 0_u64;
        for &range in &fit.parameters {
            hash = hash.rotate_left(7) ^ range.to_bits();
        }
        for a in 0..2 {
            for b in a..2 {
                hash = hash.rotate_left(7)
                    ^ fit.fit.model.structures()[0]
                        .coregionalization
                        .covariance(a, b)?
                        .to_bits();
            }
        }
        if checksum.is_some_and(|old| old != hash) {
            return Err("fit results changed across repetitions".into());
        }
        checksum = Some(hash);
        last = Some(fit);
    }
    let seconds = start.elapsed().as_secs_f64();
    let fit = last.unwrap();
    let origin = Point3::new([0.0; 3])?;
    let count = bins.iter().map(|b| b.pairs).max().unwrap() as f64;
    let mut objective = 0.0;
    for bin in &bins {
        let [a, b] = bin.variables;
        let model = &fit.fit.model;
        let predicted = model.point_covariance(origin, a, origin, b)?
            - model.point_covariance(origin, a, Point3::new(bin.lag)?, b)?;
        objective += (bin.pairs as f64 / count) * (predicted - bin.semivariance).powi(2);
    }
    if (objective - fit.fit.weighted_sse).abs() > 1e-9 * (1.0 + objective) {
        return Err(
            "fitted-model objective disagrees with independent covariance evaluation".into(),
        );
    }
    let mut out = csv::Writer::from_writer(std::io::stdout().lock());
    out.write_record(["metric", "value"])?;
    let mut metrics = vec![
        ("records".into(), source.layout().cells().to_string()),
        ("variables".into(), "2".into()),
        ("variogram_bins".into(), bins.len().to_string()),
        ("eligible_pairs".into(), pairs.to_string()),
        ("variogram_checksum".into(), vario_checksum.to_string()),
        ("fit_checksum".into(), checksum.unwrap().to_string()),
        ("repetitions".into(), repetitions.to_string()),
        (
            "preparation_seconds".into(),
            preparation_seconds.to_string(),
        ),
        ("seconds".into(), seconds.to_string()),
        (
            "fits_per_second".into(),
            (repetitions as f64 / seconds).to_string(),
        ),
        ("stop".into(), format!("{:?}", fit.stop)),
        ("evaluations".into(), fit.evaluations.to_string()),
        (
            "total_sill_iterations".into(),
            fit.total_sill_iterations.to_string(),
        ),
        ("sensitivity_rank".into(), fit.sensitivity_rank.to_string()),
        ("weighted_sse".into(), fit.fit.weighted_sse.to_string()),
        ("independent_weighted_sse".into(), objective.to_string()),
        (
            "initial_normalized_sse".into(),
            fit.objective_history[0].to_string(),
        ),
        (
            "final_normalized_sse".into(),
            fit.normalized_weighted_sse.to_string(),
        ),
    ];
    for i in 0..3 {
        metrics.push((format!("range_{i}"), fit.parameters[i].to_string()));
        metrics.push((format!("bound_{i}"), format!("{:?}", fit.bounds[i])));
        metrics.push((
            format!("sensitivity_eigenvalue_{i}"),
            fit.sensitivity_eigenvalues[i].to_string(),
        ));
    }
    for (i, value) in stddev.into_iter().enumerate() {
        metrics.push((format!("field_stddev_{i}"), value.to_string()));
    }
    for a in 0..2 {
        for b in a..2 {
            metrics.push((
                format!("standardized_sill_{a}_{b}"),
                fit.fit.model.structures()[0]
                    .coregionalization
                    .covariance(a, b)?
                    .to_string(),
            ));
        }
    }
    for (key, value) in metrics {
        out.write_record([key, value])?;
    }
    out.flush()?;
    Ok(())
}
