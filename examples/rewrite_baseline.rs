//! Measured legacy kernels and streamed dataset census for the rewrite.
//! Scope: timed legacy kernels only; the reproducible runner is scripts/benchmark_windows.ps1.

use std::{error::Error, hint::black_box, io, path::Path, time::Instant};
use terrustrial::{
    geometry::support::Support,
    systems::{
        lu::LUSystem,
        solved_systems::{sk_system::SolvedLUSKSystemBuilder, SolvedLUSystem, SolvedSystemBuilder},
    },
    variography::model_variograms::{
        composite::{CompositeVariogram, VariogramType},
        spherical::SphericalVariogram,
    },
};
use ultraviolet::{DRotor3, DVec3};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

#[derive(Default)]
struct Summary {
    count: u64,
    missing: u64,
    mean: f64,
    m2: f64,
    min: Option<f64>,
    max: Option<f64>,
}

impl Summary {
    fn add(&mut self, field: &str) -> Result<()> {
        if field.trim().is_empty() {
            self.missing = self
                .missing
                .checked_add(1)
                .ok_or("missing count overflow")?;
            return Ok(());
        }
        let value: f64 = field.trim().parse()?;
        if !value.is_finite() {
            return Err("nonfinite numeric value".into());
        }
        self.count = self.count.checked_add(1).ok_or("value count overflow")?;
        let delta = value - self.mean;
        self.mean += delta / self.count as f64;
        self.m2 += delta * (value - self.mean);
        if !self.mean.is_finite() || !self.m2.is_finite() {
            return Err("summary arithmetic overflow".into());
        }
        self.min = Some(self.min.map_or(value, |v| v.min(value)));
        self.max = Some(self.max.map_or(value, |v| v.max(value)));
        Ok(())
    }
}

fn census_reader(input: impl io::Read) -> Result<(u64, Vec<(String, Summary)>)> {
    let mut reader = csv::ReaderBuilder::new().from_reader(input);
    let headers = reader.headers()?.clone();
    if headers.is_empty() {
        return Err("missing CSV header".into());
    }
    let mut seen = std::collections::HashSet::new();
    for header in &headers {
        if header.is_empty() || !seen.insert(header) {
            return Err("empty or duplicate CSV column name".into());
        }
    }
    for required in ["x", "y", "z", "porosity.dat", "facies.dat"] {
        if !headers.iter().any(|h| h == required) {
            return Err(format!("missing required column: {required}").into());
        }
    }
    let columns: Vec<_> = headers
        .iter()
        .enumerate()
        .filter(|(_, name)| *name != "category")
        .collect();
    let mut summaries: Vec<_> = columns
        .iter()
        .map(|(_, name)| ((*name).to_owned(), Summary::default()))
        .collect();
    let mut record = csv::StringRecord::new();
    let mut rows = 0_u64;
    while reader.read_record(&mut record)? {
        rows = rows.checked_add(1).ok_or("row count overflow")?;
        for ((index, name), (_, summary)) in columns.iter().zip(&mut summaries) {
            let field = &record[*index];
            if matches!(*name, "x" | "y" | "z") && field.trim().is_empty() {
                return Err(format!("data row {rows}, column {name}: missing coordinate").into());
            }
            summary
                .add(field)
                .map_err(|error| format!("data row {rows}, column {name}: {error}"))?;
        }
    }
    if rows == 0 {
        return Err("dataset has no records".into());
    }
    Ok((rows, summaries))
}

fn census(path: &Path) -> Result<Vec<(String, String)>> {
    let file = std::fs::File::open(path)?;
    let bytes = file.metadata()?.len();
    let started = Instant::now();
    let (rows, summaries) = census_reader(file)?;
    let seconds = started.elapsed().as_secs_f64();
    let mut metrics = vec![
        ("rows".into(), rows.to_string()),
        ("input_bytes".into(), bytes.to_string()),
        ("seconds".into(), seconds.to_string()),
        (
            "rows_per_second".into(),
            (rows as f64 / seconds).to_string(),
        ),
    ];
    for (name, summary) in summaries {
        metrics.push((format!("{name}.count"), summary.count.to_string()));
        metrics.push((format!("{name}.missing"), summary.missing.to_string()));
        if summary.count > 0 {
            metrics.push((format!("{name}.min"), summary.min.unwrap().to_string()));
            metrics.push((format!("{name}.max"), summary.max.unwrap().to_string()));
            metrics.push((format!("{name}.mean"), summary.mean.to_string()));
            metrics.push((
                format!("{name}.population_variance"),
                (summary.m2 / summary.count as f64).to_string(),
            ));
        }
    }
    Ok(metrics)
}

fn kriging(iterations: u64) -> Result<Vec<(String, String)>> {
    if iterations == 0 {
        return Err("iterations must be positive".into());
    }
    let n_cond = 32;
    let n_targets = 8;
    let supports: Vec<_> = (0..n_cond)
        .map(|i| {
            Support::Point(DVec3::new(
                (i % 4) as f64,
                ((i / 4) % 4) as f64,
                (i / 16) as f64,
            ))
        })
        .chain((0..n_targets).map(|i| {
            Support::Point(DVec3::new(
                0.25 + (i % 2) as f64,
                0.25 + ((i / 2) % 2) as f64,
                0.25 + (i / 4) as f64,
            ))
        }))
        .collect();
    let values: Vec<_> = (0..n_cond).map(|i| i as f64 / n_cond as f64).collect();
    let model = CompositeVariogram::new(vec![VariogramType::Spherical(SphericalVariogram::new(
        DVec3::new(8.0, 8.0, 8.0),
        1.7,
        DRotor3::identity(),
    ))]);
    let mut system = LUSystem::new(n_targets, n_cond);
    let (mut h, mut points, mut covariance, mut indices) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut estimate = || -> Result<Vec<f64>> {
        system.build_cov_matrix(
            n_cond,
            n_targets,
            black_box(&supports),
            black_box(&model),
            &mut h,
            &mut points,
            &mut covariance,
            &mut indices,
        );
        let mut solved = SolvedLUSKSystemBuilder.build(&mut system)?;
        solved.populate_cond_values_est(black_box(&values));
        Ok(black_box(solved.estimate()))
    };
    for _ in 0..10 {
        estimate()?;
    }
    let started = Instant::now();
    let mut checksum = 0.0;
    for _ in 0..iterations {
        checksum += estimate()?.iter().sum::<f64>();
    }
    let seconds = started.elapsed().as_secs_f64();
    if !checksum.is_finite() {
        return Err("nonfinite benchmark result".into());
    }
    Ok(vec![
        ("iterations".into(), iterations.to_string()),
        ("conditioning_points".into(), n_cond.to_string()),
        ("targets_per_system".into(), n_targets.to_string()),
        ("seconds".into(), seconds.to_string()),
        (
            "systems_per_second".into(),
            (iterations as f64 / seconds).to_string(),
        ),
        ("checksum".into(), checksum.to_string()),
    ])
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    let metrics = match args.as_slice() {
        [command, path] if command == "census" => census(Path::new(path))?,
        [command, count] if command == "kriging" => {
            kriging(count.to_str().ok_or("invalid iteration count")?.parse()?)?
        }
        _ => return Err("usage: rewrite_baseline census PATH | kriging ITERATIONS".into()),
    };
    let mut writer = csv::Writer::from_writer(io::stdout().lock());
    writer.write_record(["metric", "value"])?;
    for (name, value) in metrics {
        writer.write_record([name, value])?;
    }
    writer.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    const HEADER: &str = "x,y,z,porosity.dat,facies.dat,category\n";

    #[test]
    fn census_handles_missing_attributes_and_quoted_text() {
        let input = format!("{HEADER}0,0,0,1,0,\"a,b\"\n1,0,0,,1,c\n2,0,0,3,1,d\n");
        let (rows, summaries) = census_reader(input.as_bytes()).unwrap();
        let (_, porosity) = summaries
            .iter()
            .find(|(name, _)| name == "porosity.dat")
            .unwrap();
        assert_eq!(rows, 3);
        assert_eq!(porosity.count, 2);
        assert_eq!(porosity.missing, 1);
        assert_eq!(porosity.mean, 2.0);
        assert_eq!(porosity.m2 / porosity.count as f64, 1.0);
    }

    #[test]
    fn census_rejects_bad_schema_and_invalid_observations() {
        for input in [
            String::new(),
            HEADER.to_owned(),
            "x,x,y,z,porosity.dat,facies.dat\n0,0,0,0,1,1\n".into(),
            "x,y,z\n0,0,0\n".into(),
            format!("{HEADER},0,0,1,0,a\n"),
            format!("{HEADER}0,0,0,NaN,0,a\n"),
            format!("{HEADER}0,0,0,inf,0,a\n"),
            format!("{HEADER}0,0,0,bad,0,a\n"),
            format!("{HEADER}0,0,0,1\n"),
        ] {
            assert!(
                census_reader(input.as_bytes()).is_err(),
                "accepted {input:?}"
            );
        }
    }
}
