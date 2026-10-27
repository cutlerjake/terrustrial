//! Explicit 50^3 local windows on the reservoir, or indexed anchor queries.
use std::{fs::File, time::Instant};
use terrustrial_core::{
    anisotropy::{
        field::{self, Edge, Window},
        fitting::DirectionalBin,
        AnchorField, Tensor, TensorField,
    },
    data::Point3,
    execution::{Cancellation, MemoryBudget},
    grid::{Layout, Source},
    model::Anisotropy,
    storage::grid::FileSource,
    variography::grid::{experimental, Options},
};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    let metrics=match args.get(1).map(String::as_str) {
        Some("anisotropy-field") if args.len()==4 => reservoir(&args[2],args[3].parse()?)?,
        Some("anisotropy-anchors") if args.len()==3 => anchors(args[2].parse()?)?,
        Some("anisotropy-gaussian") if args.len()==4 => gaussian(&args[2], args[3].parse()?)?,
        _=>return Err("usage: anisotropy_benchmark anisotropy-field RESERVOIR.trgrid REPETITIONS | anisotropy-anchors QUERIES".into()),
    };
    let mut out = csv::Writer::from_writer(std::io::stdout().lock());
    out.write_record(["metric", "value"])?;
    for (key, value) in metrics {
        out.write_record([key, value])?;
    }
    out.flush()?;
    Ok(())
}

fn gaussian(
    mode: &str,
    repetitions: usize,
) -> Result<Vec<(String, String)>, Box<dyn std::error::Error>> {
    use terrustrial_core::{
        anisotropy::{NonstationaryGaussian, PreparedField},
        data::{Observation, Site},
        gaussian::{
            GaussianWorkspace, NumericalTolerance, Ordinary, Uncertainty, WorkspaceOptions,
        },
        model::{Coregionalization, Covariance},
    };
    if !["direct", "prepared"].contains(&mode) || !(1..=1000000).contains(&repetitions) {
        return Err("anisotropy-gaussian direct|prepared REPETITIONS (1..=1000000)".into());
    }
    let p = |i: usize| {
        Point3::new([
            (i % 8) as f64 * 0.7,
            ((i / 8) % 8) as f64 * 0.8,
            (i / 64) as f64,
        ])
    };
    let anchors = (0..512)
        .map(|i| {
            let angle = i as f64 * 0.003;
            let (s, c) = angle.sin_cos();
            Ok((
                p(i)?,
                Tensor::from_anisotropy(&Anisotropy::new(
                    [[c, s, 0.], [-s, c, 0.], [0., 0., 1.]],
                    [1. + 0.2 * angle.sin(), 0.6, 0.4],
                )?)?,
            ))
        })
        .collect::<terrustrial_core::Result<Vec<_>>>()?;
    let model = NonstationaryGaussian {
        field: AnchorField::new(anchors, 8, 2.)?,
        coregionalization: Coregionalization::from_factor(2, 2, &[1., 0., 0.4, 0.8])?,
    };
    let site = |i: usize, v: usize| -> terrustrial_core::Result<Site> {
        let q = p(i)?.coordinates();
        Ok(Site::point(
            Point3::new([q[0] + 0.13, q[1] + 0.11, q[2] + 0.17])?,
            v,
        ))
    };
    let observations = (0..128)
        .map(|i| Observation::new(i as u64, site(i / 2, i % 2)?, (i as f64).sin(), 0.01))
        .collect::<terrustrial_core::Result<Vec<_>>>()?;
    let targets = (0..16)
        .map(|i| site(64 + i / 2, i % 2))
        .collect::<terrustrial_core::Result<Vec<_>>>()?;
    let points: Vec<_> = observations
        .iter()
        .map(|o| o.site().support.center())
        .chain(targets.iter().map(|s| s.support.center()))
        .collect();
    let values: Vec<_> = observations.iter().map(Observation::value).collect();
    let budget = MemoryBudget::new(64 << 20);
    let cancel = Cancellation::default();
    let mut workspace = GaussianWorkspace::new(128, 16, 2, &budget)?;
    let mut output = vec![0.; 16];
    let mut checksum = 0_u64;
    fn run(
        model: &impl Covariance,
        observations: &[Observation],
        targets: &[Site],
        values: &[f64],
        output: &mut [f64],
        workspace: &mut GaussianWorkspace,
        cancel: &Cancellation,
    ) -> terrustrial_core::Result<u64> {
        let plan = workspace.prepare(
            model,
            observations,
            targets,
            &Ordinary { variables: 2 },
            WorkspaceOptions {
                uncertainty: Uncertainty::Joint,
                tolerance: NumericalTolerance::default(),
                cancellation: cancel,
            },
        )?;
        plan.estimate_into(values, output)?;
        let mut hash = 0_u64;
        for value in output.iter() {
            hash = hash.rotate_left(7) ^ value.to_bits();
        }
        let c = plan.covariance().unwrap();
        for j in 0..c.ncols() {
            for i in 0..c.nrows() {
                hash = hash.rotate_left(7) ^ c[(i, j)].to_bits();
            }
        }
        Ok(hash)
    }
    let reference = run(
        &model,
        &observations,
        &targets,
        &values,
        &mut output,
        &mut workspace,
        &cancel,
    )?;
    let start = Instant::now();
    for _ in 0..repetitions {
        checksum = if mode == "prepared" {
            let prepared = NonstationaryGaussian {
                field: PreparedField::new(&model.field, &points, &budget, &cancel)?,
                coregionalization: model.coregionalization.clone(),
            };
            run(
                &prepared,
                &observations,
                &targets,
                &values,
                &mut output,
                &mut workspace,
                &cancel,
            )?
        } else {
            run(
                &model,
                &observations,
                &targets,
                &values,
                &mut output,
                &mut workspace,
                &cancel,
            )?
        };
        if checksum != reference {
            return Err("prepared Gaussian result differs from direct interpolation".into());
        }
        std::hint::black_box(checksum);
    }
    Ok(vec![
        ("seconds".into(), start.elapsed().as_secs_f64().to_string()),
        ("iterations".into(), repetitions.to_string()),
        ("checksum".into(), checksum.to_string()),
        ("observations".into(), "128".into()),
        ("targets".into(), "16".into()),
        ("anchors".into(), "512".into()),
        ("cache_included".into(), (mode == "prepared").to_string()),
    ])
}
struct Tile {
    values: Vec<Option<f64>>,
}
impl Source for Tile {
    fn layout(&self) -> Layout {
        Layout::new([50; 3]).unwrap()
    }
    fn variables(&self) -> usize {
        1
    }
    fn read(
        &mut self,
        variable: usize,
        first: u64,
        out: &mut [Option<f64>],
    ) -> terrustrial_core::Result<()> {
        if variable != 0 {
            return Err(terrustrial_core::Error::InvalidInput(
                "tile variable".into(),
            ));
        }
        let first = usize::try_from(first).map_err(|_| terrustrial_core::Error::SizeOverflow)?;
        let last = first
            .checked_add(out.len())
            .ok_or(terrustrial_core::Error::SizeOverflow)?;
        out.copy_from_slice(
            self.values
                .get(first..last)
                .ok_or_else(|| terrustrial_core::Error::InvalidInput("tile range".into()))?,
        );
        Ok(())
    }
}
fn reservoir(
    path: &str,
    repetitions: usize,
) -> Result<Vec<(String, String)>, Box<dyn std::error::Error>> {
    if !(1..=1000).contains(&repetitions) {
        return Err("repetitions in 1..=1000 required".into());
    }
    let budget = MemoryBudget::new(16 << 20);
    let cancel = Cancellation::default();
    let start = Instant::now();
    let mut source = FileSource::open(File::open(path)?, &budget)?;
    if source.layout().dimensions() != [150, 200, 200] || source.variables() != 2 {
        return Err("expected two-variable reservoir grid".into());
    }
    let _reservation = budget.reserve(125000 * 16)?;
    let mut tile = Tile {
        values: vec![None; 125000],
    };
    let offsets: Vec<_> = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1],
        [1, -1, 1],
    ]
    .into_iter()
    .flat_map(|d| [1, 2, 4].map(|h| d.map(|v| v * h)))
    .collect();
    let mut evidence = Vec::new();
    let mut sills = Vec::new();
    let mut positions = Vec::new();
    let mut pairs = 0_u64;
    let mut evidence_checksum = 0_u64;
    for z in 0..4 {
        for y in 0..4 {
            for x in 0..3 {
                for iz in 0..50 {
                    for iy in 0..50 {
                        let first = source.layout().index([x * 50, y * 50 + iy, z * 50 + iz])?;
                        let local = (50 * (iy + 50 * iz)) as usize;
                        source.read(0, first, &mut tile.values[local..local + 50])?;
                    }
                }
                let mut mean = 0.;
                let mut m2 = 0.;
                for (i, v) in tile.values.iter().enumerate() {
                    let v = v.ok_or("complete porosity required")?;
                    let d = v - mean;
                    mean += d / (i + 1) as f64;
                    m2 += d * (v - mean);
                }
                let sill = m2 / tile.values.len() as f64;
                if sill <= 0. || !sill.is_finite() {
                    return Err("invalid local porosity variance".into());
                }
                let v = experimental(
                    &mut tile,
                    Options {
                        variables: [0, 0],
                        offsets: &offsets,
                        spacing: [1.; 3],
                        tile_cells: 8192,
                        budget: &budget,
                        cancellation: &cancel,
                    },
                )?;
                let mut bins = Vec::new();
                for lag in &v.lags {
                    let gamma = lag.statistics.semivariance.ok_or("empty local lag")?;
                    let count = lag.statistics.pairs;
                    pairs += count;
                    evidence_checksum = evidence_checksum.rotate_left(7) ^ gamma.to_bits() ^ count;
                    bins.push(DirectionalBin {
                        lag: lag.offset.map(|v| v as f64),
                        semivariance: gamma,
                        pairs: count,
                    });
                }
                evidence.push(bins);
                sills.push(sill);
                positions.push(Point3::new([
                    x as f64 * 50. + 24.5,
                    y as f64 * 50. + 24.5,
                    z as f64 * 50. + 24.5,
                ])?);
            }
        }
    }
    let windows: Vec<_> = (0..48)
        .map(|i| Window {
            position: positions[i],
            bins: &evidence[i],
            structured_sill: sills[i],
            nugget: 0.,
        })
        .collect();
    let mut edges = Vec::new();
    for z in 0..4 {
        for y in 0..4 {
            for x in 0..3 {
                let i = x + 3 * (y + 4 * z);
                for (yes, j) in [(x < 2, i + 1), (y < 3, i + 3), (z < 3, i + 12)] {
                    if yes {
                        edges.push(Edge {
                            nodes: [i, j],
                            weight: 1.,
                        });
                    }
                }
            }
        }
    }
    let preparation = start.elapsed().as_secs_f64();
    let start = Instant::now();
    let mut last = None;
    let mut checksum = None;
    for _ in 0..repetitions {
        let fit = field::fit(
            &windows,
            &edges,
            field::Options {
                range_bounds: [0.25, 100.],
                regularization: 0.01,
                maximum_iterations: 50000,
                tolerance: 1e-8,
                maximum_cg_iterations: 5000,
                cg_tolerance: 1e-10,
                rank_tolerance: 1e-10,
                interpolation_neighbors: 8,
                interpolation_power: 2.,
                budget: &budget,
                cancellation: &cancel,
            },
        )?;
        if !fit.converged {
            return Err(format!("field fit did not converge: {}", fit.stationarity).into());
        }
        let hash = fit
            .field
            .anchors()
            .iter()
            .flat_map(|(_, t)| t.matrix().into_iter().flatten())
            .fold(0_u64, |a, v| a.rotate_left(7) ^ v.to_bits());
        if checksum.is_some_and(|old| old != hash) {
            return Err("field fit changed across repetitions".into());
        }
        checksum = Some(hash);
        last = Some(fit);
    }
    let seconds = start.elapsed().as_secs_f64();
    let f = last.unwrap();
    Ok(vec![
        ("records".into(), "6000000".into()),
        ("windows".into(), "48".into()),
        ("links".into(), edges.len().to_string()),
        ("bins".into(), "1008".into()),
        ("eligible_pairs".into(), pairs.to_string()),
        ("evidence_checksum".into(), evidence_checksum.to_string()),
        ("fit_checksum".into(), checksum.unwrap().to_string()),
        ("repetitions".into(), repetitions.to_string()),
        ("preparation_seconds".into(), preparation.to_string()),
        ("seconds".into(), seconds.to_string()),
        (
            "fits_per_second".into(),
            (repetitions as f64 / seconds).to_string(),
        ),
        ("coordinate_scale".into(), f.coordinate_scale.to_string()),
        ("converged".into(), f.converged.to_string()),
        ("iterations".into(), f.iterations.to_string()),
        ("cg_iterations".into(), f.cg_iterations.to_string()),
        (
            "cg_relative_residual".into(),
            f.cg_relative_residual.to_string(),
        ),
        ("stationarity".into(), f.stationarity.to_string()),
        (
            "weighted_transformed_sse".into(),
            f.weighted_transformed_sse.to_string(),
        ),
        ("roughness".into(), f.roughness.to_string()),
        ("objective".into(), f.objective.to_string()),
        (
            "weak_windows".into(),
            f.nodes
                .iter()
                .filter(|n| n.design_rank < 6)
                .count()
                .to_string(),
        ),
        (
            "excluded_bins".into(),
            f.nodes
                .iter()
                .map(|n| n.excluded_bins)
                .sum::<usize>()
                .to_string(),
        ),
        (
            "axes_at_bound".into(),
            f.nodes
                .iter()
                .map(|n| n.axes_at_bound)
                .sum::<usize>()
                .to_string(),
        ),
    ])
}
fn anchors(queries: usize) -> Result<Vec<(String, String)>, Box<dyn std::error::Error>> {
    if !(1..=10000).contains(&queries) {
        return Err("queries in 1..=10000 required".into());
    }
    let start = Instant::now();
    let mut anchors = Vec::with_capacity(100000);
    for i in 0..100000 {
        let p = Point3::new([(i % 100) as f64, (i / 100 % 100) as f64, (i / 10000) as f64])?;
        let scales = [
            1. + (i % 13) as f64 / 13.,
            1. + (i % 17) as f64 / 17.,
            1. + (i % 19) as f64 / 19.,
        ];
        anchors.push((
            p,
            Tensor::from_anisotropy(&Anisotropy::new(
                [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
                scales,
            )?)?,
        ));
    }
    let field = AnchorField::new(anchors, 8, 2.)?;
    let points: Vec<_> = (0..queries)
        .map(|i| {
            Point3::new([
                ((i * 37) % 100) as f64 + 0.3,
                ((i * 61 / 100) % 100) as f64 + 0.7,
                ((i * 19 / 10000) % 10) as f64 + 0.2,
            ])
            .unwrap()
        })
        .collect();
    let preparation = start.elapsed().as_secs_f64();
    let start = Instant::now();
    let actual: Vec<_> = points
        .iter()
        .map(|&p| field.at(p).map(|t| t.matrix()))
        .collect::<terrustrial_core::Result<_>>()?;
    let seconds = start.elapsed().as_secs_f64();
    let start = Instant::now();
    let mut maximum_error = 0.0_f64;
    for (&p, actual) in points.iter().zip(&actual) {
        let mut distances: Vec<_> = field
            .anchors()
            .iter()
            .enumerate()
            .map(|(i, (a, _))| (p.distance(*a), i))
            .collect();
        distances.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        distances.truncate(8);
        let closest = distances[0].0;
        let total = distances
            .iter()
            .map(|d| (closest / d.0).powf(2.))
            .sum::<f64>();
        for (i, row) in actual.iter().enumerate() {
            for (j, &value) in row.iter().enumerate() {
                let expected = distances
                    .iter()
                    .map(|d| {
                        (closest / d.0).powf(2.) / total * field.anchors()[d.1].1.matrix()[i][j]
                    })
                    .sum::<f64>();
                maximum_error = maximum_error.max((value - expected).abs());
            }
        }
    }
    let reference_seconds = start.elapsed().as_secs_f64();
    if maximum_error > 1e-12 {
        return Err("indexed anchor mismatch".into());
    }
    let checksum = actual
        .iter()
        .flatten()
        .flatten()
        .fold(0_u64, |a, v| a.rotate_left(7) ^ v.to_bits());
    Ok(vec![
        ("anchors".into(), "100000".into()),
        ("neighbors".into(), "8".into()),
        ("queries".into(), queries.to_string()),
        ("seconds".into(), seconds.to_string()),
        ("reference_seconds".into(), reference_seconds.to_string()),
        ("preparation_seconds".into(), preparation.to_string()),
        ("maximum_error".into(), maximum_error.to_string()),
        ("checksum".into(), checksum.to_string()),
    ])
}
