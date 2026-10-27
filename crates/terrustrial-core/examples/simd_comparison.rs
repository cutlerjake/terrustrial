//! Same-build scalar control, alternating order and bitwise output validation.
use std::{hint::black_box, time::Instant};
use terrustrial_core::{
    data::{Observation, Point3, Site},
    execution::{Cancellation, MemoryBudget},
    gaussian::{GaussianWorkspace, NumericalTolerance, Ordinary, Uncertainty, WorkspaceOptions},
    model::{Anisotropy, Coregionalization, Correlation, Covariance, Lmc, Structure},
    Result,
};

struct Scalar<'a>(&'a Lmc);
impl Covariance for Scalar<'_> {
    fn variables(&self) -> usize {
        self.0.variables()
    }
    fn point_covariance(&self, a: Point3, av: usize, b: Point3, bv: usize) -> Result<f64> {
        self.0.point_covariance(a, av, b, bv)
    }
}

// Same-build control for the incremental cross-covariance optimization. This
// retains SIMD conditioning rows while using the default scalar target columns.
struct RowsOnly<'a>(&'a Lmc);
impl Covariance for RowsOnly<'_> {
    fn variables(&self) -> usize {
        self.0.variables()
    }
    fn point_covariance(&self, a: Point3, av: usize, b: Point3, bv: usize) -> Result<f64> {
        self.0.point_covariance(a, av, b, bv)
    }
    fn covariance_row(
        &self,
        a: &Site,
        observations: &[Observation],
        out: &mut [f64],
    ) -> Result<()> {
        self.0.covariance_row(a, observations, out)
    }
}

fn run<M: Covariance>(
    model: &M,
    observations: &[Observation],
    targets: &[Site],
    iterations: usize,
    workspace: &mut GaussianWorkspace,
) -> Result<(f64, f64)> {
    let cancellation = Cancellation::default();
    let values: Vec<_> = observations.iter().map(Observation::value).collect();
    let mut out = vec![0.; targets.len()];
    let drift = Ordinary {
        variables: model.variables(),
    };
    let mut solve = || -> Result<f64> {
        let plan = workspace.prepare(
            black_box(model),
            black_box(observations),
            black_box(targets),
            &drift,
            WorkspaceOptions {
                uncertainty: Uncertainty::Joint,
                tolerance: NumericalTolerance::default(),
                cancellation: &cancellation,
            },
        )?;
        plan.estimate_into(black_box(&values), &mut out)?;
        Ok(black_box(
            out.iter().sum::<f64>() + plan.covariance().unwrap()[(0, 0)],
        ))
    };
    for _ in 0..10 {
        black_box(solve()?);
    }
    let start = Instant::now();
    let mut checksum = 0.;
    for _ in 0..iterations {
        checksum += solve()?;
    }
    Ok((start.elapsed().as_secs_f64(), checksum))
}

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let iterations: usize = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "10000".into())
        .parse()?;
    if iterations == 0 {
        return Err("positive iterations required".into());
    }
    let mut output = csv::Writer::from_writer(std::io::stdout().lock());
    output.write_record(["metric", "value"])?;
    output.write_record(["iterations", &iterations.to_string()])?;
    output.write_record(["paired_samples", "6"])?;
    for n in [32, 96] {
        for variables in [1, 3] {
            for kernel in [Correlation::Spherical, Correlation::Exponential] {
                let observations: Vec<_> = (0..n)
                    .map(|i| {
                        Observation::new(
                            i as u64,
                            Site::point(
                                Point3::new([
                                    (i % 7) as f64 * 0.3,
                                    ((i / 7) % 5) as f64 * 0.4,
                                    (i / 35) as f64 * 0.2,
                                ])
                                .unwrap(),
                                i % variables,
                            ),
                            (i as f64 * 0.7).sin(),
                            0.01,
                        )
                        .unwrap()
                    })
                    .collect();
                let targets: Vec<_> = (0..6)
                    .map(|i| {
                        Site::point(
                            Point3::new([0.21 + i as f64 * 0.17, 0.31, 0.13]).unwrap(),
                            i % variables,
                        )
                    })
                    .collect();
                let coregionalization = if variables == 1 {
                    Coregionalization::scalar(1.7)?
                } else {
                    Coregionalization::from_factor(
                        3,
                        3,
                        &[1., 0., 0., 0.3, 0.7, 0., -0.1, 0.2, 0.6],
                    )?
                };
                let (s, c) = 0.37_f64.sin_cos();
                let model = Lmc::new(vec![Structure {
                    correlation: kernel,
                    anisotropy: Anisotropy::new(
                        [[c, s, 0.], [-s, c, 0.], [0., 0., 1.]],
                        [3., 1.5, 4.],
                    )?,
                    coregionalization,
                }])?;
                let budget = MemoryBudget::new(1 << 26);
                let mut scalar = GaussianWorkspace::new(n, targets.len(), variables, &budget)?;
                let mut rows_only = GaussianWorkspace::new(n, targets.len(), variables, &budget)?;
                let mut simd = GaussianWorkspace::new(n, targets.len(), variables, &budget)?;
                let mut elapsed = [0.; 3];
                let mut checksum = 0.;
                // All six orders balance every backend's position and neighbor.
                for order in [
                    [0, 1, 2],
                    [2, 1, 0],
                    [1, 2, 0],
                    [0, 2, 1],
                    [2, 0, 1],
                    [1, 0, 2],
                ] {
                    let mut checksums = [0.; 3];
                    for backend in order {
                        let (seconds, sum) = if backend == 0 {
                            run(
                                &Scalar(&model),
                                &observations,
                                &targets,
                                iterations,
                                &mut scalar,
                            )?
                        } else if backend == 1 {
                            run(
                                &RowsOnly(&model),
                                &observations,
                                &targets,
                                iterations,
                                &mut rows_only,
                            )?
                        } else {
                            run(&model, &observations, &targets, iterations, &mut simd)?
                        };
                        elapsed[backend] += seconds;
                        checksums[backend] = sum;
                    }
                    assert_eq!(checksums[0].to_bits(), checksums[1].to_bits());
                    assert_eq!(checksums[0].to_bits(), checksums[2].to_bits());
                    checksum = checksums[0];
                }
                let prefix = format!("n{n}_v{variables}_{kernel:?}");
                for (name, value) in [
                    ("scalar_seconds", elapsed[0]),
                    ("rows_only_seconds", elapsed[1]),
                    ("simd_seconds", elapsed[2]),
                    ("checksum", checksum),
                ] {
                    output.write_record([format!("{prefix}_{name}"), value.to_string()])?;
                }
            }
        }
    }
    output.flush()?;
    Ok(())
}
