//! Same-build fresh versus fixed-conditioning reuse across changing target groups.
use std::{hint::black_box, time::Instant};
use terrustrial_core::{
    data::{Observation, Point3, Site},
    execution::{Cancellation, MemoryBudget},
    gaussian::{
        Drift, GaussianPlan, GaussianWorkspace, NumericalTolerance, Ordinary, Simple, Uncertainty,
        WorkspaceOptions,
    },
    model::{Anisotropy, Coregionalization, Correlation, Lmc, Structure},
    Result,
};
struct Case {
    model: Lmc,
    data: Vec<Observation>,
    values: Vec<f64>,
    targets: Vec<Vec<Site>>,
    uncertainty: Uncertainty,
    cancel: Cancellation,
}
impl Case {
    fn run(
        &self,
        drift: &impl Drift,
        workspace: &mut GaussianWorkspace,
        reuse: bool,
        count: usize,
    ) -> Result<u64> {
        let options = || WorkspaceOptions {
            uncertainty: self.uncertainty,
            tolerance: NumericalTolerance::default(),
            cancellation: &self.cancel,
        };
        let mut checksum = 0_u64;
        let mut prediction = [0.; 6];
        let mut collect = |plan: &GaussianPlan| -> Result<()> {
            let values = &mut prediction[..plan.weights().ncols()];
            plan.estimate_into(&self.values, values)?;
            for value in values {
                checksum = checksum.rotate_left(7) ^ value.to_bits();
            }
            if let Some(cov) = plan.covariance() {
                for j in 0..cov.ncols() {
                    for i in 0..cov.nrows() {
                        checksum = checksum.rotate_left(7) ^ cov[(i, j)].to_bits();
                    }
                }
            }
            checksum ^= plan.roundoff_corrections() as u64;
            Ok(())
        };
        if reuse {
            // Each timed run includes this initial assembly and factorization.
            let mut prepared = workspace.condition(&self.model, &self.data, drift, &self.cancel)?;
            for i in 0..count {
                collect(
                    prepared
                        .prepare(black_box(&self.targets[i % self.targets.len()]), options())?,
                )?;
            }
        } else {
            for i in 0..count {
                collect(workspace.prepare(
                    &self.model,
                    &self.data,
                    black_box(&self.targets[i % self.targets.len()]),
                    drift,
                    options(),
                )?)?;
            }
        }
        Ok(black_box(checksum))
    }
    fn compare(&self, drift: &impl Drift, count: usize) -> Result<[(f64, u64, u64); 2]> {
        let budget = MemoryBudget::new(1 << 28);
        let mut workspaces = [
            GaussianWorkspace::new(self.data.len(), 6, drift.columns(), &budget)?,
            GaussianWorkspace::new(self.data.len(), 6, drift.columns(), &budget)?,
        ];
        let mut seconds = [0.; 2];
        let mut checksum = 0;
        for pair in 0..4 {
            let mut sums = [0; 2];
            for backend in if pair % 2 == 0 { [0, 1] } else { [1, 0] } {
                self.run(drift, &mut workspaces[backend], backend == 1, 10)?;
                let start = Instant::now();
                sums[backend] = self.run(drift, &mut workspaces[backend], backend == 1, count)?;
                seconds[backend] += start.elapsed().as_secs_f64();
            }
            assert_eq!(
                sums[0], sums[1],
                "fresh/reused predictions or covariance differ"
            );
            checksum = sums[0];
        }
        let mut allocations = [0; 2];
        for backend in 0..2 {
            let mut status = Ok(0);
            let measured = allocation_counter::measure(|| {
                status = self.run(drift, &mut workspaces[backend], backend == 1, 120);
            });
            status?;
            allocations[backend] = measured.count_total;
        }
        drop(workspaces);
        assert_eq!(budget.used(), 0);
        Ok([
            (seconds[0], allocations[0], checksum),
            (seconds[1], allocations[1], checksum),
        ])
    }
}
fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let count: usize = std::env::args()
        .nth(2)
        .ok_or("target group count required")?
        .parse()?;
    if count == 0 {
        return Err("positive count required".into());
    }
    let mut output = csv::Writer::from_writer(std::io::stdout().lock());
    output.write_record(["metric", "value"])?;
    output.write_record(["groups_per_sample", &count.to_string()])?;
    output.write_record(["paired_samples", "4"])?;
    output.write_record(["allocation_groups", "120"])?;
    for n in [32, 96] {
        for variables in [1, 3] {
            let coregionalization = if variables == 1 {
                Coregionalization::scalar(1.7)?
            } else {
                Coregionalization::from_factor(3, 3, &[1., 0., 0., 0.3, 0.7, 0., -0.1, 0.2, 0.6])?
            };
            let model = Lmc::new(vec![Structure {
                correlation: Correlation::Exponential,
                anisotropy: Anisotropy::isotropic(3.)?,
                coregionalization,
            }])?;
            let data: Vec<_> = (0..n)
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
            let values = data.iter().map(Observation::value).collect();
            let targets = (0..12)
                .map(|group| {
                    (0..[1, 3, 6][group % 3])
                        .map(|i| {
                            Site::point(
                                Point3::new([
                                    0.21 + i as f64 * 0.17,
                                    0.31 + group as f64 * 0.03,
                                    0.13,
                                ])
                                .unwrap(),
                                i % variables,
                            )
                        })
                        .collect()
                })
                .collect();
            let mut case = Case {
                model,
                data,
                values,
                targets,
                uncertainty: Uncertainty::None,
                cancel: Cancellation::default(),
            };
            for uncertainty in [Uncertainty::None, Uncertainty::Joint] {
                case.uncertainty = uncertainty;
                for ordinary in [false, true] {
                    let measured = if ordinary {
                        case.compare(&Ordinary { variables }, count)?
                    } else {
                        case.compare(&Simple, count)?
                    };
                    let prefix = format!(
                        "n{n}_v{variables}_{}_{}",
                        if ordinary { "OK" } else { "SK" },
                        if uncertainty == Uncertainty::None {
                            "None"
                        } else {
                            "Joint"
                        }
                    );
                    for (name, value) in [
                        ("fresh_seconds", measured[0].0.to_string()),
                        ("reused_seconds", measured[1].0.to_string()),
                        ("fresh_allocations", measured[0].1.to_string()),
                        ("reused_allocations", measured[1].1.to_string()),
                        ("checksum", measured[0].2.to_string()),
                    ] {
                        output.write_record([format!("{prefix}_{name}"), value])?;
                    }
                }
            }
        }
    }
    output.flush()?;
    Ok(())
}
