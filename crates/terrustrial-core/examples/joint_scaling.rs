//! Joint Gaussian preparation and sampling across explicit matrix dimensions.
use std::{hint::black_box, time::Instant};
use terrustrial_core::{
    data::{Observation, Point3, Site, Support},
    execution::{Cancellation, MemoryBudget},
    gaussian::{
        GaussianPlan, GaussianWorkspace, NumericalTolerance, Ordinary, Uncertainty,
        WorkspaceOptions,
    },
    model::{Anisotropy, Coregionalization, Correlation, Lmc, Structure},
};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if !(5..=7).contains(&args.len()) {
        return Err("usage: joint_scaling OBSERVATIONS TARGETS VARIABLES ITERATIONS [dump|draw] [anisotropic|block]".into());
    }
    let mut mode = "";
    let mut geometry = "";
    for arg in &args[5..] {
        match arg.as_str() {
            "dump" | "draw" if mode.is_empty() => mode = arg,
            "anisotropic" | "block" if geometry.is_empty() => geometry = arg,
            _ => return Err("unknown or repeated benchmark option".into()),
        }
    }
    let site = |p, variable| -> terrustrial_core::Result<Site> {
        Ok(Site {
            support: if geometry == "block" {
                Support::block(Point3::new(p)?, [0.1, 0.15, 0.2], [2, 2, 2])?
            } else {
                Support::point(Point3::new(p)?)
            },
            variable,
        })
    };
    let n: usize = args[1].parse()?;
    let m: usize = args[2].parse()?;
    let v: usize = args[3].parse()?;
    let iterations: usize = args[4].parse()?;
    if !(1..=32).contains(&v)
        || !(2 * v..=2048).contains(&n)
        || !(1..=256).contains(&m)
        || !(1..=10_000_000).contains(&iterations)
    {
        return Err("require 1<=variables<=32, 2*variables<=observations<=2048, 1<=targets<=256, 1<=iterations<=10000000".into());
    }
    let model = Lmc::new(
        (0..2)
            .map(|s| {
                let factor: Vec<_> = (0..v)
                    .flat_map(|i| {
                        (0..v).map(move |j| {
                            if i == j {
                                1. + s as f64 * 0.2
                            } else {
                                0.1 * ((i * 17 + j * 7 + s) as f64).sin()
                            }
                        })
                    })
                    .collect();
                Ok(Structure {
                    correlation: if s == 0 {
                        Correlation::Exponential
                    } else {
                        Correlation::Spherical
                    },
                    anisotropy: if geometry.is_empty() {
                        Anisotropy::isotropic(2. + s as f64 * 3.)?
                    } else {
                        let (sin, cos) = 0.37_f64.sin_cos();
                        let radius = 2. + s as f64 * 3.;
                        Anisotropy::new(
                            [[cos, sin, 0.], [-sin, cos, 0.], [0., 0., 1.]],
                            [radius, radius * 0.5, radius * 1.5],
                        )?
                    },
                    coregionalization: Coregionalization::from_factor(v, v, &factor)?,
                })
            })
            .collect::<terrustrial_core::Result<Vec<_>>>()?,
    )?;
    let observations: Vec<_> = (0..n)
        .map(|i| {
            let location = i / v;
            Observation::new(
                i as u64,
                site(
                    [
                        (location % 11) as f64 * 0.3,
                        ((location / 11) % 7) as f64 * 0.4,
                        (location / 77) as f64 * 0.5,
                    ],
                    i % v,
                )?,
                (i as f64 * 0.3).sin(),
                0.01,
            )
        })
        .collect::<terrustrial_core::Result<Vec<_>>>()?;
    let targets: Vec<_> = (0..m)
        .map(|i| {
            let location = i / v;
            site(
                [
                    0.13 + (location % 7) as f64 * 0.2,
                    0.17 + ((location / 7) % 5) as f64 * 0.3,
                    0.23 + (location / 35) as f64 * 0.4,
                ],
                i % v,
            )
        })
        .collect::<terrustrial_core::Result<Vec<_>>>()?;
    let values: Vec<_> = observations.iter().map(Observation::value).collect();
    let mut mean = vec![0.; m];
    let mut noise = vec![0.; m];
    let mut draw = vec![0.; m];
    let budget = MemoryBudget::new(1 << 30);
    let cancel = Cancellation::default();
    let drift = Ordinary { variables: v };
    let options = || WorkspaceOptions {
        uncertainty: Uncertainty::Joint,
        tolerance: NumericalTolerance::default(),
        cancellation: &cancel,
    };
    let mut csv = csv::Writer::from_writer(std::io::stdout().lock());
    csv.write_record(["metric", "value"])?;
    for (key, value) in [
        ("observations", n),
        ("targets", m),
        ("variables", v),
        ("iterations", iterations),
        ("anisotropic", usize::from(!geometry.is_empty())),
        ("support_nodes", if geometry == "block" { 8 } else { 1 }),
    ] {
        csv.write_record([key, &value.to_string()])?;
    }
    if mode == "draw" {
        let mut workspace = GaussianWorkspace::new(n, m, v, &budget)?;
        let plan = workspace.prepare(&model, &observations, &targets, &drift, options())?;
        plan.estimate_into(&values, &mut mean)?;
        let (seconds, checksum, allocations) = measure(iterations, || {
            plan.sample_into(black_box(&mean), 981, 0, 7, &mut noise, &mut draw)?;
            black_box(&draw);
            Ok(draw
                .iter()
                .fold(0u64, |acc, x| acc.rotate_left(7) ^ x.to_bits()))
        })?;
        for (key, value) in [
            ("draw_seconds", seconds.to_string()),
            ("draw_checksum", checksum.to_string()),
            ("draw_allocations", allocations.count_total.to_string()),
            ("draw_allocated_bytes", allocations.bytes_total.to_string()),
        ] {
            csv.write_record([key, &value])?;
        }
        csv.flush()?;
        return Ok(());
    }
    let mut consume = |plan: &GaussianPlan| -> terrustrial_core::Result<u64> {
        plan.estimate_into(black_box(&values), &mut mean)?;
        plan.sample_into(&mean, 981, 0, 7, &mut noise, &mut draw)?;
        black_box(plan.weights());
        let mut checksum = 0u64;
        for &x in mean.iter().chain(&draw) {
            checksum = checksum.rotate_left(7) ^ x.to_bits();
        }
        for column in plan.covariance().unwrap().col_iter() {
            for &x in column.iter() {
                checksum = checksum.rotate_left(7) ^ x.to_bits();
            }
        }
        Ok(checksum)
    };
    if mode == "dump" {
        let mut workspace = GaussianWorkspace::new(n, m, v, &budget)?;
        let plan = workspace.prepare(&model, &observations, &targets, &drift, options())?;
        let prediction = plan.estimate(&values)?;
        let draws = plan.sample(&prediction, 981, 0, 7)?;
        for j in 0..m {
            for i in 0..n {
                csv.write_record([
                    format!("weight_{i}_{j}"),
                    plan.weights()[(i, j)].to_string(),
                ])?;
            }
            for i in 0..m {
                csv.write_record([
                    format!("covariance_{i}_{j}"),
                    plan.covariance().unwrap()[(i, j)].to_string(),
                ])?;
            }
            csv.write_record([format!("prediction_{j}"), prediction[j].to_string()])?;
            csv.write_record([format!("draw_{j}"), draws[j].to_string()])?;
        }
        csv.flush()?;
        return Ok(());
    }
    let mut expected = None;
    for reuse in [false, true] {
        let mut workspace = GaussianWorkspace::new(n, m, v, &budget)?;
        let (seconds, checksum, allocations) = if reuse {
            let mut conditioned = workspace.condition(&model, &observations, &drift, &cancel)?;
            measure(iterations, || {
                consume(conditioned.prepare(&targets, options())?)
            })?
        } else {
            measure(iterations, || {
                consume(workspace.prepare(&model, &observations, &targets, &drift, options())?)
            })?
        };
        if let Some(e) = expected {
            assert_eq!(checksum, e);
        } else {
            expected = Some(checksum);
        }
        for (key, value) in [
            ("seconds", seconds.to_string()),
            ("checksum", checksum.to_string()),
            (
                "allocations_per_preparation",
                allocations.count_total.to_string(),
            ),
            (
                "allocated_bytes_per_preparation",
                allocations.bytes_total.to_string(),
            ),
        ] {
            csv.write_record([format!("conditioned_{reuse}_{key}"), value])?;
        }
    }
    assert_eq!(budget.used(), 0);
    csv.flush()?;
    Ok(())
}

fn measure(
    mut iterations: usize,
    mut run: impl FnMut() -> terrustrial_core::Result<u64>,
) -> terrustrial_core::Result<(f64, u64, allocation_counter::AllocationInfo)> {
    for _ in 0..4 {
        black_box(run()?);
    }
    let start = Instant::now();
    let mut checksum = 0;
    while iterations > 0 {
        checksum = black_box(run()?);
        iterations -= 1;
    }
    let seconds = start.elapsed().as_secs_f64();
    let mut result = Ok(0);
    let allocations = allocation_counter::measure(|| result = run());
    assert_eq!(checksum, result?);
    Ok((seconds, checksum, allocations))
}
