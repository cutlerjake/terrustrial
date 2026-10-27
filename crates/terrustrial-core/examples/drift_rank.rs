//! Cost and analytic accuracy of stable and nearly dependent drift bases.
//! Usage: drift_rank stable|near16|near24|near28 ITERATIONS
use std::{hint::black_box, time::Instant};
use terrustrial_core::{
    data::{Observation, Point3, Site},
    execution::{Cancellation, MemoryBudget},
    gaussian::{
        ExternalDrift, GaussianWorkspace, NumericalTolerance, Uncertainty, WorkspaceOptions,
    },
    model::{Anisotropy, Coregionalization, Correlation, Lmc, Structure},
    Result,
};

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    let mode = args.get(1).map(String::as_str).unwrap_or("stable");
    let exponent = match mode {
        "stable" => None,
        "near16" => Some(16),
        "near24" => Some(24),
        "near28" => Some(28),
        _ => panic!("expected stable, near16, near24 or near28"),
    };
    let iterations: usize = args
        .get(2)
        .map(|s| s.parse().expect("integer iterations"))
        .unwrap_or(25000);
    assert!((1..=1_000_000).contains(&iterations));
    let model = Lmc::new(vec![Structure {
        correlation: Correlation::Nugget,
        anisotropy: Anisotropy::isotropic(1.)?,
        coregionalization: Coregionalization::scalar(1.)?,
    }])?;
    let site = |x| Site::point(Point3::new([x, 0., 0.]).unwrap(), 0);
    let observations: Vec<_> = (0..25)
        .map(|i| {
            let x = (i as f64 - 12.) / 8.;
            Observation::new(i, site(x), 3. + 2. * x, 0.).unwrap()
        })
        .collect();
    let targets = [site(1. / 16.), site(11. / 16.)];
    let drift = ExternalDrift::new(2, |s: &Site, out: &mut [f64]| {
        let x = s.support.center().coordinates()[0];
        out.copy_from_slice(&[1., exponent.map_or(x, |e| 1. + 2_f64.powi(-e) * x)]);
        Ok(())
    })?;
    let budget = MemoryBudget::new(1 << 24);
    let cancellation = Cancellation::default();
    let options = || WorkspaceOptions {
        uncertainty: Uncertainty::Joint,
        tolerance: NumericalTolerance::default(),
        cancellation: &cancellation,
    };
    let mut workspace = GaussianWorkspace::new(25, 2, 2, &budget)?;
    let plan = workspace.prepare(&model, &observations, &targets, &drift, options())?;
    let xx: f64 = observations
        .iter()
        .map(|o| o.site().support.center().coordinates()[0].powi(2))
        .sum();
    let mut weight_error = 0.0_f64;
    let mut covariance_error = 0.0_f64;
    for (j, target) in targets.iter().enumerate() {
        let tx = target.support.center().coordinates()[0];
        for (i, observation) in observations.iter().enumerate() {
            let x = observation.site().support.center().coordinates()[0];
            weight_error =
                weight_error.max((plan.weights()[(i, j)] - (1. / 25. + x * tx / xx)).abs());
        }
        for (i, other) in targets.iter().enumerate() {
            let ox = other.support.center().coordinates()[0];
            let expected = if i == j { 1. } else { 0. } + 1. / 25. + tx * ox / xx;
            covariance_error =
                covariance_error.max((plan.covariance().unwrap()[(i, j)] - expected).abs());
        }
    }
    let tolerance = exponent.map_or(1e-12, |e| 32. * f64::EPSILON / 2_f64.powi(-e));
    assert!(weight_error < tolerance && covariance_error < tolerance);
    println!("metric,value\nmode,{mode}\niterations,{iterations}\nmaximum_weight_error,{weight_error}\nmaximum_covariance_error,{covariance_error}");
    for conditioned in [false, true] {
        let mut checksum = 0u64;
        let mut observe = |plan: &terrustrial_core::gaussian::GaussianPlan| {
            checksum = checksum.wrapping_add(black_box(plan).weights()[(0, 0)].to_bits());
        };
        let (elapsed, allocations) = if conditioned {
            let mut prepared = workspace.condition(&model, &observations, &drift, &cancellation)?;
            for _ in 0..4 {
                observe(prepared.prepare(&targets, options())?);
            }
            let start = Instant::now();
            for _ in 0..iterations {
                observe(prepared.prepare(&targets, options())?);
            }
            let elapsed = start.elapsed().as_secs_f64();
            let allocations = allocation_counter::measure(|| {
                black_box(prepared.prepare(&targets, options()).unwrap());
            });
            (elapsed, allocations)
        } else {
            for _ in 0..4 {
                observe(workspace.prepare(&model, &observations, &targets, &drift, options())?);
            }
            let start = Instant::now();
            for _ in 0..iterations {
                observe(workspace.prepare(&model, &observations, &targets, &drift, options())?);
            }
            let elapsed = start.elapsed().as_secs_f64();
            let allocations = allocation_counter::measure(|| {
                black_box(
                    workspace
                        .prepare(&model, &observations, &targets, &drift, options())
                        .unwrap(),
                );
            });
            (elapsed, allocations)
        };
        println!("conditioned_{conditioned}_seconds,{elapsed}\nconditioned_{conditioned}_checksum,{checksum}\nconditioned_{conditioned}_allocations,{}\nconditioned_{conditioned}_allocated_bytes,{}", allocations.count_total, allocations.bytes_total);
    }
    Ok(())
}
