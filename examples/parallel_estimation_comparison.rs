//! Matched scalar SK group workload through both public estimation drivers.
//! Input/index/group construction is excluded equally. Timed calls include
//! neighborhood selection, preparation, prediction and output accumulation.
use std::time::Instant;
use terrustrial::{
    geometry::{ellipsoid::Ellipsoid, support::Support},
    group_operators::{generalized_sequential_kriging, ConditioningParams},
    spatial_database::{
        coordinate_system::CoordinateSystem, group_provider::GroupProvider, SpatialAcceleratedDB,
    },
    systems::solved_systems::sk_system::SolvedLUSKSystemBuilder,
    variography::model_variograms::{
        composite::{CompositeVariogram, VariogramType},
        spherical::SphericalVariogram,
    },
};
use terrustrial_core::{
    data::{Group, Observation, Point3, Site, Target},
    estimation::{estimate_groups, EstimationOptions},
    execution::{Cancellation, ExecutionOptions, MemoryBudget},
    gaussian::{NumericalTolerance, Simple, Uncertainty},
    model::{Anisotropy, Coregionalization, Correlation, Lmc, Structure},
    spatial::{Neighborhood, SpatialIndex},
};
use ultraviolet::{DRotor3, DVec3};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let repetitions: usize = std::env::args()
        .nth(1)
        .ok_or("repeat count required")?
        .parse()?;
    if repetitions == 0 {
        return Err("positive repetitions required".into());
    }
    let hard_points: Vec<_> = (0..32)
        .map(|i| DVec3::new((i % 4) as f64, (i / 4 % 4) as f64, (i / 16) as f64))
        .collect();
    let values: Vec<_> = (0..32).map(|i| (i as f64 * 0.37).sin()).collect();
    let hard = SpatialAcceleratedDB::new(
        hard_points.iter().copied().map(Support::Point).collect(),
        values.clone(),
    );
    let core_hard = SpatialIndex::new(
        hard_points
            .iter()
            .zip(&values)
            .enumerate()
            .map(|(i, (p, v))| {
                Observation::new(
                    i as u64,
                    Site::point(Point3::new([p.x, p.y, p.z]).unwrap(), 0),
                    *v,
                    0.,
                )
                .unwrap()
            })
            .collect(),
    )?;
    let targets: Vec<_> = (0..32 * 32 * 32)
        .map(|i| {
            Support::Point(DVec3::new(
                0.0625 + (i % 32) as f64 * 0.125,
                0.0625 + (i / 32 % 32) as f64 * 0.125,
                0.0625 + (i / 1024) as f64 * 0.125,
            ))
        })
        .collect();
    let groups = GroupProvider::optimized_groups(&targets, 0.125, 0.125, 0.125, 2, 2, 2);
    let core_groups: Vec<_> = (0..groups.n_groups())
        .map(|i| Group {
            id: i as u64,
            targets: groups
                .get_group(i)
                .iter()
                .enumerate()
                .map(|(j, s)| {
                    let p = s.center();
                    Target {
                        id: 1000 + (groups.get_group_range(i).start + j) as u64,
                        site: Site::point(Point3::new([p.x, p.y, p.z]).unwrap(), 0),
                    }
                })
                .collect(),
        })
        .collect();
    let model = CompositeVariogram::new(vec![VariogramType::Spherical(SphericalVariogram::new(
        DVec3::broadcast(8.),
        1.7,
        DRotor3::identity(),
    ))]);
    let core_model = Lmc::new(vec![Structure {
        correlation: Correlation::Spherical,
        anisotropy: Anisotropy::isotropic(8.)?,
        coregionalization: Coregionalization::scalar(1.7)?,
    }])?;
    let conditioning = ConditioningParams::new(
        32,
        1,
        32,
        1,
        vec![],
        vec![],
        [f64::NEG_INFINITY, f64::INFINITY],
        usize::MAX,
        false,
        false,
    );
    let ellipsoid = Ellipsoid::new(
        100.,
        100.,
        100.,
        CoordinateSystem::new(DVec3::zero(), DRotor3::identity()),
    );
    let budget = MemoryBudget::new(1 << 28);
    let cancel = Cancellation::default();
    let legacy = || {
        generalized_sequential_kriging::estimate(
            &hard,
            &conditioning,
            &model,
            ellipsoid.clone(),
            &groups,
            SolvedLUSKSystemBuilder,
        )
    };
    let core = || -> terrustrial_core::Result<Vec<f64>> {
        let mut out = Vec::with_capacity(groups.n_nodes());
        estimate_groups(
            &core_model,
            &core_hard,
            core_groups.iter().cloned(),
            &Simple,
            EstimationOptions {
                neighborhood: Neighborhood {
                    maximum: 32,
                    radius: 100.,
                }
                .into(),
                mean: None,
                uncertainty: Uncertainty::None,
                tolerance: NumericalTolerance::default(),
                budget: &budget,
                cancellation: &cancel,
            },
            ExecutionOptions {
                batch_groups: 256,
                maximum_targets_per_group: groups.max_group_size(),
                bytes_per_group: 1 << 19,
            },
            |group| {
                out.extend_from_slice(&group.values);
                Ok(())
            },
        )?;
        Ok(out)
    };
    let reference = legacy();
    let candidate = core()?;
    assert_eq!(reference.len(), candidate.len());
    let maximum_difference = reference
        .iter()
        .zip(&candidate)
        .map(|(a, b)| {
            assert!(a.is_finite() && b.is_finite());
            (a - b).abs()
        })
        .fold(0.0_f64, f64::max);
    if maximum_difference > 1e-10 {
        return Err(format!("estimation disagreement {maximum_difference}").into());
    }
    let mut legacy_seconds = 0.;
    let mut core_seconds = 0.;
    let mut legacy_checksum = 0.;
    let mut core_checksum = 0.;
    // Alternate order to reduce thermal/order bias. Both have one warmup above.
    for i in 0..repetitions {
        for use_core in [i % 2 == 0, i % 2 != 0] {
            let start = Instant::now();
            let result = if use_core { core()? } else { legacy() };
            let seconds = start.elapsed().as_secs_f64();
            if use_core {
                core_seconds += seconds;
                core_checksum += result.iter().sum::<f64>();
            } else {
                legacy_seconds += seconds;
                legacy_checksum += result.iter().sum::<f64>();
            }
        }
    }
    assert_eq!(budget.used(), 0);
    let mut out = csv::Writer::from_writer(std::io::stdout().lock());
    out.write_record(["metric", "value"])?;
    for (key, value) in [
        ("groups", groups.n_groups().to_string()),
        ("targets", groups.n_nodes().to_string()),
        ("maximum_group_targets", groups.max_group_size().to_string()),
        ("hard_observations", "32".into()),
        ("repetitions", repetitions.to_string()),
        ("legacy_seconds", legacy_seconds.to_string()),
        ("core_seconds", core_seconds.to_string()),
        (
            "maximum_absolute_difference",
            maximum_difference.to_string(),
        ),
        ("legacy_checksum", legacy_checksum.to_string()),
        ("core_checksum", core_checksum.to_string()),
    ] {
        out.write_record([key, &value])?;
    }
    out.flush()?;
    Ok(())
}
