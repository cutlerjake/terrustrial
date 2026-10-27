//! Matched scalar SK simulation through the core drivers and a seeded adapter
//! of the legacy driver. The legacy public simulate() hides its path and RNG.
//! This adapter preserves its full-path parallel preparation and serial draws,
//! using unchanged legacy search/solver primitives. Only path and innovations
//! are injected. It is not a timing of the unmodified public simulate() call.
use rand::{seq::SliceRandom, RngExt};
use rand_distr::StandardNormal;
use rayon::prelude::*;
use std::{ops::Range, time::Instant};
use terrustrial::{
    geometry::{ellipsoid::Ellipsoid, support::Support},
    group_operators::ConditioningParams,
    spatial_database::{
        coordinate_system::CoordinateSystem, group_provider::GroupProvider, ConditioningProvider,
        FilterMapResult, FilterMappedIterNearest, SpatialAcceleratedDB,
    },
    systems::{
        lu::LUSystem,
        solved_systems::{
            sk_system::{SolvedLUSKSystem, SolvedLUSKSystemBuilder},
            SolvedLUSystem, SolvedSystemBuilder,
        },
    },
    variography::model_variograms::{
        composite::{CompositeVariogram, VariogramType},
        spherical::SphericalVariogram,
    },
};
use terrustrial_core::{
    data::{Group, Observation, Point3, Site, Target},
    execution::{random_stream, Cancellation, ExecutionOptions, MemoryBudget},
    gaussian::Simple,
    model::{Anisotropy, Coregionalization, Correlation, Lmc, Structure},
    simulation::{self, SimulationOptions},
    spatial::{Neighborhood, ObservationSink, ObservationSource, SpatialIndex},
};
use ultraviolet::{DRotor3, DVec3};

const HARD_LIMIT: usize = 16;
const SIM_LIMIT: usize = 32;
const RADIUS: f64 = 2.0;
const TARGET_ID: u64 = 1000;
const SEED: u64 = 2026;
const REALIZATION: u64 = 7;

struct Workload {
    hard: SpatialAcceleratedDB<f64>,
    core_hard: SpatialIndex,
    groups: GroupProvider,
    path: Vec<usize>,
    core_groups: Vec<Group>,
    model: CompositeVariogram,
    core_model: Lmc,
    hard_params: ConditioningParams,
    sim_params: ConditioningParams,
    ellipsoid: Ellipsoid,
}

fn site(p: DVec3) -> Site {
    Site::point(Point3::new([p.x, p.y, p.z]).unwrap(), 0)
}
fn center(group: &[Support]) -> DVec3 {
    group.iter().fold(DVec3::zero(), |sum, p| sum + p.center()) / group.len() as f64
}
fn params(maximum: usize, minimum: usize) -> ConditioningParams {
    ConditioningParams::new(
        maximum,
        minimum,
        maximum,
        0,
        vec![],
        vec![],
        [f64::NEG_INFINITY, f64::INFINITY],
        usize::MAX,
        false,
        false,
    )
}
impl Workload {
    fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let hard_points: Vec<_> = (0..125)
            .map(|i| DVec3::new((i % 5) as f64, (i / 5 % 5) as f64, (i / 25) as f64))
            .collect();
        let values: Vec<_> = (0..125).map(|i| (i as f64 * 0.37).sin()).collect();
        let hard = SpatialAcceleratedDB::new(
            hard_points.iter().copied().map(Support::Point).collect(),
            values.clone(),
        );
        let core_hard = SpatialIndex::new(
            hard_points
                .iter()
                .zip(&values)
                .enumerate()
                .map(|(i, (&p, &v))| Observation::new(i as u64, site(p), v, 0.).unwrap())
                .collect(),
        )?;
        // Small deterministic jitter avoids the legacy query's unspecified tie
        // ordering. Preflight still checks every selected ID, never assumes it.
        let mut coordinates = random_stream(SEED, 0, 0, 2);
        let targets: Vec<_> = (0..32 * 32 * 32)
            .map(|i| {
                Support::Point(DVec3::new(
                    0.0625 + (i % 32) as f64 * 0.125 + coordinates.random_range(-0.002..0.002),
                    0.0625 + (i / 32 % 32) as f64 * 0.125 + coordinates.random_range(-0.002..0.002),
                    0.0625 + (i / 1024) as f64 * 0.125 + coordinates.random_range(-0.002..0.002),
                ))
            })
            .collect();
        let groups = GroupProvider::optimized_groups(&targets, 0.125, 0.125, 0.125, 2, 2, 2);
        assert_eq!(groups.n_nodes(), targets.len());
        let mut path: Vec<_> = (0..groups.n_groups()).collect();
        path.shuffle(&mut random_stream(SEED, 0, 0, 1));
        let core_groups = path
            .iter()
            .enumerate()
            .map(|(rank, &index)| Group {
                id: rank as u64,
                targets: groups
                    .get_group_range(index)
                    .map(|j| Target {
                        id: TARGET_ID + j as u64,
                        site: site(groups.get_supports()[j].center()),
                    })
                    .collect(),
            })
            .collect();
        Ok(Self {
            hard,
            core_hard,
            groups,
            path,
            core_groups,
            model: CompositeVariogram::new(vec![VariogramType::Spherical(
                SphericalVariogram::new(DVec3::broadcast(8.), 1.7, DRotor3::identity()),
            )]),
            core_model: Lmc::new(vec![Structure {
                correlation: Correlation::Spherical,
                anisotropy: Anisotropy::isotropic(8.)?,
                coregionalization: Coregionalization::scalar(1.7)?,
            }])?,
            hard_params: params(HARD_LIMIT, 1),
            sim_params: params(SIM_LIMIT, 0),
            ellipsoid: Ellipsoid::new(
                RADIUS,
                RADIUS,
                RADIUS,
                CoordinateSystem::new(DVec3::zero(), DRotor3::identity()),
            ),
        })
    }

    fn locations(&self) -> SpatialAcceleratedDB<usize> {
        let mut order = vec![0; self.groups.n_nodes()];
        for (rank, &group) in self.path.iter().enumerate() {
            order[self.groups.get_group_range(group)].fill(rank);
        }
        SpatialAcceleratedDB::new(self.groups.get_supports().to_vec(), order)
    }

    /// Untimed ID-set audit against the evolving core index. Reduction orders
    /// differ, so numerical agreement below is tolerance-based, not bitwise.
    fn audit(&self) -> Result<u64, Box<dyn std::error::Error>> {
        let locations = self.locations();
        let mut prefix = SpatialIndex::new(vec![])?;
        let cancel = Cancellation::default();
        let mut hash = 0_u64;
        for (rank, &index) in self.path.iter().enumerate() {
            let mut ellipsoid = self.ellipsoid.clone();
            let at = center(self.groups.get_group(index));
            ellipsoid.translate_to(at);
            let (hard_ids, _, _, valid) = self.hard.query(&at, &ellipsoid, &self.hard_params);
            assert!(valid);
            let previous = FilterMappedIterNearest::new(&locations, |elem| {
                if !ellipsoid.may_contain_local_point_at_sq_dist(elem.sq_dist) {
                    FilterMapResult::ExitEarly
                } else if elem.data < rank {
                    FilterMapResult::Mapped(elem)
                } else {
                    FilterMapResult::Ignore
                }
            });
            let (sim_ids, _, _, _) = previous.query(&at, &ellipsoid, &self.sim_params);
            let group = &self.core_groups[rank];
            let mut point = [0.; 3];
            for target in &group.targets {
                for (axis, x) in point.iter_mut().enumerate() {
                    *x += target.site.support.center().coordinates()[axis]
                        / group.targets.len() as f64;
                }
            }
            let point = Point3::new(point)?;
            for (kind, mut legacy_ids, found) in [
                (
                    "hard",
                    hard_ids.into_iter().map(|i| i as u64).collect::<Vec<_>>(),
                    self.core_hard.nearest(
                        point,
                        Neighborhood {
                            maximum: HARD_LIMIT,
                            radius: RADIUS,
                        },
                        &cancel,
                    )?,
                ),
                (
                    "simulated",
                    sim_ids.into_iter().map(|i| TARGET_ID + i as u64).collect(),
                    prefix.nearest(
                        point,
                        Neighborhood {
                            maximum: SIM_LIMIT,
                            radius: RADIUS,
                        },
                        &cancel,
                    )?,
                ),
            ] {
                legacy_ids.sort_unstable();
                let mut core_ids: Vec<_> = found.iter().map(Observation::id).collect();
                core_ids.sort_unstable();
                if legacy_ids != core_ids {
                    return Err(format!("{kind} neighborhood mismatch at rank {rank}: legacy {legacy_ids:?}, core {core_ids:?}").into());
                }
                hash = hash.rotate_left(7) ^ legacy_ids.len() as u64;
                for id in legacy_ids {
                    hash = hash.rotate_left(7) ^ id;
                }
            }
            for target in &group.targets {
                prefix.append(Observation::new(target.id, target.site.clone(), 0., 0.)?)?;
            }
        }
        Ok(hash)
    }

    fn legacy(&self, realization: u64) -> Vec<f64> {
        let locations = self.locations();
        // Preserve the legacy driver's separate parallel hard-conditioning pass.
        let conditioned: Vec<_> = (0..self.groups.n_groups())
            .into_par_iter()
            .map_with(self.ellipsoid.clone(), |ellipsoid, index| {
                let at = center(self.groups.get_group(index));
                ellipsoid.translate_to(at);
                self.hard.query(&at, ellipsoid, &self.hard_params).3
            })
            .collect();
        let local = LUSystem::new(self.groups.max_group_size(), HARD_LIMIT + SIM_LIMIT);
        struct Prepared {
            range: Range<usize>,
            system: SolvedLUSKSystem,
            simulated: Vec<usize>,
            values: Vec<f64>,
        }
        let plans: Vec<_> = self
            .path
            .par_iter()
            .copied()
            .enumerate()
            .map_with(
                (
                    self.ellipsoid.clone(),
                    local.clone(),
                    vec![],
                    vec![],
                    vec![],
                    vec![],
                ),
                |(ellipsoid, local, h, pt, var, ind), (rank, index)| {
                    let group = self.groups.get_group(index);
                    assert!(conditioned[index], "invalid fixture conditioning");
                    let at = center(group);
                    ellipsoid.translate_to(at);
                    let (_, values, mut supports, valid) =
                        self.hard.query(&at, ellipsoid, &self.hard_params);
                    assert!(valid);
                    let previous = FilterMappedIterNearest::new(&locations, |elem| {
                        if !ellipsoid.may_contain_local_point_at_sq_dist(elem.sq_dist) {
                            FilterMapResult::ExitEarly
                        } else if elem.data < rank && conditioned[self.path[elem.data]] {
                            FilterMapResult::Mapped(elem)
                        } else {
                            FilterMapResult::Ignore
                        }
                    });
                    let (simulated, _, sim_supports, _) =
                        previous.query(&at, ellipsoid, &self.sim_params);
                    supports.extend_from_slice(&sim_supports);
                    let n_cond = supports.len();
                    supports.extend_from_slice(group);
                    local.build_cov_matrix(
                        n_cond,
                        group.len(),
                        &supports,
                        &self.model,
                        h,
                        pt,
                        var,
                        ind,
                    );
                    Prepared {
                        range: self.groups.get_group_range(index),
                        system: SolvedLUSKSystemBuilder
                            .build(local)
                            .expect("legacy factorization"),
                        simulated,
                        values,
                    }
                },
            )
            .collect();
        let mut output = vec![0.; self.groups.n_nodes()];
        for (rank, mut plan) in plans.into_iter().enumerate() {
            plan.values
                .extend(plan.simulated.into_iter().map(|i| output[i]));
            plan.system.populate_cond_values_est(&plan.values);
            let mut rng = random_stream(SEED, realization, rank as u64, 0);
            for j in 0..plan.system.n_sim {
                plan.system.w_vec[(plan.system.n_cond + j, 0)] = rng.sample(StandardNormal);
            }
            output[plan.range].copy_from_slice(&plan.system.simulate());
        }
        output
    }

    fn core(&self, realization: u64, parallel: bool) -> terrustrial_core::Result<Vec<f64>> {
        let budget = MemoryBudget::new(1 << 28);
        let cancel = Cancellation::default();
        let options = SimulationOptions {
            seed: SEED,
            realization,
            mean: None,
            hard_neighborhood: Neighborhood {
                maximum: HARD_LIMIT,
                radius: RADIUS,
            }
            .into(),
            simulated_neighborhood: Neighborhood {
                maximum: SIM_LIMIT,
                radius: RADIUS,
            }
            .into(),
            budget: &budget,
            cancellation: &cancel,
        };
        let mut prefix = SpatialIndex::new(vec![])?;
        let mut out = vec![f64::NAN; self.groups.n_nodes()];
        let mut collect = |group: &[Observation]| {
            for o in group {
                out[(o.id() - TARGET_ID) as usize] = o.value();
            }
            Ok(())
        };
        if parallel {
            simulation::parallel::simulate_groups(
                &self.core_model,
                &self.core_hard,
                &mut prefix,
                self.core_groups.iter().cloned(),
                &Simple,
                options,
                ExecutionOptions {
                    batch_groups: 256,
                    maximum_targets_per_group: self.groups.max_group_size(),
                    bytes_per_group: 1 << 19,
                },
                &mut collect,
            )?;
        } else {
            simulation::simulate_groups(
                &self.core_model,
                &self.core_hard,
                &mut prefix,
                self.core_groups.iter().cloned(),
                &Simple,
                options,
                &mut collect,
            )?;
        }
        assert_eq!(budget.used(), 0);
        Ok(out)
    }
}

fn checksum(values: &[f64]) -> u64 {
    values.iter().fold(0_u64, |h, v| {
        assert!(v.is_finite());
        h.rotate_left(7) ^ v.to_bits()
    })
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let repetitions: usize = std::env::args()
        .nth(1)
        .ok_or("positive repeat count required")?
        .parse()?;
    if repetitions == 0 {
        return Err("positive repetitions required".into());
    }
    let work = Workload::new()?;
    let selection_checksum = work.audit()?;
    let mut maximum_difference = 0_f64;
    let mut reference_checksums = [0; 3];
    // Full output preflight over three independent realizations. This also warms
    // every driver. Matching scalar SK is not evidence for multivariate legacy SGS.
    for realization in REALIZATION..REALIZATION + 3 {
        let results = [
            work.legacy(realization),
            work.core(realization, false)?,
            work.core(realization, true)?,
        ];
        for ((&old, &serial), &parallel) in results[0].iter().zip(&results[1]).zip(&results[2]) {
            assert!(old.is_finite() && serial.is_finite() && parallel.is_finite());
            assert_eq!(
                serial.to_bits(),
                parallel.to_bits(),
                "serial/parallel mismatch"
            );
            let difference = (old - parallel).abs();
            maximum_difference = maximum_difference.max(difference);
            if difference > 1e-9 * (1. + old.abs().max(parallel.abs())) {
                return Err(format!("legacy/core mismatch: {old} versus {parallel}").into());
            }
        }
        if realization == REALIZATION {
            reference_checksums = results.each_ref().map(|r| checksum(r));
        }
    }
    let mut seconds = [0.; 3];
    for repeat in 0..repetitions {
        // Rotate the starting driver and reverse alternate cycles.
        for step in 0..3 {
            let driver = (repeat + if repeat % 2 == 0 { step } else { 2 - step }) % 3;
            let start = Instant::now();
            let result = match driver {
                0 => work.legacy(REALIZATION),
                1 => work.core(REALIZATION, false)?,
                _ => work.core(REALIZATION, true)?,
            };
            seconds[driver] += start.elapsed().as_secs_f64();
            assert_eq!(
                checksum(&result),
                reference_checksums[driver],
                "changed realization"
            );
        }
    }
    let mut out = csv::Writer::from_writer(std::io::stdout().lock());
    out.write_record(["metric", "value"])?;
    for (name, value) in [
        ("groups", work.groups.n_groups().to_string()),
        ("targets", work.groups.n_nodes().to_string()),
        (
            "maximum_group_targets",
            work.groups.max_group_size().to_string(),
        ),
        ("hard_observations", "125".into()),
        ("maximum_hard_neighbors", HARD_LIMIT.to_string()),
        ("maximum_simulated_neighbors", SIM_LIMIT.to_string()),
        ("search_radius", RADIUS.to_string()),
        ("batch_groups", "256".into()),
        ("repetitions", repetitions.to_string()),
        ("checked_realizations", "3".into()),
        ("selection_checksum", selection_checksum.to_string()),
        (
            "maximum_absolute_difference",
            maximum_difference.to_string(),
        ),
        ("legacy_adapter_seconds", seconds[0].to_string()),
        ("core_serial_seconds", seconds[1].to_string()),
        ("core_parallel_seconds", seconds[2].to_string()),
        (
            "legacy_adapter_checksum",
            reference_checksums[0].to_string(),
        ),
        ("core_serial_checksum", reference_checksums[1].to_string()),
        ("core_parallel_checksum", reference_checksums[2].to_string()),
    ] {
        out.write_record([name, &value])?;
    }
    out.flush()?;
    Ok(())
}
