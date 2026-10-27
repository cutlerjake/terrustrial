//! Grouped joint SGS with explicit path and conditioning dependencies.
mod buffers;
mod diagnostics;
use crate::execution::parallel::validate_order;
use crate::gaussian::group::{group_center, CanonicalWorkspace, Selection};
use buffers::SimulationInputs;
pub use diagnostics::{
    simulate_checkpointed_identified_with_residuals, simulate_checkpointed_with_residuals,
    simulate_groups_with_residuals, GroupResiduals,
};
use diagnostics::{Mode, Unchecked};
mod drawing;
use drawing::DrawingWorkspace;
pub mod parallel;
use crate::execution::buffer::grow;
use crate::{
    data::{Group, Observation, Site},
    error::{checked_len, invalid},
    execution::{Cancellation, MemoryBudget},
    gaussian::{
        Drift, GaussianPlan, GaussianWorkspace, KnownMean, NumericalTolerance, Uncertainty,
        WorkspaceOptions,
    },
    model::Covariance,
    spatial::{
        exact::ExactWorkspace, search::QueryWorkspace, NeighborhoodQuery, ObservationSink,
        ObservationSource,
    },
    Error, Result,
};
#[cfg(test)]
use crate::{
    data::{Point3, Target},
    gaussian::group::canonicalize_observations,
};

pub struct SimulationOptions<'a> {
    pub seed: u64,
    pub realization: u64,
    /// Known support-aware mean; None preserves the zero-mean path. With Some,
    /// covariance and drift constraints describe residuals. Stores and output
    /// retain original-scale values. Include this field's full definition and
    /// data in checkpoint identity. Callbacks must be deterministic and immutable.
    pub mean: Option<&'a dyn KnownMean>,
    pub hard_neighborhood: NeighborhoodQuery<'a>,
    pub simulated_neighborhood: NeighborhoodQuery<'a>,
    pub budget: &'a MemoryBudget,
    pub cancellation: &'a Cancellation,
}
impl SimulationOptions<'_> {
    fn validate(&self) -> Result<()> {
        self.hard_neighborhood.validate()?;
        self.simulated_neighborhood.validate()?;
        self.cancellation.check()
    }
}

/// Consumes groups in the supplied order. Distinct realizations can run in parallel
/// with independent stores. Exact target observations are included even if excluded
/// by a group-center neighborhood. Consistent repeated hard supports are coalesced.
///
/// A failed append can leave a partial group. Use `simulate_checkpointed` with the
/// `sqlite` feature for transactional group output and reproducible resumption.
pub fn simulate_groups<C, D, S, H>(
    model: &C,
    hard: &H,
    simulated: &mut S,
    groups: impl IntoIterator<Item = Group>,
    drift: &D,
    options: SimulationOptions<'_>,
    output: impl FnMut(&[Observation]) -> Result<()>,
) -> Result<()>
where
    C: Covariance,
    D: Drift,
    S: ObservationSource + ObservationSink,
    H: ObservationSource + ?Sized,
{
    run_groups::<Unchecked, _, _, _, _>(
        model,
        hard,
        simulated,
        groups,
        drift,
        options,
        |_, _| Ok(()),
        output,
    )
}
#[allow(clippy::too_many_arguments)]
fn run_groups<
    M: Mode,
    C: Covariance,
    D: Drift,
    S: ObservationSource + ObservationSink,
    H: ObservationSource + ?Sized,
>(
    model: &C,
    hard: &H,
    simulated: &mut S,
    groups: impl IntoIterator<Item = Group>,
    drift: &D,
    options: SimulationOptions<'_>,
    mut inspect: impl FnMut(&Group, &M::Report) -> Result<()>,
    mut output: impl FnMut(&[Observation]) -> Result<()>,
) -> Result<()> {
    options.validate()?;
    let mut workspace = SimulationWorkspace::<M>::new(&options)?;
    let mut previous = None;
    for group in groups {
        validate_order(group.id, &mut previous)?;
        let (result, report) = simulate_group(
            model,
            hard,
            simulated,
            &group,
            drift,
            &options,
            &mut workspace,
        )?;
        inspect(&group, &report)?;
        options.cancellation.check()?;
        for observation in result {
            simulated.append(*observation)?;
        }
        output(result)?;
    }
    Ok(())
}

/// Simulation configuration plus the identity of the full reproducibility contract.
pub struct CheckpointedOptions<'a> {
    pub simulation: SimulationOptions<'a>,
    pub identity: [u8; 32],
}

/// Derive identity from current structured inputs before starting or resuming.
/// This performs a bounded-memory prepass of hard data and the FULL path, even
/// on resume; unlike the low-level driver, a remaining suffix is not sufficient.
/// The path factory is called once per pass and must reproduce exactly the same
/// groups. It can reopen a disk-backed iterator; no iterator or full path is
/// cloned by the library. Factory/iterator buffers belong to the caller. Sources,
/// callbacks and path must remain immutable during both passes and simulation.
/// For SQLite conditioning, retain one `SqliteReadPool::open_snapshot` pool
/// across both passes and simulation to isolate reads from concurrent writes.
/// Iterator items are `Result<Group>` so disk/read failures propagate. A failure
/// during the simulation pass retains only the already committed prefix.
///
/// Built-in LMC and drifts have canonical identities. Custom components can use
/// `identity::Identified`; callback data, executable/build and backend identities
/// remain explicit caller responsibilities. Hashing does not authenticate them.
/// The library never silently falls back to an unchecked identity on error.
///
/// ```
/// # #[cfg(feature = "sqlite")]
/// # mod sqlite_example {
/// use terrustrial_core::{simulation::identity::RunProvenance, model::Lmc, gaussian::Simple, data::{Observation, Group}, simulation::{SimulationOptions, simulate_checkpointed_identified}, storage::SqliteStore};
/// fn resume(model: &Lmc, hard: &[Observation], full_path: &[Group],
///     store: &mut SqliteStore, options: SimulationOptions<'_>, provenance: RunProvenance)
///     -> terrustrial_core::Result<()> {
///     simulate_checkpointed_identified(model, hard, store,
///         || Ok(full_path.iter().cloned().map(Ok)), &Simple, options, provenance, |_| Ok(()))
/// }
/// # }
/// ```
#[allow(clippy::too_many_arguments)]
pub fn simulate_checkpointed_identified<C, D, H, G, S>(
    model: &C,
    hard: &H,
    simulated: &mut S,
    groups: impl Fn() -> Result<G>,
    drift: &D,
    options: SimulationOptions<'_>,
    provenance: identity::RunProvenance,
    committed: impl FnMut(&[Observation]) -> Result<()>,
) -> Result<()>
where
    C: Covariance + crate::identity::CanonicalFingerprint,
    D: Drift + crate::identity::CanonicalFingerprint,
    H: ObservationSource + ?Sized,
    S: crate::storage::CheckpointStore,
    G: IntoIterator<Item = Result<Group>>,
{
    options.validate()?;
    provenance.validate_mean(options.mean.is_some())?;
    let hard_id = crate::identity::observations(hard, options.cancellation)?;
    let path_id = crate::identity::try_path(groups()?, options.cancellation)?;
    let identity = identity::gaussian_run(model, drift, hard_id, path_id, &options, provenance)?;
    checkpointed_fallible(
        model,
        hard,
        simulated,
        groups()?,
        drift,
        CheckpointedOptions {
            simulation: options,
            identity: identity.to_bytes(),
        },
        committed,
    )
}

/// Durable simulation with atomic output/checkpoint commits. Supply the same path
/// (completed groups are skipped) or its remaining suffix. The supplied identity
/// must fingerprint the full run contract described by `storage::Checkpoint`.
///
/// Notifications run AFTER commit. They are not an exactly-once external sink:
/// a failed notification may already have a durable group. Read committed records
/// from the store for reliable export. Cancellation never leaves half a group.
pub fn simulate_checkpointed<C, D, H, S>(
    model: &C,
    hard: &H,
    simulated: &mut S,
    groups: impl IntoIterator<Item = Group>,
    drift: &D,
    options: CheckpointedOptions<'_>,
    committed: impl FnMut(&[Observation]) -> Result<()>,
) -> Result<()>
where
    C: Covariance,
    D: Drift,
    H: ObservationSource + ?Sized,
    S: crate::storage::CheckpointStore,
{
    checkpointed_fallible(
        model,
        hard,
        simulated,
        groups.into_iter().map(Ok),
        drift,
        options,
        committed,
    )
}

fn checkpointed_fallible<C, D, H, S>(
    model: &C,
    hard: &H,
    simulated: &mut S,
    groups: impl IntoIterator<Item = Result<Group>>,
    drift: &D,
    options: CheckpointedOptions<'_>,
    committed: impl FnMut(&[Observation]) -> Result<()>,
) -> Result<()>
where
    C: Covariance,
    D: Drift,
    H: ObservationSource + ?Sized,
    S: crate::storage::CheckpointStore,
{
    run_checkpointed::<Unchecked, _, _, _, _>(
        model,
        hard,
        simulated,
        groups,
        drift,
        options,
        |_, _| Ok(()),
        committed,
    )
}
#[allow(clippy::too_many_arguments)]
fn run_checkpointed<
    M: Mode,
    C: Covariance,
    D: Drift,
    H: ObservationSource + ?Sized,
    S: crate::storage::CheckpointStore,
>(
    model: &C,
    hard: &H,
    simulated: &mut S,
    groups: impl IntoIterator<Item = Result<Group>>,
    drift: &D,
    options: CheckpointedOptions<'_>,
    mut inspect: impl FnMut(&Group, &M::Report) -> Result<()>,
    mut committed: impl FnMut(&[Observation]) -> Result<()>,
) -> Result<()> {
    let CheckpointedOptions {
        simulation: options,
        identity,
    } = options;
    options.validate()?;
    let mut workspace = SimulationWorkspace::<M>::new(&options)?;
    let mut checkpoint = simulated.checkpoint()?;
    if checkpoint.as_ref().is_some_and(|c| c.identity != identity) {
        return Err(invalid("simulation checkpoint identity mismatch"));
    }
    if checkpoint.is_none() && !simulated.is_empty()? {
        return Err(invalid("uncheckpointed simulation store must be empty"));
    }
    let mut previous = None;
    for group in groups {
        let group = group?;
        validate_order(group.id, &mut previous)?;
        if checkpoint
            .as_ref()
            .is_some_and(|c| group.id <= c.completed_group)
        {
            continue;
        }
        let (result, report) = simulate_group(
            model,
            hard,
            simulated,
            &group,
            drift,
            &options,
            &mut workspace,
        )?;
        inspect(&group, &report)?;
        options.cancellation.check()?;
        let next = crate::storage::Checkpoint {
            identity,
            completed_group: group.id,
        };
        simulated.commit_group(result, checkpoint.as_ref(), &next, options.cancellation)?;
        checkpoint = Some(next);
        committed(result)?;
    }
    Ok(())
}
struct Searches<'a> {
    hard: QueryWorkspace<'a>,
    simulated: QueryWorkspace<'a>,
    hard_exact: ExactWorkspace,
    simulated_exact: ExactWorkspace,
    _exact_metadata: crate::execution::Reservation,
    solver: Option<GaussianWorkspace>,
    canonical: CanonicalWorkspace,
}
impl<'a> Searches<'a> {
    fn new(options: &SimulationOptions<'a>) -> Result<Self> {
        Self::with_neighborhoods(
            options.hard_neighborhood,
            options.simulated_neighborhood,
            options.budget,
        )
    }
    fn with_neighborhoods(
        hard: NeighborhoodQuery<'a>,
        simulated: NeighborhoodQuery<'a>,
        budget: &MemoryBudget,
    ) -> Result<Self> {
        Ok(Self {
            hard_exact: ExactWorkspace::default(),
            simulated_exact: ExactWorkspace::default(),
            _exact_metadata: budget.reserve(
                (2 * std::mem::size_of::<ExactWorkspace>()
                    + std::mem::size_of::<crate::execution::Reservation>()) as u64,
            )?,
            solver: None,
            canonical: CanonicalWorkspace::new(budget)?,
            hard: QueryWorkspace::new(hard, budget)?,
            simulated: QueryWorkspace::new(simulated, budget)?,
        })
    }
}
struct SimulationWorkspace<'a, M: Mode = Unchecked> {
    diagnostic: M,
    _diagnostic_metadata: Option<crate::execution::Reservation>,
    searches: Searches<'a>,
    drawing: DrawingWorkspace,
    inputs: SimulationInputs,
}
impl<'a, M: Mode> SimulationWorkspace<'a, M> {
    fn new(options: &SimulationOptions<'a>) -> Result<Self> {
        Ok(Self {
            diagnostic: M::default(),
            _diagnostic_metadata: if std::mem::size_of::<M>() == 0 {
                None
            } else {
                Some(options.budget.reserve(std::mem::size_of::<M>() as u64)?)
            },
            searches: Searches::new(options)?,
            drawing: DrawingWorkspace::new(options.budget)?,
            inputs: SimulationInputs::new(options.budget)?,
        })
    }
}
fn simulate_group<'a, M: Mode, C, D, S, H>(
    model: &C,
    hard: &H,
    simulated: &S,
    group: &Group,
    drift: &D,
    options: &SimulationOptions<'_>,
    workspace: &'a mut SimulationWorkspace<'_, M>,
) -> Result<(&'a [Observation], M::Report)>
where
    C: Covariance,
    D: Drift,
    S: ObservationSource,
    H: ObservationSource + ?Sized,
{
    let searches = &mut workspace.searches;
    let inputs = &mut workspace.inputs;
    inputs.selection.observations.clear();
    visit_group_selection(
        hard,
        simulated,
        group,
        searches,
        options.cancellation,
        |part| inputs.selection.extend(&part),
    )?;
    searches.canonical.compact(
        &mut inputs.selection.observations,
        true,
        options.cancellation,
    )?;
    inputs.fill_sites(group)?;
    let observations = &inputs.selection.observations;
    let sites = &inputs.sites;
    let solver = GaussianWorkspace::cached(
        &mut searches.solver,
        observations.len(),
        sites.len(),
        drift.columns(),
        options.budget,
    )?;
    let (plan, report) = workspace.diagnostic.prepare(
        solver,
        model,
        observations,
        sites,
        drift,
        options.budget,
        WorkspaceOptions {
            uncertainty: Uncertainty::Joint,
            tolerance: NumericalTolerance::default(),
            cancellation: options.cancellation,
        },
    )?;
    Ok((
        workspace.drawing.draw(plan, observations, group, options)?,
        report,
    ))
}
#[cfg(test)]
fn select_group<S: ObservationSource + ?Sized, H: ObservationSource + ?Sized>(
    hard: &H,
    simulated: &S,
    group: &Group,
    searches: &mut Searches<'_>,
    cancel: &Cancellation,
) -> Result<Vec<Observation>> {
    let mut selected = Vec::new();
    visit_group_selection(hard, simulated, group, searches, cancel, |part| {
        if part.is_empty() {
            return Ok(());
        }
        if selected.is_empty() {
            selected = part.into_owned();
        } else {
            selected
                .try_reserve(part.len())
                .map_err(|_| invalid("simulation selection allocation"))?;
            selected.extend_from_slice(&part);
        }
        Ok(())
    })?;
    Ok(selected)
}
fn visit_group_selection<S: ObservationSource + ?Sized, H: ObservationSource + ?Sized>(
    hard: &H,
    simulated: &S,
    group: &Group,
    searches: &mut Searches<'_>,
    cancel: &Cancellation,
    mut append: impl FnMut(std::borrow::Cow<'_, [Observation]>) -> Result<()>,
) -> Result<()> {
    // Source views may change between groups and deferred-read retries. Misses
    // are reusable only within this one selection attempt.
    searches.hard_exact.reset();
    searches.simulated_exact.reset();
    let center = group_center(group, cancel)?;
    append(searches.hard.select(hard, center, cancel)?)?;
    append(searches.simulated.select(simulated, center, cancel)?)?;
    // Preserve source overrides, ordering and exact conditioning independently
    // of neighborhood cutoffs. Each result can be borrowed or transferred.
    for target in &group.targets {
        cancel.check()?;
        append(searches.hard_exact.at_site(hard, &target.site)?)?;
        append(searches.simulated_exact.at_site(simulated, &target.site)?)?;
    }
    Ok(())
}
#[cfg(test)]
mod canonical_tests {
    use super::*;
    #[test]
    fn discarded_exact_alias_ids_still_require_consistent_records() {
        let site = |x| Site::point(Point3::new([x, 0., 0.]).unwrap(), 0);
        let mut observations = vec![
            Observation::new(1, site(0.), 3., 0.).unwrap(),
            Observation::new(2, site(0.), 3., 0.).unwrap(),
            Observation::new(2, site(1.), 3., 0.).unwrap(),
        ];
        assert!(canonicalize_observations(&mut observations, true).is_err());
    }
}

#[cfg(test)]
mod input_tests {
    use super::*;
    use crate::{
        gaussian::{Ordinary, Simple},
        model::{Anisotropy, Coregionalization, Correlation, Lmc, Structure},
        spatial::{Neighborhood, SpatialIndex},
    };
    fn options<'a>(
        budget: &'a MemoryBudget,
        cancel: &'a Cancellation,
        mean: Option<&'a dyn KnownMean>,
    ) -> SimulationOptions<'a> {
        SimulationOptions {
            seed: 765,
            realization: 4,
            mean,
            hard_neighborhood: Neighborhood {
                maximum: 32,
                radius: 100.,
            }
            .into(),
            simulated_neighborhood: Neighborhood {
                maximum: 32,
                radius: 100.,
            }
            .into(),
            budget,
            cancellation: cancel,
        }
    }
    #[test]
    fn warmed_serial_selection_preparation_and_draw_allocate_nothing() {
        fn check<M: Mode>(drift: &impl Drift) {
            let model = Lmc::new(vec![Structure {
                correlation: Correlation::Exponential,
                anisotropy: Anisotropy::isotropic(3.).unwrap(),
                coregionalization: Coregionalization::scalar(1.).unwrap(),
            }])
            .unwrap();
            let source = |offset: u64| {
                SpatialIndex::new(
                    (0..32)
                        .map(|i| {
                            Observation::new(
                                offset + i,
                                Site::point(
                                    Point3::new([(offset + i) as f64 * 0.2, 0., 0.]).unwrap(),
                                    0,
                                ),
                                (i as f64).sin(),
                                0.01,
                            )
                            .unwrap()
                        })
                        .collect(),
                )
                .unwrap()
            };
            let hard = source(0);
            let simulated = source(32);
            let groups: Vec<_> = (1..=4)
                .map(|m| Group {
                    id: m,
                    targets: (0..m)
                        .map(|j| Target {
                            id: 1000 + j,
                            site: Site::point(
                                Point3::new([0.13 + j as f64 * 0.2, 0.17, 0.]).unwrap(),
                                0,
                            ),
                        })
                        .collect(),
                })
                .collect();
            let budget = MemoryBudget::new(1 << 24);
            let cancel = Cancellation::default();
            let trend = |s: &Site| Ok(s.support.center().coordinates()[0].powi(2));
            for mean in [None, Some(&trend as &dyn KnownMean)] {
                let options = options(&budget, &cancel, mean);
                let mut workspace = SimulationWorkspace::<M>::new(&options).unwrap();
                simulate_group(
                    &model,
                    &hard,
                    &simulated,
                    &groups[3],
                    drift,
                    &options,
                    &mut workspace,
                )
                .unwrap();
                let expected: Vec<Vec<u64>> = groups
                    .iter()
                    .map(|g| {
                        simulate_group(
                            &model,
                            &hard,
                            &simulated,
                            g,
                            drift,
                            &options,
                            &mut workspace,
                        )
                        .unwrap()
                        .0
                        .iter()
                        .map(|o| o.value().to_bits())
                        .collect()
                    })
                    .collect();
                let measured = allocation_counter::measure(|| {
                    for _ in 0..20 {
                        for (group, reference) in groups.iter().zip(&expected) {
                            let (result, _) = simulate_group(
                                &model,
                                &hard,
                                &simulated,
                                group,
                                drift,
                                &options,
                                &mut workspace,
                            )
                            .unwrap();
                            assert_eq!(result.len(), reference.len());
                            assert!(result
                                .iter()
                                .zip(reference)
                                .all(|(o, bits)| o.value().to_bits() == *bits));
                        }
                    }
                });
                assert_eq!(measured.count_total, 0, "mean={}", mean.is_some());
                assert_eq!(measured.bytes_total, 0);
                drop(workspace);
                assert_eq!(budget.used(), 0);
            }
        }
        check::<Unchecked>(&Simple);
        check::<Unchecked>(&Ordinary { variables: 1 });
        check::<diagnostics::Checked>(&Simple);
        check::<diagnostics::Checked>(&Ordinary { variables: 1 });
    }
    #[test]
    fn selection_visitor_preserves_custom_overrides_and_required_exact_order() {
        use std::sync::Mutex;
        struct Custom<'a> {
            offset: u64,
            log: &'a Mutex<Vec<(u64, i32)>>,
        }
        fn record(id: u64, x: f64) -> Observation {
            Observation::new(id, Site::point(Point3::new([x, 0., 0.]).unwrap(), 0), x, 0.).unwrap()
        }
        impl ObservationSource for Custom<'_> {
            fn scan(&self, _: &mut dyn FnMut(&Observation) -> Result<()>) -> Result<()> {
                panic!("must use source overrides")
            }
            fn nearest(
                &self,
                _: Point3,
                _: Neighborhood,
                _: &Cancellation,
            ) -> Result<Vec<Observation>> {
                self.log.lock().unwrap().push((self.offset, 0));
                Ok(vec![
                    record(self.offset + 2, 0.2),
                    record(self.offset + 1, 0.1),
                ])
            }
            fn at_site(&self, site: &Site) -> Result<Vec<Observation>> {
                let x = site.support.center().coordinates()[0];
                self.log.lock().unwrap().push((self.offset, x as i32));
                let id = self.offset + if x < 0. { 10 } else { 20 };
                Ok(vec![record(id + 1, x), record(id, x)])
            }
        }
        let log = Mutex::new(Vec::new());
        let hard = Custom {
            offset: 0,
            log: &log,
        };
        let simulated = Custom {
            offset: 100,
            log: &log,
        };
        let group = Group {
            id: 1,
            targets: vec![
                Target {
                    id: 1000,
                    site: *record(0, -10.).site(),
                },
                Target {
                    id: 1001,
                    site: *record(0, 10.).site(),
                },
            ],
        };
        let budget = MemoryBudget::new(1 << 20);
        let cancel = Cancellation::default();
        let mut options = options(&budget, &cancel, None);
        options.hard_neighborhood = Neighborhood {
            maximum: 2,
            radius: 1.,
        }
        .into();
        options.simulated_neighborhood = Neighborhood {
            maximum: 2,
            radius: 1.,
        }
        .into();
        let mut searches = Searches::new(&options).unwrap();
        let owned = select_group(&hard, &simulated, &group, &mut searches, &cancel).unwrap();
        let expected_calls = vec![(0, 0), (100, 0), (0, -10), (100, -10), (0, 10), (100, 10)];
        assert_eq!(*log.lock().unwrap(), expected_calls);
        assert_eq!(
            owned.iter().map(Observation::id).collect::<Vec<_>>(),
            [1, 2, 101, 102, 11, 10, 111, 110, 21, 20, 121, 120]
        );
        log.lock().unwrap().clear();
        let mut reusable = Selection::new(&budget).unwrap();
        visit_group_selection(&hard, &simulated, &group, &mut searches, &cancel, |part| {
            reusable.extend(&part)
        })
        .unwrap();
        assert_eq!(*log.lock().unwrap(), expected_calls);
        assert!(owned
            .iter()
            .zip(&reusable.observations)
            .all(|(a, b)| a.same_bits(b)));
        searches
            .canonical
            .compact(&mut reusable.observations, true, &cancel)
            .unwrap();
        assert_eq!(
            reusable
                .observations
                .iter()
                .map(Observation::id)
                .collect::<Vec<_>>(),
            [1, 2, 11, 21]
        );
        drop(searches);
        drop(reusable);
        assert_eq!(budget.used(), 0);
    }
    struct CenterProofSource<'a> {
        name: &'static str,
        rows: Vec<Observation>,
        fail: bool,
        calls: &'a std::cell::RefCell<Vec<(&'static str, usize)>>,
    }
    impl ObservationSource for CenterProofSource<'_> {
        fn scan(&self, _: &mut dyn FnMut(&Observation) -> Result<()>) -> Result<()> {
            panic!("use source overrides")
        }
        fn nearest(
            &self,
            _: Point3,
            _: Neighborhood,
            _: &Cancellation,
        ) -> Result<Vec<Observation>> {
            Ok(Vec::new())
        }
        fn at_site_with_center_status(&self, site: &Site) -> Result<(Vec<Observation>, bool)> {
            self.calls.borrow_mut().push((self.name, site.variable));
            if self.fail {
                return Err(invalid("incomplete exact source query"));
            }
            let center = site.support.center();
            Ok((
                self.rows
                    .iter()
                    .filter(|row| row.site() == site)
                    .copied()
                    .collect(),
                !self
                    .rows
                    .iter()
                    .any(|row| row.site().support.center() == center),
            ))
        }
    }
    fn same_center_group() -> Group {
        Group {
            id: 1,
            targets: (0..7)
                .map(|variable| Target {
                    id: 1000 + variable as u64,
                    site: Site::point(Point3::new([0., 0., 0.]).unwrap(), variable),
                })
                .collect(),
        }
    }

    #[test]
    fn exact_center_selection_reuses_misses_and_resets_between_views() {
        let calls = std::cell::RefCell::new(Vec::new());
        let mut hard = CenterProofSource {
            name: "hard",
            rows: Vec::new(),
            fail: false,
            calls: &calls,
        };
        let simulated = CenterProofSource {
            name: "simulated",
            rows: Vec::new(),
            fail: false,
            calls: &calls,
        };
        let group = same_center_group();
        let budget = MemoryBudget::new(1 << 20);
        let cancel = Cancellation::default();
        let options = options(&budget, &cancel, None);
        let mut searches = Searches::new(&options).unwrap();
        assert!(
            select_group(&hard, &simulated, &group, &mut searches, &cancel)
                .unwrap()
                .is_empty()
        );
        assert_eq!(*calls.borrow(), [("hard", 0), ("simulated", 0)]);

        // A newly captured view now has aliases for variable one. An earlier
        // full-center miss must not suppress either alias in the next group.
        hard.rows = vec![
            Observation::new(2, group.targets[1].site, 5., 0.).unwrap(),
            Observation::new(1, group.targets[1].site, 5., 0.).unwrap(),
        ];
        calls.borrow_mut().clear();
        let selected = select_group(&hard, &simulated, &group, &mut searches, &cancel).unwrap();
        assert_eq!(
            selected.iter().map(Observation::id).collect::<Vec<_>>(),
            [2, 1]
        );
        assert_eq!(
            *calls.borrow(),
            [
                ("hard", 0),
                ("simulated", 0),
                ("hard", 1),
                ("hard", 2),
                ("hard", 3),
                ("hard", 4),
                ("hard", 5),
                ("hard", 6)
            ]
        );
        drop(searches);
        assert_eq!(budget.used(), 0);
    }

    #[test]
    fn exact_center_selection_retry_discards_partial_attempt_proofs() {
        let calls = std::cell::RefCell::new(Vec::new());
        let mut hard = CenterProofSource {
            name: "hard",
            rows: Vec::new(),
            fail: false,
            calls: &calls,
        };
        let mut simulated = CenterProofSource {
            name: "simulated",
            rows: Vec::new(),
            fail: true,
            calls: &calls,
        };
        let group = same_center_group();
        let budget = MemoryBudget::new(1 << 20);
        let cancel = Cancellation::default();
        let options = options(&budget, &cancel, None);
        let mut searches = Searches::new(&options).unwrap();
        assert!(select_group(&hard, &simulated, &group, &mut searches, &cancel).is_err());
        assert_eq!(*calls.borrow(), [("hard", 0), ("simulated", 0)]);
        hard.rows
            .push(Observation::new(1, group.targets[1].site, 5., 0.).unwrap());
        simulated.fail = false;
        calls.borrow_mut().clear();
        let selected = select_group(&hard, &simulated, &group, &mut searches, &cancel).unwrap();
        assert_eq!(
            selected.iter().map(Observation::id).collect::<Vec<_>>(),
            [1]
        );
        assert_eq!(&calls.borrow()[..2], [("hard", 0), ("simulated", 0)]);
    }

    #[test]
    fn exact_center_selection_checks_cancellation_before_reusing_miss() {
        let calls = std::cell::RefCell::new(Vec::new());
        let hard = CenterProofSource {
            name: "hard",
            rows: Vec::new(),
            fail: false,
            calls: &calls,
        };
        let simulated = CenterProofSource {
            name: "simulated",
            rows: Vec::new(),
            fail: false,
            calls: &calls,
        };
        let group = same_center_group();
        let budget = MemoryBudget::new(1 << 20);
        let cancel = Cancellation::default();
        let options = options(&budget, &cancel, None);
        let mut searches = Searches::new(&options).unwrap();
        let mut appended = 0;
        let result =
            visit_group_selection(&hard, &simulated, &group, &mut searches, &cancel, |_| {
                appended += 1;
                if appended == 4 {
                    cancel.cancel();
                }
                Ok(())
            });
        assert!(matches!(result, Err(Error::Cancelled)));
        assert_eq!(appended, 4);
        assert_eq!(*calls.borrow(), [("hard", 0), ("simulated", 0)]);
    }
}

pub mod high_order;
pub mod identity;
pub mod mps;
