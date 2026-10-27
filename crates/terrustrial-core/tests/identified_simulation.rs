#![cfg(feature = "sqlite")]
use terrustrial_core::{
    data::{Group, Observation, Point3, Site, Target},
    execution::{Cancellation, MemoryBudget},
    gaussian::{Ordinary, Simple},
    identity,
    model::{Anisotropy, Coregionalization, Correlation, Lmc, Structure},
    simulation::{identity::RunProvenance, simulate_checkpointed_identified, SimulationOptions},
    spatial::{Neighborhood, ObservationSource},
    storage::SqliteStore,
    Error,
};

fn model(range: f64) -> Lmc {
    Lmc::new(vec![Structure {
        correlation: Correlation::Exponential,
        anisotropy: Anisotropy::isotropic(range).unwrap(),
        coregionalization: Coregionalization::from_factor(2, 2, &[1., 0., 0.6, 0.8]).unwrap(),
    }])
    .unwrap()
}
fn hard() -> Vec<Observation> {
    vec![
        Observation::new(0, Site::point(Point3::new([0.; 3]).unwrap(), 0), -0., 0.).unwrap(),
        Observation::new(1, Site::point(Point3::new([0.; 3]).unwrap(), 1), 0.5, 0.).unwrap(),
    ]
}
fn groups() -> Vec<Group> {
    (0..12)
        .map(|i| Group {
            id: i,
            targets: (0..2)
                .map(|v| Target {
                    id: 100 + 2 * i + v as u64,
                    site: Site::point(Point3::new([i as f64 * 0.4, 0., 0.]).unwrap(), v),
                })
                .collect(),
        })
        .collect()
}
fn options<'a>(budget: &'a MemoryBudget, cancel: &'a Cancellation) -> SimulationOptions<'a> {
    SimulationOptions {
        seed: 153,
        realization: 4,
        mean: None,
        hard_neighborhood: Neighborhood {
            maximum: 2,
            radius: 10.,
        }
        .into(),
        simulated_neighborhood: Neighborhood {
            maximum: 12,
            radius: 100.,
        }
        .into(),
        budget,
        cancellation: cancel,
    }
}
fn provenance() -> RunProvenance {
    RunProvenance {
        implementation: identity::opaque("test implementation", b"source build v1").unwrap(),
        backend: identity::opaque("test backend", b"cpu-f64-native-v1").unwrap(),
        mean: None,
    }
}
type Payload = (u64, usize, [u64; 5]);
fn payloads(store: &SqliteStore) -> Vec<Payload> {
    let mut result = Vec::new();
    store
        .scan(&mut |o| {
            let [x, y, z] = o.site().support.center().coordinates();
            result.push((
                o.id(),
                o.site().variable,
                [x, y, z, o.value(), o.noise_variance()].map(f64::to_bits),
            ));
            Ok(())
        })
        .unwrap();
    result.sort_unstable();
    result
}
fn partial(path: &std::path::Path, budget: &MemoryBudget) -> SqliteStore {
    let mut store = SqliteStore::create(path, 4096, budget).unwrap();
    let cancel = Cancellation::default();
    let mut count = 0;
    let result = simulate_checkpointed_identified(
        &model(3.),
        hard().as_slice(),
        &mut store,
        || Ok(groups().into_iter().map(Ok)),
        &Simple,
        options(budget, &cancel),
        provenance(),
        |_| {
            count += 1;
            if count == 4 {
                cancel.cancel();
            }
            Ok(())
        },
    );
    assert!(matches!(result, Err(Error::Cancelled)));
    assert_eq!(store.checkpoint().unwrap().unwrap().completed_group, 3);
    assert_eq!(store.len().unwrap(), 8);
    store
}

#[test]
fn identified_restart_matches_uninterrupted_bits_and_rejects_changed_completed_prefix() {
    let directory = tempfile::tempdir().unwrap();
    let budget = MemoryBudget::new(1 << 24);
    let path = directory.path().join("interrupted.sqlite");
    drop(partial(&path, &budget));
    let mut store = SqliteStore::open(&path, 4096, &budget).unwrap();
    let before = payloads(&store);
    let checkpoint = store.checkpoint().unwrap();
    let cancel = Cancellation::default();
    // These include completed-prefix changes, which must not disappear when the
    // low-level driver skips already committed groups.
    for case in 0..13 {
        let mut data = hard();
        let mut path = groups();
        let mut opt = options(&budget, &cancel);
        let mut prov = provenance();
        let mut range = 3.;
        match case {
            0 => data[0] = Observation::new(0, *data[0].site(), 0., 0.).unwrap(),
            1 => data.reverse(),
            2 => path[0].targets[0].site = Site::point(Point3::new([0.01, 0., 0.]).unwrap(), 0),
            3 => path[11].targets[1].id += 100,
            4 => range = 4.,
            5 => opt.seed += 1,
            6 => opt.realization += 1,
            7 => {
                opt.hard_neighborhood = Neighborhood {
                    maximum: 1,
                    radius: 10.,
                }
                .into()
            }
            8 => {
                opt.simulated_neighborhood = Neighborhood {
                    maximum: 10,
                    radius: 100.,
                }
                .into()
            }
            9 => {
                prov.implementation =
                    identity::opaque("test implementation", b"source build v2").unwrap()
            }
            10 => prov.backend = identity::opaque("test backend", b"other arithmetic").unwrap(),
            11 => path.drain(..4).for_each(drop),
            12 => data[1] = Observation::new(1, *data[1].site(), 0.6, 0.).unwrap(),
            _ => unreachable!(),
        }
        let mut emitted = false;
        let result = simulate_checkpointed_identified(
            &model(range),
            data.as_slice(),
            &mut store,
            || Ok(path.iter().cloned().map(Ok)),
            &Simple,
            opt,
            prov,
            |_| {
                emitted = true;
                Ok(())
            },
        );
        assert!(
            matches!(result,Err(Error::InvalidInput(ref message)) if message=="simulation checkpoint identity mismatch"),
            "case {case}: {result:?}"
        );
        assert!(!emitted);
        assert_eq!(payloads(&store), before);
        assert_eq!(store.checkpoint().unwrap(), checkpoint);
    }
    let result = simulate_checkpointed_identified(
        &model(3.),
        hard().as_slice(),
        &mut store,
        || Ok(groups().into_iter().map(Ok)),
        &Ordinary { variables: 2 },
        options(&budget, &cancel),
        provenance(),
        |_| panic!("mismatched drift emitted"),
    );
    assert!(result.is_err());
    assert_eq!(store.checkpoint().unwrap(), checkpoint);
    let mean = |_: &Site| Ok(0.);
    let mut opt = options(&budget, &cancel);
    opt.mean = Some(&mean);
    let mut prov = provenance();
    prov.mean = Some(identity::opaque("mean", b"zero").unwrap());
    assert!(simulate_checkpointed_identified(
        &model(3.),
        hard().as_slice(),
        &mut store,
        || Ok(groups().into_iter().map(Ok)),
        &Simple,
        opt,
        prov,
        |_| panic!("mismatched mean emitted")
    )
    .is_err());
    simulate_checkpointed_identified(
        &model(3.),
        hard().as_slice(),
        &mut store,
        || Ok(groups().into_iter().map(Ok)),
        &Simple,
        options(&budget, &cancel),
        provenance(),
        |_| Ok(()),
    )
    .unwrap();
    let mut reference =
        SqliteStore::create(directory.path().join("reference.sqlite"), 4096, &budget).unwrap();
    simulate_checkpointed_identified(
        &model(3.),
        hard().as_slice(),
        &mut reference,
        || Ok(groups().into_iter().map(Ok)),
        &Simple,
        options(&budget, &cancel),
        provenance(),
        |_| Ok(()),
    )
    .unwrap();
    assert_eq!(payloads(&store), payloads(&reference));
    assert_eq!(payloads(&store)[0].2[3], (-0.0_f64).to_bits());
    assert_eq!(store.checkpoint().unwrap(), reference.checkpoint().unwrap());
    simulate_checkpointed_identified(
        &model(3.),
        hard().as_slice(),
        &mut store,
        || Ok(groups().into_iter().map(Ok)),
        &Simple,
        options(&budget, &cancel),
        provenance(),
        |_| panic!("completed run emitted twice"),
    )
    .unwrap();
    drop((store, reference));
    assert_eq!(budget.used(), 0);
}

#[test]
fn failed_or_cancelled_preflight_never_commits_output() {
    struct Failed;
    impl ObservationSource for Failed {
        fn scan(
            &self,
            visitor: &mut dyn FnMut(&Observation) -> terrustrial_core::Result<()>,
        ) -> terrustrial_core::Result<()> {
            visitor(&hard()[0])?;
            Err(Error::Numerical {
                context: "source failed",
            })
        }
    }
    let directory = tempfile::tempdir().unwrap();
    let budget = MemoryBudget::new(1 << 24);
    let cancel = Cancellation::default();
    let mut store =
        SqliteStore::create(directory.path().join("store.sqlite"), 4096, &budget).unwrap();
    assert!(matches!(
        simulate_checkpointed_identified(
            &model(3.),
            &Failed,
            &mut store,
            || Ok(groups().into_iter().map(Ok)),
            &Simple,
            options(&budget, &cancel),
            provenance(),
            |_| panic!("failed preflight emitted")
        ),
        Err(Error::Numerical { .. })
    ));
    assert!(store.is_empty().unwrap());
    assert!(store.checkpoint().unwrap().is_none());
    let invalid = [Group {
        id: 0,
        targets: vec![],
    }];
    assert!(simulate_checkpointed_identified(
        &model(3.),
        hard().as_slice(),
        &mut store,
        || Ok(invalid.iter().cloned().map(Ok)),
        &Simple,
        options(&budget, &cancel),
        provenance(),
        |_| panic!("invalid path emitted")
    )
    .is_err());
    cancel.cancel();
    assert!(matches!(
        simulate_checkpointed_identified(
            &model(3.),
            hard().as_slice(),
            &mut store,
            || Ok(groups().into_iter().map(Ok)),
            &Simple,
            options(&budget, &cancel),
            provenance(),
            |_| panic!("cancelled preflight emitted")
        ),
        Err(Error::Cancelled)
    ));
    assert!(store.is_empty().unwrap());
    assert!(store.checkpoint().unwrap().is_none());
}

#[test]
fn path_read_failures_in_either_pass_preserve_only_committed_output() {
    use std::cell::Cell;
    let directory = tempfile::tempdir().unwrap();
    let budget = MemoryBudget::new(1 << 24);
    let cancel = Cancellation::default();
    let mut store =
        SqliteStore::create(directory.path().join("store.sqlite"), 4096, &budget).unwrap();
    let failure = || Error::Io(std::io::ErrorKind::UnexpectedEof.into());
    let bad_prepass = || {
        Ok(groups()
            .into_iter()
            .enumerate()
            .map(|(i, g)| if i == 2 { Err(failure()) } else { Ok(g) }))
    };
    let result = simulate_checkpointed_identified(
        &model(3.),
        hard().as_slice(),
        &mut store,
        bad_prepass,
        &Simple,
        options(&budget, &cancel),
        provenance(),
        |_| panic!("bad prepass emitted"),
    );
    assert!(matches!(result, Err(Error::Io(_))));
    assert!(store.is_empty().unwrap());
    assert!(store.checkpoint().unwrap().is_none());
    let opens = Cell::new(0);
    let later_failure = || {
        opens.set(opens.get() + 1);
        let fail = opens.get() == 2;
        Ok(groups().into_iter().enumerate().map(move |(i, g)| {
            if fail && i == 2 {
                Err(Error::Io(std::io::ErrorKind::UnexpectedEof.into()))
            } else {
                Ok(g)
            }
        }))
    };
    let result = simulate_checkpointed_identified(
        &model(3.),
        hard().as_slice(),
        &mut store,
        later_failure,
        &Simple,
        options(&budget, &cancel),
        provenance(),
        |_| Ok(()),
    );
    assert!(matches!(result, Err(Error::Io(_))));
    assert_eq!(opens.get(), 2);
    assert_eq!(store.len().unwrap(), 4);
    assert_eq!(store.checkpoint().unwrap().unwrap().completed_group, 1);
    simulate_checkpointed_identified(
        &model(3.),
        hard().as_slice(),
        &mut store,
        || Ok(groups().into_iter().map(Ok)),
        &Simple,
        options(&budget, &cancel),
        provenance(),
        |_| Ok(()),
    )
    .unwrap();
    let mut reference =
        SqliteStore::create(directory.path().join("reference.sqlite"), 4096, &budget).unwrap();
    simulate_checkpointed_identified(
        &model(3.),
        hard().as_slice(),
        &mut reference,
        || Ok(groups().into_iter().map(Ok)),
        &Simple,
        options(&budget, &cancel),
        provenance(),
        |_| Ok(()),
    )
    .unwrap();
    assert_eq!(payloads(&store), payloads(&reference));
    assert_eq!(store.checkpoint().unwrap(), reference.checkpoint().unwrap());
    drop((store, reference));
    assert_eq!(budget.used(), 0);
}
