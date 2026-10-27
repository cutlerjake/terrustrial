//! Checkpoint orchestration must work without the SQLite feature or concrete adapter.
use terrustrial_core::{
    data::{Group, Observation, Point3, Site, Target},
    execution::{Cancellation, MemoryBudget},
    gaussian::Simple,
    identity,
    model::{Anisotropy, Coregionalization, Correlation, Lmc, Structure},
    simulation::{
        identity::RunProvenance, simulate_checkpointed_identified, simulate_groups,
        SimulationOptions,
    },
    spatial::{Neighborhood, ObservationSource, SpatialIndex},
    storage::{Checkpoint, CheckpointStore},
    Error, Result,
};
#[derive(Default)]
struct MemoryStore {
    observations: Vec<Observation>,
    checkpoint: Option<Checkpoint>,
    fail_at: Option<u64>,
}
impl ObservationSource for MemoryStore {
    fn scan(&self, visitor: &mut dyn FnMut(&Observation) -> Result<()>) -> Result<()> {
        self.observations.as_slice().scan(visitor)
    }
}
impl CheckpointStore for MemoryStore {
    fn checkpoint(&self) -> Result<Option<Checkpoint>> {
        Ok(self.checkpoint.clone())
    }
    fn is_empty(&self) -> Result<bool> {
        Ok(self.observations.is_empty())
    }
    fn commit_group(
        &mut self,
        observations: &[Observation],
        expected: Option<&Checkpoint>,
        next: &Checkpoint,
        cancel: &Cancellation,
    ) -> Result<()> {
        cancel.check()?;
        if self.checkpoint.as_ref() != expected
            || expected.is_some_and(|c| {
                c.identity != next.identity || c.completed_group >= next.completed_group
            })
        {
            return Err(Error::Numerical {
                context: "stale test writer",
            });
        }
        if self.fail_at == Some(next.completed_group) {
            return Err(Error::Numerical {
                context: "injected pre-commit failure",
            });
        }
        // Stage first, then publish both. This fixture promises process-local atomicity only.
        let mut staged = self.observations.clone();
        staged.extend_from_slice(observations);
        self.observations = staged;
        self.checkpoint = Some(next.clone());
        Ok(())
    }
}
#[test]
fn generic_checkpoint_driver_resumes_after_commit_and_notification_failures() {
    let model = Lmc::new(vec![Structure {
        correlation: Correlation::Exponential,
        anisotropy: Anisotropy::isotropic(2.).unwrap(),
        coregionalization: Coregionalization::scalar(1.).unwrap(),
    }])
    .unwrap();
    let site = |x| Site::point(Point3::new([x, 0., 0.]).unwrap(), 0);
    let hard = [Observation::new(0, site(0.), -0., 0.).unwrap()];
    let groups: Vec<_> = (0..5)
        .map(|i| Group {
            id: i,
            targets: vec![Target {
                id: 100 + i,
                site: site(i as f64 * 0.3),
            }],
        })
        .collect();
    let budget = MemoryBudget::new(16 << 20);
    let cancel = Cancellation::default();
    let options = || SimulationOptions {
        seed: 19,
        realization: 3,
        mean: None,
        hard_neighborhood: Neighborhood {
            maximum: 8,
            radius: 10.,
        }
        .into(),
        simulated_neighborhood: Neighborhood {
            maximum: 8,
            radius: 10.,
        }
        .into(),
        budget: &budget,
        cancellation: &cancel,
    };
    let provenance = RunProvenance {
        implementation: identity::opaque("test implementation", b"stable v1").unwrap(),
        backend: identity::opaque("test backend", b"cpu f64").unwrap(),
        mean: None,
    };
    let mut expected = Vec::new();
    simulate_groups(
        &model,
        hard.as_slice(),
        &mut SpatialIndex::new(vec![]).unwrap(),
        groups.clone(),
        &Simple,
        options(),
        |o| {
            expected.extend(o.iter().map(|o| (o.id(), o.value().to_bits())));
            Ok(())
        },
    )
    .unwrap();
    let mut store = MemoryStore {
        fail_at: Some(2),
        ..MemoryStore::default()
    };
    let run = |store: &mut MemoryStore, fail_notification: bool| {
        simulate_checkpointed_identified(
            &model,
            hard.as_slice(),
            store,
            || Ok(groups.clone().into_iter().map(Ok)),
            &Simple,
            options(),
            provenance,
            |_| {
                if fail_notification {
                    Err(Error::Cancelled)
                } else {
                    Ok(())
                }
            },
        )
    };
    assert!(run(&mut store, false).is_err());
    assert_eq!(store.checkpoint.as_ref().unwrap().completed_group, 1);
    assert_eq!(store.observations.len(), 2);
    assert_eq!(budget.used(), 0);
    store.fail_at = None;
    assert!(matches!(run(&mut store, true), Err(Error::Cancelled)));
    assert_eq!(store.checkpoint.as_ref().unwrap().completed_group, 2);
    assert_eq!(store.observations.len(), 3);
    assert_eq!(budget.used(), 0);
    run(&mut store, false).unwrap();
    let actual: Vec<_> = store
        .observations
        .iter()
        .map(|o| (o.id(), o.value().to_bits()))
        .collect();
    assert_eq!(actual, expected);
    assert_eq!(actual[0].1, (-0_f64).to_bits());
    run(&mut store, false).unwrap();
    assert_eq!(store.observations.len(), 5);
    let before = store.checkpoint.clone();
    let stale = Checkpoint {
        identity: [0; 32],
        completed_group: 5,
    };
    assert!(store.commit_group(&[], None, &stale, &cancel).is_err());
    assert_eq!(store.checkpoint, before);
    assert_eq!(budget.used(), 0);
}
