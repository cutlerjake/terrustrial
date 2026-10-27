//! Deterministic process termination at actual store write boundaries.
//! These hooks and the subprocess entry point do not exist in production builds.
use super::*;
use std::{
    io::Write,
    process::{Child, Command, Stdio},
    sync::OnceLock,
    time::{Duration, Instant},
};

const COUNT: usize = 4096;
const FIRST_ID: u64 = 1 << 63;
const CHILD_TEST: &str = "storage::sqlite::crash_tests::crash_child";

pub(super) fn hit(phase: &str, value: u64) {
    static CONFIG: OnceLock<Option<(String, u64, PathBuf)>> = OnceLock::new();
    let Some((selected, target, ready)) = CONFIG.get_or_init(|| {
        let phase = std::env::var("TERRUSTRIAL_TEST_CRASH_PHASE").ok()?;
        let value = std::env::var("TERRUSTRIAL_TEST_CRASH_VALUE")
            .unwrap()
            .parse()
            .unwrap();
        let ready = PathBuf::from(std::env::var_os("TERRUSTRIAL_TEST_CRASH_READY").unwrap());
        Some((phase, value, ready))
    }) else {
        return;
    };
    if phase != selected || value != *target {
        return;
    }
    signal(ready);
    // The parent kills us. No unwinding or connection destructor may perform
    // a graceful rollback before the database is reopened by another process.
    loop {
        std::thread::park();
    }
}
fn signal(ready: &Path) {
    let pending = ready.with_extension("pending");
    let mut file = std::fs::File::create(&pending).unwrap();
    file.write_all(b"ready").unwrap();
    file.sync_all().unwrap();
    std::fs::rename(pending, ready).unwrap();
}

struct Process(Child);
impl Drop for Process {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}
fn start(path: &Path, ready: &Path, mode: &str, phase: &str, value: u64) -> Process {
    Process(
        Command::new(std::env::current_exe().unwrap())
            .args(["--exact", CHILD_TEST, "--ignored", "--nocapture"])
            .env("TERRUSTRIAL_TEST_CRASH_DB", path)
            .env("TERRUSTRIAL_TEST_CRASH_READY", ready)
            .env("TERRUSTRIAL_TEST_CRASH_MODE", mode)
            .env("TERRUSTRIAL_TEST_CRASH_PHASE", phase)
            .env("TERRUSTRIAL_TEST_CRASH_VALUE", value.to_string())
            .stdout(Stdio::null())
            .stderr(Stdio::inherit())
            .spawn()
            .unwrap(),
    )
}
fn await_boundary(child: &mut Process, ready: &Path) {
    let deadline = Instant::now() + Duration::from_secs(30);
    while !ready.exists() {
        if let Some(status) = child.0.try_wait().unwrap() {
            panic!("crash child exited before its boundary: {status}");
        }
        assert!(
            Instant::now() < deadline,
            "crash child never reached its boundary"
        );
        std::thread::sleep(Duration::from_millis(10));
    }
    assert_eq!(std::fs::read(ready).unwrap(), b"ready");
}
fn kill(child: &mut Process) {
    child.0.kill().unwrap();
    assert!(!child.0.wait().unwrap().success());
}
fn finish(child: &mut Process) {
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        if let Some(status) = child.0.try_wait().unwrap() {
            assert!(status.success(), "writer child failed: {status}");
            return;
        }
        assert!(Instant::now() < deadline, "writer child did not finish");
        std::thread::sleep(Duration::from_millis(10));
    }
}
fn record(index: usize) -> Observation {
    Observation::new(
        FIRST_ID + index as u64,
        Site {
            support: Support::block(
                Point3::new([
                    if index == COUNT - 1 {
                        1e100
                    } else {
                        (index % 97) as f64 * 0.25
                    },
                    (index / 97) as f64 * 0.5,
                    (index % 3) as f64 * 0.125,
                ])
                .unwrap(),
                if index.is_multiple_of(2) {
                    [0.; 3]
                } else {
                    [0.2, -0., 0.]
                },
                if index.is_multiple_of(2) {
                    [1; 3]
                } else {
                    [2, 1, 1]
                },
            )
            .unwrap(),
            variable: index % 3,
        },
        (index as f64 * 0.125).sin(),
        0.03125,
    )
    .unwrap()
}
fn checkpoint_at(group: u64) -> Checkpoint {
    Checkpoint {
        identity: [42; 32],
        completed_group: group,
    }
}
type Payload = (u64, usize, [u64; 3], u64, u64, [u64; 3], [usize; 3]);
fn payload(observation: &Observation) -> Payload {
    (
        observation.id(),
        observation.site().variable,
        observation
            .site()
            .support
            .center()
            .coordinates()
            .map(f64::to_bits),
        observation.value().to_bits(),
        observation.noise_variance().to_bits(),
        observation.site().support.half_extents().map(f64::to_bits),
        observation.site().support.divisions(),
    )
}
fn contents(store: &SqliteStore) -> Vec<Payload> {
    let mut records = Vec::new();
    store
        .scan(&mut |o| {
            records.push(payload(o));
            Ok(())
        })
        .unwrap();
    records.sort_unstable_by_key(|row| row.0);
    records
}
fn integrity(store: &SqliteStore) {
    store.integrity_check().unwrap();
    let missing: i64 = store
        .connection
        .query_row(
            "SELECT count(*) FROM points p LEFT JOIN bounds b ON p.id=b.id
         WHERE (p.indexed=1 AND b.id IS NULL) OR (p.indexed=0 AND b.id IS NOT NULL)",
            [],
            |row| row.get(0),
        )
        .unwrap();
    let orphaned: i64 = store
        .connection
        .query_row(
            "SELECT count(*) FROM bounds b LEFT JOIN points p ON p.id=b.id WHERE p.id IS NULL",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!((missing, orphaned), (0, 0));
}

#[test]
#[ignore = "subprocess entry point; exercised by the parent crash tests"]
fn crash_child() {
    let path = PathBuf::from(std::env::var_os("TERRUSTRIAL_TEST_CRASH_DB").unwrap());
    let budget = MemoryBudget::new(1 << 24);
    let mut store = SqliteStore::open(&path, 4096, &budget).unwrap();
    match std::env::var("TERRUSTRIAL_TEST_CRASH_MODE")
        .unwrap()
        .as_str()
    {
        "commit" => store
            .commit_group(
                &(0..COUNT).map(record).collect::<Vec<_>>(),
                Some(&checkpoint_at(10)),
                &checkpoint_at(11),
                &Cancellation::default(),
            )
            .unwrap(),
        "simulation" => simulate(&mut store, &budget),
        "race" => {
            let ready = PathBuf::from(std::env::var_os("TERRUSTRIAL_TEST_CRASH_READY").unwrap());
            let index: usize = std::env::var("TERRUSTRIAL_TEST_CRASH_VALUE")
                .unwrap()
                .parse()
                .unwrap();
            // Both connections must be open before either starts its write.
            signal(&ready);
            let deadline = Instant::now() + Duration::from_secs(30);
            while !path.with_extension("go").exists() {
                assert!(Instant::now() < deadline, "parent never released writers");
                std::thread::sleep(Duration::from_millis(1));
            }
            let result = store.commit_group(
                &[record(index)],
                Some(&checkpoint_at(10)),
                &checkpoint_at(11),
                &Cancellation::default(),
            );
            let outcome = match result {
                Ok(()) => "committed",
                Err(Error::InvalidInput(message))
                    if message == "stale checkpoint or simulation identity mismatch" =>
                {
                    "stale"
                }
                other => panic!("unexpected competing writer outcome: {other:?}"),
            };
            std::fs::write(ready.with_extension("outcome"), outcome).unwrap();
            return;
        }
        mode => panic!("unknown subprocess mode: {mode}"),
    }
    panic!("child completed without reaching its configured crash boundary");
}

#[test]
fn competing_processes_commit_exactly_one_group_from_the_same_checkpoint() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("store.sqlite");
    let ready = [
        directory.path().join("first"),
        directory.path().join("second"),
    ];
    let budget = MemoryBudget::new(1 << 24);
    let mut store = SqliteStore::create(&path, 4096, &budget).unwrap();
    store
        .commit_group(&[], None, &checkpoint_at(10), &Cancellation::default())
        .unwrap();
    drop(store);
    let mut children = [
        start(&path, &ready[0], "race", "none", 0),
        start(&path, &ready[1], "race", "none", 1),
    ];
    for (child, marker) in children.iter_mut().zip(&ready) {
        await_boundary(child, marker);
    }
    signal(&path.with_extension("go"));
    for child in &mut children {
        finish(child);
    }
    let outcomes = ready.map(|p| std::fs::read_to_string(p.with_extension("outcome")).unwrap());
    let winner = match outcomes.each_ref().map(String::as_str) {
        ["committed", "stale"] => 0,
        ["stale", "committed"] => 1,
        other => panic!("expected one committed and one stale writer, got {other:?}"),
    };
    let store = SqliteStore::open(&path, 4096, &budget).unwrap();
    assert_eq!(store.checkpoint().unwrap(), Some(checkpoint_at(11)));
    assert_eq!(contents(&store), vec![payload(&record(winner))]);
    integrity(&store);
    drop(store);
    assert_eq!(budget.used(), 0);
}

#[test]
fn killed_group_writes_recover_payload_index_and_checkpoint_atomically() {
    for (phase, target) in [
        ("payload", FIRST_ID + 1500),
        ("before_commit", 11),
        ("after_commit", 11),
    ] {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("store.sqlite");
        let ready = directory.path().join("ready");
        let budget = MemoryBudget::new(1 << 24);
        let initial = Observation::new(
            1,
            Site::point(Point3::new([-1., 0., 0.]).unwrap(), 0),
            0.125,
            0.,
        )
        .unwrap();
        let mut store = SqliteStore::create(&path, 4096, &budget).unwrap();
        store
            .commit_group(
                &[initial],
                None,
                &checkpoint_at(10),
                &Cancellation::default(),
            )
            .unwrap();
        drop(store);
        let original_bytes = std::fs::read(&path).unwrap();
        let mut child = start(&path, &ready, "commit", phase, target);
        await_boundary(&mut child, &ready);
        if phase != "after_commit" {
            let journal = path.with_file_name("store.sqlite-journal");
            assert!(std::fs::metadata(journal).unwrap().len() > 512);
            // Prove the test reached spilled database writes, not just buffered
            // operations whose disappearance would require no journal recovery.
            assert_ne!(std::fs::read(&path).unwrap(), original_bytes);
        }
        kill(&mut child);
        let mut store = SqliteStore::open(&path, 4096, &budget).unwrap();
        integrity(&store);
        let committed = phase == "after_commit";
        assert_eq!(
            store.checkpoint().unwrap(),
            Some(checkpoint_at(if committed { 11 } else { 10 }))
        );
        let mut expected = vec![payload(&initial)];
        if committed {
            expected.extend((0..COUNT).map(|i| payload(&record(i))));
        }
        assert_eq!(contents(&store), expected);
        let records: Vec<_> = (0..COUNT).map(record).collect();
        let retry = store.commit_group(
            &records,
            Some(&checkpoint_at(10)),
            &checkpoint_at(11),
            &Cancellation::default(),
        );
        if committed {
            assert!(retry.is_err());
        } else {
            retry.unwrap();
        }
        expected = vec![payload(&initial)];
        expected.extend(records.iter().map(payload));
        assert_eq!(contents(&store), expected);
        // After a lost acknowledgement, a fresh checkpoint permits progress;
        // retrying the old checkpoint above never duplicates or replaces output.
        store
            .commit_group(
                &[record(COUNT)],
                Some(&checkpoint_at(11)),
                &checkpoint_at(12),
                &Cancellation::default(),
            )
            .unwrap();
        assert_eq!(store.len().unwrap(), COUNT as u64 + 2);
        integrity(&store);
        drop(store);
        assert_eq!(budget.used(), 0);
    }
}

fn simulate(store: &mut SqliteStore, budget: &MemoryBudget) {
    use crate::{
        data::{Group, Target},
        gaussian::Simple,
        model::{Anisotropy, Coregionalization, Correlation, Lmc, Structure},
        simulation::{simulate_checkpointed, CheckpointedOptions, SimulationOptions},
    };
    let model = Lmc::new(vec![Structure {
        correlation: Correlation::Exponential,
        anisotropy: Anisotropy::isotropic(3.).unwrap(),
        coregionalization: Coregionalization::from_factor(2, 2, &[1., 0., 0.6, 0.8]).unwrap(),
    }])
    .unwrap();
    let hard =
        [Observation::new(0, Site::point(Point3::new([0.; 3]).unwrap(), 0), -0.0, 0.).unwrap()];
    let groups = (0..12).map(|i| Group {
        id: i,
        targets: (0..2)
            .map(|v| Target {
                id: 100 + 2 * i + v as u64,
                site: Site {
                    support: Support::block(
                        Point3::new([i as f64 * 0.4, 0., 0.]).unwrap(),
                        if i == 0 { [0.; 3] } else { [0.1, 0., 0.] },
                        if i == 0 { [1; 3] } else { [2, 1, 1] },
                    )
                    .unwrap(),
                    variable: v,
                },
            })
            .collect(),
    });
    let mean = |s: &Site| {
        Ok(s.support.center().coordinates()[0].powi(2)
            + s.support.half_extents()[0].powi(2) / 3.
            + s.variable as f64)
    };
    simulate_checkpointed(
        &model,
        &hard[..],
        store,
        groups,
        &Simple,
        CheckpointedOptions {
            simulation: SimulationOptions {
                seed: 153,
                realization: 4,
                mean: Some(&mean),
                hard_neighborhood: Neighborhood {
                    maximum: 1,
                    radius: 0.1,
                }
                .into(),
                simulated_neighborhood: Neighborhood {
                    maximum: 12,
                    radius: 100.,
                }
                .into(),
                budget,
                cancellation: &Cancellation::default(),
            },
            identity: [7; 32],
        },
        |_| Ok(()),
    )
    .unwrap();
}

#[test]
fn killed_checkpointed_simulation_resumes_bit_for_bit_after_lost_acknowledgement() {
    let directory = tempfile::tempdir().unwrap();
    let budget = MemoryBudget::new(1 << 24);
    let path = directory.path().join("interrupted.sqlite");
    let ready = directory.path().join("ready");
    drop(SqliteStore::create(&path, 4096, &budget).unwrap());
    let mut child = start(&path, &ready, "simulation", "after_commit", 3);
    await_boundary(&mut child, &ready);
    kill(&mut child);
    let mut resumed = SqliteStore::open(&path, 4096, &budget).unwrap();
    assert_eq!(resumed.len().unwrap(), 8);
    assert_eq!(
        resumed.checkpoint().unwrap(),
        Some(Checkpoint {
            identity: [7; 32],
            completed_group: 3
        })
    );
    simulate(&mut resumed, &budget);
    let mut reference =
        SqliteStore::create(directory.path().join("reference.sqlite"), 4096, &budget).unwrap();
    simulate(&mut reference, &budget);
    assert_eq!(
        resumed.checkpoint().unwrap(),
        reference.checkpoint().unwrap()
    );
    assert_eq!(contents(&resumed), contents(&reference));
    assert_eq!(contents(&resumed)[0].3, (-0.0_f64).to_bits());
    integrity(&resumed);
    drop((resumed, reference));
    assert_eq!(budget.used(), 0);
}
