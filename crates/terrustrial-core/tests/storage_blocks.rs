#![cfg(feature = "sqlite")]
use terrustrial_core::{
    data::{Observation, Point3, Site, Support},
    execution::{Cancellation, MemoryBudget},
    model::Anisotropy,
    spatial::{
        NearestWorkspace, Neighborhood, ObservationSource, SearchNeighborhood, SearchWorkspace,
    },
    storage::{SqliteReadPool, SqliteStore},
};
fn support(center: [f64; 3], half: [f64; 3], divisions: [usize; 3], variable: usize) -> Site {
    Site {
        support: Support::block(Point3::new(center).unwrap(), half, divisions).unwrap(),
        variable,
    }
}
type Payload = (u64, usize, [u64; 8], [usize; 3]);
fn payload(o: &Observation) -> Payload {
    let c = o.site().support.center().coordinates();
    let h = o.site().support.half_extents();
    (
        o.id(),
        o.site().variable,
        [
            c[0],
            c[1],
            c[2],
            h[0],
            h[1],
            h[2],
            o.value(),
            o.noise_variance(),
        ]
        .map(f64::to_bits),
        o.site().support.divisions(),
    )
}
fn rows(source: &(impl ObservationSource + ?Sized)) -> Vec<Payload> {
    let mut out = Vec::new();
    source
        .scan(&mut |o| {
            out.push(payload(o));
            Ok(())
        })
        .unwrap();
    out.sort_unstable();
    out
}
fn fixture() -> Vec<Observation> {
    let mut rows = Vec::new();
    for i in 0..80 {
        let (half, divisions) = match i % 5 {
            0 => ([0.; 3], [1; 3]),
            1 => ([0.2, -0., 0.], [3, 1, 1]),
            2 => ([0.2, 0.3, 0.], [2, 4, 1]),
            3 => ([0.1, 0.2, 0.3], [1, 2, 3]),
            _ => ([-0., 0., -0.], [1; 3]),
        };
        let center = if i < 10 {
            [-0.; 3]
        } else if i == 79 {
            [1e100, 0., 0.]
        } else {
            [(i / 3) as f64 * 0.25, 0., 0.]
        };
        rows.push(
            Observation::new(
                u64::MAX - i,
                support(center, half, divisions, (i % 2) as usize),
                if i % 3 == 0 { -0. } else { i as f64 },
                -0.,
            )
            .unwrap(),
        );
    }
    rows.push(
        Observation::new(
            7,
            support(
                [1e-200, 0., 0.],
                [1e-201, 0., 0.],
                [usize::MAX, 1, 1],
                usize::MAX,
            ),
            2.,
            0.,
        )
        .unwrap(),
    );
    rows
}
#[test]
fn complete_support_payload_survives_scan_reopen_queries_and_snapshot_pool() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("blocks.sqlite");
    let budget = MemoryBudget::new(1 << 24);
    let cancel = Cancellation::default();
    let data = fixture();
    let mut store = SqliteStore::create(&path, 4096, &budget).unwrap();
    store.import(data.as_slice(), &cancel).unwrap();
    store.integrity_check().unwrap();
    drop(store);
    let store = SqliteStore::open(&path, 4096, &budget).unwrap();
    let pool = SqliteReadPool::open(&path, 3, 4096, &budget).unwrap();
    let snapshot = SqliteReadPool::open_snapshot(&path, 3, 4096, &budget, &cancel).unwrap();
    let expected = rows(data.as_slice());
    for source in [&store as &dyn ObservationSource, &pool, &snapshot] {
        assert_eq!(rows(source), expected);
        for observation in &data {
            assert_eq!(
                source
                    .at_site(observation.site())
                    .unwrap()
                    .iter()
                    .map(payload)
                    .collect::<Vec<_>>(),
                data.as_slice()
                    .at_site(observation.site())
                    .unwrap()
                    .iter()
                    .map(payload)
                    .collect::<Vec<_>>()
            );
        }
        for center in [[0.; 3], [1e100, 0., 0.], [1e-200, 0., 0.], [2., 0., 0.]] {
            let point = Point3::new(center).unwrap();
            let query = Neighborhood {
                maximum: 13,
                radius: 4.,
            };
            let mut oracle: Vec<_> = data
                .iter()
                .filter(|o| point.distance(o.site().support.center()) <= query.radius)
                .collect();
            oracle.sort_unstable_by(|a, b| {
                point
                    .distance(a.site().support.center())
                    .total_cmp(&point.distance(b.site().support.center()))
                    .then(a.id().cmp(&b.id()))
            });
            oracle.truncate(query.maximum);
            let oracle: Vec<_> = oracle.into_iter().map(payload).collect();
            let mut workspace = NearestWorkspace::new(query, &budget).unwrap();
            assert_eq!(
                source
                    .nearest(point, query, &cancel)
                    .unwrap()
                    .iter()
                    .map(payload)
                    .collect::<Vec<_>>(),
                oracle
            );
            assert_eq!(
                source
                    .nearest_into(point, &mut workspace, &cancel)
                    .unwrap()
                    .iter()
                    .map(payload)
                    .collect::<Vec<_>>(),
                oracle
            );
            let search = SearchNeighborhood::new(
                13,
                Anisotropy::isotropic(4.).unwrap(),
                Some(4),
                Vec::new(),
            )
            .unwrap();
            let mut actual = SearchWorkspace::new(&search, &budget).unwrap();
            let mut reference = SearchWorkspace::new(&search, &budget).unwrap();
            assert_eq!(
                source
                    .search(point, &mut actual, &cancel)
                    .unwrap()
                    .iter()
                    .map(payload)
                    .collect::<Vec<_>>(),
                data.as_slice()
                    .search(point, &mut reference, &cancel)
                    .unwrap()
                    .iter()
                    .map(payload)
                    .collect::<Vec<_>>()
            );
        }
    }
    // The column stays NULL for canonical points, and borrows fixed bytes for blocks.
    let db = rusqlite::Connection::open(&path).unwrap();
    let blob: Vec<u8> = db
        .query_row(
            "SELECT support FROM points WHERE id=?1",
            [(u64::MAX - 1) as i64],
            |r| r.get(0),
        )
        .unwrap();
    let mut bytes = Vec::new();
    for h in [0.2_f64, -0., 0.] {
        bytes.extend_from_slice(&h.to_bits().to_le_bytes());
    }
    for n in [3_u64, 1, 1] {
        bytes.extend_from_slice(&n.to_le_bytes());
    }
    assert_eq!(blob, bytes);
    let point_null: bool = db
        .query_row("SELECT support IS NULL FROM points WHERE id=-1", [], |r| {
            r.get(0)
        })
        .unwrap();
    assert!(point_null);
    drop((store, pool, snapshot, db));
    assert_eq!(budget.used(), 0);
}

#[test]
fn malformed_support_metadata_fails_decode_and_integrity_without_repair() {
    let valid = || {
        let mut b = Vec::new();
        for h in [0.2_f64, 0., 0.] {
            b.extend_from_slice(&h.to_bits().to_le_bytes());
        }
        for n in [2_u64, 1, 1] {
            b.extend_from_slice(&n.to_le_bytes());
        }
        b
    };
    let mut cases = vec![
        rusqlite::types::Value::Text("invalid".into()),
        rusqlite::types::Value::Blob(vec![0; 47]),
        rusqlite::types::Value::Blob(vec![0; 49]),
    ];
    for h in [-1_f64, f64::NAN, f64::INFINITY] {
        let mut b = valid();
        b[..8].copy_from_slice(&h.to_bits().to_le_bytes());
        cases.push(b.into());
    }
    for (offset, value) in [(24, 0_u64), (32, 2), (24, u64::MAX)] {
        let mut b = valid();
        b[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
        if value == u64::MAX {
            b[8..16].copy_from_slice(&0.2_f64.to_bits().to_le_bytes());
            b[32..40].copy_from_slice(&2_u64.to_le_bytes());
        }
        cases.push(b.into());
    }
    for value in cases {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad.sqlite");
        let budget = MemoryBudget::new(1 << 20);
        let mut store = SqliteStore::create(&path, 4096, &budget).unwrap();
        store
            .import(
                &[
                    Observation::new(0, support([0.; 3], [0.2, 0., 0.], [2, 1, 1], 0), 1., 0.)
                        .unwrap(),
                ][..],
                &Cancellation::default(),
            )
            .unwrap();
        drop(store);
        let db = rusqlite::Connection::open(&path).unwrap();
        db.execute_batch("PRAGMA ignore_check_constraints=ON")
            .unwrap();
        db.execute("UPDATE points SET support=?1", [value]).unwrap();
        drop(db);
        let before = std::fs::read(&path).unwrap();
        let store = SqliteStore::open(&path, 4096, &budget).unwrap();
        assert!(store.scan(&mut |_| Ok(())).is_err());
        assert!(store.integrity_check().is_err());
        drop(store);
        assert_eq!(std::fs::read(&path).unwrap(), before);
        assert_eq!(budget.used(), 0);
    }
}

#[test]
fn disk_blocks_match_resident_joint_estimation_and_checkpointed_simulation() {
    use terrustrial_core::{
        data::{Group, Target},
        estimation::{estimate_groups, EstimationOptions},
        execution::ExecutionOptions,
        gaussian::{NumericalTolerance, Simple, Uncertainty},
        model::{Coregionalization, Correlation, Lmc, Structure},
        simulation::{
            simulate_checkpointed, simulate_groups, CheckpointedOptions, SimulationOptions,
        },
        spatial::SpatialIndex,
    };
    let dir = tempfile::tempdir().unwrap();
    let budget = MemoryBudget::new(256 << 20);
    let cancel = Cancellation::default();
    let data: Vec<_> = (0..12)
        .map(|i| {
            Observation::new(
                i,
                support(
                    [(i / 2) as f64 * 0.6, 0., 0.],
                    if i % 3 == 0 { [0.; 3] } else { [0.15, 0., 0.] },
                    if i % 3 == 0 { [1; 3] } else { [2, 1, 1] },
                    (i % 2) as usize,
                ),
                if i == 1 { -0. } else { (i as f64).sin() },
                if i == 1 { 0. } else { 0.03 },
            )
            .unwrap()
        })
        .collect();
    let hard_path = dir.path().join("hard.sqlite");
    let mut hard = SqliteStore::create(&hard_path, 4096, &budget).unwrap();
    hard.import(data.as_slice(), &cancel).unwrap();
    drop(hard);
    let hard = SqliteReadPool::open_snapshot(&hard_path, 4, 4096, &budget, &cancel).unwrap();
    let model = Lmc::new(vec![Structure {
        correlation: Correlation::Exponential,
        anisotropy: Anisotropy::isotropic(2.).unwrap(),
        coregionalization: Coregionalization::from_factor(2, 2, &[1., 0., -0.3, 0.8]).unwrap(),
    }])
    .unwrap();
    let groups: Vec<_> = (0..6)
        .map(|i| Group {
            id: i,
            targets: (0..2)
                .map(|v| Target {
                    id: 100 + i * 2 + v as u64,
                    site: if i == 0 && v == 1 {
                        *data[1].site()
                    } else {
                        support([i as f64 * 0.5 + 0.2, 0., 0.], [0.1, 0., 0.], [2, 1, 1], v)
                    },
                })
                .collect(),
        })
        .collect();
    let query = Neighborhood {
        maximum: 16,
        radius: 10.,
    };
    let mean = |s: &Site| {
        let x = s.support.center().coordinates()[0];
        let h = s.support.half_extents()[0];
        Ok(0.2 * (x * x + h * h / 3.) + s.variable as f64)
    };
    let estimate = |source: &(dyn ObservationSource + Sync), workers| {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .unwrap();
        let mut output = Vec::new();
        pool.install(|| {
            estimate_groups(
                &model,
                source,
                groups.clone(),
                &Simple,
                EstimationOptions {
                    neighborhood: query.into(),
                    mean: Some(&mean),
                    uncertainty: Uncertainty::Joint,
                    tolerance: NumericalTolerance::default(),
                    budget: &budget,
                    cancellation: &cancel,
                },
                ExecutionOptions {
                    batch_groups: 3,
                    maximum_targets_per_group: 2,
                    bytes_per_group: 4 << 20,
                },
                |out| {
                    output.extend(out.values.iter().map(|v| v.to_bits()));
                    let c = out.covariance().unwrap();
                    for j in 0..c.ncols() {
                        for i in 0..c.nrows() {
                            output.push(c[(i, j)].to_bits());
                        }
                    }
                    Ok(())
                },
            )
        })
        .unwrap();
        output
    };
    let resident = SpatialIndex::new(data.clone()).unwrap();
    let expected = estimate(&resident, 1);
    for workers in [1, 2, 4, 8] {
        assert_eq!(estimate(&hard, workers), expected);
    }
    let options = |cancel| SimulationOptions {
        seed: 71,
        realization: 2,
        mean: Some(&mean),
        hard_neighborhood: query.into(),
        simulated_neighborhood: query.into(),
        budget: &budget,
        cancellation: cancel,
    };
    let mut resident_output = SpatialIndex::new(Vec::new()).unwrap();
    simulate_groups(
        &model,
        &resident,
        &mut resident_output,
        groups.clone(),
        &Simple,
        options(&cancel),
        |_| Ok(()),
    )
    .unwrap();
    let expected = rows(&resident_output);
    assert_eq!(expected[1].2[6], (-0_f64).to_bits());
    let out_path = dir.path().join("output.sqlite");
    let mut disk = SqliteStore::create(&out_path, 4096, &budget).unwrap();
    let stop = Cancellation::default();
    let mut count = 0;
    assert!(simulate_checkpointed(
        &model,
        &hard,
        &mut disk,
        groups.clone(),
        &Simple,
        CheckpointedOptions {
            simulation: options(&stop),
            identity: [7; 32]
        },
        |_| {
            count += 1;
            if count == 2 {
                stop.cancel();
            }
            Ok(())
        }
    )
    .is_err());
    assert_eq!(disk.len().unwrap(), 4);
    drop(disk);
    let mut disk = SqliteStore::open(&out_path, 4096, &budget).unwrap();
    simulate_checkpointed(
        &model,
        &hard,
        &mut disk,
        groups,
        &Simple,
        CheckpointedOptions {
            simulation: options(&cancel),
            identity: [7; 32],
        },
        |_| Ok(()),
    )
    .unwrap();
    assert_eq!(rows(&disk), expected);
    disk.integrity_check().unwrap();
    drop((disk, hard));
    assert_eq!(budget.used(), 0);
}
