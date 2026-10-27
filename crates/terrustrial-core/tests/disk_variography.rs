#![cfg(feature = "sqlite")]
use std::path::Path;
use terrustrial_core::{
    data::Point3,
    execution::{Cancellation, MemoryBudget},
    variography::{
        self,
        disk::{build, BuildOptions, SampleFile},
        indexed::Workspace,
        Bin, Direction, Sample,
    },
    Error, Result,
};

fn fixture(n: usize) -> Vec<Sample> {
    (0..n)
        .map(|i| Sample {
            point: Point3::new([
                (i % 37) as f64 * 0.25,
                (i / 37 % 19) as f64 * 0.3,
                (i / 703) as f64,
            ])
            .unwrap(),
            values: vec![
                (i % 13 != 0).then_some((i % 17) as f64 - 8.),
                (i % 11 != 0).then_some(4. - (i % 17) as f64 * 0.5),
                Some((i % 7) as f64),
            ],
        })
        .collect()
}
fn write(path: &Path, data: &[Sample], budget: &MemoryBudget) {
    let mut iter = data.iter().enumerate();
    let mut rows = |values: &mut [Option<f64>]| -> Result<_> {
        Ok(iter.next().map(|(i, sample)| {
            values.copy_from_slice(&sample.values);
            (u64::MAX - i as u64, sample.point)
        }))
    };
    let report = build(
        &mut rows,
        path,
        BuildOptions {
            variables: 3,
            cache_bytes: 4096,
            maximum_database_bytes: 16 << 20,
            budget,
            cancellation: &Cancellation::default(),
        },
    )
    .unwrap();
    assert_eq!(report.samples, data.len() as u64);
    assert_eq!(report.variables, 3);
    assert_eq!(
        report.database_bytes,
        std::fs::metadata(path).unwrap().len()
    );
}
fn close(a: &[Bin], b: &[Bin]) {
    assert_eq!(a.len(), b.len());
    for (a, b) in a.iter().zip(b) {
        assert_eq!(a.pairs, b.pairs);
        for (a, b) in [
            (a.mean_distance, b.mean_distance),
            (a.semivariance, b.semivariance),
        ] {
            match (a, b) {
                (Some(a), Some(b)) => {
                    assert!((a - b).abs() / (1. + b.abs()) <= 2e-12, "{a} != {b}")
                }
                (None, None) => (),
                _ => panic!("empty-bin mismatch"),
            }
        }
    }
}
fn bits(bins: &[Bin]) -> Vec<(u64, Option<u64>, Option<u64>)> {
    bins.iter()
        .map(|b| {
            (
                b.pairs,
                b.mean_distance.map(f64::to_bits),
                b.semivariance.map(f64::to_bits),
            )
        })
        .collect()
}

#[test]
fn disk_exact_pairs_match_oracle_and_reader_window_worker_counts() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("samples.sqlite");
    let data = fixture(1200);
    let budget = MemoryBudget::new(16 << 20);
    write(&path, &data, &budget);
    assert_eq!(budget.used(), 0);
    let cancel = Cancellation::default();
    let direction = Direction::new([1., -2., 0.5], 0.4, 1.).unwrap();
    let mut expected = Vec::new();
    let edges = [0.25, 0.5, 1., 2., 4.];
    for readers in [1, 2, 4, 8] {
        let mut file = SampleFile::open(&path, readers, 4096, &budget, &cancel).unwrap();
        assert_eq!(file.len(), data.len() as u64);
        assert_eq!(file.variables(), 3);
        assert!(!file.is_empty());
        assert_eq!(file.path(), path.canonicalize().unwrap());
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(readers)
            .build()
            .unwrap();
        for window in [1, 3, 16] {
            let mut workspace = Workspace::new(8, window, &budget).unwrap();
            let mut case = 0;
            pool.install(|| {
                for variables in [[0, 0], [0, 1], [1, 2]] {
                    for direction in [None, Some(&direction)] {
                        let actual = file
                            .experimental(variables, &edges, direction, &mut workspace, &cancel)
                            .unwrap();
                        if readers == 1 && window == 1 {
                            let reference = variography::experimental(
                                &data,
                                variables[0],
                                variables[1],
                                &edges,
                                direction,
                                &cancel,
                            )
                            .unwrap();
                            close(actual.bins, &reference);
                            expected.push(bits(actual.bins));
                        } else {
                            assert_eq!(bits(actual.bins), expected[case]);
                        }
                        assert_eq!(actual.population_pairs, 1200 * 1199 / 2);
                        assert!(actual.candidate_pairs <= actual.population_pairs);
                        case += 1;
                    }
                }
            });
        }
    }
    assert_eq!(budget.used(), 0);
}

#[test]
fn boundaries_extreme_coordinates_missingness_and_zero_bits_roundtrip() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("samples.sqlite");
    let tiny = f64::from_bits(1);
    let coordinates = [
        -f64::MAX,
        -1e39,
        -1.,
        -0.,
        0.,
        tiny,
        2. * tiny,
        1e-30,
        1_f64.next_down(),
        1.,
        1_f64.next_up(),
        1e39,
        f64::MAX,
    ];
    let data: Vec<_> = coordinates
        .into_iter()
        .enumerate()
        .map(|(i, x)| Sample {
            point: Point3::new([x, -0., 0.]).unwrap(),
            values: vec![Some(i as f64), Some(-(i as f64)), Some(-0.)],
        })
        .collect();
    let budget = MemoryBudget::new(1 << 20);
    write(&path, &data, &budget);
    let connection = rusqlite::Connection::open(&path).unwrap();
    let raw: Vec<u8> = connection
        .query_row("SELECT payload FROM samples WHERE ordinal=3", [], |r| {
            r.get(0)
        })
        .unwrap();
    assert_eq!(
        u64::from_le_bytes(raw[..8].try_into().unwrap()),
        (-0_f64).to_bits()
    );
    assert_eq!(
        u64::from_le_bytes(raw[8..16].try_into().unwrap()),
        (-0_f64).to_bits()
    );
    assert_eq!(
        u64::from_le_bytes(raw[43..51].try_into().unwrap()),
        (-0_f64).to_bits()
    );
    let id: i64 = connection
        .query_row("SELECT id FROM samples WHERE ordinal=0", [], |r| r.get(0))
        .unwrap();
    assert_eq!(id as u64, u64::MAX);
    drop(connection);
    let cancel = Cancellation::default();
    let mut file = SampleFile::open(&path, 2, 4096, &budget, &cancel).unwrap();
    let mut workspace = Workspace::new(4, 4, &budget).unwrap();
    let edges = [tiny, 1., 2., f64::MAX];
    let reference = variography::experimental(&data, 0, 1, &edges, None, &cancel).unwrap();
    close(
        file.experimental([0, 1], &edges, None, &mut workspace, &cancel)
            .unwrap()
            .bins,
        &reference,
    );
}

#[test]
fn failed_builds_leave_no_destination_and_release_accounting() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("samples.sqlite");
    let budget = MemoryBudget::new(1 << 20);
    for kind in 0..5 {
        let cancel = Cancellation::default();
        let mut count = 0_u64;
        let mut rows = |values: &mut [Option<f64>]| -> Result<_> {
            if count == 100 {
                match kind {
                    0 => values[0] = Some(f64::NAN),
                    1 => return Err(Error::Backend("source failure".into())),
                    2 => cancel.cancel(),
                    _ => (),
                }
            } else {
                values.fill(Some(1.));
            }
            count += 1;
            Ok((count < 1000).then_some((
                if kind == 3 { 0 } else { count },
                Point3::new([count as f64, 0., 0.]).unwrap(),
            )))
        };
        let result = build(
            &mut rows,
            &path,
            BuildOptions {
                variables: 3,
                cache_bytes: 4096,
                maximum_database_bytes: if kind == 4 { 4096 } else { 4 << 20 },
                budget: &budget,
                cancellation: &cancel,
            },
        );
        assert!(result.is_err(), "kind {kind}");
        assert!(!path.exists());
        assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 0);
        assert_eq!(budget.used(), 0);
    }
    write(&path, &fixture(1), &budget);
    let before = std::fs::read(&path).unwrap();
    let mut never =
        |_: &mut [Option<f64>]| -> Result<_> { panic!("source read for existing destination") };
    assert!(build(
        &mut never,
        &path,
        BuildOptions {
            variables: 3,
            cache_bytes: 4096,
            maximum_database_bytes: 4 << 20,
            budget: &budget,
            cancellation: &Cancellation::default()
        }
    )
    .is_err());
    assert_eq!(std::fs::read(&path).unwrap(), before);
    assert_eq!(budget.used(), 0);
}

#[test]
fn corrupted_payloads_counts_and_spatial_index_are_rejected_on_open() {
    let directory = tempfile::tempdir().unwrap();
    let original = directory.path().join("original.sqlite");
    let budget = MemoryBudget::new(1 << 20);
    write(&original, &fixture(40), &budget);
    for (i, sql) in [
        "UPDATE samples SET payload=x'01' WHERE ordinal=0",
        "UPDATE samples SET x=123 WHERE ordinal=0",
        "DELETE FROM samples WHERE ordinal=3",
        "DELETE FROM bounds WHERE ordinal=3",
        "INSERT INTO bounds VALUES(1000,0,0,0,0,0,0)",
        "UPDATE metadata SET samples=1",
        "UPDATE metadata SET variables=0",
        "UPDATE samples SET indexed=0 WHERE ordinal=3",
        "UPDATE bounds SET min_x=123,max_x=123 WHERE ordinal=3",
        "PRAGMA user_version=999",
    ]
    .iter()
    .enumerate()
    {
        let path = directory.path().join(format!("bad-{i}.sqlite"));
        std::fs::copy(&original, &path).unwrap();
        let connection = rusqlite::Connection::open(&path).unwrap();
        connection.execute_batch(sql).unwrap();
        drop(connection);
        assert!(
            SampleFile::open(&path, 2, 4096, &budget, &Cancellation::default()).is_err(),
            "{sql}"
        );
        assert_eq!(budget.used(), 0);
    }
}

#[test]
fn reader_snapshot_survives_wal_changes_and_failed_queries_are_reusable() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("samples.sqlite");
    let data = fixture(700);
    let budget = MemoryBudget::new(1 << 20);
    write(&path, &data, &budget);
    let writer = rusqlite::Connection::open(&path).unwrap();
    writer.pragma_update(None, "journal_mode", "WAL").unwrap();
    let cancel = Cancellation::default();
    let mut file = SampleFile::open(&path, 4, 4096, &budget, &cancel).unwrap();
    let mut workspace = Workspace::new(3, 4, &budget).unwrap();
    let expected = bits(
        file.experimental([0, 1], &[1., 2., 4.], None, &mut workspace, &cancel)
            .unwrap()
            .bins,
    );
    // Corrupt the newer view while all four readers retain the validated old one.
    writer
        .execute("UPDATE samples SET payload=x'01' WHERE ordinal=200", [])
        .unwrap();
    assert_eq!(
        bits(
            file.experimental([0, 1], &[1., 2., 4.], None, &mut workspace, &cancel)
                .unwrap()
                .bins
        ),
        expected
    );
    assert!(SampleFile::open(&path, 1, 4096, &budget, &cancel).is_err());
    assert!(file
        .experimental([3, 0], &[1.], None, &mut workspace, &cancel)
        .is_err());
    assert!(file
        .experimental([0, 1], &[f64::NAN], None, &mut workspace, &cancel)
        .is_err());
    cancel.cancel();
    assert!(matches!(
        file.experimental([0, 1], &[1.], None, &mut workspace, &cancel),
        Err(Error::Cancelled)
    ));
    assert_eq!(
        bits(
            file.experimental(
                [0, 1],
                &[1., 2., 4.],
                None,
                &mut workspace,
                &Cancellation::default()
            )
            .unwrap()
            .bins
        ),
        expected
    );
    drop(file);
    drop(workspace);
    drop(writer);
    assert_eq!(budget.used(), 0);
}

#[test]
fn empty_file_and_invalid_resource_limits_are_explicit() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("empty.sqlite");
    let budget = MemoryBudget::new(1 << 20);
    let cancel = Cancellation::default();
    write(&path, &[], &budget);
    assert!(SampleFile::open(&path, 0, 4096, &budget, &cancel).is_err());
    assert!(SampleFile::open(&path, 1, 4096, &MemoryBudget::new(1), &cancel).is_err());
    assert!(SampleFile::open(&path, 4, 4096, &MemoryBudget::new(20000), &cancel).is_err());
    let mut file = SampleFile::open(&path, 2, 4096, &budget, &cancel).unwrap();
    assert!(file.is_empty());
    let mut workspace = Workspace::new(1, 4, &budget).unwrap();
    let bins = file
        .experimental([0, 1], &[1.], None, &mut workspace, &cancel)
        .unwrap();
    assert_eq!(bins.population_pairs, 0);
    assert_eq!(bins.candidate_pairs, 0);
    assert_eq!(bins.bins[0].pairs, 0);
    assert!(bins.bins[0].semivariance.is_none());
    drop(file);
    drop(workspace);
    assert_eq!(budget.used(), 0);
    let mut never = |_: &mut [Option<f64>]| -> Result<_> { panic!("invalid request read source") };
    for variables in [0, usize::MAX] {
        assert!(build(
            &mut never,
            &directory.path().join("bad.sqlite"),
            BuildOptions {
                variables,
                cache_bytes: 4096,
                maximum_database_bytes: 1 << 20,
                budget: &budget,
                cancellation: &cancel
            }
        )
        .is_err());
    }
    assert_eq!(budget.used(), 0);
}

#[test]
fn inflight_cancellation_and_numerical_failure_preserve_reuse() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("dense.sqlite");
    let budget = MemoryBudget::new(1 << 20);
    let mut data = fixture(2500);
    for sample in &mut data {
        sample.point = Point3::new([0.; 3]).unwrap();
        sample.values[2] = None;
    }
    write(&path, &data, &budget);
    let cancel = Cancellation::default();
    let mut file = SampleFile::open(&path, 4, 4096, &budget, &cancel).unwrap();
    let mut workspace = Workspace::new(1, 4, &budget).unwrap();
    let barrier = std::sync::Barrier::new(2);
    std::thread::scope(|scope| {
        scope.spawn(|| {
            barrier.wait();
            std::thread::sleep(std::time::Duration::from_millis(10));
            cancel.cancel();
        });
        barrier.wait();
        assert!(matches!(
            file.experimental([0, 1], &[1.], None, &mut workspace, &cancel),
            Err(Error::Cancelled)
        ));
    });
    let result = file
        .experimental(
            [2, 2],
            &[1.],
            None,
            &mut workspace,
            &Cancellation::default(),
        )
        .unwrap();
    assert_eq!(result.bins[0].pairs, 0);
    drop(file);
    let path = directory.path().join("extreme.sqlite");
    let data = [
        Sample {
            point: Point3::new([0.; 3]).unwrap(),
            values: vec![Some(f64::MAX), Some(0.25), None],
        },
        Sample {
            point: Point3::new([0.; 3]).unwrap(),
            values: vec![Some(-f64::MAX), Some(-0.25), None],
        },
    ];
    write(&path, &data, &budget);
    let mut file = SampleFile::open(&path, 1, 4096, &budget, &Cancellation::default()).unwrap();
    assert!(matches!(
        file.experimental(
            [0, 0],
            &[1.],
            None,
            &mut workspace,
            &Cancellation::default()
        ),
        Err(Error::Numerical { .. })
    ));
    assert_eq!(
        file.experimental(
            [0, 1],
            &[1.],
            None,
            &mut workspace,
            &Cancellation::default()
        )
        .unwrap()
        .bins[0]
            .semivariance,
        Some(f64::MAX * 0.5)
    );
    drop(file);
    drop(workspace);
    assert_eq!(budget.used(), 0);
}

#[test]
fn ordered_and_scattered_source_blocks_preserve_exact_pair_membership() {
    let directory = tempfile::tempdir().unwrap();
    let budget = MemoryBudget::new(16 << 20);
    let cancel = Cancellation::default();
    let mut original = fixture(2049);
    for (i, row) in original.iter_mut().enumerate() {
        row.point = Point3::new([i as f64, (i % 7) as f64 * 0.03, (i % 11) as f64 * 0.02]).unwrap();
    }
    let edges = [1.55, 3.1];
    let reference = variography::experimental(&original, 0, 1, &edges, None, &cancel).unwrap();
    for shuffled in [false, true] {
        let rows: Vec<_> = (0..original.len())
            .map(|i| {
                original[if shuffled {
                    i * 1093 % original.len()
                } else {
                    i
                }]
                .clone()
            })
            .collect();
        let path = directory.path().join(format!("order-{shuffled}.sqlite"));
        write(&path, &rows, &budget);
        let mut expected = None;
        for readers in [1, 4] {
            let mut file = SampleFile::open(&path, readers, 4096, &budget, &cancel).unwrap();
            let mut workspace = Workspace::new(2, 5, &budget).unwrap();
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(readers)
                .build()
                .unwrap();
            pool.install(|| {
                let result = file
                    .experimental([0, 1], &edges, None, &mut workspace, &cancel)
                    .unwrap();
                close(result.bins, &reference);
                let current = bits(result.bins);
                if let Some(expected) = &expected {
                    assert_eq!(&current, expected);
                }
                expected = Some(current);
            });
        }
    }
    assert_eq!(budget.used(), 0);
}
