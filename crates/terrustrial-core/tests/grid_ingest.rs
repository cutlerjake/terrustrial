use std::{
    fs::{self, File},
    path::Path,
};
use terrustrial_core::{
    execution::{Cancellation, MemoryBudget},
    grid::{Grid, Layout, Shape, Source, Variable},
    storage::grid::ingest::{self, Options, Rows},
    Error, Result,
};

struct Input {
    rows: std::vec::IntoIter<(u64, Vec<Option<f64>>)>,
    reads: usize,
}
impl Input {
    fn new(rows: Vec<(u64, Vec<Option<f64>>)>) -> Self {
        Self {
            rows: rows.into_iter(),
            reads: 0,
        }
    }
}
impl Rows for Input {
    fn next(&mut self, values: &mut [Option<f64>]) -> Result<Option<u64>> {
        self.reads += 1;
        Ok(self.rows.next().map(|(id, v)| {
            values.copy_from_slice(&v);
            id
        }))
    }
}
fn options<'a>(
    n: u64,
    capacity: usize,
    fan_in: usize,
    budget: &'a MemoryBudget,
    cancellation: &'a Cancellation,
) -> Options<'a> {
    Options {
        layout: Layout::new([n, 1, 1]).unwrap(),
        variables: 2,
        rows_per_run: capacity,
        merge_fan_in: fan_in,
        io_buffer_bytes: 64,
        disk_limit_bytes: 1 << 30,
        budget,
        cancellation,
    }
}
fn attributes(i: u64) -> Vec<Option<f64>> {
    vec![
        if i == 0 {
            Some(-0.0)
        } else if i.is_multiple_of(13) {
            None
        } else {
            Some(i as f64 / 10.0)
        },
        if i.is_multiple_of(7) {
            None
        } else {
            Some(-(i as f64))
        },
    ]
}
fn directory_count(path: &Path) -> usize {
    fs::read_dir(path).unwrap().count()
}

#[test]
fn arbitrary_order_multi_pass_ingestion_is_bit_identical_to_the_column_writer() -> Result<()> {
    let root = tempfile::tempdir()?;
    let budget = MemoryBudget::new(1 << 20);
    let cancel = Cancellation::default();
    let n = 127;
    let mut reference = Grid::new(
        Shape::new([n as usize, 1, 1])?,
        vec![Variable::Continuous { scale: 1.0 }; 2],
        (0..n).flat_map(attributes).collect(),
        &budget,
    )?;
    let mut expected = vec![];
    terrustrial_core::storage::grid::write(&mut reference, &mut expected, 11, &budget, &cancel)?;
    let used = budget.used();
    for (capacity, fan_in) in [(1, 2), (7, 3), (64, 4), (512, 8)] {
        let rows = (0..n)
            .map(|j| {
                let i = (j * 37) % n;
                (i, attributes(i))
            })
            .collect();
        let mut input = Input::new(rows);
        let output = root.path().join(format!("{capacity}-{fan_in}.trgrid"));
        let report = ingest::import(
            &mut input,
            &output,
            options(n, capacity, fan_in, &budget, &cancel),
        )?;
        assert_eq!(report.rows, n);
        assert_eq!(report.initial_runs, n.div_ceil(capacity as u64));
        assert_eq!(report.output_bytes, expected.len() as u64);
        assert!(report.maximum_working_reservation_bytes < 1 << 20);
        assert_eq!(fs::read(&output)?, expected);
        assert_eq!(budget.used(), used);
        if capacity == 1 {
            assert_eq!(report.merge_passes, 7);
        }
        if capacity == 512 {
            assert_eq!(report.merge_passes, 0);
        }
        let mut file =
            terrustrial_core::storage::grid::FileSource::open(File::open(&output)?, &budget)?;
        let mut zero = [None];
        file.read(0, 0, &mut zero)?;
        assert_eq!(zero[0].unwrap().to_bits(), (-0.0_f64).to_bits());
    }
    assert_eq!(directory_count(root.path()), 4);
    Ok(())
}

#[test]
fn duplicates_missing_rows_invalid_values_and_source_failure_do_not_publish() -> Result<()> {
    let root = tempfile::tempdir()?;
    let budget = MemoryBudget::new(1 << 20);
    let cancel = Cancellation::default();
    for (ids, capacity) in [
        (vec![0, 1, 2, 2], 4),
        (vec![0, 1, 2, 2], 1),
        (vec![0, 1, 2], 2),
        (vec![0, 1, 2, 4], 2),
        (vec![0, 1, 2, 3, 0], 2),
    ] {
        let mut input = Input::new(ids.into_iter().map(|i| (i, attributes(i))).collect());
        let output = root.path().join("fail.trgrid");
        assert!(ingest::import(
            &mut input,
            &output,
            options(4, capacity, 2, &budget, &cancel)
        )
        .is_err());
        assert!(!output.exists());
        assert_eq!(directory_count(root.path()), 0);
        assert_eq!(budget.used(), 0);
    }
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut input = Input::new(vec![(0, vec![Some(bad), None])]);
        assert!(ingest::import(
            &mut input,
            &root.path().join("bad"),
            options(1, 2, 2, &budget, &cancel)
        )
        .is_err());
        assert_eq!(directory_count(root.path()), 0);
    }
    struct Failing(usize);
    impl Rows for Failing {
        fn next(&mut self, out: &mut [Option<f64>]) -> Result<Option<u64>> {
            if self.0 == 5 {
                return Err(Error::Io(std::io::Error::other("injected input failure")));
            }
            out.fill(None);
            let i = self.0;
            self.0 += 1;
            Ok(Some(i as u64))
        }
    }
    assert!(matches!(
        ingest::import(
            &mut Failing(0),
            &root.path().join("bad"),
            options(8, 2, 2, &budget, &cancel)
        ),
        Err(Error::Io(_))
    ));
    assert_eq!(directory_count(root.path()), 0);
    assert_eq!(budget.used(), 0);
    Ok(())
}

#[test]
fn disk_and_memory_limits_cover_spill_merge_and_output_phases() -> Result<()> {
    let root = tempfile::tempdir()?;
    let budget = MemoryBudget::new(1 << 20);
    let cancel = Cancellation::default();
    for (limit, capacity) in [(1, 2), (184, 2), (300, 2), (350, 8)] {
        let mut input = Input::new((0..8).map(|i| (i, attributes(i))).collect());
        let mut settings = options(8, capacity, 2, &budget, &cancel);
        settings.disk_limit_bytes = limit;
        assert!(
            matches!(
                ingest::import(&mut input, &root.path().join("limited"), settings),
                Err(Error::DiskBudget { .. })
            ),
            "{limit}, {capacity}"
        );
        assert_eq!(directory_count(root.path()), 0);
        assert_eq!(budget.used(), 0);
    }
    for (limit, capacity, fan_in) in [(1, 2, 2), (600, 2, 4)] {
        let budget = MemoryBudget::new(limit);
        let mut input = Input::new((0..8).map(|i| (i, attributes(i))).collect());
        assert!(matches!(
            ingest::import(
                &mut input,
                &root.path().join("limited"),
                options(8, capacity, fan_in, &budget, &cancel)
            ),
            Err(Error::MemoryBudget { .. })
        ));
        assert_eq!(directory_count(root.path()), 0);
        assert_eq!(budget.used(), 0);
    }
    Ok(())
}

#[test]
fn cancellation_releases_runs_and_existing_destinations_are_never_replaced() -> Result<()> {
    let root = tempfile::tempdir()?;
    let budget = MemoryBudget::new(1 << 20);
    let cancel = Cancellation::default();
    struct Cancelling {
        count: u64,
        cancel: Cancellation,
    }
    impl Rows for Cancelling {
        fn next(&mut self, values: &mut [Option<f64>]) -> Result<Option<u64>> {
            let i = self.count;
            self.count += 1;
            values.fill(None);
            if self.count == 5 {
                self.cancel.cancel();
            }
            Ok(Some(i))
        }
    }
    let mut source = Cancelling {
        count: 0,
        cancel: cancel.clone(),
    };
    let mut large = options(1_000_000_000_000, 2, 2, &budget, &cancel);
    large.disk_limit_bytes = 1 << 46;
    assert!(matches!(
        ingest::import(&mut source, &root.path().join("cancelled"), large),
        Err(Error::Cancelled)
    ));
    assert_eq!(directory_count(root.path()), 0);
    assert_eq!(budget.used(), 0);
    let cancel = Cancellation::default();
    let destination = root.path().join("existing");
    fs::write(&destination, b"keep me")?;
    let mut input = Input::new(vec![(0, attributes(0))]);
    assert!(ingest::import(&mut input, &destination, options(1, 1, 2, &budget, &cancel)).is_err());
    assert_eq!(input.reads, 0);
    assert_eq!(fs::read(&destination)?, b"keep me");
    struct Racing<'a>(&'a Path, bool);
    impl Rows for Racing<'_> {
        fn next(&mut self, values: &mut [Option<f64>]) -> Result<Option<u64>> {
            if self.1 {
                return Ok(None);
            }
            self.1 = true;
            fs::write(self.0, b"concurrent writer")?;
            values.fill(None);
            Ok(Some(0))
        }
    }
    let race = root.path().join("race");
    assert!(matches!(
        ingest::import(
            &mut Racing(&race, false),
            &race,
            options(1, 1, 2, &budget, &cancel)
        ),
        Err(Error::Io(_))
    ));
    assert_eq!(fs::read(&race)?, b"concurrent writer");
    assert_eq!(directory_count(root.path()), 2);
    assert_eq!(budget.used(), 0);
    Ok(())
}
