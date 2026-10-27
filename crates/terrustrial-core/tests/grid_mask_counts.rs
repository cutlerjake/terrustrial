#![cfg(feature = "fft")]
use terrustrial_core::{
    execution::{Cancellation, MemoryBudget},
    grid::{Layout, Source},
    variography::grid::counts::Workspace,
    Error, Result,
};

struct Data {
    layout: Layout,
    values: Vec<Vec<Option<f64>>>,
    reads: u128,
    fail: bool,
    cancel: Option<Cancellation>,
}
impl Source for Data {
    fn layout(&self) -> Layout {
        self.layout
    }
    fn variables(&self) -> usize {
        self.values.len()
    }
    fn read(&mut self, variable: usize, first: u64, out: &mut [Option<f64>]) -> Result<()> {
        if self.fail {
            return Err(Error::InvalidInput("injected source failure".into()));
        }
        self.reads += out.len() as u128;
        out.copy_from_slice(&self.values[variable][first as usize..first as usize + out.len()]);
        if let Some(cancel) = &self.cancel {
            cancel.cancel();
        }
        Ok(())
    }
}
fn data(shape: [u64; 3]) -> Data {
    let layout = Layout::new(shape).unwrap();
    Data {
        layout,
        values: (0..32)
            .map(|v| {
                (0..layout.cells())
                    .map(|i| {
                        (!(i * 7 + v * 3).is_multiple_of(5 + v % 4)).then_some(if i % 3 == 0 {
                            f64::MAX
                        } else {
                            -0.0
                        })
                    })
                    .collect()
            })
            .collect(),
        reads: 0,
        fail: false,
        cancel: None,
    }
}
fn coordinates(layout: Layout, i: usize) -> [i64; 3] {
    let [nx, ny, _] = layout.dimensions();
    [
        (i as u64 % nx) as i64,
        (i as u64 / nx % ny) as i64,
        (i as u64 / nx / ny) as i64,
    ]
}
fn oracle(data: &Data, variables: [usize; 2]) -> std::collections::BTreeMap<[i64; 3], u64> {
    let n = data.layout.cells() as usize;
    let mut result = std::collections::BTreeMap::new();
    for i in 0..n {
        if variables.iter().any(|&v| data.values[v][i].is_none()) {
            continue;
        }
        for j in 0..n {
            if variables.iter().any(|&v| data.values[v][j].is_none()) {
                continue;
            }
            let a = coordinates(data.layout, i);
            let b = coordinates(data.layout, j);
            *result
                .entry(std::array::from_fn(|axis| b[axis] - a[axis]))
                .or_default() += 1;
        }
    }
    result
}

#[test]
fn all_signed_lags_match_independent_ordered_pair_enumeration() {
    let budget = MemoryBudget::new(32 << 20);
    let cancel = Cancellation::default();
    for workers in [1, 2, 4, 8] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .unwrap();
        for tile in [1, 7, 128] {
            let mut workspace = Workspace::new(tile, &budget).unwrap();
            for shape in [
                [5, 4, 3],
                [1, 7, 1],
                [17, 1, 1],
                [2, 2, 2],
                [1, 1, 1],
                [9, 3, 2],
            ] {
                let mut source = data(shape);
                for variables in [[0, 0], [0, 1], [31, 7]] {
                    let expected = oracle(&source, variables);
                    let before = source.reads;
                    pool.install(|| {
                        let result = workspace.compute(&mut source, variables, &cancel).unwrap();
                        assert_eq!(result.layout(), source.layout);
                        assert_eq!(result.source_values_read, source.reads - before);
                        assert_eq!(
                            result.eligible_cells(),
                            *expected.get(&[0; 3]).unwrap_or(&0)
                        );
                        for z in -(shape[2] as i64)..=shape[2] as i64 {
                            for y in -(shape[1] as i64)..=shape[1] as i64 {
                                for x in -(shape[0] as i64)..=shape[0] as i64 {
                                    let offset = [x, y, z];
                                    assert_eq!(
                                        result.pairs(offset),
                                        *expected.get(&offset).unwrap_or(&0),
                                        "shape={shape:?}, lag={offset:?}, variables={variables:?}"
                                    );
                                    assert_eq!(
                                        result.pairs(offset),
                                        result.pairs(offset.map(|x| -x))
                                    );
                                }
                            }
                        }
                        assert_eq!(result.pairs([i64::MIN, 0, 0]), 0);
                        assert_eq!(result.pairs([0, i64::MAX, 0]), 0);
                    });
                }
            }
        }
    }
    assert_eq!(budget.used(), 0);
}

#[test]
fn complete_and_empty_masks_do_not_allocate_transform_storage() {
    for present in [true, false] {
        let budget = MemoryBudget::new(32 * 5);
        let mut source = data([13, 11, 7]);
        source
            .values
            .iter_mut()
            .for_each(|v| v.fill(present.then_some(-0.)));
        let mut workspace = Workspace::new(5, &budget).unwrap();
        let counts = workspace
            .compute(&mut source, [0, 31], &Cancellation::default())
            .unwrap();
        assert!(!counts.uses_transform());
        assert_eq!(
            counts.pairs([2, -3, 1]),
            if present { 11 * 8 * 6 } else { 0 }
        );
        assert_eq!(counts.pairs([0; 3]), if present { 13 * 11 * 7 } else { 0 });
        assert_eq!(counts.source_values_read, 2 * 13 * 11 * 7);
        assert_eq!(budget.used(), 32 * 5);
        drop(workspace);
        assert_eq!(budget.used(), 0);
    }
}

#[test]
fn failures_release_new_reservations_and_preserve_workspace_reuse() {
    let budget = MemoryBudget::new(32 * 5);
    let mut workspace = Workspace::new(5, &budget).unwrap();
    let mut source = data([8, 4, 2]);
    assert!(matches!(
        workspace.compute(&mut source, [0, 1], &Cancellation::default()),
        Err(Error::MemoryBudget { .. })
    ));
    assert_eq!(budget.used(), 32 * 5);
    source.values.iter_mut().for_each(|v| v.fill(Some(1.)));
    assert_eq!(
        workspace
            .compute(&mut source, [0, 1], &Cancellation::default())
            .unwrap()
            .pairs([1, 0, 0]),
        7 * 4 * 2
    );
    assert!(workspace
        .compute(&mut source, [32, 0], &Cancellation::default())
        .is_err());
    assert!(Workspace::new(0, &budget).is_err());
    assert!(Workspace::new(usize::MAX, &budget).is_err());
    drop(workspace);
    assert_eq!(budget.used(), 0);

    let budget = MemoryBudget::new(1 << 20);
    let mut workspace = Workspace::new(5, &budget).unwrap();
    let mut source = data([8, 4, 2]);
    let expected = oracle(&source, [0, 1]);
    workspace
        .compute(&mut source, [0, 1], &Cancellation::default())
        .unwrap();
    let retained = budget.used();
    source.fail = true;
    assert!(workspace
        .compute(&mut source, [0, 1], &Cancellation::default())
        .is_err());
    source.fail = false;
    let old = source.values[0][0];
    source.values[0][0] = Some(f64::NAN);
    assert!(workspace
        .compute(&mut source, [0, 1], &Cancellation::default())
        .is_err());
    source.values[0][0] = old;
    let cancelled = Cancellation::default();
    source.cancel = Some(cancelled.clone());
    assert!(matches!(
        workspace.compute(&mut source, [0, 1], &cancelled),
        Err(Error::Cancelled)
    ));
    source.cancel = None;
    assert_eq!(budget.used(), retained);
    let counts = workspace
        .compute(&mut source, [0, 1], &Cancellation::default())
        .unwrap();
    for (lag, count) in expected {
        assert_eq!(counts.pairs(lag), count);
    }
    drop(workspace);
    assert_eq!(budget.used(), 0);
}

#[test]
fn warmed_transform_reuse_has_no_calling_thread_allocations() {
    let budget = MemoryBudget::new(2 << 20);
    let mut workspace = Workspace::new(31, &budget).unwrap();
    let mut source = data([31, 9, 2]);
    let cancel = Cancellation::default();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    pool.install(|| {
        let expected = workspace
            .compute(&mut source, [0, 1], &cancel)
            .unwrap()
            .pairs([3, -2, 1]);
        let allocation = allocation_counter::measure(|| {
            let result = workspace.compute(&mut source, [0, 1], &cancel).unwrap();
            assert!(result.uses_transform());
            assert_eq!(result.pairs([3, -2, 1]), expected);
        });
        assert_eq!(allocation.count_total, 0);
        assert_eq!(allocation.bytes_total, 0);
    });
    drop(workspace);
    assert_eq!(budget.used(), 0);
}

#[test]
fn file_source_counts_match_resident_joint_masks() {
    let budget = MemoryBudget::new(4 << 20);
    let cancel = Cancellation::default();
    let mut source = data([9, 7, 3]);
    let expected = oracle(&source, [31, 7]);
    let mut file = tempfile::tempfile().unwrap();
    terrustrial_core::storage::grid::write(&mut source, &mut file, 11, &budget, &cancel).unwrap();
    let mut file = terrustrial_core::storage::grid::FileSource::open(file, &budget).unwrap();
    let mut workspace = Workspace::new(13, &budget).unwrap();
    let counts = workspace.compute(&mut file, [31, 7], &cancel).unwrap();
    for (lag, count) in expected {
        assert_eq!(counts.pairs(lag), count);
    }
    drop(workspace);
    drop(file);
    assert_eq!(budget.used(), 0);
}

#[test]
fn logical_huge_source_cancels_after_bounded_read_without_global_allocation() {
    struct Logical {
        cancel: Cancellation,
        read: usize,
    }
    impl Source for Logical {
        fn layout(&self) -> Layout {
            Layout::new([1 << 40, 1, 1]).unwrap()
        }
        fn variables(&self) -> usize {
            1
        }
        fn read(&mut self, _: usize, _: u64, out: &mut [Option<f64>]) -> Result<()> {
            self.read += out.len();
            out.fill(Some(1.));
            self.cancel.cancel();
            Ok(())
        }
    }
    let budget = MemoryBudget::new(512);
    let cancel = Cancellation::default();
    let mut source = Logical {
        cancel: cancel.clone(),
        read: 0,
    };
    let mut workspace = Workspace::new(16, &budget).unwrap();
    assert!(matches!(
        workspace.compute(&mut source, [0, 0], &cancel),
        Err(Error::Cancelled)
    ));
    assert_eq!(source.read, 16);
    assert_eq!(budget.used(), 512);
    drop(workspace);
    assert_eq!(budget.used(), 0);
}
