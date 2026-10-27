#![cfg(feature = "fft")]
use std::collections::BTreeMap;
use terrustrial_core::{
    execution::{Cancellation, MemoryBudget},
    grid::{Layout, Source},
    variography::grid::{
        self as variography,
        counts::{self, Extent},
        spectral::{Accuracy, Method, Workspace},
    },
    Error, Result,
};

struct Local {
    layout: Layout,
    extent: Extent,
    holes: bool,
    reads: u128,
    invalid_exterior: bool,
}
impl Local {
    fn new(shape: [u64; 3], origin: [u64; 3], dimensions: [u64; 3], holes: bool) -> Self {
        Self {
            layout: Layout::new(shape).unwrap(),
            extent: Extent { origin, dimensions },
            holes,
            reads: 0,
            invalid_exterior: false,
        }
    }
    fn value(&self, variable: usize, point: [u64; 3]) -> Option<f64> {
        if (0..3).any(|a| {
            point[a] < self.extent.origin[a]
                || point[a] - self.extent.origin[a] >= self.extent.dimensions[a]
        }) {
            // A large singly present attribute must not expand joint support or
            // change its normalization. Every exterior value is still validated.
            return (variable == 0).then_some(if self.invalid_exterior {
                f64::NAN
            } else {
                f64::MAX
            });
        }
        let local: [u64; 3] = std::array::from_fn(|a| point[a] - self.extent.origin[a]);
        let [nx, ny, _] = self.extent.dimensions;
        let i = local[0] + nx * (local[1] + ny * local[2]);
        if self.holes
            && i != 0
            && i + 1 != self.extent.dimensions.iter().product::<u64>()
            && (i + variable as u64 * 3) % 11 == 5
        {
            None
        } else {
            Some(((i * i * 7 + i * 5 + variable as u64 * 17) % 47) as f64 - 23.)
        }
    }
    fn oracle(&self) -> BTreeMap<[i64; 3], (u64, f64)> {
        let mut points = vec![];
        let [nx, ny, nz] = self.extent.dimensions;
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let local = [x, y, z];
                    let point = std::array::from_fn(|a| local[a] + self.extent.origin[a]);
                    if let (Some(a), Some(b)) = (self.value(0, point), self.value(1, point)) {
                        points.push((local.map(|v| v as i64), a, b));
                    }
                }
            }
        }
        let mut result = BTreeMap::new();
        for a in &points {
            for b in &points {
                let lag = std::array::from_fn(|i| b.0[i] - a.0[i]);
                let entry = result.entry(lag).or_insert((0, 0.));
                entry.0 += 1;
                entry.1 += 0.5 * (a.1 - b.1) * (a.2 - b.2);
            }
        }
        result
    }
}
impl Source for Local {
    fn layout(&self) -> Layout {
        self.layout
    }
    fn variables(&self) -> usize {
        2
    }
    fn read(&mut self, variable: usize, first: u64, out: &mut [Option<f64>]) -> Result<()> {
        let [nx, ny, _] = self.layout.dimensions();
        self.reads += out.len() as u128;
        for (i, value) in out.iter_mut().enumerate() {
            let i = first + i as u64;
            *value = self.value(variable, [i % nx, i / nx % ny, i / nx / ny]);
        }
        Ok(())
    }
}

#[test]
fn translated_boxes_holes_rectangles_and_coalesced_reads_match_independent_pairs() -> Result<()> {
    let budget = MemoryBudget::new(8 << 20);
    let cancel = Cancellation::default();
    let mut workspace = Workspace::new(19, &budget)?;
    let mut counts = counts::Workspace::new(13, &budget)?;
    for workers in [1, 4] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .unwrap();
        pool.install(|| -> Result<()> {
            for (shape, origin, dimensions) in [
                ([19, 17, 8], [11, 8, 4], [5, 4, 3]),
                ([19, 17, 8], [0, 0, 0], [5, 4, 3]),
                ([7, 11, 5], [0, 2, 1], [7, 5, 3]),
                ([7, 5, 9], [0, 0, 5], [7, 5, 3]),
                ([7, 5, 9], [3, 2, 6], [1, 1, 1]),
                ([7, 5, 9], [1, 1, 6], [3, 2, 1]),
            ] {
                for holes in [false, true] {
                    let mut source = Local::new(shape, origin, dimensions, holes);
                    let expected = source.oracle();
                    let count = counts.compute(&mut source, [0, 1], &cancel)?;
                    assert_eq!(count.layout().dimensions(), shape);
                    assert_eq!(count.eligible_extent(), Some(Extent { origin, dimensions }));
                    if !holes {
                        assert!(!count.uses_transform());
                    }
                    let mut map = workspace.compute(
                        &mut source,
                        [0, 1],
                        [1.; 3],
                        Accuracy::default(),
                        &cancel,
                    )?;
                    assert_eq!(map.eligible_extent(), count.eligible_extent());
                    assert_eq!(map.layout().dimensions(), shape);
                    for z in -(shape[2] as i64)..=shape[2] as i64 {
                        for y in -(shape[1] as i64)..=shape[1] as i64 {
                            for x in -(shape[0] as i64)..=shape[0] as i64 {
                                let lag = [x, y, z];
                                let (pairs, sum) = expected.get(&lag).copied().unwrap_or_default();
                                assert_eq!(count.pairs(lag), pairs, "{lag:?}");
                                let answer = map.lag(lag)?;
                                assert_eq!(answer.statistics.pairs, pairs);
                                if pairs > 0 {
                                    let reference = sum / pairs as f64;
                                    assert!(
                                        (answer.statistics.semivariance.unwrap() - reference).abs()
                                            <= 1e-10 * (1. + reference.abs())
                                    );
                                    let exact = map.lag_direct(lag)?;
                                    assert!(
                                        (exact.statistics.semivariance.unwrap() - reference).abs()
                                            <= 1e-12 * (1. + reference.abs())
                                    );
                                } else {
                                    assert!(answer.statistics.semivariance.is_none());
                                }
                            }
                        }
                    }
                }
            }
            Ok(())
        })?;
    }
    Ok(())
}

#[test]
fn million_cell_source_uses_small_extent_storage_and_bounded_followup_reads() -> Result<()> {
    let budget = MemoryBudget::new(64 << 10);
    let cancel = Cancellation::default();
    let mut source = Local::new([4096, 128, 2], [4000, 120, 0], [5, 4, 2], true);
    let expected = source.oracle();
    let cells = source.layout.cells();
    let extent_cells: u128 = source
        .extent
        .dimensions
        .iter()
        .map(|v| *v as u128)
        .product();
    let mut workspace = Workspace::new(31, &budget)?;
    let map_reads;
    {
        let mut map =
            workspace.compute(&mut source, [0, 1], [1.; 3], Accuracy::default(), &cancel)?;
        // One complete pass, one cropped integer-mask pass, one cropped FFT pass.
        assert_eq!(map.source_values_read, 2 * cells as u128 + 4 * extent_cells);
        let before = map.source_values_read;
        let answer = map.lag_direct([1, 0, 0])?;
        assert_eq!(answer.statistics.pairs, expected[&[1, 0, 0]].0);
        assert_eq!(map.source_values_read - before, 4 * 4 * 4 * 2);
        map_reads = map.source_values_read;
        assert_eq!(map.lag([1000, 0, 0])?.method, Method::Analytic);
    }
    assert_eq!(map_reads, source.reads);
    assert!(budget.used() < 64 << 10);
    // The full padded geometry would require millions of coefficients. This is
    // one million streamed logical cells, not actual beyond-physical-RAM evidence.
    source.invalid_exterior = true;
    assert!(workspace
        .compute(&mut source, [0, 1], [1.; 3], Accuracy::default(), &cancel)
        .is_err());
    source.invalid_exterior = false;
    assert!(workspace
        .compute(&mut source, [0, 1], [1.; 3], Accuracy::default(), &cancel)?
        .lag([1, 0, 0])
        .is_ok());
    drop(workspace);
    assert_eq!(budget.used(), 0);
    Ok(())
}

#[test]
fn crop_preserves_original_streaming_reduction_and_file_results() -> Result<()> {
    let budget = MemoryBudget::new(8 << 20);
    let cancel = Cancellation::default();
    let mut source = Local::new([19, 17, 8], [11, 8, 4], [5, 4, 3], true);
    let offsets = [[1, 0, 0], [-2, 1, 0], [0, -2, 1], [1, 2, -1], [8, 0, 0]];
    let expected = variography::experimental(
        &mut source,
        variography::Options {
            variables: [0, 1],
            offsets: &offsets,
            spacing: [1.; 3],
            tile_cells: 19,
            budget: &budget,
            cancellation: &cancel,
        },
    )?;
    let mut file = tempfile::tempfile()?;
    terrustrial_core::storage::grid::write(&mut source, &mut file, 23, &budget, &cancel)?;
    let mut file = terrustrial_core::storage::grid::FileSource::open(file, &budget)?;
    let mut workspace = Workspace::new(13, &budget)?;
    let mut disk_workspace = Workspace::new(29, &budget)?;
    let mut resident =
        workspace.compute(&mut source, [0, 1], [1.; 3], Accuracy::default(), &cancel)?;
    let mut disk =
        disk_workspace.compute(&mut file, [0, 1], [1.; 3], Accuracy::default(), &cancel)?;
    for lag in &expected.lags {
        assert_eq!(
            resident.lag_direct(lag.offset)?.statistics.semivariance,
            lag.statistics.semivariance
        );
        assert_eq!(
            disk.lag_direct(lag.offset)?.statistics.semivariance,
            lag.statistics.semivariance
        );
        assert_eq!(
            disk.lag(lag.offset)?.statistics.semivariance,
            resident.lag(lag.offset)?.statistics.semivariance
        );
    }
    Ok(())
}

#[test]
fn resized_and_analytic_workloads_release_stale_paired_caches_before_admission() -> Result<()> {
    let cancel = Cancellation::default();
    let mut large = Local::new([64, 16, 4], [0; 3], [64, 16, 4], false);
    let mut smaller = Local::new([32, 16, 4], [0; 3], [32, 16, 4], true);
    let limit = {
        let budget = MemoryBudget::new(16 << 20);
        let mut workspace = Workspace::new(31, &budget)?;
        workspace.compute(&mut large, [0, 1], [1.; 3], Accuracy::default(), &cancel)?;
        budget.used()
    };
    let budget = MemoryBudget::new(limit);
    let mut workspace = Workspace::new(31, &budget)?;
    workspace.compute(&mut large, [0, 1], [1.; 3], Accuracy::default(), &cancel)?;
    assert_eq!(budget.used(), limit);
    // Retaining the old floating cache while allocating the new integer cache
    // would exceed this exact budget even though the new paired workspace fits.
    workspace.compute(&mut smaller, [0, 1], [1.; 3], Accuracy::default(), &cancel)?;
    assert!(budget.used() < limit);
    workspace.compute(&mut large, [0, 1], [1.; 3], Accuracy::default(), &cancel)?;
    assert_eq!(budget.used(), limit);
    let mut constant = Local::new([8, 8, 2], [1, 1, 1], [1, 1, 1], false);
    assert_eq!(
        workspace
            .compute(&mut constant, [0, 1], [1.; 3], Accuracy::default(), &cancel)?
            .lag([0; 3])?
            .method,
        Method::Analytic
    );
    assert_eq!(budget.used(), 31 * 96);
    let too_large = Local::new([128, 32, 4], [0; 3], [128, 32, 4], true);
    let mut too_large = too_large;
    assert!(matches!(
        workspace.compute(
            &mut too_large,
            [0, 1],
            [1.; 3],
            Accuracy::default(),
            &cancel
        ),
        Err(Error::MemoryBudget { .. })
    ));
    workspace.compute(&mut smaller, [0, 1], [1.; 3], Accuracy::default(), &cancel)?;
    drop(workspace);
    assert_eq!(budget.used(), 0);
    Ok(())
}
