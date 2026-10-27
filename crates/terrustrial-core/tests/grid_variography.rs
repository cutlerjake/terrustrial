use std::{
    fs::{File, OpenOptions},
    io::{Cursor, Seek, SeekFrom, Write},
    path::PathBuf,
};
use terrustrial_core::{
    execution::{Cancellation, MemoryBudget},
    grid::{Grid, Layout, Shape, Source, Variable},
    variography::{
        self,
        grid::{experimental, Options},
    },
    Error, Result,
};

fn settings<'a>(
    offsets: &'a [[i64; 3]],
    variables: [usize; 2],
    tile_cells: usize,
    budget: &'a MemoryBudget,
    cancellation: &'a Cancellation,
) -> Options<'a> {
    Options {
        variables,
        offsets,
        tile_cells,
        spacing: [1.0; 3],
        budget,
        cancellation,
    }
}
fn fixture(budget: &MemoryBudget) -> Result<Grid> {
    let values = (0..60)
        .flat_map(|i| {
            let (x, y, z) = (i % 5, (i / 5) % 4, i / 20);
            [
                if i % 7 == 0 {
                    None
                } else {
                    Some((x + 3 * y + 2 * z) as f64)
                },
                if i % 11 == 0 {
                    None
                } else {
                    Some((2 * x) as f64 - (y + z) as f64)
                },
            ]
        })
        .collect();
    Grid::new(
        Shape::new([5, 4, 3])?,
        vec![Variable::Continuous { scale: 1.0 }; 2],
        values,
        budget,
    )
}

#[test]
fn all_bounded_lags_match_an_independent_pair_enumeration_and_tile_sizes() -> Result<()> {
    let budget = MemoryBudget::new(1 << 24);
    let cancel = Cancellation::default();
    let mut source = fixture(&budget)?;
    let offsets: Vec<_> = (-3..=3)
        .flat_map(|z| {
            (-4..=4).flat_map(move |y| {
                (-5..=5).filter_map(move |x| ([x, y, z] != [0; 3]).then_some([x, y, z]))
            })
        })
        .collect();
    let mut samples = vec![];
    for i in 0..60 {
        samples.push(variography::Sample {
            point: terrustrial_core::data::Point3::new(
                source.shape().coordinates(i)?.map(|v| v as f64),
            )?,
            values: source.values()[i * 2..i * 2 + 2].to_vec(),
        });
    }
    for variables in [[0, 0], [1, 1], [0, 1], [1, 0]] {
        let result = experimental(
            &mut source,
            settings(&offsets, variables, 1, &budget, &cancel),
        )?;
        let tiled = experimental(
            &mut source,
            settings(&offsets, variables, 13, &budget, &cancel),
        )?;
        for (lag, tile) in result.lags.iter().zip(&tiled.lags) {
            let mut pairs = 0;
            let mut sum = 0.0;
            for a in &samples {
                for b in &samples {
                    let delta = std::array::from_fn(|d| {
                        (b.point.coordinates()[d] - a.point.coordinates()[d]) as i64
                    });
                    if delta != lag.offset {
                        continue;
                    }
                    let [va, vb] = variables;
                    if let (Some(aa), Some(ab), Some(ba), Some(bb)) =
                        (a.values[va], a.values[vb], b.values[va], b.values[vb])
                    {
                        pairs += 1;
                        sum += 0.5 * (aa - ba) * (ab - bb);
                    }
                }
            }
            assert_eq!(lag.statistics.pairs, pairs);
            assert_eq!(lag.statistics.semivariance, tile.statistics.semivariance);
            if pairs > 0 {
                assert!((lag.statistics.semivariance.unwrap() - sum / pairs as f64).abs() < 1e-12);
            } else {
                assert!(lag.statistics.semivariance.is_none());
            }
        }
    }
    Ok(())
}

struct Temporary {
    path: PathBuf,
    _directory: tempfile::TempDir,
}
impl Temporary {
    fn new() -> Self {
        let directory = tempfile::tempdir().unwrap();
        Self {
            path: directory.path().join("grid.bin"),
            _directory: directory,
        }
    }
    fn create(&self) -> File {
        OpenOptions::new()
            .create_new(true)
            .write(true)
            .read(true)
            .open(&self.path)
            .unwrap()
    }
}

#[test]
fn column_file_roundtrips_missing_signed_zero_and_matches_in_memory_variography() -> Result<()> {
    let budget = MemoryBudget::new(1 << 24);
    let cancel = Cancellation::default();
    let mut source = fixture(&budget)?;
    let temp = Temporary::new();
    terrustrial_core::storage::grid::write(&mut source, temp.create(), 7, &budget, &cancel)?;
    let used = budget.used();
    {
        let mut disk =
            terrustrial_core::storage::grid::FileSource::open(File::open(&temp.path)?, &budget)?;
        assert_eq!(disk.layout(), source.layout());
        assert_eq!(Source::variables(&disk), 2);
        let offsets = [[1, 0, 0], [0, -2, 0], [-1, 1, 1], [i64::MIN, 0, 0]];
        let a = experimental(&mut source, settings(&offsets, [0, 1], 3, &budget, &cancel))?;
        let b = experimental(&mut disk, settings(&offsets, [0, 1], 17, &budget, &cancel))?;
        for (a, b) in a.lags.iter().zip(&b.lags) {
            assert_eq!(a.statistics.pairs, b.statistics.pairs);
            assert_eq!(a.statistics.semivariance, b.statistics.semivariance);
        }
        assert!(disk.read(2, 0, &mut [None]).is_err());
        assert!(disk.read(0, 60, &mut [None]).is_err());
        disk.read(1, 60, &mut [])?;
    }
    assert_eq!(budget.used(), used);
    let mut special = Grid::new(
        Shape::new([3, 1, 1])?,
        vec![Variable::Continuous { scale: 1.0 }],
        vec![Some(-0.0), None, Some(2.0)],
        &budget,
    )?;
    let other = Temporary::new();
    terrustrial_core::storage::grid::write(&mut special, other.create(), 100, &budget, &cancel)?;
    let mut disk =
        terrustrial_core::storage::grid::FileSource::open(File::open(&other.path)?, &budget)?;
    let mut values = [None; 3];
    disk.read(0, 0, &mut values)?;
    assert_eq!(values[0].unwrap().to_bits(), (-0.0_f64).to_bits());
    assert_eq!(values[1], None);
    assert_eq!(values[2], Some(2.0));
    Ok(())
}

#[test]
fn corrupt_truncated_and_oversized_grid_files_fail_explicitly() -> Result<()> {
    let budget = MemoryBudget::new(1 << 24);
    let cancel = Cancellation::default();
    let mut source = fixture(&budget)?;
    let mut bytes = vec![];
    terrustrial_core::storage::grid::write(&mut source, &mut bytes, 8, &budget, &cancel)?;
    for mutate in 0..5 {
        let mut corrupt = bytes.clone();
        match mutate {
            0 => {
                corrupt.pop();
            }
            1 => corrupt[0] = 0,
            2 => corrupt[8..16].fill(0),
            3 => corrupt.extend([0]),
            4 => corrupt[8..16].copy_from_slice(&u64::MAX.to_le_bytes()),
            _ => unreachable!(),
        }
        let temp = Temporary::new();
        temp.create().write_all(&corrupt)?;
        assert!(terrustrial_core::storage::grid::FileSource::open(
            File::open(&temp.path)?,
            &budget
        )
        .is_err());
    }
    for mutate in 0..2 {
        let temp = Temporary::new();
        let mut file = temp.create();
        file.write_all(&bytes)?;
        file.seek(SeekFrom::Start(40))?;
        if mutate == 0 {
            file.write_all(&[2])?;
        } else {
            file.write_all(&[1])?;
            file.write_all(&f64::NAN.to_le_bytes())?;
        }
        let mut source = terrustrial_core::storage::grid::FileSource::open(file, &budget)?;
        assert!(source.read(0, 0, &mut [None]).is_err());
    }
    Ok(())
}

struct Generated {
    layout: Layout,
    maximum_read: usize,
    reads: usize,
    cancel: Cancellation,
}

#[test]
fn scaled_accumulation_handles_finite_extreme_terms_and_exact_cancellation() -> Result<()> {
    let budget = MemoryBudget::new(1 << 20);
    let cancel = Cancellation::default();
    let mut source = Grid::new(
        Shape::new([3, 1, 1])?,
        vec![Variable::Continuous { scale: 1.0 }; 2],
        vec![
            Some(0.0),
            Some(-f64::MAX),
            Some(2.0),
            Some(0.0),
            Some(4.0),
            Some(-f64::MAX),
        ],
        &budget,
    )?;
    // The two cross increments are +MAX and -MAX. Their mean is exactly zero;
    // neither overflowing the raw sum nor subtracting extreme running means
    // is necessary. Each half product itself remains representable.
    let result = experimental(
        &mut source,
        settings(&[[1, 0, 0]], [0, 1], 2, &budget, &cancel),
    )?;
    assert_eq!(result.lags[0].statistics.pairs, 2);
    assert_eq!(result.lags[0].statistics.semivariance, Some(0.0));
    Ok(())
}
impl Source for Generated {
    fn layout(&self) -> Layout {
        self.layout
    }
    fn variables(&self) -> usize {
        2
    }
    fn read(&mut self, variable: usize, first: u64, out: &mut [Option<f64>]) -> Result<()> {
        self.maximum_read = self.maximum_read.max(out.len());
        self.reads += 1;
        for (i, v) in out.iter_mut().enumerate() {
            *v = Some(if variable == 0 {
                (first + i as u64) as f64
            } else {
                -((first + i as u64) as f64)
            });
        }
        if self.reads == 8 {
            self.cancel.cancel();
        }
        Ok(())
    }
}
#[test]
fn logical_grids_larger_than_ram_use_bounded_reads_and_cancel_without_leaking() -> Result<()> {
    let budget = MemoryBudget::new(8192);
    let cancel = Cancellation::default();
    let mut source = Generated {
        layout: Layout::new([u64::MAX, 1, 1])?,
        maximum_read: 0,
        reads: 0,
        cancel: cancel.clone(),
    };
    assert!(matches!(
        experimental(
            &mut source,
            settings(&[[1, 0, 0]], [0, 1], 32, &budget, &cancel)
        ),
        Err(Error::Cancelled)
    ));
    assert_eq!(source.maximum_read, 32);
    assert_eq!(budget.used(), 0);
    let cancel = Cancellation::default();
    assert!(matches!(
        terrustrial_core::storage::grid::write(
            &mut source,
            Cursor::new(vec![]),
            32,
            &budget,
            &cancel
        ),
        Err(Error::SizeOverflow)
    ));
    assert!(Layout::new([u64::MAX, 2, 1]).is_err());
    Ok(())
}

#[test]
fn missingness_extreme_values_negative_cross_variograms_and_invalid_requests() -> Result<()> {
    let budget = MemoryBudget::new(1 << 24);
    let cancel = Cancellation::default();
    let mut source = Grid::new(
        Shape::new([2, 1, 1])?,
        vec![Variable::Continuous { scale: 1.0 }; 2],
        vec![Some(f64::MAX), Some(0.0), Some(-f64::MAX), Some(-0.5)],
        &budget,
    )?;
    let result = experimental(
        &mut source,
        settings(&[[1, 0, 0]], [0, 1], 1, &budget, &cancel),
    )?;
    assert_eq!(result.lags[0].statistics.semivariance, Some(f64::MAX / 2.0));
    assert!(matches!(
        experimental(
            &mut source,
            settings(&[[1, 0, 0]], [0, 0], 1, &budget, &cancel)
        ),
        Err(Error::Numerical { .. })
    ));
    for offsets in [vec![], vec![[0; 3]], vec![[1, 0, 0], [1, 0, 0]]] {
        assert!(
            experimental(&mut source, settings(&offsets, [0, 1], 1, &budget, &cancel)).is_err()
        );
    }
    for (variables, tile) in [([0, 2], 1), ([0, 1], 0), ([0, 1], usize::MAX)] {
        // Huge requested tile is clamped to source length and is therefore valid.
        let result = experimental(
            &mut source,
            settings(&[[1, 0, 0]], variables, tile, &budget, &cancel),
        );
        assert_eq!(result.is_ok(), tile == usize::MAX);
    }
    let tiny = MemoryBudget::new(1);
    assert!(matches!(
        experimental(
            &mut source,
            settings(&[[1, 0, 0]], [0, 1], 1, &tiny, &cancel)
        ),
        Err(Error::MemoryBudget { .. })
    ));
    assert_eq!(tiny.used(), 0);
    let mut negative = Grid::new(
        Shape::new([3, 1, 1])?,
        vec![Variable::Continuous { scale: 1.0 }; 2],
        vec![Some(0.0), Some(0.0), Some(1.0), Some(-1.0), Some(2.0), None],
        &budget,
    )?;
    let result = experimental(
        &mut negative,
        settings(&[[1, 0, 0]], [0, 1], 2, &budget, &cancel),
    )?;
    assert_eq!(result.lags[0].statistics.pairs, 1);
    assert_eq!(result.lags[0].statistics.semivariance, Some(-0.5));
    Ok(())
}
