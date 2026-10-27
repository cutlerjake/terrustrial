use std::{
    fs::File,
    io::{Seek, SeekFrom, Write},
    sync::Barrier,
};
use terrustrial_core::{
    execution::{Cancellation, MemoryBudget},
    grid::{Layout, Source},
    storage::grid::FileSource,
    Result,
};

struct Generated {
    cells: u64,
    variables: usize,
}
fn value(variable: usize, index: u64) -> Option<f64> {
    match (index + variable as u64) % 17 {
        0 => None,
        1 => Some(-0.0),
        2 => Some(f64::MAX),
        _ => Some((variable as u64 * 1_000_000 + index) as f64),
    }
}
impl Source for Generated {
    fn layout(&self) -> Layout {
        Layout::new([self.cells, 1, 1]).unwrap()
    }
    fn variables(&self) -> usize {
        self.variables
    }
    fn read(&mut self, variable: usize, first: u64, output: &mut [Option<f64>]) -> Result<()> {
        for (i, out) in output.iter_mut().enumerate() {
            *out = value(variable, first + i as u64);
        }
        Ok(())
    }
}
fn fixture(cells: u64, variables: usize) -> Result<File> {
    let mut file = tempfile::tempfile()?;
    terrustrial_core::storage::grid::write(
        &mut Generated { cells, variables },
        &mut file,
        1024,
        &MemoryBudget::new(1 << 20),
        &Cancellation::default(),
    )?;
    Ok(file)
}
fn check(output: &[Option<f64>], variable: usize, first: u64) {
    for (i, out) in output.iter().enumerate() {
        assert_eq!(
            out.map(f64::to_bits),
            value(variable, first + i as u64).map(f64::to_bits),
            "variable {variable}, cell {}",
            first + i as u64
        );
    }
}

#[test]
fn cloned_handles_read_independent_columns_concurrently() -> Result<()> {
    let file = fixture(32_768, 8)?;
    let budget = MemoryBudget::new(8 * 65536);
    let sources: Vec<_> = (0..8)
        .map(|_| FileSource::open(file.try_clone().unwrap(), &budget).unwrap())
        .collect();
    let barrier = Barrier::new(sources.len());
    std::thread::scope(|scope| {
        for (worker, mut source) in sources.into_iter().enumerate() {
            let barrier = &barrier;
            scope.spawn(move || {
                let mut output = [None; 127];
                barrier.wait();
                for iteration in 0..2000 {
                    let variable = (worker + iteration) % 8;
                    let first = ((worker * 3907 + iteration * 7919) % (32768 - 127)) as u64;
                    source.read(variable, first, &mut output).unwrap();
                    check(&output, variable, first);
                }
            });
        }
    });
    Ok(())
}

#[test]
fn cloned_handles_open_concurrently_and_stream_large_reads() -> Result<()> {
    let file = fixture(20_003, 8)?;
    let budget = MemoryBudget::new(8 * 65536);
    let barrier = Barrier::new(8);
    std::thread::scope(|scope| {
        for worker in 0..8 {
            let (file, budget, barrier) = (&file, &budget, &barrier);
            scope.spawn(move || {
                let mut output = vec![None; 10_000];
                barrier.wait();
                for iteration in 0..32 {
                    let mut source = FileSource::open(file.try_clone().unwrap(), budget).unwrap();
                    let first = ((iteration * 743 + worker * 29) % 10004) as u64;
                    source.read(worker, first, &mut output).unwrap();
                    check(&output, worker, first);
                }
            });
        }
    });
    assert_eq!(budget.used(), 0);
    Ok(())
}

#[test]
fn random_boundaries_column_tails_and_bulk_bypass_preserve_bits_without_allocations() -> Result<()>
{
    let file = fixture(20_003, 32)?;
    let budget = MemoryBudget::new(65536);
    let mut source = FileSource::open(file, &budget)?;
    assert_eq!(budget.used(), 65536);
    let mut output = vec![None; 20_003];
    let lengths = [0, 1, 2, 1819, 1820, 1821, 7279, 7281, 7282, 10_000, 20_003];
    let counted = allocation_counter::measure(|| {
        for iteration in 0..500 {
            let variable = iteration * 7 % 32;
            let n = lengths[iteration % lengths.len()];
            let first = if iteration % 3 == 0 {
                20_003 - n
            } else {
                iteration * 7919 % (20_004 - n)
            } as u64;
            source.read(variable, first, &mut output[..n]).unwrap();
            check(&output[..n], variable, first);
        }
    });
    assert_eq!(counted.count_total, 0);
    assert_eq!(counted.bytes_total, 0);
    assert!(source.read(32, 0, &mut []).is_err());
    assert!(source.read(0, 20_004, &mut []).is_err());
    assert!(source.read(0, 20_003, &mut output[..1]).is_err());
    drop(source);
    assert_eq!(budget.used(), 0);
    assert!(FileSource::open(fixture(1, 1)?, &MemoryBudget::new(65535)).is_err());
    Ok(())
}

#[test]
fn prefetched_corruption_is_lazy_and_explicit_invalidation_observes_edits() -> Result<()> {
    let mut file = fixture(4001, 2)?;
    // Both corrupt records share a prefetched page with valid requested values.
    file.seek(SeekFrom::Start(40 + 150 * 9))?;
    file.write_all(&[2])?;
    file.seek(SeekFrom::Start(40 + 151 * 9))?;
    file.write_all(&[1])?;
    file.write_all(&f64::NAN.to_le_bytes())?;
    let budget = MemoryBudget::new(65536);
    let mut source = FileSource::open(file.try_clone()?, &budget)?;
    let mut output = [None; 10];
    source.read(0, 0, &mut output)?;
    check(&output, 0, 0);
    for first in [150, 151] {
        assert!(source.read(0, first, &mut output[..1]).is_err());
        // Cache hits must repeat the payload validation, including after errors.
        assert!(source.read(0, first, &mut output[..1]).is_err());
    }
    source.read(0, 152, &mut output)?;
    check(&output, 0, 152);
    file.seek(SeekFrom::Start(40 + 150 * 9))?;
    for replacement in [-0.0_f64, f64::MAX] {
        file.write_all(&[1])?;
        file.write_all(&replacement.to_le_bytes())?;
    }
    source.invalidate_cache();
    source.read(0, 150, &mut output[..2])?;
    assert_eq!(output[0].unwrap().to_bits(), (-0.0_f64).to_bits());
    assert_eq!(output[1], Some(f64::MAX));
    // A failed page fill must not poison later valid reads.
    file.set_len(40 + 20 * 9)?;
    source.invalidate_cache();
    assert!(source.read(1, 3900, &mut output).is_err());
    file.seek(SeekFrom::Start(0))?;
    terrustrial_core::storage::grid::write(
        &mut Generated {
            cells: 4001,
            variables: 2,
        },
        &mut file,
        1024,
        &MemoryBudget::new(1 << 20),
        &Cancellation::default(),
    )?;
    // File contents and layout are restored exactly. Reopen after length changes.
    drop(source);
    let mut source = FileSource::open(file, &budget)?;
    source.read(1, 3900, &mut output)?;
    check(&output, 1, 3900);
    Ok(())
}
