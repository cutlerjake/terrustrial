//! Streaming grid persistence and external-sort ingestion.
mod file;
pub mod ingest;
use crate::{
    error::invalid,
    execution::{Cancellation, MemoryBudget},
    grid::{Layout, Source},
    Error, Result,
};
pub use file::FileSource;
use std::io::Write;
const MAGIC: &[u8; 8] = b"TRGRID01";
const HEADER: u64 = 40;
const RECORD: u64 = 9;
const BUFFER: usize = 64 * 1024;

fn file_size(layout: Layout, variables: usize) -> Result<u64> {
    if variables == 0 {
        return Err(invalid("positive grid variable count required"));
    }
    u64::try_from(variables)
        .ok()
        .and_then(|v| v.checked_mul(layout.cells()))
        .and_then(|v| v.checked_mul(RECORD))
        .and_then(|v| v.checked_add(HEADER))
        .ok_or(Error::SizeOverflow)
}
/// Stream columns into TRGRID01. Failure may leave a partial destination: callers
/// needing atomic publication should write a new temporary file and rename it
/// only after success. Input must remain immutable throughout the operation.
pub fn write(
    source: &mut impl Source,
    mut output: impl Write,
    tile_cells: usize,
    budget: &MemoryBudget,
    cancel: &Cancellation,
) -> Result<()> {
    cancel.check()?;
    if tile_cells == 0 {
        return Err(invalid("positive grid tile size required"));
    }
    let layout = source.layout();
    let variables = source.variables();
    file_size(layout, variables)?;
    let capacity = tile_cells.min(usize::try_from(layout.cells()).unwrap_or(usize::MAX));
    let bytes = capacity
        .checked_mul(std::mem::size_of::<Option<f64>>() + RECORD as usize)
        .ok_or(Error::SizeOverflow)?;
    let _reservation = budget.reserve(u64::try_from(bytes).map_err(|_| Error::SizeOverflow)?)?;
    let mut values = vec![None; capacity];
    let mut buffer = vec![0_u8; capacity * RECORD as usize];
    output.write_all(MAGIC)?;
    for value in layout.dimensions().into_iter().chain([variables as u64]) {
        output.write_all(&value.to_le_bytes())?;
    }
    for variable in 0..variables {
        let mut first = 0;
        while first < layout.cells() {
            cancel.check()?;
            let n = (layout.cells() - first).min(capacity as u64) as usize;
            source.read(variable, first, &mut values[..n])?;
            for (value, bytes) in values[..n]
                .iter()
                .zip(buffer.as_chunks_mut::<{ RECORD as usize }>().0)
            {
                let (valid, value) = match value {
                    Some(v) if v.is_finite() => (1, *v),
                    None => (0, 0.0),
                    _ => return Err(invalid("nonfinite grid source value")),
                };
                bytes[0] = valid;
                bytes[1..].copy_from_slice(&value.to_le_bytes());
            }
            output.write_all(&buffer[..n * RECORD as usize])?;
            first += n as u64;
        }
    }
    output.flush()?;
    Ok(())
}
