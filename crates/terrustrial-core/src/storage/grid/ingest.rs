//! Coordinate-keyed grid ingestion by bounded external merge sort. No global
//! cell bitmap, row index, or list of run paths is retained in memory.
use super::{file_size, Layout, HEADER, MAGIC, RECORD};
use crate::{
    error::{checked_len, invalid},
    execution::{Cancellation, MemoryBudget},
    Error, Result,
};
use std::{
    cmp::Reverse,
    collections::BinaryHeap,
    fs::{self, File},
    io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
};

/// Supplies one complete attribute row and its x-fast global cell index.
/// Fill EVERY value slot before returning Some(index); None marks end of input.
/// Coordinates can be validated/flattened with `Layout::index`. Values may be
/// missing, but present values must be finite. No ordering is required.
pub trait Rows {
    fn next(&mut self, values: &mut [Option<f64>]) -> Result<Option<u64>>;
}

pub struct Options<'a> {
    pub layout: Layout,
    pub variables: usize,
    pub rows_per_run: usize,
    /// At least two. Bounds simultaneously open input files and read buffers.
    pub merge_fan_in: usize,
    pub io_buffer_bytes: usize,
    /// Logical file lengths, including intermediate runs and final output.
    /// Not a quota on filesystem metadata, allocation blocks, or unrelated files.
    pub disk_limit_bytes: u64,
    pub budget: &'a MemoryBudget,
    pub cancellation: &'a Cancellation,
}
#[derive(Clone, Debug)]
pub struct Report {
    pub rows: u64,
    pub initial_runs: u64,
    pub merge_passes: u64,
    pub output_bytes: u64,
    pub peak_accounted_disk_bytes: u64,
    pub total_logical_bytes_written: u64,
    pub maximum_working_reservation_bytes: u64,
}
#[derive(Clone, Copy)]
struct Entry {
    cell: u64,
    row: usize,
}
struct Disk {
    limit: u64,
    used: u64,
    peak: u64,
    written: u64,
}
impl Disk {
    fn reserve(&mut self, bytes: u64) -> Result<()> {
        let used = self.used.checked_add(bytes).ok_or(Error::SizeOverflow)?;
        if used > self.limit {
            return Err(Error::DiskBudget {
                requested: bytes,
                available: self.limit - self.used,
            });
        }
        self.used = used;
        self.peak = self.peak.max(used);
        self.written = self.written.checked_add(bytes).ok_or(Error::SizeOverflow)?;
        Ok(())
    }
    fn release(&mut self, bytes: u64) {
        self.used -= bytes;
    }
}
fn run_path(directory: &Path, pass: u64, index: u64) -> PathBuf {
    directory.join(format!("run-{pass}-{index}"))
}
fn reservation(bytes: usize) -> Result<u64> {
    u64::try_from(bytes).map_err(|_| Error::SizeOverflow)
}

/// Import exactly one row for EVERY cell. A missing attribute is represented by
/// None, not by omitting the row. Duplicates, gaps, invalid values, I/O errors,
/// resource exhaustion and cancellation fail without publishing a destination.
///
/// The destination must be new. Sorted runs and the staged output live in a
/// private tempfile directory beside it. Final data/header are synchronized,
/// then persist_noclobber publishes without replacing an existing target. This
/// is not a cross-platform crash-durability or directory-fsync guarantee. Scratch
/// cleanup is best-effort on ordinary return; process termination can leave files.
/// Memory is bounded by rows_per_run and merge_fan_in, independent of row count.
pub fn import(rows: &mut impl Rows, destination: &Path, options: Options<'_>) -> Result<Report> {
    options.cancellation.check()?;
    let output_bytes = file_size(options.layout, options.variables)?;
    if output_bytes > options.disk_limit_bytes {
        return Err(Error::DiskBudget {
            requested: output_bytes,
            available: options.disk_limit_bytes,
        });
    }
    if options.rows_per_run == 0
        || options.merge_fan_in < 2
        || options.io_buffer_bytes == 0
        || destination.file_name().is_none()
    {
        return Err(invalid("grid ingestion configuration or destination"));
    }
    let capacity = options
        .rows_per_run
        .min(usize::try_from(options.layout.cells()).unwrap_or(usize::MAX));
    let row_bytes = checked_len(&[options.variables, RECORD as usize])?
        .checked_add(8)
        .ok_or(Error::SizeOverflow)?;
    let capacity_values = checked_len(&[capacity, options.variables])?;
    let sort_bytes = checked_len(&[capacity, std::mem::size_of::<Entry>()])?
        .checked_add(checked_len(&[
            capacity_values,
            std::mem::size_of::<Option<f64>>(),
        ])?)
        .and_then(|v| v.checked_add(row_bytes))
        .and_then(|v| v.checked_add(options.io_buffer_bytes))
        .ok_or(Error::SizeOverflow)?;
    // Fail before filesystem mutation for an impossible minimum phase.
    let first_reservation = options.budget.reserve(reservation(sort_bytes)?)?;
    if destination.try_exists()? {
        return Err(invalid("grid destination already exists"));
    }
    let parent = destination
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let scratch = tempfile::Builder::new()
        .prefix(".terrustrial-grid-")
        .tempdir_in(parent)?;
    let mut disk = Disk {
        limit: options.disk_limit_bytes,
        used: 0,
        peak: 0,
        written: 0,
    };
    let mut count = 0_u64;
    let mut runs = 0_u64;
    {
        let mut entries = Vec::with_capacity(capacity);
        let mut values = Vec::with_capacity(capacity_values);
        let mut encoded = vec![0; row_bytes];
        let mut done = false;
        while !done {
            entries.clear();
            values.clear();
            for row in 0..capacity {
                options.cancellation.check()?;
                let start = values.len();
                values.resize(start + options.variables, None);
                let Some(cell) = rows.next(&mut values[start..])? else {
                    values.truncate(start);
                    done = true;
                    break;
                };
                if cell >= options.layout.cells()
                    || values[start..].iter().flatten().any(|v| !v.is_finite())
                {
                    return Err(invalid("grid row index or nonfinite value"));
                }
                count = count.checked_add(1).ok_or(Error::SizeOverflow)?;
                if count > options.layout.cells() {
                    return Err(invalid("more rows than grid cells"));
                }
                entries.push(Entry { cell, row });
            }
            if entries.is_empty() {
                break;
            }
            entries.sort_unstable_by_key(|entry| entry.cell);
            if entries.windows(2).any(|r| r[0].cell == r[1].cell) {
                return Err(invalid("duplicate grid cell within an input batch"));
            }
            let bytes = (entries.len() as u64)
                .checked_mul(row_bytes as u64)
                .ok_or(Error::SizeOverflow)?;
            disk.reserve(bytes)?;
            let mut writer = BufWriter::with_capacity(
                options.io_buffer_bytes,
                File::create_new(run_path(scratch.path(), 0, runs))?,
            );
            for entry in &entries {
                options.cancellation.check()?;
                encoded[..8].copy_from_slice(&entry.cell.to_le_bytes());
                let row = &values[entry.row * options.variables..][..options.variables];
                for (value, bytes) in row.iter().zip(encoded[8..].as_chunks_mut::<9>().0) {
                    bytes[0] = u8::from(value.is_some());
                    bytes[1..].copy_from_slice(&value.unwrap_or(0.0).to_le_bytes());
                }
                writer.write_all(&encoded)?;
            }
            writer.flush()?;
            runs = runs.checked_add(1).ok_or(Error::SizeOverflow)?;
        }
    }
    drop(first_reservation);
    if count != options.layout.cells() {
        return Err(invalid(format!(
            "incomplete grid: received {count} rows for {} cells",
            options.layout.cells()
        )));
    }
    let initial_runs = runs;
    let mut maximum_working = reservation(sort_bytes)?;
    let mut pass = 0_u64;
    while runs > 1 {
        options.cancellation.check()?;
        let mut first = 0;
        let mut next_runs = 0;
        while first < runs {
            let end = first.saturating_add(options.merge_fan_in as u64).min(runs);
            let working = merge(
                scratch.path(),
                pass,
                first..end,
                next_runs,
                row_bytes,
                &options,
                &mut disk,
            )?;
            maximum_working = maximum_working.max(working);
            first = end;
            next_runs += 1;
        }
        runs = next_runs;
        pass = pass.checked_add(1).ok_or(Error::SizeOverflow)?;
    }
    let source = run_path(scratch.path(), pass, 0);
    let transpose_bytes = checked_len(&[capacity, row_bytes])?
        .checked_add(checked_len(&[capacity, RECORD as usize])?)
        .ok_or(Error::SizeOverflow)?;
    let _reservation = options.budget.reserve(reservation(transpose_bytes)?)?;
    maximum_working = maximum_working.max(reservation(transpose_bytes)?);
    disk.reserve(output_bytes)?;
    let mut staged = tempfile::Builder::new()
        .prefix("complete-")
        .tempfile_in(scratch.path())?;
    staged.as_file_mut().set_len(output_bytes)?;
    {
        let mut input = File::open(&source)?;
        if input.metadata()?.len()
            != count
                .checked_mul(row_bytes as u64)
                .ok_or(Error::SizeOverflow)?
        {
            return Err(invalid("merged grid row count does not match input"));
        }
        let mut records = vec![0; capacity * row_bytes];
        let mut column = vec![0; capacity * RECORD as usize];
        let mut first = 0_u64;
        while first < count {
            options.cancellation.check()?;
            let n = (count - first).min(capacity as u64) as usize;
            input.read_exact(&mut records[..n * row_bytes])?;
            for (i, record) in records[..n * row_bytes].chunks_exact(row_bytes).enumerate() {
                let cell = u64::from_le_bytes(record[..8].try_into().unwrap());
                if cell != first + i as u64 {
                    return Err(invalid("duplicate or missing grid cell in merged input"));
                }
                for value in record[8..].as_chunks::<9>().0 {
                    if value[0] > 1
                        || (value[0] == 1
                            && !f64::from_le_bytes(value[1..].try_into().unwrap()).is_finite())
                    {
                        return Err(invalid("invalid merged grid payload"));
                    }
                }
            }
            for variable in 0..options.variables {
                options.cancellation.check()?;
                for (record, out) in records[..n * row_bytes]
                    .chunks_exact(row_bytes)
                    .zip(column.as_chunks_mut::<9>().0)
                {
                    out.copy_from_slice(&record[8 + variable * 9..][..9]);
                }
                staged.as_file_mut().seek(SeekFrom::Start(
                    HEADER + ((variable as u64) * count + first) * RECORD,
                ))?;
                staged.write_all(&column[..n * 9])?;
            }
            first += n as u64;
        }
    }
    options.cancellation.check()?;
    staged.as_file_mut().seek(SeekFrom::Start(0))?;
    staged.write_all(MAGIC)?;
    for field in options
        .layout
        .dimensions()
        .into_iter()
        .chain([options.variables as u64])
    {
        staged.write_all(&field.to_le_bytes())?;
    }
    staged.as_file().sync_all()?;
    options.cancellation.check()?;
    // Noclobber also handles another writer creating the destination after the
    // early existence check. On failure the staged file remains RAII-owned.
    staged
        .persist_noclobber(destination)
        .map_err(|e| Error::Io(e.error))?;
    Ok(Report {
        rows: count,
        initial_runs,
        merge_passes: pass,
        output_bytes,
        peak_accounted_disk_bytes: disk.peak,
        total_logical_bytes_written: disk.written,
        maximum_working_reservation_bytes: maximum_working,
    })
}

struct Cursor {
    reader: BufReader<File>,
    record: Vec<u8>,
    remaining: u64,
    previous: Option<u64>,
}
impl Cursor {
    fn advance(&mut self) -> Result<Option<u64>> {
        if self.remaining == 0 {
            return Ok(None);
        }
        self.reader.read_exact(&mut self.record)?;
        let key = u64::from_le_bytes(self.record[..8].try_into().unwrap());
        if self.previous.is_some_and(|old| old >= key) {
            return Err(invalid("unsorted or duplicate intermediate grid run"));
        }
        self.previous = Some(key);
        self.remaining -= 1;
        Ok(Some(key))
    }
}
fn merge(
    directory: &Path,
    pass: u64,
    range: std::ops::Range<u64>,
    output: u64,
    row_bytes: usize,
    options: &Options<'_>,
    disk: &mut Disk,
) -> Result<u64> {
    let count = usize::try_from(range.end - range.start).map_err(|_| Error::SizeOverflow)?;
    let per_cursor = options
        .io_buffer_bytes
        .checked_add(row_bytes)
        .and_then(|v| v.checked_add(192))
        .ok_or(Error::SizeOverflow)?;
    let bytes = checked_len(&[count, per_cursor])?
        .checked_add(options.io_buffer_bytes)
        .ok_or(Error::SizeOverflow)?;
    let _reservation = options.budget.reserve(reservation(bytes)?)?;
    let mut cursors = Vec::with_capacity(count);
    let mut heap = BinaryHeap::with_capacity(count);
    let mut total = 0_u64;
    for index in range.clone() {
        options.cancellation.check()?;
        let file = File::open(run_path(directory, pass, index))?;
        let length = file.metadata()?.len();
        if length == 0 || length % (row_bytes as u64) != 0 {
            return Err(invalid("invalid intermediate grid run length"));
        }
        total = total.checked_add(length).ok_or(Error::SizeOverflow)?;
        let mut cursor = Cursor {
            reader: BufReader::with_capacity(options.io_buffer_bytes, file),
            record: vec![0; row_bytes],
            remaining: length / row_bytes as u64,
            previous: None,
        };
        let key = cursor.advance()?.unwrap();
        heap.push(Reverse((key, cursors.len())));
        cursors.push(cursor);
    }
    disk.reserve(total)?;
    let mut writer = BufWriter::with_capacity(
        options.io_buffer_bytes,
        File::create_new(run_path(directory, pass + 1, output))?,
    );
    let mut previous = None;
    while let Some(Reverse((key, index))) = heap.pop() {
        options.cancellation.check()?;
        if previous.is_some_and(|old| old >= key) {
            return Err(invalid("duplicate grid cell across input batches"));
        }
        previous = Some(key);
        writer.write_all(&cursors[index].record)?;
        if let Some(key) = cursors[index].advance()? {
            heap.push(Reverse((key, index)));
        }
    }
    writer.flush()?;
    drop(writer);
    drop(cursors); // Windows requires readers to close before deletion.
    for index in range {
        fs::remove_file(run_path(directory, pass, index))?;
    }
    disk.release(total);
    reservation(bytes)
}
