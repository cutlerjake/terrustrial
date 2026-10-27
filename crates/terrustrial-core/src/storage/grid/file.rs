use super::{file_size, Layout, Source, BUFFER, HEADER, MAGIC, RECORD};
use crate::{
    error::invalid,
    execution::{MemoryBudget, Reservation},
    Error, Result,
};
use std::{fs::File, io};

// Four independently retained regions accommodate two variables at two lag
// endpoints. Page boundaries are record aligned, including across columns.
const SLOTS: usize = 4;
const PAGE_RECORDS: usize = BUFFER / SLOTS / RECORD as usize;
const PAGE_BYTES: usize = PAGE_RECORDS * RECORD as usize;

/// Versioned column-major file: 8-byte magic, three LE u64 dimensions and one
/// LE u64 variable count, then 9-byte records (validity byte, LE f64 payload).
/// Missing records have validity 0; validity 1 requires a finite f64. Other tags
/// are errors. Payloads are validated lazily when requested, including cached
/// payloads; length/header are validated at open.
///
/// Short reads share a bounded 64 KiB cache. Large reads stream through the same
/// buffer without retaining pages. No mmap or whole-input allocation is used.
/// The file must remain immutable during operations. After modifying payloads
/// between operations, call [`Self::invalidate_cache`] on every affected reader.
/// Reopen readers after changing the header or length.
///
/// Positioned reads permit concurrent readers created from `File::try_clone`
/// on Unix and Windows. Windows reads can move the shared cursor, so other users
/// of the handle must also use positioned IO. On other platforms, file-source
/// reads use a shared lock around seek/read; external users of cloned handles
/// must not access their cursor concurrently with file-source operations.
pub struct FileSource {
    file: File,
    layout: Layout,
    variables: usize,
    buffer: Vec<u8>,
    pages: [Option<u64>; SLOTS],
    recency: [usize; SLOTS],
    _reservation: Reservation,
}

impl FileSource {
    pub fn open(file: File, budget: &MemoryBudget) -> Result<Self> {
        let mut header = [0_u8; HEADER as usize];
        read_exact_at(&file, &mut header, 0)?;
        if &header[..8] != MAGIC {
            return Err(invalid("unrecognized grid file version"));
        }
        let fields: [u64; 4] = std::array::from_fn(|i| {
            u64::from_le_bytes(header[8 + i * 8..16 + i * 8].try_into().unwrap())
        });
        let layout = Layout::new(fields[..3].try_into().unwrap())?;
        let variables = usize::try_from(fields[3]).map_err(|_| Error::SizeOverflow)?;
        if file.metadata()?.len() != file_size(layout, variables)? {
            return Err(invalid("grid file length does not match header"));
        }
        let reservation = budget.reserve(BUFFER as u64)?;
        let mut buffer = Vec::new();
        buffer
            .try_reserve_exact(BUFFER)
            .map_err(|_| Error::Backend("grid read buffer allocation failed".into()))?;
        buffer.resize(BUFFER, 0);
        Ok(Self {
            file,
            layout,
            variables,
            buffer,
            pages: [None; SLOTS],
            recency: std::array::from_fn(|i| i),
            _reservation: reservation,
        })
    }

    /// Discard prefetched payloads after external edits between operations.
    /// This performs no IO or allocation and does not revalidate the header or
    /// file length. Concurrent file mutation during an operation is unsupported.
    pub fn invalidate_cache(&mut self) {
        self.pages.fill(None);
    }

    fn page(&mut self, page: u64) -> Result<usize> {
        let position = match self
            .recency
            .iter()
            .position(|&slot| self.pages[slot] == Some(page))
        {
            Some(position) => position,
            None => {
                let slot = self.recency[SLOTS - 1];
                // A short/failed read may have overwritten part of the victim.
                self.pages[slot] = None;
                let first = page * PAGE_RECORDS as u64;
                let records = (self.layout.cells() * self.variables as u64 - first)
                    .min(PAGE_RECORDS as u64) as usize;
                read_exact_at(
                    &self.file,
                    &mut self.buffer
                        [slot * PAGE_BYTES..slot * PAGE_BYTES + records * RECORD as usize],
                    HEADER + first * RECORD,
                )?;
                self.pages[slot] = Some(page);
                SLOTS - 1
            }
        };
        self.recency[..=position].rotate_right(1);
        Ok(self.recency[0])
    }
}

impl Source for FileSource {
    fn layout(&self) -> Layout {
        self.layout
    }
    fn variables(&self) -> usize {
        self.variables
    }
    fn read(&mut self, variable: usize, first: u64, mut output: &mut [Option<f64>]) -> Result<()> {
        self.layout.check_read(first, output.len())?;
        if variable >= self.variables {
            return Err(invalid("grid variable"));
        }
        let mut first = variable as u64 * self.layout.cells() + first;
        if output.len() >= BUFFER / RECORD as usize {
            self.invalidate_cache();
            for chunk in output.chunks_mut(BUFFER / RECORD as usize) {
                let bytes = &mut self.buffer[..chunk.len() * RECORD as usize];
                read_exact_at(&self.file, bytes, HEADER + first * RECORD)?;
                decode(bytes, chunk)?;
                first += chunk.len() as u64;
            }
        } else {
            while !output.is_empty() {
                let slot = self.page(first / PAGE_RECORDS as u64)?;
                let within = (first % PAGE_RECORDS as u64) as usize;
                let count = output.len().min(PAGE_RECORDS - within);
                let (chunk, rest) = output.split_at_mut(count);
                let start = slot * PAGE_BYTES + within * RECORD as usize;
                decode(&self.buffer[start..start + count * RECORD as usize], chunk)?;
                first += count as u64;
                output = rest;
            }
        }
        Ok(())
    }
}

fn decode(bytes: &[u8], output: &mut [Option<f64>]) -> Result<()> {
    for (out, bytes) in output
        .iter_mut()
        .zip(bytes.as_chunks::<{ RECORD as usize }>().0)
    {
        *out = match bytes[0] {
            0 => None,
            1 => {
                let value = f64::from_le_bytes(bytes[1..].try_into().unwrap());
                if !value.is_finite() {
                    return Err(invalid("nonfinite grid file value"));
                }
                Some(value)
            }
            _ => return Err(invalid("invalid grid validity byte")),
        };
    }
    Ok(())
}

#[cfg(unix)]
fn read_exact_at(file: &File, output: &mut [u8], offset: u64) -> io::Result<()> {
    use std::os::unix::fs::FileExt;
    file.read_exact_at(output, offset)
}
#[cfg(windows)]
fn read_exact_at(file: &File, output: &mut [u8], offset: u64) -> io::Result<()> {
    use std::os::windows::fs::FileExt;
    read_exact_with(output, offset, |bytes, at| file.seek_read(bytes, at))
}
#[cfg(not(any(unix, windows)))]
fn read_exact_at(mut file: &File, output: &mut [u8], offset: u64) -> io::Result<()> {
    use std::io::{Read, Seek, SeekFrom};
    static CURSOR: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _guard = CURSOR
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    file.seek(SeekFrom::Start(offset))?;
    file.read_exact(output)
}

#[cfg(any(windows, test))]
fn read_exact_with(
    mut output: &mut [u8],
    mut offset: u64,
    mut read: impl FnMut(&mut [u8], u64) -> io::Result<usize>,
) -> io::Result<()> {
    while !output.is_empty() {
        match read(output, offset) {
            Ok(0) => return Err(io::ErrorKind::UnexpectedEof.into()),
            Ok(n) => {
                output = &mut output[n..];
                offset = offset
                    .checked_add(n as u64)
                    .ok_or(io::ErrorKind::InvalidInput)?;
            }
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn failed_page_fill_invalidates_only_the_overwritten_victim() -> Result<()> {
        use std::io::Write;
        let mut file = tempfile::tempfile()?;
        file.write_all(MAGIC)?;
        for field in [10_000_u64, 1, 1, 1] {
            file.write_all(&field.to_le_bytes())?;
        }
        file.set_len(HEADER + 10_000 * RECORD)?;
        let mut source = FileSource::open(file.try_clone()?, &MemoryBudget::new(BUFFER as u64))?;
        for page in 0..SLOTS as u64 {
            source.page(page)?;
        }
        let victim = source.recency[SLOTS - 1];
        let before = source.pages;
        // Deliberately violate file immutability to inject a short physical read
        // after part of the victim buffer has been overwritten.
        file.set_len(HEADER + (SLOTS as u64 * PAGE_RECORDS as u64 + 7) * RECORD)?;
        assert!(source.page(SLOTS as u64).is_err());
        for (slot, previous) in before.iter().enumerate() {
            assert_eq!(
                source.pages[slot],
                if slot == victim { None } else { *previous }
            );
        }
        assert!(source.page(SLOTS as u64).is_err());
        Ok(())
    }

    #[test]
    fn exact_read_handles_partial_interrupted_eof_and_failed_reads() {
        let mut out = [0; 7];
        let mut calls = 0;
        read_exact_with(&mut out, 19, |bytes, offset| {
            calls += 1;
            if calls % 2 == 1 {
                return Err(io::ErrorKind::Interrupted.into());
            }
            let n = bytes.len().min(2);
            for (i, value) in bytes[..n].iter_mut().enumerate() {
                *value = (offset + i as u64) as u8;
            }
            Ok(n)
        })
        .unwrap();
        assert_eq!(out, [19, 20, 21, 22, 23, 24, 25]);
        assert_eq!(calls, 8);
        for kind in [
            io::ErrorKind::UnexpectedEof,
            io::ErrorKind::PermissionDenied,
        ] {
            let result = read_exact_with(&mut out, 0, |_, _| match kind {
                io::ErrorKind::UnexpectedEof => Ok(0),
                _ => Err(kind.into()),
            });
            assert_eq!(result.unwrap_err().kind(), kind);
        }
        read_exact_with(&mut [], 0, |_, _| panic!("empty read")).unwrap();
        assert_eq!(
            read_exact_with(&mut out, u64::MAX, |_, _| Ok(1))
                .unwrap_err()
                .kind(),
            io::ErrorKind::InvalidInput
        );
    }
}
