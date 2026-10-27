//! Portable binary point persistence and a bounded-memory scan reference.
use crate::{
    data::{Observation, Point3, Site},
    error::invalid,
    execution::Cancellation,
    spatial::ObservationSource,
    Error, Result,
};
use std::{
    fs::{File, OpenOptions},
    io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
};

/// Portable, append-only point-observation format: 8-byte version then 56-byte records.
/// Queries are O(N log k) scans. This is a bounded-memory oracle, not the production
/// out-of-core spatial index. Blocks are rejected rather than losing support semantics.
pub struct PointFile {
    path: PathBuf,
}
const MAGIC: &[u8; 8] = b"TRPOINT1";
impl PointFile {
    pub fn create(
        path: impl AsRef<Path>,
        source: &(impl ObservationSource + ?Sized),
        cancel: &Cancellation,
    ) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)?;
        let result = (|| {
            let mut output = BufWriter::new(file);
            output.write_all(MAGIC)?;
            source.scan(&mut |observation| {
                cancel.check()?;
                write_observation(&mut output, observation)
            })?;
            output.flush()?;
            output.get_ref().sync_all()?;
            Ok(())
        })();
        if let Err(error) = result {
            let _ = std::fs::remove_file(&path);
            return Err(error);
        }
        Ok(Self { path })
    }
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let out = Self {
            path: path.as_ref().to_path_buf(),
        };
        let mut file = File::open(&out.path)?;
        let bytes = file.metadata()?.len();
        let mut magic = [0; 8];
        file.read_exact(&mut magic)?;
        if &magic != MAGIC || (bytes - 8) % 56 != 0 {
            return Err(invalid("unsupported or truncated point file"));
        }
        Ok(out)
    }
    pub fn path(&self) -> &Path {
        &self.path
    }
}
impl ObservationSource for PointFile {
    fn scan(&self, visitor: &mut dyn FnMut(&Observation) -> Result<()>) -> Result<()> {
        let mut input = BufReader::new(File::open(&self.path)?);
        let mut magic = [0; 8];
        input.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err(invalid("point-file version"));
        }
        let mut bytes = [0; 56];
        loop {
            if input.read(&mut bytes[..1])? == 0 {
                break;
            }
            input.read_exact(&mut bytes[1..])?;
            let obs = decode_observation(&bytes)?;
            visitor(&obs)?;
        }
        Ok(())
    }
}
fn write_observation(output: &mut impl Write, observation: &Observation) -> Result<()> {
    // A block with one quadrature point is still block support; only point geometry
    // is accepted through the explicit point predicate.
    if !observation.site().support.is_point() {
        return Err(invalid("point file cannot encode block support"));
    }
    let p = observation.site().support.center().coordinates();
    let words = [
        observation.id(),
        u64::try_from(observation.site().variable).map_err(|_| Error::SizeOverflow)?,
        p[0].to_bits(),
        p[1].to_bits(),
        p[2].to_bits(),
        observation.value().to_bits(),
        observation.noise_variance().to_bits(),
    ];
    for word in words {
        output.write_all(&word.to_le_bytes())?;
    }
    Ok(())
}

fn decode_observation(bytes: &[u8; 56]) -> Result<Observation> {
    let word = |i: usize| u64::from_le_bytes(bytes[i * 8..i * 8 + 8].try_into().unwrap());
    Observation::new(
        word(0),
        Site::point(
            Point3::new([
                f64::from_bits(word(2)),
                f64::from_bits(word(3)),
                f64::from_bits(word(4)),
            ])?,
            usize::try_from(word(1)).map_err(|_| Error::SizeOverflow)?,
        ),
        f64::from_bits(word(5)),
        f64::from_bits(word(6)),
    )
}
/// Independent seek cursor for bounded-memory random sampling. Record indices
/// refer to file order, not observation IDs. Do not modify the file while reading.
pub struct PointReader {
    reader: BufReader<File>,
    count: u64,
}
impl PointFile {
    pub fn reader(&self) -> Result<PointReader> {
        Self::open(&self.path)?;
        let file = File::open(&self.path)?;
        let count = (file.metadata()?.len() - 8) / 56;
        Ok(PointReader {
            reader: BufReader::new(file),
            count,
        })
    }
}
impl PointReader {
    pub fn len(&self) -> u64 {
        self.count
    }
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }
    pub fn get(&mut self, index: u64) -> Result<Observation> {
        if index >= self.count {
            return Err(invalid("point-file record index out of bounds"));
        }
        let offset = index
            .checked_mul(56)
            .and_then(|x| x.checked_add(8))
            .ok_or(Error::SizeOverflow)?;
        self.reader.seek(SeekFrom::Start(offset))?;
        let mut bytes = [0; 56];
        self.reader.read_exact(&mut bytes)?;
        decode_observation(&bytes)
    }
}
