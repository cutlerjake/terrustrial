//! Bounded reads of immutable regular-grid columns. Global indices are u64;
//! reading a window does not require the whole grid to fit in addressable RAM.
use crate::{error::invalid, Error, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Layout {
    dimensions: [u64; 3],
    cells: u64,
}
impl Layout {
    pub fn new(dimensions: [u64; 3]) -> Result<Self> {
        if dimensions.contains(&0) {
            return Err(invalid("positive grid dimensions required"));
        }
        let cells = dimensions
            .iter()
            .try_fold(1_u64, |a, b| a.checked_mul(*b).ok_or(Error::SizeOverflow))?;
        Ok(Self { dimensions, cells })
    }
    pub fn dimensions(self) -> [u64; 3] {
        self.dimensions
    }
    pub fn cells(self) -> u64 {
        self.cells
    }
    pub fn index(self, coordinates: [u64; 3]) -> Result<u64> {
        if (0..3).any(|a| coordinates[a] >= self.dimensions[a]) {
            return Err(invalid("grid coordinate"));
        }
        Ok(coordinates[0]
            + self.dimensions[0] * (coordinates[1] + self.dimensions[1] * coordinates[2]))
    }
    pub(crate) fn check_read(self, first: u64, len: usize) -> Result<()> {
        let len = u64::try_from(len).map_err(|_| Error::SizeOverflow)?;
        if first > self.cells || len > self.cells - first {
            return Err(invalid("grid read range"));
        }
        Ok(())
    }
}

/// Stable x-fast columns throughout an operation. Implementations MUST fill
/// every output slot, retain missing values as None, and propagate errors.
/// Selected values are checked by consumers; arbitrary source implementations
/// are responsible for their own storage, synchronization and resource limits.
pub trait Source {
    fn layout(&self) -> Layout;
    fn variables(&self) -> usize;
    fn read(&mut self, variable: usize, first: u64, output: &mut [Option<f64>]) -> Result<()>;
}
impl Source for Grid {
    fn layout(&self) -> Layout {
        Layout::new(self.shape().dimensions().map(|d| d as u64)).expect("validated in-memory shape")
    }
    fn variables(&self) -> usize {
        self.variables().len()
    }
    fn read(&mut self, variable: usize, first: u64, output: &mut [Option<f64>]) -> Result<()> {
        self.layout().check_read(first, output.len())?;
        if variable >= self.variables().len() {
            return Err(invalid("grid variable"));
        }
        let first = usize::try_from(first).map_err(|_| Error::SizeOverflow)?;
        let variables = self.variables().len();
        for (i, value) in output.iter_mut().enumerate() {
            *value = self.values()[(first + i) * variables + variable];
        }
        Ok(())
    }
}

mod resident;
pub use resident::{Boundary, Event, Grid, Shape, Variable};
