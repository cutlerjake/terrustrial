//! Aligned training-grid replicates and serial high-order simulation. This is an
//! exact scan of complete geometric replicates, not a nearest-pattern shortlist.
//! Grids are in memory; the underlying estimator also accepts streaming sources.
use super::{aggregate::PrefixSource, ReplicateSource, ReplicateVisitor};
use crate::{
    error::{checked_len, invalid},
    execution::{Cancellation, MemoryBudget},
    grid::{Boundary, Grid},
    Error, Result,
};
use std::collections::HashSet;

#[derive(Clone, Copy, Debug)]
pub struct Location {
    pub offset: [i64; 3],
    pub variable: usize,
}

pub struct GridReplicates<'a> {
    grid: &'a Grid,
    targets: &'a [usize],
    locations: &'a [Location],
    boundary: Boundary,
    budget: &'a MemoryBudget,
    cancellation: &'a Cancellation,
}
impl<'a> GridReplicates<'a> {
    /// Scan centers in stable x-fast order. Skip an entire replicate if ANY target
    /// or conditioning value is missing or outside a bounded TI. The fitted row
    /// count reports the retained support; no missing coordinate is marginalized.
    /// When used as a `PrefixSource`, retain the initial consecutive observed
    /// conditioning values instead, ending the prefix at its first missing value.
    pub fn new(
        grid: &'a Grid,
        targets: &'a [usize],
        locations: &'a [Location],
        boundary: Boundary,
        budget: &'a MemoryBudget,
        cancellation: &'a Cancellation,
    ) -> Result<Self> {
        cancellation.check()?;
        if targets.is_empty()
            || targets.iter().any(|v| *v >= grid.variables().len())
            || locations
                .iter()
                .any(|v| v.variable >= grid.variables().len())
        {
            return Err(invalid("high-order grid replicate attributes"));
        }
        let count = targets
            .len()
            .checked_add(locations.len())
            .ok_or(Error::SizeOverflow)?;
        let _reservation = budget
            .reserve(u64::try_from(checked_len(&[count, 64])?).map_err(|_| Error::SizeOverflow)?)?;
        let mut seen = HashSet::new();
        if targets.iter().any(|v| !seen.insert(*v)) {
            return Err(invalid("repeated high-order target attribute"));
        }
        let mut seen = HashSet::new();
        if locations
            .iter()
            .any(|v| !seen.insert((v.offset, v.variable)))
        {
            return Err(invalid("repeated high-order conditioning location"));
        }
        Ok(Self {
            grid,
            targets,
            locations,
            boundary,
            budget,
            cancellation,
        })
    }
}
impl ReplicateSource for GridReplicates<'_> {
    fn scan(&self, visitor: &mut ReplicateVisitor<'_>) -> Result<()> {
        self.scan_rows(visitor, false)
    }
}
impl PrefixSource for GridReplicates<'_> {
    fn scan_prefixes(&self, visitor: &mut ReplicateVisitor<'_>) -> Result<()> {
        self.scan_rows(visitor, true)
    }
}
impl GridReplicates<'_> {
    fn scan_rows(&self, visitor: &mut ReplicateVisitor<'_>, partial: bool) -> Result<()> {
        let count = self
            .targets
            .len()
            .checked_add(self.locations.len())
            .ok_or(Error::SizeOverflow)?;
        let _reservation = self
            .budget
            .reserve(u64::try_from(checked_len(&[count, 8])?).map_err(|_| Error::SizeOverflow)?)?;
        let mut target = vec![0.0; self.targets.len()];
        let mut condition = vec![0.0; self.locations.len()];
        'center: for center in 0..self.grid.shape().cells() {
            self.cancellation.check()?;
            for (out, &variable) in target.iter_mut().zip(self.targets) {
                let Some(value) = self.grid.get(center, variable)? else {
                    continue 'center;
                };
                *out = value;
            }
            let mut count = 0;
            for (out, location) in condition.iter_mut().zip(self.locations) {
                let value = self
                    .grid
                    .shape()
                    .offset(center, location.offset, self.boundary)
                    .map(|cell| self.grid.get(cell, location.variable))
                    .transpose()?
                    .flatten();
                match value {
                    Some(value) => {
                        *out = value;
                        count += 1;
                    }
                    None if partial => break,
                    None => continue 'center,
                }
            }
            visitor(&target, &condition[..count])?;
        }
        Ok(())
    }
}
