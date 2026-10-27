//! Shared budgeted storage for selected observations and target sites.
use crate::execution::Reservation;
use crate::{data::Observation, execution::MemoryBudget, Error, Result};

use crate::execution::buffer::grow;

pub(crate) struct Selection {
    pub(crate) observations: Vec<Observation>,
    reservation: Reservation,
}
impl Selection {
    pub(crate) fn new(budget: &MemoryBudget) -> Result<Self> {
        Ok(Self {
            observations: Vec::new(),
            reservation: budget.reserve(0)?,
        })
    }
    pub(crate) fn extend(&mut self, observations: &[Observation]) -> Result<()> {
        if observations.is_empty() {
            return Ok(());
        }
        let total = self
            .observations
            .len()
            .checked_add(observations.len())
            .ok_or(Error::SizeOverflow)?;
        grow(&mut self.observations, total, &mut self.reservation)?;
        self.observations.extend_from_slice(observations);
        Ok(())
    }
}
