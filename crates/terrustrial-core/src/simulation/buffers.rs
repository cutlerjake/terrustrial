//! Target and selection buffers used only by simulation.
use super::*;
use crate::execution::Reservation;

pub(super) struct SimulationInputs {
    pub(super) selection: Selection,
    pub(super) sites: Vec<Site>,
    sites_reservation: Reservation,
}
impl SimulationInputs {
    pub(super) fn new(budget: &MemoryBudget) -> Result<Self> {
        Ok(Self {
            selection: Selection::new(budget)?,
            sites: Vec::new(),
            sites_reservation: budget.reserve(0)?,
        })
    }
    pub(super) fn fill_sites(&mut self, group: &Group) -> Result<()> {
        grow(
            &mut self.sites,
            group.targets.len(),
            &mut self.sites_reservation,
        )?;
        self.sites.clear();
        self.sites
            .extend(group.targets.iter().map(|target| target.site));
        Ok(())
    }
}
