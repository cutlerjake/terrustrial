//! Bounded page-tree traversal state, independent of persistence.
use crate::{error::invalid, spatial::hierarchy::Bounds, Result};
#[derive(Clone, Copy, Debug)]
pub(crate) struct Node {
    pub id: u64,
    pub bounds: Bounds,
    pub first_rank: u64,
    pub records: u64,
}

impl crate::spatial::hierarchy::Bounded for Node {
    fn bounds(&self) -> Bounds {
        self.bounds
    }
}

#[derive(Clone, Copy)]
pub(crate) struct Pending {
    pub(crate) distance: f64,
    pub(crate) node: Node,
    pub(crate) depth: usize,
}
#[derive(Default)]
pub(crate) struct Frontier {
    entries: Vec<Pending>,
}
impl Frontier {
    pub(crate) fn bytes() -> usize {
        128 * std::mem::size_of::<Pending>()
    }
    pub(crate) fn new() -> Result<Self> {
        let mut entries = Vec::new();
        entries
            .try_reserve_exact(128)
            .map_err(|_| invalid("packed frontier allocation"))?;
        Ok(Self { entries })
    }
    pub(crate) fn clear(&mut self) {
        self.entries.clear();
    }
    pub(crate) fn push(&mut self, pending: Pending) -> bool {
        if self.entries.len() == 128 {
            return false;
        }
        let mut i = self.entries.len();
        self.entries.push(pending);
        while i > 0 {
            let parent = (i - 1) / 2;
            if self.entries[parent].distance <= pending.distance {
                break;
            }
            self.entries[i] = self.entries[parent];
            i = parent;
        }
        self.entries[i] = pending;
        true
    }
    pub(crate) fn pop(&mut self) -> Option<Pending> {
        let first = *self.entries.first()?;
        let last = self.entries.pop().unwrap();
        if !self.entries.is_empty() {
            let mut i = 0;
            while i * 2 + 1 < self.entries.len() {
                let mut child = i * 2 + 1;
                if child + 1 < self.entries.len()
                    && self.entries[child + 1].distance < self.entries[child].distance
                {
                    child += 1;
                }
                if last.distance <= self.entries[child].distance {
                    break;
                }
                self.entries[i] = self.entries[child];
                i = child;
            }
            self.entries[i] = last;
        }
        Some(first)
    }
}
