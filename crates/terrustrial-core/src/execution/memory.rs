//! Shared logical memory accounting.
use crate::{Error, Result};
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};
/// Shared accounting for explicitly reserved payload/workspace bytes.
/// This is not an operating-system RSS limit or a replacement for fallible allocation.
#[derive(Clone, Debug)]
pub struct MemoryBudget(Arc<BudgetInner>);
#[derive(Debug)]
struct BudgetInner {
    limit: u64,
    used: AtomicU64,
}
#[derive(Debug)]
pub struct Reservation {
    budget: MemoryBudget,
    bytes: u64,
}
impl Reservation {
    /// Transfer an existing charge to an independently retained allocation.
    pub(crate) fn split(&mut self, bytes: u64) -> Result<Self> {
        if bytes > self.bytes {
            return Err(Error::MemoryBudget {
                requested: bytes,
                available: self.bytes,
            });
        }
        self.bytes -= bytes;
        Ok(Self {
            budget: self.budget.clone(),
            bytes,
        })
    }
    pub fn bytes(&self) -> u64 {
        self.bytes
    }
    /// Adjust a retained buffer's logical charge. Failed growth leaves it intact.
    pub(crate) fn resize(&mut self, bytes: u64) -> Result<()> {
        if bytes > self.bytes {
            let mut additional = self.budget.reserve(bytes - self.bytes)?;
            // Transfer the extra charge while allowing its Arc to drop normally.
            additional.bytes = 0;
        } else {
            self.budget
                .0
                .used
                .fetch_sub(self.bytes - bytes, Ordering::AcqRel);
        }
        self.bytes = bytes;
        Ok(())
    }
}
impl MemoryBudget {
    pub fn new(limit: u64) -> Self {
        Self(Arc::new(BudgetInner {
            limit,
            used: AtomicU64::new(0),
        }))
    }
    pub fn used(&self) -> u64 {
        self.0.used.load(Ordering::Acquire)
    }
    pub fn limit(&self) -> u64 {
        self.0.limit
    }
    pub fn reserve(&self, bytes: u64) -> Result<Reservation> {
        let mut used = self.used();
        loop {
            let available = self.0.limit - used;
            if bytes > available {
                return Err(Error::MemoryBudget {
                    requested: bytes,
                    available,
                });
            }
            match self.0.used.compare_exchange_weak(
                used,
                used + bytes,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    return Ok(Reservation {
                        budget: self.clone(),
                        bytes,
                    })
                }
                Err(actual) => used = actual,
            }
        }
    }
}
impl Drop for Reservation {
    fn drop(&mut self) {
        self.budget.0.used.fetch_sub(self.bytes, Ordering::AcqRel);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn resized_reservations_conserve_shared_budget_and_failed_growth() {
        let budget = MemoryBudget::new(1024);
        let barrier = std::sync::Barrier::new(8);
        std::thread::scope(|scope| {
            for _ in 0..8 {
                let budget = &budget;
                let barrier = &barrier;
                scope.spawn(move || {
                    let mut held = budget.reserve(32).unwrap();
                    held.resize(64).unwrap();
                    barrier.wait();
                    assert_eq!(budget.used(), 512);
                    assert!(held.resize(2048).is_err());
                    assert_eq!(held.bytes(), 64);
                    barrier.wait();
                    held.resize(16).unwrap();
                    barrier.wait();
                    assert_eq!(budget.used(), 128);
                    barrier.wait();
                });
            }
        });
        assert_eq!(budget.used(), 0);
    }
}
