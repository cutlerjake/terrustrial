//! Bounded batch admission and scheduling shared by independent driver workers.
use super::{Cancellation, MemoryBudget, Reservation};
use crate::{
    data::Group,
    error::{checked_len, invalid},
    Error, Result,
};

#[derive(Clone, Copy, Debug)]
pub struct ExecutionOptions {
    /// Maximum admitted groups. Legacy drivers use batches; rolling drivers
    /// recycle individual slots without waiting for a complete batch.
    pub batch_groups: usize,
    pub maximum_targets_per_group: usize,
    /// Allowance for EACH persistent worker arena and EACH retained group slot.
    /// Drawing scratch is retained within the worker allowance.
    /// The parent reserves (batch_groups + active_workers) times this amount,
    /// plus window metadata. Deferred packed reads additionally reserve up to
    /// four blocking-service arenas. Exhaustion never reduces a neighborhood.
    pub bytes_per_group: u64,
}
impl ExecutionOptions {
    /// Initialize each native estimation or rolling-simulation worker once.
    /// The same cloned hook can initialize a packed writer separately. Existing
    /// Rayon workers and the caller are unaffected; core overlap across runs is
    /// the caller's responsibility. See [`super::ThreadStart`].
    pub fn with_thread_start(self, thread_start: super::ThreadStart) -> super::ExecutionConfig {
        super::ExecutionConfig {
            options: self,
            thread_start: Some(thread_start),
        }
    }
    pub(crate) fn worker_count(&self) -> usize {
        rayon::current_num_threads().min(self.batch_groups)
    }
    pub(crate) fn reserve(&self, budget: &MemoryBudget) -> Result<Reservation> {
        if self.batch_groups == 0
            || self.maximum_targets_per_group == 0
            || self.bytes_per_group == 0
        {
            return Err(invalid("parallel execution limits"));
        }
        let metadata = u64::try_from(checked_len(&[self.maximum_targets_per_group, 512])?)
            .map_err(|_| Error::SizeOverflow)?;
        let per_group = self
            .bytes_per_group
            .checked_add(metadata)
            .and_then(|v| v.checked_add(256))
            .ok_or(Error::SizeOverflow)?;
        let total = per_group
            .checked_mul(self.batch_groups as u64)
            .ok_or(Error::SizeOverflow)?;
        let workers = self
            .bytes_per_group
            .checked_mul(self.worker_count() as u64)
            .ok_or(Error::SizeOverflow)?;
        budget.reserve(total.checked_add(workers).ok_or(Error::SizeOverflow)?)
    }
    pub(crate) fn batch_into(
        &self,
        groups: &mut impl Iterator<Item = Group>,
        previous: &mut Option<u64>,
        cancel: &Cancellation,
        batch: &mut Vec<Group>,
    ) -> Result<()> {
        batch.clear();
        batch
            .try_reserve_exact(self.batch_groups)
            .map_err(|_| invalid("parallel batch allocation"))?;
        for group in groups.take(self.batch_groups) {
            cancel.check()?;
            validate_order(group.id, previous)?;
            if group.targets.is_empty() || group.targets.len() > self.maximum_targets_per_group {
                return Err(invalid("parallel group target limit"));
            }
            batch.push(group);
        }
        Ok(())
    }
}

pub(crate) fn validate_order(id: u64, previous: &mut Option<u64>) -> Result<()> {
    if previous.is_some_and(|old| id <= old) {
        return Err(invalid("group IDs must increase strictly in input order"));
    }
    *previous = Some(id);
    Ok(())
}
pub(crate) fn prepare_batches<W: Send, T: Send>(
    batch: &[Group],
    workers: &mut [W],
    slots: &mut [T],
    job: impl Fn(usize, &Group, &mut W, &mut T) + Sync,
) {
    if batch.is_empty() {
        return;
    }
    // Split by worker count, distributing the remainder one group per worker.
    // Fixed ceil-sized chunks otherwise leave workers idle (16 groups / 14
    // workers becomes only eight chunks). Each arena remains exclusively owned.
    fn split<W: Send, T: Send>(
        first: usize,
        batch: &[Group],
        workers: &mut [W],
        slots: &mut [T],
        job: &(impl Fn(usize, &Group, &mut W, &mut T) + Sync),
    ) {
        if workers.len() == 1 {
            for (offset, (group, slot)) in batch.iter().zip(slots).enumerate() {
                job(first + offset, group, &mut workers[0], slot);
            }
            return;
        }
        let middle = workers.len() / 2;
        let count =
            batch.len() / workers.len() * middle + (batch.len() % workers.len()).min(middle);
        let (left_workers, right_workers) = workers.split_at_mut(middle);
        let (left_slots, right_slots) = slots.split_at_mut(count);
        let (left_groups, right_groups) = batch.split_at(count);
        rayon::join(
            || split(first, left_groups, left_workers, left_slots, job),
            || split(first + count, right_groups, right_workers, right_slots, job),
        );
    }
    assert!(!workers.is_empty());
    let active = workers.len().min(batch.len());
    split(
        0,
        batch,
        &mut workers[..active],
        &mut slots[..batch.len()],
        &job,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uneven_batches_use_every_available_arena_and_preserve_output_order() {
        for (count, worker_count) in [(16, 14), (17, 14), (3, 14), (31, 3), (0, 4), (1, 1)] {
            let groups: Vec<_> = (0..count)
                .map(|id| Group {
                    id: id as u64,
                    targets: vec![],
                })
                .collect();
            let mut workers = vec![0; worker_count];
            let mut slots = vec![usize::MAX; count];
            prepare_batches(
                &groups,
                &mut workers,
                &mut slots,
                |index, group, used, slot| {
                    assert_eq!(index as u64, group.id);
                    *used += 1;
                    *slot = index;
                },
            );
            assert_eq!(slots, (0..count).collect::<Vec<_>>());
            let active = count.min(worker_count);
            assert!(workers[..active].iter().all(|n| *n > 0));
            assert!(workers[active..].iter().all(|n| *n == 0));
            assert_eq!(workers.iter().sum::<usize>(), count);
            if active > 0 {
                assert!(
                    workers[..active].iter().max().unwrap()
                        - workers[..active].iter().min().unwrap()
                        <= 1
                );
            }
        }
    }
}
