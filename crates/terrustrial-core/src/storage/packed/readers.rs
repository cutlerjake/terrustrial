//! Available-reader checkout and bounded duplicate-load coordination.
//!
//! Pool state is held only for checkout/return. No pool guard spans SQLite,
//! decoding or cache insertion. A load keeps its key after returning the reader
//! so a second request cannot decode the same page concurrently.
use super::*;
use std::{sync::Condvar, time::Duration};

struct State {
    free: Vec<Reader>,
    active: Vec<Option<Key>>,
}
pub(super) struct Pool {
    state: Mutex<State>,
    changed: Condvar,
    _reservation: Reservation,
}
pub(super) enum Acquired<'a> {
    Cached(Arc<Cached>),
    Load(Load<'a>),
}
pub(super) struct Load<'a> {
    pool: &'a Pool,
    slot: usize,
    reader: Option<Reader>,
}
impl Pool {
    pub(super) fn new(readers: Vec<Reader>, budget: &MemoryBudget) -> Result<Self> {
        let count = readers.len();
        let bytes = count
            .checked_mul(std::mem::size_of::<Reader>() + 2 * std::mem::size_of::<Option<Key>>())
            .ok_or(Error::SizeOverflow)?;
        let reservation = budget.reserve(bytes as u64)?;
        let mut active = Vec::new();
        active
            .try_reserve_exact(count * 2)
            .map_err(|_| invalid("packed load slots allocation"))?;
        active.resize(count * 2, None);
        Ok(Self {
            state: Mutex::new(State {
                free: readers,
                active,
            }),
            changed: Condvar::new(),
            _reservation: reservation,
        })
    }
    pub(super) fn acquire(
        &self,
        key: Key,
        cancel: &Cancellation,
        mut cached: impl FnMut() -> Result<Option<Arc<Cached>>>,
    ) -> Result<Acquired<'_>> {
        let pool_lock = Stamp::now();
        let mut state = lock(&self.state)?;
        pool_lock.record(Metric::PagePoolLock);
        loop {
            cancel.check()?;
            // Completion inserts before releasing its active slot. Checking
            // under the pool lock closes the check/claim race with that release.
            if let Some(value) = cached()? {
                return Ok(Acquired::Cached(value));
            }
            if !state.active.contains(&Some(key)) && !state.free.is_empty() {
                if let Some(slot) = state.active.iter().position(Option::is_none) {
                    state.active[slot] = Some(key);
                    let reader = state.free.pop();
                    return Ok(Acquired::Load(Load {
                        pool: self,
                        slot,
                        reader,
                    }));
                }
            }
            // Cancellation currently exposes an atomic token, not a wake hook.
            // Notifications handle ordinary progress; the bounded timeout lets
            // a cancelled waiter leave even when every reader is stalled.
            let _span = Span::new(Metric::ReaderWait);
            state = self
                .changed
                .wait_timeout(state, Duration::from_millis(10))
                .map_err(|_| invalid("packed reader pool poisoned"))?
                .0;
        }
    }
}
impl Load<'_> {
    pub(super) fn connection(&self) -> &Connection {
        &self
            .reader
            .as_ref()
            .expect("reader already released")
            .connection
    }
    pub(super) fn release_reader(&mut self) {
        let _span = Span::new(Metric::PageRelease);
        if let Some(reader) = self.reader.take() {
            let mut state = self.pool.state.lock().unwrap_or_else(|e| e.into_inner());
            state.free.push(reader);
            self.pool.changed.notify_all();
        }
    }
}
impl Drop for Load<'_> {
    fn drop(&mut self) {
        let mut state = self.pool.state.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(reader) = self.reader.take() {
            state.free.push(reader);
        }
        state.active[self.slot] = None;
        self.pool.changed.notify_all();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;

    fn key(page: u64) -> Key {
        Key {
            page,
            generation: 0,
            payload: false,
        }
    }
    fn pool(count: usize, budget: &MemoryBudget) -> Pool {
        Pool::new(
            (0..count)
                .map(|_| Reader {
                    connection: Connection::open_in_memory().unwrap(),
                    _reservation: budget.reserve(4096).unwrap(),
                })
                .collect(),
            budget,
        )
        .unwrap()
    }
    fn claim<'a>(pool: &'a Pool, page: u64, cancel: &Cancellation) -> Load<'a> {
        match pool.acquire(key(page), cancel, || Ok(None)).unwrap() {
            Acquired::Load(load) => load,
            Acquired::Cached(_) => panic!("unexpected cache hit"),
        }
    }
    #[test]
    fn unrelated_pages_use_free_readers_while_duplicate_waits() {
        let budget = MemoryBudget::new(1 << 20);
        let pool = pool(2, &budget);
        let cancel = Cancellation::default();
        let first = claim(&pool, 2, &cancel);
        std::thread::scope(|scope| {
            let (tx, rx) = mpsc::channel();
            let p = &pool;
            let c = &cancel;
            scope.spawn(move || {
                let _duplicate = claim(p, 2, c);
                tx.send(()).unwrap();
            });
            // These IDs formerly used the same connection modulo two.
            let mut unrelated = claim(&pool, 4, &cancel);
            assert!(rx.try_recv().is_err());
            unrelated.release_reader();
            let _third = claim(&pool, 6, &cancel);
            drop(first);
            rx.recv_timeout(Duration::from_secs(2)).unwrap();
        });
        drop(pool);
        assert_eq!(budget.used(), 0);
    }
    #[test]
    fn cancellation_does_not_need_the_stalled_reader_to_return() {
        let budget = MemoryBudget::new(1 << 20);
        let pool = pool(1, &budget);
        let cancel = Cancellation::default();
        let first = claim(&pool, 1, &cancel);
        std::thread::scope(|scope| {
            let (tx, rx) = mpsc::channel();
            let c = &cancel;
            let p = &pool;
            scope.spawn(move || {
                tx.send(matches!(
                    p.acquire(key(2), c, || Ok(None)),
                    Err(Error::Cancelled)
                ))
                .unwrap();
            });
            cancel.cancel();
            assert!(rx.recv_timeout(Duration::from_secs(2)).unwrap());
        });
        drop(first);
        let _next = claim(&pool, 1, &Cancellation::default());
    }
}
