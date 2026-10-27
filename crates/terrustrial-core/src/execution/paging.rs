//! Cooperative cache-only queries on numerical workers. A miss transfers the
//! job to the blocking service; that service retries preparation once with
//! ordinary synchronous reads. Cache hits never make the extra queue trip.
use std::cell::Cell;
thread_local! { static CACHE_ONLY: Cell<bool> = const { Cell::new(false) }; }
#[cfg(feature = "sqlite")]
pub(crate) fn cache_only() -> bool {
    CACHE_ONLY.with(Cell::get)
}
pub(crate) fn with_cache_only<T>(enabled: bool, f: impl FnOnce() -> T) -> T {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            CACHE_ONLY.with(|flag| flag.set(self.0));
        }
    }
    let previous = CACHE_ONLY.with(|flag| flag.replace(enabled));
    let _restore = Restore(previous);
    f()
}
