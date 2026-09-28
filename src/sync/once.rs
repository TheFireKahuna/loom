use crate::sync::atomic::{AtomicBool, Ordering};
use crate::sync::Mutex;

/// Mock implementation of `std::sync::Once`.
///
/// `const`, as `std`'s is. A `static` one is fresh in every execution: its
/// flag and lock register afresh per execution (`rt::Registration`). The
/// first `call_once` runs its closure while racing callers block on the lock;
/// every call that returns happens after the closure completed.
#[derive(Debug)]
pub struct Once {
    done: AtomicBool,
    lock: Mutex<()>,
}

impl Once {
    /// Creates a new `Once` value.
    #[allow(clippy::new_without_default)]
    pub const fn new() -> Once {
        Once {
            done: AtomicBool::new(false),
            lock: Mutex::new(()),
        }
    }

    /// Runs `f` if no `call_once` on this value has completed, else returns
    /// once one has.
    #[track_caller]
    pub fn call_once<F: FnOnce()>(&self, f: F) {
        // Acquire: a completed initializer happens before this return.
        if self.done.load(Ordering::Acquire) {
            return;
        }
        let _g = self.lock.lock().unwrap();
        if self.done.load(Ordering::Relaxed) {
            return;
        }
        f();
        // Release: publishes the initializer to every later fast-path load.
        self.done.store(true, Ordering::Release);
    }

    /// Whether some `call_once` has completed.
    #[track_caller]
    pub fn is_completed(&self) -> bool {
        self.done.load(Ordering::Acquire)
    }
}
