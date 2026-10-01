use crate::rt;
use crate::sync::atomic::{AtomicU8, Ordering};

use std::fmt;

const INCOMPLETE: u8 = 0;
const RUNNING: u8 = 1;
const COMPLETE: u8 = 2;
const POISONED: u8 = 3;

/// Mock implementation of `std::sync::Once`.
///
/// `const`, as `std`'s is, and fresh in every execution: its state and wait
/// queue register afresh per execution (`rt::Registration`). Modelled on
/// `std`'s futex `Once`, lock-free: the caller that moves the state from
/// incomplete (or poisoned, for `call_once_force`) to running runs the
/// closure, then publishes completion with a `Release` store and wakes every
/// waiter. Waiters block on a wait queue that carries no causality and
/// re-read the state with `Acquire`, so every return happens after the
/// closure, and a waiter gains no ordering from another waiter.
pub struct Once {
    state: AtomicU8,
    waiters: rt::Registration<rt::Futex>,
}

/// Mock implementation of `std::sync::OnceState`: the state of a [`Once`]
/// as [`Once::call_once_force`]'s closure sees it.
#[derive(Debug)]
pub struct OnceState {
    poisoned: bool,
}

impl OnceState {
    /// Whether the [`Once`] was poisoned before this closure was called.
    pub fn is_poisoned(&self) -> bool {
        self.poisoned
    }
}

/// Publishes the running closure's outcome when dropped: complete, or
/// poisoned if the closure panicked.
struct Completion<'a> {
    once: &'a Once,
    set_to: u8,
}

impl Drop for Completion<'_> {
    #[track_caller]
    fn drop(&mut self) {
        self.once.state.store(self.set_to, Ordering::Release);
        self.once.waiters.get().wake_all(self.set_to, location!());
    }
}

impl Once {
    /// Creates a new `Once` value.
    #[allow(clippy::new_without_default)]
    pub const fn new() -> Once {
        Once {
            state: AtomicU8::new(INCOMPLETE),
            waiters: rt::Registration::new(),
        }
    }

    /// Runs `f` if no call on this value has completed, else returns once one
    /// has. Panics if an earlier closure panicked, poisoning the `Once`.
    #[track_caller]
    pub fn call_once<F: FnOnce()>(&self, f: F) {
        if self.is_completed() {
            return;
        }

        let mut f = Some(f);
        self.call(false, &mut |_| f.take().unwrap()());
    }

    /// As [`call_once`](Self::call_once), but a poisoned `Once` runs `f`
    /// instead of panicking, and `f` learns of the poison from its
    /// [`OnceState`].
    #[track_caller]
    pub fn call_once_force<F: FnOnce(&OnceState)>(&self, f: F) {
        if self.is_completed() {
            return;
        }

        let mut f = Some(f);
        self.call(true, &mut |state| f.take().unwrap()(state));
    }

    /// Whether some call has completed. Reads with `Acquire`: a `true`
    /// happens after the closure, and a completion this thread is not ordered
    /// after may read as `false`.
    #[track_caller]
    pub fn is_completed(&self) -> bool {
        self.state.load(Ordering::Acquire) == COMPLETE
    }

    /// Blocks until some call has completed. Panics if the `Once` is
    /// poisoned.
    #[track_caller]
    pub fn wait(&self) {
        if !self.is_completed() {
            self.wait_until_complete(false);
        }
    }

    /// Blocks until some call has completed, waiting through poison.
    #[track_caller]
    pub fn wait_force(&self) {
        if !self.is_completed() {
            self.wait_until_complete(true);
        }
    }

    #[track_caller]
    fn call(&self, ignore_poisoning: bool, f: &mut dyn FnMut(&OnceState)) {
        let mut state = self.state.load(Ordering::Acquire);

        loop {
            match state {
                COMPLETE => return,
                POISONED if !ignore_poisoning => {
                    panic!("Once instance has previously been poisoned");
                }
                INCOMPLETE | POISONED => {
                    if let Err(now) = self.state.compare_exchange(
                        state,
                        RUNNING,
                        Ordering::Acquire,
                        Ordering::Acquire,
                    ) {
                        state = now;
                        continue;
                    }
                    self.waiters.get().set(RUNNING);

                    let mut completion = Completion {
                        once: self,
                        set_to: POISONED,
                    };
                    f(&OnceState {
                        poisoned: state == POISONED,
                    });
                    completion.set_to = COMPLETE;
                    return;
                }
                _ => state = self.block_on(state),
            }
        }
    }

    #[track_caller]
    fn wait_until_complete(&self, ignore_poisoning: bool) {
        let mut state = self.state.load(Ordering::Acquire);

        loop {
            match state {
                COMPLETE => return,
                POISONED if !ignore_poisoning => {
                    panic!("Once instance has previously been poisoned");
                }
                _ => state = self.block_on(state),
            }
        }
    }

    /// Wait for the state to move on from `seen`, then re-read it. If it
    /// already had — `seen` was stale — the newer state lies beyond what this
    /// thread has seen of the word, so the monitor wait returns at once,
    /// having observed it, and the re-read reaches it.
    #[track_caller]
    fn block_on(&self, seen: u8) -> u8 {
        if !self.waiters.get().wait(seen, location!()) {
            crate::hint::monitor_wait(&self.state);
        }
        self.state.load(Ordering::Acquire)
    }
}

impl fmt::Debug for Once {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Once").finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::Once;
    use crate::rt;
    use crate::sync::Arc;
    use crate::thread;

    /// A waiter whose read of the state was already stale when it reached the
    /// wait queue still returns once the state completes, and spends no
    /// forced-progress wait doing so: no store it has seen elsewhere is made
    /// unreadable to it.
    #[test]
    fn a_stale_wait_is_not_a_spin() {
        crate::model(|| {
            let once = Arc::new(Once::new());
            let o2 = once.clone();
            let th = thread::spawn(move || o2.call_once(|| {}));
            once.wait();
            let yielded = rt::execution(|execution| execution.threads.active().last_yield);
            assert_eq!(yielded, None, "Once::wait entered the forced-progress model");
            th.join().unwrap();
        });
    }
}
