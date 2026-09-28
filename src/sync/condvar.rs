use super::{LockResult, MutexGuard};
use crate::rt;

use std::time::Duration;

/// Mock implementation of `std::sync::Condvar`.
#[derive(Debug)]
pub struct Condvar {
    object: rt::Registration<rt::Condvar>,
}

/// A type indicating whether a timed wait on a condition variable returned due
/// to a time out or not.
#[derive(Debug)]
pub struct WaitTimeoutResult(bool);

impl Condvar {
    /// Creates a new condition variable which is ready to be waited on and notified.
    ///
    /// `const`, as `std`'s is; see `rt::Registration` for how the two contexts
    /// register.
    pub const fn new() -> Condvar {
        Condvar {
            object: rt::Registration::new(),
        }
    }

    /// Blocks the current thread until this condition variable receives a notification.
    ///
    /// Like the real `std` condvar, this wait is subject to spurious
    /// wakeups: loom explores executions in which the wait returns without
    /// a notification (at most one spurious wake per condvar per
    /// execution), so callers must re-check their condition in a loop.
    #[track_caller]
    pub fn wait<'a, T>(&self, mut guard: MutexGuard<'a, T>) -> LockResult<MutexGuard<'a, T>> {
        // Release the RefCell borrow guard allowing another thread to lock the
        // data
        guard.unborrow();

        // Wait until notified
        self.object.get().wait(&guard.rt(), false, location!());

        // Borrow the mutex guarded data again
        guard.reborrow();

        Ok(guard)
    }

    /// Waits on this condition variable for a notification, timing out after a
    /// specified duration.
    ///
    /// Loom has no clock: `_dur` is ignored and the timeout is modeled as a
    /// nondeterministic branch instead — executions where the wait is woken
    /// by a notification and executions where the timeout fires first are
    /// both explored. A wait no notification can ever reach always resolves
    /// via the timeout (never a reported deadlock). On the timed-out branch
    /// the mutex is reacquired, like `std`, and
    /// [`WaitTimeoutResult::timed_out`] returns `true`.
    #[track_caller]
    pub fn wait_timeout<'a, T>(
        &self,
        mut guard: MutexGuard<'a, T>,
        _dur: Duration,
    ) -> LockResult<(MutexGuard<'a, T>, WaitTimeoutResult)> {
        // Release the RefCell borrow guard allowing another thread to lock the
        // data
        guard.unborrow();

        // Wait until notified or "timed out"
        let timed_out = self.object.get().wait(&guard.rt(), true, location!());

        // Borrow the mutex guarded data again
        guard.reborrow();

        Ok((guard, WaitTimeoutResult(timed_out)))
    }

    /// Blocks the current thread until `condition` returns `false`, re-checking
    /// it under the lock after every wakeup — spurious ones included, which
    /// loom explores.
    #[track_caller]
    pub fn wait_while<'a, T, F>(
        &self,
        mut guard: MutexGuard<'a, T>,
        mut condition: F,
    ) -> LockResult<MutexGuard<'a, T>>
    where
        F: FnMut(&mut T) -> bool,
    {
        while condition(&mut *guard) {
            guard = self.wait(guard)?;
        }
        Ok(guard)
    }

    /// Waits on this condition variable while `condition` returns `true`,
    /// timing out after a specified duration.
    ///
    /// As `std`'s: `condition` is checked first and after every wakeup, and
    /// the result reports a timeout only if the deadline passed with
    /// `condition` still `true`. The deadline is the timeout
    /// [`wait_timeout`](Self::wait_timeout) models: it passes when a wait's
    /// timeout branch is taken, and a spurious wakeup leaves it pending.
    #[track_caller]
    pub fn wait_timeout_while<'a, T, F>(
        &self,
        mut guard: MutexGuard<'a, T>,
        dur: Duration,
        mut condition: F,
    ) -> LockResult<(MutexGuard<'a, T>, WaitTimeoutResult)>
    where
        F: FnMut(&mut T) -> bool,
    {
        let mut deadline_passed = false;
        loop {
            if !condition(&mut *guard) {
                return Ok((guard, WaitTimeoutResult(false)));
            }
            if deadline_passed {
                return Ok((guard, WaitTimeoutResult(true)));
            }
            let (next, result) = self.wait_timeout(guard, dur)?;
            guard = next;
            deadline_passed = result.timed_out();
        }
    }

    /// Wakes up one blocked thread on this condvar.
    #[track_caller]
    pub fn notify_one(&self) {
        self.object.get().notify_one(location!());
    }

    /// Wakes up all blocked threads on this condvar.
    #[track_caller]
    pub fn notify_all(&self) {
        self.object.get().notify_all(location!());
    }
}

impl WaitTimeoutResult {
    /// Returns `true` if the wait was known to have timed out.
    pub fn timed_out(&self) -> bool {
        self.0
    }
}

impl Default for Condvar {
    fn default() -> Self {
        Self::new()
    }
}
