use crate::rt::object;
use crate::rt::{self, thread, Access, Mutex, VersionVec};

use std::collections::VecDeque;

use tracing::trace;

use super::Location;

#[derive(Debug, Copy, Clone)]
pub(crate) struct Condvar {
    state: object::Ref<State>,
}

#[derive(Debug)]
pub(super) struct State {
    /// Tracks access to the mutex
    last_access: Option<Access>,

    /// True if a wait on the condvar already woke on its own (spuriously /
    /// by its timeout) this execution
    did_spur: bool,

    /// Threads waiting on the condvar
    waiters: VecDeque<thread::Id>,
}

impl Condvar {
    /// Create a new condition variable object
    pub(crate) fn new() -> Condvar {
        super::execution(|execution| {
            let state = execution.objects.insert_with(
                || State {
                    last_access: None,
                    did_spur: false,
                    waiters: VecDeque::new(),
                },
                |state| {
                    state.last_access = None;
                    state.did_spur = false;
                    state.waiters.clear();
                },
            );

            trace!(?state, "Condvar::new");

            Condvar { state }
        })
    }

    /// Blocks the current thread until this condition variable receives a
    /// notification, the thread wakes spuriously, or — for a timed wait —
    /// its timeout fires. Returns `true` iff the wait timed out.
    pub(crate) fn wait(&self, mutex: &Mutex, timed: bool, location: Location) -> bool {
        self.state.branch_opaque(location);

        // Decide up front whether this wait wakes on its own: the timeout
        // firing for a timed wait, a spurious wakeup for an untimed one.
        // Modeled like `Notify`'s spurious branch — a path branch explored
        // both ways — and bounded the same way: at most one self-wake per
        // condvar per execution, which keeps wait loops finite.
        let wake_self = rt::execution(|execution| {
            let spurious = if self.state.get(&execution.objects).might_spur() {
                execution.path.branch_spurious()
            } else {
                false
            };

            if spurious {
                self.state.get_mut(&mut execution.objects).did_spur = true;
            }

            spurious
        });

        if wake_self {
            trace!(state = ?self.state, ?timed, "Condvar::wait: self-wake");

            // Even a wait that never blocks releases and reacquires the
            // lock: other threads may run (and take it) in between —
            // `acquire_lock`'s branch is the schedule point.
            mutex.release_lock();
            mutex.acquire_lock(location);

            return timed;
        }

        rt::execution(|execution| {
            trace!(state = ?self.state, ?mutex, ?timed, "Condvar::wait");

            let state = self.state.get_mut(&mut execution.objects);

            // Track the current thread as a waiter
            state.waiters.push_back(execution.threads.active_id());
        });

        // Release the lock
        mutex.release_lock();

        // Disable the current thread; a timed wait can additionally be
        // revived by the deadlock rescue in `Execution::schedule` (its
        // timeout firing when nothing else can run).
        if timed {
            rt::park_timed(location);
        } else {
            rt::park(location);
        }

        // A notification dequeues its target before waking it; any other
        // wake (the timeout rescue, a stray `Thread::unpark`) leaves the
        // entry behind. Dequeue ourselves so a later notification is not
        // spent on a thread that already returned from its wait.
        let timed_out = rt::execution(|execution| {
            let id = execution.threads.active_id();
            let state = self.state.get_mut(&mut execution.objects);

            if let Some(pos) = state.waiters.iter().position(|&th| th == id) {
                state.waiters.remove(pos);
                timed
            } else {
                false
            }
        });

        // Acquire the lock again
        mutex.acquire_lock(location);

        timed_out
    }

    /// Wakes up one blocked thread on this condvar.
    pub(crate) fn notify_one(&self, location: Location) {
        self.state.branch_opaque(location);

        rt::execution(|execution| {
            let state = self.state.get_mut(&mut execution.objects);

            // Notify the first waiter
            let thread = state.waiters.pop_front();

            trace!(state = ?self.state, ?thread, "Condvar::notify_one");

            if let Some(thread) = thread {
                execution.threads.unpark(thread);
            }
        })
    }

    /// Wakes up all blocked threads on this condvar.
    pub(crate) fn notify_all(&self, location: Location) {
        self.state.branch_opaque(location);

        rt::execution(|execution| {
            let state = self.state.get_mut(&mut execution.objects);

            trace!(state = ?self.state, threads = ?state.waiters, "Condvar::notify_all");

            for thread in state.waiters.drain(..) {
                execution.threads.unpark(thread);
            }
        })
    }
}

impl State {
    fn might_spur(&self) -> bool {
        !self.did_spur
    }

    pub(super) fn last_dependent_access(&self) -> Option<&Access> {
        self.last_access.as_ref()
    }

    pub(crate) fn set_last_access(&mut self, path_id: usize, version: &VersionVec) {
        Access::set_or_create(&mut self.last_access, path_id, version);
    }
}
