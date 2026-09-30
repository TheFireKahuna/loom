use crate::rt::object;
use crate::rt::{thread, Access, Execution, Location, Synchronize, VersionVec};

use std::collections::HashMap;
use std::sync::atomic::Ordering::{Acquire, Release};

#[derive(Debug, Copy, Clone)]
pub(crate) struct RwLock {
    state: object::Ref<State>,
}

#[derive(Debug, PartialEq)]
enum Locked {
    /// Each reader with the count of its live read guards: a thread's second
    /// guard does not end the read lock when it drops.
    Read(HashMap<thread::Id, usize>),
    Write(thread::Id),
}

/// What a thread's pending operation on the lock is. Only `Read` and `Write`
/// block: another thread taking the lock disables those, never a pending `Try*`,
/// which is to fail there instead.
#[derive(Debug, Copy, Clone, PartialEq)]
pub(super) enum Action {
    /// Read lock
    Read,

    /// Write lock
    Write,

    /// `try_read`
    TryRead,

    /// `try_write`
    TryWrite,

    /// Either guard's unlock
    Unlock,
}

#[derive(Debug)]
pub(super) struct State {
    /// A single `thread::Id` when Write locked.
    /// A set of `thread::Id` when Read locked.
    lock: Option<Locked>,

    /// Tracks access to the rwlock.
    ///
    /// Every op on the lock, unlock included, races the last one. A blocking
    /// acquire must race the unlock that admits it although it cannot run
    /// first: a bounded search reaches some schedules only by switching, at
    /// the unlock, to a thread that then blocks there, at no preemption cost.
    last_access: Option<Access>,

    /// Causality transfers between threads
    synchronize: Synchronize,
}

impl RwLock {
    /// Common RwLock function
    pub(crate) fn new() -> RwLock {
        super::execution(|execution| {
            let state = execution.objects.insert(State {
                lock: None,
                last_access: None,
                synchronize: Synchronize::new(),
            });

            RwLock { state }
        })
    }

    /// Acquire the read lock.
    /// Fail to acquire read lock if already *write* locked.
    pub(crate) fn acquire_read_lock(&self, location: Location) {
        self.state
            .branch_disable(Action::Read, self.is_write_locked(), location);

        assert!(
            self.post_acquire_read_lock(),
            "expected to be able to acquire read lock"
        );
    }

    /// Acquire write lock.
    /// Fail to acquire write lock if either read or write locked.
    pub(crate) fn acquire_write_lock(&self, location: Location) {
        self.state.branch_disable(
            Action::Write,
            self.is_write_locked() || self.is_read_locked(),
            location,
        );

        assert!(
            self.post_acquire_write_lock(),
            "expected to be able to acquire write lock"
        );
    }

    pub(crate) fn try_acquire_read_lock(&self, location: Location) -> bool {
        self.state.branch_action(Action::TryRead, location);
        self.post_acquire_read_lock()
    }

    pub(crate) fn try_acquire_write_lock(&self, location: Location) -> bool {
        self.state.branch_action(Action::TryWrite, location);
        self.post_acquire_write_lock()
    }

    /// A read guard's unlock: a step of its own, so a peer's `try_write` can
    /// find the lock held however little the critical section does.
    pub(crate) fn release_read_lock(&self, location: Location) {
        self.branch_release(location);
        super::execution(|execution| {
            let state = self.state.get_mut(&mut execution.objects);
            let thread_id = execution.threads.active_id();

            state
                .synchronize
                .sync_store(&mut execution.threads, Release);

            // Establish sequential consistency between the lock's operations.
            execution.threads.seq_cst();

            let readers = match &mut state.lock {
                Some(Locked::Read(readers)) => readers,
                _ => panic!("invalid internal loom state"),
            };

            let guards = readers
                .get_mut(&thread_id)
                .expect("invalid internal loom state");
            *guards -= 1;
            if *guards == 0 {
                readers.remove(&thread_id);
            }

            if readers.is_empty() {
                state.lock = None;

                self.unlock_threads(execution, thread_id);
            }
        });
    }

    /// A write guard's unlock, a step of its own as a read unlock is.
    pub(crate) fn release_write_lock(&self, location: Location) {
        self.branch_release(location);
        super::execution(|execution| {
            let state = self.state.get_mut(&mut execution.objects);

            state.lock = None;

            state
                .synchronize
                .sync_store(&mut execution.threads, Release);

            // Establish sequential consistency between the lock's operations.
            execution.threads.seq_cst();

            let thread_id = execution.threads.active_id();

            self.unlock_threads(execution, thread_id);
        });
    }

    fn branch_release(&self, location: Location) {
        if super::execution(|execution| execution.threads.is_active()) {
            self.state.branch_action(Action::Unlock, location);
        }
    }

    fn unlock_threads(&self, execution: &mut Execution, thread_id: thread::Id) {
        // TODO: This and the above function look very similar.
        // Refactor the two to DRY the code.
        for (id, thread) in execution.threads.iter_mut() {
            if id == thread_id {
                continue;
            }

            let obj = thread
                .operation
                .as_ref()
                .map(|operation| operation.object());

            if obj == Some(self.state.erase()) {
                thread.set_runnable();
            }
        }
    }

    /// Returns `true` if RwLock is read locked
    fn is_read_locked(&self) -> bool {
        super::execution(|execution| {
            let lock = &self.state.get(&execution.objects).lock;
            matches!(lock, Some(Locked::Read(_)))
        })
    }

    /// Returns `true` if RwLock is write locked.
    fn is_write_locked(&self) -> bool {
        super::execution(|execution| {
            let lock = &self.state.get(&execution.objects).lock;
            matches!(lock, Some(Locked::Write(_)))
        })
    }

    fn post_acquire_read_lock(&self) -> bool {
        super::execution(|execution| {
            let state = self.state.get_mut(&mut execution.objects);
            let thread_id = execution.threads.active_id();

            // Set the lock to the current thread
            let mut already_locked = false;
            state.lock = match state.lock.take() {
                None => Some(Locked::Read(HashMap::from([(thread_id, 1)]))),
                Some(Locked::Read(mut threads)) => {
                    *threads.entry(thread_id).or_insert(0) += 1;
                    Some(Locked::Read(threads))
                }
                Some(Locked::Write(writer)) => {
                    already_locked = true;
                    Some(Locked::Write(writer))
                }
            };

            // The RwLock is already Write locked, so we cannot acquire a read lock on it.
            if already_locked {
                return false;
            }

            state.synchronize.sync_load(&mut execution.threads, Acquire);

            execution.threads.seq_cst();

            // Block all writer threads from attempting to acquire the RwLock
            for (id, th) in execution.threads.iter_mut() {
                if id == thread_id {
                    continue;
                }

                let op = match th.operation.as_ref() {
                    Some(op) if op.object() == self.state.erase() => op,
                    _ => continue,
                };

                if op.action() == Action::Write {
                    let location = op.location();
                    th.set_blocked(location, false);
                }
            }

            true
        })
    }

    fn post_acquire_write_lock(&self) -> bool {
        super::execution(|execution| {
            let state = self.state.get_mut(&mut execution.objects);
            let thread_id = execution.threads.active_id();

            // Set the lock to the current thread
            state.lock = match state.lock {
                Some(Locked::Read(_)) => return false,
                Some(Locked::Write(_)) => return false,
                None => Some(Locked::Write(thread_id)),
            };

            state.synchronize.sync_load(&mut execution.threads, Acquire);

            // Establish sequential consistency between locks
            execution.threads.seq_cst();

            // Block all other threads waiting to acquire the RwLock
            for (id, th) in execution.threads.iter_mut() {
                if id == thread_id {
                    continue;
                }

                match th.operation.as_ref() {
                    Some(op)
                        if op.object() == self.state.erase()
                            && (op.action() == Action::Read || op.action() == Action::Write) =>
                    {
                        let location = op.location();
                        th.set_blocked(location, false);
                    }
                    _ => continue,
                };
            }

            true
        })
    }
}

impl State {
    pub(crate) fn last_dependent_access(&self) -> Option<&Access> {
        self.last_access.as_ref()
    }

    pub(crate) fn set_last_access(&mut self, path_id: usize, version: &VersionVec) {
        Access::set_or_create(&mut self.last_access, path_id, version)
    }
}
