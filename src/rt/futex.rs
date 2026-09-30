use crate::rt::object;
use crate::rt::{self, thread, Access, Location, VersionVec};

use tracing::trace;

/// A wait queue on an atomic word, as a futex is: `wait` blocks only while
/// the word still holds the value the waiter expects, and a wake carries no
/// causality — the waiter synchronizes by re-reading the word itself.
///
/// The word here is a shadow of the atomic's latest value in modification
/// order, kept by its owner in the same step as each store to the atomic,
/// which is what a futex's compare reads.
#[derive(Debug, Copy, Clone)]
pub(crate) struct Futex {
    state: object::Ref<State>,
}

#[derive(Debug)]
pub(super) struct State {
    word: u8,
    waiters: Vec<thread::Id>,
    last_access: Option<Access>,
}

impl Futex {
    pub(crate) fn new() -> Futex {
        rt::execution(|execution| {
            let state = execution.objects.insert_with(
                || State {
                    word: 0,
                    waiters: Vec::new(),
                    last_access: None,
                },
                |state| {
                    state.word = 0;
                    state.waiters.clear();
                    state.last_access = None;
                },
            );

            trace!(?state, "Futex::new");

            Futex { state }
        })
    }

    /// Record a store of `word` to the atomic this shadows, made by the
    /// caller's current step. Not a scheduling point: the store's own is.
    pub(crate) fn set(&self, word: u8) {
        rt::execution(|execution| self.state.get_mut(&mut execution.objects).word = word);
    }

    /// Block until woken if the word is still `expected`. Returns whether it
    /// blocked.
    pub(crate) fn wait(&self, expected: u8, location: Location) -> bool {
        self.state.branch_opaque(location);

        let blocks = rt::execution(|execution| {
            let id = execution.threads.active_id();
            let state = self.state.get_mut(&mut execution.objects);

            trace!(state = ?self.state, word = state.word, expected, "Futex::wait");

            if state.word == expected {
                state.waiters.push(id);
                true
            } else {
                false
            }
        });

        if blocks {
            rt::park(location);
        }

        blocks
    }

    /// Store `word` and wake every waiter.
    pub(crate) fn wake_all(&self, word: u8, location: Location) {
        self.state.branch_opaque(location);

        rt::execution(|execution| {
            let state = self.state.get_mut(&mut execution.objects);
            state.word = word;

            trace!(state = ?self.state, waiters = ?state.waiters, "Futex::wake_all");

            for thread in std::mem::take(&mut state.waiters) {
                execution.threads.wake(thread);
            }
        })
    }
}

impl State {
    pub(super) fn last_dependent_access(&self) -> Option<&Access> {
        self.last_access.as_ref()
    }

    pub(super) fn set_last_access(&mut self, path_id: usize, version: &VersionVec) {
        Access::set_or_create(&mut self.last_access, path_id, version);
    }
}
