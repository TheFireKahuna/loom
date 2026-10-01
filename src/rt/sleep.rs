//! Sleep sets for the unbounded search (`rt::dpor`).
//!
//! A thread goes to sleep at a schedule branch when a sibling alternative
//! explored before the current one already covers the executions that differ
//! only by running this thread earlier: every behavior reachable by
//! scheduling it now is reachable in that sibling's subtree by commuting its
//! next operation forward across the independent operations in between. It
//! wakes the moment a conflicting operation executes — past that point the
//! commutation argument no longer holds. Scheduling a still-sleeping thread
//! therefore proves the whole remainder redundant, and the execution
//! finishes as a non-exploring scout (`Execution::skip`), closing the
//! subtree.
//!
//! "Explored before" is the order the branch's alternatives are taken in: the
//! wakeup tree's order at a branch the path owns, and the claim order at a
//! branch frozen for sharding, where alternatives run concurrently and the
//! claim word's history is the one order every worker agrees on
//! (`Path::branch_thread`). Either way the set an alternative defers to is
//! fixed when the alternative is taken, so every execution of its subtree
//! starts from the same sleep set: a wakeup tree inserted under one sleep set
//! is never replayed under a larger one, which would cut the very executions
//! whose races open the alternatives the tree still owes.
//!
//! Never engaged under a preemption bound. The commutation the whole scheme
//! rests on is budget-blind: moving the sleeper's operation to the front of
//! the deferred-to subtree inserts a context switch and a switch-back, so
//! the covering linearization can cost up to two preemptions more than the
//! execution it covers — and be truncated by the very bound the pruned
//! execution satisfied (the interaction studied by Coons, Musuvathi &
//! McKinley, OOPSLA'13). This is empirical, not hypothetical: the
//! differential fuzz corpus (`tests/reduction.rs`) produces behavior loss
//! within seconds when deference is allowed under a bound, and measures the
//! defensible residue at a few percent. `Path::branch_thread` reports no
//! coverable siblings on bounded walks, and `Builder::new_execution` turns
//! the whole policy off.

use crate::rt::thread;

/// Threads whose next operation is covered by a sibling subtree.
///
/// One word of per-execution state, recomputed from scratch on every replay —
/// there is no carcass storage here to leak across epochs.
#[derive(Debug, Default)]
pub(crate) struct SleepSet {
    /// Bit per thread index. `Path::new` asserts `MAX_THREADS <= 16`.
    asleep: u16,
}

impl SleepSet {
    pub(crate) fn clear(&mut self) {
        self.asleep = 0;
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.asleep == 0
    }

    /// The sleepers, a bit per thread index.
    pub(crate) fn mask(&self) -> u16 {
        self.asleep
    }

    pub(crate) fn contains(&self, id: thread::Id) -> bool {
        self.asleep & (1u16 << id.as_usize()) != 0
    }

    /// Put the covered siblings of a just-passed branch to sleep.
    pub(crate) fn cover(&mut self, covered: u16) {
        self.asleep |= covered;
    }

    /// Wake these threads: a conflicting operation has executed, so their
    /// covered-elsewhere justification is spent.
    pub(crate) fn wake(&mut self, woken: u16) {
        self.asleep &= !woken;
    }

    /// Drop the whole set. For the transitions the commutation argument says
    /// nothing about — time firing a timed wait instead of an operation.
    pub(crate) fn wake_all(&mut self) {
        self.asleep = 0;
    }
}
