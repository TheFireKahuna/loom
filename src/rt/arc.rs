use crate::rt::object;
use crate::rt::{self, Access, Location, Synchronize, VersionVec};

use std::sync::atomic::Ordering::{Acquire, Release, SeqCst};

use tracing::trace;
#[derive(Debug)]
pub(crate) struct Arc {
    state: object::Ref<State>,
}

#[derive(Debug)]
pub(super) struct State {
    /// Reference count
    ref_cnt: usize,

    /// Live `Weak` handles. The allocation outlives the value while any does.
    weak_cnt: usize,

    /// Location where the arc was allocated
    allocated: Location,

    /// Causality transfers between threads
    ///
    /// Only updated on on ref dec and acquired before drop
    synchronize: Synchronize,

    /// Tracks access to the arc object
    last_ref_inc: Option<Access>,
    last_ref_dec: Option<Access>,
    last_ref_inspect: Option<Access>,
    last_ref_modification: Option<RefModify>,
}

/// Actions performed on the Arc
///
/// Clones are only dependent with inspections. Drops are dependent between each
/// other.
#[derive(Debug, Copy, Clone, PartialEq)]
pub(super) enum Action {
    /// Clone the arc
    RefInc,

    /// Drop the Arc
    RefDec,

    /// Inspect internals (such as get ref count). This is done with SeqCst
    /// causality
    Inspect,
}

/// Actions which modify the Arc's reference count
///
/// This is used to ascertain dependence for Action::Inspect
#[derive(Debug, Copy, Clone, PartialEq)]
enum RefModify {
    /// Corresponds to Action::RefInc
    RefInc,

    /// Corresponds to Action::RefDec
    RefDec,
}

impl Arc {
    pub(crate) fn new(location: Location) -> Arc {
        rt::execution(|execution| {
            let state = execution.objects.insert(State {
                ref_cnt: 1,
                weak_cnt: 0,
                allocated: location,
                synchronize: Synchronize::new(),
                last_ref_inc: None,
                last_ref_dec: None,
                last_ref_inspect: None,
                last_ref_modification: None,
            });

            trace!(?state, %location, "Arc::new");

            Arc { state }
        })
    }

    pub(crate) fn ref_inc(&self, location: Location) {
        self.branch(Action::RefInc, location);

        rt::execution(|execution| {
            let state = self.state.get_mut(&mut execution.objects);
            state.ref_cnt = state.ref_cnt.checked_add(1).expect("overflow");

            trace!(state = ?self.state, ref_cnt = ?state.ref_cnt, %location, "Arc::ref_inc");
        })
    }

    /// The object of an `Arc::new_cyclic` under construction: no strong
    /// reference until [`Arc::strong_init`], so every upgrade meanwhile fails.
    pub(crate) fn new_cyclic(location: Location) -> Arc {
        let arc = Arc::new(location);
        rt::execution(|execution| arc.state.get_mut(&mut execution.objects).ref_cnt = 0);
        arc
    }

    /// The first strong reference of a `new_cyclic` object, once its value
    /// exists.
    pub(crate) fn strong_init(&self) {
        rt::execution(|execution| {
            let state = self.state.get_mut(&mut execution.objects);
            assert_eq!(state.ref_cnt, 0, "[loom internal bug] cyclic Arc already live");
            state.ref_cnt = 1;
        })
    }

    /// `Arc::downgrade` or `Weak::clone`: one more weak handle. Like a clone,
    /// it needs a live handle, so it changes no answer another op can give
    /// except through the counts.
    pub(crate) fn weak_inc(&self, location: Location) {
        self.branch(Action::RefInc, location);
        rt::execution(|execution| {
            let state = self.state.get_mut(&mut execution.objects);
            state.weak_cnt = state.weak_cnt.checked_add(1).expect("overflow");
        })
    }

    /// Dropping a `Weak`: a release, as a strong drop is, which a later
    /// `get_mut` acquires.
    pub(crate) fn weak_dec(&self, location: Location) {
        self.branch(Action::RefDec, location);
        rt::execution(|execution| {
            let state = self.state.get_mut(&mut execution.objects);
            assert!(state.weak_cnt >= 1, "Weak is already released");
            state.weak_cnt -= 1;
            state.synchronize.sync_store(&mut execution.threads, Release);
        })
    }

    /// `Weak::upgrade`: one more strong reference if any is left, acquiring
    /// every earlier release when it succeeds, as `std`'s `Acquire` increment
    /// does. Dependent with the drops: whether it succeeds is the race.
    pub(crate) fn upgrade(&self, location: Location) -> bool {
        self.branch(Action::RefDec, location);
        rt::execution(|execution| {
            let state = self.state.get_mut(&mut execution.objects);
            if state.ref_cnt == 0 {
                return false;
            }
            state.ref_cnt += 1;
            state.synchronize.sync_load(&mut execution.threads, Acquire);
            true
        })
    }

    /// `(strong, weak)` counts, observed like `strong_count`.
    #[track_caller]
    pub(crate) fn counts(&self) -> (usize, usize) {
        self.branch(Action::Inspect, location!());
        rt::execution(|execution| {
            let state = self.state.get_mut(&mut execution.objects);
            state.synchronize.sync_load(&mut execution.threads, SeqCst);
            (state.ref_cnt, state.weak_cnt)
        })
    }

    /// `try_unwrap`'s test: the only strong reference, whatever `Weak`s remain
    /// (they fail to upgrade once it is taken).
    pub(crate) fn sole_strong(&self, location: Location) -> bool {
        self.branch(Action::RefDec, location);
        rt::execution(|execution| {
            let state = self.state.get_mut(&mut execution.objects);
            assert!(state.ref_cnt >= 1, "Arc is released");
            state.synchronize.sync_load(&mut execution.threads, Acquire);
            state.ref_cnt == 1
        })
    }

    /// Validate a `get_mut` call: unique when this is the only strong
    /// reference and no `Weak` could upgrade to a second.
    pub(crate) fn get_mut(&self, location: Location) -> bool {
        self.branch(Action::RefDec, location);

        rt::execution(|execution| {
            let state = self.state.get_mut(&mut execution.objects);

            assert!(state.ref_cnt >= 1, "Arc is released");

            // Synchronize the threads
            state.synchronize.sync_load(&mut execution.threads, Acquire);

            let is_only_ref = state.ref_cnt == 1 && state.weak_cnt == 0;

            trace!(state = ?self.state, ?is_only_ref, %location, "Arc::get_mut");

            is_only_ref
        })
    }

    /// Returns true if the memory should be dropped.
    pub(crate) fn ref_dec(&self, location: Location) -> bool {
        self.branch(Action::RefDec, location);

        rt::execution(|execution| {
            let state = self.state.get_mut(&mut execution.objects);

            assert!(state.ref_cnt >= 1, "Arc is already released");

            // Decrement the ref count
            state.ref_cnt -= 1;

            trace!(state = ?self.state, ref_cnt = ?state.ref_cnt, %location, "Arc::ref_dec");

            // Synchronize the threads.
            state
                .synchronize
                .sync_store(&mut execution.threads, Release);

            if state.ref_cnt == 0 {
                // Final ref count, the arc will be dropped. This requires
                // acquiring the causality
                //
                // In the real implementation, this is done with a fence.
                state.synchronize.sync_load(&mut execution.threads, Acquire);
                true
            } else {
                false
            }
        })
    }

    #[track_caller]
    pub(crate) fn strong_count(&self) -> usize {
        self.branch(Action::Inspect, location!());

        rt::execution(|execution| {
            let state = self.state.get_mut(&mut execution.objects);
            assert!(state.ref_cnt > 0, "Arc is already released");

            // Synchronize the threads.
            state.synchronize.sync_load(&mut execution.threads, SeqCst);

            state.ref_cnt
        })
    }

    fn branch(&self, action: Action, location: Location) {
        let r = self.state;
        r.branch_action(action, location);
        assert!(
            r.ref_eq(self.state),
            "Internal state mutated during branch. This is \
                usually due to a bug in the algorithm being tested writing in \
                an invalid memory location."
        );
    }
}

impl State {
    pub(super) fn check_for_leaks(&self, index: usize) {
        if self.ref_cnt != 0 || self.weak_cnt != 0 {
            if self.allocated.is_captured() {
                panic!(
                    "Arc leaked.\n  Allocated: {}\n      Index: {}",
                    self.allocated, index
                );
            } else {
                panic!("Arc leaked.\n  Index: {}", index);
            }
        }
    }

    pub(super) fn last_dependent_access(&self, action: Action) -> Option<&Access> {
        match action {
            // RefIncs are not dependent w/ RefDec, only inspections
            Action::RefInc => self.last_ref_inspect.as_ref(),
            Action::RefDec => self.last_ref_dec.as_ref(),
            Action::Inspect => match self.last_ref_modification {
                Some(RefModify::RefInc) => self.last_ref_inc.as_ref(),
                Some(RefModify::RefDec) => self.last_ref_dec.as_ref(),
                None => None,
            },
        }
    }

    pub(super) fn set_last_access(&mut self, action: Action, path_id: usize, version: &VersionVec) {
        match action {
            Action::RefInc => {
                self.last_ref_modification = Some(RefModify::RefInc);
                Access::set_or_create(&mut self.last_ref_inc, path_id, version)
            }
            Action::RefDec => {
                self.last_ref_modification = Some(RefModify::RefDec);
                Access::set_or_create(&mut self.last_ref_dec, path_id, version)
            }
            Action::Inspect => Access::set_or_create(&mut self.last_ref_inspect, path_id, version),
        }
    }
}
