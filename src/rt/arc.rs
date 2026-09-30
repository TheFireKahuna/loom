use crate::rt::object;
use crate::rt::{self, Access, Location, Synchronize, VersionVec};

use std::sync::atomic::Ordering::{self, Acquire, Relaxed, Release};

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

    /// The strong count's release sequence: every decrement releases into it,
    /// and an acquiring read or fence after a read of it acquires it.
    strong: Synchronize,

    /// The weak count's, which `std` keeps in a separate atomic: nothing
    /// released here reaches an acquire of `strong`, nor the reverse. The
    /// strong references collectively hold one implicit weak reference, which
    /// the step that ends the strong count releases.
    weak: Synchronize,

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

    /// Read a count with `Relaxed`, or `Acquire` for `Weak::weak_count`.
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
                strong: Synchronize::new(),
                weak: Synchronize::new(),
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
    /// exists: `std`'s `Release` increment, which a successful upgrade
    /// acquires.
    pub(crate) fn strong_init(&self) {
        rt::execution(|execution| {
            let state = self.state.get_mut(&mut execution.objects);
            assert_eq!(state.ref_cnt, 0, "[loom internal bug] cyclic Arc already live");
            state.ref_cnt = 1;
            state.strong.sync_store(&mut execution.threads, Release);
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

    /// Dropping a `Weak`: `std`'s `Release` decrement of the weak count, and
    /// the `Acquire` fence of the one that ends the allocation.
    pub(crate) fn weak_dec(&self, location: Location) {
        self.branch(Action::RefDec, location);
        rt::execution(|execution| {
            let state = self.state.get_mut(&mut execution.objects);
            assert!(state.weak_cnt >= 1, "Weak is already released");
            state.weak_cnt -= 1;
            state.weak.sync_store(&mut execution.threads, Release);
            if state.weak_cnt == 0 && state.ref_cnt == 0 {
                state.weak.sync_load(&mut execution.threads, Acquire);
            }
        })
    }

    /// `Weak::upgrade`: `std`'s increment of a nonzero strong count, `Acquire`
    /// on success and `Relaxed` on failure. Dependent with the drops: whether
    /// it succeeds is the race.
    pub(crate) fn upgrade(&self, location: Location) -> bool {
        self.branch(Action::RefDec, location);
        rt::execution(|execution| {
            let state = self.state.get_mut(&mut execution.objects);
            if state.ref_cnt == 0 {
                state.strong.sync_load(&mut execution.threads, Relaxed);
                return false;
            }
            state.ref_cnt += 1;
            state.strong.sync_load(&mut execution.threads, Acquire);
            state.strong.sync_store(&mut execution.threads, Acquire);
            true
        })
    }

    /// The strong count, read `Relaxed` as `std`'s `strong_count`s read it.
    #[track_caller]
    pub(crate) fn strong_count(&self) -> usize {
        self.branch(Action::Inspect, location!());
        rt::execution(|execution| {
            let state = self.state.get_mut(&mut execution.objects);
            state.strong.sync_load(&mut execution.threads, Relaxed);
            state.ref_cnt
        })
    }

    /// The count of `Weak` handles, read with `order`: `Relaxed` for
    /// `Arc::weak_count`, `Acquire` for `Weak::weak_count`.
    #[track_caller]
    pub(crate) fn weak_count(&self, order: Ordering) -> usize {
        self.branch(Action::Inspect, location!());
        rt::execution(|execution| {
            let state = self.state.get_mut(&mut execution.objects);
            state.weak.sync_load(&mut execution.threads, order);
            state.weak_cnt
        })
    }

    /// `try_unwrap`'s `Relaxed` CAS of the strong count from 1 to 0, with the
    /// `Acquire` fence and the implicit weak drop of its success. One step, so
    /// an upgrade precedes it, and it fails, or follows it and fails itself.
    pub(crate) fn try_unwrap(&self, location: Location) -> bool {
        self.branch(Action::RefDec, location);
        rt::execution(|execution| {
            let state = self.state.get_mut(&mut execution.objects);
            assert!(state.ref_cnt >= 1, "Arc is released");
            state.strong.sync_load(&mut execution.threads, Relaxed);
            if state.ref_cnt != 1 {
                return false;
            }
            state.ref_cnt = 0;
            state.strong.sync_store(&mut execution.threads, Relaxed);
            state.strong.sync_load(&mut execution.threads, Acquire);
            state.release_implicit_weak(&mut execution.threads);
            true
        })
    }

    /// `make_mut`'s first step, its `Acquire` CAS of the strong count from 1
    /// to 0 (`Relaxed` on failure, when it clones). Until
    /// [`make_mut_settle`](Self::make_mut_settle) no upgrade succeeds.
    pub(crate) fn make_mut_take(&self, location: Location) -> bool {
        self.branch(Action::RefDec, location);
        rt::execution(|execution| {
            let state = self.state.get_mut(&mut execution.objects);
            assert!(state.ref_cnt >= 1, "Arc is released");
            if state.ref_cnt != 1 {
                state.strong.sync_load(&mut execution.threads, Relaxed);
                return false;
            }
            state.ref_cnt = 0;
            state.strong.sync_load(&mut execution.threads, Acquire);
            state.strong.sync_store(&mut execution.threads, Acquire);
            true
        })
    }

    /// `make_mut`'s second step, its `Relaxed` read of the weak count: with no
    /// `Weak` left, the `Release` store that restores the strong reference
    /// (true: the value stays in place); otherwise the drop of the implicit
    /// weak reference, and the value moves out (false).
    pub(crate) fn make_mut_settle(&self, location: Location) -> bool {
        self.branch(Action::RefDec, location);
        rt::execution(|execution| {
            let state = self.state.get_mut(&mut execution.objects);
            assert_eq!(state.ref_cnt, 0, "[loom internal bug] make_mut lost its take");
            state.weak.sync_load(&mut execution.threads, Relaxed);
            if state.weak_cnt == 0 {
                state.ref_cnt = 1;
                state.strong.sync_store(&mut execution.threads, Release);
                true
            } else {
                state.release_implicit_weak(&mut execution.threads);
                false
            }
        })
    }

    /// `get_mut`'s uniqueness test, as `std`'s: an `Acquire` CAS locking a
    /// weak count with no `Weak` (`Relaxed` when it fails), an `Acquire` read
    /// of the strong count, and the `Release` store that unlocks. The weak
    /// lock keeps every `Weak` op out of the window, so one step is exact.
    pub(crate) fn get_mut(&self, location: Location) -> bool {
        self.branch(Action::RefDec, location);

        rt::execution(|execution| {
            let state = self.state.get_mut(&mut execution.objects);

            assert!(state.ref_cnt >= 1, "Arc is released");

            if state.weak_cnt != 0 {
                state.weak.sync_load(&mut execution.threads, Relaxed);
                return false;
            }
            state.weak.sync_load(&mut execution.threads, Acquire);
            state.strong.sync_load(&mut execution.threads, Acquire);
            state.weak.sync_store(&mut execution.threads, Release);

            let is_only_ref = state.ref_cnt == 1;

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
            state.strong.sync_store(&mut execution.threads, Release);

            if state.ref_cnt == 0 {
                // Final ref count, the arc will be dropped. This requires
                // acquiring the causality
                //
                // In the real implementation, this is done with a fence.
                state.strong.sync_load(&mut execution.threads, Acquire);
                state.release_implicit_weak(&mut execution.threads);
                true
            } else {
                false
            }
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
    /// The strong references' implicit weak reference, dropped when the strong
    /// count ends: a `Release` decrement of the weak count, and the `Acquire`
    /// fence that frees the allocation when no `Weak` remains.
    fn release_implicit_weak(&mut self, threads: &mut rt::thread::Set) {
        self.weak.sync_store(threads, Release);
        if self.weak_cnt == 0 {
            self.weak.sync_load(threads, Acquire);
        }
    }

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
