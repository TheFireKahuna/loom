use crate::rt::alloc::Allocation;
use crate::rt::sleep::SleepSet;
use crate::rt::{lazy_static, object, thread, Path};

use rustc_hash::FxHashMap;
use std::any::Any;
use std::fmt::{self, Write};

use tracing::{info, trace};

pub(crate) struct Execution {
    /// Uniquely identifies an execution
    pub(super) id: Id,

    /// Execution path taken
    pub(crate) path: Path,

    pub(crate) threads: thread::Set,

    pub(crate) lazy_statics: lazy_static::Set,

    /// All loom aware objects part of this execution run.
    pub(super) objects: object::Store,

    /// Maps raw allocations to LeakTrack objects
    pub(super) raw_allocations: FxHashMap<usize, Allocation>,

    pub(crate) arc_objs: FxHashMap<*const (), std::sync::Arc<super::Arc>>,

    /// Registrations of `const`-constructed atomic cells, keyed by the cell's
    /// globally-unique identity (`rt::atomic::Atomic::cell_id`).
    ///
    /// A cell built in a `const` context cannot register with an execution at
    /// construction — there is none — so it registers on its first access of
    /// each execution and looks itself up here afterwards. Keyed on the minted
    /// identity rather than the cell's address so that moving the cell carries
    /// its history with it, and so a later cell reusing a freed address is a
    /// distinct entry rather than an alias.
    ///
    /// Per-`Execution`, so parallel workers — which share the `static`s a
    /// const constructor exists to serve, but never share an `Execution`
    /// (`model::check_parallel`) — each hold their own registration of the one
    /// cell. Cleared per iteration alongside every other object ref.
    pub(super) deferred_atomics: FxHashMap<u64, object::Ref<super::atomic::State>>,

    /// The same for `const`-built locks and condvars (`rt::Registration`).
    pub(super) deferred_objects: FxHashMap<u64, super::Deferred>,

    /// This execution's instances of `const`-built locks' data, by lock
    /// identity, in first-touch order (`rt::instance`). Dropped at the end
    /// of the execution with the lazy statics, in reverse.
    pub(super) lock_data: Vec<(u64, super::registration::Instance)>,

    /// The address space materialized cells live in: committed ranges, the
    /// cells registered in them keyed by address (their identity, since such a
    /// cell carries no identity word and cannot move), and outstanding resets.
    /// Cleared per iteration — it holds object refs and epochs in this
    /// execution's thread numbering.
    pub(super) vm: super::atomic::Vm,

    /// Report memory still committed when an execution ends
    /// (`Builder::check_committed_leaks`).
    pub(crate) check_committed_leaks: bool,

    /// The operation whose access records the previous `schedule()` call
    /// updated (via `set_last_access`), if any. This is what makes the
    /// DPOR backtrack scan event-driven — see `schedule()`.
    dpor_update: Option<object::Operation>,

    /// Capture locations for significant events
    pub(crate) location: bool,

    /// Log execution output to STDOUT
    pub(crate) log: bool,

    /// Reincarnate objects in place across iterations (`Builder::reuse_objects`).
    pub(crate) reuse_objects: bool,

    /// Threads asleep at the current point of the execution (`rt::sleep`).
    sleep: SleepSet,

    /// Prune sleep-set-redundant executions (`Builder::sleep_sets`, gated off
    /// under a preemption bound).
    pub(crate) sleep_sets: bool,

    /// Executions this instance cut short as sleep-set redundant. Cumulative
    /// across iterations; never reset.
    ///
    /// A scout still runs its mark scan: its branches are non-exploring, so
    /// its races land on exploring ancestors. That can open an alternative a
    /// full walk would not have (a few extra executions on small trees), but
    /// suppressing the marks loses behaviors outright under forward
    /// deference — the deferred-to subtree is explored *later*, and scout
    /// races are part of how its obligations get planted (fuzz seed 28).
    pub(crate) pruned: usize,
}

#[derive(Debug, Eq, PartialEq, Hash, Clone, Copy)]
pub(crate) struct Id(usize);

/// Why an execution failed. Raised once, on the thread that called
/// `Builder::check`, after exploration has stopped.
pub(crate) enum Failure {
    /// A model thread panicked. Its hook has already reported the panic at
    /// its site, so this is re-raised without running the hook again.
    Panic(Box<dyn Any + Send>),

    /// A failure the runtime detected, not yet reported anywhere.
    Report(String),
}

impl Failure {
    pub(crate) fn raise(self) -> ! {
        match self {
            Failure::Panic(payload) => std::panic::resume_unwind(payload),
            Failure::Report(report) => panic!("{report}"),
        }
    }
}

impl Execution {
    /// Create a new execution.
    ///
    /// This is only called at the start of a fuzz run. The same instance is
    /// reused across permutations.
    pub(crate) fn new(
        max_threads: usize,
        max_branches: usize,
        preemption_bound: Option<usize>,
        exploring: bool,
    ) -> Execution {
        let id = Id::new();
        let threads = thread::Set::new(id, max_threads);

        let preemption_bound =
            preemption_bound.map(|bound| bound.try_into().expect("preemption_bound too big"));

        Execution {
            id,
            path: Path::new(max_branches, preemption_bound, exploring),
            threads,
            lazy_statics: lazy_static::Set::new(),
            objects: object::Store::with_capacity(max_branches),
            raw_allocations: FxHashMap::default(),
            arc_objs: FxHashMap::default(),
            deferred_atomics: FxHashMap::default(),
            deferred_objects: FxHashMap::default(),
            lock_data: Vec::new(),
            vm: super::atomic::Vm::default(),
            check_committed_leaks: false,
            dpor_update: None,
            location: false,
            log: false,
            reuse_objects: true,
            sleep: SleepSet::default(),
            sleep_sets: true,
            pruned: 0,
        }
    }

    /// Create state to track a new thread
    pub(crate) fn new_thread(&mut self, symmetry: Option<thread::Symmetry>) -> thread::Id {
        let thread_id = self.threads.new_thread(symmetry);
        let active_id = self.threads.active_id();

        let (active, new) = self.threads.active2_mut(thread_id);

        new.causality.join(&active.causality);
        new.dpor_vv.join(&active.dpor_vv);

        // Bump causality in order to ensure CausalCell accurately detects
        // incorrect access when first action.
        new.causality.inc(thread_id);
        active.causality.inc(active_id);

        thread_id
    }

    /// Resets the execution state for the next execution run. Returns `false`
    /// when the path is fully explored.
    pub(crate) fn step(&mut self) -> bool {
        if !self.path.step() {
            return false;
        }

        self.reset_iteration();
        true
    }

    /// Reset every per-iteration structure in place, keeping its allocations:
    /// the object store keeps its entries as reincarnation carcasses
    /// (`begin_epoch`), the maps keep their tables, the thread set its
    /// backing storage. Also the seam a pooled `Execution` crosses when a
    /// parallel worker reuses it for a fresh subtree.
    pub(crate) fn reset_iteration(&mut self) {
        let id = Id::new();
        self.id = id;

        if self.reuse_objects {
            self.objects.begin_epoch();
        } else {
            self.objects.clear();
        }
        self.lazy_statics.reset();
        self.raw_allocations.clear();
        self.arc_objs.clear();
        // Deferred cells re-register on their first access of the next
        // iteration; their identities persist (they live in the cells), their
        // registrations do not.
        self.deferred_atomics.clear();
        self.deferred_objects.clear();
        debug_assert!(self.lock_data.is_empty(), "lock data outlived its execution");
        self.lock_data.clear();
        self.vm.clear();
        self.threads.clear(id);
        self.sleep.clear();

        // Object refs do not survive the iteration reset.
        self.dpor_update = None;
    }

    /// Returns `true` if a switch is required
    pub(crate) fn schedule(&mut self) -> bool {
        use crate::rt::path::Thread;

        // Implementation of the DPOR algorithm.

        let curr_thread = self.threads.active_id();

        {
            let objects = &self.objects;
            let path = &mut self.path;
            let dirty = self.dpor_update;

            // Event-driven backtrack scan. A (pending op, object) pair can
            // only produce a backtrack point it has not already produced
            // when one of its inputs changed since the pair was last
            // checked, and between two `schedule()` calls exactly three
            // things change: the just-ran thread's pending operation (set
            // immediately before this call), its `dpor_vv` (grown when the
            // previous call activated it — and it is always `curr_thread`
            // here), and the access records the previous call's operation
            // updated through `set_last_access` (`dirty`: its object's, and
            // S's when it is an SC operation — `Operation::may_affect`). Every other
            // pair's check is a pure function of unchanged inputs whose
            // marks were already inserted — `Path::backtrack` is
            // idempotent and time-invariant within an iteration — so
            // re-running it cannot alter exploration, only burn time.
            // (A `dpor_vv` that grew can also *stop* producing a mark the
            // stale inputs produced, but never start: version vectors only
            // grow, so `happens_before` flips false→true only.)
            for (th_id, th) in self.threads.iter() {
                let operation = match th.operation {
                    Some(operation) => operation,
                    None => continue,
                };

                if th_id != curr_thread && !dirty.is_some_and(|d| d.may_affect(&operation)) {
                    continue;
                }

                // Every dependent access that is concurrent with this
                // operation (not ordered before it) is a race DPOR must
                // explore both ways: track a backtrack point at each.
                objects.for_each_dependent_access(operation, |access| {
                    if access.happens_before(&th.dpor_vv) {
                        // The previous access happened before this access,
                        // thus there is no race.
                        return;
                    }

                    // Track backtracking point
                    path.backtrack(access.path_id(), th_id);
                });
            }
        }

        // A thread in a timed wait is schedulable: scheduling it is its
        // timeout firing, with its pending operation on the object it waits
        // on, so DPOR orders the firing against that object's notifications
        // both ways. With nothing else runnable (spinners wait on the others
        // and do not count) any timeout may fire — time is the only mover
        // left, never a deadlock. While another thread can run, a thread's
        // timeout may fire early once per execution, which bounds every
        // timed-wait loop.
        let others_runnable = self.threads.iter().any(|(_, th)| th.is_runnable());
        let timed = self.threads.iter().any(|(_, th)| th.is_blocked_timed());

        // Threads symmetry holds back from being scheduled for now
        // (`thread::symmetric`). Shown as disabled below: unschedulable
        // and never a DPOR alternative, which is the entire reduction.
        let pinned = self.threads.symmetry_pinned_mask();

        // It's important to avoid pre-emption as much as possible
        let mut initial = Some(self.threads.active_id());

        // If the thread is not runnable, then we can pick any arbitrary other
        // runnable thread; failing that, a timeout fires before a spinner
        // runs again.
        if !self.threads.active().is_runnable() {
            initial = None;

            for (i, th) in self.threads.iter() {
                if !th.is_runnable() || pinned & (1 << i.as_usize()) != 0 {
                    continue;
                }

                if let Some(ref mut init) = initial {
                    if th.yield_count < self.threads[*init].yield_count {
                        *init = i;
                    }
                } else {
                    initial = Some(i)
                }
            }

            if initial.is_none() {
                initial = self
                    .threads
                    .iter()
                    .find(|(_, th)| th.may_time_out(others_runnable))
                    .map(|(i, _)| i);
            }
        }

        // Timeouts are not operations the sleep-set commutation argument
        // covers: a firing's enabledness hangs on what else is runnable. No
        // thread sleeps across a point where one could fire.
        if timed {
            self.sleep.wake_all();
        }

        // A sleeping thread must not be *started* here: its next operation is
        // covered by a sibling subtree (`rt::sleep`). Prefer any non-sleeping
        // runnable thread; when none exists the node is sleep-set blocked —
        // every continuation is redundant — so finish as a scout.
        if self.sleep_sets && !self.sleep.is_empty() {
            if let Some(id) = initial {
                if self.sleep.contains(id) {
                    let replacement = self
                        .threads
                        .iter()
                        .filter(|&(i, th)| {
                            th.is_runnable()
                                && pinned & (1 << i.as_usize()) == 0
                                && !self.sleep.contains(i)
                        })
                        .min_by_key(|&(_, th)| th.yield_count)
                        .map(|(i, _)| i);

                    match replacement {
                        Some(other) => initial = Some(other),
                        None => {
                            if !self.path.is_skipping() {
                                self.pruned += 1;
                                self.path.skip_branch();
                            }
                        }
                    }
                }
            }
        }

        let path_id = self.path.pos();

        let (next, covered) = self.path.branch_thread(self.id, {
            self.threads.iter().map(|(i, th)| {
                let is_pinned = pinned & (1 << i.as_usize()) != 0;

                if initial.is_none() && th.is_runnable() && !is_pinned {
                    initial = Some(i);
                }

                if initial == Some(i) {
                    Thread::Active
                } else if th.is_yield() {
                    Thread::Yield
                } else if th.may_time_out(others_runnable) {
                    Thread::Skip
                } else if !th.is_runnable() || is_pinned {
                    Thread::Disabled
                } else {
                    Thread::Skip
                }
            })
        });

        if self.sleep_sets && !timed {
            self.sleep.cover(covered);

            // Replay scheduled a thread that is still asleep: everything from
            // here is a reordering the deferred-to subtree already covers.
            if let Some(id) = next {
                if self.sleep.contains(id) && !self.path.is_skipping() {
                    self.pruned += 1;
                    self.path.skip_branch();
                }
            }
        }

        // No thread can run. Unless all threads have terminated, the test has
        // deadlocked: the execution fails with the active thread left in
        // place, and that thread switches out to the driver for good.
        if next.is_none() && !self.threads.iter().all(|(_, th)| th.is_terminated()) {
            std::hint::cold_path();
            super::Scheduler::fail(Failure::Report(self.deadlock_report()));
            return true;
        }

        let switched = Some(self.threads.active_id()) != next;

        self.threads.set_active(next);

        if !self.threads.is_active() {
            return true;
        }

        if self.threads.active().is_blocked_timed() {
            trace!(thread = ?self.threads.active_id(), ?others_runnable, "timeout fires");
            self.threads.active_mut().fire_timeout(others_runnable);
        }

        // The chosen thread is scheduled now: any symmetry pin waiting on it
        // releases from the next branch on.
        self.threads.active_mut().started = true;

        // TODO: refactor
        if let Some(operation) = self.threads.active().operation {
            // The operation about to run wakes any sleeper it conflicts with:
            // from here on, scheduling that sleeper is no longer a commutation
            // of an explored subtree.
            if self.sleep_sets && !self.sleep.is_empty() {
                let mut woken = 0u16;

                for (id, th) in self.threads.iter() {
                    if self.sleep.contains(id) {
                        if let Some(pending) = th.operation {
                            if pending.conflicts_with(&operation) {
                                woken |= 1u16 << id.as_usize();
                            }
                        }
                    }
                }

                self.sleep.wake(woken);
            }

            let threads = &mut self.threads;
            let th_id = threads.active_id();

            // The DPOR clock must dominate every conflicting predecessor,
            // i.e. all threads' dependent accesses, not just the most
            // recent one overall.
            {
                let active = threads.active_mut();
                active.dpor_prior = (!self.path.is_bounded()).then_some(active.dpor_vv);
                let dpor_vv = &mut active.dpor_vv;
                self.objects.for_each_dependent_access(operation, |access| {
                    dpor_vv.join(access.version());
                });
            }

            threads.active_mut().dpor_vv.inc(th_id);

            self.objects
                .set_last_access(operation, th_id, path_id, &threads.active().dpor_vv);

            // This is the only place access records change; the next
            // `schedule()` call's backtrack scan re-examines exactly the
            // pending operations targeting this object.
            self.dpor_update = Some(operation);
        } else {
            self.dpor_update = None;
        }

        // Reactivate yielded threads, but only if the current active thread is
        // not yielded.
        for (id, th) in self.threads.iter_mut() {
            if th.is_yield() && Some(id) != next {
                th.set_runnable();
            }
        }

        if switched {
            info!("~~~~~~~~ THREAD {} ~~~~~~~~", self.threads.active_id());
        }

        curr_thread != self.threads.active_id()
    }

    /// Every thread and what it waits on, for a deadlock's failure report.
    #[cold]
    fn deadlock_report(&self) -> String {
        use crate::rt::thread::State;

        let mut report = String::from("deadlock: no thread can make progress");
        let mut located = false;

        for (id, th) in self.threads.iter() {
            let _ = write!(report, "\n    thread {}: ", id.as_usize());

            let location = match th.state {
                State::Blocked { location, timed } => {
                    // A pending operation on a stored object is what the
                    // thread blocked on; a thread that parked has none.
                    let object = th.operation.and_then(|operation| {
                        let obj = operation.object();
                        Some((self.objects.kind(obj)?, obj.index()))
                    });

                    match object {
                        Some((kind, index)) => {
                            let _ = write!(report, "blocked on {kind} #{index}");
                        }
                        None => report.push_str("parked"),
                    }

                    if timed {
                        report.push_str(" (timed)");
                    }

                    Some(location)
                }
                State::Runnable { .. } => {
                    report.push_str("runnable");
                    None
                }
                State::Yield => {
                    report.push_str("yielded");
                    None
                }
                State::Terminated => {
                    report.push_str("terminated");
                    None
                }
            };

            if let Some(location) = location.filter(|l| l.is_captured()) {
                let _ = write!(report, " at {location}");
                located = true;
            }
        }

        if !located {
            report.push_str("\n(set LOOM_LOCATION=1 to name where each thread blocked)");
        }

        report
    }

    /// Hand out the execution's lock data to be dropped by the caller,
    /// outside the execution borrow, last-touched first.
    pub(crate) fn take_lock_data(&mut self) -> Vec<super::registration::Instance> {
        self.lock_data.drain(..).rev().map(|(_, instance)| instance).collect()
    }

    /// Wake every sleeping thread: an operation whose dependence the sleep set
    /// cannot see from `Operation::conflicts_with` ran.
    pub(crate) fn wake_sleepers(&mut self) {
        self.sleep.wake_all();
    }

    /// Panics if any leaks were detected
    pub(crate) fn check_for_leaks(&self) {
        self.objects.check_for_leaks();

        if self.check_committed_leaks {
            self.vm.check_for_leaks();
        }
    }
}

impl fmt::Debug for Execution {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt.debug_struct("Execution")
            .field("path", &self.path)
            .field("threads", &self.threads)
            .finish()
    }
}

impl Id {
    pub(crate) fn new() -> Id {
        use std::sync::atomic::AtomicUsize;
        use std::sync::atomic::Ordering::Relaxed;

        // The number picked here is arbitrary. It is mostly to avoid collision
        // with "zero" to aid with debugging.
        static NEXT_ID: AtomicUsize = AtomicUsize::new(46_413_762);

        let next = NEXT_ID.fetch_add(1, Relaxed);

        Id(next)
    }
}
