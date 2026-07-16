use crate::rt::alloc::Allocation;
use crate::rt::{lazy_static, object, thread, Path};

use rustc_hash::FxHashMap;
use std::fmt;

use tracing::info;

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

    /// The object whose access records the previous `schedule()` call
    /// updated (via `set_last_access`), if any. This is what makes the
    /// DPOR backtrack scan event-driven — see `schedule()`.
    dpor_update: Option<object::Ref>,

    /// Maximum number of concurrent threads
    pub(super) max_threads: usize,

    pub(super) max_history: usize,

    /// Capture locations for significant events
    pub(crate) location: bool,

    /// Log execution output to STDOUT
    pub(crate) log: bool,
}

#[derive(Debug, Eq, PartialEq, Hash, Clone, Copy)]
pub(crate) struct Id(usize);

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
            dpor_update: None,
            max_threads,
            max_history: 7,
            location: false,
            log: false,
        }
    }

    /// Create state to track a new thread
    pub(crate) fn new_thread(&mut self) -> thread::Id {
        let thread_id = self.threads.new_thread();
        let active_id = self.threads.active_id();

        let (active, new) = self.threads.active2_mut(thread_id);

        new.causality.join(&active.causality);
        new.dpor_vv.join(&active.dpor_vv);

        // Bump causality in order to ensure CausalCell accurately detects
        // incorrect access when first action.
        new.causality[thread_id] += 1;
        active.causality[active_id] += 1;

        thread_id
    }

    /// Resets the execution state for the next execution run
    pub(crate) fn step(self) -> Option<Self> {
        let id = Id::new();
        let max_threads = self.max_threads;
        let max_history = self.max_history;
        let location = self.location;
        let log = self.log;
        let mut path = self.path;
        let mut objects = self.objects;
        let mut lazy_statics = self.lazy_statics;
        let mut raw_allocations = self.raw_allocations;
        let mut arc_objs = self.arc_objs;

        let mut threads = self.threads;

        if !path.step() {
            return None;
        }

        objects.clear();
        lazy_statics.reset();
        raw_allocations.clear();
        arc_objs.clear();

        threads.clear(id);

        Some(Execution {
            id,
            path,
            threads,
            objects,
            lazy_statics,
            raw_allocations,
            arc_objs,
            // Object refs do not survive the iteration reset.
            dpor_update: None,
            max_threads,
            max_history,
            location,
            log,
        })
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
            // here), and the access records of the one object the previous
            // call passed to `set_last_access` (`dirty`). Every other
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

                if th_id != curr_thread && Some(operation.object()) != dirty {
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

        // A thread blocked in a timed wait can always end its block on its
        // own — its timeout fires. If no thread can run otherwise, time is
        // the only mover left: fire every pending timeout instead of
        // reporting a deadlock the clock would have resolved. The woken
        // wait tells a timeout apart from a notification by its wait-queue
        // entry (see `rt::Condvar::wait`).
        if !self
            .threads
            .iter()
            .any(|(_, th)| th.is_runnable() || th.is_yield())
        {
            for (_, th) in self.threads.iter_mut() {
                if th.is_blocked_timed() {
                    th.set_runnable();
                }
            }
        }

        // It's important to avoid pre-emption as much as possible
        let mut initial = Some(self.threads.active_id());

        // If the thread is not runnable, then we can pick any arbitrary other
        // runnable thread.
        if !self.threads.active().is_runnable() {
            initial = None;

            for (i, th) in self.threads.iter() {
                if !th.is_runnable() {
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
        }

        let path_id = self.path.pos();

        let next = self.path.branch_thread(self.id, {
            self.threads.iter().map(|(i, th)| {
                if initial.is_none() && th.is_runnable() {
                    initial = Some(i);
                }

                if initial == Some(i) {
                    Thread::Active
                } else if th.is_yield() {
                    Thread::Yield
                } else if !th.is_runnable() {
                    Thread::Disabled
                } else {
                    Thread::Skip
                }
            })
        });

        let switched = Some(self.threads.active_id()) != next;

        self.threads.set_active(next);

        // There is no active thread. Unless all threads have terminated, the
        // test has deadlocked.
        if !self.threads.is_active() {
            let terminal = self.threads.iter().all(|(_, th)| th.is_terminated());

            assert!(
                terminal,
                "deadlock; threads = {:?}",
                self.threads
                    .iter()
                    .map(|(i, th)| { (i, th.state) })
                    .collect::<Vec<_>>()
            );

            return true;
        }

        // TODO: refactor
        if let Some(operation) = self.threads.active().operation {
            let threads = &mut self.threads;
            let th_id = threads.active_id();

            // The DPOR clock must dominate every conflicting predecessor,
            // i.e. all threads' dependent accesses, not just the most
            // recent one overall.
            {
                let dpor_vv = &mut threads.active_mut().dpor_vv;
                self.objects.for_each_dependent_access(operation, |access| {
                    dpor_vv.join(access.version());
                });
            }

            threads.active_mut().dpor_vv[th_id] += 1;

            self.objects
                .set_last_access(operation, th_id, path_id, &threads.active().dpor_vv);

            // This is the only place access records change; the next
            // `schedule()` call's backtrack scan re-examines exactly the
            // pending operations targeting this object.
            self.dpor_update = Some(operation.object());
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

    /// Panics if any leaks were detected
    pub(crate) fn check_for_leaks(&self) {
        self.objects.check_for_leaks();
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
