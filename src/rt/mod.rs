#[macro_use]
mod location;
pub(crate) use self::location::Location;

mod access;
use self::access::Access;

mod alloc;
pub(crate) use self::alloc::{alloc, dealloc, Allocation};

mod arc;
pub(crate) use self::arc::Arc;

mod atomic;
pub(crate) use self::atomic::{
    fence, publish, reset, unpublish, zero_exclusive, Atomic, ModelOps, FULL_MASK,
};
// Re-exported further by `sync::atomic::materialized`, which is why these are
// `pub` rather than `pub(crate)` — `rt` itself is private, so nothing leaks
// until that module chooses to.
pub use self::atomic::{Atomic as ConstructedCell, Cell1, Cell16, Cell2, Cell4, Cell8};

pub(crate) mod cell;
pub(crate) use self::cell::Cell;

mod condvar;
pub(crate) use self::condvar::Condvar;

mod execution;
pub(crate) use self::execution::Execution;

mod notify;
pub(crate) use self::notify::Notify;

mod num;
pub(crate) use self::num::Numeric;

#[macro_use]
pub(crate) mod object;

mod mpsc;
pub(crate) use self::mpsc::Channel;

mod mutex;
pub(crate) use self::mutex::Mutex;

mod path;
pub(crate) use self::path::Path;

mod rwlock;
pub(crate) use self::rwlock::RwLock;

mod scheduler;
pub(crate) use self::scheduler::Scheduler;

mod sleep;

mod synchronize;
pub(crate) use self::synchronize::Synchronize;

pub(crate) mod lazy_static;
pub(crate) mod thread;

mod vv;
pub(crate) use self::vv::VersionVec;

use tracing::trace;

/// Maximum number of threads that can be included in a model.
pub const MAX_THREADS: usize = 5;

/// Maximum number of atomic store history to track per-cell.
pub(crate) const MAX_ATOMIC_HISTORY: usize = 7;

pub(crate) fn spawn<F>(stack_size: Option<usize>, symmetric: bool, f: F) -> crate::rt::thread::Id
where
    F: FnOnce() + 'static,
{
    let id = execution(|execution| execution.new_thread(symmetric));

    trace!(thread = ?id, "spawn");

    Scheduler::spawn(
        stack_size,
        Box::new(move || {
            f();
            thread_done();
        }),
    );

    id
}

/// Marks the current thread as blocked until a modeled primitive wakes it
/// (`thread::Set::wake`). Independent of the `std::thread::park` token.
pub(crate) fn park(location: Location) {
    block(location, false);
}

/// Marks the current thread as blocked in a timed wait: the block can end
/// on its own (the wait's timeout firing), which `Execution::schedule`
/// models by waking the thread when nothing else can run.
pub(crate) fn park_timed(location: Location) {
    block(location, true);
}

fn block(location: Location, timed: bool) {
    let switch = execution(|execution| {
        let thread = execution.threads.active_id();
        let active = execution.threads.active_mut();

        trace!(?thread, ?timed, "block");

        active.set_blocked(location, timed);
        active.operation = None;
        execution.schedule()
    });

    if switch {
        Scheduler::switch();
    }
}

/// `std::thread::park`: return once the thread's token is available,
/// consuming it — the acquire half of unpark→park synchronization — or
/// spuriously, which `std` permits.
///
/// Both halves operate on the thread's park object, dependent with every
/// `unpark` of it, so the search explores each unpark on either side of the
/// token check. The spurious return is explored at most once per thread per
/// execution: a caller's park loop then re-checks and parks for real, so every
/// such loop terminates while every call site can still be the one that spurs.
pub(crate) fn park_thread(location: Location) {
    let id = execution(|execution| execution.threads.active_id());

    branch_park(id, location);

    let parked = execution(|execution| {
        let active = execution.threads.active_mut();

        if active.take_park_token() {
            trace!(thread = ?id, "park: token");
            return false;
        }

        if active.may_spur_park() && execution.path.branch_spurious() {
            trace!(thread = ?id, "park: spurious");
            execution.threads.active_mut().spend_park_spur();
            return false;
        }

        trace!(thread = ?id, "park: blocked");
        execution.threads.active_mut().set_parked(location);
        true
    });

    if !parked {
        return;
    }

    block_parked();

    // Only `unpark` wakes a parked thread; re-check its token as a park-object
    // operation of its own, ordered after the unpark that woke it.
    branch_park(id, location);

    execution(|execution| {
        let taken = execution.threads.active_mut().take_park_token();
        assert!(taken, "[loom internal bug] parked thread woken without a token");
    });
}

/// Give up the processor after `set_parked`.
fn block_parked() {
    let switch = execution(|execution| {
        execution.threads.active_mut().operation = None;
        execution.schedule()
    });

    if switch {
        Scheduler::switch();
    }
}

fn branch_park(id: thread::Id, location: Location) {
    branch(|execution| {
        execution.threads.active_mut().operation = Some(object::Operation::park(id, location));
    });
}

/// `std::thread::Thread::unpark` of `target`: make its token available
/// (`thread::Set::unpark`). An operation on the target's park object.
pub(crate) fn unpark_thread(target: thread::Id, location: Location) {
    branch(|execution| {
        execution.threads.active_mut().operation =
            Some(object::Operation::unpark(target, location));
    });

    execution(|execution| {
        trace!(?target, "unpark");
        execution.threads.unpark(target);
    });
}

/// A `SeqCst` fence's scheduling point: an operation on the SC total order S,
/// dependent with every other SC fence and SC access, so the search explores
/// the fence on either side of each.
pub(crate) fn branch_sc_fence(location: Location) {
    branch(|execution| {
        execution.threads.active_mut().operation = Some(object::Operation::sc_fence(location));
    });
}

/// Add an execution branch point.
fn branch<F, R>(f: F) -> R
where
    F: FnOnce(&mut Execution) -> R,
{
    let (ret, switch) = execution(|execution| {
        let ret = f(execution);
        let switch = execution.schedule();

        trace!(?switch, "branch");

        (ret, switch)
    });

    if switch {
        Scheduler::switch();
    }

    ret
}

fn synchronize<F, R>(f: F) -> R
where
    F: FnOnce(&mut Execution) -> R,
{
    execution(|execution| {
        execution.threads.begin_op();
        trace!("synchronize");
        f(execution)
    })
}

/// Yield the thread.
///
/// This enables concurrent algorithms that require other threads to make
/// progress.
///
/// Using this as a hint might be necessary to reduce the number of branches
/// being investigated by loom. This might be necessary when testing spin locks,
/// since each iteration constitutes a branch point which might easily cause a
/// combinatorial explosion.
///
/// Note that in loom, [`spin_loop`] and [`spin_loop_hint`] is an alias of this
/// function.
///
/// [`spin_loop`]: crate::hint::spin_loop
/// [`spin_loop_hint`]: crate::sync::atomic::spin_loop_hint
///
/// # Examples
///
/// Testing a raw spin lock under loom.
///
/// This is only provided as an example for when using [`spin_loop`] and
/// [`yield_now`] could be appropriate. Using a spin lock is almost always worse
/// than using a [`Mutex`] directly which spins internally before parking the
/// thread if contention is detected to save on system resources.
///
/// [`Mutex`]: std::sync::Mutex
/// [`spin_loop_hint`]: crate::sync::atomic::spin_loop_hint
/// [`spin_loop`]: crate::hint::spin_loop
///
/// ```no_run
/// use loom::sync::atomic::AtomicBool;
/// use loom::hint;
/// use loom::thread;
///
/// use std::sync::Arc;
/// use std::sync::atomic::Ordering::{Acquire, Relaxed, SeqCst};
///
/// struct Lock {
///     locked: AtomicBool,
/// }
///
/// impl Lock {
///     fn new() -> Self {
///         Lock {
///             locked: AtomicBool::new(false),
///         }
///     }
///
///     fn spin(&self) {
///         while self.locked.load(Relaxed) {
///             hint::spin_loop();
///         }
///     }
///
///     fn lock(&self) {
///         loop {
///             self.spin();
///
///             if self.locked.compare_exchange(false, true, Acquire, Relaxed).is_ok() {
///                 break;
///             }
///
///             thread::yield_now();
///         }
///     }
///
///     fn unlock(&self) {
///         self.locked.store(false, SeqCst);
///     }
/// }
///
/// # /*
/// #[test]
/// # */
/// fn test_concurrent_logic() {
///     loom::model(|| {
///         let v1 = Arc::new(Lock::new());
///         let v2 = v1.clone();
///
///         let t1 = thread::spawn(move || {
///             v2.lock();
///             // critical section.
///             v2.unlock();
///         });
///
///         v1.lock();
///         // critical section.
///         v1.unlock();
///
///         t1.join().unwrap();
///     });
/// }
/// ```
pub fn yield_now() {
    let switch = execution(|execution| {
        let thread = execution.threads.active_id();

        execution.threads.active_mut().set_yield();
        execution.threads.active_mut().operation = None;
        let switch = execution.schedule();

        trace!(?thread, ?switch, "yield_now");

        switch
    });

    if switch {
        Scheduler::switch();
    }
}

pub(crate) fn execution<F, R>(f: F) -> R
where
    F: FnOnce(&mut Execution) -> R,
{
    Scheduler::with_execution(f)
}

/// Run the active thread's `thread_local` destructors, on the thread,
/// inside the execution: each value is taken out under the execution
/// borrow, then dropped as ordinary user code, so tracked operations in
/// `Drop` impls are explored like any other op. Values drop in reverse
/// creation order. A destructor may access (or lazily create) other
/// locals: newly created values are destroyed in a later pass, and
/// re-accessing a destroyed key errors (`AccessError`), like `std`.
pub(crate) fn drop_locals() {
    loop {
        let local = execution(|execution| {
            let thread = execution.threads.active_id();
            let local = execution.threads.active_mut().take_next_local();

            trace!(?thread, dropping = local.is_some(), "drop_locals");

            local
        });

        match local {
            // Drop as user code of the still-live thread.
            Some(local) => drop(local),
            None => return,
        }
    }
}

pub fn thread_done() {
    // Locals are normally dropped earlier, before the join handle is
    // notified (see `thread::spawn_internal`); sweep up any created since,
    // e.g. by a `lazy_static` value's own teardown.
    drop_locals();

    execution(|execution| {
        let thread = execution.threads.active_id();

        execution.threads.active_mut().operation = None;
        execution.threads.active_mut().set_terminated();
        let switch = execution.schedule();
        trace!(?thread, ?switch, "thread_done: terminate");
    });
}

/// Tells loom to explore possible concurrent executions starting at this point.
pub fn explore() {
    execution(|execution| {
        execution.path.explore_state();
    })
}

/// Tells loom to stop exploring possible concurrent executions starting at this
/// point.
///
/// Exploration can be enabled again with `explore`.
pub fn stop_exploring() {
    execution(|execution| {
        execution.path.critical();
    })
}

/// Tells loom to stop exploring possible concurrent execution starting at this
/// point.
///
/// Unlike `stop_exploring`, exploration cannot be restarted by `explore`.
pub fn skip_branch() {
    execution(|execution| execution.path.skip_branch())
}

/// Explore both boolean outcomes at this point, resolved each way across
/// executions — a bounded two-valued path branch (reuses the `Spurious`
/// branch machinery, like `Notify` and `Condvar`). The primitive for a data
/// race whose winner the *environment* picks nondeterministically — e.g. a
/// kernel timer-vs-alert race whose outcome is expressed through neither
/// scheduling nor a modeled atomic, so loom has no other way to branch on it.
pub fn nondet_bool() -> bool {
    execution(|execution| execution.path.branch_spurious())
}
