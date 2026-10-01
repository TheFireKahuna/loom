//! Mocked versions of [`std::hint`] functions.

pub use crate::rt::spin_loop;

/// Why a monitor wait returned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MonitorWake {
    /// A store or read-modify-write to the watched location landed
    /// modification-order-after everything this thread had seen there.
    Stored,

    /// The wait's timeout fired first ([`monitor_wait_timeout`] only).
    TimedOut,
}

pub(crate) mod sealed {
    use super::MonitorWake;

    /// The arm half of [`Monitored`](super::Monitored), kept out of reach so
    /// that only loom's atomic cells and lane views can be watched.
    pub trait Arm {
        /// Arm on the watched bits and sleep, `timed` or not.
        fn arm(&self, timed: bool) -> MonitorWake;
    }
}

/// A location a hardware monitor can watch: one of loom's atomic cells, whose
/// whole width it watches, or a typed lane view of one, which watches the
/// lane's bits alone.
pub trait Monitored: sealed::Arm {}

/// Waits for a store to `location`, as a hardware monitor does
/// (`MONITORX`/`MWAITX`, `UMONITOR`/`UMWAIT`, `WFE`).
///
/// The monitor arms on what this thread has already observed of the location:
/// the store its last read there returned, or anything newer it has seen. The
/// thread then takes no step until some thread stores to the watched bits, or
/// read-modify-writes them, modification-order-after that. A store that had
/// already landed so when the wait began, between the read that found the
/// condition false and this call, ends the wait at once. So the poll loop a
/// fair scheduler would carry is written
///
/// ```
/// # loom::model(|| {
/// # use loom::sync::atomic::{AtomicU32, Ordering};
/// # let flag = AtomicU32::new(1);
/// while flag.load(Ordering::Acquire) == 0 {
///     loom::hint::monitor_wait(&flag);
/// }
/// # });
/// ```
///
/// and is explored as exactly that wait: no step is spent re-reading an
/// unchanged value, and no other thread's progress is assumed.
///
/// The wake is a coherence observation with no value and no synchronization:
/// the thread's reads of the location return nothing older than the store
/// that woke it, and it acquires nothing. Stores to bits outside the watched
/// ones never wake it; a lane view watches its lane alone.
///
/// The search treats the waiting thread's next step as a read of the watched
/// bits: it is dependent with every store and read-modify-write to them and
/// with nothing else. Where modification order has not yet placed a store
/// against what the waiter has seen, the search explores both placements,
/// before (it never wakes the waiter) and after (it does). A waiter that no
/// store can wake is a deadlock, and the failure report names the location.
///
/// An untimed wait never returns spuriously. Hardware monitors may, and a
/// caller that must survive that uses [`monitor_wait_timeout`], whose timeout
/// is the same observable return without a store.
#[track_caller]
pub fn monitor_wait<M: Monitored + ?Sized>(location: &M) -> MonitorWake {
    location.arm(false)
}

/// [`monitor_wait`] with a deadline: besides a store, the wait ends when its
/// timeout fires, which the search offers as it does every timed wait's.
#[track_caller]
pub fn monitor_wait_timeout<M: Monitored + ?Sized>(location: &M) -> MonitorWake {
    location.arm(true)
}

/// Informs the compiler that this point in the code is not reachable, enabling
/// further optimizations.
///
/// This is a mocked version of the standard library's
/// [`std::hint::unreachable_unchecked`]. Loom's wrapper of this function
/// unconditionally panics.
///
/// # Safety
///
/// Technically, this function is safe to call (unlike the standard library's
/// version), as it always panics rather than invoking UB. However, this
/// function is marked as `unsafe` because it's intended to be used as a
/// simulated version of [`std::hint::unreachable_unchecked`], which is unsafe.
///
/// See [the documentation for
/// `std::hint::unreachable_unchecked`](std::hint::unreachable_unchecked#Safety)
/// for safety details.
#[track_caller]
pub unsafe fn unreachable_unchecked() -> ! {
    unreachable!("unreachable_unchecked was reached!");
}
