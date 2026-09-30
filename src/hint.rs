//! Mocked versions of [`std::hint`] functions.

/// Signals the processor that it is entering a busy-wait spin-loop.
///
/// Loom models it as what a spin loop is: a wait for another thread's
/// progress. The spinner is not scheduled again while any other thread can
/// run — the switch away costs no preemption — and a store it had already
/// seen superseded is not returned to it again, so a loop that spins until a
/// peer's store lands always ends. This is the primitive for every poll loop
/// in a model; [`thread::yield_now`](crate::thread::yield_now) is a plain
/// scheduling point and gives a poll loop no progress.
///
/// Under [`Builder::preemption_bound`](crate::model::Builder::preemption_bound)
/// the switch after a spin is free, but once an execution has spent the whole
/// bound the search explores no other thread at that switch than the one the
/// scheduler picks, which keeps a spin loop's state space exhaustible.
pub fn spin_loop() {
    crate::rt::spin_loop();
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
