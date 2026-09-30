//! Mock implementation of `std::sync::atomic`.

#[allow(clippy::module_inception)]
mod atomic;

mod bool;
pub use self::bool::AtomicBool;

// `atomic_int!` is shared with `materialized`, which is declared after it.
#[macro_use]
mod int;
pub use self::int::{AtomicI16, AtomicI32, AtomicI8, AtomicIsize};
pub use self::int::{AtomicU16, AtomicU32, AtomicU8, AtomicUsize};

mod lane;
pub use self::lane::{LaneU32Of64, LaneU32Of128, LaneU64Of128};

/// The constructed cells' backing marker.
///
/// Public only because it is the default type parameter of the lane views, so
/// it appears in their signature. Opaque: no constructor, no field, and nothing
/// to do with one but let it be inferred.
pub use crate::rt::ConstructedCell;

pub mod materialized;

#[cfg(target_has_atomic = "64")]
pub use self::int::{AtomicI64, AtomicU64};

// Unlike `std`, the 128-bit atomics are available on every target: loom
// simulates atomic operations, so hardware support is not required.
pub use self::int::{AtomicI128, AtomicU128};

mod ptr;
pub use self::ptr::AtomicPtr;

mod generic;
pub use self::generic::{Atomic, AtomicPrimitive};

#[doc(no_inline)]
pub use std::sync::atomic::Ordering;

/// Signals the processor that it is entering a busy-wait spin-loop.
///
/// For loom, this is an alias of [`spin_loop`] but is provided as a reflection
/// of the deprecated [`core::sync::atomic::spin_loop_hint`] function. See the
/// [`spin_loop`] documentation for more information on what effect using this
/// has on loom.
///
/// [`spin_loop`]: crate::hint::spin_loop
pub fn spin_loop_hint() {
    crate::hint::spin_loop();
}

/// An atomic fence.
pub fn fence(order: Ordering) {
    crate::rt::fence(order);
}

/// A compiler memory fence.
///
/// Orders memory operations only against a signal handler interrupting the
/// same thread. Loom runs no such handler, and a thread's own operations are
/// already modelled in program order, so this has no effect on the model; it
/// performs no modelled operation and is not a scheduling point.
///
/// # Panics
///
/// Panics if `order` is [`Relaxed`](Ordering::Relaxed), as `core`'s does.
#[inline]
#[track_caller]
pub fn compiler_fence(order: Ordering) {
    if let Ordering::Relaxed = order {
        panic!("there is no such thing as a relaxed fence");
    }
}
