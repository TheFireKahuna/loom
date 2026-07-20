//! Atomics laid out exactly like the integer they model.
//!
//! An ordinary [`loom::sync::atomic`](crate::sync::atomic) cell caches its
//! model registration inline, so it is wider than the integer it stands for and
//! can only come from a constructor. That is invisible to most code, but it is
//! fatal to anything that obtains its cells by *reinterpreting memory* rather
//! than by constructing them — a pool carving fixed-size slots out of
//! demand-committed pages, say. Such code has to be restructured for the
//! checker build, and then the checker is verifying a structure that is not the
//! one that ships.
//!
//! The cells here carry no value and no identity word — nothing but the
//! modelled integer's size and alignment — so:
//!
//! - `size_of` and `align_of` match it exactly, at every width, and
//! - the all-zeroes pattern is a valid, unregistered cell holding zero.
//!
//! A zeroed region of memory is therefore a valid array of these, and the same
//! allocation path can run under the checker as in production.
//!
//! # Declaring the memory is mandatory
//!
//! A materialized cell takes its identity from **where it lives**, so the
//! memory holding it must be declared with [`publish`]. A cell outside every
//! published range has no identity and its first access panics.
//!
//! That is deliberate. Identity-by-position is what lets these cells exist at
//! `u8` and `u16` widths at all — an identity *word* needs bits the narrow
//! widths do not have, and a minted id would accumulate without bound because a
//! materialized cell re-registers every execution. Requiring the declaration is
//! the price, and it buys a second thing: [`publish`] records the publishing
//! thread's causality as the cells' genesis, so a reader that reaches a cell
//! without synchronizing-with the publication is reported, exactly as
//! [`AtomicU64::new`](crate::sync::atomic::AtomicU64::new) reports it for a
//! constructed cell.
//!
//! # Cost
//!
//! Every operation resolves its registration through a per-execution table
//! instead of reading a cached ref — there is nowhere to cache one. Measured at
//! 1.05–1.10x a constructed cell (`examples/materialized_cost.rs`). Prefer the
//! ordinary types unless a cell genuinely has to be materialized.

use super::atomic::Atomic;
use crate::rt;

use std::sync::atomic::Ordering;

/// Declare that the calling thread has published `len` bytes of zeroed memory
/// at `ptr`.
///
/// Cells materialized inside the range take this thread's causality as their
/// genesis, so an access by a thread that has not synchronized-with the
/// publication is reported as a causality violation — the check a constructed
/// cell gets from [`AtomicU64::new`](crate::sync::atomic::AtomicU64::new), and
/// the reason to call this rather than rely on the default.
///
/// Without a declaration, a materialized cell is modelled as having preceded
/// the execution. That is accurate for a `static`, and a silent
/// under-approximation for memory a thread handed out at runtime: no access to
/// it can ever be reported as unsynchronized.
///
/// Declaring a range again supersedes the earlier declaration for those
/// addresses, which is what a decommit-then-recommit cycle needs. Declarations
/// do not outlive an execution.
///
/// This describes memory to the model; it neither reads nor writes it, and
/// `ptr` need not be dereferenceable.
#[track_caller]
pub fn publish(ptr: *const u8, len: usize) {
    rt::publish(ptr as usize, len)
}

atomic_int!(@materialized AtomicU8, u8, Atomic<u8, rt::Cell1>);
atomic_int!(@materialized AtomicI8, i8, Atomic<i8, rt::Cell1>);
atomic_int!(@materialized AtomicU16, u16, Atomic<u16, rt::Cell2>);
atomic_int!(@materialized AtomicI16, i16, Atomic<i16, rt::Cell2>);
atomic_int!(@materialized AtomicU32, u32, Atomic<u32, rt::Cell4>);
atomic_int!(@materialized AtomicI32, i32, Atomic<i32, rt::Cell4>);
atomic_int!(@materialized AtomicU64, u64, Atomic<u64, rt::Cell8>);
atomic_int!(@materialized AtomicI64, i64, Atomic<i64, rt::Cell8>);
atomic_int!(@materialized AtomicU128, u128, Atomic<u128, rt::Cell16>);
atomic_int!(@materialized AtomicI128, i128, Atomic<i128, rt::Cell16>);

#[cfg(target_pointer_width = "64")]
atomic_int!(@materialized AtomicUsize, usize, Atomic<usize, rt::Cell8>);
#[cfg(target_pointer_width = "64")]
atomic_int!(@materialized AtomicIsize, isize, Atomic<isize, rt::Cell8>);
#[cfg(target_pointer_width = "32")]
atomic_int!(@materialized AtomicUsize, usize, Atomic<usize, rt::Cell4>);
#[cfg(target_pointer_width = "32")]
atomic_int!(@materialized AtomicIsize, isize, Atomic<isize, rt::Cell4>);

// The property this module exists to provide. Asserted rather than documented:
// if it stops holding, a cell can no longer be reinterpreted from a zeroed
// region and the module is pointless.
const _: () = {
    use std::mem::{align_of, size_of};

    macro_rules! same_layout {
        ($($cell:ty => $int:ty),* $(,)?) => {$(
            assert!(size_of::<$cell>() == size_of::<$int>());
            assert!(align_of::<$cell>() == align_of::<$int>());
        )*};
    }

    same_layout! {
        AtomicU8 => u8,
        AtomicI8 => i8,
        AtomicU16 => u16,
        AtomicI16 => i16,
        AtomicU32 => u32,
        AtomicI32 => i32,
        AtomicU64 => u64,
        AtomicI64 => i64,
        AtomicU128 => u128,
        AtomicI128 => i128,
        AtomicUsize => usize,
        AtomicIsize => isize,
    }
};
