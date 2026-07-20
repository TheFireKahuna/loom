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
//! The cells here carry no inline value and no cached registration — only an
//! identity word, minted on first access — so:
//!
//! - `size_of` and `align_of` match the modelled integer, and
//! - the all-zeroes bit pattern is a valid, unregistered cell holding zero.
//!
//! A zeroed region of memory is therefore a valid array of these, and the same
//! allocation path can run under the checker as in production.
//!
//! # Cost
//!
//! Every operation resolves its registration through a per-execution table
//! instead of reading a cached ref — there is nowhere to cache one. Prefer the
//! ordinary types unless a cell genuinely has to be materialized.
//!
//! # Genesis
//!
//! A materialized cell's initial value is zero by construction, and its
//! initialization is modelled as preceding the execution. Unlike
//! [`AtomicU64::new`](crate::sync::atomic::AtomicU64::new), it therefore does
//! not detect an unsynchronized publication *of the cell itself*.

use super::atomic::Atomic;
use crate::rt;

use std::sync::atomic::Ordering;

atomic_int!(@materialized AtomicU64, u64, Atomic<u64, rt::CellId>);
atomic_int!(@materialized AtomicI64, i64, Atomic<i64, rt::CellId>);
atomic_int!(@materialized AtomicUsize, usize, Atomic<usize, rt::CellId>);
atomic_int!(@materialized AtomicIsize, isize, Atomic<isize, rt::CellId>);
atomic_int!(@materialized AtomicU128, u128, Atomic<u128, rt::CellId16>);
atomic_int!(@materialized AtomicI128, i128, Atomic<i128, rt::CellId16>);

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
        AtomicU64 => u64,
        AtomicI64 => i64,
        AtomicUsize => usize,
        AtomicIsize => isize,
        AtomicU128 => u128,
        AtomicI128 => i128,
    }
};
