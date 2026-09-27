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
//! the price, and it buys a second thing: [`publish`] records the commit as the
//! cells' genesis, so a thread that reaches a cell without synchronizing-with
//! any commit of its memory is reported, as
//! [`AtomicU64::new`](crate::sync::atomic::AtomicU64::new) reports an
//! unsynchronized access to a constructed cell.
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

/// Model the `MEM_COMMIT` verb over `len` bytes at `ptr`: the calling thread
/// has committed the range, and every byte of it reads zero until written.
///
/// A materialized cell has no identity outside committed memory, so its first
/// access panics unless a commit covers it. The commit is also the cells'
/// genesis: an access must happen-after some commit of the cell's memory — the
/// accessing thread's own, or one it synchronized with through the program's
/// own release — or it is reported as a causality violation.
///
/// A commit synchronizes with nothing. Committing a range that is already
/// committed changes nothing in it — that is what an idempotent commit is — and
/// lets the caller use the memory, without ordering it after whoever committed
/// first. To discard the contents of live memory, use [`reset`]. Commits do not
/// outlive an execution.
///
/// This describes memory to the model; it neither reads nor writes it, and
/// `ptr` need not be dereferenceable.
#[track_caller]
pub fn publish(ptr: *const u8, len: usize) {
    rt::publish(ptr as usize, len)
}

/// Declare that the calling thread has bulk-zeroed `len` bytes at `ptr` while
/// owning them exclusively.
///
/// A materialized cell keeps its value in the model, not in the bytes it
/// occupies, so a `memset` over the raw memory is invisible: without this call
/// the cells keep whatever they last held. A pool that recycles a record by
/// zeroing it before any typed reference exists has to say so here.
///
/// The write is non-atomic, and checked as such — a peer that has not
/// synchronized-with the caller is reported, exactly as it would be for
/// `with_mut`. Cells in the range that were never registered are already zero,
/// so the range need not have been touched.
#[track_caller]
pub fn zero_exclusive(ptr: *mut u8, len: usize) {
    rt::zero_exclusive(ptr as usize, len, location!())
}

/// Model the `MEM_RESET` verb over `len` bytes at `ptr`, rounded out to whole
/// 4 KiB pages as the kernel rounds it: the contents are no longer of interest,
/// but the mapping stays.
///
/// A reset promises no zeros. It marks the pages clean, and until a page is
/// written again the kernel may discard it at any moment, after which every
/// byte of it reads zero; a write dirties the page and cancels the discard. So
/// each page reads its old contents until a discard that may never come, and
/// zero from then on — even to the thread that reset it — and a thread that has
/// seen the discard through one cell sees it through every cell of the page.
/// The checker explores every such point per page, so code that assumes a
/// reset page reads zero is reported by the execution where it does not.
///
/// Use this, not [`zero_exclusive`], whenever a reader may legally still be
/// walking the span: a concurrent atomic access is admissible and sees old or
/// zero. A concurrent non-atomic access is reported, as against a store. A
/// discard synchronizes with nothing.
#[track_caller]
pub fn reset(ptr: *mut u8, len: usize) {
    rt::reset(ptr as usize, len, location!())
}

/// Model the `MEM_DECOMMIT` verb over `len` bytes at `ptr`: the mapping itself
/// goes, not merely its contents.
///
/// The inverse of [`publish`], and an assertion that no thread can still reach
/// the range, checked from both sides. It is a non-atomic write to every cell
/// in the range, so an earlier access by a thread the decommit does not
/// happen-after is reported. And the cells lose their registrations and the
/// range leaves the committed set, so a **later access panics** as a use after
/// decommit — decommitted memory faults on real hardware, and a
/// stale-but-plausible value here would pass exactly the executions the
/// caller's unreachability argument exists to forbid.
///
/// Use [`reset`] instead wherever a reader may legally still be in the span.
/// A later [`publish`] re-registers the range's cells at zero, which is what
/// re-committing decommitted pages hands back.
///
/// This describes memory to the model; it neither reads nor writes it.
#[track_caller]
pub fn unpublish(ptr: *const u8, len: usize) {
    rt::unpublish(ptr as usize, len)
}

pub use super::ptr::materialized::AtomicPtr;

/// The cells' backing markers, one per width.
///
/// Public only because a lane view names its cell's backing in its own type
/// (`LaneU64Of128<'_, Cell16>`). They are opaque: no constructor, no field, and
/// nothing to do with one but let it be inferred.
pub use crate::rt::{Cell1, Cell16, Cell2, Cell4, Cell8};

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

// Typed sub-word lane views, as on the constructed cells. The view types are
// generic over the backing, so these are the same views — only the cell they
// borrow differs.

impl AtomicU64 {
    /// An aligned 32-bit lane view at `byte_offset` (0 or 4).
    #[track_caller]
    pub fn lane_u32(&self, byte_offset: usize) -> super::LaneU32Of64<'_, rt::Cell8> {
        super::LaneU32Of64::new(&self.0, byte_offset)
    }
}

impl AtomicU128 {
    /// An aligned 32-bit lane view at `byte_offset` (0, 4, 8, or 12).
    #[track_caller]
    pub fn lane_u32(&self, byte_offset: usize) -> super::LaneU32Of128<'_, rt::Cell16> {
        super::LaneU32Of128::new(&self.0, byte_offset)
    }

    /// An aligned 64-bit lane view at `byte_offset` (0 or 8).
    #[track_caller]
    pub fn lane_u64(&self, byte_offset: usize) -> super::LaneU64Of128<'_, rt::Cell16> {
        super::LaneU64Of128::new(&self.0, byte_offset)
    }
}

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
