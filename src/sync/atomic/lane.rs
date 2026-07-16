//! Typed sub-word lane views into a wider atomic cell.
//!
//! A lane is the modelled counterpart of a production sub-word atomic view —
//! an `&AtomicU32`/`&AtomicU64` pointer-cast into an aligned slice of a
//! 16-byte (or 8-byte) single-copy-atomic cell (the ntlib carve-out #5 shape:
//! x86-64 `+cx16` / AArch64 LSE2 guarantee every aligned power-of-two
//! sub-access of the cell is individually atomic and coherent with the wide
//! ops). Each lane method is **one** modelled linearization point on exactly
//! the lane's bits, built on the cell's masked ops:
//!
//! - stores/RMWs touch only the lane's region — the untouched bits are
//!   physically preserved, and disjoint-lane ops keep their DPOR independence
//!   (mask-intersection pruning);
//! - loads are **cell-coherent**: independently stale per lane for separate
//!   masked stores, but never behind a whole-cell op the loading thread has
//!   already observed through any lane (`rt::atomic` module docs — the
//!   single-copy-atomicity claim).
//!
//! Byte offsets are offsets into the cell's **little-endian** in-memory
//! representation — offset `k` names value bits `8k..8k+width` — matching a
//! pointer-cast lane on the LE targets the production carve-out runs on.
//! Constructors assert lane-aligned, in-bounds offsets.

use super::Atomic;

use std::sync::atomic::Ordering;

macro_rules! lane_type {
    (
        $(#[$m:meta])*
        $name:ident, $parent:ty, $lane:ty
    ) => {
        $(#[$m])*
        #[derive(Debug)]
        pub struct $name<'a> {
            cell: &'a Atomic<$parent>,
            shift: u32,
        }

        impl<'a> $name<'a> {
            /// One lane's bytes.
            const LANE_BYTES: usize = std::mem::size_of::<$lane>();
            /// The whole cell's bytes.
            const CELL_BYTES: usize = std::mem::size_of::<$parent>();

            pub(crate) fn new(cell: &'a Atomic<$parent>, byte_offset: usize) -> Self {
                assert!(
                    byte_offset % Self::LANE_BYTES == 0
                        && byte_offset + Self::LANE_BYTES <= Self::CELL_BYTES,
                    "misaligned or out-of-bounds lane: byte offset {} for a {}-byte \
                     lane of a {}-byte cell",
                    byte_offset,
                    Self::LANE_BYTES,
                    Self::CELL_BYTES,
                );
                $name {
                    cell,
                    shift: (byte_offset * 8) as u32,
                }
            }

            /// The lane's bits within the cell.
            fn mask(&self) -> $parent {
                (<$lane>::MAX as $parent) << self.shift
            }

            fn to_cell(&self, v: $lane) -> $parent {
                (v as $parent) << self.shift
            }

            fn from_cell(&self, v: $parent) -> $lane {
                (v >> self.shift) as $lane
            }

            /// One-step masked RMW returning the lane's previous value.
            #[track_caller]
            fn rmw(&self, order: Ordering, f: impl FnOnce($lane) -> $lane) -> $lane {
                let mask = self.mask();
                let prior = self
                    .cell
                    .rmw_masked::<_, std::convert::Infallible>(mask, order, order, |cur| {
                        Ok((cur & !mask) | (self.to_cell(f(self.from_cell(cur))) & mask))
                    })
                    .unwrap();
                self.from_cell(prior)
            }

            /// Loads the lane's value (cell-coherent — module docs).
            #[track_caller]
            pub fn load(&self, order: Ordering) -> $lane {
                self.from_cell(self.cell.load_masked(self.mask(), order))
            }

            /// Stores into the lane, preserving every other bit of the cell.
            #[track_caller]
            pub fn store(&self, val: $lane, order: Ordering) {
                self.cell.store_masked(self.mask(), self.to_cell(val), order)
            }

            /// Stores into the lane, returning the previous lane value.
            #[track_caller]
            pub fn swap(&self, val: $lane, order: Ordering) -> $lane {
                self.rmw(order, |_| val)
            }

            /// Lane compare-exchange as one modelled step: the compare consults
            /// only the lane's bits; on success exactly those bits are
            /// replaced, every other bit preserved verbatim from the value at
            /// the linearization point.
            #[track_caller]
            pub fn compare_exchange(
                &self,
                current: $lane,
                new: $lane,
                success: Ordering,
                failure: Ordering,
            ) -> Result<$lane, $lane> {
                let mask = self.mask();
                self.cell
                    .rmw_masked(mask, success, failure, |cur| {
                        if cur & mask == self.to_cell(current) {
                            Ok((cur & !mask) | (self.to_cell(new) & mask))
                        } else {
                            Err(self.from_cell(cur))
                        }
                    })
                    .map(|prior| self.from_cell(prior))
            }

            /// [`Self::compare_exchange`] (loom models the weak form as the
            /// strong one, like the full-width atomics).
            #[track_caller]
            pub fn compare_exchange_weak(
                &self,
                current: $lane,
                new: $lane,
                success: Ordering,
                failure: Ordering,
            ) -> Result<$lane, $lane> {
                self.compare_exchange(current, new, success, failure)
            }

            /// Adds to the lane (wrapping at the lane's width), returning the
            /// previous lane value.
            #[track_caller]
            pub fn fetch_add(&self, val: $lane, order: Ordering) -> $lane {
                self.rmw(order, |v| v.wrapping_add(val))
            }

            /// Subtracts from the lane (wrapping), returning the previous
            /// lane value.
            #[track_caller]
            pub fn fetch_sub(&self, val: $lane, order: Ordering) -> $lane {
                self.rmw(order, |v| v.wrapping_sub(val))
            }

            /// Bitwise "and" on the lane, returning the previous lane value.
            #[track_caller]
            pub fn fetch_and(&self, val: $lane, order: Ordering) -> $lane {
                self.rmw(order, |v| v & val)
            }

            /// Bitwise "or" on the lane, returning the previous lane value.
            #[track_caller]
            pub fn fetch_or(&self, val: $lane, order: Ordering) -> $lane {
                self.rmw(order, |v| v | val)
            }

            /// Bitwise "xor" on the lane, returning the previous lane value.
            #[track_caller]
            pub fn fetch_xor(&self, val: $lane, order: Ordering) -> $lane {
                self.rmw(order, |v| v ^ val)
            }

            /// Bitwise "nand" on the lane, returning the previous lane value.
            #[track_caller]
            pub fn fetch_nand(&self, val: $lane, order: Ordering) -> $lane {
                self.rmw(order, |v| !(v & val))
            }

            /// Stores the maximum of the lane and `val`, returning the
            /// previous lane value.
            #[track_caller]
            pub fn fetch_max(&self, val: $lane, order: Ordering) -> $lane {
                self.rmw(order, |v| v.max(val))
            }

            /// Stores the minimum of the lane and `val`, returning the
            /// previous lane value.
            #[track_caller]
            pub fn fetch_min(&self, val: $lane, order: Ordering) -> $lane {
                self.rmw(order, |v| v.min(val))
            }
        }
    };
}

lane_type!(
    /// An aligned 32-bit lane of an [`AtomicU128`](super::AtomicU128) — see
    /// the module docs.
    LaneU32Of128,
    u128,
    u32
);

lane_type!(
    /// An aligned 64-bit lane of an [`AtomicU128`](super::AtomicU128) — see
    /// the module docs.
    LaneU64Of128,
    u128,
    u64
);

lane_type!(
    /// An aligned 32-bit lane of an [`AtomicU64`](super::AtomicU64) — see
    /// the module docs.
    LaneU32Of64,
    u64,
    u32
);
