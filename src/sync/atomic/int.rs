use super::lane::{LaneU32Of64, LaneU32Of128, LaneU64Of128};
use super::Atomic;

use std::sync::atomic::Ordering;

#[rustfmt::skip] // rustfmt cannot properly format multi-line concat!.
macro_rules! atomic_int {
    ($name: ident, $int_type: ty) => {
        #[doc = concat!(
            " Mock implementation of `std::sync::atomic::", stringify!($name), "`.\n\n\
             NOTE: Unlike `std::sync::atomic::", stringify!($name), "`, \
             this type has a different in-memory representation than `",
             stringify!($int_type), "`.",
        )]
        #[derive(Debug)]
        pub struct $name(Atomic<$int_type>);

        impl $name {
            #[doc = concat!(" Creates a new instance of `", stringify!($name), "`.")]
            #[track_caller]
            pub fn new(v: $int_type) -> Self {
                Self(Atomic::new(v, location!()))
            }

            /// Get access to a mutable reference to the inner value.
            #[track_caller]
            pub fn with_mut<R>(&mut self, f: impl FnOnce(&mut $int_type) -> R) -> R {
                self.0.with_mut(f)
            }

            /// Load the value without any synchronization.
            ///
            /// # Safety
            ///
            /// An unsynchronized atomic load technically always has undefined behavior.
            /// However, if the atomic value is not currently visible by other threads,
            /// this *should* always be equivalent to a non-atomic load of an un-shared
            /// integer value.
            #[track_caller]
            pub unsafe fn unsync_load(&self) -> $int_type {
                self.0.unsync_load()
            }

            /// Consumes the atomic and returns the contained value.
            #[track_caller]
            pub fn into_inner(self) -> $int_type {
                // SAFETY: ownership guarantees that no other threads are concurrently
                // accessing the atomic value.
                unsafe { self.unsync_load() }
            }

            /// Loads a value from the atomic integer.
            #[track_caller]
            pub fn load(&self, order: Ordering) -> $int_type {
                self.0.load(order)
            }

            /// Sub-word load: reads only the bits under `mask` (other bits
            /// zero). The masked lane is independently coherent for separate
            /// masked stores — it may return a stale lane value while another
            /// lane is seen fresh — but **cell-coherent** against whole-cell
            /// ops: it can never read the lane from behind a multi-lane op
            /// the thread has already observed through any lane (single-copy
            /// atomicity; `rt::atomic` module docs). Models an aligned
            /// sub-word load inside a wider single-copy-atomic cell (spec
            /// carve-out #5). A load spanning more than one region still
            /// returns a single consistent (non-torn) snapshot.
            #[track_caller]
            pub fn load_masked(&self, mask: $int_type, order: Ordering) -> $int_type {
                self.0.load_masked(mask, order)
            }

            /// Stores a value into the atomic integer.
            #[track_caller]
            pub fn store(&self, val: $int_type, order: Ordering) {
                self.0.store(val, order)
            }

            /// Sub-word store: writes only the bits under `mask`, leaving the
            /// rest of the cell untouched (a *weak* store to that lane,
            /// independently coherent from the other lanes). Models an aligned
            /// sub-word store inside a wider single-copy-atomic cell (spec
            /// carve-out #5).
            #[track_caller]
            pub fn store_masked(&self, mask: $int_type, val: $int_type, order: Ordering) {
                self.0.store_masked(mask, val, order)
            }

            /// Stores a value into the atomic integer, returning the previous value.
            #[track_caller]
            pub fn swap(&self, val: $int_type, order: Ordering) -> $int_type {
                self.0.swap(val, order)
            }

            /// Stores a value into the atomic integer if the current value is the same as the `current` value.
            #[track_caller]
            pub fn compare_and_swap(
                &self,
                current: $int_type,
                new: $int_type,
                order: Ordering,
            ) -> $int_type {
                self.0.compare_and_swap(current, new, order)
            }

            /// Stores a value into the atomic if the current value is the same as the `current` value.
            #[track_caller]
            pub fn compare_exchange(
                &self,
                current: $int_type,
                new: $int_type,
                success: Ordering,
                failure: Ordering,
            ) -> Result<$int_type, $int_type> {
                self.0.compare_exchange(current, new, success, failure)
            }

            /// Stores a value into the atomic if the current value is the same as the current value.
            #[track_caller]
            pub fn compare_exchange_weak(
                &self,
                current: $int_type,
                new: $int_type,
                success: Ordering,
                failure: Ordering,
            ) -> Result<$int_type, $int_type> {
                self.compare_exchange(current, new, success, failure)
            }

            /// Adds to the current value, returning the previous value.
            #[track_caller]
            pub fn fetch_add(&self, val: $int_type, order: Ordering) -> $int_type {
                self.0.rmw(|v| v.wrapping_add(val), order)
            }

            /// Subtracts from the current value, returning the previous value.
            #[track_caller]
            pub fn fetch_sub(&self, val: $int_type, order: Ordering) -> $int_type {
                self.0.rmw(|v| v.wrapping_sub(val), order)
            }

            /// Bitwise "and" with the current value.
            #[track_caller]
            pub fn fetch_and(&self, val: $int_type, order: Ordering) -> $int_type {
                self.0.rmw(|v| v & val, order)
            }

            /// Bitwise "nand" with the current value.
            #[track_caller]
            pub fn fetch_nand(&self, val: $int_type, order: Ordering) -> $int_type {
                self.0.rmw(|v| !(v & val), order)
            }

            /// Bitwise "or" with the current value.
            #[track_caller]
            pub fn fetch_or(&self, val: $int_type, order: Ordering) -> $int_type {
                self.0.rmw(|v| v | val, order)
            }

            /// Bitwise "xor" with the current value.
            #[track_caller]
            pub fn fetch_xor(&self, val: $int_type, order: Ordering) -> $int_type {
                self.0.rmw(|v| v ^ val, order)
            }

            /// Stores the maximum of the current and provided value, returning the previous value
            #[track_caller]
            pub fn fetch_max(&self, val: $int_type, order: Ordering) -> $int_type {
                self.0.rmw(|v| v.max(val), order)
            }

            /// Stores the minimum of the current and provided value, returning the previous value
            #[track_caller]
            pub fn fetch_min(&self, val: $int_type, order: Ordering) -> $int_type {
                self.0.rmw(|v| v.min(val), order)
            }

            /// Single-step read-modify-write over only the bits under `mask`,
            /// returning the previous value of those bits (other bits zero).
            ///
            /// Unlike [`Self::fetch_update`] (a load followed by a CAS — two
            /// modelled steps that can interleave), this is **one** modelled
            /// atomic step reading the most recent value of the masked lane.
            /// It models a *sub-word* RMW on a wider single-copy-atomic cell
            /// (spec carve-out #5): the masked lane is independently coherent
            /// from the rest of the cell. `f` receives the current value (only
            /// the masked bits meaningful) and must depend only on those bits;
            /// its result's masked bits are written.
            #[track_caller]
            pub fn fetch_modify<F>(&self, mask: $int_type, f: F, order: Ordering) -> $int_type
            where
                F: FnOnce($int_type) -> $int_type,
            {
                self.0
                    .rmw_masked::<_, ()>(mask, order, order, |v| Ok(f(v)))
                    .unwrap()
            }

            /// Masked compare-exchange as **one** modelled atomic step: the
            /// compare consults only the bits under `mask`, and on success
            /// exactly those bits are replaced by `new`'s (all other bits
            /// preserved verbatim from the value at the linearization
            /// point). Models an aligned sub-word CAS inside a wider
            /// single-copy-atomic cell (spec carve-out #5). Returns the full
            /// previous value on both arms.
            #[track_caller]
            pub fn compare_exchange_masked(
                &self,
                mask: $int_type,
                current: $int_type,
                new: $int_type,
                success: Ordering,
                failure: Ordering,
            ) -> Result<$int_type, $int_type> {
                self.0.rmw_masked(mask, success, failure, |actual| {
                    if actual & mask == current & mask {
                        Ok((actual & !mask) | (new & mask))
                    } else {
                        Err(actual)
                    }
                })
            }

            /// Fetches the value, and applies a function to it that returns an optional new value.
            /// Returns a [`Result`] of [`Ok`]`(previous_value)` if the function returned
            /// [`Some`]`(_)`, else [`Err`]`(previous_value)`.
            #[track_caller]
            pub fn fetch_update<F>(
                &self,
                set_order: Ordering,
                fetch_order: Ordering,
                f: F,
            ) -> Result<$int_type, $int_type>
            where
                F: FnMut($int_type) -> Option<$int_type>,
            {
                self.0.fetch_update(set_order, fetch_order, f)
            }
        }

        impl Default for $name {
            fn default() -> Self {
                Self::new(Default::default())
            }
        }

        impl From<$int_type> for $name {
            fn from(v: $int_type) -> Self {
                Self::new(v)
            }
        }
    };
}

atomic_int!(AtomicU8, u8);
atomic_int!(AtomicU16, u16);
atomic_int!(AtomicU32, u32);
atomic_int!(AtomicUsize, usize);

// Typed sub-word lane views (see `lane`): the modelled counterpart of a
// production pointer-cast `&AtomicU32`/`&AtomicU64` into an aligned slice of
// a wider single-copy-atomic cell. Byte offsets index the cell's
// little-endian representation; constructors assert alignment and bounds.

#[cfg(target_has_atomic = "64")]
impl AtomicU64 {
    /// An aligned 32-bit lane view at `byte_offset` (0 or 4).
    #[track_caller]
    pub fn lane_u32(&self, byte_offset: usize) -> LaneU32Of64<'_> {
        LaneU32Of64::new(&self.0, byte_offset)
    }
}

impl AtomicU128 {
    /// An aligned 32-bit lane view at `byte_offset` (0, 4, 8, or 12).
    #[track_caller]
    pub fn lane_u32(&self, byte_offset: usize) -> LaneU32Of128<'_> {
        LaneU32Of128::new(&self.0, byte_offset)
    }

    /// An aligned 64-bit lane view at `byte_offset` (0 or 8).
    #[track_caller]
    pub fn lane_u64(&self, byte_offset: usize) -> LaneU64Of128<'_> {
        LaneU64Of128::new(&self.0, byte_offset)
    }
}

atomic_int!(AtomicI8, i8);
atomic_int!(AtomicI16, i16);
atomic_int!(AtomicI32, i32);
atomic_int!(AtomicIsize, isize);

#[cfg(target_has_atomic = "64")]
atomic_int!(AtomicU64, u64);

#[cfg(target_has_atomic = "64")]
atomic_int!(AtomicI64, i64);

// The 128-bit atomics are provided unconditionally: loom simulates atomic
// operations, so no hardware 128-bit atomic support (`target_has_atomic =
// "128"`) is required to model them.
atomic_int!(AtomicU128, u128);

atomic_int!(AtomicI128, i128);
