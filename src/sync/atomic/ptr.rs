use super::atomic::Atomic;

use std::sync::atomic::Ordering;

#[rustfmt::skip] // rustfmt cannot properly format multi-line concat!.
macro_rules! atomic_ptr {
    // Constructed cells: the default backing caches its registration inline, so
    // it is wider than `*mut T` and must be built by a constructor.
    ($name: ident) => {
        atomic_ptr!(@ops [] $name, Atomic<*mut T>);

        impl<T> $name<T> {
            /// Creates a new instance of `AtomicPtr`. `const`, as `core`'s is; see
            /// [`AtomicUsize::new`](crate::sync::atomic::AtomicUsize::new) for how
            /// the two contexts register. In `const` evaluation only a null
            /// pointer exists, and a non-null one is a compile error here.
            #[track_caller]
            pub const fn new(v: *mut T) -> $name<T> {
                let created = std::panic::Location::caller();
                $name(core::intrinsics::const_eval_select(
                    (v, created),
                    Atomic::<*mut T>::deferred_null,
                    Atomic::<*mut T>::eager_ptr,
                ))
            }

            /// Creates a new `AtomicPtr` initialized with a null pointer. `const`;
            /// registers as [`new`](Self::new) does.
            #[track_caller]
            #[must_use]
            pub const fn null() -> $name<T> {
                $name::new(std::ptr::null_mut())
            }

            /// Creates a null `AtomicPtr` with registration deferred to first
            /// access whatever the context; see
            /// [`AtomicUsize::const_new`](crate::sync::atomic::AtomicUsize::const_new).
            #[track_caller]
            pub const fn const_null() -> $name<T> {
                $name(Atomic::const_new(0))
            }

            /// Returns a mutable reference to the underlying pointer, as `std`'s
            /// does. The borrow is an exclusive access the model checks like
            /// [`with_mut`](Self::with_mut)'s.
            #[track_caller]
            pub fn get_mut(&mut self) -> &mut *mut T {
                self.0.get_mut()
            }

            /// Consumes the atomic and returns the contained value.
            #[track_caller]
            pub fn into_inner(self) -> *mut T {
                // SAFETY: ownership guarantees that no other threads are
                // concurrently accessing the atomic value.
                unsafe { self.unsync_load() }
            }
        }

        impl<T> Default for $name<T> {
            fn default() -> $name<T> {
                $name::new(std::ptr::null_mut())
            }
        }

        impl<T> From<*mut T> for $name<T> {
            /// Converts a `*mut T` into an `AtomicPtr<T>`.
            #[track_caller]
            fn from(p: *mut T) -> Self {
                Self::new(p)
            }
        }
    };

    // Materialized cells: the backing is nothing but layout, so the cell matches
    // `*mut T` and is reinterpretable from zeroed memory. No constructor —
    // `ZEROED` is the only way to name one, and it is the null pointer.
    (@materialized $name: ident, $backing: ty) => {
        atomic_ptr!(
            @ops [#[cfg_attr(feature = "zerocopy", derive(zerocopy::FromZeros))]]
            $name, $backing
        );

        impl<T> $name<T> {
            /// An unregistered `AtomicPtr` holding null — bit-identical to the
            /// all-zeroes pattern, so a zeroed region is a valid array of these.
            pub const ZEROED: Self = $name(<$backing>::zeroed());
        }

        impl<T> Default for $name<T> {
            fn default() -> $name<T> {
                $name::ZEROED
            }
        }
    };

    (@ops [$(#[$extra:meta])*] $name: ident, $backing: ty) => {
        #[doc = concat!(
            " Mock implementation of `std::sync::atomic::", stringify!($name), "`.",
        )]
        $(#[$extra])*
        #[repr(transparent)]
        pub struct $name<T>($backing);

        impl<T> std::fmt::Debug for $name<T> {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                self.0.fmt_peek(f, stringify!($name), std::fmt::Debug::fmt)
            }
        }

        impl<T> std::fmt::Pointer for $name<T> {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                self.0.fmt_peek(f, stringify!($name), std::fmt::Pointer::fmt)
            }
        }

        impl<T> $name<T> {
            /// Load the value without any synchronization.
            ///
            /// # Safety
            ///
            /// An unsynchronized atomic load technically always has undefined
            /// behavior. However, if the atomic value is not currently visible by
            /// other threads, this *should* always be equivalent to a non-atomic
            /// load of an un-shared `*mut T` value.
            pub unsafe fn unsync_load(&self) -> *mut T {
                self.0.unsync_load()
            }

            /// Get access to a mutable reference to the inner value.
            #[track_caller]
            pub fn with_mut<R>(&mut self, f: impl FnOnce(&mut *mut T) -> R) -> R {
                self.0.with_mut(f)
            }

            /// Loads a value from the pointer.
            #[track_caller]
            pub fn load(&self, order: Ordering) -> *mut T {
                self.0.load(order)
            }

            /// Stores a value into the pointer.
            #[track_caller]
            pub fn store(&self, val: *mut T, order: Ordering) {
                self.0.store(val, order)
            }

            /// Stores a value into the pointer, returning the previous value.
            #[track_caller]
            pub fn swap(&self, val: *mut T, order: Ordering) -> *mut T {
                self.0.swap(val, order)
            }

            /// Stores a value into the pointer if the current value is the same as
            /// the `current` value.
            #[track_caller]
            pub fn compare_and_swap(
                &self,
                current: *mut T,
                new: *mut T,
                order: Ordering,
            ) -> *mut T {
                self.0.compare_and_swap(current, new, order)
            }

            /// Stores a value into the pointer if the current value is the same as
            /// the `current` value.
            #[track_caller]
            pub fn compare_exchange(
                &self,
                current: *mut T,
                new: *mut T,
                success: Ordering,
                failure: Ordering,
            ) -> Result<*mut T, *mut T> {
                self.0.compare_exchange(current, new, success, failure)
            }

            /// Stores a value into the atomic if the current value is the same as
            /// the current value.
            ///
            /// May fail spuriously even when the current value equals `current`, as the
            /// target's does: only where it lowers to a single LL/SC attempt, or everywhere
            /// with `LOOM_SPURIOUS_WEAK_CAS=1` set.
            #[track_caller]
            pub fn compare_exchange_weak(
                &self,
                current: *mut T,
                new: *mut T,
                success: Ordering,
                failure: Ordering,
            ) -> Result<*mut T, *mut T> {
                self.0.compare_exchange_weak(current, new, success, failure)
            }

            /// Fetches the value, and applies a function to it that returns an
            /// optional new value. Returns a [`Result`] of
            /// [`Ok`]`(previous_value)` if the function returned [`Some`]`(_)`,
            /// else [`Err`]`(previous_value)`.
            #[track_caller]
            pub fn try_update<F>(
                &self,
                set_order: Ordering,
                fetch_order: Ordering,
                f: F,
            ) -> Result<*mut T, *mut T>
            where
                F: FnMut(*mut T) -> Option<*mut T>,
            {
                self.0.try_update(set_order, fetch_order, f)
            }

            /// An alias for [`Self::try_update`], deprecated as `core`'s is.
            #[track_caller]
            #[deprecated(
                since = "1.99.0",
                note = "renamed to `try_update` for consistency",
                suggestion = "try_update"
            )]
            pub fn fetch_update<F>(
                &self,
                set_order: Ordering,
                fetch_order: Ordering,
                f: F,
            ) -> Result<*mut T, *mut T>
            where
                F: FnMut(*mut T) -> Option<*mut T>,
            {
                self.try_update(set_order, fetch_order, f)
            }

            /// Fetches the value, and applies a function to it that returns a new
            /// value. The new value is stored and the old value is returned.
            ///
            /// [`Self::try_update`] with a function that always returns a new value,
            /// so it takes the same modelled steps: may call `f` more than once if
            /// the value changes between the load and the compare-exchange.
            #[track_caller]
            pub fn update(
                &self,
                set_order: Ordering,
                fetch_order: Ordering,
                mut f: impl FnMut(*mut T) -> *mut T,
            ) -> *mut T {
                match self.try_update(set_order, fetch_order, |p| Some(f(p))) {
                    Ok(prev) => prev,
                    Err(_) => unreachable!("`f` always supplies a new value"),
                }
            }

            // The address RMWs below are each one modelled step on the stored
            // pointer. A modelled cell carries its pointer as an exposed address,
            // so the stored and returned pointers keep the provenance of the one
            // stored, as `core`'s do, on the permissive-provenance terms every
            // loom pointer cell has.

            /// Offsets the pointer's address by adding `val` (in units of `T`),
            /// returning the previous pointer. Wraps, as `wrapping_add` does.
            #[track_caller]
            pub fn fetch_ptr_add(&self, val: usize, order: Ordering) -> *mut T {
                self.fetch_byte_add(val.wrapping_mul(size_of::<T>()), order)
            }

            /// Offsets the pointer's address by subtracting `val` (in units of
            /// `T`), returning the previous pointer. Wraps, as `wrapping_sub` does.
            #[track_caller]
            pub fn fetch_ptr_sub(&self, val: usize, order: Ordering) -> *mut T {
                self.fetch_byte_sub(val.wrapping_mul(size_of::<T>()), order)
            }

            /// Offsets the pointer's address by adding `val` bytes, returning the
            /// previous pointer.
            #[track_caller]
            pub fn fetch_byte_add(&self, val: usize, order: Ordering) -> *mut T {
                self.0.rmw(|p| p.wrapping_byte_add(val), order)
            }

            /// Offsets the pointer's address by subtracting `val` bytes, returning
            /// the previous pointer.
            #[track_caller]
            pub fn fetch_byte_sub(&self, val: usize, order: Ordering) -> *mut T {
                self.0.rmw(|p| p.wrapping_byte_sub(val), order)
            }

            /// Bitwise "or" of the pointer's address with `val`, returning the
            /// previous pointer.
            #[track_caller]
            pub fn fetch_or(&self, val: usize, order: Ordering) -> *mut T {
                self.0.rmw(|p| p.map_addr(|a| a | val), order)
            }

            /// Bitwise "and" of the pointer's address with `val`, returning the
            /// previous pointer.
            #[track_caller]
            pub fn fetch_and(&self, val: usize, order: Ordering) -> *mut T {
                self.0.rmw(|p| p.map_addr(|a| a & val), order)
            }

            /// Bitwise "xor" of the pointer's address with `val`, returning the
            /// previous pointer.
            #[track_caller]
            pub fn fetch_xor(&self, val: usize, order: Ordering) -> *mut T {
                self.0.rmw(|p| p.map_addr(|a| a ^ val), order)
            }
        }
    };
}

atomic_ptr!(AtomicPtr);

/// The materialized pointer cell — see [`crate::sync::atomic::materialized`].
pub mod materialized {
    use super::Atomic;
    use crate::rt;

    use std::sync::atomic::Ordering;

    #[cfg(target_pointer_width = "64")]
    atomic_ptr!(@materialized AtomicPtr, Atomic<*mut T, rt::Cell8>);
    #[cfg(target_pointer_width = "32")]
    atomic_ptr!(@materialized AtomicPtr, Atomic<*mut T, rt::Cell4>);

    const _: () = {
        use std::mem::{align_of, size_of};

        assert!(size_of::<AtomicPtr<u8>>() == size_of::<*mut u8>());
        assert!(align_of::<AtomicPtr<u8>>() == align_of::<*mut u8>());
    };
}
