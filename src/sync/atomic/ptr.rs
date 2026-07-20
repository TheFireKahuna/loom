use super::Atomic;

use std::sync::atomic::Ordering;

#[rustfmt::skip] // rustfmt cannot properly format multi-line concat!.
macro_rules! atomic_ptr {
    // Constructed cells: the default backing caches its registration inline, so
    // it is wider than `*mut T` and must be built by a constructor.
    ($name: ident) => {
        atomic_ptr!(@ops [] $name, Atomic<*mut T>);

        impl<T> $name<T> {
            /// Creates a new instance of `AtomicPtr`.
            #[track_caller]
            pub fn new(v: *mut T) -> $name<T> {
                $name(Atomic::new(v, location!()))
            }

            /// Creates a null `AtomicPtr` in a `const` context.
            ///
            /// Null rather than a general `const_new(v: *mut T)` because casting
            /// a pointer to an integer is not permitted in a `const fn`, so the
            /// initial value could not be recorded. This is not a real
            /// restriction: a non-null pointer constant is not available in a
            /// `const` context either.
            ///
            /// Registration is deferred to first access; see
            /// [`AtomicUsize::const_new`](crate::sync::atomic::AtomicUsize::const_new)
            /// for what that changes and when to prefer [`new`](Self::new).
            pub const fn const_null() -> $name<T> {
                $name(Atomic::const_new(0))
            }
        }

        impl<T> Default for $name<T> {
            fn default() -> $name<T> {
                $name::new(std::ptr::null_mut())
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
                self.0.fmt(f)
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

            /// Consumes the atomic and returns the contained value.
            #[track_caller]
            pub fn into_inner(self) -> *mut T {
                // SAFETY: ownership guarantees that no other threads are
                // concurrently accessing the atomic value.
                unsafe { self.unsync_load() }
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
            #[track_caller]
            pub fn compare_exchange_weak(
                &self,
                current: *mut T,
                new: *mut T,
                success: Ordering,
                failure: Ordering,
            ) -> Result<*mut T, *mut T> {
                self.compare_exchange(current, new, success, failure)
            }

            /// Fetches the value, and applies a function to it that returns an
            /// optional new value. Returns a [`Result`] of
            /// [`Ok`]`(previous_value)` if the function returned [`Some`]`(_)`,
            /// else [`Err`]`(previous_value)`.
            #[track_caller]
            pub fn fetch_update<F>(
                &self,
                set_order: Ordering,
                fetch_order: Ordering,
                f: F,
            ) -> Result<*mut T, *mut T>
            where
                F: FnMut(*mut T) -> Option<*mut T>,
            {
                self.0.fetch_update(set_order, fetch_order, f)
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
