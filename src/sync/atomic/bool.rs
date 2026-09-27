use super::atomic::Atomic;

use std::sync::atomic::Ordering;

/// Mock implementation of `std::sync::atomic::AtomicBool`.
///
/// NOTE: Unlike `std::sync::atomic::AtomicBool`, this type has a different
/// in-memory representation than `bool`.
pub struct AtomicBool(Atomic<bool>);

impl AtomicBool {
    /// Creates a new instance of `AtomicBool`. `const`, as `core`'s is; see
    /// [`AtomicUsize::new`](crate::sync::atomic::AtomicUsize::new) for how the
    /// two contexts register.
    #[track_caller]
    pub const fn new(v: bool) -> AtomicBool {
        AtomicBool(Atomic::new(v as u128))
    }

    /// Creates a new instance of `AtomicBool` with registration deferred to
    /// first access whatever the context; see
    /// [`AtomicUsize::const_new`](crate::sync::atomic::AtomicUsize::const_new).
    pub const fn const_new(v: bool) -> AtomicBool {
        AtomicBool(Atomic::const_new(v as u128))
    }

    /// Load the value without any synchronization.
    ///
    /// # Safety
    ///
    /// An unsynchronized atomic load technically always has undefined behavior.
    /// However, if the atomic value is not currently visible by other threads,
    /// this *should* always be equivalent to a non-atomic load of an un-shared
    /// `bool` value.
    #[track_caller]
    pub unsafe fn unsync_load(&self) -> bool {
        self.0.unsync_load()
    }

    /// Consumes the atomic and returns the contained value.
    #[track_caller]
    pub fn into_inner(self) -> bool {
        // SAFETY: ownership guarantees that no other threads are concurrently
        // accessing the atomic value.
        unsafe { self.unsync_load() }
    }

    /// Loads a value from the atomic bool.
    #[track_caller]
    pub fn load(&self, order: Ordering) -> bool {
        self.0.load(order)
    }

    /// Stores a value into the atomic bool.
    #[track_caller]
    pub fn store(&self, val: bool, order: Ordering) {
        self.0.store(val, order)
    }

    /// Stores a value into the atomic bool, returning the previous value.
    #[track_caller]
    pub fn swap(&self, val: bool, order: Ordering) -> bool {
        self.0.swap(val, order)
    }

    /// Stores a value into the atomic bool if the current value is the same as the `current` value.
    #[track_caller]
    pub fn compare_and_swap(&self, current: bool, new: bool, order: Ordering) -> bool {
        self.0.compare_and_swap(current, new, order)
    }

    /// Stores a value into the atomic if the current value is the same as the `current` value.
    #[track_caller]
    pub fn compare_exchange(
        &self,
        current: bool,
        new: bool,
        success: Ordering,
        failure: Ordering,
    ) -> Result<bool, bool> {
        self.0.compare_exchange(current, new, success, failure)
    }

    /// Stores a value into the atomic if the current value is the same as the current value.
    #[track_caller]
    pub fn compare_exchange_weak(
        &self,
        current: bool,
        new: bool,
        success: Ordering,
        failure: Ordering,
    ) -> Result<bool, bool> {
        self.compare_exchange(current, new, success, failure)
    }

    /// Logical "and" with the current value.
    #[track_caller]
    pub fn fetch_and(&self, val: bool, order: Ordering) -> bool {
        self.0.rmw(|v| v & val, order)
    }

    /// Logical "nand" with the current value.
    #[track_caller]
    pub fn fetch_nand(&self, val: bool, order: Ordering) -> bool {
        self.0.rmw(|v| !(v & val), order)
    }

    /// Logical "or" with the current value.
    #[track_caller]
    pub fn fetch_or(&self, val: bool, order: Ordering) -> bool {
        self.0.rmw(|v| v | val, order)
    }

    /// Logical "xor" with the current value.
    #[track_caller]
    pub fn fetch_xor(&self, val: bool, order: Ordering) -> bool {
        self.0.rmw(|v| v ^ val, order)
    }

    /// Logical "not" with the current value: one read-modify-write, returning
    /// the previous value.
    #[track_caller]
    pub fn fetch_not(&self, order: Ordering) -> bool {
        self.fetch_xor(true, order)
    }

    /// Fetches the value, and applies a function to it that returns an optional new value. Returns
    /// a [`Result`] of [`Ok`]`(previous_value)` if the function returned [`Some`]`(_)`, else
    /// [`Err`]`(previous_value)`.
    #[track_caller]
    pub fn try_update<F>(
        &self,
        set_order: Ordering,
        fetch_order: Ordering,
        f: F,
    ) -> Result<bool, bool>
    where
        F: FnMut(bool) -> Option<bool>,
    {
        self.0.try_update(set_order, fetch_order, f)
    }

    /// Fetches the value, and applies a function to it that returns a new value.
    /// The new value is stored and the old value is returned.
    ///
    /// [`Self::try_update`] with a function that always returns a new value, so it
    /// takes the same modelled steps: may call `f` more than once if the value
    /// changes between the load and the compare-exchange.
    #[track_caller]
    pub fn update(
        &self,
        set_order: Ordering,
        fetch_order: Ordering,
        mut f: impl FnMut(bool) -> bool,
    ) -> bool {
        match self.try_update(set_order, fetch_order, |v| Some(f(v))) {
            Ok(prev) => prev,
            Err(_) => unreachable!("`f` always supplies a new value"),
        }
    }
}

impl std::fmt::Debug for AtomicBool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt_peek(f, "AtomicBool", std::fmt::Debug::fmt)
    }
}

impl Default for AtomicBool {
    fn default() -> AtomicBool {
        AtomicBool::new(Default::default())
    }
}

impl From<bool> for AtomicBool {
    fn from(b: bool) -> Self {
        Self::new(b)
    }
}
