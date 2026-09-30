use crate::rt;

use std::fmt;
use std::mem::ManuallyDrop;
use std::ops;
use std::ptr::{self, NonNull};
use std::sync::{LockResult, TryLockError, TryLockResult};

/// Mock implementation of `std::sync::Mutex`.
///
/// A mutex built at runtime holds its data inline. One built in `const`
/// evaluation — a `static` above all — holds only the pristine value it was
/// built with and never touches it: each execution works on a copy of its
/// own, made on first use and dropped when the mutex is, or at the end of the
/// execution with the lazy statics. So a `static` mutex's data is fresh in
/// every execution, and no two executions, or models, share it.
pub struct Mutex<T: ?Sized> {
    object: rt::Registration<rt::Mutex>,
    fresh: rt::Fresh,
    data: ManuallyDrop<std::sync::Mutex<T>>,
}

/// Mock implementation of `std::sync::MutexGuard`.
#[derive(Debug)]
pub struct MutexGuard<'a, T: ?Sized> {
    lock: &'a Mutex<T>,
    data: Option<std::sync::MutexGuard<'a, T>>,
}

impl<T> Mutex<T> {
    /// Creates a new mutex in an unlocked state ready for use. `const`, as
    /// `std`'s is; see `rt::Registration` for how the two contexts register.
    pub const fn new(data: T) -> Mutex<T> {
        Mutex {
            data: ManuallyDrop::new(std::sync::Mutex::new(data)),
            fresh: rt::Fresh::of::<std::sync::Mutex<T>>(),
            object: rt::Registration::new(),
        }
    }

    /// Consumes this mutex, returning the underlying data.
    pub fn into_inner(self) -> LockResult<T> {
        let mut this = ManuallyDrop::new(self);

        let data = match this.object.minted().and_then(rt::take_instance) {
            // SAFETY: the instance is a boxed `std::sync::Mutex<T>` (`fresh`
            // was made for it at construction, where `T` is this `T`), now
            // owned here; the pristine inline value is a template, left
            // undropped like any unused bitwise copy of a `const` value.
            Some(instance) => unsafe {
                *Box::from_raw(instance.into_raw().cast::<std::sync::Mutex<T>>().as_ptr())
            },
            // SAFETY: `this` is never used again, and its only other fields
            // own nothing.
            None => unsafe { ManuallyDrop::take(&mut this.data) },
        };

        Ok(data.into_inner().unwrap())
    }
}

impl<T: ?Sized> Mutex<T> {
    /// The data this execution works on: inline for a runtime-built mutex,
    /// this execution's instance for a `const`-built one.
    fn data(&self) -> &std::sync::Mutex<T> {
        if !self.object.is_deferred() {
            return &self.data;
        }

        let pristine: *const std::sync::Mutex<T> = &*self.data;

        // SAFETY: a deferred mutex never touches its inline data, which is
        // the value `fresh` was made for. The instance is a bitwise copy of
        // it, so it has the same type and metadata, and it lives until the
        // execution ends or this mutex is dropped, neither of which can
        // happen while `self` is borrowed within the execution.
        unsafe {
            let instance = rt::instance(
                self.object.id(),
                NonNull::new_unchecked(pristine as *mut u8),
                self.fresh,
            );
            &*ptr::from_raw_parts(instance.as_ptr().cast_const(), ptr::metadata(pristine))
        }
    }

    /// Acquires a mutex, blocking the current thread until it is able to do so.
    #[track_caller]
    pub fn lock(&self) -> LockResult<MutexGuard<'_, T>> {
        self.object.get().acquire_lock(location!());

        Ok(MutexGuard {
            lock: self,
            data: Some(self.data().lock().unwrap()),
        })
    }

    /// Attempts to acquire this lock.
    ///
    /// If the lock could not be acquired at this time, then `Err` is returned.
    /// Otherwise, an RAII guard is returned. The lock will be unlocked when the
    /// guard is dropped.
    ///
    /// This function does not block.
    #[track_caller]
    pub fn try_lock(&self) -> TryLockResult<MutexGuard<'_, T>> {
        if self.object.get().try_acquire_lock(location!()) {
            Ok(MutexGuard {
                lock: self,
                data: Some(self.data().lock().unwrap()),
            })
        } else {
            Err(TryLockError::WouldBlock)
        }
    }

    /// Returns a mutable reference to the underlying data.
    pub fn get_mut(&mut self) -> LockResult<&mut T> {
        let data: *const std::sync::Mutex<T> = self.data();

        // SAFETY: `&mut self` excludes every guard and every other reference
        // to the data, inline or instance.
        Ok(unsafe { &mut *data.cast_mut() }.get_mut().unwrap())
    }
}

impl<T: ?Sized> Drop for Mutex<T> {
    fn drop(&mut self) {
        // A `const`-built mutex used in this execution drops its instance;
        // the inline value is then a template. Otherwise the inline value is
        // the data.
        if let Some(instance) = self.object.minted().and_then(rt::take_instance) {
            drop(instance);
        } else {
            // SAFETY: dropped once, here, and never used again.
            unsafe { ManuallyDrop::drop(&mut self.data) }
        }
    }
}

impl<T: ?Sized + fmt::Debug> fmt::Debug for Mutex<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // A `const`-built mutex's inline value is a template: never locked,
        // not even to be shown.
        if self.object.is_deferred() {
            f.debug_struct("Mutex").finish_non_exhaustive()
        } else {
            f.debug_struct("Mutex").field("data", &&*self.data).finish()
        }
    }
}

impl<T: ?Sized + Default> Default for Mutex<T> {
    /// Creates a `Mutex<T>`, with the `Default` value for T.
    fn default() -> Self {
        Self::new(Default::default())
    }
}

impl<T> From<T> for Mutex<T> {
    /// Creates a new mutex in an unlocked state ready for use.
    /// This is equivalent to [`Mutex::new`].
    fn from(t: T) -> Self {
        Self::new(t)
    }
}

impl<'a, T: ?Sized + 'a> MutexGuard<'a, T> {
    pub(super) fn unborrow(&mut self) {
        self.data = None;
    }

    pub(super) fn reborrow(&mut self) {
        self.data = Some(self.lock.data().lock().unwrap());
    }

    pub(super) fn rt(&self) -> rt::Mutex {
        self.lock.object.get()
    }
}

impl<'a, T: ?Sized> ops::Deref for MutexGuard<'a, T> {
    type Target = T;

    fn deref(&self) -> &T {
        self.data.as_ref().unwrap().deref()
    }
}

impl<'a, T: ?Sized> ops::DerefMut for MutexGuard<'a, T> {
    fn deref_mut(&mut self) -> &mut T {
        self.data.as_mut().unwrap().deref_mut()
    }
}

impl<'a, T: ?Sized + 'a> Drop for MutexGuard<'a, T> {
    #[track_caller]
    fn drop(&mut self) {
        self.data = None;
        self.lock.object.get().unlock(location!());
    }
}
