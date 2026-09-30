use crate::rt;

use core::fmt;
use std::mem::ManuallyDrop;
use std::ops;
use std::ptr::{self, NonNull};
use std::sync::{LockResult, TryLockError, TryLockResult};

/// Mock implementation of `std::sync::RwLock`
///
/// Holds its data as [`Mutex`](crate::sync::Mutex) does: inline when built
/// at runtime, and when built in `const` evaluation a pristine value each
/// execution copies afresh on first use, so a `static` rwlock's data is fresh
/// in every execution and shared by none.
pub struct RwLock<T: ?Sized> {
    object: rt::Registration<rt::RwLock>,
    fresh: rt::Fresh,
    data: ManuallyDrop<std::sync::RwLock<T>>,
}

/// Mock implementation of `std::sync::RwLockReadGuard`
pub struct RwLockReadGuard<'a, T: ?Sized> {
    lock: &'a RwLock<T>,
    data: Option<std::sync::RwLockReadGuard<'a, T>>,
}

/// Mock implementation of `std::sync::rwLockWriteGuard`
pub struct RwLockWriteGuard<'a, T: ?Sized> {
    lock: &'a RwLock<T>,
    /// `data` is an Option so that the Drop impl can drop the std guard and release the std lock
    /// before releasing the loom mock lock, as that might cause another thread to acquire the lock
    data: Option<std::sync::RwLockWriteGuard<'a, T>>,
}

impl<T> RwLock<T> {
    /// Creates a new rwlock in an unlocked state ready for use. `const`, as
    /// `std`'s is; see `rt::Registration` for how the two contexts register.
    pub const fn new(data: T) -> RwLock<T> {
        RwLock {
            data: ManuallyDrop::new(std::sync::RwLock::new(data)),
            fresh: rt::Fresh::of::<std::sync::RwLock<T>>(),
            object: rt::Registration::new(),
        }
    }

    /// Consumes this `RwLock`, returning the underlying data.
    pub fn into_inner(self) -> LockResult<T> {
        let mut this = ManuallyDrop::new(self);

        let data = match this.object.minted().and_then(rt::take_instance) {
            // SAFETY: as `Mutex::into_inner`: the instance is a boxed
            // `std::sync::RwLock<T>`, now owned here, and the inline value a
            // template left undropped.
            Some(instance) => unsafe {
                *Box::from_raw(instance.into_raw().cast::<std::sync::RwLock<T>>().as_ptr())
            },
            // SAFETY: `this` is never used again, and its only other fields
            // own nothing.
            None => unsafe { ManuallyDrop::take(&mut this.data) },
        };

        Ok(data.into_inner().expect("loom::RwLock state corrupt"))
    }
}

impl<T: ?Sized> RwLock<T> {
    /// The data this execution works on, as `Mutex::data`.
    fn data(&self) -> &std::sync::RwLock<T> {
        if !self.object.is_deferred() {
            return &self.data;
        }

        let pristine: *const std::sync::RwLock<T> = &*self.data;

        // SAFETY: as `Mutex::data`: the inline value is never touched and is
        // what `fresh` was made for; the instance is its bitwise copy, living
        // until the execution ends or this rwlock drops.
        unsafe {
            let instance = rt::instance(
                self.object.id(),
                NonNull::new_unchecked(pristine as *mut u8),
                self.fresh,
            );
            &*ptr::from_raw_parts(instance.as_ptr().cast_const(), ptr::metadata(pristine))
        }
    }

    /// Locks this rwlock with shared read access, blocking the current
    /// thread until it can be acquired.
    ///
    /// The calling thread will be blocked until there are no more writers
    /// which hold the lock. There may be other readers currently inside the
    /// lock when this method returns. This method does not provide any
    /// guarantees with respect to the ordering of whether contentious readers
    /// or writers will acquire the lock first.
    #[track_caller]
    pub fn read(&self) -> LockResult<RwLockReadGuard<'_, T>> {
        self.object.get().acquire_read_lock(location!());

        Ok(RwLockReadGuard {
            lock: self,
            data: Some(self.data().try_read().expect("loom::RwLock state corrupt")),
        })
    }

    /// Attempts to acquire this rwlock with shared read access.
    ///
    /// If the access could not be granted at this time, then Err is returned.
    /// Otherwise, an RAII guard is returned which will release the shared
    /// access when it is dropped.
    ///
    /// This function does not block.
    #[track_caller]
    pub fn try_read(&self) -> TryLockResult<RwLockReadGuard<'_, T>> {
        if self.object.get().try_acquire_read_lock(location!()) {
            Ok(RwLockReadGuard {
                lock: self,
                data: Some(self.data().try_read().expect("loom::RwLock state corrupt")),
            })
        } else {
            Err(TryLockError::WouldBlock)
        }
    }

    /// Locks this rwlock with exclusive write access, blocking the current
    /// thread until it can be acquired.
    ///
    /// This function will not return while other writers or other readers
    /// currently have access to the lock.
    #[track_caller]
    pub fn write(&self) -> LockResult<RwLockWriteGuard<'_, T>> {
        self.object.get().acquire_write_lock(location!());

        Ok(RwLockWriteGuard {
            lock: self,
            data: Some(self.data().try_write().expect("loom::RwLock state corrupt")),
        })
    }

    /// Attempts to lock this rwlock with exclusive write access.
    ///
    /// If the lock could not be acquired at this time, then Err is returned.
    /// Otherwise, an RAII guard is returned which will release the lock when
    /// it is dropped.
    ///
    /// This function does not block.
    #[track_caller]
    pub fn try_write(&self) -> TryLockResult<RwLockWriteGuard<'_, T>> {
        if self.object.get().try_acquire_write_lock(location!()) {
            Ok(RwLockWriteGuard {
                lock: self,
                data: Some(self.data().try_write().expect("loom::RwLock state corrupt")),
            })
        } else {
            Err(TryLockError::WouldBlock)
        }
    }

    /// Returns a mutable reference to the underlying data.
    pub fn get_mut(&mut self) -> LockResult<&mut T> {
        let data: *const std::sync::RwLock<T> = self.data();

        // SAFETY: `&mut self` excludes every guard and every other reference
        // to the data, inline or instance.
        Ok(unsafe { &mut *data.cast_mut() }
            .get_mut()
            .expect("loom::RwLock state corrupt"))
    }
}

impl<T: ?Sized> Drop for RwLock<T> {
    fn drop(&mut self) {
        // As `Mutex`'s: a used `const`-built rwlock drops its instance, else
        // the inline value is the data.
        if let Some(instance) = self.object.minted().and_then(rt::take_instance) {
            drop(instance);
        } else {
            // SAFETY: dropped once, here, and never used again.
            unsafe { ManuallyDrop::drop(&mut self.data) }
        }
    }
}

impl<T: ?Sized + fmt::Debug> fmt::Debug for RwLock<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // A `const`-built rwlock's inline value is a template, never locked.
        if self.object.is_deferred() {
            f.debug_struct("RwLock").finish_non_exhaustive()
        } else {
            f.debug_struct("RwLock").field("data", &&*self.data).finish()
        }
    }
}

impl<T: Default> Default for RwLock<T> {
    /// Creates a `RwLock<T>`, with the `Default` value for T.
    fn default() -> Self {
        Self::new(Default::default())
    }
}

impl<T> From<T> for RwLock<T> {
    /// Creates a new rwlock in an unlocked state ready for use.
    /// This is equivalent to [`RwLock::new`].
    fn from(t: T) -> Self {
        Self::new(t)
    }
}

impl<T: fmt::Debug> fmt::Debug for RwLockReadGuard<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        (**self).fmt(f)
    }
}

impl<'a, T: ?Sized> ops::Deref for RwLockReadGuard<'a, T> {
    type Target = T;

    fn deref(&self) -> &T {
        self.data.as_ref().unwrap().deref()
    }
}

impl<'a, T: ?Sized + 'a> Drop for RwLockReadGuard<'a, T> {
    #[track_caller]
    fn drop(&mut self) {
        self.data = None;
        self.lock.object.get().release_read_lock(location!())
    }
}

impl<T: fmt::Debug> fmt::Debug for RwLockWriteGuard<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        (**self).fmt(f)
    }
}

impl<'a, T: ?Sized> ops::Deref for RwLockWriteGuard<'a, T> {
    type Target = T;

    fn deref(&self) -> &T {
        self.data.as_ref().unwrap().deref()
    }
}

impl<'a, T: ?Sized> ops::DerefMut for RwLockWriteGuard<'a, T> {
    fn deref_mut(&mut self) -> &mut T {
        self.data.as_mut().unwrap().deref_mut()
    }
}

impl<'a, T: ?Sized + 'a> Drop for RwLockWriteGuard<'a, T> {
    #[track_caller]
    fn drop(&mut self) {
        self.data = None;
        self.lock.object.get().release_write_lock(location!())
    }
}
