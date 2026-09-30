use crate::rt::{self, Condvar, Futex, Mutex, RwLock};

use std::ptr::NonNull;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering::Relaxed;

/// Process-global identity source for `const`-built sync objects.
static NEXT_ID: AtomicU64 = AtomicU64::new(1);

/// A sync object's registration with the execution: made at construction when
/// that runs inside a model, and deferred to the object's first use in each
/// execution when it runs in `const` evaluation, where no execution exists.
///
/// A deferred object is keyed by an identity minted on first use, never by
/// address, so it may move; a `static` gets a fresh registration every
/// execution, as its state must. The data a deferred lock guards is fresh
/// every execution too (`instance`).
#[derive(Debug)]
pub(crate) struct Registration<R> {
    eager: Option<R>,
    id: AtomicU64,
}

/// A registrable sync object.
pub(crate) trait Registrable: Copy {
    fn create() -> Self;
    fn wrap(self) -> Deferred;
    fn unwrap(deferred: Deferred) -> Self;
}

/// A deferred object's registration in the current execution.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Deferred {
    Mutex(Mutex),
    RwLock(RwLock),
    Condvar(Condvar),
    Futex(Futex),
}

impl<R: Registrable> Registration<R> {
    /// Registers now at runtime, or on first use when evaluated in `const`.
    pub(crate) const fn new() -> Registration<R> {
        core::intrinsics::const_eval_select((), Self::deferred, Self::eager)
    }

    const fn deferred() -> Registration<R> {
        Registration {
            eager: None,
            id: AtomicU64::new(0),
        }
    }

    fn eager() -> Registration<R> {
        Registration {
            eager: Some(R::create()),
            id: AtomicU64::new(0),
        }
    }

    /// The object's registration in the current execution.
    #[inline]
    pub(crate) fn get(&self) -> R {
        match self.eager {
            Some(r) => r,
            None => self.resolve(),
        }
    }

    /// Whether the object was built in `const` evaluation.
    pub(crate) fn is_deferred(&self) -> bool {
        self.eager.is_none()
    }

    /// The deferred object's identity, minted on first use.
    pub(crate) fn id(&self) -> u64 {
        // Racing exploration workers share a `static`: settle on one id.
        match self.id.load(Relaxed) {
            0 => {
                let fresh = NEXT_ID.fetch_add(1, Relaxed);
                match self.id.compare_exchange(0, fresh, Relaxed, Relaxed) {
                    Ok(_) => fresh,
                    Err(won) => won,
                }
            }
            id => id,
        }
    }

    /// The deferred object's identity, if it has been used.
    pub(crate) fn minted(&self) -> Option<u64> {
        match self.id.load(Relaxed) {
            0 => None,
            id => Some(id),
        }
    }

    #[cold]
    fn resolve(&self) -> R {
        let id = self.id();
        if let Some(&d) = rt::execution(|execution| execution.deferred_objects.get(&id).copied()).as_ref() {
            return R::unwrap(d);
        }
        let r = R::create();
        rt::execution(|execution| execution.deferred_objects.insert(id, r.wrap()));
        r
    }
}

macro_rules! registrable {
    ($ty:ident, $create:expr) => {
        impl Registrable for $ty {
            fn create() -> Self {
                $create
            }

            fn wrap(self) -> Deferred {
                Deferred::$ty(self)
            }

            fn unwrap(deferred: Deferred) -> Self {
                match deferred {
                    Deferred::$ty(r) => r,
                    other => unreachable!("[loom internal bug] id bound to {other:?}"),
                }
            }
        }
    };
}

registrable!(Mutex, Mutex::new(true));
registrable!(RwLock, RwLock::new());
registrable!(Condvar, Condvar::new());
registrable!(Futex, Futex::new());

/// How a `const`-built lock's data enters an execution: `copy` makes a boxed
/// bitwise copy of the lock's pristine inline data, `drop` frees one. Both
/// are for the inline data's type at construction, where it is sized.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Fresh {
    copy: unsafe fn(NonNull<u8>) -> NonNull<u8>,
    drop: unsafe fn(NonNull<u8>),
}

impl Fresh {
    /// For inline data of type `L`.
    pub(crate) const fn of<L>() -> Fresh {
        /// # Safety
        /// `pristine` points to a valid `L` that nothing writes.
        unsafe fn copy<L>(pristine: NonNull<u8>) -> NonNull<u8> {
            // SAFETY: the caller's; the copy is a new owner of an `L` that
            // owns no heap, as every value `const` evaluation builds.
            let value = unsafe { pristine.cast::<L>().read() };
            NonNull::from(Box::leak(Box::new(value))).cast()
        }

        /// # Safety
        /// `instance` came from `copy::<L>` and is not used again.
        unsafe fn drop<L>(instance: NonNull<u8>) {
            // SAFETY: the caller's.
            std::mem::drop(unsafe { Box::from_raw(instance.cast::<L>().as_ptr()) });
        }

        Fresh {
            copy: copy::<L>,
            drop: drop::<L>,
        }
    }
}

/// A `const`-built lock's data in one execution: a bitwise copy of the
/// pristine value, owned by the execution. Dropped when its lock is, or at
/// the end of the execution with the lazy statics.
#[derive(Debug)]
pub(crate) struct Instance {
    ptr: NonNull<u8>,
    drop: unsafe fn(NonNull<u8>),
}

impl Instance {
    /// The data, as the `L` the lock's `Fresh` was made for; the caller takes
    /// ownership.
    pub(crate) fn into_raw(self) -> NonNull<u8> {
        let ptr = self.ptr;
        std::mem::forget(self);
        ptr
    }
}

impl Drop for Instance {
    fn drop(&mut self) {
        // SAFETY: `ptr` came from the `copy` paired with this `drop`, and the
        // instance is its only owner.
        unsafe { (self.drop)(self.ptr) }
    }
}

/// The current execution's instance of the deferred lock `id`'s data, copied
/// on first touch from `pristine` with `fresh`.
///
/// # Safety
/// `pristine` points to the lock's inline data, which `fresh` was made for
/// and which nothing ever writes.
pub(crate) unsafe fn instance(id: u64, pristine: NonNull<u8>, fresh: Fresh) -> NonNull<u8> {
    rt::execution(|execution| {
        if let Some((_, instance)) = execution.lock_data.iter().find(|(k, _)| *k == id) {
            return instance.ptr;
        }

        let instance = Instance {
            // SAFETY: the caller's.
            ptr: unsafe { (fresh.copy)(pristine) },
            drop: fresh.drop,
        };
        let ptr = instance.ptr;
        execution.lock_data.push((id, instance));
        ptr
    })
}

/// Take the deferred lock `id`'s instance out of the current execution, if
/// there is an execution and the lock has an instance in it.
pub(crate) fn take_instance(id: u64) -> Option<Instance> {
    rt::Scheduler::try_with_execution(|execution| {
        let pos = execution.lock_data.iter().position(|(k, _)| *k == id)?;
        Some(execution.lock_data.remove(pos).1)
    })
    .flatten()
}
