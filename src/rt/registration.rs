use crate::rt::{self, Condvar, Mutex, RwLock};

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
/// execution, as its state must. The data a `static` lock guards is the
/// caller's and is not reset.
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

    #[cold]
    fn resolve(&self) -> R {
        // Racing exploration workers share a `static`: settle on one id.
        let id = match self.id.load(Relaxed) {
            0 => {
                let fresh = NEXT_ID.fetch_add(1, Relaxed);
                match self.id.compare_exchange(0, fresh, Relaxed, Relaxed) {
                    Ok(_) => fresh,
                    Err(won) => won,
                }
            }
            id => id,
        };
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
