//! Mock implementation of `std::sync`.

mod arc;
pub mod atomic;
mod barrier;
mod condvar;
pub mod mpsc;
mod mutex;
mod notify;
mod once;
mod rwlock;

pub use self::arc::{Arc, Weak};
pub use self::barrier::{Barrier, BarrierWaitResult};
pub use self::condvar::{Condvar, WaitTimeoutResult};
pub use self::mutex::{Mutex, MutexGuard};
pub use self::notify::Notify;
pub use self::once::{Once, OnceState};
pub use self::rwlock::{RwLock, RwLockReadGuard, RwLockWriteGuard};

#[doc(no_inline)]
pub use std::sync::{LockResult, PoisonError, TryLockError, TryLockResult};
