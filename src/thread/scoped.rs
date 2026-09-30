//! Mock implementation of `std::thread::scope`.

use super::{init_current, Builder, Thread, ThreadId};
use crate::rt::{self, Location};
use crate::sync::atomic::AtomicUsize;

use std::marker::PhantomData;
use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};
use std::sync::atomic::Ordering::{Acquire, Relaxed, Release};
use std::sync::{Arc, Mutex};
use std::{fmt, io};

/// Mock implementation of `std::thread::Scope`.
///
/// A scope to spawn scoped threads in. See [`scope`] for details.
pub struct Scope<'scope, 'env: 'scope> {
    data: Arc<ScopeData>,
    scope: PhantomData<&'scope mut &'scope ()>,
    env: PhantomData<&'env mut &'env ()>,
}

/// Mock implementation of `std::thread::ScopedJoinHandle`.
///
/// An owned permission to join on a scoped thread (block on its termination).
pub struct ScopedJoinHandle<'scope, T> {
    packet: Arc<Packet<'scope, T>>,
    thread: Thread,
}

/// What `scope` waits on: a modelled count of threads whose closure — and
/// everything it borrowed from `'scope` — has not yet been dropped, and the
/// wake the thread that takes it to zero sends.
struct ScopeData {
    num_running_threads: AtomicUsize,
    all_done: rt::Notify,
}

/// A scoped thread's result slot, shared by the thread and its handle.
struct Packet<'scope, T> {
    result: Mutex<Option<std::thread::Result<T>>>,
    done: rt::Notify,
    /// Notified when the main function returns, before the TLS destructors.
    finished: rt::Notify,
    _scope: PhantomData<&'scope ()>,
}

/// Mock implementation of `std::thread::scope`.
///
/// Creates a scope for spawning scoped threads, which may borrow non-`'static`
/// data. Every thread spawned in the scope that has not been joined manually is
/// joined before this returns, and its closure's effects happen-before the
/// return, as `std`'s modelled counter handshake gives them.
///
/// A panic in a scoped thread fails the model, as a panic in any loom thread
/// does, so the `std` behaviour of re-raising it here as "a scoped thread
/// panicked" does not arise.
#[track_caller]
pub fn scope<'env, F, T>(f: F) -> T
where
    F: for<'scope> FnOnce(&'scope Scope<'scope, 'env>) -> T,
{
    let scope = Scope {
        data: Arc::new(ScopeData {
            num_running_threads: AtomicUsize::new(0),
            all_done: rt::Notify::new(false, false),
        }),
        scope: PhantomData,
        env: PhantomData,
    };

    // Catch a panic from `f` so no scoped thread can outlive the borrows it
    // was spawned with while this frame unwinds.
    let result = catch_unwind(AssertUnwindSafe(|| f(&scope)));

    let location = location!();
    while scope.data.num_running_threads.load(Acquire) != 0 {
        scope.data.all_done.wait(location);
    }

    match result {
        Ok(result) => result,
        Err(e) => resume_unwind(e),
    }
}

impl<'scope, 'env> Scope<'scope, 'env> {
    /// Mock implementation of `std::thread::Scope::spawn`.
    ///
    /// Spawns a new thread within a scope, returning a [`ScopedJoinHandle`] for
    /// it. Unlike non-scoped threads, the closure may borrow non-`'static`
    /// data from outside the scope.
    #[track_caller]
    pub fn spawn<F, T>(&'scope self, f: F) -> ScopedJoinHandle<'scope, T>
    where
        F: FnOnce() -> T + Send + 'scope,
        T: Send + 'scope,
    {
        spawn_scoped(self, f, None, None, location!())
    }
}

impl Builder {
    /// Mock implementation of `std::thread::Builder::spawn_scoped`.
    ///
    /// Spawns a new scoped thread using the settings set through this
    /// `Builder`.
    #[track_caller]
    pub fn spawn_scoped<'scope, 'env, F, T>(
        self,
        scope: &'scope Scope<'scope, 'env>,
        f: F,
    ) -> io::Result<ScopedJoinHandle<'scope, T>>
    where
        F: FnOnce() -> T + Send + 'scope,
        T: Send + 'scope,
    {
        Ok(spawn_scoped(scope, f, self.name, self.stack_size, location!()))
    }
}

fn spawn_scoped<'scope, F, T>(
    scope: &'scope Scope<'scope, '_>,
    f: F,
    name: Option<String>,
    stack_size: Option<usize>,
    location: Location,
) -> ScopedJoinHandle<'scope, T>
where
    F: FnOnce() -> T + Send + 'scope,
    T: Send + 'scope,
{
    let packet = Arc::new(Packet {
        result: Mutex::new(None),
        done: rt::Notify::new(true, false),
        finished: rt::Notify::new(false, false),
        _scope: PhantomData,
    });

    scope.data.num_running_threads.fetch_add(1, Relaxed);

    let body: Box<dyn FnOnce() + 'scope> = {
        let name = name.clone();
        let packet = packet.clone();
        let data = scope.data.clone();
        Box::new(move || {
            rt::execution(|execution| {
                init_current(execution, name);
            });

            *packet.result.lock().unwrap() = Some(Ok(f()));

            packet.finished.notify(location);

            rt::drop_locals();

            packet.done.notify(location);

            // The last `'scope` borrow this thread holds. Dropped before the
            // count falls, so `scope` cannot return while it is live.
            drop(packet);

            if data.num_running_threads.fetch_sub(1, Release) == 1 {
                data.all_done.notify(location);
            }
        })
    };

    // SAFETY: `scope` does not return — and so no `'scope` borrow ends —
    // until `num_running_threads` is zero, which this body brings about only
    // after dropping every `'scope` value it owns. What runs after the
    // decrement (`notify`, and dropping `data`) owns nothing of `'scope`. A
    // panicking model leaks the suspended coroutines rather than unwinding
    // them, so no path runs this body's drop glue once `scope` has returned.
    let body: Box<dyn FnOnce() + 'static> = unsafe { std::mem::transmute(body) };

    let id = rt::spawn(stack_size, None, body);

    ScopedJoinHandle {
        packet,
        thread: Thread {
            id: ThreadId { id },
            name,
        },
    }
}

impl<'scope, T> ScopedJoinHandle<'scope, T> {
    /// Mock implementation of `std::thread::ScopedJoinHandle::thread`.
    #[must_use]
    pub fn thread(&self) -> &Thread {
        &self.thread
    }

    /// Mock implementation of `std::thread::ScopedJoinHandle::join`.
    ///
    /// Waits for the associated thread to finish; its effects happen-before
    /// the return.
    #[track_caller]
    pub fn join(self) -> std::thread::Result<T> {
        self.packet.done.wait(location!());
        self.packet.result.lock().unwrap().take().unwrap()
    }

    /// Mock implementation of `std::thread::ScopedJoinHandle::is_finished`:
    /// true once the main function returns, before the thread's TLS
    /// destructors run. A `true` does not synchronize with the thread.
    #[track_caller]
    pub fn is_finished(&self) -> bool {
        self.packet.finished.is_notified(location!())
    }
}

impl fmt::Debug for Scope<'_, '_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Scope").finish_non_exhaustive()
    }
}

impl<T> fmt::Debug for ScopedJoinHandle<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ScopedJoinHandle").finish_non_exhaustive()
    }
}
