use crate::rt;

use std::sync::Mutex;
use std::task::Waker;

/// Mock implementation of `tokio::sync::AtomicWaker`.
#[derive(Debug)]
pub struct AtomicWaker {
    waker: Mutex<Option<Waker>>,
    object: rt::Mutex,
}

impl AtomicWaker {
    /// Create a new instance of `AtomicWaker`.
    pub fn new() -> AtomicWaker {
        AtomicWaker {
            waker: Mutex::new(None),
            object: rt::Mutex::new(false),
        }
    }

    /// Registers the current task to be notified on calls to `wake`.
    ///
    /// The lock stands for `futures`' lock-free registration states, and a
    /// `wake` holds it only while it takes the waker. There, a `register`
    /// that lands in that window wakes its own task and returns, and the
    /// executor re-polls until a `register` stores the waker once the take is
    /// done: a busy wait on the waking thread. Blocking on the lock is that
    /// wait, ending in the same stored waker, without the re-polls that only
    /// spend steps until the take finishes.
    #[track_caller]
    pub fn register(&self, waker: Waker) {
        self.object.acquire_lock(location!());
        *self.waker.lock().unwrap() = Some(waker);
        self.object.unlock(location!());
    }

    /// Registers the current task to be woken without consuming the value.
    pub fn register_by_ref(&self, waker: &Waker) {
        self.register(waker.clone());
    }

    /// Notifies the task that last called `register`.
    pub fn wake(&self) {
        if let Some(waker) = self.take_waker() {
            waker.wake();
        }
    }

    /// Attempts to take the `Waker` value out of the `AtomicWaker` with the
    /// intention that the caller will wake the task later.
    #[track_caller]
    pub fn take_waker(&self) -> Option<Waker> {
        self.object.acquire_lock(location!());

        let ret = self.waker.lock().unwrap().take();

        self.object.unlock(location!());

        ret
    }
}

impl Default for AtomicWaker {
    fn default() -> Self {
        AtomicWaker::new()
    }
}

#[cfg(test)]
mod tests {
    use super::AtomicWaker;
    use crate::rt;
    use crate::sync::Arc;
    use crate::thread;

    use std::future::poll_fn;
    use std::task::Poll;

    /// A `register` that collides with a `wake` in progress waits for it, and
    /// spends no forced-progress wait doing so: no store the task has seen
    /// elsewhere is made unreadable to it.
    #[test]
    fn a_colliding_register_is_not_a_spin() {
        crate::model(|| {
            let waker = Arc::new(AtomicWaker::new());
            let w2 = waker.clone();
            let th = thread::spawn(move || w2.wake());

            crate::future::block_on(poll_fn(|cx| {
                waker.register_by_ref(cx.waker());
                let yielded = rt::execution(|execution| execution.threads.active().last_yield);
                assert_eq!(yielded, None, "register entered the forced-progress model");
                Poll::Ready(())
            }));
            th.join().unwrap();
        });
    }
}
