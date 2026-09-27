#![deny(warnings, rust_2018_idioms)]

//! A panic in a model thread reaches the panic hook once — at its own site —
//! and fails the model with its own message, even when the panicking thread's
//! own unwinding would block on a lock its peer holds.
//!
//! A binary of its own: it installs a process-wide panic hook.

use loom::sync::{Arc, Mutex};
use loom::thread;

use std::panic;
use std::sync::atomic::Ordering::Relaxed;

/// Takes the lock when dropped, as a guard type restoring shared state does.
struct LockOnDrop(Arc<Mutex<()>>);

impl Drop for LockOnDrop {
    fn drop(&mut self) {
        drop(self.0.lock().unwrap());
    }
}

#[test]
fn panic_in_model_thread_is_reported_once() {
    static SEEN: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

    let next = panic::take_hook();
    panic::set_hook(Box::new(move |info| {
        if info
            .payload_as_str()
            .is_some_and(|s| s.contains("original failure"))
        {
            SEEN.fetch_add(1, Relaxed);
        }
        next(info);
    }));

    let payload = panic::catch_unwind(|| {
        let mut builder = loom::model::Builder::new();
        builder.threads = 1;
        builder.check(|| {
            let lock = Arc::new(Mutex::new(()));

            let th = {
                let lock = lock.clone();
                thread::spawn(move || {
                    let _restore = LockOnDrop(lock);
                    panic!("original failure");
                })
            };

            // Joined with the lock held: the panicking thread's unwinding
            // blocks on it, a deadlock inside the unwind.
            let _held = lock.lock().unwrap();
            th.join().unwrap();
        });
    })
    .expect_err("the model thread panics");

    let _ = panic::take_hook();

    let message = payload
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| payload.downcast_ref::<&str>().copied());
    assert_eq!(message, Some("original failure"));
    assert_eq!(SEEN.load(Relaxed), 1);
}
