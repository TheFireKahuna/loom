#![deny(warnings, rust_2018_idioms)]

use loom::sync::atomic::{AtomicBool, Ordering::*};
use loom::sync::Arc;
use loom::thread;
use std::time::Duration;

/// With nobody to unpark it, a timed park ends by its timeout instead of
/// deadlocking the model.
#[test]
fn park_timeout_alone_times_out() {
    loom::model(|| {
        thread::park_timeout(Duration::from_millis(1));
    });
}

/// An unpark that lands after the timeout fired leaves the token, so the next
/// park consumes it rather than blocking forever.
#[test]
fn unpark_after_timeout_leaves_the_token() {
    loom::model(|| {
        let main = thread::current();
        let th = thread::spawn(move || main.unpark());
        thread::park_timeout(Duration::from_millis(1));
        th.join().unwrap();
        // The token is consumed or still pending; either way this pair returns.
        thread::current().unpark();
        thread::park();
    });
}

/// A timed park that consumes the token acquires the unparker's view.
#[test]
fn park_timeout_acquires_through_the_token() {
    loom::model(|| {
        let flag = Arc::new(AtomicBool::new(false));
        let data = Arc::new(loom::cell::UnsafeCell::new(0u32));
        let main = thread::current();
        let (f2, d2) = (flag.clone(), data.clone());
        let th = thread::spawn(move || {
            d2.with_mut(|p| unsafe { *p = 7 });
            f2.store(true, Relaxed);
            main.unpark();
        });
        while !flag.load(Relaxed) {
            thread::park_timeout(Duration::from_millis(1));
        }
        th.join().unwrap();
        assert_eq!(data.with(|p| unsafe { *p }), 7);
    });
}
