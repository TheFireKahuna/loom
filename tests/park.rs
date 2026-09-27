#![deny(warnings, rust_2018_idioms)]
//! `thread::park` / `Thread::unpark` against `std`'s contract: unpark
//! synchronizes with the park that consumes its token — not with the target
//! at the moment of the unpark — and park may return spuriously.

use loom::cell::UnsafeCell;
use loom::sync::atomic::AtomicUsize;
use loom::thread;

use std::collections::HashSet;
use std::sync::atomic::Ordering::{Acquire, Relaxed, Release};
use std::sync::{Arc, Mutex};

/// An unpark the target never consumes by parking gives it nothing: reading
/// the data after seeing a relaxed flag races with the write.
#[test]
#[should_panic(expected = "Causality violation")]
fn unpark_synchronizes_only_through_park() {
    loom::model(|| {
        let data = Arc::new(UnsafeCell::new(0usize));
        let go = Arc::new(AtomicUsize::new(0));

        let waiter = {
            let (data, go) = (data.clone(), go.clone());
            thread::spawn(move || {
                while go.load(Relaxed) == 0 {
                    thread::yield_now();
                }
                data.with(|p| unsafe { *p });
            })
        };

        let target = waiter.thread().clone();
        let waker = {
            let (data, go) = (data.clone(), go.clone());
            thread::spawn(move || {
                data.with_mut(|p| unsafe { *p = 1 });
                target.unpark();
                go.store(1, Relaxed);
            })
        };

        waker.join().unwrap();
        waiter.join().unwrap();
    });
}

/// The canonical park loop is race-free and never deadlocks, whichever side
/// of the waiter's check the unpark lands on, spurious return included.
#[test]
fn park_loop_is_sound() {
    loom::model(|| {
        let data = Arc::new(UnsafeCell::new(0usize));
        let ready = Arc::new(AtomicUsize::new(0));

        let waiter = {
            let (data, ready) = (data.clone(), ready.clone());
            thread::spawn(move || {
                while ready.load(Acquire) == 0 {
                    thread::park();
                }
                assert_eq!(1, data.with(|p| unsafe { *p }));
            })
        };

        data.with_mut(|p| unsafe { *p = 1 });
        ready.store(1, Release);
        waiter.thread().unpark();

        waiter.join().unwrap();
    });
}

/// A park may return spuriously, before any unpark; one that consumes the
/// token sees everything before the unpark.
#[test]
fn park_consumes_the_token_or_returns_spuriously() {
    let seen = Arc::new(Mutex::new(HashSet::new()));
    let sink = seen.clone();
    loom::model(move || {
        let flag = Arc::new(AtomicUsize::new(0));

        let waiter = {
            let flag = flag.clone();
            thread::spawn(move || {
                thread::park();
                flag.load(Relaxed)
            })
        };

        flag.store(1, Relaxed);
        waiter.thread().unpark();

        let r = waiter.join().unwrap();
        sink.lock().unwrap().insert(r);
    });
    let seen = seen.lock().unwrap().clone();
    assert_eq!(
        seen,
        HashSet::from([0, 1]),
        "0 is reachable only by a spurious return, 1 by consuming the token"
    );
}

/// The positive half of `unpark_synchronizes_only_through_park`: once the
/// unpark has made the token available, the park that consumes it acquires
/// the unparker's view, and the data read after it does not race.
#[test]
fn park_acquires_the_unparkers_view() {
    loom::model(|| {
        let data = Arc::new(UnsafeCell::new(0usize));
        let go = Arc::new(AtomicUsize::new(0));

        let waiter = {
            let (data, go) = (data.clone(), go.clone());
            thread::spawn(move || {
                while go.load(Relaxed) == 0 {
                    thread::yield_now();
                }
                thread::park();
                assert_eq!(1, data.with(|p| unsafe { *p }));
            })
        };

        let target = waiter.thread().clone();
        data.with_mut(|p| unsafe { *p = 1 });
        target.unpark();
        go.store(1, Relaxed);

        waiter.join().unwrap();
    });
}
