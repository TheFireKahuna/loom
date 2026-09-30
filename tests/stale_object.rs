#![deny(warnings, rust_2018_idioms)]

//! An object built at run time belongs to its execution. Kept into a later
//! one, its index names a slot that execution has not reincarnated, and the
//! access is refused in every build rather than reading the slot's carcass.

use loom::model::Builder;
use loom::sync::atomic::AtomicUsize;
use loom::thread;

use std::sync::atomic::Ordering::{Relaxed, SeqCst};
use std::sync::Mutex;

fn builder() -> Builder {
    let mut b = Builder::new();
    // One worker: the rigs pass an object between executions through a static.
    b.threads = 1;
    b
}

// The kept cell's first access in the next execution precedes any object that
// execution creates.
#[test]
#[should_panic(expected = "used outside the execution that created it")]
fn a_kept_cell_is_refused_at_its_first_access() {
    static KEEP: Mutex<Option<AtomicUsize>> = Mutex::new(None);
    builder().check(|| {
        if let Some(c) = KEEP.lock().unwrap().take() {
            c.load(SeqCst);
            std::mem::forget(c);
            return;
        }
        let (_a, _b) = (AtomicUsize::new(1), AtomicUsize::new(2));
        let c = AtomicUsize::new(3);
        c.store(42, SeqCst);
        *KEEP.lock().unwrap() = Some(c);
        let x = std::sync::Arc::new(AtomicUsize::new(0));
        let x2 = x.clone();
        let t = thread::spawn(move || x2.store(1, Relaxed));
        x.load(Relaxed);
        t.join().unwrap();
    });
}

// An access with no scheduling point of its own is refused the same way.
#[test]
#[should_panic(expected = "used outside the execution that created it")]
fn a_kept_cell_is_refused_at_an_unsync_load() {
    static KEEP: Mutex<Option<AtomicUsize>> = Mutex::new(None);
    builder().check(|| {
        if let Some(c) = KEEP.lock().unwrap().take() {
            unsafe { c.unsync_load() };
            std::mem::forget(c);
            return;
        }
        let (_a, _b) = (AtomicUsize::new(1), AtomicUsize::new(2));
        let c = AtomicUsize::new(3);
        c.store(42, SeqCst);
        *KEEP.lock().unwrap() = Some(c);
        let x = std::sync::Arc::new(AtomicUsize::new(0));
        let x2 = x.clone();
        let t = thread::spawn(move || x2.store(1, Relaxed));
        x.load(Relaxed);
        t.join().unwrap();
    });
}
