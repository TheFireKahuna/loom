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

fn bounded() -> loom::model::Builder {
    let mut builder = loom::model::Builder::new();
    builder.preemption_bound = Some(3);
    builder.threads = 1;
    builder
}

/// A timeout fires on its own, so a peer spinning for the timed thread's
/// effect is waiting on time, not on a deadlock or an endless spin.
#[test]
fn a_timeout_ends_a_peers_spin() {
    bounded().check(|| {
        let flag = Arc::new(AtomicBool::new(false));
        let f2 = flag.clone();
        let th = thread::spawn(move || {
            thread::park_timeout(Duration::from_millis(1));
            f2.store(true, Release);
        });
        while !flag.load(Acquire) {
            loom::hint::spin_loop();
        }
        th.join().unwrap();
    });
}

/// The same with two timed parks in a row: the second times out as the
/// first does, however many timeouts came before it.
#[test]
fn successive_timeouts_end_a_peers_spin() {
    bounded().check(|| {
        let flag = Arc::new(AtomicBool::new(false));
        let f2 = flag.clone();
        let th = thread::spawn(move || {
            thread::park_timeout(Duration::from_millis(1));
            thread::park_timeout(Duration::from_millis(1));
            f2.store(true, Release);
        });
        while !flag.load(Acquire) {
            loom::hint::spin_loop();
        }
        th.join().unwrap();
    });
}

/// `thread::sleep` is a timed wait nothing ends early: a sleeping thread
/// wakes on its own, and a poll loop that sleeps between polls completes.
#[test]
fn a_sleep_poll_loop_completes() {
    bounded().check(|| {
        let flag = Arc::new(AtomicBool::new(false));
        let f2 = flag.clone();
        let th = thread::spawn(move || f2.store(true, Release));
        while !flag.load(Acquire) {
            thread::sleep(Duration::from_millis(1));
        }
        th.join().unwrap();
    });
}

/// A timeout can fire while its unparker is still runnable: the unpark then
/// finds the thread already returned, and the token is left.
#[test]
fn a_timeout_fires_before_a_runnable_unparker() {
    use std::sync::atomic::{AtomicBool as StdBool, Ordering::SeqCst};
    static EARLY: StdBool = StdBool::new(false);

    bounded().check(|| {
        let unparked = Arc::new(AtomicBool::new(false));
        let main = thread::current();
        let u2 = unparked.clone();
        let th = thread::spawn(move || {
            u2.store(true, SeqCst);
            main.unpark();
        });
        thread::park_timeout(Duration::from_millis(1));
        if !unparked.load(SeqCst) {
            EARLY.store(true, SeqCst);
        }
        th.join().unwrap();
    });

    assert!(EARLY.load(SeqCst), "no timeout fired before the unparker ran");
}

/// A sleeper resumed by its own timeout sits at a free branch: another thread
/// may run there at no preemption. At bound 0 the sleeper still sees a peer's
/// store made during the sleep before a second peer runs.
#[test]
fn a_sleep_resumed_by_its_timeout_frees_the_switch() {
    let seen: std::sync::Arc<std::sync::Mutex<std::collections::BTreeSet<(bool, bool)>>> =
        Default::default();
    let out = seen.clone();
    let mut builder = loom::model::Builder::new();
    builder.preemption_bound = Some(0);
    builder.threads = 1;
    builder.check(move || {
        let first = Arc::new(AtomicBool::new(false));
        let second = Arc::new(AtomicBool::new(false));
        let f1 = first.clone();
        let t1 = thread::spawn(move || f1.store(true, Relaxed));
        let f2 = second.clone();
        let t2 = thread::spawn(move || f2.store(true, Relaxed));
        thread::sleep(Duration::from_millis(1));
        out.lock()
            .unwrap()
            .insert((first.load(Relaxed), second.load(Relaxed)));
        t1.join().unwrap();
        t2.join().unwrap();
    });
    assert!(
        seen.lock().unwrap().contains(&(false, true)),
        "the sleeper never saw the second store alone: {:?}",
        seen.lock().unwrap()
    );
}
