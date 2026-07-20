#![deny(warnings, rust_2018_idioms)]

use loom::sync::atomic::AtomicUsize;
use loom::sync::{Condvar, Mutex};
use loom::thread;

use std::sync::atomic::Ordering::SeqCst;
use std::sync::Arc;
use std::time::Duration;

#[test]
fn notify_one() {
    loom::model(|| {
        let inc = Arc::new(Inc::new());

        for _ in 0..1 {
            let inc = inc.clone();
            thread::spawn(move || inc.inc());
        }

        inc.wait();
    });
}

#[test]
fn notify_all() {
    loom::model(|| {
        let inc = Arc::new(Inc::new());

        let mut waiters = Vec::new();
        for _ in 0..2 {
            let inc = inc.clone();
            waiters.push(thread::spawn(move || inc.wait()));
        }

        thread::spawn(move || inc.inc_all()).join().expect("inc");

        for th in waiters {
            th.join().expect("waiter");
        }
    });
}

/// With a concurrent notifier, `wait_timeout` must explore both outcomes:
/// woken by the notification, and the timeout firing first.
#[test]
fn wait_timeout_explores_both_outcomes() {
    // `std` atomics: accumulated across iterations, checked after the model.
    static TIMED_OUT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    static WOKEN: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

    loom::model(|| {
        let pair = Arc::new((Mutex::new(false), Condvar::new()));

        let notifier = {
            let pair = pair.clone();
            thread::spawn(move || {
                let (mutex, condvar) = &*pair;
                *mutex.lock().unwrap() = true;
                condvar.notify_one();
            })
        };

        let (mutex, condvar) = &*pair;
        let mut ready = mutex.lock().unwrap();
        let mut timed_out = false;

        while !*ready {
            let (guard, result) = condvar.wait_timeout(ready, Duration::from_millis(1)).unwrap();
            ready = guard;

            if result.timed_out() {
                timed_out = true;
                break;
            }
        }

        drop(ready);
        notifier.join().unwrap();

        if timed_out {
            TIMED_OUT.fetch_add(1, SeqCst);
        } else {
            WOKEN.fetch_add(1, SeqCst);
        }
    });

    assert!(TIMED_OUT.load(SeqCst) > 0, "timed-out branch never explored");
    assert!(WOKEN.load(SeqCst) > 0, "notified branch never explored");
}

/// A `wait_timeout` no notification can ever reach must resolve via the
/// timeout — never a reported deadlock. Reaching the end of the model is
/// that witness.
///
/// A timed wait may also surface the condvar's bounded pre-deadline spurious
/// wake (`rt::condvar`), which is a legal, explored outcome — so absorb it the
/// way a real caller does. The budget is at most one self-wake per condvar per
/// execution, so this re-waits at most once before the timeout resolves it,
/// and the loop is finite in every execution.
#[test]
fn wait_timeout_without_notifier_times_out() {
    loom::model(|| {
        let mutex = Mutex::new(());
        let condvar = Condvar::new();

        let mut guard = mutex.lock().unwrap();
        loop {
            let (g, result) = condvar.wait_timeout(guard, Duration::from_millis(1)).unwrap();
            guard = g;
            if result.timed_out() {
                break;
            }
        }
        drop(guard);
    });
}

/// The timeout rescue also fires with other (untimed) waiters in the mix:
/// a joiner blocked on the timed-out thread must not be reported as a
/// deadlock either. The join returning is that witness.
///
/// The bounded spurious wake is absorbed here for the same reason as in
/// `wait_timeout_without_notifier_times_out`.
#[test]
fn wait_timeout_without_notifier_cross_thread() {
    loom::model(|| {
        thread::spawn(|| {
            let mutex = Mutex::new(());
            let condvar = Condvar::new();

            let mut guard = mutex.lock().unwrap();
            loop {
                let (g, result) = condvar.wait_timeout(guard, Duration::from_millis(1)).unwrap();
                guard = g;
                if result.timed_out() {
                    break;
                }
            }
        })
        .join()
        .unwrap();
    });
}

/// Untimed `wait` explores spurious wakeups: a wake with the predicate
/// still false (no notification has been sent at that point). The
/// re-check loop must absorb it and the wait must still complete.
#[test]
fn wait_explores_spurious_wakeup() {
    static SPURIOUS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

    loom::model(|| {
        let pair = Arc::new((Mutex::new(false), Condvar::new()));

        let notifier = {
            let pair = pair.clone();
            thread::spawn(move || {
                let (mutex, condvar) = &*pair;
                *mutex.lock().unwrap() = true;
                condvar.notify_one();
            })
        };

        let (mutex, condvar) = &*pair;
        let mut ready = mutex.lock().unwrap();

        while !*ready {
            ready = condvar.wait(ready).unwrap();

            // The notifier sets the flag before notifying, so waking with
            // the flag still clear is a spurious wakeup.
            if !*ready {
                SPURIOUS.fetch_add(1, SeqCst);
            }
        }

        drop(ready);
        notifier.join().unwrap();
    });

    assert!(SPURIOUS.load(SeqCst) > 0, "spurious wakeup never explored");
}

/// Spurious wakeups must not erase deadlock detection: an untimed wait no
/// notification can ever reach is still a deadlock.
#[test]
#[should_panic]
fn wait_without_notifier_deadlocks() {
    loom::model(|| {
        let mutex = Mutex::new(());
        let condvar = Condvar::new();

        let guard = mutex.lock().unwrap();
        let _guard = condvar.wait(guard).unwrap();
    });
}

struct Inc {
    num: AtomicUsize,
    mutex: Mutex<()>,
    condvar: Condvar,
}

impl Inc {
    fn new() -> Inc {
        Inc {
            num: AtomicUsize::new(0),
            mutex: Mutex::new(()),
            condvar: Condvar::new(),
        }
    }

    fn wait(&self) {
        let mut guard = self.mutex.lock().unwrap();

        loop {
            let val = self.num.load(SeqCst);
            if 1 == val {
                break;
            }

            guard = self.condvar.wait(guard).unwrap();
        }
    }

    fn inc(&self) {
        self.num.store(1, SeqCst);
        drop(self.mutex.lock().unwrap());
        self.condvar.notify_one();
    }

    fn inc_all(&self) {
        self.num.store(1, SeqCst);
        drop(self.mutex.lock().unwrap());
        self.condvar.notify_all();
    }
}
