#![deny(warnings, rust_2018_idioms)]

use loom::sync::atomic::{AtomicBool, AtomicUsize};
use loom::sync::{Condvar, Mutex};
use loom::thread;

use std::sync::atomic::Ordering::{Acquire, Relaxed, Release, SeqCst};
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

fn bounded() -> loom::model::Builder {
    let mut builder = loom::model::Builder::new();
    builder.preemption_bound = Some(3);
    builder.threads = 1;
    builder
}

/// Waiters whose timeouts fire before the notifier runs, counted by the
/// notifier.
fn early_timeouts(waiters: usize) -> std::collections::BTreeSet<usize> {
    let seen: Arc<std::sync::Mutex<std::collections::BTreeSet<usize>>> = Default::default();
    let out = seen.clone();
    bounded().check(move || {
        let s = Arc::new((Mutex::new(false), Condvar::new()));
        let early = Arc::new(AtomicUsize::new(0));
        let hs: Vec<_> = (0..waiters)
            .map(|_| {
                let (s, early) = (s.clone(), early.clone());
                thread::spawn(move || {
                    let g = s.0.lock().unwrap();
                    if !*g {
                        let (g, r) = s.1.wait_timeout(g, Duration::from_millis(1)).unwrap();
                        if r.timed_out() && !*g {
                            early.fetch_add(1, SeqCst);
                        }
                    }
                })
            })
            .collect();
        {
            let mut g = s.0.lock().unwrap();
            out.lock().unwrap().insert(early.load(SeqCst));
            *g = true;
            s.1.notify_all();
        }
        for h in hs {
            h.join().unwrap();
        }
    });
    let seen = seen.lock().unwrap().clone();
    seen
}

/// One waiter's timeout can fire before the notifier runs.
#[test]
fn one_waiter_times_out_early() {
    assert!(early_timeouts(1).contains(&1));
}

/// So can each of two waiters': the early timeout is a waiter's, not the
/// condvar's.
#[test]
fn two_waiters_both_time_out_early() {
    assert!(early_timeouts(2).contains(&2));
}

/// A peer spinning for a timed waiter's effect is ended by the waiter's
/// timeouts, however many it waits through.
#[test]
fn successive_wait_timeouts_end_a_peers_spin() {
    bounded().check(|| {
        let flag = Arc::new(AtomicBool::new(false));
        let s = Arc::new((Mutex::new(()), Condvar::new()));
        let (f2, s2) = (flag.clone(), s.clone());
        let th = thread::spawn(move || {
            let g = s2.0.lock().unwrap();
            let (g, _) = s2.1.wait_timeout(g, Duration::from_millis(1)).unwrap();
            let (_g, _) = s2.1.wait_timeout(g, Duration::from_millis(1)).unwrap();
            f2.store(true, Release);
        });
        while !flag.load(Acquire) {
            loom::hint::spin_loop();
        }
        th.join().unwrap();
    });
}

/// A wait that times out before a runnable notifier gets the lock sees a third
/// thread's store made while it was blocked. Bound 0 reaches it: the waiter
/// blocking frees the switch to the storer, the storer finishing frees the
/// waiter's timeout, and the notifier runs last.
#[test]
fn timeout_after_a_free_switch_sees_the_store() {
    let seen: Arc<std::sync::Mutex<std::collections::BTreeSet<(usize, bool, usize)>>> =
        Default::default();
    let out = seen.clone();
    let mut builder = loom::model::Builder::new();
    builder.preemption_bound = Some(0);
    builder.threads = 1;
    builder.check(move || {
        let cell = Arc::new(AtomicUsize::new(0));
        let s = Arc::new((Mutex::new(0), Condvar::new()));
        let s1 = s.clone();
        let notifier = thread::spawn(move || {
            *s1.0.lock().unwrap() += 1;
            s1.1.notify_one();
        });
        let c2 = cell.clone();
        let storer = thread::spawn(move || c2.store(1, Relaxed));
        let g = s.0.lock().unwrap();
        let (g, r) = s.1.wait_timeout(g, Duration::from_millis(1)).unwrap();
        out.lock()
            .unwrap()
            .insert((*g, r.timed_out(), cell.load(Relaxed)));
        drop(g);
        notifier.join().unwrap();
        storer.join().unwrap();
    });
    assert!(
        seen.lock().unwrap().contains(&(0, true, 1)),
        "no early timeout saw the store: {:?}",
        seen.lock().unwrap()
    );
}
