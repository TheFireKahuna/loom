#![deny(warnings, rust_2018_idioms)]

//! `static` locks and `Once`s are fresh in every execution — state and data —
//! and belong to no process-wide lock: models that share one run
//! concurrently, and a failed model leaves nothing behind for the next.

use loom::cell::UnsafeCell;
use loom::sync::atomic::{AtomicBool, AtomicUsize};
use loom::sync::{Arc, Condvar, Mutex, Once, RwLock};
use loom::thread;

use std::panic::{self, AssertUnwindSafe};
use std::sync::atomic::Ordering::{Relaxed, SeqCst};

fn bounded() -> loom::model::Builder {
    let mut builder = loom::model::Builder::new();
    builder.preemption_bound = Some(3);
    builder.threads = 1;
    builder
}

static FRESH_MUTEX: Mutex<u32> = Mutex::new(0);
static FRESH_RWLOCK: RwLock<Vec<u32>> = RwLock::new(Vec::new());

/// Each execution starts from the value the static was built with.
#[test]
fn static_lock_data_is_fresh_every_execution() {
    bounded().check(|| {
        let x = Arc::new(AtomicUsize::new(0));
        let x2 = x.clone();
        let th = thread::spawn(move || {
            x2.store(1, Relaxed);
            FRESH_RWLOCK.write().unwrap().push(1);
        });
        x.load(Relaxed);
        *FRESH_MUTEX.lock().unwrap() += 1;
        th.join().unwrap();
        assert_eq!(*FRESH_MUTEX.lock().unwrap(), 1);
        assert_eq!(*FRESH_RWLOCK.read().unwrap(), [1]);
    });
}

static HELD_RWLOCK: RwLock<u32> = RwLock::new(0);
static HELD_MUTEX: Mutex<u32> = Mutex::new(0);
static FAILED_ONCE: Once = Once::new();

/// A model that fails holding a static lock, or inside a static `Once`'s
/// closure, leaves the next model using them untouched.
#[test]
fn a_failed_model_leaves_statics_usable() {
    let failed = panic::catch_unwind(AssertUnwindSafe(|| {
        bounded().check(|| {
            let _w = HELD_RWLOCK.write().unwrap();
            let _m = HELD_MUTEX.lock().unwrap();
            panic!("deliberate");
        })
    }));
    assert!(failed.is_err());

    let failed = panic::catch_unwind(AssertUnwindSafe(|| {
        bounded().check(|| FAILED_ONCE.call_once(|| panic!("deliberate")))
    }));
    assert!(failed.is_err());

    bounded().check(|| {
        *HELD_RWLOCK.write().unwrap() += 1;
        *HELD_MUTEX.lock().unwrap() += 1;
        let ran = Arc::new(AtomicBool::new(false));
        let r2 = ran.clone();
        FAILED_ONCE.call_once(move || r2.store(true, Relaxed));
        assert!(ran.load(Relaxed));
        assert!(!FAILED_ONCE.is_completed() || ran.load(Relaxed));
    });
}

static SHARED_RWLOCK: RwLock<u32> = RwLock::new(0);
static SHARED_MUTEX: Mutex<u32> = Mutex::new(0);

fn shared_static_model() {
    let mut builder = bounded();
    builder.threads = 2;
    builder.budgeted = false;
    builder.probe = std::time::Duration::ZERO;
    builder.check(|| {
        let x = Arc::new(AtomicUsize::new(0));
        let x2 = x.clone();
        let th = thread::spawn(move || {
            for _ in 0..2 {
                x2.fetch_add(1, Relaxed);
            }
            *SHARED_MUTEX.lock().unwrap() += 1;
        });
        {
            let mut g = SHARED_RWLOCK.write().unwrap();
            for _ in 0..2 {
                x.fetch_add(1, Relaxed);
            }
            *g += 1;
            assert_eq!(*g, 1);
        }
        th.join().unwrap();
        assert_eq!(*SHARED_MUTEX.lock().unwrap(), 1);
    });
}

/// Concurrent models, each sharded over workers, share the statics.
#[test]
fn concurrent_models_share_static_locks() {
    let models: Vec<_> = (0..3).map(|_| std::thread::spawn(shared_static_model)).collect();
    for model in models {
        model.join().unwrap();
    }
}

/// A `Once` waiter acquires from the initializer only: another waiter's
/// writes before its own `call_once` do not happen before this one returns.
#[test]
#[should_panic(expected = "Causality violation")]
fn once_waiters_do_not_synchronize_with_each_other() {
    bounded().check(|| {
        let once = Arc::new(Once::new());
        let x = Arc::new(UnsafeCell::new(0u32));
        let b_done = Arc::new(AtomicBool::new(false));
        let init = {
            let once = once.clone();
            thread::spawn(move || once.call_once(|| loom::hint::spin_loop()))
        };
        let b = {
            let (once, x, b_done) = (once.clone(), x.clone(), b_done.clone());
            thread::spawn(move || {
                x.with_mut(|p| unsafe { *p = 1 });
                once.call_once(|| {});
                b_done.store(true, Relaxed);
            })
        };
        once.call_once(|| {});
        if b_done.load(Relaxed) {
            x.with(|p| unsafe { *p });
        }
        init.join().unwrap();
        b.join().unwrap();
    });
}

/// The initializer happens before every return from the `Once`.
#[test]
fn once_initializer_happens_before_every_return() {
    bounded().check(|| {
        let once = Arc::new(Once::new());
        let x = Arc::new(UnsafeCell::new(0u32));
        let th = {
            let (once, x) = (once.clone(), x.clone());
            thread::spawn(move || {
                once.call_once(|| x.with_mut(|p| unsafe { *p = 7 }));
                assert_eq!(x.with(|p| unsafe { *p }), 7);
            })
        };
        once.wait();
        assert!(once.is_completed());
        assert_eq!(x.with(|p| unsafe { *p }), 7);
        th.join().unwrap();
    });
}

/// `call_once_force` runs on a fresh `Once` with no poison to report, and
/// every other caller sees it done.
#[test]
fn call_once_force_runs_once() {
    bounded().check(|| {
        let once = Arc::new(Once::new());
        let runs = Arc::new(AtomicUsize::new(0));
        let th = {
            let (once, runs) = (once.clone(), runs.clone());
            thread::spawn(move || {
                once.call_once_force(|state| {
                    assert!(!state.is_poisoned());
                    runs.fetch_add(1, SeqCst);
                })
            })
        };
        once.call_once(|| {
            runs.fetch_add(1, SeqCst);
        });
        once.wait_force();
        th.join().unwrap();
        assert_eq!(runs.load(SeqCst), 1);
    });
}

/// `is_completed` reads with Acquire and may miss a completion it is not
/// ordered after.
#[test]
fn is_completed_may_read_stale() {
    use std::sync::atomic::AtomicBool as StdBool;
    static STALE: StdBool = StdBool::new(false);

    bounded().check(|| {
        let once = Arc::new(Once::new());
        let flag = Arc::new(AtomicBool::new(false));
        let th = {
            let (once, flag) = (once.clone(), flag.clone());
            thread::spawn(move || {
                once.call_once(|| {});
                flag.store(true, Relaxed);
            })
        };
        if flag.load(Relaxed) && !once.is_completed() {
            STALE.store(true, SeqCst);
        }
        th.join().unwrap();
    });

    assert!(STALE.load(SeqCst));
}

static UNSIZED: Mutex<[u32; 2]> = Mutex::new([0; 2]);

/// A `static` lock used through an unsized view works on its execution's
/// instance like any other.
#[test]
fn a_static_lock_is_fresh_through_an_unsized_view() {
    bounded().check(|| {
        let lock: &Mutex<[u32]> = &UNSIZED;
        lock.lock().unwrap()[1] += 1;
        assert_eq!(*UNSIZED.lock().unwrap(), [0, 1]);
    });
}

static CV_LOCK: Mutex<bool> = Mutex::new(false);
static CV: Condvar = Condvar::new();

/// A `static` condvar has no data and registers afresh each execution: a
/// waiter a failed model left queued on it is gone in the next model.
#[test]
fn a_static_condvar_carries_nothing_between_models() {
    let failed = panic::catch_unwind(AssertUnwindSafe(|| {
        bounded().check(|| {
            let th = thread::spawn(|| {
                let g = CV_LOCK.lock().unwrap();
                let _g = CV.wait_while(g, |ready| !*ready).unwrap();
            });
            let _ = &th;
            panic!("deliberate");
        })
    }));
    assert!(failed.is_err());

    bounded().check(|| {
        let th = thread::spawn(|| {
            *CV_LOCK.lock().unwrap() = true;
            CV.notify_one();
        });
        let g = CV_LOCK.lock().unwrap();
        let g = CV.wait_while(g, |ready| !*ready).unwrap();
        assert!(*g);
        drop(g);
        th.join().unwrap();
    });
}
