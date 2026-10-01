#![deny(warnings, rust_2018_idioms)]

use loom::hint::{monitor_wait, monitor_wait_timeout, MonitorWake};
use loom::sync::atomic::{AtomicU128, AtomicU32, Ordering::*};
use loom::sync::Arc;
use loom::thread;

use std::collections::BTreeSet;
use std::sync::atomic::{AtomicBool as StdBool, AtomicUsize as StdUsize, Ordering as Std};
use std::sync::Mutex as StdMutex;

/// One worker, so a `std` flag a thread sets right after a modelled step
/// tells the threads that run later in the same execution that it ran.
fn serial() -> loom::model::Builder {
    let mut builder = loom::model::Builder::new();
    builder.threads = 1;
    builder
}

/// Wide enough to see every lane of the 128-bit cell.
const LANE0: u128 = u32::MAX as u128;

/// A store that lands before the wait begins, after the read that found the
/// condition false, ends the wait at once; one that lands later wakes the
/// sleeper. Either way the waiter's next read returns the store: the wake is
/// a coherence observation of it. A wake the model failed to deliver would
/// leave the waiter asleep with its peer finished, a deadlock.
#[test]
fn a_store_before_or_during_the_wait_wakes_the_waiter() {
    static STORED: StdBool = StdBool::new(false);
    static BEFORE: StdUsize = StdUsize::new(0);
    static AFTER: StdUsize = StdUsize::new(0);

    serial().check(|| {
        STORED.store(false, Std::Relaxed);
        let x = Arc::new(AtomicU32::new(0));
        let x2 = x.clone();
        let th = thread::spawn(move || {
            x2.store(1, Relaxed);
            STORED.store(true, Std::Relaxed);
        });

        if x.load(Relaxed) == 0 {
            let landed = STORED.load(Std::Relaxed);
            assert_eq!(monitor_wait(&*x), MonitorWake::Stored);
            if landed { &BEFORE } else { &AFTER }.fetch_add(1, Std::Relaxed);
            assert_eq!(x.load(Relaxed), 1, "the waiter read older than its wake");
        }
        th.join().unwrap();
    });

    assert!(BEFORE.load(Std::Relaxed) > 0, "no store landed before the wait");
    assert!(AFTER.load(Std::Relaxed) > 0, "no store landed after the check");
}

/// A read-modify-write is a store to the watched location.
#[test]
fn an_rmw_wakes_the_waiter() {
    loom::model(|| {
        let x = Arc::new(AtomicU32::new(0));
        let x2 = x.clone();
        let th = thread::spawn(move || {
            x2.fetch_add(1, Relaxed);
        });
        while x.load(Relaxed) == 0 {
            assert_eq!(monitor_wait(&*x), MonitorWake::Stored);
        }
        th.join().unwrap();
    });
}

/// A compare-exchange that fails stores nothing, so it wakes no one.
#[test]
fn a_failed_compare_exchange_does_not_wake() {
    loom::model(|| {
        let x = Arc::new(AtomicU32::new(0));
        let x2 = x.clone();
        let th = thread::spawn(move || {
            assert!(x2.compare_exchange(5, 6, AcqRel, Acquire).is_err());
        });
        assert_eq!(x.load(Relaxed), 0);
        assert_eq!(monitor_wait_timeout(&*x), MonitorWake::TimedOut);
        th.join().unwrap();
    });
}

/// A store to another cell does not wake a monitor of this one.
#[test]
fn a_store_to_another_location_does_not_wake() {
    loom::model(|| {
        let x = Arc::new(AtomicU32::new(0));
        let y = Arc::new(AtomicU32::new(0));
        let y2 = y.clone();
        let th = thread::spawn(move || y2.store(1, Release));
        assert_eq!(x.load(Relaxed), 0);
        assert_eq!(monitor_wait_timeout(&*x), MonitorWake::TimedOut);
        th.join().unwrap();
    });
}

/// A lane's monitor watches the lane alone: a store to a disjoint lane of the
/// same cell never wakes it, and a store to its own lane does.
#[test]
fn a_store_to_a_disjoint_lane_does_not_wake() {
    loom::model(|| {
        let cell = Arc::new(AtomicU128::new(0));
        let c2 = cell.clone();
        let th = thread::spawn(move || c2.lane_u32(4).store(1, Release));
        assert_eq!(cell.lane_u32(0).load(Relaxed), 0);
        assert_eq!(monitor_wait_timeout(&cell.lane_u32(0)), MonitorWake::TimedOut);
        th.join().unwrap();
    });
}

#[test]
fn a_store_to_the_watched_lane_wakes() {
    loom::model(|| {
        let cell = Arc::new(AtomicU128::new(0));
        let c2 = cell.clone();
        let th = thread::spawn(move || {
            c2.lane_u32(4).store(1, Relaxed);
            c2.lane_u32(0).store(1, Relaxed);
        });
        while cell.lane_u32(0).load(Relaxed) == 0 {
            assert_eq!(monitor_wait(&cell.lane_u32(0)), MonitorWake::Stored);
        }
        th.join().unwrap();
    });
}

/// A whole-cell store writes every lane, so it wakes a lane's monitor.
#[test]
fn a_whole_cell_store_wakes_a_lane_monitor() {
    loom::model(|| {
        let cell = Arc::new(AtomicU128::new(0));
        let c2 = cell.clone();
        let th = thread::spawn(move || c2.store(1 << 64 | 1, Relaxed));
        while cell.lane_u32(0).load(Relaxed) == 0 {
            assert_eq!(monitor_wait(&cell.lane_u32(0)), MonitorWake::Stored);
        }
        th.join().unwrap();
    });
}

/// A wide compare-exchange that carries the watched lane through verbatim
/// writes no store there, so it does not wake that lane's monitor.
#[test]
fn a_preserving_rmw_does_not_wake_the_lane_it_carries() {
    loom::model(|| {
        let cell = Arc::new(AtomicU128::new(0));
        let c2 = cell.clone();
        let th = thread::spawn(move || {
            let _ = c2.compare_exchange_preserving(LANE0, 0, 1 << 64, AcqRel, Acquire);
        });
        assert_eq!(cell.lane_u32(0).load(Relaxed), 0);
        assert_eq!(monitor_wait_timeout(&cell.lane_u32(0)), MonitorWake::TimedOut);
        th.join().unwrap();
    });
}

/// With nothing to store, a timed wait ends by its timeout.
#[test]
fn the_timeout_ends_an_unwoken_wait() {
    loom::model(|| {
        let x = AtomicU32::new(0);
        assert_eq!(x.load(Relaxed), 0);
        assert_eq!(monitor_wait_timeout(&x), MonitorWake::TimedOut);
    });
}

/// A waiter no store can wake is a deadlock, and the report names the
/// location it waits on.
#[test]
#[should_panic(expected = "waiting for a store to atomic #")]
fn a_waiter_nothing_can_wake_is_a_deadlock() {
    loom::model(|| {
        let x = AtomicU32::new(0);
        assert_eq!(x.load(Relaxed), 0);
        monitor_wait(&x);
    });
}

#[test]
#[should_panic(expected = "bits 0xffffffff00000000")]
fn the_deadlock_report_names_the_lane() {
    loom::model(|| {
        let cell = Arc::new(AtomicU128::new(0));
        let c2 = cell.clone();
        let th = thread::spawn(move || c2.lane_u32(0).store(1, Relaxed));
        assert_eq!(cell.lane_u32(4).load(Relaxed), 0);
        monitor_wait(&cell.lane_u32(4));
        th.join().unwrap();
    });
}

/// The wake orders the waiter's reads of the location and nothing else: data
/// stored before the flag without a release is not acquired through it.
#[test]
fn the_wake_carries_no_synchronization() {
    let seen: std::sync::Arc<StdMutex<BTreeSet<u32>>> = Default::default();
    let out = seen.clone();

    loom::model(move || {
        let data = Arc::new(AtomicU32::new(0));
        let flag = Arc::new(AtomicU32::new(0));
        let (d2, f2) = (data.clone(), flag.clone());
        let th = thread::spawn(move || {
            d2.store(1, Relaxed);
            f2.store(1, Relaxed);
        });
        while flag.load(Relaxed) == 0 {
            monitor_wait(&*flag);
        }
        out.lock().unwrap().insert(data.load(Relaxed));
        th.join().unwrap();
    });

    assert_eq!(*seen.lock().unwrap(), BTreeSet::from([0, 1]));
}

/// A store modification order has not placed against what the waiter has
/// seen is explored on both sides. Beyond, it wakes the waiter; before, it
/// never does, and the waiter, which saw the store after it, can no longer
/// read it.
#[test]
fn an_unplaced_store_is_explored_on_both_sides() {
    static PEER_STORED: StdBool = StdBool::new(false);
    static WOKEN: StdUsize = StdUsize::new(0);
    static PLACED_BEFORE: StdUsize = StdUsize::new(0);

    serial().check(|| {
        PEER_STORED.store(false, Std::Relaxed);
        let x = Arc::new(AtomicU32::new(0));
        let (x1, x2) = (x.clone(), x.clone());
        let a = thread::spawn(move || x1.store(1, Relaxed));
        let b = thread::spawn(move || {
            x2.store(2, Relaxed);
            PEER_STORED.store(true, Std::Relaxed);
        });

        if x.load(Relaxed) == 1 {
            match monitor_wait_timeout(&*x) {
                MonitorWake::Stored => {
                    assert_eq!(x.load(Relaxed), 2);
                    WOKEN.fetch_add(1, Std::Relaxed);
                }
                MonitorWake::TimedOut if PEER_STORED.load(Std::Relaxed) => {
                    assert_eq!(x.load(Relaxed), 1, "a store placed before 1 was read after it");
                    PLACED_BEFORE.fetch_add(1, Std::Relaxed);
                }
                MonitorWake::TimedOut => {}
            }
        }
        a.join().unwrap();
        b.join().unwrap();
    });

    assert!(WOKEN.load(Std::Relaxed) > 0, "the store was never placed beyond");
    assert!(PLACED_BEFORE.load(Std::Relaxed) > 0, "the store was never placed before");
}
