#![deny(warnings, rust_2018_idioms)]

use loom::sync::atomic::AtomicUsize;
use loom::thread;

use std::collections::HashSet;
use std::sync::atomic::Ordering::{Relaxed, SeqCst};
use std::sync::{Arc, Mutex};

// Loom currently does not support load buffering.
#[test]
#[ignore]
fn load_buffering() {
    let values = Arc::new(Mutex::new(HashSet::new()));
    let values_ = values.clone();
    loom::model(move || {
        let x = Arc::new(AtomicUsize::new(0));
        let y = Arc::new(AtomicUsize::new(0));

        let th = {
            let (x, y) = (x.clone(), y.clone());
            thread::spawn(move || {
                x.store(y.load(Relaxed), Relaxed);
            })
        };

        let a = x.load(Relaxed);
        y.store(1, Relaxed);

        th.join().unwrap();
        values.lock().unwrap().insert(a);
    });
    assert!(values_.lock().unwrap().contains(&1));
}

#[test]
fn store_buffering() {
    let values = Arc::new(Mutex::new(HashSet::new()));
    let values_ = values.clone();
    loom::model(move || {
        let x = Arc::new(AtomicUsize::new(0));
        let y = Arc::new(AtomicUsize::new(0));

        let a = {
            let (x, y) = (x.clone(), y.clone());
            thread::spawn(move || {
                x.store(1, Relaxed);
                y.load(Relaxed)
            })
        };

        y.store(1, Relaxed);
        let b = x.load(Relaxed);

        let a = a.join().unwrap();
        values.lock().unwrap().insert((a, b));
    });
    assert!(values_.lock().unwrap().contains(&(0, 0)));
}

// `SeqCst` accesses are ordered by the SC total order S, so the
// store-buffering both-miss outcome must NOT be explored when every access in
// the Dekker pattern is `SeqCst` — without any `fence(SeqCst)` in the tested
// code.
#[test]
fn store_buffering_seq_cst() {
    let values = Arc::new(Mutex::new(HashSet::new()));
    let values_ = values.clone();
    loom::model(move || {
        let x = Arc::new(AtomicUsize::new(0));
        let y = Arc::new(AtomicUsize::new(0));

        let a = {
            let (x, y) = (x.clone(), y.clone());
            thread::spawn(move || {
                x.store(1, SeqCst);
                y.load(SeqCst)
            })
        };

        y.store(1, SeqCst);
        let b = x.load(SeqCst);

        let a = a.join().unwrap();
        values.lock().unwrap().insert((a, b));
    });
    let values = values_.lock().unwrap();
    assert!(!values.contains(&(0, 0)), "SC accesses must forbid the store-buffering both-miss");
    // The three SC-consistent outcomes are still explored.
    assert!(values.contains(&(1, 1)));
    assert!(values.contains(&(0, 1)));
    assert!(values.contains(&(1, 0)));
}

// Same edge with the arm spelled as an SC read-modify-write (the
// "push RMW then read the partner word" shape): the RMW's write half joins the
// SC order so the partner's SC load cannot miss it.
#[test]
fn store_buffering_seq_cst_rmw() {
    let values = Arc::new(Mutex::new(HashSet::new()));
    let values_ = values.clone();
    loom::model(move || {
        let x = Arc::new(AtomicUsize::new(0));
        let y = Arc::new(AtomicUsize::new(0));

        let a = {
            let (x, y) = (x.clone(), y.clone());
            thread::spawn(move || {
                x.fetch_or(1, SeqCst);
                y.load(SeqCst)
            })
        };

        y.fetch_or(1, SeqCst);
        let b = x.load(SeqCst);

        let a = a.join().unwrap();
        values.lock().unwrap().insert((a, b));
    });
    let values = values_.lock().unwrap();
    assert!(!values.contains(&(0, 0)), "SC RMWs must forbid the store-buffering both-miss");
    assert!(values.contains(&(1, 1)));
    assert!(values.contains(&(0, 1)));
    assert!(values.contains(&(1, 0)));
}

// SC accesses restrict only SC reads — they must NOT manufacture
// happens-before with unrelated SC operations. A relaxed load racing a
// relaxed store stays racy even when both threads also perform SC accesses
// to other cells: both stale and fresh reads remain explorable.
#[test]
fn sc_access_no_hb_leak() {
    let values = Arc::new(Mutex::new(HashSet::new()));
    let values_ = values.clone();
    loom::model(move || {
        let x = Arc::new(AtomicUsize::new(0));
        let a = Arc::new(AtomicUsize::new(0));
        let b = Arc::new(AtomicUsize::new(0));

        let th = {
            let (x, a) = (x.clone(), a.clone());
            thread::spawn(move || {
                x.store(1, Relaxed);
                a.store(1, SeqCst);
            })
        };

        b.store(1, SeqCst);
        let v = x.load(Relaxed);

        th.join().unwrap();
        values.lock().unwrap().insert(v);
    });
    let values = values_.lock().unwrap();
    assert!(values.contains(&0), "unrelated SC accesses must not force relaxed visibility");
    assert!(values.contains(&1));
}

// IRIW (independent reads of independent writes): two writers to two cells,
// two readers that each read both cells in the opposite order. Under SC there
// is a single total order over the four SC ops, so the two readers must agree
// on which write came first: the outcome where reader 1 sees x-before-y and
// reader 2 sees y-before-x is forbidden. This is the case a per-location SC
// model must still get right even though the two writes are to different cells
// and share no happens-before edge.
#[test]
fn iriw_seq_cst() {
    let seen = Arc::new(Mutex::new(false));
    let seen_ = seen.clone();
    loom::model(move || {
        let x = Arc::new(AtomicUsize::new(0));
        let y = Arc::new(AtomicUsize::new(0));

        let wx = {
            let x = x.clone();
            thread::spawn(move || x.store(1, SeqCst))
        };
        let wy = {
            let y = y.clone();
            thread::spawn(move || y.store(1, SeqCst))
        };
        let r1 = {
            let (x, y) = (x.clone(), y.clone());
            thread::spawn(move || (x.load(SeqCst), y.load(SeqCst)))
        };
        let r2 = {
            let (x, y) = (x.clone(), y.clone());
            thread::spawn(move || (y.load(SeqCst), x.load(SeqCst)))
        };

        wx.join().unwrap();
        wy.join().unwrap();
        let (x1, y1) = r1.join().unwrap();
        let (y2, x2) = r2.join().unwrap();

        // Forbidden: r1 saw x=1 before y (so x precedes y in S) while r2 saw
        // y=1 before x (so y precedes x in S).
        if (x1, y1, y2, x2) == (1, 0, 1, 0) {
            *seen.lock().unwrap() = true;
        }
    });
    assert!(!*seen_.lock().unwrap(), "SC forbids the IRIW disagreement outcome");
}

// RWC (read-write-causality): T1 stores x; T2 reads x then stores y; T3 reads
// y then reads x. Under SC, if T2 observes x=1 and T3 observes y=1, then T3
// must also observe x=1 — the (a=1, b=1, c=0) outcome is forbidden.
#[test]
fn rwc_seq_cst() {
    let seen = Arc::new(Mutex::new(false));
    let seen_ = seen.clone();
    loom::model(move || {
        let x = Arc::new(AtomicUsize::new(0));
        let y = Arc::new(AtomicUsize::new(0));

        let t1 = {
            let x = x.clone();
            thread::spawn(move || x.store(1, SeqCst))
        };
        let t2 = {
            let (x, y) = (x.clone(), y.clone());
            thread::spawn(move || {
                let a = x.load(SeqCst);
                y.store(1, SeqCst);
                a
            })
        };
        let t3 = {
            let (x, y) = (x.clone(), y.clone());
            thread::spawn(move || {
                let b = y.load(SeqCst);
                let c = x.load(SeqCst);
                (b, c)
            })
        };

        t1.join().unwrap();
        let a = t2.join().unwrap();
        let (b, c) = t3.join().unwrap();

        if (a, b, c) == (1, 1, 0) {
            *seen.lock().unwrap() = true;
        }
    });
    assert!(!*seen_.lock().unwrap(), "SC forbids the RWC outcome (1, 1, 0)");
}

// Completeness guard: the SC read rule must not over-prune. Two SC stores race
// a single SC load of the same cell; because the load can be scheduled before
// either, after the first, or after both, and the two stores are mutually
// mo-incomparable until ordered, every value {0, 1, 2} must remain reachable.
#[test]
fn sc_load_still_sees_every_racing_store() {
    let values = Arc::new(Mutex::new(HashSet::new()));
    let values_ = values.clone();
    loom::model(move || {
        let x = Arc::new(AtomicUsize::new(0));

        let w1 = {
            let x = x.clone();
            thread::spawn(move || x.store(1, SeqCst))
        };
        let w2 = {
            let x = x.clone();
            thread::spawn(move || x.store(2, SeqCst))
        };

        let v = x.load(SeqCst);

        w1.join().unwrap();
        w2.join().unwrap();
        values.lock().unwrap().insert(v);
    });
    let values = values_.lock().unwrap();
    for expected in [0, 1, 2] {
        assert!(
            values.contains(&expected),
            "SC load over-pruned: {expected} unreachable, saw {values:?}",
        );
    }
}
