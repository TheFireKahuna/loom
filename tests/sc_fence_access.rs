#![deny(warnings, rust_2018_idioms)]

//! Interaction between a `SeqCst` *fence* in one thread and a `SeqCst`
//! *access* in another — the mixed part of the C++20 SC total order S
//! (`psc_base`, the `[Esc] ; scb ; sb? ; [Fsc]` and `[Fsc] ; sb? ; scb ;
//! [Esc]` clauses; equivalently the fence rules [atomics.order] p4–p7).
//!
//! The pure cases are covered elsewhere: SC access ↔ SC access in
//! `tests/litmus.rs`, and SC fence ↔ SC fence in `tests/fence.rs`. These
//! tests pin the cases where one endpoint is a fence and the other is an SC
//! access — the outcomes that plain acquire/release (and, before the two SC
//! orders were unified, loom) leave reachable.

use loom::sync::atomic::{fence, AtomicUsize};
use loom::thread;

use std::collections::HashSet;
use std::sync::atomic::Ordering::{Relaxed, SeqCst};
use std::sync::{Arc, Mutex};

/// Store-buffering with the two arms spelled differently: one thread orders
/// its relaxed store against its relaxed load with a `fence(SeqCst)`, the
/// other uses bare `SeqCst` accesses (no fence). Under a single SC order S the
/// both-miss `(0, 0)` outcome is forbidden — the fence in T1 must order against
/// the SC accesses in T2. This is the canonical case the unified S closes:
/// with the fence order and the access order kept separate it stays reachable.
#[test]
fn sb_fence_vs_sc_access() {
    let values = Arc::new(Mutex::new(HashSet::new()));
    let values_ = values.clone();
    loom::model(move || {
        let x = Arc::new(AtomicUsize::new(0));
        let y = Arc::new(AtomicUsize::new(0));

        // Fence arm: relaxed store, SC fence, relaxed load.
        let t1 = {
            let (x, y) = (x.clone(), y.clone());
            thread::spawn(move || {
                x.store(1, Relaxed);
                fence(SeqCst);
                y.load(Relaxed)
            })
        };

        // Access arm: bare SC store and SC load, no fence.
        y.store(1, SeqCst);
        let r2 = x.load(SeqCst);

        let r1 = t1.join().unwrap();
        values.lock().unwrap().insert((r1, r2));
    });
    let values = values_.lock().unwrap();
    assert!(
        !values.contains(&(0, 0)),
        "a SeqCst fence must order against a SeqCst access: (0, 0) forbidden, saw {values:?}",
    );
    // The three SC-consistent outcomes stay reachable — the fence adds ordering,
    // it does not collapse the interleaving.
    assert!(values.contains(&(1, 1)), "saw {values:?}");
    assert!(values.contains(&(0, 1)), "saw {values:?}");
    assert!(values.contains(&(1, 0)), "saw {values:?}");
}

/// The same edge with the arms swapped (SC accesses in the spawned thread, the
/// fence in the main thread) — guards that promotion (p5/p7) and the fence-read
/// scope (p4/p6) are both wired symmetrically, not just for one thread role.
#[test]
fn sb_sc_access_vs_fence() {
    let values = Arc::new(Mutex::new(HashSet::new()));
    let values_ = values.clone();
    loom::model(move || {
        let x = Arc::new(AtomicUsize::new(0));
        let y = Arc::new(AtomicUsize::new(0));

        // Access arm.
        let t1 = {
            let (x, y) = (x.clone(), y.clone());
            thread::spawn(move || {
                x.store(1, SeqCst);
                y.load(SeqCst)
            })
        };

        // Fence arm.
        y.store(1, Relaxed);
        fence(SeqCst);
        let r2 = x.load(Relaxed);

        let r1 = t1.join().unwrap();
        values.lock().unwrap().insert((r1, r2));
    });
    let values = values_.lock().unwrap();
    assert!(
        !values.contains(&(0, 0)),
        "a SeqCst access must order against a SeqCst fence: (0, 0) forbidden, saw {values:?}",
    );
    assert!(values.contains(&(1, 1)), "saw {values:?}");
    assert!(values.contains(&(0, 1)), "saw {values:?}");
    assert!(values.contains(&(1, 0)), "saw {values:?}");
}

/// IRIW (independent reads of independent writes) where one of the two readers
/// splits its pair of relaxed loads with a `SeqCst` fence instead of using SC
/// loads. The writers are SC accesses. Under one SC order S the fence orders
/// the fence-reader's two loads against the SC writes, so the two readers must
/// still agree on which write came first — the `(x=1, y=0)` vs `(y=1, x=0)`
/// disagreement is forbidden. There is no rf edge to carry acquire/release
/// synchronization here, so this fails outright unless the fence and the SC
/// accesses share one order (it is the fence-read rule p4/p6 pulling an SC
/// *access* into the fence's scope).
#[test]
fn iriw_fence_reader() {
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
        // Pure SC reader: sees the order as x-then-y or not.
        let r1 = {
            let (x, y) = (x.clone(), y.clone());
            thread::spawn(move || (x.load(SeqCst), y.load(SeqCst)))
        };
        // Fence reader: two relaxed loads split by an SC fence.
        let r2 = {
            let (x, y) = (x.clone(), y.clone());
            thread::spawn(move || {
                let c = y.load(Relaxed);
                fence(SeqCst);
                let d = x.load(Relaxed);
                (c, d)
            })
        };

        wx.join().unwrap();
        wy.join().unwrap();
        let (a, b) = r1.join().unwrap();
        let (c, d) = r2.join().unwrap();

        // Forbidden: r1 saw x before y while r2 saw y before x.
        if (a, b, c, d) == (1, 0, 1, 0) {
            *seen.lock().unwrap() = true;
        }
    });
    assert!(
        !*seen_.lock().unwrap(),
        "SC forbids the IRIW disagreement even when one reader orders with a fence",
    );
}

/// Completeness guard for the fence-read scope: the restriction keys on
/// SC-ranked stores only. A plain relaxed store whose thread never fences is
/// never SC-ranked, so a load after an unrelated `SeqCst` fence must still be
/// free to read the stale *or* the fresh value — the fence must not manufacture
/// coherence against a write that is not in S.
#[test]
fn fence_does_not_over_restrict_relaxed_store() {
    let values = Arc::new(Mutex::new(HashSet::new()));
    let values_ = values.clone();
    loom::model(move || {
        let x = Arc::new(AtomicUsize::new(0));

        let t1 = {
            let x = x.clone();
            thread::spawn(move || {
                // No fence here: this store never enters S.
                x.store(1, Relaxed);
            })
        };

        fence(SeqCst);
        let r = x.load(Relaxed);

        t1.join().unwrap();
        values.lock().unwrap().insert(r);
    });
    let values = values_.lock().unwrap();
    assert!(values.contains(&0), "fresh-only: fence wrongly forced visibility, saw {values:?}");
    assert!(values.contains(&1), "stale-only: saw {values:?}");
}

/// Completeness guard for promotion: a `SeqCst` fence in the *creating* thread
/// promotes that thread's stores — including a cell's initial store, which is
/// modification-order-minimal. Promoting it must stay inert (it supersedes
/// nothing), so a racing relaxed load still reaches every value. This pins the
/// hazard that a late-promoted, mo-early store must never act as a newer SC
/// witness.
#[test]
fn init_store_promotion_is_inert() {
    let values = Arc::new(Mutex::new(HashSet::new()));
    let values_ = values.clone();
    loom::model(move || {
        // `x` is created on the main thread; its initial store's creator is the
        // main thread, which then executes a `SeqCst` fence.
        let x = Arc::new(AtomicUsize::new(0));

        let t1 = {
            let x = x.clone();
            thread::spawn(move || x.load(Relaxed))
        };

        x.store(1, Relaxed);
        fence(SeqCst);

        let r = t1.join().unwrap();
        values.lock().unwrap().insert(r);
    });
    let values = values_.lock().unwrap();
    assert!(values.contains(&0), "saw {values:?}");
    assert!(values.contains(&1), "saw {values:?}");
}
