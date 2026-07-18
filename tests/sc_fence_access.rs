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
//!
//! The tail of the file also exercises the pure fence↔fence (`psc_F`) path
//! carried by loom's `seq_cst_causality` frontier, in both directions: that it
//! forbids the cyclic outcomes (IRIW with reader-side fences, including
//! `Relaxed`/`Release` writers that never enter S), and — the harder guard —
//! that it does NOT over-constrain: genuinely independent fences stay
//! unordered, and it draws the `psc`-cycle line exactly even when the closing
//! edge routes through release/acquire `hb`.

use loom::sync::atomic::{fence, AtomicUsize};
use loom::thread;

use std::collections::HashSet;
use std::sync::atomic::Ordering::{Acquire, Relaxed, Release, SeqCst};
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

/// IRIW where BOTH readers split their two `Relaxed` loads with a `SeqCst`
/// fence and the writers are plain `Relaxed` — no store ever enters S as an SC
/// write. RC11 still forbids the `(1,0,1,0)` disagreement: the two fences close
/// a `psc_F` cycle `F —po→ r(=0) —fr→ W —rf→ r'(=1) —po→ F'` and its mirror,
/// which is mode-independent (no `sw` edge appears in it). The SC-ranked
/// store-exclusion never fires here (nothing is SC-ranked), so this is enforced
/// entirely by the `seq_cst_causality` frontier. The existing `iriw_fence_reader`
/// covers SC *writers*; this pins the pure fence↔fence case.
#[test]
fn iriw_both_fence_readers_relaxed_writes() {
    let seen = Arc::new(Mutex::new(false));
    let seen_ = seen.clone();
    loom::model(move || {
        let x = Arc::new(AtomicUsize::new(0));
        let y = Arc::new(AtomicUsize::new(0));

        let wx = {
            let x = x.clone();
            thread::spawn(move || x.store(1, Relaxed))
        };
        let wy = {
            let y = y.clone();
            thread::spawn(move || y.store(1, Relaxed))
        };
        let r1 = {
            let (x, y) = (x.clone(), y.clone());
            thread::spawn(move || {
                let a = x.load(Relaxed);
                fence(SeqCst);
                let b = y.load(Relaxed);
                (a, b)
            })
        };
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

        if (a, b, c, d) == (1, 0, 1, 0) {
            *seen.lock().unwrap() = true;
        }
    });
    assert!(
        !*seen_.lock().unwrap(),
        "SC fences between both readers' loads must forbid IRIW disagreement \
         even with relaxed writes",
    );
}

/// Same shape as above but with `Release` writes and `Relaxed` reads — still no
/// SC store anywhere, so still forbidden purely by the fence↔fence frontier.
#[test]
fn iriw_both_fence_readers_release_writes() {
    let seen = Arc::new(Mutex::new(false));
    let seen_ = seen.clone();
    loom::model(move || {
        let x = Arc::new(AtomicUsize::new(0));
        let y = Arc::new(AtomicUsize::new(0));

        let wx = {
            let x = x.clone();
            thread::spawn(move || x.store(1, Release))
        };
        let wy = {
            let y = y.clone();
            thread::spawn(move || y.store(1, Release))
        };
        let r1 = {
            let (x, y) = (x.clone(), y.clone());
            thread::spawn(move || {
                let a = x.load(Relaxed);
                fence(SeqCst);
                let b = y.load(Relaxed);
                (a, b)
            })
        };
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

        if (a, b, c, d) == (1, 0, 1, 0) {
            *seen.lock().unwrap() = true;
        }
    });
    assert!(
        !*seen_.lock().unwrap(),
        "SC fences between both readers' loads must forbid IRIW disagreement \
         with release writes",
    );
}

/// Positive control for the two tests above: with the reader fences REMOVED
/// (plain release/acquire, no SC anywhere), RC11 ALLOWS the IRIW disagreement,
/// so loom MUST reach `(1,0,1,0)`. This proves the exploration can find the
/// outcome and that it is the fences — not an exploration shortfall — that
/// forbid it above.
#[test]
fn iriw_no_fences_reaches_disagreement() {
    let seen = Arc::new(Mutex::new(false));
    let seen_ = seen.clone();
    loom::model(move || {
        let x = Arc::new(AtomicUsize::new(0));
        let y = Arc::new(AtomicUsize::new(0));

        let wx = {
            let x = x.clone();
            thread::spawn(move || x.store(1, Release))
        };
        let wy = {
            let y = y.clone();
            thread::spawn(move || y.store(1, Release))
        };
        let r1 = {
            let (x, y) = (x.clone(), y.clone());
            thread::spawn(move || (x.load(Relaxed), y.load(Relaxed)))
        };
        let r2 = {
            let (x, y) = (x.clone(), y.clone());
            thread::spawn(move || (y.load(Relaxed), x.load(Relaxed)))
        };

        wx.join().unwrap();
        wy.join().unwrap();
        let (a, b) = r1.join().unwrap();
        let (c, d) = r2.join().unwrap();

        if (a, b, c, d) == (1, 0, 1, 0) {
            *seen.lock().unwrap() = true;
        }
    });
    assert!(
        *seen_.lock().unwrap(),
        "control: without fences the IRIW disagreement must be reachable, else \
         the 'forbidden' results above are not trustworthy",
    );
}

/// Frontier must NOT over-order two SC fences with no coherence path between
/// them.
///
/// ```text
/// Producer: data.store(1, Relaxed); flag.store(1, Release)
/// T1:       r = flag.load(Acquire); fence(SeqCst)        // T1 may see data=1
/// T2:       fence(SeqCst); d = data.load(Relaxed)        // d=0 must be legal
/// ```
///
/// T1 acquires `flag`, so it may observe `data=1`, and its fence pushes that
/// view into the frontier; T2's fence then absorbs the frontier. But there is
/// no `hb;eco;hb` between the two fences (T1 has nothing after its fence for an
/// `eco` step to reach), so RC11 does not order them and does not force T2's
/// load. `(r=1, d=0)` is legal and MUST stay reachable.
#[test]
fn frontier_does_not_over_order_independent_fences() {
    let values = Arc::new(Mutex::new(HashSet::new()));
    let values_ = values.clone();
    loom::model(move || {
        let data = Arc::new(AtomicUsize::new(0));
        let flag = Arc::new(AtomicUsize::new(0));

        let prod = {
            let (data, flag) = (data.clone(), flag.clone());
            thread::spawn(move || {
                data.store(1, Relaxed);
                flag.store(1, Release);
            })
        };
        let t1 = {
            let flag = flag.clone();
            thread::spawn(move || {
                let r = flag.load(Acquire);
                fence(SeqCst);
                r
            })
        };
        let t2 = {
            let data = data.clone();
            thread::spawn(move || {
                fence(SeqCst);
                data.load(Relaxed)
            })
        };

        prod.join().unwrap();
        let r = t1.join().unwrap();
        let d = t2.join().unwrap();
        values.lock().unwrap().insert((r, d));
    });
    let values = values_.lock().unwrap();
    assert!(
        values.contains(&(1, 0)),
        "over-constraint: (r=1, d=0) is legal (no psc edge) yet unreachable — \
         the frontier is manufacturing fence ordering. saw {values:?}",
    );
}

/// Same, across a three-fence chain: the frontier accrues causality in
/// execution order, so an earlier unrelated fence's view could transitively
/// reach a later fence. A relaxed reader behind a third fence must still be free
/// to miss a value only an earlier, unrelated fenced thread observed.
#[test]
fn frontier_does_not_leak_transitively() {
    let values = Arc::new(Mutex::new(HashSet::new()));
    let values_ = values.clone();
    loom::model(move || {
        let data = Arc::new(AtomicUsize::new(0));
        let flag = Arc::new(AtomicUsize::new(0));

        let prod = {
            let (data, flag) = (data.clone(), flag.clone());
            thread::spawn(move || {
                data.store(1, Relaxed);
                flag.store(1, Release);
            })
        };
        let t1 = {
            let flag = flag.clone();
            thread::spawn(move || {
                let r = flag.load(Acquire);
                fence(SeqCst);
                r
            })
        };
        let t2 = thread::spawn(|| fence(SeqCst));
        let t3 = {
            let data = data.clone();
            thread::spawn(move || {
                fence(SeqCst);
                data.load(Relaxed)
            })
        };

        prod.join().unwrap();
        let r = t1.join().unwrap();
        t2.join().unwrap();
        let d = t3.join().unwrap();
        values.lock().unwrap().insert((r, d));
    });
    let values = values_.lock().unwrap();
    assert!(
        values.contains(&(1, 0)),
        "transitive over-constraint: (r=1, d=0) legal but unreachable across a \
         three-fence chain. saw {values:?}",
    );
}

/// The precise `psc`-cycle line, closing through release/acquire `hb`. A
/// `Relaxed` `done` signal orders the two fences via `psc_F` (relaxed `rf` is in
/// `eco`, flanked by `po`) WITHOUT creating any `hb`:
///
/// ```text
/// Producer: data.store(1, Relaxed); flag.store(1, Release)
/// T1:       r = flag.load(Acquire); fence(SeqCst); done.store(1, Relaxed)
/// T2:       obs = done.load(Relaxed); fence(SeqCst); d = data.load(Relaxed)
/// ```
///
/// `(r=1, obs=1, d=0)` is FORBIDDEN by a `psc` cycle:
///   - `F1 —hb→ Wdone —rf→ Rdone —hb→ F2`  (needs obs=1)  ⇒ F1 psc_F F2
///   - `F2 —hb→ Rd —fr→ Wd1 —hb→ F1`        (Wd1—hb→F1 needs r=1, via the
///     release/acquire on `flag`)          ⇒ F2 psc_F F1
/// The cycle exists ONLY when `r=1`; with `r=0` the closing edge is gone, so
/// `(0,1,0)` stays reachable. Forbidding `(1,1,0)` while allowing `(0,1,0)` and
/// `(1,1,1)` is the exact cycle signature — a coarse "fence = barrier"
/// over-approximation would forbid all three. This is the publish(Release) →
/// fence → acquire-consumer shape of the hazard-pointer reservation fences.
#[test]
fn frontier_matches_psc_cycle_through_hb() {
    let values = Arc::new(Mutex::new(HashSet::new()));
    let values_ = values.clone();
    loom::model(move || {
        let data = Arc::new(AtomicUsize::new(0));
        let flag = Arc::new(AtomicUsize::new(0));
        let done = Arc::new(AtomicUsize::new(0));

        let prod = {
            let (data, flag) = (data.clone(), flag.clone());
            thread::spawn(move || {
                data.store(1, Relaxed);
                flag.store(1, Release);
            })
        };
        let t1 = {
            let (flag, done) = (flag.clone(), done.clone());
            thread::spawn(move || {
                let r = flag.load(Acquire);
                fence(SeqCst);
                done.store(1, Relaxed);
                r
            })
        };
        let t2 = {
            let (data, done) = (data.clone(), done.clone());
            thread::spawn(move || {
                let obs = done.load(Relaxed);
                fence(SeqCst);
                let d = data.load(Relaxed);
                (obs, d)
            })
        };

        prod.join().unwrap();
        let r = t1.join().unwrap();
        let (obs, d) = t2.join().unwrap();
        values.lock().unwrap().insert((r, obs, d));
    });
    let values = values_.lock().unwrap();
    assert!(
        !values.contains(&(1, 1, 0)),
        "unsound: loom reached the psc-cyclic (r=1, obs=1, d=0). saw {values:?}",
    );
    assert!(
        values.contains(&(0, 1, 0)),
        "over-constraint: (0,1,0) is legal (r=0 breaks the F2→F1 edge). \
         saw {values:?}",
    );
    assert!(
        values.contains(&(1, 1, 1)),
        "over-constraint: (1,1,1) is legal (d=1, no fr edge). saw {values:?}",
    );
}

/// Calibration for the frontier-faithfulness tests above: a genuine SB+fences
/// (Dekker) between two fences — the StoreLoad shape of every SC-fence site in
/// this workspace — where `psc_F` DOES order the fences and forbids the
/// both-miss. Proves the frontier actually forbids when a coherence path is
/// present, so the "must stay reachable" assertions above are real
/// distinguishers, not a checker that never forbids anything.
#[test]
fn sb_both_fences_forbids_both_miss() {
    let values = Arc::new(Mutex::new(HashSet::new()));
    let values_ = values.clone();
    loom::model(move || {
        let data = Arc::new(AtomicUsize::new(0));
        let flag = Arc::new(AtomicUsize::new(0));

        let t1 = {
            let (data, flag) = (data.clone(), flag.clone());
            thread::spawn(move || {
                data.store(1, Relaxed);
                fence(SeqCst);
                flag.load(Relaxed)
            })
        };
        let t2 = {
            let (data, flag) = (data.clone(), flag.clone());
            thread::spawn(move || {
                flag.store(1, Relaxed);
                fence(SeqCst);
                data.load(Relaxed)
            })
        };

        let f = t1.join().unwrap();
        let d = t2.join().unwrap();
        values.lock().unwrap().insert((f, d));
    });
    let values = values_.lock().unwrap();
    assert!(
        !values.contains(&(0, 0)),
        "SB+fences both-miss (0,0) must be forbidden. saw {values:?}",
    );
    assert!(values.contains(&(1, 0)), "saw {values:?}");
    assert!(values.contains(&(0, 1)), "saw {values:?}");
    assert!(values.contains(&(1, 1)), "saw {values:?}");
}
