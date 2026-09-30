#![deny(warnings, rust_2018_idioms)]

//! The whole-cell coherence floor of a typed lane `load()`, litmus by litmus.
//!
//! A peer's wide store `W` writes lanes A and B of one cell in one
//! single-copy-atomic event. The active thread observes `W` through lane B —
//! by reading it, or by writing lane B coherence-after it — and then loads
//! lane A. Reading lane A from before `W` closes the cycle
//! `W →co/rf observation →po L →fr W`, and whether the target forbids that
//! cycle depends only on whether it orders the observation before `L`. loom
//! models the target it is built for, so each outcome's expectation is that
//! target's verdict: AArch64 (and every target loom has no stronger rule for)
//! from herd7, x86 from TSO.
//!
//! The AArch64 verdicts are herd7 7.58 (`aarch64.cat`, `-variant mixed`) on a
//! 64-bit cell with 32-bit lanes; the shape carries to the 16-byte cell
//! unchanged, since no rule involved depends on the access size. The x86
//! verdicts follow from TSO: loads are ordered with loads and locked RMWs are
//! full barriers, but a store may pass a later load unless `mfence`, `xchg`
//! or another locked instruction intervenes.
//!
//! | Observation, then lane-A load                   | AArch64 | x86    |
//! |--------------------------------------------------|---------|--------|
//! | relaxed read of B                                | allowed | forbid |
//! | relaxed read of B, acquire load of A             | allowed | forbid |
//! | acquire read of B                                | forbid  | forbid |
//! | relaxed read of B, `fence(Acquire)`              | forbid  | forbid |
//! | relaxed RMW on B reading `W`'s half              | allowed | forbid |
//! | acquire RMW on B reading `W`'s half              | forbid  | forbid |
//! | a peer read B, released; we acquired             | forbid  | forbid |
//! | relaxed read of B, own relaxed store to B        | allowed | forbid |
//! | ... own release store, acquire load of A         | allowed | forbid |
//! | ... own store, `fence(SeqCst)`                   | forbid  | forbid |
//! | ... own `SeqCst` store, `SeqCst` load of A       | forbid  | forbid |
//! | uncommunicating `SeqCst` fences                  | allowed | allowed|
//!
//! The own-store rows read lane B first: that read is what makes the store
//! coherence-after the wide op in the model, which fixes modification order
//! only from what the writer has seen. On x86 that read alone floors.
//!
//! The floor binds every access to lane A the observation is ordered before,
//! not only a load: an RMW's read, a failed compare-exchange's read, and where
//! a store lands in lane A's modification order. A store is ordered after an
//! earlier read by an acquire on the read, an acquire fence between them, or
//! its own release (AArch64 `bob`: `[A];po`, `[R];po;[dmb.ld]`, `po;[L]`), and
//! on x86 always.
//!
//! | Observation, then lane-A access                  | AArch64 | x86    |
//! |--------------------------------------------------|---------|--------|
//! | acquire read of B, relaxed RMW of A              | forbid  | forbid |
//! | relaxed read of B, relaxed RMW of A              | allowed | forbid |
//! | acquire read of B, failed relaxed CAS of A       | forbid  | forbid |
//! | acquire read of B, relaxed store to A            | forbid  | forbid |
//! | relaxed read of B, relaxed store to A            | allowed | forbid |
//! | relaxed read of B, release store to A            | forbid  | forbid |

use loom::sync::atomic::{fence, AtomicU128, AtomicU32, Ordering, Ordering::*};
use loom::thread;

use std::collections::BTreeSet;
use std::sync::{Arc, Mutex};

const LANE_A: u128 = u64::MAX as u128;

/// The wide op: lane A = 1 and lane B = 1 in one event.
const WIDE: u128 = 1 | (1 << 64);

type Outcomes = Arc<Mutex<BTreeSet<(u64, u64)>>>;

/// Explore `body` to exhaustion against a peer's wide store, collecting what
/// it returns.
fn explore<F>(body: F) -> BTreeSet<(u64, u64)>
where
    F: Fn(&AtomicU128) -> Option<(u64, u64)> + Send + Sync + 'static,
{
    let seen: Outcomes = Default::default();
    let seen_ = seen.clone();
    loom::model(move || {
        let x = Arc::new(AtomicU128::new(0));
        // Split the cell into lane A / lane B.
        x.store_masked(LANE_A, 0, Relaxed);

        let w = {
            let x = x.clone();
            thread::spawn(move || x.store(WIDE, Relaxed))
        };
        let out = body(&x);
        w.join().unwrap();
        if let Some(out) = out {
            seen_.lock().unwrap().insert(out);
        }
    });
    let seen = seen.lock().unwrap().clone();
    seen
}

/// Read lane B, then lane A: `(b, a)`.
fn read_route(b_order: Ordering, fence_between: bool, a_order: Ordering) -> BTreeSet<(u64, u64)> {
    explore(move |x| {
        let b = x.lane_u64(8).load(b_order);
        if fence_between {
            fence(Acquire);
        }
        let a = x.lane_u64(0).load(a_order);
        Some((b, a))
    })
}

/// Read lane B, write it, then load lane A; keep only executions where the
/// read saw the wide op, which puts the own store coherence-after it:
/// `(b, a)`.
fn write_route(
    b_order: Ordering,
    fence_between: Option<Ordering>,
    a_order: Ordering,
) -> BTreeSet<(u64, u64)> {
    explore(move |x| {
        let b = x.lane_u64(8).load(Relaxed);
        x.lane_u64(8).store(2, b_order);
        if let Some(order) = fence_between {
            fence(order);
        }
        let a = x.lane_u64(0).load(a_order);
        (b == 1).then_some((b, a))
    })
}

const TRAVELLED: (u64, u64) = (1, 0);

/// Whether loom stands in for x86 (TSO) here; otherwise the AArch64 rules.
const X86: bool = cfg!(any(target_arch = "x86", target_arch = "x86_64"));

/// `outcome` is reachable exactly when the target allows it.
fn assert_target(seen: &BTreeSet<(u64, u64)>, outcome: (u64, u64), allowed: bool) {
    assert_eq!(
        seen.contains(&outcome),
        allowed,
        "{outcome:?} must be {} on this target: {seen:?}",
        if allowed { "reachable" } else { "unreachable" }
    );
}

// AArch64 RR-rlx: `LDR W0,[x+4]; LDR W2,[x]` — Sometimes. x86: loads stay in
// order — forbidden.
#[test]
fn relaxed_sibling_read_floors_only_on_x86() {
    let seen = read_route(Relaxed, false, Relaxed);
    assert_target(&seen, TRAVELLED, !X86);
}

// AArch64 RR-rlx-acqL: `LDR W0,[x+4]; LDAR W2,[x]` — Sometimes. An acquire on
// the floored load orders what follows it, not what precedes it.
#[test]
fn acquire_on_the_floored_load_alone_floors_only_on_x86() {
    let seen = read_route(Relaxed, false, Acquire);
    assert_target(&seen, TRAVELLED, !X86);
}

// AArch64 RR-acq: `LDAR W0,[x+4]; LDR W2,[x]` — Never.
#[test]
fn acquire_sibling_read_floors() {
    let seen = read_route(Acquire, false, Relaxed);
    assert!(!seen.contains(&TRAVELLED), "{seen:?}");
    assert!(seen.contains(&(1, 1)) && seen.contains(&(0, 0)), "{seen:?}");
}

// AArch64 RR-dmbld: `LDR W0,[x+4]; DMB ISHLD; LDR W2,[x]` — Never.
#[test]
fn acquire_fence_after_sibling_read_floors() {
    let seen = read_route(Relaxed, true, Relaxed);
    assert!(!seen.contains(&TRAVELLED), "{seen:?}");
}

// AArch64 RMW-rlx: `SWP W4,W0,[x+4]; LDR W2,[x]` — Sometimes. x86: a locked
// RMW — forbidden.
#[test]
fn relaxed_sibling_rmw_floors_only_on_x86() {
    let seen = explore(|x| {
        let b = x.lane_u64(8).swap(2, Relaxed);
        let a = x.lane_u64(0).load(Relaxed);
        Some((b, a))
    });
    assert_target(&seen, TRAVELLED, !X86);
}

// AArch64 RMW-acq: `SWPA W4,W0,[x+4]; LDR W2,[x]` — Never.
#[test]
fn acquire_sibling_rmw_floors() {
    let seen = explore(|x| {
        let b = x.lane_u64(8).swap(2, Acquire);
        let a = x.lane_u64(0).load(Relaxed);
        Some((b, a))
    });
    assert!(!seen.contains(&TRAVELLED), "{seen:?}");
}

// AArch64 RR-xrel: a peer reads lane B relaxed and releases a flag; this
// thread acquires the flag and reads lane A — Never. The release orders the
// peer's read before its flag store; the acquire orders the flag load before
// the lane-A load.
#[test]
fn a_peer_observation_reached_through_synchronization_floors() {
    let seen: Outcomes = Default::default();
    let seen_ = seen.clone();
    loom::model(move || {
        let x = Arc::new(AtomicU128::new(0));
        let flag = Arc::new(AtomicU32::new(0));
        x.store_masked(LANE_A, 0, Relaxed);

        let w = {
            let x = x.clone();
            thread::spawn(move || x.store(WIDE, Relaxed))
        };
        let r = {
            let (x, flag) = (x.clone(), flag.clone());
            thread::spawn(move || {
                if x.lane_u64(8).load(Relaxed) == 1 {
                    flag.store(1, Release);
                }
            })
        };
        if flag.load(Acquire) == 1 {
            let a = x.lane_u64(0).load(Relaxed);
            seen_.lock().unwrap().insert((1, a));
        }
        w.join().unwrap();
        r.join().unwrap();
    });
    let seen = seen.lock().unwrap().clone();
    assert!(!seen.contains(&TRAVELLED), "{seen:?}");
    assert!(seen.contains(&(1, 1)), "{seen:?}");
}

// AArch64 RWR-rlx: `LDR W0,[x+4]; STR W4,[x+4]; LDR W2,[x]` — Sometimes. x86:
// the store may sit in the store buffer, but the lane-B read before it is
// ordered — forbidden.
#[test]
fn own_sibling_store_after_relaxed_read_floors_only_on_x86() {
    let seen = write_route(Relaxed, None, Relaxed);
    assert_target(&seen, TRAVELLED, !X86);
}

// AArch64 RWR-rel-apr: `LDR; STLR W4,[x+4]; LDAPR W2,[x]` — Sometimes.
#[test]
fn own_release_store_then_acquire_load_floors_only_on_x86() {
    let seen = write_route(Release, None, Acquire);
    assert_target(&seen, TRAVELLED, !X86);
}

// AArch64 RWR-dmbsy: `LDR; STR W4,[x+4]; DMB ISH; LDR W2,[x]` — Never.
#[test]
fn seq_cst_fence_after_own_sibling_store_floors() {
    let seen = write_route(Relaxed, Some(SeqCst), Relaxed);
    assert!(!seen.contains(&TRAVELLED), "{seen:?}");
    assert!(seen.contains(&(1, 1)), "{seen:?}");
}

// AArch64 RWR-sc: `LDR; STLR W4,[x+4]; LDAR W2,[x]` — Never.
#[test]
fn own_seq_cst_store_then_seq_cst_load_floors() {
    let seen = write_route(SeqCst, None, SeqCst);
    assert!(!seen.contains(&TRAVELLED), "{seen:?}");
    assert!(seen.contains(&(1, 1)), "{seen:?}");
}

// The same rule through a preserving wide CAS, which floors the lane it
// carried through the store it read there: a relaxed read of the lane it
// wrote orders nothing on AArch64, so the carried lane may still read older.
#[test]
fn relaxed_read_of_a_preserving_op_floors_the_carried_lane_only_on_x86() {
    const O_MASK: u128 = (u32::MAX as u128) << 64;
    const OWNER: u128 = 0xABCD << 64;

    let seen: Outcomes = Default::default();
    let seen_ = seen.clone();
    loom::model(move || {
        let x = Arc::new(AtomicU128::new(0));

        let w = {
            let x = x.clone();
            thread::spawn(move || {
                x.lane_u32(8).store(0xABCD, Relaxed);
                x.compare_exchange_preserving(O_MASK, OWNER, 1 << 96, Relaxed, Relaxed)
                    .expect("uncontended CAS must succeed");
            })
        };
        let v = x.lane_u32(12).load(Relaxed);
        let o = x.lane_u32(8).load(Relaxed);
        w.join().unwrap();
        seen_.lock().unwrap().insert((u64::from(v), u64::from(o)));
    });
    let seen = seen.lock().unwrap().clone();
    assert_target(&seen, (1, 0), !X86);
}

// AArch64 SCF-noc: a peer reads lane B then runs `DMB ISH`; this thread runs
// `DMB ISH` then reads lane A — Sometimes, even with the peer's fence first in
// S. Two `SeqCst` fences with nothing communicated between them order neither
// thread's accesses for the other. A host (non-model) flag witnesses which
// fence the schedule committed first, so only the S-order that could wrongly
// floor is recorded.
#[test]
fn seq_cst_fences_without_communication_do_not_floor() {
    use std::sync::atomic::AtomicBool;

    let seen: Outcomes = Default::default();
    let seen_ = seen.clone();
    loom::model(move || {
        let x = Arc::new(AtomicU128::new(0));
        let peer_fenced = Arc::new(AtomicBool::new(false));
        x.store_masked(LANE_A, 0, Relaxed);

        let w = {
            let x = x.clone();
            thread::spawn(move || x.store(WIDE, Relaxed))
        };
        let r = {
            let (x, peer_fenced) = (x.clone(), peer_fenced.clone());
            thread::spawn(move || {
                let b = x.lane_u64(8).load(Relaxed);
                fence(SeqCst);
                peer_fenced.store(true, std::sync::atomic::Ordering::Relaxed);
                b
            })
        };
        fence(SeqCst);
        let after_peer = peer_fenced.load(std::sync::atomic::Ordering::Relaxed);
        let a = x.lane_u64(0).load(Relaxed);
        w.join().unwrap();
        let b = r.join().unwrap();
        if after_peer {
            seen_.lock().unwrap().insert((b, a));
        }
    });
    let seen = seen.lock().unwrap().clone();
    assert_target(&seen, TRAVELLED, true);
}

// The futex value↔presence Dekker. A waiter's whole-cell CAS (lane V == 2,
// install the queue in lane S) races a waker's release-only swap of lane V
// followed by an acquire load of lane S. AArch64 FUTEX-swpl-ldapr: `CASAL` |
// `SWPL W4,W0,[x]; LDAPR W6,[x+4]` — Sometimes: the swap reads the CAS's
// lane V yet the lane-S load misses its push (Rust `Release` swap then
// `Acquire` load under `+rcpc`). x86: the swap is `xchg`, a full barrier —
// forbidden.
#[test]
fn release_swap_then_acquire_presence_load_misses_push_only_on_aarch64() {
    const V: u128 = u64::MAX as u128;
    let seen: Outcomes = Default::default();
    let seen_ = seen.clone();
    loom::model(move || {
        let x = Arc::new(AtomicU128::new(2));
        // Split the cell into lane V (low qword) / lane S (high qword).
        x.store_masked(V, 2, Relaxed);

        let waiter = {
            let x = x.clone();
            thread::spawn(move || x.compare_exchange(2, 2 | (1 << 64), AcqRel, Relaxed).is_ok())
        };
        let v = x.lane_u64(0).swap(0, Release);
        let s = x.lane_u64(8).load(Acquire);
        let pushed = waiter.join().unwrap();
        if pushed {
            seen_.lock().unwrap().insert((v, s));
        }
    });
    let seen = seen.lock().unwrap().clone();
    // The swap read the waiter's lane V (2) yet the presence load saw no push.
    assert_target(&seen, (2, 0), !X86);
    assert!(seen.contains(&(2, 1)), "{seen:?}");
}

// The same Dekker with a `SeqCst` store as the publish and the push CAS
// running after the presence load, from a snapshot the waiter took before
// the store (as the futex push does). AArch64 FUTEX-stlr-ldapr: `CASAL` |
// `STLR W4,[x]; LDAPR W6,[x+4]` — Sometimes: the CAS may read lane V from
// before the store even though it lands after the presence load. x86: the
// store is `xchg`, so the presence load follows it, and the later-landing CAS
// follows it too and reads it — forbidden. This is the lane floor's dual:
// it restricts the RMW's read, not the load.
#[test]
fn seq_cst_store_then_presence_load_orders_a_later_push_only_on_x86() {
    const V: u128 = u64::MAX as u128;
    let seen: Outcomes = Default::default();
    let seen_ = seen.clone();
    loom::model(move || {
        let x = Arc::new(AtomicU128::new(0));
        x.store_masked(V, 0, Relaxed);

        let waiter = {
            let x = x.clone();
            thread::spawn(move || {
                let snap = x.load(Acquire);
                snap & V == 0 && x.compare_exchange(snap, snap | (1 << 64), AcqRel, Relaxed).is_ok()
            })
        };
        x.lane_u64(0).store(1, SeqCst);
        let s = x.lane_u64(8).load(Acquire);
        let pushed = waiter.join().unwrap();
        seen_.lock().unwrap().insert((u64::from(pushed), s));
    });
    let seen = seen.lock().unwrap().clone();
    // The push succeeded (it read lane V from before the store) yet the
    // presence load missed it.
    assert_target(&seen, (1, 0), !X86);
    assert!(seen.contains(&(1, 1)) && seen.contains(&(0, 0)), "{seen:?}");
}

// The dual with the presence load on another thread, one that happens-after
// the store's barrier (here through spawn, as the futex's two-notifier rig
// does): on x86 the load still follows the `xchg` in memory order, so the
// later-landing push reads the store.
#[test]
fn seq_cst_store_orders_a_later_push_for_a_reader_that_happens_after_it() {
    const V: u128 = u64::MAX as u128;
    let seen: Outcomes = Default::default();
    let seen_ = seen.clone();
    loom::model(move || {
        let x = Arc::new(AtomicU128::new(0));
        x.store_masked(V, 0, Relaxed);

        let waiter = {
            let x = x.clone();
            thread::spawn(move || {
                let snap = x.load(Acquire);
                snap & V == 0 && x.compare_exchange(snap, snap | (1 << 64), AcqRel, Relaxed).is_ok()
            })
        };
        x.lane_u64(0).store(1, SeqCst);
        let reader = {
            let x = x.clone();
            thread::spawn(move || x.lane_u64(8).load(Acquire))
        };
        let s = reader.join().unwrap();
        let pushed = waiter.join().unwrap();
        seen_.lock().unwrap().insert((u64::from(pushed), s));
    });
    let seen = seen.lock().unwrap().clone();
    assert_target(&seen, (1, 0), !X86);
    assert!(seen.contains(&(1, 1)) && seen.contains(&(0, 0)), "{seen:?}");
}

/// A peer stores lane A, then lane B at `b_order`; the active thread reads the
/// whole cell in one relaxed load: `(b, a)`.
fn wide_read_of_ordered_lane_stores(b_order: Ordering) -> BTreeSet<(u64, u64)> {
    wide_access_of_ordered_lane_stores(b_order, false, |x| x.load(Relaxed))
}

/// As above, the peer optionally fencing `Release` between its stores, and the
/// active thread reading through `read`.
fn wide_access_of_ordered_lane_stores(
    b_order: Ordering,
    fence_between: bool,
    read: fn(&AtomicU128) -> u128,
) -> BTreeSet<(u64, u64)> {
    let seen: Outcomes = Default::default();
    let seen_ = seen.clone();
    loom::model(move || {
        let x = Arc::new(AtomicU128::new(0));
        x.store_masked(LANE_A, 0, Relaxed);

        let w = {
            let x = x.clone();
            thread::spawn(move || {
                x.lane_u64(0).store(1, Relaxed);
                if fence_between {
                    fence(Release);
                }
                x.lane_u64(8).store(1, b_order);
            })
        };
        let v = read(&x);
        w.join().unwrap();
        seen_
            .lock()
            .unwrap()
            .insert(((v >> 64) as u64, v as u64));
    });
    let seen = seen.lock().unwrap().clone();
    seen
}

// One single-copy-atomic read of both lanes cannot take lane B's store and
// miss the lane-A store ordered before it: `W_A →bob W_B →rfe R →fre W_A` is
// an `ob` cycle in AArch64 (`STR; STLR` against one `LDP`/`LDR Q`), and x86
// commits stores in order to a load that reads both.
#[test]
fn wide_read_never_sees_a_release_lane_store_without_the_earlier_one() {
    let seen = wide_read_of_ordered_lane_stores(Release);
    assert_target(&seen, TRAVELLED, false);
    assert!(seen.contains(&(0, 0)) && seen.contains(&(1, 1)), "{seen:?}");
}

// Without the release, AArch64 lets the two stores commit out of order
// (`STR; STR` against one wide read — Sometimes); x86 still commits in order.
#[test]
fn wide_read_sees_relaxed_lane_stores_reordered_only_on_aarch64() {
    let seen = wide_read_of_ordered_lane_stores(Relaxed);
    assert_target(&seen, TRAVELLED, !X86);
}

// `fence(Release)` (`DMB ISH`) orders the earlier store as `STLR` would.
#[test]
fn wide_read_never_sees_a_fenced_lane_store_without_the_earlier_one() {
    let seen = wide_access_of_ordered_lane_stores(Relaxed, true, |x| x.load(Relaxed));
    assert_target(&seen, TRAVELLED, false);
}

// A wide RMW's read is the same single-copy-atomic event (`CASP`,
// `cmpxchg16b`).
#[test]
fn wide_rmw_never_sees_a_release_lane_store_without_the_earlier_one() {
    let seen = wide_access_of_ordered_lane_stores(Release, false, |x| x.fetch_add(0, Relaxed));
    assert_target(&seen, TRAVELLED, false);
    assert!(seen.contains(&(0, 0)) && seen.contains(&(1, 1)), "{seen:?}");
}

/// Read lane B at `b_order`, then run `access` on lane A: `(b, a)`, where `a`
/// is what the access read.
fn rmw_route(b_order: Ordering, access: fn(&AtomicU128) -> u64) -> BTreeSet<(u64, u64)> {
    explore(move |x| {
        let b = x.lane_u64(8).load(b_order);
        Some((b, access(x)))
    })
}

// AArch64 RMW-after-acq: `LDAR W0,[x+4]; LDADD W4,W2,[x]` — Never.
#[test]
fn acquire_sibling_read_floors_a_later_rmw() {
    let seen = rmw_route(Acquire, |x| x.lane_u64(0).fetch_add(0, Relaxed));
    assert!(!seen.contains(&TRAVELLED), "{seen:?}");
    assert!(seen.contains(&(1, 1)) && seen.contains(&(0, 0)), "{seen:?}");
}

// AArch64 RMW-after-rlx: `LDR W0,[x+4]; LDADD W4,W2,[x]` — Sometimes. x86:
// loads stay in order, and the RMW's read is one.
#[test]
fn relaxed_sibling_read_floors_a_later_rmw_only_on_x86() {
    let seen = rmw_route(Relaxed, |x| x.lane_u64(0).fetch_add(0, Relaxed));
    assert_target(&seen, TRAVELLED, !X86);
}

// A failed compare-exchange is a load at its failure ordering: `LDAR; CAS`
// failing — Never.
#[test]
fn acquire_sibling_read_floors_a_later_failed_cas() {
    let seen = rmw_route(Acquire, |x| {
        x.lane_u64(0)
            .compare_exchange(7, 8, Relaxed, Relaxed)
            .expect_err("lane A never holds 7")
    });
    assert!(!seen.contains(&TRAVELLED), "{seen:?}");
    assert!(seen.contains(&(1, 1)) && seen.contains(&(0, 0)), "{seen:?}");
}

/// Read lane B at `b_order`, then store 2 to lane A at `a_order`; once the
/// wide op joins, read where lane A ended: `(b, final a)`.
fn store_route(b_order: Ordering, a_order: Ordering) -> BTreeSet<(u64, u64)> {
    let seen: Outcomes = Default::default();
    let seen_ = seen.clone();
    loom::model(move || {
        let x = Arc::new(AtomicU128::new(0));
        x.store_masked(LANE_A, 0, Relaxed);

        let w = {
            let x = x.clone();
            thread::spawn(move || x.store(WIDE, Relaxed))
        };
        let b = x.lane_u64(8).load(b_order);
        x.lane_u64(0).store(2, a_order);
        w.join().unwrap();
        let a = x.lane_u64(0).load(Relaxed);
        seen_.lock().unwrap().insert((b, a));
    });
    let seen = seen.lock().unwrap().clone();
    seen
}

/// Lane A ended at the wide op's half although the store read its lane B
/// first: the store landed modification-order-before the wide op.
const STORE_TRAVELLED: (u64, u64) = (1, 1);

// AArch64 W-after-acq: `LDAR W0,[x+4]; STR W4,[x]` with `W` coherence-after
// the store — Never.
#[test]
fn acquire_sibling_read_floors_a_later_store() {
    let seen = store_route(Acquire, Relaxed);
    assert!(!seen.contains(&STORE_TRAVELLED), "{seen:?}");
    assert!(seen.contains(&(1, 2)) && seen.contains(&(0, 1)), "{seen:?}");
}

// AArch64 W-after-rlx: `LDR W0,[x+4]; STR W4,[x]` — Sometimes: nothing orders
// the read before the store. x86: a store never passes an earlier load.
#[test]
fn relaxed_sibling_read_floors_a_later_store_only_on_x86() {
    let seen = store_route(Relaxed, Relaxed);
    assert_target(&seen, STORE_TRAVELLED, !X86);
}

// AArch64 W-rel-after-rlx: `LDR W0,[x+4]; STLR W4,[x]` — Never: a release
// store is ordered after everything before it.
#[test]
fn relaxed_sibling_read_floors_a_later_release_store() {
    let seen = store_route(Relaxed, Release);
    assert!(!seen.contains(&STORE_TRAVELLED), "{seen:?}");
}
