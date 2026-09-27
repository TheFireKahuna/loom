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
