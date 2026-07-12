//! Mixed-size (sub-word) atomic sub-locations.
//!
//! A single `AtomicU128` cell may be accessed at sub-word granularity — an
//! aligned lane written on its own while the rest of the word is untouched
//! (spec carve-out #5: `lse2` single-copy atomicity + per-byte coherence).
//! loom models the cell as a set of independently-coherent regions. These
//! tests pin the two properties that make that model faithful:
//!
//!  1. **Lane independence** — two masked writes to different lanes may be
//!     observed in either order (a welded single ring wrongly serializes them).
//!  2. **Wide-op single-copy atomicity** — a full-width store/CAS is seen
//!     all-or-none; a 128-bit load never tears one, even on a split cell.
//!
//! And the two together: the consistency that keeps wide ops atomic must not
//! over-couple genuinely independent lanes.

#![deny(warnings, rust_2018_idioms)]

use loom::sync::atomic::AtomicU128;
use loom::thread;

use std::collections::HashSet;
use std::sync::atomic::{
    AtomicUsize,
    Ordering::{Relaxed, SeqCst},
};
use std::sync::{Arc, Mutex};

// Lane A = low 64 bits, lane B = high 64 bits.
const LANE_A: u128 = u64::MAX as u128;
const LANE_B: u128 = (u64::MAX as u128) << 64;

// The ntlib futex-word shape: value ⊂ high, and a disjoint low half.
const VALUE: u128 = (u32::MAX as u128) << 96; // bits 96..128
const HIGH: u128 = (u64::MAX as u128) << 64; // bits 64..128 (owner+value)
const S_MASK: u128 = u64::MAX as u128; // bits 0..64

/// Two masked stores to different lanes, program-ordered A then B, may be
/// observed reordered by a reader that reads lane B fresh but lane A still
/// stale. That outcome is reachable **only** because the lanes are
/// independently coherent — a single welded ring (each masked write an RMW
/// snapshot of the whole word) forbids reading an older lane-A value once the
/// newer lane-B write has been seen.
#[test]
fn masked_lanes_reorder_independently() {
    let seen = Arc::new(Mutex::new(HashSet::new()));
    let seen_ = seen.clone();
    loom::model(move || {
        let x = Arc::new(AtomicU128::new(0));

        let w = {
            let x = x.clone();
            thread::spawn(move || {
                x.store_masked(LANE_A, 1, Relaxed);
                x.store_masked(LANE_B, 1 << 64, Relaxed);
            })
        };

        // Read lane B, then lane A.
        let b = (x.load_masked(LANE_B, Relaxed) >> 64) as u64;
        let a = x.load_masked(LANE_A, Relaxed) as u64;

        w.join().unwrap();
        seen.lock().unwrap().insert((a, b));
    });

    let seen = seen_.lock().unwrap();
    // Lane B fresh (1) while lane A is still stale (0): the reorder.
    assert!(
        seen.contains(&(0, 1)),
        "lane independence not explored; a welded ring would forbid it: {:?}",
        *seen
    );
    // Sanity: the ordered observations are of course also reachable.
    assert!(seen.contains(&(0, 0)));
    assert!(seen.contains(&(1, 1)));
}

/// A full-width store to a **split** cell is single-copy-atomic: a concurrent
/// 128-bit load never sees one lane updated and the other not. The cell is
/// first split into lanes by a masked op, so this exercises the wide-op
/// consistency path (shared `op_id` across siblings), not the trivial
/// single-region case.
#[test]
fn wide_store_never_tears() {
    let seen = Arc::new(Mutex::new(HashSet::new()));
    let seen_ = seen.clone();
    loom::model(move || {
        let x = Arc::new(AtomicU128::new(0));

        // Split the cell into lane A / lane B.
        x.store_masked(LANE_A, 0, Relaxed);

        let w = {
            let x = x.clone();
            thread::spawn(move || {
                // Wide store: both lanes set in one op.
                x.store(1u128 | (1u128 << 64), Relaxed);
            })
        };

        let v = x.load(Relaxed);
        let a = (v & 1) as u64;
        let b = ((v >> 64) & 1) as u64;

        w.join().unwrap();
        seen.lock().unwrap().insert((a, b));
    });

    let seen = seen_.lock().unwrap();
    assert!(!seen.contains(&(1, 0)), "torn wide store observed: {:?}", *seen);
    assert!(!seen.contains(&(0, 1)), "torn wide store observed: {:?}", *seen);
    // Both all-or-none outcomes are reachable.
    assert!(seen.contains(&(0, 0)) && seen.contains(&(1, 1)), "{:?}", *seen);
}

/// Same, for a full-width compare-exchange (RMW) on a split cell: the wide CAS
/// commits every lane or none, so a concurrent 128-bit load never tears it.
#[test]
fn wide_cas_never_tears() {
    let seen = Arc::new(Mutex::new(HashSet::new()));
    let seen_ = seen.clone();
    loom::model(move || {
        let x = Arc::new(AtomicU128::new(0));

        // Split into lanes.
        x.store_masked(LANE_A, 0, Relaxed);

        let w = {
            let x = x.clone();
            thread::spawn(move || {
                let _ = x.compare_exchange(0, 1u128 | (1u128 << 64), Relaxed, Relaxed);
            })
        };

        let v = x.load(Relaxed);
        let a = (v & 1) as u64;
        let b = ((v >> 64) & 1) as u64;

        w.join().unwrap();
        seen.lock().unwrap().insert((a, b));
    });

    let seen = seen_.lock().unwrap();
    assert!(!seen.contains(&(1, 0)), "torn wide CAS observed: {:?}", *seen);
    assert!(!seen.contains(&(0, 1)), "torn wide CAS observed: {:?}", *seen);
    assert!(seen.contains(&(0, 0)) && seen.contains(&(1, 1)), "{:?}", *seen);
}

/// The wide-op consistency filter must not over-couple **independent** lanes:
/// two disjoint masked stores, observed through a single full-width load, are
/// still seen in all four combinations. (The filter keys on shared `op_id`;
/// distinct ops to disjoint lanes share none, so it never links them.)
#[test]
fn full_load_keeps_disjoint_lanes_independent() {
    let seen = Arc::new(Mutex::new(HashSet::new()));
    let seen_ = seen.clone();
    loom::model(move || {
        let x = Arc::new(AtomicU128::new(0));

        let wa = {
            let x = x.clone();
            thread::spawn(move || x.store_masked(VALUE, 1 << 96, Relaxed))
        };
        let ws = {
            let x = x.clone();
            thread::spawn(move || x.store_masked(S_MASK, 1, Relaxed))
        };

        let v = x.load(Relaxed);
        let value = ((v >> 96) & 1) as u64;
        let s = (v & 1) as u64;

        wa.join().unwrap();
        ws.join().unwrap();
        seen.lock().unwrap().insert((value, s));
    });

    let seen = seen_.lock().unwrap();
    for combo in [(0, 0), (1, 0), (0, 1), (1, 1)] {
        assert!(
            seen.contains(&combo),
            "disjoint lanes over-coupled; missing {:?} in {:?}",
            combo,
            *seen
        );
    }
}

/// The ntlib nesting: the value lane (bits 96..128) is written both on its own
/// and, atomically with the owner lane, under the wider HIGH mask (bits
/// 64..128) — VALUE ⊂ HIGH. This drives the refining split across a nested
/// mask and a masked-CAS spanning two regions, and must yield only coherent
/// snapshots (never a value bit the code never wrote).
#[test]
fn nested_value_within_high() {
    let seen = Arc::new(Mutex::new(HashSet::new()));
    let seen_ = seen.clone();
    loom::model(move || {
        let x = Arc::new(AtomicU128::new(0));

        // Value lane written alone.
        let wv = {
            let x = x.clone();
            thread::spawn(move || x.store_masked(VALUE, 2 << 96, Relaxed))
        };

        // Owner+value written together (HIGH), overlapping the value lane:
        // owner = 1, value = 1.
        let _ = x.compare_exchange_masked(HIGH, 0, (1 << 64) | (1 << 96), Relaxed, Relaxed);

        let v = x.load(Relaxed);
        let value = (v >> 96) as u64;
        let owner = ((v >> 64) & (u32::MAX as u128)) as u64;

        wv.join().unwrap();
        seen.lock().unwrap().insert((owner, value));
    });

    let seen = seen_.lock().unwrap();
    // Every observed value bit must be one actually written: 0 (init), 1 (HIGH
    // CAS), or 2 (masked store). Owner is 0 (init) or 1 (HIGH CAS). No torn or
    // invented lane values.
    for &(owner, value) in seen.iter() {
        assert!(owner == 0 || owner == 1, "invented owner {}: {:?}", owner, *seen);
        assert!(
            value == 0 || value == 1 || value == 2,
            "invented value {}: {:?}",
            value,
            *seen
        );
        // The HIGH CAS writes owner and value together: if the owner is set,
        // it was the CAS, so the value it wrote (1) must be visible too — unless
        // the later masked store (2) overwrote just the value lane. Owner set
        // therefore implies value ∈ {1, 2}, never 0.
        if owner == 1 {
            assert!(value == 1 || value == 2, "torn HIGH CAS: {:?}", *seen);
        }
    }
}

/// The pruning must be **sound**: mask-filtering the dependence check must not
/// let a thread's later access to one lane shadow its earlier access to
/// another, or a peer op on the earlier lane is filtered out and the racing
/// schedule is silently never explored. This is the cross-lane analog of
/// `sc_repro.rs` (the DPOR shadowing hole).
///
/// A 2-thread race would not catch it — the reorder backtrack can re-anchor on
/// the peer's (unshadowed) record. It needs three: thread A runs a lane-B CAS
/// then a lane-A store (the shadower); thread C runs a competing lane-B CAS.
/// A and C's CASes are the *only* lane-B ops, so the backtrack that schedules
/// C's CAS ahead of A's can anchor only on A's lane-B record. If A's lane-A
/// store shadowed it, that reorder is lost and C's CAS never wins — so the
/// final value 2 becomes unreachable. RMWs (not plain loads) are essential:
/// a CAS succeeds only on the true schedule order, not via stale-value
/// branching. Per-region access records keep A's lane-B record alive, so both
/// winners remain reachable. (Verified to FAIL when the shadow is injected.)
#[test]
fn pruning_does_not_shadow_across_lanes() {
    let seen = Arc::new(Mutex::new(HashSet::new()));
    let seen_ = seen.clone();
    loom::model(move || {
        let x = Arc::new(AtomicU128::new(0));

        let a = {
            let x = x.clone();
            thread::spawn(move || {
                // Lane-B CAS (the record that must not be shadowed) ...
                let _ = x.compare_exchange_masked(LANE_B, 0, 1 << 64, SeqCst, SeqCst);
                // ... then a lane-A store (the would-be shadower).
                x.store_masked(LANE_A, 1, SeqCst);
            })
        };
        let c = {
            let x = x.clone();
            thread::spawn(move || {
                // Competing lane-B CAS; wins only if scheduled before A's.
                let _ = x.compare_exchange_masked(LANE_B, 0, 2 << 64, SeqCst, SeqCst);
            })
        };

        a.join().unwrap();
        c.join().unwrap();
        let b = (x.load(SeqCst) >> 64) as u64;
        seen.lock().unwrap().insert(b);
    });

    let seen = seen_.lock().unwrap();
    assert!(seen.contains(&1), "A's CAS never won: {:?}", *seen);
    assert!(
        seen.contains(&2),
        "C's CAS never scheduled ahead of A's — A's lane-B record was shadowed \
         by its later lane-A store: {:?}",
        *seen
    );
}

/// Mask-intersection DPOR pruning: two threads each writing a lane commute
/// when the lanes are **disjoint** (no backtrack point between them, so the
/// interleavings collapse), while the same writes to a **shared** lane are
/// genuinely dependent and fully explored. The contrast is the pruning at
/// work — it is what keeps the finer sub-location model from exploding the
/// schedule space for genuinely independent lanes.
#[test]
fn dpor_prunes_disjoint_lanes() {
    fn iterations(shared: bool) -> usize {
        let n = Arc::new(AtomicUsize::new(0));
        let n_ = n.clone();
        loom::model(move || {
            n.fetch_add(1, Relaxed);
            let x = Arc::new(AtomicU128::new(0));
            let t = {
                let x = x.clone();
                thread::spawn(move || {
                    x.store_masked(LANE_A, 1, Relaxed);
                    x.store_masked(LANE_A, 2, Relaxed);
                    x.store_masked(LANE_A, 3, Relaxed);
                })
            };
            // The main thread writes the disjoint high lane, or (shared) the
            // same low lane the spawned thread writes.
            let (mask, base) = if shared { (LANE_A, 0) } else { (LANE_B, 64) };
            x.store_masked(mask, 4 << base, Relaxed);
            x.store_masked(mask, 5 << base, Relaxed);
            x.store_masked(mask, 6 << base, Relaxed);
            t.join().unwrap();
        });
        n_.load(Relaxed)
    }

    let disjoint = iterations(false);
    let shared = iterations(true);
    // Disjoint lanes commute to essentially a single schedule; the shared lane
    // fans out across all interleavings.
    assert!(
        disjoint * 5 <= shared,
        "DPOR did not prune disjoint lanes: disjoint={} shared={}",
        disjoint,
        shared
    );
}

/// Basic correctness: a masked store to one lane leaves the other lane's bits
/// exactly as they were.
#[test]
fn masked_store_preserves_other_lane() {
    loom::model(|| {
        let x = AtomicU128::new(0);
        x.store(1u128 | (1u128 << 64), Relaxed); // A=1, B=1
        x.store_masked(LANE_A, 5, Relaxed); // A=5, B untouched
        let v = x.load(Relaxed);
        assert_eq!(v & LANE_A, 5, "lane A wrong: {:#x}", v);
        assert_eq!(v & LANE_B, 1u128 << 64, "lane B clobbered: {:#x}", v);
    });
}
