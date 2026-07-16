//! Typed sub-word lane views (`lane_u32` / `lane_u64`).
//!
//! The lanes are thin sugar over the masked ops (whose model fidelity is
//! pinned by `mixed_size.rs`); these tests pin the *lane algebra* — offsets,
//! masking, sibling preservation, RMW/CAS semantics — plus one concurrency
//! shape per op class, so a regression in the shift/mask plumbing cannot hide
//! behind the masked-op layer.

#![deny(warnings, rust_2018_idioms)]

use loom::sync::atomic::{AtomicU64, AtomicU128};
use loom::thread;

use std::sync::atomic::Ordering::{Relaxed, SeqCst};
use std::sync::Arc;

/// The ntlib futex-word shape: S = bits 0..64, owner = 64..96, value = 96..128.
const VALUE_OFF: usize = 12;
const OWNER_OFF: usize = 8;
const HIGH_OFF: usize = 8;
const S_OFF: usize = 0;

/// Every aligned offset addresses exactly its own bits: a store through each
/// lane lands at the right shift and leaves every sibling untouched.
#[test]
fn lane_offsets_address_disjoint_bits() {
    loom::model(|| {
        let x = AtomicU128::new(0);
        x.lane_u32(0).store(0x1111_1111, Relaxed);
        x.lane_u32(4).store(0x2222_2222, Relaxed);
        x.lane_u32(8).store(0x3333_3333, Relaxed);
        x.lane_u32(12).store(0x4444_4444, Relaxed);
        assert_eq!(
            x.load(Relaxed),
            0x4444_4444_3333_3333_2222_2222_1111_1111u128
        );
        // The 64-bit views compose the same bytes.
        assert_eq!(x.lane_u64(0).load(Relaxed), 0x2222_2222_1111_1111);
        assert_eq!(x.lane_u64(8).load(Relaxed), 0x4444_4444_3333_3333);
    });
}

/// Lane RMWs return the old lane value, wrap at the lane width, and never
/// carry into a sibling — the futex-word "value op vs queue churn" isolation.
#[test]
fn lane_rmw_algebra_and_isolation() {
    loom::model(|| {
        let x = AtomicU128::new(0);
        x.lane_u64(S_OFF).store(u64::MAX, Relaxed); // queue bits all-set
        let v = x.lane_u32(VALUE_OFF);

        assert_eq!(v.fetch_add(5, SeqCst), 0);
        assert_eq!(v.fetch_sub(1, SeqCst), 5);
        assert_eq!(v.fetch_or(0xF0, SeqCst), 4);
        assert_eq!(v.fetch_and(0x1F, SeqCst), 0xF4);
        assert_eq!(v.fetch_xor(0x1, SeqCst), 0x14);
        assert_eq!(v.swap(u32::MAX, SeqCst), 0x15);
        // Wrap stays inside the lane: no carry into the owner lane.
        assert_eq!(v.fetch_add(1, SeqCst), u32::MAX);
        assert_eq!(v.load(Relaxed), 0);
        assert_eq!(v.fetch_max(7, SeqCst), 0);
        assert_eq!(v.fetch_min(3, SeqCst), 7);
        assert_eq!(v.fetch_nand(0xFF, SeqCst), 3);
        assert_eq!(v.load(Relaxed), !3u32);

        assert_eq!(x.lane_u64(S_OFF).load(Relaxed), u64::MAX, "sibling disturbed");
        assert_eq!(x.lane_u32(OWNER_OFF).load(Relaxed), 0, "owner disturbed");
    });
}

/// Lane CAS: compares only the lane, replaces only the lane, reports the
/// observed lane value on failure.
#[test]
fn lane_cas_semantics() {
    loom::model(|| {
        let x = AtomicU128::new(0);
        x.lane_u64(S_OFF).store(0xDEAD, Relaxed);

        let v = x.lane_u32(VALUE_OFF);
        assert_eq!(v.compare_exchange(0, 7, SeqCst, Relaxed), Ok(0));
        assert_eq!(v.compare_exchange(0, 9, SeqCst, Relaxed), Err(7));
        assert_eq!(v.compare_exchange_weak(7, 8, SeqCst, Relaxed), Ok(7));
        assert_eq!(x.lane_u64(S_OFF).load(Relaxed), 0xDEAD, "sibling disturbed");
    });
}

/// Two threads doing RMWs on disjoint lanes: both updates always land — the
/// lanes are genuinely independent atomics over one cell.
#[test]
fn concurrent_disjoint_lane_rmws_both_land() {
    loom::model(|| {
        let x = Arc::new(AtomicU128::new(0));

        let t = {
            let x = x.clone();
            thread::spawn(move || {
                x.lane_u32(VALUE_OFF).fetch_add(1, SeqCst);
            })
        };
        x.lane_u64(S_OFF).fetch_add(1, SeqCst);
        t.join().unwrap();

        assert_eq!(x.lane_u32(VALUE_OFF).load(SeqCst), 1);
        assert_eq!(x.lane_u64(S_OFF).load(SeqCst), 1);
    });
}

/// A lane CAS races a full-width CAS over the same bits: exactly one wins —
/// the lane op and the wide op are atomic against each other (the carve-out's
/// central claim, exercised through the lane API).
#[test]
fn lane_cas_races_wide_cas_one_winner() {
    loom::model(|| {
        let x = Arc::new(AtomicU128::new(0));

        let t = {
            let x = x.clone();
            thread::spawn(move || {
                x.compare_exchange(0, 1u128 << 96, SeqCst, SeqCst).is_ok()
            })
        };
        let lane_won = x
            .lane_u32(VALUE_OFF)
            .compare_exchange(0, 2, SeqCst, SeqCst)
            .is_ok();
        let wide_won = t.join().unwrap();

        // Both compare the same (value) bits against 0: one, and only one,
        // can observe 0 — unless the loser ran first and lost to nothing.
        let v = x.lane_u32(VALUE_OFF).load(SeqCst);
        match (lane_won, wide_won) {
            (true, false) => assert_eq!(v, 2),
            (false, true) => assert_eq!(v, 1),
            (true, true) => panic!("both CASes claimed the same bits"),
            (false, false) => panic!("neither CAS won on an untouched word"),
        }
    });
}

/// The 32-bit lane of an `AtomicU64` (the pool's `{generation, succ}` atom
/// shape): a lane `fetch_add` bumps only the low dword.
#[test]
fn atomic_u64_lane_u32() {
    loom::model(|| {
        let x = AtomicU64::new(0xAAAA_BBBB_0000_0000);
        assert_eq!(x.lane_u32(0).fetch_add(1, SeqCst), 0);
        assert_eq!(x.load(Relaxed), 0xAAAA_BBBB_0000_0001);
        assert_eq!(x.lane_u32(4).load(Relaxed), 0xAAAA_BBBB);
    });
}

/// Misaligned lane offsets are construction errors, not silent misreads.
#[test]
#[should_panic(expected = "misaligned or out-of-bounds lane")]
fn misaligned_lane_panics() {
    loom::model(|| {
        let x = AtomicU128::new(0);
        let _ = x.lane_u64(HIGH_OFF - 2);
    });
}

/// Out-of-bounds lane offsets are construction errors too.
#[test]
#[should_panic(expected = "misaligned or out-of-bounds lane")]
fn out_of_bounds_lane_panics() {
    loom::model(|| {
        let x = AtomicU128::new(0);
        let _ = x.lane_u32(16);
    });
}
