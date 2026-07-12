//! RMW-atomicity modification-order tests.
//!
//! C11 requires an atomic RMW's write to sit *immediately* after the store
//! it read in the cell's modification order — no other store may split the
//! pair. `rt::atomic` enforces this with the `close_rmw_atomicity` closure
//! and the single-lane `mo_before` marker; these tests pin the fix from both
//! sides: the forbidden outcomes stay unreachable (soundness of the model's
//! *checking*), and every legal weak outcome is still explored (no
//! over-constraint — loom's cardinal rule is no false negatives).

#![deny(warnings, rust_2018_idioms)]

use loom::sync::atomic::AtomicU64;
use loom::sync::Arc;
use loom::thread;

use std::collections::HashSet;
use std::sync::atomic::Ordering::{Acquire, Relaxed, Release};
use std::sync::Mutex;

/// A plain store racing a committed CAS must land mo-after the CAS's write,
/// never mo-incomparable to it.
///
/// Main stores 2, a spawned thread CASes 2 -> 0x5_0000_0002, main stores 3.
/// If the CAS read 2, its write is immediately mo-after 2, and main's 3 —
/// mo-after main's own 2, unable to split the pair — is forced mo-after the
/// CAS write: final order `0, 2, 0x5_0000_0002, 3`. If the CAS read 3 it
/// fails and writes nothing. Either way the post-join Acquire load returns
/// 3. Before the fix, 3 landed mo-incomparable to the CAS write and the
/// final load could return the "zombie" 0x5_0000_0002.
#[test]
fn store_racing_cas_orders_after_the_rmw_write() {
    let finals = std::sync::Arc::new(Mutex::new(HashSet::new()));
    let finals_ = finals.clone();

    loom::model(move || {
        let m = Arc::new(AtomicU64::new(0));
        m.store(2, Release);

        let mc = m.clone();
        let th = thread::spawn(move || {
            let _ = mc.compare_exchange(2, (5u64 << 32) | 2, Release, Relaxed);
        });

        let g = (m.load(Relaxed) as u32).wrapping_add(1);
        m.store(g as u64, Release);

        th.join().unwrap();

        let fin = m.load(Acquire);
        finals_.lock().unwrap().insert(fin);
        assert_eq!(fin, 3, "non-C11 final state {fin:#x}");
    });

    assert_eq!(*finals.lock().unwrap(), HashSet::from([3]));
}

/// The atomicity edge must not over-constrain: a store that is genuinely
/// concurrent with an RMW pair may still legally land mo-*after* the RMW's
/// write, and the model must keep exploring those extensions.
///
/// Two concurrent plain stores (1 and 2) race a concurrent `fetch_add(10)`.
/// Exactly these `(rmw_prev, final)` outcomes are C11-legal, enumerating the
/// total modification orders (the RMW pair `x -> x+10` stays adjacent, the
/// unread plain store lands before the read store or after the write):
///
///   read 0: `0,+10,1,2` / `0,+10,2,1`         -> (0,1), (0,2)
///   read 1: `0,1,11,2`  / `0,2,1,11`          -> (1,2), (1,11)
///   read 2: `0,2,12,1`  / `0,1,2,12`          -> (2,1), (2,12)
///
/// Missing pairs here mean the fix pruned legal executions (false
/// negatives); extra pairs mean unsound exploration.
#[test]
fn rmw_atomicity_does_not_overconstrain_concurrent_stores() {
    let seen = std::sync::Arc::new(Mutex::new(HashSet::new()));
    let seen_ = seen.clone();

    loom::model(move || {
        let m = Arc::new(AtomicU64::new(0));

        let m1 = m.clone();
        let t1 = thread::spawn(move || m1.store(1, Release));

        let m2 = m.clone();
        let t2 = thread::spawn(move || m2.store(2, Release));

        let m3 = m.clone();
        let t3 = thread::spawn(move || m3.fetch_add(10, Release));

        t1.join().unwrap();
        t2.join().unwrap();
        let prev = t3.join().unwrap();

        let fin = m.load(Acquire);
        seen_.lock().unwrap().insert((prev, fin));
    });

    let allowed: HashSet<(u64, u64)> =
        HashSet::from([(0, 1), (0, 2), (1, 2), (1, 11), (2, 1), (2, 12)]);
    assert_eq!(*seen.lock().unwrap(), allowed);
}

/// Two concurrent RMWs can never read the same store: the first write is
/// immediately mo-after the shared candidate, making it non-maximal for the
/// second. The classic atomic-counter guarantee.
#[test]
fn concurrent_fetch_adds_read_distinct_stores() {
    loom::model(|| {
        let m = Arc::new(AtomicU64::new(0));

        let m2 = m.clone();
        let th = thread::spawn(move || m2.fetch_add(1, Relaxed));

        let v1 = m.fetch_add(1, Relaxed);
        let v2 = th.join().unwrap();

        assert_ne!(v1, v2);
        assert_eq!(m.load(Acquire), 2);
    });
}

/// The closure must chase RMW *chains*: with `store 1; {fetch_add(10);
/// fetch_add(100)} racing store 2`, each RMW pair is individually
/// unsplittable — the racing store may land before the first pair's read,
/// between the two pairs, or after the second, but never inside a pair, and
/// a zombie intermediate (final 111 with 2 mo-incomparable) must be
/// unreachable.
///
///   both read the chain:  `0,1,11,111,2`  -> final 2
///   2 between the pairs:  `0,1,11,2,102`  -> final 102
///   2 before both pairs:  `0,1,2,12,112`  -> final 112
#[test]
fn store_cannot_split_an_rmw_chain() {
    let finals = std::sync::Arc::new(Mutex::new(HashSet::new()));
    let finals_ = finals.clone();

    loom::model(move || {
        let m = Arc::new(AtomicU64::new(0));
        m.store(1, Release);

        let mc = m.clone();
        let th = thread::spawn(move || {
            mc.fetch_add(10, Release);
            mc.fetch_add(100, Release);
        });

        m.store(2, Release);

        th.join().unwrap();

        let fin = m.load(Acquire);
        finals_.lock().unwrap().insert(fin);
        assert!(
            fin == 2 || fin == 102 || fin == 112,
            "store split an RMW pair: final {fin}"
        );
    });

    assert_eq!(*finals.lock().unwrap(), HashSet::from([2, 102, 112]));
}
