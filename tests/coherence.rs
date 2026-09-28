//! Modification-order coherence of the atomic model.
//!
//! Each test explores a small program to exhaustion and checks the set of
//! outcomes against C++20 [intro.races]/[atomics.order]: an outcome no
//! coherent modification order admits must never appear, and a legal one the
//! model used to drop must appear.

#![deny(warnings, rust_2018_idioms)]

use loom::cell::UnsafeCell;
use loom::sync::atomic::{fence, AtomicU128, AtomicU64, AtomicUsize};
use loom::thread;

use std::collections::HashSet;
use std::hash::Hash;
use std::sync::atomic::Ordering::{Acquire, Relaxed, Release, SeqCst};
use std::sync::{Arc, Mutex};

/// Every outcome `f` produces across the whole exploration.
fn outcomes<T>(f: impl Fn() -> T + Send + Sync + 'static) -> HashSet<T>
where
    T: Hash + Eq + Clone + Send + 'static,
{
    let set = Arc::new(Mutex::new(HashSet::new()));
    let sink = set.clone();
    loom::model(move || {
        let v = f();
        sink.lock().unwrap().insert(v);
    });
    let out = set.lock().unwrap().clone();
    out
}

/// Modification order is transitive. `W` stores 1 then 2; `T` stores 3. `R`
/// reading 3 then 1 puts 3 before 1, hence before 2; main reading 2 then 3
/// would put 2 before 3 — a cycle.
#[test]
fn mo_is_transitive() {
    let out = outcomes(|| {
        let x = Arc::new(AtomicUsize::new(0));
        let w = {
            let x = x.clone();
            thread::spawn(move || {
                x.store(1, Relaxed);
                x.store(2, Relaxed);
            })
        };
        let t = {
            let x = x.clone();
            thread::spawn(move || x.store(3, Relaxed))
        };
        let r = {
            let x = x.clone();
            thread::spawn(move || (x.load(Relaxed), x.load(Relaxed)))
        };
        let main = (x.load(Relaxed), x.load(Relaxed));
        w.join().unwrap();
        t.join().unwrap();
        (r.join().unwrap(), main)
    });

    assert!(
        !out.contains(&((3, 1), (2, 3))),
        "coherence cycle 3 < 1 < 2 < 3"
    );
}

/// An SC RMW read obeys the SC read rule and records it. `T1`'s SC store of
/// 1 precedes the SC fetch_add in S whenever `T1`'s SC load of y reads 0
/// (store buffering through y), so the fetch_add reading `T2`'s relaxed 2
/// puts 1 before 2, and main cannot then read 2 followed by 1.
#[test]
fn sc_rmw_read_follows_its_sc_predecessors() {
    let out = outcomes(|| {
        let x = Arc::new(AtomicUsize::new(0));
        let y = Arc::new(AtomicUsize::new(0));
        let t1 = {
            let (x, y) = (x.clone(), y.clone());
            thread::spawn(move || {
                x.store(1, SeqCst);
                y.load(SeqCst)
            })
        };
        let t2 = {
            let x = x.clone();
            thread::spawn(move || x.store(2, Relaxed))
        };
        let t3 = {
            let (x, y) = (x.clone(), y.clone());
            thread::spawn(move || {
                y.store(1, SeqCst);
                x.fetch_add(10, SeqCst)
            })
        };
        let main = (x.load(Relaxed), x.load(Relaxed));
        let r1 = t1.join().unwrap();
        t2.join().unwrap();
        let r3 = t3.join().unwrap();
        r1 == 0 && r3 == 2 && main == (2, 1)
    });

    assert!(
        !out.contains(&true),
        "SC RMW read 2 ahead of the SC store of 1"
    );
}

/// Nothing sits between an RMW's read store and its write, in either
/// direction. The fetch_add reads 1 and writes 11; `Q` reading 2 then 11
/// puts 2 before 11, so before 1. Main — reading only after `Q`, through a
/// relaxed flag that orders the schedule but not the memory — cannot then
/// read 1 followed by 2.
#[test]
fn rmw_atomicity_orders_predecessors_of_the_write() {
    let out = outcomes(|| {
        let x = Arc::new(AtomicUsize::new(0));
        let flag = Arc::new(AtomicUsize::new(0));
        let t1 = {
            let x = x.clone();
            thread::spawn(move || x.store(1, Relaxed))
        };
        let t2 = {
            let x = x.clone();
            thread::spawn(move || x.store(2, Relaxed))
        };
        let t3 = {
            let x = x.clone();
            thread::spawn(move || x.fetch_add(10, Relaxed))
        };
        let q = {
            let (x, flag) = (x.clone(), flag.clone());
            thread::spawn(move || {
                let q = (x.load(Relaxed), x.load(Relaxed));
                flag.store(1, Relaxed);
                q
            })
        };
        let main = (flag.load(Relaxed) == 1).then(|| (x.load(Relaxed), x.load(Relaxed)));
        t1.join().unwrap();
        t2.join().unwrap();
        let r = t3.join().unwrap();
        let q = q.join().unwrap();
        r == 1 && q == (2, 11) && main == Some((1, 2))
    });

    assert!(!out.contains(&true), "store 2 split the RMW pair 1 -> 11");
}

/// A failed compare-exchange is a load at its failure ordering, and may read
/// a stale value — here the 0 that `T1`'s relaxed flag does not order away.
#[test]
fn failed_cas_may_read_a_stale_value() {
    let out = outcomes(|| {
        let x = Arc::new(AtomicUsize::new(0));
        let f = Arc::new(AtomicUsize::new(0));
        let t1 = {
            let (x, f) = (x.clone(), f.clone());
            thread::spawn(move || {
                x.store(1, Relaxed);
                f.store(1, Relaxed);
            })
        };
        let r = (f.load(Relaxed) == 1).then(|| x.compare_exchange(1, 2, Relaxed, Relaxed));
        t1.join().unwrap();
        r
    });

    assert!(
        out.contains(&Some(Err(0))),
        "failed CAS never read the stale 0"
    );
    assert!(out.contains(&Some(Ok(1))));
}

/// A failing CAS over a split cell reads a single-copy-atomic snapshot: it
/// may read from before the wide store — which the relaxed flag does not
/// order away — but never half of it.
#[test]
fn failed_wide_cas_reads_a_stale_whole_snapshot() {
    const LO: u128 = u64::MAX as u128;
    const WIDE: u128 = (1 << 64) | 1;

    let out = outcomes(|| {
        let x = Arc::new(AtomicU128::new(0));
        let f = Arc::new(AtomicUsize::new(0));
        // Split the cell into two lanes.
        x.store_masked(LO, 0, Relaxed);
        let t = {
            let (x, f) = (x.clone(), f.clone());
            thread::spawn(move || {
                x.store(WIDE, Relaxed);
                f.store(1, Relaxed);
            })
        };
        let r = (f.load(Relaxed) == 1).then(|| x.compare_exchange(7, 8, Relaxed, Relaxed));
        t.join().unwrap();
        r
    });

    assert!(
        out.contains(&Some(Err(0))),
        "the failing CAS never read the stale snapshot"
    );
    assert!(out.contains(&Some(Err(WIDE))));
    assert!(
        out.iter()
            .all(|r| matches!(r, None | Some(Err(0)) | Some(Err(WIDE)))),
        "torn CAS read: {out:?}"
    );
}

/// A failing preserving CAS takes the same stale reads as a plain one.
#[test]
fn failed_preserving_cas_may_read_a_stale_value() {
    const HI: u128 = !0u128 << 64;

    let out = outcomes(|| {
        let x = Arc::new(AtomicU128::new(0));
        let f = Arc::new(AtomicUsize::new(0));
        let t1 = {
            let (x, f) = (x.clone(), f.clone());
            thread::spawn(move || {
                x.store(1, Relaxed);
                f.store(1, Relaxed);
            })
        };
        let r = (f.load(Relaxed) == 1)
            .then(|| x.compare_exchange_preserving(HI, 1, 2, Relaxed, Relaxed));
        t1.join().unwrap();
        r
    });

    assert!(
        out.contains(&Some(Err(0))),
        "failed preserving CAS never read the stale 0"
    );
}

/// A full-width load after a preserving CAS sees the CAS all-or-none. The
/// CAS compared the high lane against `T1`'s 1 and wrote 5 into the low
/// lane; a snapshot with that 5 cannot show the high lane older than 1.
#[test]
fn wide_load_sees_a_preserving_cas_whole() {
    const HI: u128 = !0u128 << 64;

    let out = outcomes(|| {
        let x = Arc::new(AtomicU128::new(0));
        let t1 = {
            let x = x.clone();
            thread::spawn(move || x.store_masked(HI, 1 << 64, Relaxed))
        };
        let t2 = {
            let x = x.clone();
            thread::spawn(move || {
                x.compare_exchange_preserving(HI, 1 << 64, (1 << 64) | 5, Relaxed, Relaxed)
                    .is_ok()
            })
        };
        let t3 = {
            let x = x.clone();
            thread::spawn(move || x.load(Relaxed))
        };
        t1.join().unwrap();
        let ok = t2.join().unwrap();
        let v = t3.join().unwrap();
        (ok, v as u64, (v >> 64) as u64)
    });

    assert!(
        !out.contains(&(true, 5, 0)),
        "torn snapshot of the preserving CAS"
    );
    assert!(out.contains(&(true, 5, 1)));
}

/// A weak compare-exchange fails spuriously only where the target's does: on
/// x86-64, whose `lock cmpxchg` never does, it behaves as the strong one.
#[test]
#[cfg(target_arch = "x86_64")]
fn weak_cas_follows_the_target() {
    if std::env::var_os("LOOM_SPURIOUS_WEAK_CAS").is_some() {
        return;
    }
    let out = outcomes(|| {
        let x = AtomicUsize::new(0);
        x.compare_exchange_weak(0, 1, Relaxed, Relaxed)
    });

    assert_eq!(out, HashSet::from([Ok(0)]));
}

/// Every store a thread has not yet passed stays readable, however many
/// there are: main may read any of `T1`'s nine values.
#[test]
fn history_keeps_every_readable_store() {
    let out = outcomes(|| {
        let x = Arc::new(AtomicUsize::new(0));
        let f = Arc::new(AtomicUsize::new(0));
        let t1 = {
            let (x, f) = (x.clone(), f.clone());
            thread::spawn(move || {
                for i in 1..=8 {
                    x.store(i, Relaxed);
                }
                f.store(1, Relaxed);
            })
        };
        let r = (f.load(Relaxed) == 1).then(|| x.load(Relaxed));
        t1.join().unwrap();
        r
    });

    for v in 0..=8 {
        assert!(out.contains(&Some(v)), "store of {v} was unreadable");
    }
}

/// A store keeps constraining reads while some thread has not passed it.
/// `T2` stores 100 and then reads main's 1, putting 100 before 1; `T3`,
/// having read 1, may not read 100 however many stores main makes after 1.
#[test]
fn a_seen_store_keeps_its_floor() {
    let out = Arc::new(Mutex::new(HashSet::new()));
    let sink = out.clone();

    let mut builder = loom::model::Builder::new();
    builder.preemption_bound = Some(2);
    builder.check(move || {
        let x = Arc::new(AtomicUsize::new(0));
        let t2 = {
            let x = x.clone();
            thread::spawn(move || {
                x.store(100, Relaxed);
                x.load(Relaxed)
            })
        };
        let t3 = {
            let x = x.clone();
            thread::spawn(move || (x.load(Relaxed), x.load(Relaxed)))
        };
        for i in 1..=7 {
            x.store(i, Relaxed);
        }
        let t2 = t2.join().unwrap();
        let t3 = t3.join().unwrap();
        sink.lock().unwrap().insert((t2, t3));
    });

    let out = out.lock().unwrap();
    assert!(!out.contains(&(1, (1, 100))), "T3 read 100 after 1");
    assert!(out.contains(&(1, (1, 7))));
}

/// A store an acquire fence still owes its release to stays in the history.
/// Main reads `T1`'s release store relaxed, stores eight more values past
/// it, then fences: the fence must still synchronize with `T1`.
#[test]
fn acquire_fence_draws_on_a_passed_store() {
    loom::model(|| {
        let data = Arc::new(UnsafeCell::new(0usize));
        let x = Arc::new(AtomicUsize::new(0));
        let t1 = {
            let (data, x) = (data.clone(), x.clone());
            thread::spawn(move || {
                data.with_mut(|p| unsafe { *p = 1 });
                x.store(1, Release);
            })
        };
        if x.load(Relaxed) == 1 {
            for i in 2..=9 {
                x.store(i, Relaxed);
            }
            fence(Acquire);
            assert_eq!(data.with(|p| unsafe { *p }), 1);
        }
        t1.join().unwrap();
    });
}

/// More live stores than the model tracks is a loud failure, never a silent
/// drop.
#[test]
#[should_panic(expected = "the most the model tracks")]
fn history_overflow_is_loud() {
    let mut builder = loom::model::Builder::new();
    builder.preemption_bound = Some(0);
    builder.check(|| {
        let x = Arc::new(AtomicUsize::new(0));
        let t1 = {
            let x = x.clone();
            thread::spawn(move || {
                for i in 1..=40 {
                    x.store(i, Relaxed);
                }
            })
        };
        x.load(Relaxed);
        t1.join().unwrap();
    });
}

/// `with_mut` writes a store mo-after every other: after it, no older value
/// is readable, whichever of two unordered stores it found last.
#[test]
fn with_mut_supersedes_every_store() {
    let out = outcomes(|| {
        let mut x = Arc::new(AtomicUsize::new(0));
        let t1 = {
            let x = x.clone();
            thread::spawn(move || x.store(1, Relaxed))
        };
        let t2 = {
            let x = x.clone();
            thread::spawn(move || x.store(2, Relaxed))
        };
        t1.join().unwrap();
        t2.join().unwrap();
        let cell = Arc::get_mut(&mut x).unwrap();
        let found = cell.with_mut(|v| std::mem::replace(v, 5));
        (found, x.load(Relaxed))
    });

    assert!(
        out.iter().all(|&(_, after)| after == 5),
        "stale value after with_mut: {out:?}"
    );
    assert!(out.contains(&(1, 5)) && out.contains(&(2, 5)));
}

/// `unsync_load` reads the mo-last store and fixes it as such: a later load
/// returns the same value.
#[test]
fn unsync_load_reads_the_mo_last_store() {
    let out = outcomes(|| {
        let x = Arc::new(AtomicUsize::new(0));
        let t1 = {
            let x = x.clone();
            thread::spawn(move || x.store(1, Relaxed))
        };
        let t2 = {
            let x = x.clone();
            thread::spawn(move || x.store(2, Relaxed))
        };
        t1.join().unwrap();
        t2.join().unwrap();
        let seen = unsafe { x.unsync_load() };
        (seen, x.load(Relaxed))
    });

    assert!(
        out.iter().all(|&(seen, later)| seen == later),
        "unsync_load disagreed: {out:?}"
    );
    assert!(out.contains(&(1, 1)) && out.contains(&(2, 2)));
}

/// A successful RMW may read a store that is not modification-order-last and
/// go in immediately after it, ahead of the stores that follow. `T2`'s
/// fetch_add, run after it saw `T1`'s relaxed flag, may still read the 0
/// that `T1`'s store of 1 then overwrites: order 0, 10, 1.
#[test]
fn rmw_may_read_a_store_that_is_not_last() {
    let out = outcomes(|| {
        let x = Arc::new(AtomicUsize::new(0));
        let flag = Arc::new(AtomicUsize::new(0));
        let t1 = {
            let (x, flag) = (x.clone(), flag.clone());
            thread::spawn(move || {
                x.store(1, Relaxed);
                flag.store(1, Relaxed);
            })
        };
        let r = (flag.load(Relaxed) == 1).then(|| x.fetch_add(10, Relaxed));
        t1.join().unwrap();
        (r, x.load(Relaxed))
    });

    assert!(
        out.contains(&(Some(0), 1)),
        "the RMW never went in ahead of the store of 1: {out:?}"
    );
    assert!(out.contains(&(Some(1), 11)));
    assert!(!out.contains(&(Some(0), 10)), "the store of 1 was lost: {out:?}");
}

/// A masked store unordered with a wide store may land after it: once both
/// happen before a full load, the load may see the lane store over the wide
/// one.
#[test]
fn lane_store_may_land_after_a_wide_store() {
    const LO: u64 = u32::MAX as u64;

    let out = outcomes(|| {
        let x = Arc::new(AtomicU64::new(0));
        let t1 = {
            let x = x.clone();
            thread::spawn(move || x.store(0x1_0000_0001, Relaxed))
        };
        let t2 = {
            let x = x.clone();
            thread::spawn(move || x.store_masked(LO, 2, Relaxed))
        };
        t1.join().unwrap();
        t2.join().unwrap();
        x.load(Relaxed)
    });

    assert!(out.contains(&0x1_0000_0001));
    assert!(
        out.contains(&0x1_0000_0002),
        "the lane store never landed after the wide store: {out:?}"
    );
}

/// A wide load may forward its own lane store and read the other lane from
/// before a peer's wide store the forwarded store follows (AArch64,
/// herd7 `RR-wide-fwd`): final `0x2_0000_0001` with the load reading high 2,
/// low 0.
#[test]
fn wide_load_may_forward_its_own_lane_store() {
    const HI: u64 = !(u32::MAX as u64);

    let out = outcomes(|| {
        let x = Arc::new(AtomicU64::new(0));
        let t = {
            let x = x.clone();
            thread::spawn(move || x.store(0x1_0000_0001, Relaxed))
        };
        x.store_masked(HI, 2 << 32, Relaxed);
        let v = x.load(Relaxed);
        t.join().unwrap();
        (x.load(Relaxed), v)
    });

    assert!(
        out.contains(&(0x2_0000_0001, 0x2_0000_0000)),
        "the forwarding read was never explored: {out:?}"
    );
}

/// A preserving op's record keeps binding the carried lane however many
/// preserving ops follow it: after nine CASes carrying the high lane, a full
/// load showing the first CAS's low lane still cannot show the high lane
/// older than the store the CAS compared.
#[test]
fn preserving_records_outlive_later_preserving_ops() {
    const HI: u128 = !0u128 << 64;

    let out = Arc::new(Mutex::new(HashSet::new()));
    let sink = out.clone();

    let mut builder = loom::model::Builder::new();
    builder.preemption_bound = Some(2);
    builder.check(move || {
        let x = Arc::new(AtomicU128::new(0));
        let t1 = {
            let x = x.clone();
            thread::spawn(move || x.store_masked(HI, 1 << 64, Relaxed))
        };
        let t2 = {
            let x = x.clone();
            thread::spawn(move || {
                let mut ok = 0;
                for i in 0..9u128 {
                    let cur = (1 << 64) | i;
                    if x
                        .compare_exchange_preserving(HI, cur, cur + 1, Relaxed, Relaxed)
                        .is_ok()
                    {
                        ok += 1;
                    }
                }
                ok
            })
        };
        let t3 = {
            let x = x.clone();
            thread::spawn(move || x.load(Relaxed))
        };
        t1.join().unwrap();
        let ok = t2.join().unwrap();
        let v = t3.join().unwrap();
        sink.lock().unwrap().insert((ok, v as u64, (v >> 64) as u64));
    });

    let out = out.lock().unwrap();
    assert!(
        !out.iter().any(|&(_, lo, hi)| lo > 0 && hi == 0),
        "torn snapshot of a preserving CAS: {out:?}"
    );
    assert!(out.iter().any(|&(ok, lo, hi)| ok == 9 && lo == 1 && hi == 1));
}

/// An SC RMW goes into S after every SC operation that already ran, so it may
/// not slot in ahead of a store an earlier SC load read: that load would be
/// coherence-ordered after it yet before it in S. `T2`'s SC load reads 1 and
/// precedes `T3`'s SC fetch_add in S (store buffering through y reading 0),
/// so the fetch_add may not read the 0 that 1 overwrites.
#[test]
fn sc_rmw_may_not_slot_in_before_an_sc_read() {
    let out = outcomes(|| {
        let x = Arc::new(AtomicUsize::new(0));
        let y = Arc::new(AtomicUsize::new(0));
        let t1 = {
            let x = x.clone();
            thread::spawn(move || x.store(1, Relaxed))
        };
        let t2 = {
            let (x, y) = (x.clone(), y.clone());
            thread::spawn(move || (x.load(SeqCst), y.load(SeqCst)))
        };
        let t3 = {
            let (x, y) = (x.clone(), y.clone());
            thread::spawn(move || {
                y.store(1, SeqCst);
                x.fetch_add(10, SeqCst)
            })
        };
        t1.join().unwrap();
        let (r1, r2) = t2.join().unwrap();
        let r3 = t3.join().unwrap();
        (r1, r2, r3, x.load(Relaxed))
    });

    assert!(
        !out.contains(&(1, 0, 0, 1)),
        "the SC RMW went in ahead of a store an earlier SC load read: {out:?}"
    );
}

/// An SC store goes after every store an earlier SC load read. `T2`'s SC load
/// of 1 precedes `T3`'s SC store of 2 in S (store buffering through y), so 2
/// cannot be modification-order-before 1: `T4` may not read 2 then 1.
#[test]
fn sc_store_follows_what_earlier_sc_loads_read() {
    let out = outcomes(|| {
        let x = Arc::new(AtomicUsize::new(0));
        let y = Arc::new(AtomicUsize::new(0));
        let t1 = {
            let x = x.clone();
            thread::spawn(move || x.store(1, Relaxed))
        };
        let t2 = {
            let (x, y) = (x.clone(), y.clone());
            thread::spawn(move || (x.load(SeqCst), y.load(SeqCst)))
        };
        let t3 = {
            let (x, y) = (x.clone(), y.clone());
            thread::spawn(move || {
                y.store(1, SeqCst);
                x.store(2, SeqCst);
            })
        };
        let t4 = {
            let x = x.clone();
            thread::spawn(move || (x.load(Relaxed), x.load(Relaxed)))
        };
        t1.join().unwrap();
        let (r1, r2) = t2.join().unwrap();
        t3.join().unwrap();
        (r1, r2, t4.join().unwrap())
    });

    assert!(
        !out.contains(&(1, 0, (2, 1))),
        "an SC store landed before a store an earlier SC load read: {out:?}"
    );
}

/// Loads that fix modification order commute in the search, and must lose
/// nothing by it. `W1` and `W2` race; each reader reads twice, and its second
/// read fixes the order of the two stores. Every pair of reader histories that
/// one of the two orders admits is reachable; the pair needing both is not.
#[test]
fn order_fixing_loads_lose_no_history() {
    let out = outcomes(|| {
        let x = Arc::new(AtomicUsize::new(0));
        let ws: Vec<_> = [1, 2]
            .into_iter()
            .map(|v| {
                let x = x.clone();
                thread::spawn(move || x.store(v, Relaxed))
            })
            .collect();
        let r = {
            let x = x.clone();
            thread::spawn(move || (x.load(Relaxed), x.load(Relaxed)))
        };
        let own = (x.load(Relaxed), x.load(Relaxed));
        for w in ws {
            w.join().unwrap();
        }
        (r.join().unwrap(), own)
    });

    // A history is coherent with `mo` when it never reads backwards in it.
    let coherent = |mo: [usize; 3], (a, b): (usize, usize)| {
        let at = |v| mo.iter().position(|&m| m == v).unwrap();
        at(a) <= at(b)
    };
    let mut expected = HashSet::new();
    for mo in [[0, 1, 2], [0, 2, 1]] {
        for h1 in [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 1), (2, 2)] {
            for h2 in [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 1), (2, 2)] {
                if coherent(mo, h1) && coherent(mo, h2) {
                    expected.insert((h1, h2));
                }
            }
        }
    }
    assert_eq!(out, expected);
}
