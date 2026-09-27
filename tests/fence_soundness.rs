#![deny(warnings, rust_2018_idioms)]
//! What fences may and may not establish, against C++20 [atomics.fences] and
//! [atomics.order] p4.
//!
//! A `SeqCst` fence orders the single total order S and so restricts which
//! values coherence lets an operation read or overwrite, but it creates no
//! happens-before of its own: a data race across two SC fences is still a
//! race. Release/acquire fences synchronize through the reads and writes on
//! either side of them, whatever became of the store since.

use loom::cell::UnsafeCell;
use loom::sync::atomic::{fence, AtomicUsize};
use loom::thread;

use std::collections::HashSet;
use std::sync::atomic::Ordering::{Acquire, Relaxed, Release, SeqCst};
use std::sync::{Arc, Mutex};

/// Every outcome `f` returns across the whole exploration.
fn outcomes<T>(f: impl Fn() -> T + Send + Sync + 'static) -> HashSet<T>
where
    T: std::hash::Hash + Eq + Clone + Send + 'static,
{
    let seen = Arc::new(Mutex::new(HashSet::new()));
    let sink = seen.clone();
    loom::model(move || {
        let v = f();
        sink.lock().unwrap().insert(v);
    });
    let out = seen.lock().unwrap().clone();
    out
}

/// Two SC fences with a relaxed flag between them are not a release/acquire
/// pair: when the reader sees the flag, its read of the data races with the
/// write (C++20: hb is (sb ∪ sw)+, and no sw edge exists here).
#[test]
#[should_panic(expected = "Causality violation")]
fn sc_fences_create_no_happens_before() {
    loom::model(|| {
        let data = Arc::new(UnsafeCell::new(0usize));
        let flag = Arc::new(AtomicUsize::new(0));

        let writer = {
            let (data, flag) = (data.clone(), flag.clone());
            thread::spawn(move || {
                data.with_mut(|p| unsafe { *p = 1 });
                fence(SeqCst);
                flag.store(1, Relaxed);
            })
        };

        let reader = thread::spawn(move || {
            fence(SeqCst);
            if flag.load(Relaxed) == 1 {
                data.with(|p| unsafe { *p });
            }
        });

        writer.join().unwrap();
        reader.join().unwrap();
    });
}

/// The positive half: a release fence before the flag store and an acquire
/// fence after the flag load do synchronize, SC or not, and SB with SC fences
/// on both sides still forbids the both-miss outcome.
#[test]
fn fences_synchronize_through_the_flag() {
    for (release, acquire) in [(Release, Acquire), (SeqCst, SeqCst), (SeqCst, Acquire)] {
        loom::model(move || {
            let data = Arc::new(UnsafeCell::new(0usize));
            let flag = Arc::new(AtomicUsize::new(0));

            let writer = {
                let (data, flag) = (data.clone(), flag.clone());
                thread::spawn(move || {
                    data.with_mut(|p| unsafe { *p = 1 });
                    fence(release);
                    flag.store(1, Relaxed);
                })
            };

            if flag.load(Relaxed) == 1 {
                fence(acquire);
                assert_eq!(1, data.with(|p| unsafe { *p }));
            }

            writer.join().unwrap();
        });
    }

    let seen = outcomes(|| {
        let x = Arc::new(AtomicUsize::new(0));
        let y = Arc::new(AtomicUsize::new(0));

        let other = {
            let (x, y) = (x.clone(), y.clone());
            thread::spawn(move || {
                y.store(1, Relaxed);
                fence(SeqCst);
                x.load(Relaxed)
            })
        };

        x.store(1, Relaxed);
        fence(SeqCst);
        let a = y.load(Relaxed);
        (a, other.join().unwrap())
    });
    assert!(!seen.contains(&(0, 0)), "SB+SC fences reached (0, 0): {seen:?}");
    assert_eq!(seen.len(), 3, "SB+SC fences lost a legal outcome: {seen:?}");
}

/// An acquire fence synchronizes with the release of every store an earlier
/// read of its thread saw ([atomics.fences] p4) — including one pushed out of
/// its cell's bounded history before the fence runs.
#[test]
fn acquire_fence_synchronizes_with_an_evicted_store() {
    loom::model(|| {
        let data = Arc::new(UnsafeCell::new(0usize));
        let flag = Arc::new(AtomicUsize::new(0));

        let writer = {
            let (data, flag) = (data.clone(), flag.clone());
            thread::spawn(move || {
                data.with_mut(|p| unsafe { *p = 1 });
                flag.store(1, Release);
            })
        };

        if flag.load(Relaxed) == 1 {
            // Enough stores of the reader's own to evict the one it read.
            for v in 2..12 {
                flag.store(v, Relaxed);
            }
            fence(Acquire);
            assert_eq!(1, data.with(|p| unsafe { *p }));
        }

        writer.join().unwrap();
    });
}

/// p4.2/p4.3 against an SC access in another thread. `x = 1` happens before
/// T1's fence F, so an SC operation later than F in S may not be
/// coherence-ordered before it; T1's `z = 1` follows F, so it may not be
/// coherence-ordered before an SC operation earlier than F. Ending with
/// `r = 1`, `rx = 0`, `z == 2` needs `F <S Wz2 <S Rx <S F`.
#[test]
fn sc_fence_orders_against_sc_accesses_of_other_threads() {
    let seen = outcomes(|| {
        let x = Arc::new(AtomicUsize::new(0));
        let flag = Arc::new(AtomicUsize::new(0));
        let z = Arc::new(AtomicUsize::new(0));

        let t0 = {
            let (x, flag) = (x.clone(), flag.clone());
            thread::spawn(move || {
                x.store(1, Relaxed);
                flag.store(1, Release);
            })
        };
        let t1 = {
            let (flag, z) = (flag.clone(), z.clone());
            thread::spawn(move || {
                let r = flag.load(Acquire);
                if r == 1 {
                    fence(SeqCst);
                    z.store(1, Relaxed);
                }
                r
            })
        };
        let t2 = {
            let (x, z) = (x.clone(), z.clone());
            thread::spawn(move || {
                z.store(2, SeqCst);
                x.load(SeqCst)
            })
        };

        t0.join().unwrap();
        let r = t1.join().unwrap();
        let rx = t2.join().unwrap();
        (r, rx, z.load(Relaxed))
    });
    assert!(
        !seen.contains(&(1, 0, 2)),
        "reached the S-cyclic (1, 0, 2): {seen:?}"
    );
    for legal in [(1, 0, 1), (1, 1, 2), (1, 1, 1), (0, 0, 2), (0, 1, 2)] {
        assert!(seen.contains(&legal), "lost {legal:?}: {seen:?}");
    }
}

/// A fence's scope reaches every operation it happens before, not only its
/// own thread's (p4.3): `r0 = 0` and `ry = 1` put `Wx <S Ry0 <S Wy <S F`, and
/// F happens before the main thread's load of `x` through the release/acquire
/// on `flag`, so that load may not read the `x = 0` that `Wx` superseded.
/// Nothing but S orders `Wx` before F — no happens-before path joins them.
#[test]
fn fence_scope_travels_with_synchronization() {
    let seen = outcomes(|| {
        let x = Arc::new(AtomicUsize::new(0));
        let y = Arc::new(AtomicUsize::new(0));
        let flag = Arc::new(AtomicUsize::new(0));

        let t0 = {
            let (x, y) = (x.clone(), y.clone());
            thread::spawn(move || {
                x.store(1, SeqCst);
                y.load(SeqCst)
            })
        };
        let t3 = {
            let y = y.clone();
            thread::spawn(move || y.store(1, SeqCst))
        };
        let t1 = {
            let (y, flag) = (y.clone(), flag.clone());
            thread::spawn(move || {
                let ry = y.load(Acquire);
                fence(SeqCst);
                flag.store(1, Release);
                ry
            })
        };

        let rx = if flag.load(Acquire) == 1 {
            Some(x.load(Relaxed))
        } else {
            None
        };

        let r0 = t0.join().unwrap();
        t3.join().unwrap();
        let ry = t1.join().unwrap();
        (r0, ry, rx)
    });
    assert!(
        !seen.contains(&(0, 1, Some(0))),
        "a load the fence happens before read under an SC store earlier in S: {seen:?}"
    );
    assert!(seen.contains(&(0, 1, Some(1))), "{seen:?}");
    assert!(seen.contains(&(1, 1, Some(0))), "{seen:?}");
}
