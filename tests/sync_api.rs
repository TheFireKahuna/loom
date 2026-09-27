#![deny(warnings)]

//! The `std::cell` / `std::sync` surface loom mirrors as compositions of
//! modelled primitives. Where an item synchronizes, a tracked access witnesses
//! it: without the edge, loom reports the access as a race.

use std::collections::BTreeSet;
use std::sync::{Arc as StdArc, Mutex as StdMutex};
use std::time::Duration;

use loom::cell::{Cell, UnsafeCell};
use loom::sync::atomic::{AtomicUsize, Ordering::*};
use loom::sync::{Arc, Barrier, Condvar, Mutex};
use loom::thread;

/// A tracked cell shared across threads, so that loom — not the type system —
/// judges whether its accesses are ordered.
struct Racy<T>(UnsafeCell<T>);

// SAFETY: every access goes through loom's tracking, which reports the
// unordered ones this impl lets the tests attempt.
unsafe impl<T: Send> Sync for Racy<T> {}

impl<T> std::ops::Deref for Racy<T> {
    type Target = UnsafeCell<T>;
    fn deref(&self) -> &UnsafeCell<T> {
        &self.0
    }
}

fn racy<T>(v: T) -> Racy<T> {
    Racy(UnsafeCell::new(v))
}

type Outcomes<T> = StdArc<StdMutex<BTreeSet<T>>>;

fn outcomes<T>() -> Outcomes<T> {
    StdArc::new(StdMutex::new(BTreeSet::new()))
}

#[test]
fn unsafe_cell_replace_is_a_tracked_write() {
    loom::model(|| {
        let c = Arc::new(racy(1));
        let t = {
            let c = c.clone();
            thread::spawn(move || unsafe { c.replace(2) })
        };
        assert_eq!(t.join().unwrap(), 1);
        assert_eq!(unsafe { c.replace(3) }, 2);
        assert_eq!(c.with(|p| unsafe { *p }), 3);
    });
}

#[test]
#[should_panic]
fn unsafe_cell_replace_races_with_an_unsynchronized_read() {
    loom::model(|| {
        let c = Arc::new(racy(1));
        let t = {
            let c = c.clone();
            thread::spawn(move || unsafe { c.replace(2) })
        };
        c.with(|p| unsafe { *p });
        t.join().unwrap();
    });
}

#[test]
fn cell_update() {
    loom::model(|| {
        let c = Cell::new(5);
        c.update(|x| x * 2);
        assert_eq!(c.get(), 10);
    });
}

#[test]
fn arc_into_inner_returns_to_exactly_one() {
    fn sum(slots: [Racy<i32>; 2]) -> i32 {
        slots.iter().map(|s| s.with(|p| unsafe { *p })).sum()
    }
    loom::model(|| {
        let a = Arc::new([racy(0), racy(0)]);
        let b = a.clone();
        let t = thread::spawn(move || {
            b[1].with_mut(|p| unsafe { *p = 1 });
            Arc::into_inner(b).map(sum)
        });
        a[0].with_mut(|p| unsafe { *p = 1 });
        let mine = Arc::into_inner(a).map(sum);
        let theirs = t.join().unwrap();

        // Exactly one gets the value, and it reads the other's write: the
        // loser's release is acquired by the winner's decrement, or the read
        // would be reported.
        match (mine, theirs) {
            (Some(v), None) | (None, Some(v)) => assert_eq!(v, 2),
            other => panic!("into_inner returned {other:?}"),
        }
    });
}

#[test]
fn arc_unwrap_or_clone_and_is_unique() {
    loom::model(|| {
        let mut a = Arc::new(5);
        assert!(Arc::is_unique(&a));
        let b = a.clone();
        assert!(!Arc::is_unique(&a));
        assert_eq!(Arc::unwrap_or_clone(b), 5);
        assert!(Arc::is_unique(&a));
        *Arc::get_mut(&mut a).unwrap() = 6;
        assert_eq!(Arc::unwrap_or_clone(a), 6);
    });
}

#[test]
fn condvar_wait_while_sees_the_flag() {
    loom::model(|| {
        let pair = Arc::new((Mutex::new(false), Condvar::new()));
        let t = {
            let pair = pair.clone();
            thread::spawn(move || {
                *pair.0.lock().unwrap() = true;
                pair.1.notify_one();
            })
        };
        let guard = pair.1.wait_while(pair.0.lock().unwrap(), |ready| !*ready).unwrap();
        assert!(*guard);
        drop(guard);
        t.join().unwrap();
    });
}

#[test]
fn condvar_wait_timeout_while_reports_both_outcomes() {
    let seen = outcomes();
    let s = seen.clone();
    loom::model(move || {
        let pair = Arc::new((Mutex::new(false), Condvar::new()));
        let t = {
            let pair = pair.clone();
            thread::spawn(move || {
                *pair.0.lock().unwrap() = true;
                pair.1.notify_one();
            })
        };
        let (guard, result) = pair
            .1
            .wait_timeout_while(pair.0.lock().unwrap(), Duration::from_millis(1), |ready| {
                !*ready
            })
            .unwrap();
        // A timeout is reported exactly when the condition still holds.
        assert_eq!(result.timed_out(), !*guard);
        s.lock().unwrap().insert(result.timed_out());
        drop(guard);
        t.join().unwrap();
    });
    assert_eq!(*seen.lock().unwrap(), BTreeSet::from([false, true]));
}

#[test]
fn barrier_releases_all_and_elects_one_leader() {
    loom::model(|| {
        const N: usize = 3;
        let barrier = Arc::new(Barrier::new(N));
        let slots = Arc::new([(); N].map(|()| racy(0usize)));
        let leaders = Arc::new(AtomicUsize::new(0));

        let run = |i: usize, barrier: Arc<Barrier>, slots: Arc<[Racy<usize>; N]>, leaders: Arc<AtomicUsize>| {
            slots[i].with_mut(|p| unsafe { *p = i + 1 });
            if barrier.wait().is_leader() {
                leaders.fetch_add(1, Relaxed);
            }
            // Every write before the barrier happens-before every return from
            // it; a missing edge would be reported here.
            (0..N).map(|j| slots[j].with(|p| unsafe { *p })).sum::<usize>()
        };

        let handles: Vec<_> = (1..N)
            .map(|i| {
                let (b, s, l) = (barrier.clone(), slots.clone(), leaders.clone());
                thread::spawn(move || run(i, b, s, l))
            })
            .collect();
        let mine = run(0, barrier.clone(), slots.clone(), leaders.clone());
        for h in handles {
            assert_eq!(h.join().unwrap(), 6);
        }
        assert_eq!(mine, 6);
        assert_eq!(leaders.load(Relaxed), 1);
    });
}

#[test]
fn barrier_is_reusable_and_trivial_barriers_lead() {
    loom::model(|| {
        assert!(Barrier::new(0).wait().is_leader());
        assert!(Barrier::new(1).wait().is_leader());

        let barrier = Arc::new(Barrier::new(2));
        let t = {
            let b = barrier.clone();
            thread::spawn(move || [b.wait().is_leader(), b.wait().is_leader()])
        };
        let mine = [barrier.wait().is_leader(), barrier.wait().is_leader()];
        let theirs = t.join().unwrap();
        assert!(mine[0] ^ theirs[0]);
        assert!(mine[1] ^ theirs[1]);
    });
}
