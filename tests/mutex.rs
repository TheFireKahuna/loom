#![deny(warnings, rust_2018_idioms)]

use loom::cell::UnsafeCell;
use loom::sync::atomic::AtomicUsize;
use loom::sync::Mutex;
use loom::thread;

use std::rc::Rc;
use std::sync::atomic::Ordering::SeqCst;

#[test]
fn mutex_enforces_mutual_exclusion() {
    loom::model(|| {
        let data = Rc::new((Mutex::new(0), AtomicUsize::new(0)));

        let ths: Vec<_> = (0..2)
            .map(|_| {
                let data = data.clone();

                thread::spawn(move || {
                    let mut locked = data.0.lock().unwrap();

                    let prev = data.1.fetch_add(1, SeqCst);
                    assert_eq!(prev, *locked);
                    *locked += 1;
                })
            })
            .collect();

        for th in ths {
            th.join().unwrap();
        }

        let locked = data.0.lock().unwrap();

        assert_eq!(*locked, data.1.load(SeqCst));
    });
}

#[test]
fn mutex_establishes_seq_cst() {
    loom::model(|| {
        struct Data {
            cell: UnsafeCell<usize>,
            flag: Mutex<bool>,
        }

        let data = Rc::new(Data {
            cell: UnsafeCell::new(0),
            flag: Mutex::new(false),
        });

        {
            let data = data.clone();

            thread::spawn(move || {
                unsafe { data.cell.with_mut(|v| *v = 1) };
                *data.flag.lock().unwrap() = true;
            });
        }

        let flag = *data.flag.lock().unwrap();

        if flag {
            let v = unsafe { data.cell.with(|v| *v) };
            assert_eq!(v, 1);
        }
    });
}

#[test]
fn mutex_into_inner() {
    loom::model(|| {
        let lock = Rc::new(Mutex::new(0));

        let ths: Vec<_> = (0..2)
            .map(|_| {
                let lock = lock.clone();

                thread::spawn(move || {
                    *lock.lock().unwrap() += 1;
                })
            })
            .collect();

        for th in ths {
            th.join().unwrap();
        }

        let lock = Rc::try_unwrap(lock).unwrap().into_inner().unwrap();
        assert_eq!(lock, 2);
    })
}

// An unlock is a step of its own: a peer's `try_lock` may run while a critical
// section with no modelled op inside it still holds the lock.
#[test]
fn try_lock_sees_an_empty_critical_section_held() {
    use loom::sync::Arc;
    use std::sync::atomic::AtomicBool;

    static FAILED: AtomicBool = AtomicBool::new(false);
    loom::model(|| {
        let m = Arc::new(Mutex::new(()));
        let m2 = m.clone();
        let t = thread::spawn(move || drop(m2.lock().unwrap()));
        if m.try_lock().is_err() {
            FAILED.store(true, SeqCst);
        }
        t.join().unwrap();
    });
    assert!(FAILED.load(SeqCst), "try_lock never saw the lock held");
}

// A blocking acquire races the unlock that would let it in, even while it is
// disabled: at bound 1, main is preempted at its first unlock, the peer stores
// and then blocks on the lock, and main runs on without a second preemption,
// reading the store with the peer's acquisition still to come. Every other
// schedule of this outcome takes two preemptions.
#[test]
fn a_blocked_acquire_races_the_unlock_that_admits_it() {
    use loom::sync::atomic::AtomicUsize as LoomUsize;
    use loom::sync::atomic::Ordering::Acquire;
    use loom::sync::Arc;
    use std::collections::BTreeSet;

    for bound in [Some(1), None] {
        let seen = std::sync::Arc::new(std::sync::Mutex::new(BTreeSet::new()));
        let seen_ = seen.clone();
        let mut b = loom::model::Builder::new();
        b.preemption_bound = bound;
        b.threads = 1;
        b.check(move || {
            let m = Arc::new(Mutex::new(0usize));
            let c = Arc::new(LoomUsize::new(0));
            let (m2, c2) = (m.clone(), c.clone());
            let t = thread::spawn(move || {
                c2.store(1, SeqCst);
                let mut g = m2.lock().unwrap();
                *g += 1;
                *g
            });
            let first = {
                let mut g = m.lock().unwrap();
                *g += 1;
                *g
            };
            let read = c.load(Acquire);
            let second = {
                let mut g = m.lock().unwrap();
                *g += 2;
                *g
            };
            let peer = t.join().unwrap();
            seen_.lock().unwrap().insert((first, read, second, peer));
        });
        assert!(
            seen.lock().unwrap().contains(&(1, 1, 3, 4)),
            "bound {bound:?}: the peer's store before main's read, with its lock after main's second, was never explored"
        );
    }
}

// Without a bound, a blocking acquire races the acquire before it, past the
// unlock between: running it before that unlock is impossible, before that
// acquire is the other order of the critical sections.
#[test]
fn unbounded_search_runs_both_orders_of_two_critical_sections() {
    use loom::sync::Arc;
    use std::collections::BTreeSet;

    let seen = std::sync::Arc::new(std::sync::Mutex::new(BTreeSet::new()));
    let seen_ = seen.clone();
    let mut b = loom::model::Builder::new();
    b.preemption_bound = None;
    b.threads = 1;
    b.check(move || {
        let m = Arc::new(Mutex::new(0usize));
        let m2 = m.clone();
        let t = thread::spawn(move || {
            let mut g = m2.lock().unwrap();
            *g += 1;
            *g
        });
        let main = {
            let mut g = m.lock().unwrap();
            *g += 2;
            *g
        };
        let peer = t.join().unwrap();
        seen_.lock().unwrap().insert((main, peer));
    });
    let seen = seen.lock().unwrap();
    assert!(
        seen.contains(&(2, 3)) && seen.contains(&(3, 1)),
        "both orders of the critical sections: {seen:?}"
    );
}
