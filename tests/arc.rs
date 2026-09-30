#![deny(warnings, rust_2018_idioms)]

use loom::cell::UnsafeCell;
use loom::sync::atomic::AtomicBool;
use loom::sync::atomic::Ordering::{Acquire, Release};
use loom::sync::Arc;
use loom::sync::Notify;
use loom::thread;

struct State {
    data: UnsafeCell<usize>,
    guard: AtomicBool,
}

impl Drop for State {
    fn drop(&mut self) {
        self.data.with(|ptr| unsafe {
            assert_eq!(1, *ptr);
        });
    }
}

#[test]
fn basic_usage() {
    loom::model(|| {
        let num = Arc::new(State {
            data: UnsafeCell::new(0),
            guard: AtomicBool::new(false),
        });

        let num2 = num.clone();
        thread::spawn(move || {
            num2.data.with_mut(|ptr| unsafe { *ptr = 1 });
            num2.guard.store(true, Release);
        });

        loop {
            if num.guard.load(Acquire) {
                num.data.with(|ptr| unsafe {
                    assert_eq!(1, *ptr);
                });
                break;
            }

            thread::yield_now();
        }
    });
}

#[test]
fn sync_in_drop() {
    loom::model(|| {
        let num = Arc::new(State {
            data: UnsafeCell::new(0),
            guard: AtomicBool::new(false),
        });

        let num2 = num.clone();
        thread::spawn(move || {
            num2.data.with_mut(|ptr| unsafe { *ptr = 1 });
            num2.guard.store(true, Release);
            drop(num2);
        });

        drop(num);
    });
}

#[test]
#[should_panic]
fn detect_mem_leak() {
    loom::model(|| {
        let num = Arc::new(State {
            data: UnsafeCell::new(0),
            guard: AtomicBool::new(false),
        });

        std::mem::forget(num);
    });
}

#[test]
fn try_unwrap_succeeds() {
    loom::model(|| {
        let num = Arc::new(0usize);
        let num2 = Arc::clone(&num);
        drop(num2);
        let _ = Arc::try_unwrap(num).unwrap();
    });
}

#[test]
fn try_unwrap_fails() {
    loom::model(|| {
        let num = Arc::new(0usize);
        let num2 = Arc::clone(&num);
        let num = Arc::try_unwrap(num).unwrap_err();

        drop(num2);

        let _ = Arc::try_unwrap(num).unwrap();
    });
}

#[test]
fn try_unwrap_multithreaded() {
    loom::model(|| {
        let num = Arc::new(0usize);
        let num2 = Arc::clone(&num);
        let can_drop = Arc::new(Notify::new());
        let thread = {
            let can_drop = can_drop.clone();
            thread::spawn(move || {
                can_drop.wait();
                drop(num2);
            })
        };

        // The other thread is holding the other arc clone, so we can't unwrap the arc.
        let num = Arc::try_unwrap(num).unwrap_err();

        // Allow the thread to proceed.
        can_drop.notify();

        // After the thread drops the other clone, the arc should be
        // unwrappable.
        thread.join().unwrap();
        let _ = Arc::try_unwrap(num).unwrap();
    });
}

mod weak {
    use loom::sync::atomic::{AtomicUsize, Ordering::Relaxed};
    use loom::sync::{Arc, Weak};
    use loom::thread;

    // The last strong drop races an upgrade: both outcomes are explored, and
    // an upgrade ahead of the drop may read the cell either side of the store.
    #[test]
    fn upgrade_races_the_last_drop() {
        let seen = std::sync::Arc::new(std::sync::Mutex::new(std::collections::HashSet::new()));
        let seen_ = seen.clone();
        loom::model(move || {
            let a = Arc::new(AtomicUsize::new(0));
            let w = Arc::downgrade(&a);
            let t = thread::spawn(move || w.upgrade().map(|a| a.load(Relaxed)));
            a.store(1, Relaxed);
            drop(a);
            seen_.lock().unwrap().insert(t.join().unwrap());
        });
        assert_eq!(*seen.lock().unwrap(), [None, Some(0), Some(1)].into());
    }

    #[test]
    fn weak_blocks_get_mut_but_not_try_unwrap() {
        loom::model(|| {
            let mut a = Arc::new(1);
            let w = Arc::downgrade(&a);
            assert!(Arc::get_mut(&mut a).is_none());
            assert_eq!((Arc::strong_count(&a), Arc::weak_count(&a)), (1, 1));
            assert_eq!(Arc::try_unwrap(a).ok(), Some(1));
            assert!(w.upgrade().is_none());
            assert_eq!(w.weak_count(), 0);
            assert!(Weak::<u8>::new().upgrade().is_none());
        });
    }

    #[test]
    fn new_cyclic_and_make_mut() {
        struct Node {
            me: Weak<Node>,
            v: u32,
        }
        impl Clone for Node {
            fn clone(&self) -> Node {
                Node { me: Weak::new(), v: self.v }
            }
        }
        loom::model(|| {
            let mut n = Arc::new_cyclic(|me| {
                assert!(me.upgrade().is_none());
                Node { me: me.clone(), v: 1 }
            });
            assert!(Arc::ptr_eq(&n.me.upgrade().unwrap(), &n));
            // Only its own `Weak` shares it: moved, not cloned.
            Arc::make_mut(&mut n).v = 2;
            assert_eq!(n.v, 2);
            assert!(n.me.upgrade().is_none());

            let other = n.clone();
            Arc::make_mut(&mut n).v = 3;
            assert_eq!((n.v, other.v), (3, 2));
        });
    }
}

// `std`'s orderings on the two counts: each call below is the atomic step, or
// steps, `std` performs, with the orderings `std` gives them.
mod std_parity {
    use loom::cell::UnsafeCell;
    use loom::sync::atomic::{AtomicBool, Ordering::Relaxed};
    use loom::sync::Arc;
    use loom::thread;

    fn builder() -> loom::model::Builder {
        let mut b = loom::model::Builder::new();
        b.preemption_bound = Some(3);
        b
    }

    // `try_unwrap` is one CAS of the strong count from 1 to 0: an upgrade
    // either precedes it, and it fails, or follows it, and the upgrade fails.
    #[test]
    fn try_unwrap_is_one_step_against_upgrade() {
        let seen = std::sync::Arc::new(std::sync::Mutex::new(std::collections::HashSet::new()));
        let seen_ = seen.clone();
        builder().check(move || {
            let a = Arc::new(5u32);
            let w = Arc::downgrade(&a);
            let t = thread::spawn(move || w.upgrade());
            let unwrapped = Arc::try_unwrap(a).is_ok();
            let upgraded = t.join().unwrap().is_some();
            seen_.lock().unwrap().insert((unwrapped, upgraded));
        });
        assert_eq!(*seen.lock().unwrap(), [(true, false), (false, true)].into());
    }

    // `make_mut` with only a `Weak` sharing the value races an upgrade: the
    // upgrade wins, and the value is cloned, or loses, and it is moved.
    #[test]
    fn make_mut_move_out_against_upgrade() {
        let seen = std::sync::Arc::new(std::sync::Mutex::new(std::collections::HashSet::new()));
        let seen_ = seen.clone();
        builder().check(move || {
            let mut a = Arc::new(5u32);
            let w = Arc::downgrade(&a);
            let t = thread::spawn(move || w.upgrade().map(|a| *a));
            *Arc::make_mut(&mut a) += 1;
            assert_eq!(*a, 6);
            seen_.lock().unwrap().insert(t.join().unwrap());
        });
        assert_eq!(*seen.lock().unwrap(), [None, Some(5)].into());
    }

    // `std`'s `make_mut` takes the strong count to 0 before it reads the weak
    // count, so an upgrade in between fails while the `Weak` it came through
    // may still be dropped in time for the value to stay in place.
    #[test]
    fn make_mut_window_fails_an_upgrade_that_leaves_the_value_in_place() {
        let seen = std::sync::Arc::new(std::sync::Mutex::new(std::collections::HashSet::new()));
        let seen_ = seen.clone();
        builder().check(move || {
            let mut a = Arc::new(5u32);
            let before = Arc::as_ptr(&a);
            let w = Arc::downgrade(&a);
            let t = thread::spawn(move || w.upgrade().is_some());
            Arc::make_mut(&mut a);
            let in_place = Arc::as_ptr(&a) == before;
            seen_.lock().unwrap().insert((t.join().unwrap(), in_place));
        });
        assert!(seen.lock().unwrap().contains(&(false, true)));
    }

    // `strong_count` is a `Relaxed` load: reading 1 does not synchronize with
    // the drop that made it 1.
    #[test]
    #[should_panic(expected = "Causality violation")]
    fn strong_count_does_not_acquire() {
        builder().check(|| {
            let a = Arc::new(UnsafeCell::new(0u32));
            let b = a.clone();
            let t = thread::spawn(move || {
                b.with_mut(|p| unsafe { *p = 1 });
                drop(b);
            });
            while Arc::strong_count(&a) != 1 {
                thread::yield_now();
            }
            a.with(|p| unsafe { *p });
            t.join().unwrap();
        });
    }

    // Dropping a `Weak` releases the weak count; `upgrade` acquires the strong
    // count. Nothing orders one after the other.
    #[test]
    #[should_panic(expected = "Causality violation")]
    fn weak_drop_does_not_sync_with_upgrade() {
        builder().check(|| {
            let data = Arc::new(UnsafeCell::new(0u32));
            let owner = Arc::new(());
            let w1 = Arc::downgrade(&owner);
            let w2 = Arc::downgrade(&owner);
            let flag = Arc::new(AtomicBool::new(false));
            let (d2, f2) = (data.clone(), flag.clone());
            let t = thread::spawn(move || {
                d2.with_mut(|p| unsafe { *p = 1 });
                drop(w1);
                f2.store(true, Relaxed);
            });
            while !flag.load(Relaxed) {
                thread::yield_now();
            }
            let _s = w2.upgrade().unwrap();
            data.with(|p| unsafe { *p });
            t.join().unwrap();
        });
    }

    // A failed `try_unwrap` is a `Relaxed` CAS: it acquires nothing.
    #[test]
    #[should_panic(expected = "Causality violation")]
    fn failed_try_unwrap_does_not_acquire() {
        builder().check(|| {
            let data = Arc::new(UnsafeCell::new(0u32));
            let owner = Arc::new(());
            let keep = owner.clone();
            let other = owner.clone();
            let flag = Arc::new(AtomicBool::new(false));
            let (d2, f2) = (data.clone(), flag.clone());
            let t = thread::spawn(move || {
                d2.with_mut(|p| unsafe { *p = 1 });
                drop(other);
                f2.store(true, Relaxed);
            });
            while !flag.load(Relaxed) {
                thread::yield_now();
            }
            assert!(Arc::try_unwrap(owner).is_err());
            data.with(|p| unsafe { *p });
            drop(keep);
            t.join().unwrap();
        });
    }

    // A `get_mut` that a live `Weak` fails takes its lock CAS's `Relaxed`
    // failure: it acquires nothing.
    #[test]
    #[should_panic(expected = "Causality violation")]
    fn get_mut_failed_by_a_weak_does_not_acquire() {
        builder().check(|| {
            let data = Arc::new(UnsafeCell::new(0u32));
            let mut owner = Arc::new(());
            let other = owner.clone();
            let w = Arc::downgrade(&owner);
            let flag = Arc::new(AtomicBool::new(false));
            let (d2, f2) = (data.clone(), flag.clone());
            let t = thread::spawn(move || {
                d2.with_mut(|p| unsafe { *p = 1 });
                drop(other);
                f2.store(true, Relaxed);
            });
            while !flag.load(Relaxed) {
                thread::yield_now();
            }
            assert!(Arc::get_mut(&mut owner).is_none());
            data.with(|p| unsafe { *p });
            drop(w);
            t.join().unwrap();
        });
    }

    // A successful `get_mut` acquires both counts, so it orders the peer's
    // strong and weak drops before the read.
    #[test]
    fn get_mut_success_acquires_both_counts() {
        builder().check(|| {
            let data = Arc::new(UnsafeCell::new(0u32));
            let mut owner = Arc::new(());
            let other = owner.clone();
            let w = Arc::downgrade(&owner);
            let (d2, done) = (data.clone(), Arc::new(AtomicBool::new(false)));
            let done2 = done.clone();
            let t = thread::spawn(move || {
                d2.with_mut(|p| unsafe { *p = 1 });
                drop(other);
                drop(w);
                done2.store(true, Relaxed);
            });
            if Arc::get_mut(&mut owner).is_some() {
                data.with(|p| unsafe { *p });
            }
            t.join().unwrap();
            let _ = done;
        });
    }
}
