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

            loom::hint::spin_loop();
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
                loom::hint::spin_loop();
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
                loom::hint::spin_loop();
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
                loom::hint::spin_loop();
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
                loom::hint::spin_loop();
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

/// DPOR over an `Arc`'s counts: an inspection must be reversed against every
/// count change concurrent with it, not only the latest of one kind.
mod dependence {
    use loom::sync::{Arc, Mutex};
    use loom::thread;
    use std::collections::BTreeSet;
    use std::sync::Mutex as StdMutex;

    fn counts<F>(bound: Option<usize>, sleep_sets: bool, f: F) -> BTreeSet<usize>
    where
        F: Fn() -> usize + Send + Sync + 'static,
    {
        let seen: std::sync::Arc<StdMutex<BTreeSet<usize>>> = Default::default();
        let out = seen.clone();
        let mut builder = loom::model::Builder::new();
        builder.threads = 1;
        builder.preemption_bound = bound;
        builder.sleep_sets = sleep_sets;
        builder.check(move || {
            let n = f();
            out.lock().unwrap().insert(n);
        });
        let seen = seen.lock().unwrap().clone();
        seen
    }

    const CONFIGS: [(Option<usize>, bool); 4] =
        [(None, false), (None, true), (Some(2), false), (Some(3), false)];

    /// A clone in one thread, then a drop in another that is joined before
    /// main inspects: the drop must not hide the clone, which main may read
    /// either side of.
    #[test]
    fn a_later_drop_does_not_hide_a_concurrent_clone() {
        let model = || {
            let a = Arc::new(0usize);
            let gate = Arc::new(Mutex::new(()));
            let held = gate.lock().unwrap();

            let cloner = {
                let (a1, gate) = (a.clone(), gate.clone());
                thread::spawn(move || {
                    let a2 = a1.clone();
                    drop(gate.lock().unwrap());
                    drop(a2);
                    drop(a1);
                })
            };
            let dropper = {
                let b1 = a.clone();
                thread::spawn(move || drop(b1))
            };
            dropper.join().unwrap();
            let n = Arc::strong_count(&a);
            drop(held);
            cloner.join().unwrap();
            n
        };

        for (bound, sleep_sets) in CONFIGS {
            assert_eq!(
                counts(bound, sleep_sets, model),
                BTreeSet::from([2, 3]),
                "bound = {bound:?}, sleep_sets = {sleep_sets}"
            );
        }
    }

    /// An inspection before a concurrent drop: the drop must be reversed
    /// against it, so main reads the count before and after the drop.
    #[test]
    fn a_drop_is_reversed_against_an_earlier_inspection() {
        let model = || {
            let a = Arc::new(0usize);
            let b1 = a.clone();
            let dropper = thread::spawn(move || drop(b1));
            let n = Arc::strong_count(&a);
            dropper.join().unwrap();
            n
        };

        for (bound, sleep_sets) in CONFIGS {
            assert_eq!(
                counts(bound, sleep_sets, model),
                BTreeSet::from([1, 2]),
                "bound = {bound:?}, sleep_sets = {sleep_sets}"
            );
        }
    }

    /// An upgrade after a concurrent `Weak::strong_count` must be reversed
    /// against it: the weak side reads the count before and after.
    #[test]
    fn an_upgrade_is_reversed_against_an_earlier_weak_inspection() {
        let model = || {
            let a = Arc::new(0usize);
            let w1 = Arc::downgrade(&a);
            let w2 = w1.clone();
            let upgrader = thread::spawn(move || w1.upgrade());
            let n = w2.strong_count();
            let upgraded = upgrader.join().unwrap();
            drop((upgraded, w2, a));
            n
        };

        for (bound, sleep_sets) in CONFIGS {
            assert_eq!(
                counts(bound, sleep_sets, model),
                BTreeSet::from([1, 2]),
                "bound = {bound:?}, sleep_sets = {sleep_sets}"
            );
        }
    }

    /// A `Weak` dropped after a concurrent `weak_count` must be reversed
    /// against it.
    #[test]
    fn a_weak_drop_is_reversed_against_an_earlier_weak_count() {
        let model = || {
            let a = Arc::new(0usize);
            let w = Arc::downgrade(&a);
            let dropper = thread::spawn(move || drop(w));
            let n = Arc::weak_count(&a);
            dropper.join().unwrap();
            n
        };

        for (bound, sleep_sets) in CONFIGS {
            assert_eq!(
                counts(bound, sleep_sets, model),
                BTreeSet::from([0, 1]),
                "bound = {bound:?}, sleep_sets = {sleep_sets}"
            );
        }
    }
}

/// Drops that do not end the strong count commute: only the final drop
/// conflicts with the others, and every drop with every count read.
mod drop_independence {
    use loom::sync::Arc;
    use loom::thread;
    use std::collections::BTreeSet;
    use std::sync::Mutex as StdMutex;

    /// Records which thread ran the value's destructor: the final dropper.
    struct Last(std::sync::Arc<StdMutex<Option<String>>>);

    impl Drop for Last {
        fn drop(&mut self) {
            *self.0.lock().unwrap() = Some(format!("{:?}", thread::current().id()));
        }
    }

    /// Two threads drop their clones while main keeps its own: neither drop
    /// is final, so the drops add no order to explore beyond what the same
    /// threads doing nothing already have.
    #[test]
    fn non_final_drops_are_explored_once() {
        let executions = |drop_clones: bool| {
            loom::model::Builder::new()
                .check(move || {
                    let a = Arc::new(());
                    let clones = [a.clone(), a.clone()];
                    let mut kept = Vec::new();
                    let hs: Vec<_> = clones
                        .into_iter()
                        .map(|c| {
                            let c = if drop_clones { Some(c) } else { kept.push(c); None };
                            thread::spawn(move || drop(c))
                        })
                        .collect();
                    for h in hs {
                        h.join().unwrap();
                    }
                    // Without the racing drops, main drops the clones itself.
                    drop(kept);
                    drop(a);
                })
                .executions
        };
        assert_eq!(executions(true), executions(false));
    }

    /// Three holders race to drop: every thread can be the final dropper, at
    /// every bound, though non-final drops no longer reorder among themselves.
    #[test]
    fn every_thread_can_drop_last() {
        for bound in [None, Some(1), Some(2), Some(3)] {
            let seen: std::sync::Arc<StdMutex<BTreeSet<String>>> = Default::default();
            let out = seen.clone();
            let mut builder = loom::model::Builder::new();
            builder.threads = 1;
            builder.preemption_bound = bound;
            builder.check(move || {
                let who = std::sync::Arc::new(StdMutex::new(None));
                let a = Arc::new(Last(who.clone()));
                let hs: Vec<_> = (0..2)
                    .map(|_| {
                        let c = a.clone();
                        thread::spawn(move || drop(c))
                    })
                    .collect();
                drop(a);
                for h in hs {
                    h.join().unwrap();
                }
                let last = who.lock().unwrap().take().unwrap();
                out.lock().unwrap().insert(last);
            });
            let seen = seen.lock().unwrap().clone();
            assert_eq!(seen.len(), 3, "bound = {bound:?}: {seen:?}");
        }
    }
}
