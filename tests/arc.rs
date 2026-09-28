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
