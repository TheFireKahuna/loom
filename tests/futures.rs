#![cfg(feature = "futures")]
#![deny(warnings, rust_2018_idioms)]

use loom::future::{block_on, AtomicWaker};
use loom::sync::atomic::AtomicUsize;
use loom::thread;

use futures_util::future::poll_fn;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::Arc;
use std::task::Poll;

struct Chan {
    num: AtomicUsize,
    task: AtomicWaker,
}

#[test]
fn atomic_waker_valid() {
    use std::task::Poll::*;

    const NUM_NOTIFY: usize = 2;

    loom::model(|| {
        let chan = Arc::new(Chan {
            num: AtomicUsize::new(0),
            task: AtomicWaker::new(),
        });

        for _ in 0..NUM_NOTIFY {
            let chan = chan.clone();

            thread::spawn(move || {
                chan.num.fetch_add(1, Relaxed);
                chan.task.wake();
            });
        }

        block_on(poll_fn(move |cx| {
            chan.task.register_by_ref(cx.waker());

            if NUM_NOTIFY == chan.num.load(Relaxed) {
                return Ready(());
            }

            Pending
        }));
    });
}

// Tests futures spuriously poll as this is a very common pattern
#[test]
fn spurious_poll() {
    use loom::sync::atomic::AtomicBool;
    use loom::sync::atomic::Ordering::{Acquire, Release};

    let poll_thrice = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let actual = poll_thrice.clone();

    loom::model(move || {
        let gate = Arc::new(AtomicBool::new(false));
        let mut cnt = 0;

        let num_poll = block_on(poll_fn(|cx| {
            if cnt == 0 {
                let gate = gate.clone();
                let waker = cx.waker().clone();

                thread::spawn(move || {
                    gate.store(true, Release);
                    waker.wake();
                });
            }

            cnt += 1;

            if gate.load(Acquire) {
                Poll::Ready(cnt)
            } else {
                Poll::Pending
            }
        }));

        if num_poll == 3 {
            poll_thrice.store(true, Release);
        }

        assert!(num_poll > 0 && num_poll <= 3, "actual = {}", num_poll);
    });

    assert!(actual.load(Acquire));
}

/// A spurious wake of `block_on` is one step of the waiting thread, not a
/// wait for the others' progress that makes stale values unreadable: a
/// re-poll with no wake behind it can read again the value the first poll
/// read, even after the first poll saw, through another cell, that the
/// peer's store had landed. A second poll that reads 0 had no wake behind it
/// (the peer's wake carries its store).
#[test]
fn a_spurious_wake_is_a_step_not_a_spin() {
    use loom::sync::Mutex;
    use std::task::Waker;

    let stale_repoll = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let reached = stale_repoll.clone();

    loom::model(move || {
        let x = Arc::new(AtomicUsize::new(0));
        let landed = Arc::new(AtomicUsize::new(0));
        let slot = Arc::new(Mutex::new(None::<Waker>));
        let (x2, landed2, slot2) = (x.clone(), landed.clone(), slot.clone());

        let th = thread::spawn(move || {
            x2.store(1, Relaxed);
            landed2.store(1, Relaxed);
            if let Some(waker) = slot2.lock().unwrap().take() {
                waker.wake();
            }
        });

        let mut polls = 0;
        let mut saw_landed = false;
        block_on(poll_fn(|cx| {
            *slot.lock().unwrap() = Some(cx.waker().clone());
            polls += 1;
            if polls == 1 {
                saw_landed = landed.load(Relaxed) == 1;
            }
            if x.load(Relaxed) == 1 {
                return Poll::Ready(());
            }
            if polls == 2 {
                if saw_landed {
                    stale_repoll.store(true, Relaxed);
                }
                return Poll::Ready(());
            }
            Poll::Pending
        }));
        th.join().unwrap();
    });

    assert!(
        reached.load(Relaxed),
        "a spurious re-poll never read the value its first poll read after the store landed"
    );
}
