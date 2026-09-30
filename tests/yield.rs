#![deny(warnings, rust_2018_idioms)]

use loom::sync::atomic::AtomicUsize;
use loom::thread;

use std::collections::BTreeSet;
use std::panic::{self, AssertUnwindSafe};
use std::sync::atomic::Ordering::{Relaxed, SeqCst};
use std::sync::{Arc, Mutex};

/// A poll loop that spins through `hint::spin_loop` waits for the peer: the
/// spinner does not run again while another thread can.
#[test]
fn spin_loop_completes() {
    loom::model(|| {
        let inc = Arc::new(AtomicUsize::new(0));

        {
            let inc = inc.clone();
            thread::spawn(move || {
                inc.store(1, Relaxed);
            });
        }

        loop {
            if 1 == inc.load(Relaxed) {
                return;
            }

            loom::hint::spin_loop();
        }
    });
}

fn outcomes(bound: Option<usize>, yield_first: bool) -> BTreeSet<usize> {
    let seen: Arc<Mutex<BTreeSet<usize>>> = Default::default();
    let out = seen.clone();
    let mut builder = loom::model::Builder::new();
    builder.preemption_bound = bound;
    builder.check(move || {
        let y = Arc::new(AtomicUsize::new(0));
        let t = {
            let y = y.clone();
            thread::spawn(move || {
                if yield_first {
                    thread::yield_now();
                }
                y.store(1, SeqCst);
            })
        };
        let r = y.load(SeqCst);
        t.join().unwrap();
        out.lock().unwrap().insert(r);
    });
    let seen = seen.lock().unwrap().clone();
    seen
}

/// `yield_now` is a scheduling point and nothing more, as `std`'s: the
/// yielder may run on, so a yield hides no order.
#[test]
fn yield_now_hides_no_order() {
    for bound in [None, Some(1), Some(2), Some(3)] {
        assert_eq!(outcomes(bound, true), outcomes(bound, false), "bound = {bound:?}");
    }
}

/// A poll loop that yields through `yield_now` promises no progress: once no
/// preemption hands the processor to the peer, it polls until the branch
/// limit, and the failure names the primitive a poll loop needs.
#[test]
fn a_yield_now_poll_loop_reaches_the_branch_limit() {
    let mut builder = loom::model::Builder::new();
    builder.preemption_bound = Some(1);
    builder.max_branches = 200;

    let result = panic::catch_unwind(AssertUnwindSafe(|| {
        builder.check(|| {
            let flag = Arc::new(AtomicUsize::new(0));
            let t = {
                let flag = flag.clone();
                thread::spawn(move || flag.store(1, Relaxed))
            };
            while flag.load(Relaxed) == 0 {
                thread::yield_now();
            }
            t.join().unwrap();
        })
    }));

    let payload = result.expect_err("a yield_now poll loop was given progress");
    let message = payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_default();
    assert!(message.contains("hint::spin_loop"), "{message}");
}
