#![deny(warnings, rust_2018_idioms)]

//! A failing model fails its test once, with its own diagnosis: no user
//! destructor runs against a failed execution, so none can turn the failure
//! into a double panic that aborts the process.

use loom::model::Builder;
use loom::sync::atomic::AtomicUsize;
use loom::sync::{Arc, Mutex};
use loom::thread;

use std::any::Any;
use std::hint::black_box;
use std::panic::{self, AssertUnwindSafe};
use std::sync::atomic::Ordering::Relaxed;
use std::time::{Duration, Instant};

/// Performs a tracked operation when dropped.
struct Touch(Arc<AtomicUsize>);

impl Drop for Touch {
    fn drop(&mut self) {
        self.0.fetch_add(1, Relaxed);
    }
}

fn message(payload: Box<dyn Any + Send>) -> String {
    match payload.downcast::<String>() {
        Ok(s) => *s,
        Err(payload) => match payload.downcast::<&'static str>() {
            Ok(s) => s.to_string(),
            Err(_) => panic!("model failed with a non-string payload"),
        },
    }
}

fn serial() -> Builder {
    let mut builder = Builder::new();
    builder.threads = 1;
    builder
}

/// Two threads take two mutexes in opposite orders, each holding a guard
/// whose `Drop` performs a tracked operation.
fn lock_order_inversion() {
    let a = Arc::new(Mutex::new(()));
    let b = Arc::new(Mutex::new(()));
    let hits = Arc::new(AtomicUsize::new(0));

    let th1 = {
        let (a, b, hits) = (a.clone(), b.clone(), hits.clone());
        thread::spawn(move || {
            let _touch = Touch(hits);
            let _a = a.lock().unwrap();
            let _b = b.lock().unwrap();
        })
    };
    let th2 = thread::spawn(move || {
        let _touch = Touch(hits);
        let _b = b.lock().unwrap();
        let _a = a.lock().unwrap();
    });

    th1.join().unwrap();
    th2.join().unwrap();
}

#[test]
#[should_panic(expected = "deadlock")]
fn deadlock_with_tracked_op_in_drop() {
    serial().check(lock_order_inversion);
}

#[test]
fn deadlock_report_names_what_each_thread_waits_on() {
    let payload = panic::catch_unwind(|| serial().check(lock_order_inversion))
        .expect_err("the lock-order inversion deadlocks");
    let report = message(payload);

    assert!(report.starts_with("deadlock"), "{report}");
    assert!(report.contains("thread 0: blocked on Notify"), "{report}");
    assert!(report.contains("thread 1: blocked on Mutex"), "{report}");
    assert!(report.contains("thread 2: blocked on Mutex"), "{report}");
}

#[test]
fn deadlock_report_locates_each_blocked_thread() {
    let mut builder = serial();
    builder.location = true;

    let payload = panic::catch_unwind(AssertUnwindSafe(|| builder.check(lock_order_inversion)))
        .expect_err("the lock-order inversion deadlocks");
    let report = message(payload);

    let located = report.lines().filter(|l| l.contains("model_failure.rs:")).count();
    assert_eq!(located, 3, "{report}");
}

#[test]
fn deadlock_report_names_a_parked_thread() {
    let payload = panic::catch_unwind(|| {
        serial().check(|| {
            let hits = Arc::new(AtomicUsize::new(0));
            let th = thread::spawn(move || {
                let _touch = Touch(hits);
                // Nothing unparks this thread: a lost wakeup.
                thread::park();
            });
            th.join().unwrap();
        })
    })
    .expect_err("the unmatched park deadlocks");
    let report = message(payload);

    assert!(report.starts_with("deadlock"), "{report}");
    assert!(report.contains("thread 0: blocked on Notify"), "{report}");
    assert!(report.contains("thread 1: parked"), "{report}");
}

#[test]
#[should_panic(expected = "original failure")]
fn panic_with_tracked_drops_reports_the_original_message() {
    serial().check(|| {
        let hits = Arc::new(AtomicUsize::new(0));
        let lock = Arc::new(Mutex::new(()));
        let _main = Touch(hits.clone());

        let th = {
            let (hits, lock) = (hits.clone(), lock.clone());
            thread::spawn(move || {
                let _touch = Touch(hits);
                let _guard = lock.lock().unwrap();
                panic!("original failure");
            })
        };

        drop(lock.lock().unwrap());
        th.join().unwrap();
    });
}

/// Takes the lock when dropped, as a guard type restoring shared state does.
struct LockOnDrop(Arc<Mutex<()>>);

impl Drop for LockOnDrop {
    fn drop(&mut self) {
        drop(self.0.lock().unwrap());
    }
}

#[test]
#[should_panic(expected = "original failure")]
fn panic_whose_unwinding_would_deadlock_reports_the_original_message() {
    serial().check(|| {
        let lock = Arc::new(Mutex::new(()));

        let th = {
            let lock = lock.clone();
            thread::spawn(move || {
                let _restore = LockOnDrop(lock);
                panic!("original failure");
            })
        };

        let _held = lock.lock().unwrap();
        th.join().unwrap();
    });
}

#[derive(Debug, PartialEq)]
struct Custom(u32);

#[test]
fn panic_payload_that_unwinds_to_the_thread_root_is_preserved() {
    let payload = panic::catch_unwind(|| {
        serial().check(|| {
            let th = thread::spawn(|| {
                // No tracked operation between here and the thread's root.
                panic::panic_any(Custom(7));
            });
            th.join().unwrap();
        })
    })
    .expect_err("the model thread panics");

    assert_eq!(payload.downcast_ref::<Custom>(), Some(&Custom(7)));
}

/// A state space no run finishes within the tests' time budget: four threads
/// of mutually dependent read-modify-writes.
fn unbounded_model() {
    let x = Arc::new(AtomicUsize::new(0));

    let ths: Vec<_> = (0..3)
        .map(|_| {
            let x = x.clone();
            thread::spawn(move || {
                for _ in 0..8 {
                    x.fetch_add(1, Relaxed);
                }
            })
        })
        .collect();

    for _ in 0..8 {
        x.fetch_add(1, Relaxed);
    }

    for th in ths {
        th.join().unwrap();
    }
}

fn check_times_out(threads: usize) {
    let mut builder = Builder::new();
    builder.max_duration = Some(Duration::from_millis(300));
    builder.threads = threads;
    builder.budgeted = false;
    builder.probe = Duration::ZERO;
    builder.preemption_bound = None;

    let start = Instant::now();
    let payload = panic::catch_unwind(AssertUnwindSafe(|| builder.check(unbounded_model)))
        .expect_err("a run cut short by max_duration must fail");
    let elapsed = start.elapsed();
    let report = message(payload);

    assert!(report.contains("max_duration"), "{report}");
    assert!(elapsed < Duration::from_secs(10), "stopped after {elapsed:?}");
}

#[test]
fn max_duration_fails_a_serial_run() {
    check_times_out(1);
}

#[test]
fn max_duration_fails_a_sharded_run() {
    check_times_out(4);
}

#[inline(never)]
fn recurse(depth: usize) -> usize {
    let frame = black_box([depth as u8; 1024]);

    if depth == 0 {
        return frame[0] as usize;
    }

    recurse(depth - 1) + frame[depth % 1024] as usize
}

#[test]
fn model_threads_run_on_a_full_size_stack() {
    loom::model(|| {
        let th = thread::spawn(|| black_box(recurse(256)));
        black_box(recurse(256));
        th.join().unwrap();
    });
}
