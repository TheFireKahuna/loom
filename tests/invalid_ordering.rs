#![deny(warnings, rust_2018_idioms)]

use loom::sync::atomic::{AtomicU128, AtomicUsize};
use std::sync::atomic::Ordering::*;

#[test]
#[should_panic(expected = "there is no such thing as a release load")]
fn release_load_panics() {
    loom::model(|| {
        AtomicUsize::new(0).load(Release);
    });
}

#[test]
#[should_panic(expected = "there is no such thing as an acquire-release load")]
fn acq_rel_lane_load_panics() {
    loom::model(|| {
        AtomicU128::new(0).lane_u32(4).load(AcqRel);
    });
}

#[test]
#[should_panic(expected = "there is no such thing as an acquire store")]
fn acquire_store_panics() {
    loom::model(|| {
        AtomicUsize::new(0).store(1, Acquire);
    });
}

#[test]
#[should_panic(expected = "there is no such thing as a release failure ordering")]
fn release_failure_ordering_panics() {
    loom::model(|| {
        let _ = AtomicUsize::new(0).compare_exchange(1, 2, AcqRel, Release);
    });
}

#[test]
// core's `try_update` loads with `fetch_order` before its first compare.
#[should_panic(expected = "there is no such thing as an acquire-release load")]
fn acq_rel_try_update_fetch_order_panics() {
    loom::model(|| {
        let _ = AtomicUsize::new(0).try_update(AcqRel, AcqRel, |v| Some(v + 1));
    });
}

#[test]
#[should_panic(expected = "there is no such thing as an acquire-release failure ordering")]
fn acq_rel_weak_failure_ordering_panics() {
    loom::model(|| {
        let _ = AtomicUsize::new(0).compare_exchange_weak(0, 1, SeqCst, AcqRel);
    });
}

#[test]
fn unconditional_rmw_accepts_every_ordering() {
    loom::model(|| {
        let a = AtomicUsize::new(0);
        for order in [Relaxed, Acquire, Release, AcqRel, SeqCst] {
            a.fetch_add(1, order);
            a.swap(0, order);
        }
        let _ = a.compare_exchange(0, 1, Release, Relaxed);
        let _ = a.compare_exchange(1, 2, Relaxed, SeqCst);
    });
}
