//! Spurious failure of `compare_exchange_weak` under the Rust contract.
//!
//! Every test here sets `LOOM_SPURIOUS_WEAK_CAS=1` before its model, so the
//! spurious arm is explored whatever the host's own weak compare-exchange
//! does. The override is read once per process, and this binary holds only
//! tests that want it.

#![deny(warnings, rust_2018_idioms)]

use loom::sync::atomic::AtomicUsize;

use std::collections::HashSet;
use std::hash::Hash;
use std::sync::atomic::Ordering::{AcqRel, Acquire, Relaxed};
use std::sync::{Arc, Mutex};

/// Every outcome `f` produces across the whole exploration, spurious weak
/// failures included.
fn outcomes<T>(f: impl Fn() -> T + Send + Sync + 'static) -> HashSet<T>
where
    T: Hash + Eq + Clone + Send + 'static,
{
    // Every test in the binary sets the same value before its first
    // operation, so none can observe it unset.
    std::env::set_var("LOOM_SPURIOUS_WEAK_CAS", "1");

    let set = Arc::new(Mutex::new(HashSet::new()));
    let sink = set.clone();
    loom::model(move || {
        let v = f();
        sink.lock().unwrap().insert(v);
    });
    let out = set.lock().unwrap().clone();
    out
}

/// A weak compare-exchange may fail spuriously, reporting the value that
/// matched.
#[test]
fn weak_cas_fails_spuriously() {
    let out = outcomes(|| {
        let x = AtomicUsize::new(0);
        x.compare_exchange_weak(0, 1, Relaxed, Relaxed)
    });

    assert_eq!(out, HashSet::from([Ok(0), Err(0)]));
}

/// A retry loop terminates: the attempt right after a spurious failure does
/// not fail spuriously.
#[test]
fn weak_cas_retry_loop_terminates() {
    let out = outcomes(|| {
        let x = AtomicUsize::new(0);
        let mut attempts = 0;
        while x.compare_exchange_weak(0, 1, AcqRel, Acquire).is_err() {
            attempts += 1;
        }
        attempts
    });

    assert_eq!(out, HashSet::from([0, 1]));
}

/// `try_update` retries through the weak form, so its closure can see the
/// same value twice.
#[test]
fn try_update_retries_a_spurious_failure() {
    let out = outcomes(|| {
        let x = AtomicUsize::new(3);
        let mut calls = 0;
        let r = x.try_update(Relaxed, Relaxed, |v| {
            calls += 1;
            Some(v + 1)
        });
        (r, calls)
    });

    assert_eq!(out, HashSet::from([(Ok(3), 1), (Ok(3), 2)]));
}

/// A spurious failure is the exclusive monitor lost; a store landing on the
/// cell between two attempts clears it again, even one writing the value the
/// attempt expects. So a retry loop may fail spuriously twice running around
/// a peer's store, and still terminates once the peer is done.
#[test]
fn weak_cas_fails_spuriously_again_after_a_store_lands() {
    let out = outcomes(|| {
        let x = Arc::new(AtomicUsize::new(0));
        let t = {
            let x = x.clone();
            loom::thread::spawn(move || x.store(0, Relaxed))
        };
        let mut attempts = 0;
        while x.compare_exchange_weak(0, 1, AcqRel, Acquire).is_err() {
            attempts += 1;
        }
        t.join().unwrap();
        attempts
    });

    assert_eq!(out, HashSet::from([0, 1, 2]));
}
