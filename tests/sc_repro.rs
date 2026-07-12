#![deny(warnings, rust_2018_idioms)]

use loom::sync::atomic::AtomicUsize;
use loom::thread;

use std::collections::HashSet;
use std::sync::atomic::Ordering::{Acquire, SeqCst};
use std::sync::{Arc, Mutex};

// The DPOR shadowing hole (fork fix; also present upstream in 0.7.2): objects
// tracked ONE last access, so a spawned thread's own leading load overwrote
// the record of the peer's load — its following CAS was then only checked
// against its own load (trivially ordered, "no race") and the child's
// load→CAS prefix was never explored ahead of the peer's plain load.
#[test]
fn load_cas_prefix_explored_before_peer_load() {
    let values = Arc::new(Mutex::new(HashSet::new()));
    let values_ = values.clone();
    loom::model(move || {
        let x = Arc::new(AtomicUsize::new(0));
        let xc = x.clone();
        let th = thread::spawn(move || {
            let _ = xc.load(Acquire);
            let _ = xc.compare_exchange(0, 1, SeqCst, SeqCst);
        });
        let v = x.load(Acquire);
        th.join().unwrap();
        values.lock().unwrap().insert(v);
    });
    let values = values_.lock().unwrap();
    assert!(values.contains(&0), "peer-load-first never explored");
    assert!(
        values.contains(&1),
        "the child's load→CAS prefix was never explored ahead of the peer's load"
    );
}

// Same hole with a plain store after the leading load.
#[test]
fn load_store_prefix_explored_before_peer_load() {
    let values = Arc::new(Mutex::new(HashSet::new()));
    let values_ = values.clone();
    loom::model(move || {
        let x = Arc::new(AtomicUsize::new(0));
        let xc = x.clone();
        let th = thread::spawn(move || {
            let _ = xc.load(Acquire);
            xc.store(1, SeqCst);
        });
        let v = x.load(Acquire);
        th.join().unwrap();
        values.lock().unwrap().insert(v);
    });
    let values = values_.lock().unwrap();
    assert!(values.contains(&0), "peer-load-first never explored");
    assert!(
        values.contains(&1),
        "the child's load→store prefix was never explored ahead of the peer's load"
    );
}

// A child publishes via an SC RMW; the parent reads the same cell with
// Acquire. Both the stale and the fresh read must be explored.
#[test]
fn acquire_load_races_sc_rmw() {
    let values = Arc::new(Mutex::new(HashSet::new()));
    let values_ = values.clone();
    loom::model(move || {
        let x = Arc::new(AtomicUsize::new(0));
        let xc = x.clone();
        let th = thread::spawn(move || {
            let _ = xc.compare_exchange(0, 1, SeqCst, SeqCst);
        });
        let v = x.load(Acquire);
        th.join().unwrap();
        values.lock().unwrap().insert(v);
    });
    let values = values_.lock().unwrap();
    assert!(values.contains(&0), "stale read never explored");
    assert!(values.contains(&1), "fresh read never explored");
}
