#![deny(warnings)]

//! Measurement and contract tests for preemption-bounded exploration.
//!
//! Loom's bounded mode implements BPOR (Coons, Musuvathi & McKinley,
//! OOPSLA'13, Algorithm 3): standard DPOR backtrack points plus a
//! conservative point at the most recent context-switch boundary strictly
//! before the race. `Stats::conservative` attributes completed executions to
//! those conservative points, which is the measured ceiling on what a
//! bound-aware optimal DPOR could remove.

use std::collections::BTreeSet;
use std::sync::{Arc, Mutex};

use loom::sync::atomic::{AtomicUsize, Ordering::*};
use loom::thread;

fn builder(bound: Option<usize>, threads: usize) -> loom::model::Builder {
    let mut b = loom::model::Builder::new();

    // Pin everything the environment could otherwise leak in.
    b.preemption_bound = bound;
    b.threads = threads;
    b.max_permutations = None;
    b.max_duration = None;
    b.log = false;
    b.budgeted = false;
    b.stats = true;
    b
}

/// Two writers raced against a reader: enough dependent accesses that the
/// bounded walk provably plants conservative backtrack points.
fn racy_model(log: &Arc<Mutex<BTreeSet<usize>>>) -> impl Fn() + Send + Sync + 'static {
    let log = log.clone();

    move || {
        let x = Arc::new(AtomicUsize::new(0));

        let t1 = {
            let x = x.clone();
            thread::spawn(move || {
                x.store(1, SeqCst);
                x.store(2, SeqCst);
            })
        };
        let t2 = {
            let x = x.clone();
            thread::spawn(move || {
                x.fetch_add(10, SeqCst);
            })
        };

        t1.join().unwrap();
        t2.join().unwrap();

        log.lock().unwrap().insert(x.load(SeqCst));
    }
}

/// Attribution invariants: the count is a subset of executions, is stable
/// serial-vs-sharded in total, and vanishes without a bound (conservative
/// points only exist under one).
#[test]
fn conservative_attribution_invariants() {
    let log = Arc::new(Mutex::new(BTreeSet::new()));
    let bounded = builder(Some(1), 1).check(racy_model(&log));

    assert!(bounded.conservative <= bounded.executions);
    assert!(
        bounded.conservative > 0,
        "this model plants conservative points under bound 1; \
         attribution saw none (executions = {})",
        bounded.executions
    );

    let log = Arc::new(Mutex::new(BTreeSet::new()));
    let unbounded = builder(None, 1).check(racy_model(&log));

    assert_eq!(
        unbounded.conservative, 0,
        "conservative points cannot exist without a preemption bound"
    );
}

/// The behaviors a bounded run reaches must not change with the measurement
/// on: attribution observes the walk, never steers it.
#[test]
fn attribution_is_behavior_preserving() {
    let with = Arc::new(Mutex::new(BTreeSet::new()));
    let stats = builder(Some(1), 1).check(racy_model(&with));

    let without = Arc::new(Mutex::new(BTreeSet::new()));
    let mut plain = builder(Some(1), 1);
    plain.stats = false;
    let baseline = plain.check(racy_model(&without));

    assert_eq!(stats.executions, baseline.executions);
    assert_eq!(*with.lock().unwrap(), *without.lock().unwrap());
    assert_eq!(baseline.conservative, 0, "off means not collected");
}

/// Documents the divergence from BPOR's exploration gate (Algorithm 2,
/// Line 12): BPOR creates backtrack points unconditionally and refuses only
/// explorations whose *own* cost exceeds the bound — and scheduling a thread
/// where the previous thread just blocked or exited is free (Definition 2.5).
/// Loom instead refuses to create any point once the prefix has spent the
/// bound, so at `k = 0` it explores exactly one order of two free-running
/// writers even though both orders contain zero preemptions.
///
/// Un-ignore when the mark gate distinguishes free switches
/// (`initial_active == None`) from preemptive ones.
#[test]
#[ignore = "bound-gate coverage hole: free alternatives at saturated branches are refused"]
fn bound_zero_reaches_all_zero_preemption_orders() {
    let log: Arc<Mutex<BTreeSet<usize>>> = Arc::new(Mutex::new(BTreeSet::new()));
    let out = log.clone();

    builder(Some(0), 1).check(move || {
        let x = Arc::new(AtomicUsize::new(0));

        let t1 = {
            let x = x.clone();
            thread::spawn(move || x.store(1, SeqCst))
        };
        let t2 = {
            let x = x.clone();
            thread::spawn(move || x.store(2, SeqCst))
        };

        t1.join().unwrap();
        t2.join().unwrap();

        out.lock().unwrap().insert(x.load(SeqCst));
    });

    // Run t1 to completion, then t2: zero preemptions, final x == 2.
    // Run t2 to completion, then t1: zero preemptions, final x == 1.
    // Preemption-bounded coverage at k = 0 includes both.
    let seen = log.lock().unwrap();
    assert_eq!(
        *seen,
        BTreeSet::from([1, 2]),
        "a 0-preemption final state was never reached under bound 0"
    );
}
