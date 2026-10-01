#![deny(warnings)]

//! Snapshot resumption (`Builder::snapshot`) against replay from the start.
//!
//! A snapshot restore claims to change only how an execution reaches its
//! divergence branch, never which executions exist nor what any of them
//! observes. Each model here runs under both, at several bounds and worker
//! counts, and must produce the same execution count and the same set of
//! behaviours.
//!
//! The binary installs `loom::alloc::Model`, whose contract the models keep:
//! a behaviour leaves the model only as a `u64` in a set allocated before the
//! check, which an execution inserts into without allocating.

use std::collections::hash_map::DefaultHasher;
use std::collections::BTreeSet;
use std::hash::{Hash, Hasher};
use std::sync::atomic::AtomicU64 as StdAtomicU64;
use std::sync::atomic::Ordering as StdOrdering;
use std::sync::Arc;

use loom::sync::atomic::{AtomicUsize, Ordering::*};
use loom::sync::{Condvar, Mutex};
use loom::thread;

#[global_allocator]
static ALLOC: loom::alloc::Model<std::alloc::System> = loom::alloc::Model(std::alloc::System);

/// A fixed-capacity set of nonzero `u64`s that inserts without allocating.
struct Sink(Box<[StdAtomicU64]>);

impl Sink {
    fn new() -> Arc<Sink> {
        Arc::new(Sink((0..1 << 16).map(|_| StdAtomicU64::new(0)).collect()))
    }

    fn insert(&self, value: u64) {
        let value = value | 1;
        let mask = self.0.len() - 1;
        let mut i = (value as usize).wrapping_mul(0x9E37_79B9) & mask;
        loop {
            match self.0[i].compare_exchange(0, value, StdOrdering::Relaxed, StdOrdering::Relaxed) {
                Ok(_) => return,
                Err(seen) if seen == value => return,
                Err(_) => i = (i + 1) & mask,
            }
        }
    }

    fn values(&self) -> BTreeSet<u64> {
        self.0
            .iter()
            .map(|v| v.load(StdOrdering::Relaxed))
            .filter(|&v| v != 0)
            .collect()
    }
}

fn hash(values: &[usize]) -> u64 {
    let mut h = DefaultHasher::new();
    values.hash(&mut h);
    h.finish()
}

fn explore<M>(
    threads: usize,
    bound: Option<usize>,
    sleep_sets: bool,
    snapshot: Option<usize>,
    model: M,
) -> (BTreeSet<u64>, usize)
where
    M: Fn() -> Vec<usize> + Send + Sync + 'static,
{
    let sink = Sink::new();
    let out = sink.clone();

    let mut builder = loom::model::Builder::new();
    builder.threads = threads;
    builder.preemption_bound = bound;
    builder.sleep_sets = sleep_sets;
    builder.max_permutations = None;
    builder.max_duration = None;
    builder.log = false;
    builder.probe = std::time::Duration::ZERO;
    builder.budgeted = false;
    builder.snapshot = snapshot;

    let stats = builder.check(move || {
        let observed = model();
        sink.insert(hash(&observed));
    });

    (out.values(), stats.executions)
}

fn assert_snapshots_preserve_exploration<M>(name: &str, model: M)
where
    M: Fn() -> Vec<usize> + Send + Sync + Clone + 'static,
{
    // Bounded, then unbounded (source sets and wakeup trees) with and
    // without sleep sets. Sharded unbounded runs settle frozen branches
    // through the claim record in whatever order workers reach them, so
    // their execution count is not fixed; their behaviours are.
    let configs = [
        (Some(1), true),
        (Some(2), true),
        (Some(3), true),
        (None, true),
        (None, false),
    ];
    for (bound, sleep_sets) in configs {
        let (base, base_n) = explore(1, bound, sleep_sets, None, model.clone());
        for threads in [1, 4] {
            for spacing in [1, 3, 8] {
                let (seen, n) = explore(threads, bound, sleep_sets, Some(spacing), model.clone());
                if threads == 1 || bound.is_some() {
                    assert_eq!(
                        n, base_n,
                        "{name}: bound {bound:?}, sleep sets {sleep_sets}, {threads} worker(s), \
                         spacing {spacing}: {n} executions against {base_n} replayed"
                    );
                }
                assert_eq!(
                    seen, base,
                    "{name}: bound {bound:?}, sleep sets {sleep_sets}, {threads} worker(s), \
                     spacing {spacing}: behaviours differ from replay"
                );
            }
        }
    }
}

fn store_buffering() -> Vec<usize> {
    let x = Arc::new(AtomicUsize::new(0));
    let y = Arc::new(AtomicUsize::new(0));

    let (x1, y1) = (x.clone(), y.clone());
    let a = thread::spawn(move || {
        x1.store(1, Relaxed);
        y1.load(Relaxed)
    });
    let (x2, y2) = (x.clone(), y.clone());
    let b = thread::spawn(move || {
        y2.store(1, Relaxed);
        x2.load(Relaxed)
    });

    vec![a.join().unwrap(), b.join().unwrap()]
}

/// Heap state a restore must rewind: boxes built and mutated across threads.
fn heap_and_mutex() -> Vec<usize> {
    let shared = Arc::new(Mutex::new(Vec::<Box<usize>>::new()));
    let mut handles = Vec::new();
    for t in 0..2 {
        let shared = shared.clone();
        handles.push(thread::spawn(move || {
            for i in 0..2 {
                shared.lock().unwrap().push(Box::new(t * 10 + i));
            }
        }));
    }
    for h in handles {
        h.join().unwrap();
    }
    let v = shared.lock().unwrap();
    v.iter().map(|b| **b).collect()
}

fn rmw_contention() -> Vec<usize> {
    let n = Arc::new(AtomicUsize::new(0));
    let mut handles = Vec::new();
    for _ in 0..3 {
        let n = n.clone();
        handles.push(thread::spawn(move || {
            let seen = n.load(Acquire);
            let _ = n.compare_exchange(seen, seen + 1, AcqRel, Acquire);
            seen
        }));
    }
    let mut out: Vec<usize> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    out.push(n.load(Relaxed));
    out
}

fn condvar_handoff() -> Vec<usize> {
    let pair = Arc::new((Mutex::new(0usize), Condvar::new()));
    let p = pair.clone();
    let t = thread::spawn(move || {
        let (lock, cv) = &*p;
        *lock.lock().unwrap() = 1;
        cv.notify_one();
    });
    let (lock, cv) = &*pair;
    let mut waits = 0;
    let mut guard = lock.lock().unwrap();
    while *guard == 0 {
        guard = cv.wait(guard).unwrap();
        waits += 1;
    }
    drop(guard);
    t.join().unwrap();
    vec![waits]
}

/// A thread spawned late, after snapshots of the execution exist.
fn late_spawn() -> Vec<usize> {
    let x = Arc::new(AtomicUsize::new(0));
    let x1 = x.clone();
    let a = thread::spawn(move || {
        x1.fetch_add(1, AcqRel);
        x1.fetch_add(1, AcqRel);
    });
    x.store(5, Release);
    let first = x.load(Acquire);
    let x2 = x.clone();
    let b = thread::spawn(move || x2.swap(9, AcqRel));
    a.join().unwrap();
    let swapped = b.join().unwrap();
    vec![first, swapped, x.load(Acquire)]
}

#[test]
fn snapshot_store_buffering() {
    assert_snapshots_preserve_exploration("store_buffering", store_buffering);
}

#[test]
fn snapshot_heap_and_mutex() {
    assert_snapshots_preserve_exploration("heap_and_mutex", heap_and_mutex);
}

#[test]
fn snapshot_rmw_contention() {
    assert_snapshots_preserve_exploration("rmw_contention", rmw_contention);
}

#[test]
fn snapshot_condvar_handoff() {
    assert_snapshots_preserve_exploration("condvar_handoff", condvar_handoff);
}

#[test]
fn snapshot_late_spawn() {
    assert_snapshots_preserve_exploration("late_spawn", late_spawn);
}

