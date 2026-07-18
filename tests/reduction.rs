#![deny(warnings)]

//! Differential coverage tests for the two exploration reductions:
//! sleep sets (`Builder::sleep_sets`) and parallel sharding
//! (`Builder::threads`).
//!
//! Both claim to change only *how many times* the search visits a behavior,
//! never *which* behaviors it reaches. That claim is what these tests check,
//! and it is not a claim to take on argument alone: sleep sets are proved
//! sound for unbounded DPOR, but combining a partial-order reduction with a
//! preemption bound is a known source of silently lost coverage (Coons,
//! Musuvathi & Emmi, *Bounded Partial-Order Reduction*, OOPSLA'13 — a bound
//! can cut the sibling subtree that a sleep set is pruning against). So every
//! model here is explored under each bound the reductions might interact
//! with, and the set of observed behaviors is compared against an
//! unreduced serial run at that same bound.
//!
//! A behavior is a per-execution fingerprint: every value each thread read,
//! tagged by reader, plus whatever final state the model records. Losing an
//! interleaving that any thread could distinguish drops a fingerprint and
//! fails the comparison.

use std::collections::BTreeSet;
use std::sync::{Arc, Mutex};

use loom::sync::atomic::AtomicU128;
use loom::sync::atomic::{AtomicUsize, Ordering::*};
use loom::thread;

/// Collects one fingerprint per execution.
#[derive(Clone, Default)]
struct Log(Arc<Mutex<Vec<String>>>);

impl Log {
    fn record(&self, what: impl std::fmt::Display) {
        self.0.lock().unwrap().push(what.to_string());
    }

    /// Order-independent fingerprint. Entries are tagged by their writer, so
    /// sorting canonicalizes without collapsing distinct observations.
    fn fingerprint(&self) -> String {
        let mut entries = self.0.lock().unwrap().clone();
        entries.sort();
        entries.join(",")
    }
}

/// Run `model` to exhaustion and return every behavior it exhibited, with the
/// number of executions it took to get there.
fn explore<M>(threads: usize, sleep_sets: bool, bound: Option<usize>, model: M) -> (BTreeSet<String>, usize)
where
    M: Fn(&Log) + Send + Sync + 'static,
{
    let seen: Arc<Mutex<BTreeSet<String>>> = Arc::new(Mutex::new(BTreeSet::new()));
    let out = seen.clone();

    let mut builder = loom::model::Builder::new();

    // Pin every knob the ambient environment could otherwise supply: these
    // tests compare runs against each other, so a stray `LOOM_*` variable
    // would compare two different questions.
    builder.threads = threads;
    builder.sleep_sets = sleep_sets;
    builder.preemption_bound = bound;
    builder.max_permutations = None;
    builder.max_duration = None;
    builder.log = false;

    // These models are far too small to reach the probe on their own, and a
    // run that never shards would test nothing here. `budgeted` off likewise:
    // under a parallel test harness the shared budget would otherwise hand
    // back one worker and the comparison would be against itself.
    builder.probe = std::time::Duration::ZERO;
    builder.budgeted = false;

    let stats = builder.check(move || {
        let log = Log::default();
        model(&log);
        out.lock().unwrap().insert(log.fingerprint());
    });

    let seen = seen.lock().unwrap().clone();
    (seen, stats.executions)
}

/// Every bound the reductions are asked to coexist with, plus the unbounded
/// search they are proved against.
const BOUNDS: &[Option<usize>] = &[None, Some(1), Some(2), Some(3)];

/// The core assertion: at each bound, neither reduction may lose a behavior
/// that an unreduced serial search finds, nor invent one it does not.
fn assert_reductions_preserve_coverage<M>(name: &str, model: M)
where
    M: Fn(&Log) + Send + Sync + Clone + 'static,
{
    for &bound in BOUNDS {
        let (baseline, plain) = explore(1, false, bound, model.clone());

        for &(threads, sleep_sets) in &[(1, true), (4, false), (4, true)] {
            let (reduced, count) = explore(threads, sleep_sets, bound, model.clone());

            let lost: Vec<_> = baseline.difference(&reduced).collect();
            let gained: Vec<_> = reduced.difference(&baseline).collect();

            assert!(
                lost.is_empty(),
                "{name}: bound={bound:?} threads={threads} sleep_sets={sleep_sets} \
                 LOST {} of {} behaviors ({} executions vs {plain} unreduced): {lost:?}",
                lost.len(),
                baseline.len(),
                count,
            );

            // A behavior the unreduced search cannot reach means the
            // reduction changed the model's semantics, not just its
            // scheduling — at least as serious as losing one.
            assert!(
                gained.is_empty(),
                "{name}: bound={bound:?} threads={threads} sleep_sets={sleep_sets} \
                 INVENTED {} behaviors: {gained:?}",
                gained.len(),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Models
//
// Each is small enough to explore exhaustively at every bound, and shaped so
// that a dropped interleaving changes what some thread can read.
// ---------------------------------------------------------------------------

/// Store buffering. Two threads write one location and read the other; both
/// reads seeing 0 is the relaxed-only outcome.
fn store_buffering(log: &Log) {
    let x = Arc::new(AtomicUsize::new(0));
    let y = Arc::new(AtomicUsize::new(0));

    let t = {
        let (x, y, log) = (x.clone(), y.clone(), log.clone());
        thread::spawn(move || {
            x.store(1, Relaxed);
            log.record(format_args!("t1:y={}", y.load(Relaxed)));
        })
    };

    y.store(1, Relaxed);
    log.record(format_args!("t0:x={}", x.load(Relaxed)));

    t.join().unwrap();
}

/// Message passing over a release/acquire pair, with a third thread racing on
/// the flag — enough dependence to keep sleep sets from pruning trivially.
fn message_passing(log: &Log) {
    let data = Arc::new(AtomicUsize::new(0));
    let flag = Arc::new(AtomicUsize::new(0));

    let writer = {
        let (data, flag) = (data.clone(), flag.clone());
        thread::spawn(move || {
            data.store(7, Relaxed);
            flag.store(1, Release);
        })
    };

    let disturber = {
        let flag = flag.clone();
        thread::spawn(move || {
            flag.store(2, Relaxed);
        })
    };

    let f = flag.load(Acquire);
    log.record(format_args!("t0:flag={f}"));

    if f == 1 {
        log.record(format_args!("t0:data={}", data.load(Relaxed)));
    }

    writer.join().unwrap();
    disturber.join().unwrap();
}

/// Three threads on three separate cells: every operation commutes with every
/// other, which is the case sleep sets exist to collapse.
fn disjoint_objects(log: &Log) {
    let cells: Vec<_> = (0..3).map(|_| Arc::new(AtomicUsize::new(0))).collect();

    let handles: Vec<_> = cells
        .iter()
        .enumerate()
        .skip(1)
        .map(|(i, cell)| {
            let (cell, log) = (cell.clone(), log.clone());
            thread::spawn(move || {
                cell.store(i, Relaxed);
                log.record(format_args!("t{i}:c={}", cell.load(Relaxed)));
            })
        })
        .collect();

    cells[0].store(9, Relaxed);
    log.record(format_args!("t0:c={}", cells[0].load(Relaxed)));

    for h in handles {
        h.join().unwrap();
    }
}

/// Disjoint *lanes* of one 128-bit cell. Independence here comes from the
/// mask-scoped dependence relation rather than from distinct objects, so this
/// covers the path unique to this fork.
fn disjoint_lanes(log: &Log) {
    const LOW: u128 = u64::MAX as u128;
    const HIGH: u128 = (u64::MAX as u128) << 64;

    let cell = Arc::new(AtomicU128::new(0));

    let t = {
        let (cell, log) = (cell.clone(), log.clone());
        thread::spawn(move || {
            cell.store_masked(HIGH, 1u128 << 64, Relaxed);
            log.record(format_args!("t1:low={}", cell.load_masked(LOW, Relaxed)));
        })
    };

    cell.store_masked(LOW, 1, Relaxed);
    log.record(format_args!(
        "t0:high={}",
        cell.load_masked(HIGH, Relaxed) >> 64
    ));

    t.join().unwrap();

    // A whole-cell load spans both lanes: one linearization point over
    // regions the threads touched independently.
    log.record(format_args!("final:{}", cell.load(Relaxed)));
}

/// Read-modify-write contention. Every operation conflicts, so a sleep set
/// must never prune here; this is the model that catches over-eager
/// independence.
fn rmw_contention(log: &Log) {
    let n = Arc::new(AtomicUsize::new(0));

    let t = {
        let (n, log) = (n.clone(), log.clone());
        thread::spawn(move || {
            log.record(format_args!("t1:swap={}", n.fetch_add(1, AcqRel)));
        })
    };

    log.record(format_args!("t0:swap={}", n.fetch_add(10, AcqRel)));

    t.join().unwrap();
    log.record(format_args!("final:{}", n.load(Acquire)));
}

/// A mutex-guarded counter. Lock operations are opaque, so this checks that
/// the conservative side of the independence relation holds up.
fn mutex_counter(log: &Log) {
    use loom::sync::Mutex as LoomMutex;

    let lock = Arc::new(LoomMutex::new(0usize));
    let free = Arc::new(AtomicUsize::new(0));

    let t = {
        let (lock, free, log) = (lock.clone(), free.clone(), log.clone());
        thread::spawn(move || {
            free.store(1, Relaxed);
            let mut g = lock.lock().unwrap();
            *g += 1;
            log.record(format_args!("t1:count={}", *g));
        })
    };

    {
        let mut g = lock.lock().unwrap();
        *g += 10;
        log.record(format_args!("t0:count={}", *g));
    }

    log.record(format_args!("t0:free={}", free.load(Relaxed)));
    t.join().unwrap();
}

/// Mixed: one pair of threads racing on a shared cell while a third works on
/// its own. Sleep sets must prune the independent thread's reorderings
/// without touching the contended pair's.
fn mixed_independence(log: &Log) {
    let shared = Arc::new(AtomicUsize::new(0));
    let private = Arc::new(AtomicUsize::new(0));

    let racer = {
        let (shared, log) = (shared.clone(), log.clone());
        thread::spawn(move || {
            shared.store(1, Relaxed);
            log.record(format_args!("t1:shared={}", shared.load(Relaxed)));
        })
    };

    let loner = {
        let (private, log) = (private.clone(), log.clone());
        thread::spawn(move || {
            private.store(5, Relaxed);
            log.record(format_args!("t2:private={}", private.load(Relaxed)));
        })
    };

    shared.store(2, Relaxed);
    log.record(format_args!("t0:shared={}", shared.load(Relaxed)));

    racer.join().unwrap();
    loner.join().unwrap();
}

// ---------------------------------------------------------------------------
// Coverage equivalence
// ---------------------------------------------------------------------------

#[test]
fn coverage_store_buffering() {
    assert_reductions_preserve_coverage("store_buffering", store_buffering);
}

#[test]
fn coverage_message_passing() {
    assert_reductions_preserve_coverage("message_passing", message_passing);
}

#[test]
fn coverage_disjoint_objects() {
    assert_reductions_preserve_coverage("disjoint_objects", disjoint_objects);
}

#[test]
fn coverage_disjoint_lanes() {
    assert_reductions_preserve_coverage("disjoint_lanes", disjoint_lanes);
}

#[test]
fn coverage_rmw_contention() {
    assert_reductions_preserve_coverage("rmw_contention", rmw_contention);
}

#[test]
fn coverage_mutex_counter() {
    assert_reductions_preserve_coverage("mutex_counter", mutex_counter);
}

#[test]
fn coverage_mixed_independence() {
    assert_reductions_preserve_coverage("mixed_independence", mixed_independence);
}

// ---------------------------------------------------------------------------
// The reductions have to actually reduce
//
// Coverage equivalence is satisfied trivially by a reduction that does
// nothing, so pin the other side down too.
// ---------------------------------------------------------------------------

/// Threads that both touch two cells: partially independent, and big enough
/// at a loose bound to contain redundancy worth cutting.
fn interleaved_cells() {
    let x = Arc::new(AtomicUsize::new(0));
    let y = Arc::new(AtomicUsize::new(0));

    let handles: Vec<_> = (0..2)
        .map(|i| {
            let (x, y) = (x.clone(), y.clone());
            thread::spawn(move || {
                x.store(i + 1, Relaxed);
                y.store(i + 1, Relaxed);
                x.load(Relaxed);
                y.load(Relaxed);
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }
}

#[test]
fn sleep_sets_cut_redundant_executions() {
    // Only asserted at a bound loose enough for redundancy to exist. A tight
    // preemption bound is itself a strong reduction — it cuts the repeated
    // interleavings before a sleep set gets to see them — so at bounds 1 and 2
    // these models leave the reduction with nothing to do. That is a property
    // of the bound, not a defect here, and `..._leave_fully_dependent_models_alone`
    // pins down that being inert is what "nothing to do" looks like.
    let mut builder = loom::model::Builder::new();
    builder.threads = 1;
    builder.preemption_bound = Some(3);
    builder.max_branches = 10_000;

    let without = {
        builder.sleep_sets = false;
        builder.check(interleaved_cells).executions
    };

    let with = {
        builder.sleep_sets = true;
        builder.check(interleaved_cells).executions
    };

    assert!(
        with < without,
        "sleep sets explored {with} executions, no better than {without} without them"
    );
}

#[test]
fn sleep_sets_leave_fully_dependent_models_alone() {
    // Nothing commutes, so there is nothing to sleep through: the reduction
    // should be inert rather than "helpful".
    let (_, without) = explore(1, false, Some(2), rmw_contention);
    let (_, with) = explore(1, true, Some(2), rmw_contention);

    assert_eq!(
        with, without,
        "sleep sets changed the execution count of a model with no \
         independent operations"
    );
}

/// The tree a sharded run walks must not depend on how many workers walk it.
///
/// This is the invariant that catches the two ways sharding goes wrong, both
/// of which are otherwise silent. If a donated subtree cannot record a
/// backtrack point in the prefix it no longer owns, the count *falls* as
/// workers are added and interleavings are lost; if a branch stays markable in
/// two copies at once, the count *rises* and subtrees are walked twice. Only
/// an exactly worker-count-independent total says neither is happening.
#[test]
fn sharding_walks_the_same_tree_at_every_worker_count() {
    let models: &[(&str, fn(&Log))] = &[
        ("store_buffering", store_buffering),
        ("message_passing", message_passing),
        ("disjoint_lanes", disjoint_lanes),
        ("rmw_contention", rmw_contention),
        ("mixed_independence", mixed_independence),
        ("mutex_counter", mutex_counter),
    ];

    // Checked with sleep sets off, where the invariant is exact. Splitting a
    // subtree also seeds each side's sleep set with what the other took, so
    // with them on the amount of reduction legitimately depends on where the
    // splits landed — sound, but not a fixed number to compare against.
    for &(name, model) in models {
        for &bound in BOUNDS {
            let (_, two) = explore(2, false, bound, model);

            for workers in [3, 4, 8] {
                let (_, n) = explore(workers, false, bound, model);

                assert_eq!(
                    n, two,
                    "{name}: bound={bound:?} explored {n} executions on \
                     {workers} workers but {two} on 2 — sharding changed the \
                     tree, which means lost or duplicated subtrees"
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Failures still surface
//
// A reduction that silently swallowed an assertion failure would pass every
// coverage test above, because the fingerprint is only recorded on the
// executions that complete.
// ---------------------------------------------------------------------------

fn racy_model_panics(threads: usize, sleep_sets: bool) -> bool {
    let mut builder = loom::model::Builder::new();
    builder.threads = threads;
    builder.sleep_sets = sleep_sets;
    builder.preemption_bound = Some(2);
    builder.probe = std::time::Duration::ZERO;
    builder.budgeted = false;

    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        builder.check(|| {
            let x = Arc::new(AtomicUsize::new(0));

            let t = {
                let x = x.clone();
                thread::spawn(move || x.store(1, Relaxed))
            };

            // Fails exactly on the interleavings where the store lands first.
            assert_eq!(x.load(Relaxed), 0, "store was observed");

            t.join().unwrap();
        });
    }))
    .is_err()
}

#[test]
fn reductions_still_find_a_real_bug() {
    for &(threads, sleep_sets) in &[(1, false), (1, true), (4, false), (4, true)] {
        assert!(
            racy_model_panics(threads, sleep_sets),
            "threads={threads} sleep_sets={sleep_sets}: a model that fails on \
             some interleaving was reported clean"
        );
    }
}
