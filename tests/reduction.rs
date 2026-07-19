#![deny(warnings)]

//! Differential coverage tests for parallel sharding (`Builder::threads`).
//!
//! Sharding claims to change only *which worker* visits a subtree, never
//! which behaviors the search reaches nor how many executions it takes to
//! reach them. That claim is not one to take on argument alone: combining a
//! partial-order reduction with a preemption bound is a known source of
//! silently lost coverage (Coons, Musuvathi & Emmi, *Bounded Partial-Order
//! Reduction*, OOPSLA'13 — a bound can cut the sibling subtree the reduction
//! is pruning against). So every model here is explored under each bound, and
//! the observed behaviors are compared against a serial run at that bound.
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
fn explore<M>(threads: usize, bound: Option<usize>, model: M) -> (BTreeSet<String>, usize)
where
    M: Fn(&Log) + Send + Sync + 'static,
{
    explore_with(threads, bound, model, true, true)
}

fn explore_with<M>(
    threads: usize,
    bound: Option<usize>,
    model: M,
    reuse_objects: bool,
    sleep_sets: bool,
) -> (BTreeSet<String>, usize)
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
    builder.reuse_objects = reuse_objects;
    builder.sleep_sets = sleep_sets;

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

/// The core assertion: at each bound, sharding may not lose a behavior a
/// serial search finds, nor invent one it does not.
fn assert_reductions_preserve_coverage<M>(name: &str, model: M)
where
    M: Fn(&Log) + Send + Sync + Clone + 'static,
{
    for &bound in BOUNDS {
        let (baseline, plain) = explore(1, bound, model.clone());

        for &threads in &[2, 4, 8] {
            let (reduced, count) = explore(threads, bound, model.clone());

            let lost: Vec<_> = baseline.difference(&reduced).collect();
            let gained: Vec<_> = reduced.difference(&baseline).collect();

            assert!(
                lost.is_empty(),
                "{name}: bound={bound:?} threads={threads} \
                 LOST {} of {} behaviors ({count} executions vs {plain} serial): {lost:?}",
                lost.len(),
                baseline.len(),
            );

            // A behavior the serial search cannot reach means sharding changed
            // the model's semantics, not just who explored it — at least as
            // serious as losing one.
            assert!(
                gained.is_empty(),
                "{name}: bound={bound:?} threads={threads} \
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

/// Threads that both touch two cells: partially independent, and the largest
/// tree here at a loose bound.
fn interleaved_cells(_: &Log) {
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

/// A sharded run must walk exactly the tree a serial run walks — not a
/// superset, not a subset, and not something that depends on how many workers
/// happened to be running.
///
/// This is the invariant that catches the two ways sharding goes wrong, both
/// of which are otherwise silent. If a donated subtree cannot record a
/// backtrack point in the prefix it no longer owns, the count *falls* and
/// interleavings are lost; if a branch stays markable in two copies at once,
/// the count *rises* and subtrees are walked twice.
///
/// Compared against **serial**, deliberately. Comparing worker counts only to
/// each other passes just as happily when every one of them over-explores by
/// the same factor, which is exactly how an earlier design hid a 5.3x tax.
///
/// Sleep sets are pinned off here: their pruning at a *shared* branch reads
/// the claim record at passage time, which is deliberately time-sensitive
/// (later marks only prune more), so exact counts are a property of the
/// unpruned walk. Coverage under sleep sets is asserted separately by
/// `sleep_sets_cover_every_behavior_of_the_full_walk`.
#[test]
fn sharding_walks_the_same_tree_as_a_serial_run() {
    let models: &[(&str, fn(&Log))] = &[
        ("store_buffering", store_buffering),
        ("message_passing", message_passing),
        ("disjoint_lanes", disjoint_lanes),
        ("rmw_contention", rmw_contention),
        ("mixed_independence", mixed_independence),
        ("mutex_counter", mutex_counter),
        ("interleaved_cells", interleaved_cells),
    ];

    for &(name, model) in models {
        for &bound in BOUNDS {
            let (_, serial) = explore_with(1, bound, model, true, false);

            for workers in [2, 3, 4, 8] {
                let (_, n) = explore_with(workers, bound, model, true, false);

                assert_eq!(
                    n, serial,
                    "{name}: bound={bound:?} explored {n} executions on \
                     {workers} workers but {serial} serially — sharding changed \
                     the tree, which means lost or duplicated subtrees"
                );
            }
        }
    }
}

/// Object reincarnation (`Builder::reuse_objects`) claims a recycled object
/// is extensionally identical to a freshly constructed one. If any reused
/// state leaked across executions — a stale store in a ring, a surviving
/// access record, a partition that failed to collapse — the checker would
/// walk a different tree, so behaviors *and* execution counts must match a
/// virgin-store run exactly.
#[test]
fn reincarnation_walks_the_same_tree_as_virgin_stores() {
    let models: &[(&str, fn(&Log))] = &[
        ("store_buffering", store_buffering),
        ("message_passing", message_passing),
        ("disjoint_lanes", disjoint_lanes),
        ("rmw_contention", rmw_contention),
        ("mixed_independence", mixed_independence),
        ("mutex_counter", mutex_counter),
        ("interleaved_cells", interleaved_cells),
    ];

    for &(name, model) in models {
        for &bound in BOUNDS {
            let (virgin_seen, virgin_n) = explore_with(1, bound, model, false, true);
            let (reused_seen, reused_n) = explore_with(1, bound, model, true, true);

            assert_eq!(
                virgin_seen, reused_seen,
                "{name}: bound={bound:?} reincarnated objects changed the \
                 observable behaviors — reused state leaked across executions"
            );
            assert_eq!(
                reused_n, virgin_n,
                "{name}: bound={bound:?} explored {reused_n} executions with \
                 reincarnation but {virgin_n} with virgin stores — a recycled \
                 object is not extensionally identical to a fresh one"
            );
        }
    }
}

/// Sleep sets claim to walk a *subset* of the full tree that still reaches
/// every behavior: an execution is only pruned when the interleavings it
/// leads to reorder independent operations of a subtree some sibling
/// alternative covers. Losing a behavior here means the independence relation
/// or the deference order is wrong — the one failure mode that would make the
/// checker silently unsound.
///
/// Counts are asserted as an upper bound only. Serially the pruned walk is
/// deterministic, but at shared branches pruning grows with the marks that
/// have landed by passage time, so a sharded count may fall anywhere at or
/// below the full walk's.
#[test]
fn sleep_sets_cover_every_behavior_of_the_full_walk() {
    let models: &[(&str, fn(&Log))] = &[
        ("store_buffering", store_buffering),
        ("message_passing", message_passing),
        ("disjoint_objects", disjoint_objects),
        ("disjoint_lanes", disjoint_lanes),
        ("rmw_contention", rmw_contention),
        ("mixed_independence", mixed_independence),
        ("mutex_counter", mutex_counter),
        ("interleaved_cells", interleaved_cells),
    ];

    for &(name, model) in models {
        for &bound in BOUNDS {
            let (full_seen, full_n) = explore_with(1, bound, model, true, false);
            let (slept_seen, slept_n) = explore_with(1, bound, model, true, true);

            let lost: Vec<_> = full_seen.difference(&slept_seen).collect();
            assert!(
                lost.is_empty(),
                "{name}: bound={bound:?} sleep sets LOST {} of {} behaviors \
                 ({slept_n} executions vs {full_n} full): {lost:?}",
                lost.len(),
                full_seen.len(),
            );
            assert_eq!(
                slept_seen, full_seen,
                "{name}: bound={bound:?} sleep sets INVENTED behaviors"
            );
            // Scout races can open an ancestor alternative the full walk
            // would not have — tolerate a few extra executions, trip on
            // runaway overshoot.
            assert!(
                slept_n <= full_n + full_n / 8 + 8,
                "{name}: bound={bound:?} pruned walk took {slept_n} executions, \
                 full walk {full_n} — a sleep set may never add exploration"
            );

            for workers in [2, 4, 8] {
                let (sharded_seen, sharded_n) =
                    explore_with(workers, bound, model, true, true);

                assert_eq!(
                    sharded_seen, full_seen,
                    "{name}: bound={bound:?} workers={workers} sharded sleep-set \
                     walk changed the observable behaviors"
                );
                assert!(
                    sharded_n <= full_n + full_n / 8 + 8,
                    "{name}: bound={bound:?} workers={workers} sharded pruned \
                     walk took {sharded_n} executions, full walk {full_n}"
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Wake precision
//
// The one silently-unsound failure mode of a sleep set is an under-precise
// wake: a sleeper not woken by a genuinely conflicting operation keeps its
// covered-elsewhere justification past the point it expired, and the
// interleavings pruned after that are not covered by anything. These models
// pin the two subtle axes of the conflict relation: lane-mask overlap within
// one cell, and load/store kind. (Over-waking is merely less pruning.)
// ---------------------------------------------------------------------------

/// Three threads on one 128-bit cell. The low-lane writers conflict with each
/// other and with the low-lane readers but not with the high-lane writer; a
/// wake keyed on object identity alone would be sound here, one keyed on a
/// wrong mask would not.
fn lane_wake_precision(log: &Log) {
    const LOW: u128 = u64::MAX as u128;
    const HIGH: u128 = (u64::MAX as u128) << 64;

    let cell = Arc::new(AtomicU128::new(0));

    let low_writer = {
        let (cell, log) = (cell.clone(), log.clone());
        thread::spawn(move || {
            cell.store_masked(LOW, 1, Relaxed);
            log.record(format_args!("t1:low={}", cell.load_masked(LOW, Relaxed)));
        })
    };

    let high_writer = {
        let cell = cell.clone();
        thread::spawn(move || {
            cell.store_masked(HIGH, 1u128 << 64, Relaxed);
        })
    };

    cell.store_masked(LOW, 2, Relaxed);
    log.record(format_args!("t0:low={}", cell.load_masked(LOW, Relaxed)));

    low_writer.join().unwrap();
    high_writer.join().unwrap();

    log.record(format_args!("final:{:x}", cell.load(Relaxed)));
}

/// Loads must not wake sleepers on other loads, but a store must wake both.
/// Two readers race one writer; the readers' mutual order is irrelevant, the
/// writer's position relative to each read is everything.
fn load_kind_precision(log: &Log) {
    let x = Arc::new(AtomicUsize::new(0));

    let reader = {
        let (x, log) = (x.clone(), log.clone());
        thread::spawn(move || {
            log.record(format_args!("t1:x={}", x.load(Relaxed)));
            log.record(format_args!("t1:x'={}", x.load(Relaxed)));
        })
    };

    let writer = {
        let x = x.clone();
        thread::spawn(move || {
            x.store(1, Relaxed);
        })
    };

    log.record(format_args!("t0:x={}", x.load(Relaxed)));

    reader.join().unwrap();
    writer.join().unwrap();
}

#[test]
fn coverage_lane_wake_precision() {
    assert_reductions_preserve_coverage("lane_wake_precision", lane_wake_precision);
}

#[test]
fn coverage_load_kind_precision() {
    assert_reductions_preserve_coverage("load_kind_precision", load_kind_precision);
}

// ---------------------------------------------------------------------------
// Randomized differential fuzzing
//
// The hand-written models above test the failure modes we thought of. The
// fuzzer tests the ones we did not: random small programs — threads × ops ×
// cells × orderings, including lane-masked ops on one wide cell and a
// blocking mutex — each explored to exhaustion by the full walk and the
// pruned walk, asserting the behavior sets are identical. Seeds are fixed,
// so a failure names a reproducible program.
// ---------------------------------------------------------------------------

/// Deterministic xorshift; no dependency, stable across platforms.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// One randomly generated straight-line op.
#[derive(Clone, Copy, Debug)]
enum FuzzOp {
    /// `cells[i].store(v)`
    Store(usize, usize),
    /// log `cells[i].load()`
    Load(usize),
    /// log `cells[i].fetch_add(v)`
    Rmw(usize, usize),
    /// `wide.store_masked(lane l)`
    WideStore(u32, u64),
    /// log `wide.load_masked(lane l)`
    WideLoad(u32),
    /// lock the mutex, add v, log the total
    Mutex(usize),
}

const ORDERINGS: [loom::sync::atomic::Ordering; 5] = [Relaxed, Acquire, Release, AcqRel, SeqCst];

impl FuzzOp {
    fn generate(rng: &mut Rng) -> FuzzOp {
        match rng.below(12) {
            0..=2 => FuzzOp::Store(rng.below(2) as usize, 1 + rng.below(3) as usize),
            3..=5 => FuzzOp::Load(rng.below(2) as usize),
            6..=7 => FuzzOp::Rmw(rng.below(2) as usize, 1 + rng.below(3) as usize),
            8 => FuzzOp::WideStore(rng.below(2) as u32, 1 + rng.below(3)),
            9 => FuzzOp::WideLoad(rng.below(2) as u32),
            _ => FuzzOp::Mutex(1 + rng.below(3) as usize),
        }
    }

    /// A load-shaped ordering for reads, store-shaped for writes, either for
    /// RMWs — mirroring what real code can legally write.
    fn ordering(&self, rng: &mut Rng) -> loom::sync::atomic::Ordering {
        match self {
            FuzzOp::Load(_) | FuzzOp::WideLoad(_) => [Relaxed, Acquire, SeqCst][rng.below(3) as usize],
            FuzzOp::Store(..) | FuzzOp::WideStore(..) => {
                [Relaxed, Release, SeqCst][rng.below(3) as usize]
            }
            _ => ORDERINGS[rng.below(5) as usize],
        }
    }

    fn run(
        self,
        ord: loom::sync::atomic::Ordering,
        who: usize,
        cells: &[Arc<AtomicUsize>],
        wide: &Arc<AtomicU128>,
        mutex: &Arc<loom::sync::Mutex<usize>>,
        log: &Log,
    ) {
        match self {
            FuzzOp::Store(c, v) => cells[c].store(v, ord),
            FuzzOp::Load(c) => log.record(format_args!("t{who}:c{c}={}", cells[c].load(ord))),
            FuzzOp::Rmw(c, v) => {
                log.record(format_args!("t{who}:r{c}={}", cells[c].fetch_add(v, ord)))
            }
            FuzzOp::WideStore(lane, v) => {
                let mask = (u64::MAX as u128) << (64 * lane);
                wide.store_masked(mask, (v as u128) << (64 * lane), ord);
            }
            FuzzOp::WideLoad(lane) => {
                let mask = (u64::MAX as u128) << (64 * lane);
                log.record(format_args!(
                    "t{who}:w{lane}={:x}",
                    wide.load_masked(mask, ord) >> (64 * lane)
                ));
            }
            FuzzOp::Mutex(v) => {
                let mut g = mutex.lock().unwrap();
                *g += v;
                log.record(format_args!("t{who}:m={}", *g));
            }
        }
    }
}

/// A generated program: per-thread straight-line op lists (thread 0 is the
/// model's main thread).
#[derive(Clone, Debug)]
struct FuzzProgram {
    threads: Vec<Vec<(FuzzOp, loom::sync::atomic::Ordering)>>,
}

impl FuzzProgram {
    fn generate(seed: u64) -> FuzzProgram {
        // Seed 0 is a xorshift fixed point; offset by a constant.
        let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);

        let threads = (0..2 + rng.below(2) as usize)
            .map(|_| {
                (0..2 + rng.below(3) as usize)
                    .map(|_| {
                        let op = FuzzOp::generate(&mut rng);
                        let ord = op.ordering(&mut rng);
                        (op, ord)
                    })
                    .collect()
            })
            .collect();

        FuzzProgram { threads }
    }

    fn run(&self, log: &Log) {
        let cells: Vec<_> = (0..2).map(|_| Arc::new(AtomicUsize::new(0))).collect();
        let wide = Arc::new(AtomicU128::new(0));
        let mutex = Arc::new(loom::sync::Mutex::new(0usize));

        let handles: Vec<_> = self
            .threads
            .iter()
            .enumerate()
            .skip(1)
            .map(|(who, ops)| {
                let ops = ops.clone();
                let (cells, wide, mutex, log) =
                    (cells.clone(), wide.clone(), mutex.clone(), log.clone());
                thread::spawn(move || {
                    for (op, ord) in ops {
                        op.run(ord, who, &cells, &wide, &mutex, &log);
                    }
                })
            })
            .collect();

        for &(op, ord) in &self.threads[0] {
            op.run(ord, 0, &cells, &wide, &mutex, &log);
        }

        for h in handles {
            h.join().unwrap();
        }

        log.record(format_args!(
            "final:{},{},{:x},{}",
            cells[0].load(Relaxed),
            cells[1].load(Relaxed),
            wide.load(Relaxed),
            *mutex.lock().unwrap(),
        ));
    }
}

/// The property, over programs nobody designed: the pruned walk reaches
/// exactly the behaviors of the full walk. `LOOM_FUZZ_PROGRAMS` scales the
/// corpus for soak runs.
#[test]
fn fuzz_pruned_walk_matches_full_walk() {
    let programs: u64 = std::env::var("LOOM_FUZZ_PROGRAMS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(64);

    // Localization overrides: pin one seed and/or one configuration.
    let only_seed: Option<u64> = std::env::var("LOOM_FUZZ_SEED").ok().and_then(|v| v.parse().ok());
    let force_workers: Option<usize> = std::env::var("LOOM_FUZZ_WORKERS")
        .ok()
        .and_then(|v| v.parse().ok());
    let force_bound: Option<usize> = std::env::var("LOOM_FUZZ_BOUND")
        .ok()
        .and_then(|v| v.parse().ok());

    for seed in only_seed.map(|s| s..s + 1).unwrap_or(0..programs) {
        let program = Arc::new(FuzzProgram::generate(seed));

        // Rotate through bound × workers configurations so the corpus covers
        // the matrix without exploring every program four times.
        let bound = force_bound
            .map(|b| if b == 0 { None } else { Some(b) })
            .unwrap_or([None, Some(2)][(seed % 2) as usize]);
        let workers = force_workers.unwrap_or([1, 4][(seed / 2 % 2) as usize]);

        let run = |sleep: bool| {
            let program = program.clone();
            explore_with(
                workers,
                bound,
                move |log| program.run(log),
                true,
                sleep,
            )
        };

        let (full_seen, full_n) = run(false);
        let (slept_seen, slept_n) = run(true);

        assert_eq!(
            slept_seen, full_seen,
            "seed={seed} bound={bound:?} workers={workers}: pruned walk changed \
             the behavior set ({slept_n} vs {full_n} executions)\nprogram: {:#?}",
            program,
        );
        // Tolerance: a scout's races mark exploring ancestors (its own
        // branches are non-exploring), which can open an alternative the
        // full walk would not have — a few extra executions on tiny trees,
        // never a coverage change. The bound trips on runaway overshoot.
        assert!(
            slept_n <= full_n + full_n / 8 + 8,
            "seed={seed} bound={bound:?} workers={workers}: pruned walk took \
             {slept_n} executions, full walk {full_n}"
        );
    }
}

// ---------------------------------------------------------------------------
// Failures still surface
//
// A reduction that silently swallowed an assertion failure would pass every
// coverage test above, because the fingerprint is only recorded on the
// executions that complete.
// ---------------------------------------------------------------------------

fn racy_model_panics(threads: usize) -> bool {
    let mut builder = loom::model::Builder::new();
    builder.threads = threads;
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
    for &threads in &[1, 2, 4, 8] {
        assert!(
            racy_model_panics(threads),
            "threads={threads}: a model that fails on some interleaving was \
             reported clean"
        );
    }
}
