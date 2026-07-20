//! What a materialized cell costs, against a constructed one.
//!
//! A constructed cell caches its registration inline and reads it directly on
//! every operation. A materialized cell has nowhere to cache one — its whole
//! representation is an identity word — so every operation resolves through
//! `Execution::deferred_atomics`. This measures that.
//!
//! Each pair of rigs is identical but for the cell type, and the execution
//! counts are asserted equal, so the difference is per-operation cost and not
//! a differently sized state space.
//!
//! `cargo run --release --example materialized_cost`

use loom::model::Builder;
use loom::sync::atomic::materialized::AtomicU64 as Thin;
use loom::sync::atomic::AtomicU64 as Fat;
use loom::sync::atomic::Ordering::Relaxed;
use loom::sync::Arc;
use loom::thread;

use std::time::{Duration, Instant};

/// A zeroed buffer reinterpreted as cells.
struct Region(Vec<u64>);

impl Region {
    fn new(cells: usize) -> Region {
        Region(vec![0u64; cells])
    }

    fn cell(&self, i: usize) -> &Thin {
        assert!(i < self.0.len());
        // SAFETY: `Thin` is asserted to match `u64`'s size and alignment and
        // the all-zeroes pattern is a valid unregistered cell; the buffer is
        // zeroed, `u64`-aligned, and the index is bounds-checked above.
        unsafe { &*(self.0.as_ptr().add(i) as *const Thin) }
    }
}

/// Best of `TRIALS` — single-shot timings here are worth little, and the
/// minimum is the least noise-contaminated estimator of the underlying cost.
const TRIALS: usize = 3;

fn run(
    label: &str,
    branches: usize,
    f: impl Fn() + Sync + Send + Clone + 'static,
) -> (usize, Duration) {
    let mut best = Duration::MAX;
    let mut executions = 0;

    for _ in 0..TRIALS {
        let mut builder = Builder::new();
        builder.max_branches = branches;

        let start = Instant::now();
        let stats = builder.check(f.clone());
        let elapsed = start.elapsed();

        best = best.min(elapsed);
        executions = stats.executions;
    }

    assert!(
        best >= Duration::from_millis(50),
        "{label}: {best:?} is too short to measure — enlarge the rig"
    );

    println!(
        "  {label:<14} {executions:>9} execs  {:>9.3}s  {:>10.2} us/exec",
        best.as_secs_f64(),
        best.as_secs_f64() * 1e6 / executions as f64,
    );

    (executions, best)
}

fn compare(title: &str, fat: (usize, Duration), thin: (usize, Duration)) {
    assert_eq!(
        fat.0, thin.0,
        "{title}: state spaces differ ({} vs {}), so the times are not comparable",
        fat.0, thin.0
    );

    let ratio = thin.1.as_secs_f64() / fat.1.as_secs_f64();
    println!("  => materialized is {ratio:.2}x constructed\n");
}

/// One thread, many operations: the closest this gets to isolating the
/// per-operation cost. (The load-value branches still make it explore, so it is
/// not a single execution — it is one thread's ops repeated across many.)
fn sequential(ops: usize) {
    println!("sequential, {ops} ops per execution");

    let fat = run("constructed", ops * 8 + 1024, move || {
        let cell = Fat::new(0);
        for _ in 0..ops {
            cell.store(cell.load(Relaxed) + 1, Relaxed);
        }
    });

    let thin = run("materialized", ops * 8 + 1024, move || {
        let region = Region::new(1);
        let cell = region.cell(0);
        for _ in 0..ops {
            cell.store(cell.load(Relaxed) + 1, Relaxed);
        }
    });

    compare("sequential", fat, thin);
}

/// Two threads contending on one cell: per-operation cost inside a real
/// exploration, where scheduling and coherence work dilute it.
fn contended(ops: usize) {
    println!("contended, 2 threads x {ops} rmw");

    let fat = run("constructed", 100_000, move || {
        let cell = Arc::new(Fat::new(0));
        let peer = cell.clone();

        let t = thread::spawn(move || {
            for _ in 0..ops {
                peer.fetch_add(1, Relaxed);
            }
        });

        for _ in 0..ops {
            cell.fetch_add(1, Relaxed);
        }

        t.join().unwrap();
    });

    let thin = run("materialized", 100_000, move || {
        let region = Arc::new(Region::new(1));
        let peer = region.clone();

        let t = thread::spawn(move || {
            for _ in 0..ops {
                peer.cell(0).fetch_add(1, Relaxed);
            }
        });

        for _ in 0..ops {
            region.cell(0).fetch_add(1, Relaxed);
        }

        t.join().unwrap();
    });

    compare("contended", fat, thin);
}

/// Many distinct cells, each touched repeatedly — the pool-like shape, and the
/// one where the registration table is largest and its lookups least
/// predictable.
///
/// Note this compares whole workloads, not just the lookup: the constructed
/// side also pays to build `cells` cells and to carry them at ~40 bytes each
/// against 8. That is a real difference between the representations, but it is
/// not the per-operation cost `sequential` isolates.
fn many_cells(cells: usize, rounds: usize) {
    println!("many cells, {cells} cells x {rounds} rounds");

    let fat = run("constructed", cells * rounds * 8 + 1024, move || {
        let all: Vec<Fat> = (0..cells).map(|_| Fat::new(0)).collect();
        for _ in 0..rounds {
            for c in &all {
                c.store(1, Relaxed);
            }
        }
    });

    let thin = run("materialized", cells * rounds * 8 + 1024, move || {
        let region = Region::new(cells);
        for _ in 0..rounds {
            for i in 0..cells {
                region.cell(i).store(1, Relaxed);
            }
        }
    });

    compare("many cells", fat, thin);
}

/// The shape that stresses object-store reincarnation: several cells whose
/// *first touch* order differs per schedule, so their store indices churn
/// across executions and collide with the other object variants (`Arc`,
/// threads) that a run also creates.
///
/// This is the rig for the truncate-vs-overwrite question in
/// `Store::insert_with` — a variant mismatch is only reachable when numbering
/// moves, and numbering only moves for lazily registered cells.
fn churn(cells: usize, rounds: usize) {
    println!("churn, 2 threads x {rounds} passes over {cells} cells, opposite orders");

    let thin = run("materialized", 200_000, move || {
        let region = Arc::new(Region::new(cells));
        let peer = region.clone();

        let t = thread::spawn(move || {
            for _ in 0..rounds {
                for i in 0..cells {
                    peer.cell(i).fetch_add(1, Relaxed);
                }
            }
        });

        for _ in 0..rounds {
            for i in (0..cells).rev() {
                region.cell(i).fetch_add(1, Relaxed);
            }
        }

        t.join().unwrap();
    });

    // Measured both ways at 3 cells x 4 passes (28048 executions): truncate
    // 0.138s, overwrite 0.135s — inside noise. The cliff the truncate was
    // changed to avoid is real in principle and unreachable in practice here,
    // because lazy numbering only permutes *atomic* indices among themselves
    // and `recycle` absorbs a same-variant carcass. Kept as the rig that would
    // catch it if a future object graph moves the variant boundary.
    println!(
        "  (no constructed counterpart: those are numbered at construction, \
         so their indices cannot churn)\n  best {:.3}s over {} execs\n",
        thin.1.as_secs_f64(),
        thin.0
    );
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let run_all = args.is_empty();
    let want = |name: &str| run_all || args.iter().any(|a| a == name);

    if want("--sequential") {
        sequential(50_000);
    }
    if want("--contended") {
        contended(9);
    }
    if want("--many-cells") {
        many_cells(4_000, 250);
    }
    if want("--churn") {
        churn(3, 4);
    }
}
