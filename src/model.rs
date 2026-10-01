//! Model concurrent programs.

use crate::rt::{self, Execution, Failure, Scheduler};
use std::panic::{self, AssertUnwindSafe};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use tracing::{info, subscriber};
use tracing_subscriber::{fmt, EnvFilter};

const DEFAULT_MAX_THREADS: usize = 5;
const DEFAULT_MAX_BRANCHES: usize = 1_000;

/// Stack size of a model thread, in bytes: a real thread's, so that a deep
/// call chain or a panic hook printing a backtrace fits where it would
/// outside the model. The stacks are committed when built and reused across
/// executions, so the cost is address space and commit charge per pooled
/// thread, not per execution; pages the thread never touches stay
/// non-resident.
const DEFAULT_STACK_SIZE: usize = 1 << 20;

/// How often a worker offers part of its remaining subtree to idle peers.
/// Each check is an uncontended mutex acquire, so this only has to be small
/// enough that the initial fan-out is not the bottleneck.
const DONATE_INTERVAL: usize = 16;

/// How deep into the tree a branch may be frozen, and so shared.
///
/// This costs no exploration at any setting — freezing changes who walks an
/// alternative, never which alternatives exist. The execution count on
/// `fused_same_word_claim_races_cancel` is 2,266,951 at every value from 4 to
/// 10,000. All it decides is how far down a task can still be subdivided.
///
/// That makes the failure mode one-sided. Once a task's floor passes this
/// depth it can never be split again, so a shallow setting strands whole
/// subtrees on one worker: 58s at 4 and 8, 43s at 16, 33s at 32, against 12s
/// from 96 up. Above 96 the curve is flat to 10,000, inside a run-to-run
/// spread of about ±3s. 256 sits in that flat region an order of magnitude
/// clear of the cliff, and stays bounded so a pathological model cannot grow
/// an unbounded frozen prefix.
const SPLIT_DEPTH: usize = 256;

/// How long a model gets to finish as a plain reduced serial walk before the
/// run restarts under a worker pool. Short enough that restarting is noise
/// against a model that needs sharding at all, long enough that ordinary
/// models never reach it.
const PROBE: Duration = Duration::from_secs(1);


/// What a completed check explored.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct Stats {
    /// Number of distinct executions run.
    pub executions: usize,

    /// Number of OS threads that explored the state space.
    pub threads: usize,

    /// Executions cut short because a sleep set proved every continuation
    /// redundant — their behaviors are covered by a sibling subtree. Counted
    /// inside `executions`.
    pub pruned: usize,

    /// Executions that sit under at least one schedule alternative opened
    /// only by the preemption bound's conservative backtrack points (Coons,
    /// Musuvathi & McKinley, OOPSLA'13, Algorithm 3 Line 9). Counted inside
    /// `executions`; only collected when [`Builder::stats`] is set, else 0.
    ///
    /// This is the measured ceiling on what a bound-aware optimal DPOR
    /// (e.g. TACAS'23 slack bounding) could remove: these executions exist
    /// only to compensate for the bound, though some re-derive behaviors an
    /// optimal search would still have to reach another way.
    pub conservative: usize,
}

/// Configure a model
#[derive(Debug)]
#[non_exhaustive] // Support adding more fields in the future
pub struct Builder {
    /// Max number of threads to check as part of the execution.
    ///
    /// This should be set as low as possible and must be less than
    /// [`MAX_THREADS`](crate::MAX_THREADS).
    pub max_threads: usize,

    /// Maximum number of thread switches per permutation.
    ///
    /// Defaults to `LOOM_MAX_BRANCHES` environment variable.
    pub max_branches: usize,

    /// Maximum number of permutations to explore.
    ///
    /// Defaults to `LOOM_MAX_PERMUTATIONS` environment variable.
    pub max_permutations: Option<usize>,

    /// Maximum amount of time to spend on checking. A run that reaches it
    /// before exploring the whole state space fails: the unexplored part may
    /// hold the bug. Every exploring worker stops within one execution.
    ///
    /// Defaults to `LOOM_MAX_DURATION` environment variable, in seconds.
    pub max_duration: Option<Duration>,

    /// Maximum number of thread preemptions to explore
    ///
    /// Without a bound (`None`) the search is optimal dynamic partial-order
    /// reduction: source sets and wakeup trees (Abdulla et al., JACM 2017),
    /// which reaches every equivalence class of executions, most of them
    /// once. Under a bound it is bounded partial-order reduction (Coons et
    /// al., OOPSLA'13), complete for the executions within the bound.
    ///
    /// A preemption is a switch away from a thread that could have continued,
    /// including one that called [`thread::yield_now`](crate::thread::yield_now).
    /// A switch after a thread blocks or finishes is free, and so is the
    /// switch after a [`hint::spin_loop`](crate::hint::spin_loop) — but once an
    /// execution has spent the whole bound, the scheduler explores no
    /// alternative at a spin either: spins recur on every pass of a wait loop,
    /// and left open at a spent bound they would make such a loop's state
    /// space inexhaustible. So a bounded search may miss an order reachable
    /// only by switching to a different thread at a spin after the bound is
    /// spent.
    ///
    /// Defaults to `LOOM_MAX_PREEMPTIONS` environment variable.
    pub preemption_bound: Option<usize>,

    /// When doing an exhaustive check, uses the file to store and load the
    /// check progress
    ///
    /// Defaults to `LOOM_CHECKPOINT_FILE` environment variable.
    pub checkpoint_file: Option<PathBuf>,

    /// How often to write the checkpoint file
    ///
    /// Defaults to `LOOM_CHECKPOINT_INTERVAL` environment variable.
    pub checkpoint_interval: usize,

    /// When `true` loom won't start state exploration until `explore_state` is
    /// called.
    pub expect_explicit_explore: bool,

    /// When `true`, locations are captured on each loom operation.
    ///
    /// Note that is is **very** expensive. It is recommended to first isolate a
    /// failing iteration using `LOOM_CHECKPOINT_FILE`, then enable location
    /// tracking.
    ///
    /// Defaults to `LOOM_LOCATION` environment variable.
    pub location: bool,

    /// Log execution output to stdout.
    ///
    /// Defaults to existence of `LOOM_LOG` environment variable.
    pub log: bool,

    /// Stack size, in bytes, of each model thread — the model's main thread
    /// and every thread spawned without an explicit
    /// [`thread::Builder::stack_size`](crate::thread::Builder::stack_size).
    ///
    /// Defaults to the `LOOM_STACK_SIZE` environment variable, else 1 MiB.
    pub stack_size: usize,

    /// Number of OS threads exploring the state space concurrently.
    ///
    /// The search tree is split at its shallowest unexplored branches and
    /// handed out, so workers explore disjoint subtrees and the union is the
    /// same state space a serial run covers. What changes is ordering: which
    /// worker reaches a failing execution first is not deterministic, so
    /// reproducing one is easier with this set to 1.
    ///
    /// Defaults to the `LOOM_THREADS` environment variable, else the machine's
    /// parallelism. This is a ceiling, not a reservation: workers beyond the
    /// first are drawn from a process-wide budget, so a model that runs
    /// alongside fifteen others in a test harness quietly gets one worker
    /// while the same model run on its own gets the machine. Nothing has to be
    /// configured per test for that to hold.
    pub threads: usize,

    /// How long a model may run as a plain serial walk before it is handed to
    /// a worker pool.
    ///
    /// A pool is overhead a model that finishes in microseconds should not pay,
    /// and that is most of them. Nothing is lost by waiting: the walk's path
    /// crosses over with it, so the pool resumes where the probe stopped. Zero
    /// shards immediately — mainly useful for testing the sharded path on a
    /// model too small to reach the probe on its own.
    ///
    /// Defaults to the `LOOM_PROBE_MS` environment variable, else one second.
    pub probe: Duration,

    /// Draw workers beyond the first from the process-wide budget, so that
    /// concurrently-running models share the machine instead of each claiming
    /// all of it. On by default; that sharing is what makes [`threads`] safe
    /// to default to the whole machine.
    ///
    /// Turning it off makes [`threads`] an exact count. Tests that need a
    /// specific degree of sharding want this; ordinary runs do not.
    ///
    /// [`threads`]: Self::threads
    pub budgeted: bool,

    /// How deep into the tree a branch may be frozen for sharing.
    ///
    /// Deeper means a task can still be subdivided further down, which is what
    /// keeps workers fed on an unbalanced tree. It buys that for free: what
    /// gets explored is identical at every setting, so this trades only load
    /// balance against how much of the tree carries shared bookkeeping. Only
    /// consulted when a run actually shards.
    ///
    /// Defaults to the `LOOM_SPLIT_DEPTH` environment variable, else a value
    /// measured against the nt-sync futex models.
    pub split_depth: usize,

    /// Reincarnate model objects in place across executions instead of
    /// freeing and reallocating them (see `rt::object::Store::begin_epoch`).
    ///
    /// On by default; consecutive executions rebuild an almost-identical
    /// object graph, and reuse removes that rebuild from the allocator
    /// entirely. Off forces every execution to construct its objects from
    /// scratch — the reference behavior the reuse path is tested against.
    pub reuse_objects: bool,

    /// Prune executions that only reorder independent operations of a
    /// subtree already covered at some branch (sleep sets, see `rt::sleep`).
    /// Coverage is unchanged: every observable behavior of the full walk is
    /// still reached.
    ///
    /// On by default; `LOOM_SLEEP_SETS=0` turns it off, restoring the
    /// exploration the pruned walk is tested against. Without sleep sets a
    /// bounded walk's sharded execution counts are identical at every worker
    /// count; an unbounded walk's are not, since a branch frozen for sharding
    /// takes its reversals as source sets rather than into a wakeup tree
    /// (`rt::dpor`), and which branches freeze depends on timing.
    ///
    /// Inert when `preemption_bound` is set. Deference is only sound when
    /// the covered-elsewhere subtree is fully explored, and a bound truncates
    /// it: a covering linearization can cost up to two more preemptions than
    /// the execution it covers (Coons, Musuvathi & McKinley, OOPSLA'13
    /// study this interaction). Differential fuzzing confirms behavior loss
    /// within seconds if deference is allowed under a bound, and measures the
    /// sound residue (BPOR's classical rule) at a few percent — not worth an
    /// unproven soundness posture.
    pub sleep_sets: bool,

    /// Collect per-execution attribution into [`Stats`] (currently: how many
    /// executions exist only because of the preemption bound's conservative
    /// backtrack points, see [`Stats::conservative`]). Off by default — it
    /// walks the path's schedule branches once per execution.
    ///
    /// Defaults to the presence of the `LOOM_STATS` environment variable,
    /// which also prints the stats line at the end of the run.
    pub stats: bool,

    /// Report materialized memory still committed when an execution ends: a
    /// `publish` with no matching `unpublish` never returns its commit charge.
    ///
    /// Off by default, because memory legitimately outliving one execution — a
    /// region held by a `static` and committed afresh in each — is committed
    /// at every execution's end by design.
    pub check_committed_leaks: bool,

    /// Snapshot each execution every this many branches, and resume the next
    /// execution from the deepest snapshot at or before the branch where it
    /// leaves the previous one's path, instead of replaying that prefix. The
    /// set of executions explored is the same either way.
    ///
    /// Takes effect only in a binary that installs
    /// [`alloc::Model`](crate::alloc::Model) as its global allocator, whose
    /// contract the model must keep.
    ///
    /// Defaults to the `LOOM_SNAPSHOT` environment variable, else 64;
    /// `LOOM_SNAPSHOT=0` replays every execution from the start.
    pub snapshot: Option<usize>,
}

impl Builder {
    /// Create a new `Builder` instance with default values.
    pub fn new() -> Builder {
        use std::env;

        let checkpoint_interval = env::var("LOOM_CHECKPOINT_INTERVAL")
            .map(|v| {
                v.parse()
                    .expect("invalid value for `LOOM_CHECKPOINT_INTERVAL`")
            })
            .unwrap_or(20_000);

        let max_branches = env::var("LOOM_MAX_BRANCHES")
            .map(|v| v.parse().expect("invalid value for `LOOM_MAX_BRANCHES`"))
            .unwrap_or(DEFAULT_MAX_BRANCHES);

        let location = env::var("LOOM_LOCATION").is_ok();

        let log = env::var("LOOM_LOG").is_ok();

        let max_duration = env::var("LOOM_MAX_DURATION")
            .map(|v| {
                let secs = v.parse().expect("invalid value for `LOOM_MAX_DURATION`");
                Duration::from_secs(secs)
            })
            .ok();

        let max_permutations = env::var("LOOM_MAX_PERMUTATIONS")
            .map(|v| {
                v.parse()
                    .expect("invalid value for `LOOM_MAX_PERMUTATIONS`")
            })
            .ok();

        let preemption_bound = env::var("LOOM_MAX_PREEMPTIONS")
            .map(|v| v.parse().expect("invalid value for `LOOM_MAX_PREEMPTIONS`"))
            .ok();

        let checkpoint_file = env::var("LOOM_CHECKPOINT_FILE")
            .map(|v| v.parse().expect("invalid value for `LOOM_CHECKPOINT_FILE`"))
            .ok();

        let threads = env::var("LOOM_THREADS")
            .map(|v| v.parse().expect("invalid value for `LOOM_THREADS`"))
            .unwrap_or_else(|_| cpus());

        Builder {
            max_threads: DEFAULT_MAX_THREADS,
            max_branches,
            max_duration,
            max_permutations,
            preemption_bound,
            checkpoint_file,
            checkpoint_interval,
            expect_explicit_explore: false,
            location,
            log,
            stack_size: env::var("LOOM_STACK_SIZE")
                .map(|v| v.parse().expect("invalid value for `LOOM_STACK_SIZE`"))
                .unwrap_or(DEFAULT_STACK_SIZE),
            threads,
            budgeted: true,
            split_depth: env::var("LOOM_SPLIT_DEPTH")
                .map(|v| v.parse().expect("invalid value for `LOOM_SPLIT_DEPTH`"))
                .unwrap_or(SPLIT_DEPTH),
            probe: env::var("LOOM_PROBE_MS")
                .map(|v| {
                    Duration::from_millis(v.parse().expect("invalid value for `LOOM_PROBE_MS`"))
                })
                .unwrap_or(PROBE),
            reuse_objects: env::var("LOOM_REUSE_OBJECTS")
                .map(|v| v != "0")
                .unwrap_or(true),
            sleep_sets: env::var("LOOM_SLEEP_SETS")
                .map(|v| v != "0")
                .unwrap_or(true),
            stats: env::var_os("LOOM_STATS").is_some(),
            check_committed_leaks: false,
            snapshot: Some(
                env::var("LOOM_SNAPSHOT")
                    .map(|v| v.parse().expect("invalid value for `LOOM_SNAPSHOT`"))
                    .unwrap_or(64),
            )
            .filter(|&spacing| spacing > 0),
        }
    }

    /// Number of OS threads to explore with, resolved against the settings
    /// that require a single ordered walk.
    ///
    /// Checkpointing serializes one path to disk and resumes from it, and
    /// logging narrates one execution after another; both describe a single
    /// walk of the tree and would be nonsense interleaved.
    fn workers(&self) -> usize {
        if self.checkpoint_file.is_some() || self.log {
            return 1;
        }

        self.threads.max(1)
    }

    /// Set the checkpoint file.
    pub fn checkpoint_file(&mut self, file: &str) -> &mut Self {
        self.checkpoint_file = Some(file.into());
        self
    }

    /// Check the provided model.
    ///
    /// # Panics
    ///
    /// When an execution fails, once, after exploration has stopped: a
    /// deadlock panics with a report of what each thread waits on, and a
    /// panic in a model thread is re-raised with its own payload. A failed
    /// execution is not unwound further: once a model thread panics, the
    /// first tracked operation a destructor attempts during its unwinding
    /// stops that thread for good and fails the model with the panic's
    /// message — even if the panic would have been caught inside the model.
    ///
    /// Also panics when `max_duration` passes before the state space is
    /// fully explored.
    pub fn check<F>(&self, f: F) -> Stats
    where
        F: Fn() + Sync + Send + 'static,
    {
        Scheduler::install_panic_hook();

        let f = Arc::new(f);

        // One worker is always ours; the rest are whatever the machine has
        // spare right now, held until this run finishes.
        let want = self.workers() - 1;
        let grant = if self.budgeted {
            Grant::claim(want)
        } else {
            Grant(0)
        };
        let workers = 1 + if self.budgeted { grant.0 } else { want };

        let start = Instant::now();

        // Spinning up a worker pool for a model that finishes in microseconds
        // is all overhead, and that is the large majority of them. So every run
        // starts as a plain serial walk and only hands over once it is still
        // going after `PROBE`.
        //
        // The handover carries the walk's own path across, so the pool picks
        // the tree up exactly where the probe left it. Nothing is explored
        // twice and nothing is thrown away — which is why the probe can be
        // generous without a large model paying for it.
        let probe = if workers == 1 {
            None
        } else {
            Some(start + self.probe)
        };

        let (stats, timed_out) = match self.on_driver(|| self.check_serial(&f, start, probe)) {
            Walk::Done(stats) => (stats, false),
            Walk::TimedOut(stats) => (stats, true),
            Walk::Failed(failure) => failure.raise(),
            Walk::Handover(probed, seed) => {
                let (stats, timed_out) =
                    self.check_parallel(&f, workers, start, seed, probed.executions);

                let stats = Stats {
                    pruned: stats.pruned + probed.pruned,
                    conservative: stats.conservative + probed.conservative,
                    ..stats
                };

                (stats, timed_out)
            }
        };

        drop(grant);

        if timed_out {
            panic!(
                "loom model exceeded max_duration ({:?}) after {} executions, \
                 before exploring its whole state space",
                self.max_duration.unwrap_or_default(),
                stats.executions,
            );
        }

        // `LOOM_LOG` forces a serial walk, so it cannot report what a sharded
        // run explored. This can, and the execution count is the number that
        // says whether a reduction is doing anything.
        if std::env::var_os("LOOM_STATS").is_some() {
            rt::snapshot::stats::report(stats.executions as u64);
            eprintln!(
                "loom: {} executions ({} pruned, {} bound-conservative), {} worker(s), bound={:?}, {:.2}s",
                stats.executions,
                stats.pruned,
                stats.conservative,
                stats.threads,
                self.preemption_bound,
                start.elapsed().as_secs_f64(),
            );
        }

        stats
    }

    /// Run `walk` on a thread of its own, named for the caller.
    ///
    /// A failed execution can leave its driver's OS thread with a raised
    /// panic count — the unwind it abandoned — so every walk that runs
    /// executions owns the thread it does so on, and that thread ends with
    /// the walk.
    fn on_driver<T: Send>(&self, walk: impl FnOnce() -> T + Send) -> T {
        let dispatch = tracing::dispatcher::get_default(|d| d.clone());

        let driver = std::thread::scope(|scope| {
            driver_thread()
                .spawn_scoped(scope, || {
                    tracing::dispatcher::with_default(&dispatch, walk)
                })
                .expect("failed to spawn the loom driver thread")
                .join()
        });

        match driver {
            Ok(value) => value,
            Err(payload) => panic::resume_unwind(payload),
        }
    }

    /// The per-thread state executions run on: snapshotting when `snapshot`
    /// is set and the binary installs [`crate::alloc::Model`], replaying from
    /// the start otherwise.
    fn new_engine(&self) -> Engine {
        // A tracing subscriber's spans count references outside the world,
        // which a restore would replay.
        let spacing = self.snapshot.filter(|_| {
            rt::snapshot::available() && !self.log
        });

        let world = spacing.and_then(|spacing| {
            let world =
                rt::snapshot::World::new(|| self.new_execution(), self.max_threads, self.stack_size)?;
            Some((world, spacing))
        });

        match world {
            Some((world, spacing)) => Engine::Snapshot {
                world,
                snapshots: rt::snapshot::Snapshots::new(spacing),
            },
            None => Engine::Replay {
                execution: self.new_execution(),
                scheduler: Scheduler::new(self.max_threads, self.stack_size),
            },
        }
    }

    fn new_execution(&self) -> Execution {
        let mut execution = Execution::new(
            self.max_threads,
            self.max_branches,
            self.preemption_bound,
            !self.expect_explicit_explore,
        );

        execution.log = self.log;
        execution.location = self.location;
        execution.reuse_objects = self.reuse_objects;
        execution.check_committed_leaks = self.check_committed_leaks;
        execution.sleep_sets = self.sleep_sets && self.preemption_bound.is_none();
        execution
    }

    /// The configured ceiling the run has hit, if any. Checked once per
    /// execution; the clock is read only when `max_duration` is set, and a
    /// read is noise against an execution.
    fn limit_reached(&self, executions: usize, start: Instant) -> Option<Limit> {
        if let Some(max) = self.max_permutations {
            if executions >= max {
                return Some(Limit::Permutations);
            }
        }

        if let Some(max) = self.max_duration {
            if start.elapsed() >= max {
                return Some(Limit::Duration);
            }
        }

        None
    }

    /// Walk the whole tree on this thread, until it is exhausted, a ceiling
    /// is hit, or an execution fails. If `probe` passes first, hands the rest
    /// of the tree back: the model is large enough to be worth a worker pool.
    fn check_serial<F>(&self, f: &Arc<F>, start: Instant, probe: Option<Instant>) -> Walk
    where
        F: Fn() + Sync + Send + 'static,
    {
        let mut i = 1;
        let mut conservative = 0;
        let mut _span = tracing::info_span!("iter", message = i).entered();

        let mut engine = self.new_engine();

        if let Some(ref path) = self.checkpoint_file {
            if path.exists() {
                let saved = checkpoint::load_execution_path(path);

                // The saved prefix was explored, and its backtrack points
                // placed, under the bound it was written with; resumed under
                // another, the walk would be neither bound's.
                let bound = saved.preemption_bound().map(usize::from);
                assert_eq!(
                    bound, self.preemption_bound,
                    "checkpoint {} was written under preemption_bound {:?}, \
                     but this Builder's is {:?}",
                    path.display(),
                    bound,
                    self.preemption_bound,
                );

                let mut saved = saved;
                saved.set_max_branches(self.max_branches);
                engine.take_path(saved);
            }
        }

        loop {
            if i % self.checkpoint_interval == 0 {
                info!(parent: None, "");
                info!(
                    parent: None,
                    " ================== Iteration {} ==================", i
                );
                info!(parent: None, "");

                if let Some(ref path) = self.checkpoint_file {
                    checkpoint::store_execution_path(&engine.execution().path, path);
                }
            }

            // Only a run that is deciding whether to shard reads the clock
            // here, and for it one read per execution is noise against the
            // execution itself. Checking on the donation cadence instead would
            // overshoot the probe by up to a full interval, and those
            // executions are thrown away along with the tree they built.
            if let Some(deadline) = probe {
                if Instant::now() >= deadline {
                    let probed = self.stats(i - 1, 1, engine.execution().pruned, conservative);
                    return Walk::Handover(probed, engine.into_path());
                }
            }

            if let Err(failure) = engine.run(f) {
                // Its user values must not be destroyed (`Scheduler::run`).
                engine.leak();
                return Walk::Failed(failure);
            }

            engine.check_for_leaks();

            if self.stats && engine.execution().path.conservative_attributed() {
                conservative += 1;
            }

            // Create the next iteration's `tracing` span before trying to step to the next
            // execution, as the `Execution` will capture the current span when
            // it's reset.
            _span = tracing::info_span!(parent: None, "iter", message = i + 1).entered();
            if !engine.step() {
                info!(parent: None, "Completed in {} iterations", i);
                return Walk::Done(self.stats(i, 1, engine.execution().pruned, conservative));
            }

            match self.limit_reached(i, start) {
                None => {}
                Some(Limit::Permutations) => {
                    return Walk::Done(self.stats(i, 1, engine.execution().pruned, conservative))
                }
                Some(Limit::Duration) => {
                    return Walk::TimedOut(self.stats(
                        i,
                        1,
                        engine.execution().pruned,
                        conservative,
                    ))
                }
            }

            i += 1;
        }
    }

    /// Explore the tree with `workers` threads over a shared pool of subtrees.
    ///
    /// `seed` is the tree still to walk — whole, or whatever the probe left.
    /// One worker takes it; whenever peers go idle, a worker freezes its
    /// shallowest branches and posts what nobody has claimed. Workers never
    /// share an `Execution`, a `Scheduler`, or a coroutine — only the task pool
    /// and the frozen prefixes — so each explores exactly as a serial walk
    /// does, one path at a time.
    ///
    /// `probed` executions already ran before the handover; they count
    /// against `max_permutations` and are included in the returned stats.
    ///
    /// Returns the stats and whether the run stopped at `max_duration`.
    fn check_parallel<F>(
        &self,
        f: &Arc<F>,
        workers: usize,
        start: Instant,
        mut seed: rt::Path,
        probed: usize,
    ) -> (Stats, bool)
    where
        F: Fn() + Sync + Send + 'static,
    {
        seed.set_split_depth(self.split_depth);

        let shared = Shared::new(seed, probed);

        // Worker threads do not inherit the caller's `tracing` subscriber.
        let dispatch = tracing::dispatcher::get_default(|d| d.clone());

        std::thread::scope(|scope| {
            for _ in 0..workers {
                let shared = &shared;
                let f = f.clone();
                let dispatch = dispatch.clone();

                driver_thread()
                    .spawn_scoped(scope, move || {
                        tracing::dispatcher::with_default(&dispatch, || {
                            // The `Scheduler` is built inside the guarded
                            // scope so that a driver-side panic (a leak
                            // report) unwinds through its `Drop` with the
                            // thread flagged as panicking.
                            let run = panic::catch_unwind(AssertUnwindSafe(|| {
                                self.worker(shared, &f, workers, start);
                            }));

                            if let Err(payload) = run {
                                shared.fail(Failure::Panic(payload));
                            }
                        });
                    })
                    .expect("failed to spawn a loom worker thread");
            }
        });

        let executions = shared.executions.load(Relaxed);
        let pruned = shared.pruned.load(Relaxed);
        let conservative = shared.conservative.load(Relaxed);
        let timed_out = shared.timed_out.load(Relaxed);

        if let Some(failure) = shared.into_failure() {
            failure.raise();
        }

        info!(parent: None, "Completed in {} iterations", executions);
        (self.stats(executions, workers, pruned, conservative), timed_out)
    }

    fn worker<F>(&self, shared: &Shared, f: &Arc<F>, workers: usize, start: Instant)
    where
        F: Fn() + Sync + Send + 'static,
    {
        // One `Execution` for the worker's whole life: taking a new subtree
        // swaps the path in and epoch-resets the rest, so the object store's
        // reincarnation carcasses carry across tasks, not just iterations.
        let mut engine = self.new_engine();

        let mut conservative = 0;

        while let Some(path) = shared.take(workers) {
            engine.take_path(path);

            loop {
                if let Err(failure) = engine.run(f) {
                    // Its user values must not be destroyed (`Scheduler::run`),
                    // and this thread runs no further execution.
                    engine.leak();
                    shared.fail(failure);
                    return;
                }

                engine.check_for_leaks();

                if self.stats && engine.execution().path.conservative_attributed() {
                    conservative += 1;
                }

                let done = shared.executions.fetch_add(1, Relaxed) + 1;

                if let Some(limit) = self.limit_reached(done, start) {
                    if limit == Limit::Duration {
                        shared.timed_out.store(true, Relaxed);
                    }

                    shared.stop();
                    shared.pruned.fetch_add(engine.execution().pruned, Relaxed);
                    shared.conservative.fetch_add(conservative, Relaxed);
                    return;
                }

                if !engine.step() {
                    break;
                }

                if done % DONATE_INTERVAL == 0 {
                    shared.donate(&mut engine.execution().path);
                }

                // A peer failed or hit a ceiling: the verdict is in.
                if shared.stopped() {
                    shared.pruned.fetch_add(engine.execution().pruned, Relaxed);
                    shared.conservative.fetch_add(conservative, Relaxed);
                    return;
                }
            }
        }

        shared.pruned.fetch_add(engine.execution().pruned, Relaxed);
        shared.conservative.fetch_add(conservative, Relaxed);
    }

    fn stats(&self, executions: usize, threads: usize, pruned: usize, conservative: usize) -> Stats {
        let _ = self;

        Stats {
            executions,
            threads,
            pruned,
            conservative,
        }
    }
}

/// The state a thread runs executions on.
enum Engine {
    /// Every execution replays its path from the start.
    Replay {
        execution: Execution,
        scheduler: Scheduler,
    },

    /// An execution resumes from the deepest snapshot its path allows.
    Snapshot {
        world: rt::snapshot::World,
        snapshots: rt::snapshot::Snapshots,
    },
}

impl Engine {
    fn execution(&mut self) -> &mut Execution {
        match self {
            Engine::Replay { execution, .. } => execution,
            Engine::Snapshot { world, .. } => &mut world.inner().execution,
        }
    }

    /// Start a fresh subtree.
    fn take_path(&mut self, path: rt::Path) {
        match self {
            Engine::Replay { execution, .. } => {
                execution.path = path;
                execution.path.detach();
                execution.reset_iteration();
            }
            Engine::Snapshot { world, snapshots } => {
                world.inner().execution.path = path;
                world.inner().execution.path.detach();
                snapshots.clear();
            }
        }
    }

    /// Hand the path over and drop the rest.
    fn into_path(mut self) -> rt::Path {
        let fresh = rt::Path::new(0, None, false);
        std::mem::replace(&mut self.execution().path, fresh)
    }

    /// Abandon this state after a failed execution.
    fn leak(self) {
        match self {
            Engine::Replay { execution, .. } => std::mem::forget(execution),
            Engine::Snapshot { world, .. } => world.leak(),
        }
    }

    fn check_for_leaks(&mut self) {
        match self {
            Engine::Replay { execution, .. } => execution.check_for_leaks(),
            Engine::Snapshot { world, .. } => {
                let execution: *mut Execution = &mut world.inner().execution;
                // SAFETY: the world outlives the call.
                world.routed(|| unsafe { (*execution).check_for_leaks() });
            }
        }
    }

    /// Step the path; `false` once the tree is exhausted.
    fn step(&mut self) -> bool {
        match self {
            Engine::Replay { execution, .. } => execution.step(),
            Engine::Snapshot { world, .. } => {
                let execution: *mut Execution = &mut world.inner().execution;
                // SAFETY: the world outlives the call. The step log the
                // reversals extend is world state; the path stays outside.
                world.routed(|| unsafe { (*execution).step_path() })
            }
        }
    }

    /// Run the next execution.
    fn run<F>(&mut self, f: &Arc<F>) -> Result<(), Failure>
    where
        F: Fn() + Sync + Send + 'static,
    {
        match self {
            Engine::Replay {
                execution,
                scheduler,
            } => scheduler.run(execution, body(f)),
            Engine::Snapshot { world, snapshots } => run_checked(world, snapshots, f),
        }
    }
}

/// [`run_snapshot`], and under `LOOM_SNAPSHOT_CHECK` a replay of every
/// resumed execution from the start, which must leave the same branch record
/// and the same observations.
fn run_checked<F>(
    world: &mut rt::snapshot::World,
    snapshots: &mut rt::snapshot::Snapshots,
    f: &Arc<F>,
) -> Result<(), Failure>
where
    F: Fn() + Sync + Send + 'static,
{
    if !rt::snapshot::check() {
        return run_snapshot(world, snapshots, f).0;
    }

    let before = world.inner().execution.path.duplicate();
    let (result, resumed) = run_snapshot(world, snapshots, f);
    result?;
    world.inner().execution.path.assert_outside();
    if !resumed {
        return Ok(());
    }

    let restored = world.inner().execution.path.record();
    let restored_steps = world.inner().execution.steps.record();
    let restored_observed = observed(&world.inner().execution);
    let after = std::mem::replace(&mut world.inner().execution.path, before);

    start(world, f, usize::MAX);
    drive(world, snapshots, |_, _| unreachable!("[loom internal bug] snapshot during a check replay"))?;

    world.inner().execution.path.assert_outside();
    let replayed = world.inner().execution.path.record();
    let replayed_steps = world.inner().execution.steps.record();
    let replayed_observed = observed(&world.inner().execution);
    assert!(
        restored == replayed
            && restored_steps == replayed_steps
            && restored_observed == replayed_observed,
        "loom: an execution resumed from a snapshot differs from its replay \
         (path records equal: {}, step logs equal: {}, observations equal: {})",
        restored == replayed,
        restored_steps == replayed_steps,
        restored_observed == replayed_observed,
    );

    world.inner().execution.path = after;
    rt::snapshot::stats::add(&rt::snapshot::stats::CHECKED, 1);
    Ok(())
}

/// The digest of what an execution observed (`thread::Set::observed`).
fn observed(execution: &Execution) -> u64 {
    execution.threads.observed().wrapping_add(execution.path.observed)
}

/// Run the next execution from the deepest usable snapshot, or from the
/// start when there is none, taking snapshots as it goes. Also says whether
/// it resumed from a snapshot.
fn run_snapshot<F>(
    world: &mut rt::snapshot::World,
    snapshots: &mut rt::snapshot::Snapshots,
    f: &Arc<F>,
) -> (Result<(), Failure>, bool)
where
    F: Fn() + Sync + Send + 'static,
{
    use rt::snapshot::stats;

    let spacing = snapshots.spacing();
    let divergence = world.inner().execution.path.divergence();

    let resumed = match divergence {
        None => {
            snapshots.clear();
            false
        }
        Some(divergence) => snapshots.restorable(divergence).is_some(),
    };

    if resumed {
        let pos = snapshots.restore(world);
        stats::add(&stats::RESTORED, 1);

        let next = next_snapshot(&world.inner().execution.path, pos, divergence, spacing);
        world.inner().execution.path.snapshot_at(next);
        rt::snapshot::clear_due();
    } else {
        stats::add(&stats::STARTED, 1);
        let next = next_snapshot(&world.inner().execution.path, 0, divergence, spacing);
        start(world, f, next);
    }

    let result = drive(world, snapshots, |world, snapshots| {
        let bytes = snapshots.take(world);
        stats::add(&stats::TAKEN, 1);
        stats::add(&stats::BYTES, bytes as u64);

        let pos = world.inner().execution.path.pos_now();
        if Some(pos) == divergence {
            stats::add(&stats::ON_TARGET, 1);
        }
        world.inner().execution.path.snapshot_at(pos + spacing);
    });

    (result, resumed)
}

/// Begin an execution of `f` from the start, its first snapshot due at
/// branch `next`.
fn start<F>(world: &mut rt::snapshot::World, f: &Arc<F>, next: usize)
where
    F: Fn() + Sync + Send + 'static,
{
    let inner: *mut rt::snapshot::Inner = world.inner();
    // SAFETY: the world outlives the call, and the scheduler and the
    // execution are disjoint fields.
    world.routed(|| unsafe {
        (*inner).execution.reset_iteration();
        (*inner).execution.path.snapshot_at(next);
        (*inner).scheduler.start(&mut (*inner).execution, body(f));
    });
    rt::snapshot::clear_due();
}

/// Drive the started execution to its end, handing every snapshot request
/// to `snapshot`.
fn drive(
    world: &mut rt::snapshot::World,
    snapshots: &mut rt::snapshot::Snapshots,
    mut snapshot: impl FnMut(&mut rt::snapshot::World, &mut rt::snapshot::Snapshots),
) -> Result<(), Failure> {
    use rt::scheduler::Drive;

    loop {
        let inner: *mut rt::snapshot::Inner = world.inner();
        // SAFETY: the world outlives the call.
        let drive = world.routed(|| unsafe { (*inner).scheduler.drive() });

        // A snapshot taken before a coroutine was rebuilt names its old stack.
        if world.inner().scheduler.take_rebuilt() {
            snapshots.clear();
        }

        match drive {
            Drive::Done(result) => return result,
            Drive::Snapshot => snapshot(world, snapshots),
        }
    }
}

/// Where an execution resuming at `pos` and diverging at `divergence` takes
/// its first snapshot: at the divergence branch when that branch has another
/// alternative to come, whose execution then resumes there; otherwise a
/// spacing on.
fn next_snapshot(path: &rt::Path, pos: usize, divergence: Option<usize>, spacing: usize) -> usize {
    match divergence {
        Some(divergence) if pos < divergence && path.has_alternative(divergence) => {
            divergence.min(pos + spacing)
        }
        _ => pos + spacing,
    }
}

/// A pointer to the model closure, which `Builder::check` keeps alive across
/// every execution. An execution holds no count on it: a restore would
/// replay the count's changes.
struct Body<F>(*const F);

// SAFETY: `F: Sync`, and the pointee outlives every execution.
unsafe impl<F: Sync> Send for Body<F> {}

/// The root of model thread 0: the model closure, then the execution's
/// teardown.
fn body<F>(f: &Arc<F>) -> impl FnOnce() + Send + 'static
where
    F: Fn() + Sync + Send + 'static,
{
    let f = Body::<F>(&**f);

    move || {
        let f = f;
        // SAFETY: `Body`'s.
        unsafe { (*f.0)() };

        // Run the main thread's `thread_local` destructors before
        // the lazy_statics tear down: a TLS destructor may still
        // read a static, never the reverse.
        rt::drop_locals();

        let lazy_statics = rt::execution(|execution| execution.lazy_statics.drop());

        // drop outside of execution
        drop(lazy_statics);

        // Then the execution's instances of `const`-built locks' data, which
        // a lazy static's value may have borrowed.
        let lock_data = rt::execution(|execution| execution.take_lock_data());
        drop(lock_data);

        rt::thread_done();
    }
}

/// A thread to drive executions on, named for the thread that asked for the
/// check, so a panic reported from one of its model threads names the test.
fn driver_thread() -> std::thread::Builder {
    let builder = std::thread::Builder::new();

    match std::thread::current().name() {
        Some(name) => builder.name(name.to_string()),
        None => builder,
    }
}

/// How a serial walk ended.
enum Walk {
    /// The tree is exhausted, or `max_permutations` was reached.
    Done(Stats),

    /// `max_duration` passed first.
    TimedOut(Stats),

    /// An execution failed.
    Failed(Failure),

    /// The probe passed first: the walk so far, and the tree still to walk.
    Handover(Stats, rt::Path),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Limit {
    Permutations,
    Duration,
}

fn cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

/// Process-wide budget of extra exploration workers.
///
/// Every model run owns one worker outright and draws the rest from here,
/// returning them when it finishes. This is what lets sharding be on by
/// default: test harnesses run test functions concurrently, and without a
/// shared budget each concurrent model would spawn a full machine's worth of
/// workers and the whole run would thrash. A model with the machine to itself
/// takes all of it; sixteen concurrent models get one worker each and behave
/// exactly as they did before sharding existed. Nothing needs configuring per
/// test either way.
fn budget() -> &'static Mutex<usize> {
    static BUDGET: std::sync::OnceLock<Mutex<usize>> = std::sync::OnceLock::new();

    BUDGET.get_or_init(|| Mutex::new(cpus().saturating_sub(1)))
}

/// Extra workers claimed for one model run, returned on drop.
struct Grant(usize);

impl Grant {
    /// Claim up to `want` extra workers, taking whatever is free right now.
    /// Never blocks: a model always makes progress on its own worker, so
    /// waiting for capacity could only trade throughput for latency.
    fn claim(want: usize) -> Grant {
        let mut free = budget().lock().unwrap();
        let got = want.min(*free);
        *free -= got;

        Grant(got)
    }
}

impl Drop for Grant {
    fn drop(&mut self) {
        *budget().lock().unwrap() += self.0;
    }
}

/// The pool of subtrees still to explore, plus the run's shared verdict.
struct Shared {
    state: Mutex<QueueState>,
    wake: Condvar,
    /// Executions the run has explored, the serial probe's included: what
    /// `max_permutations` is checked against.
    executions: AtomicUsize,

    /// Sum of exited workers' sleep-set prunes.
    pruned: AtomicUsize,

    /// Sum of exited workers' bound-conservative-attributed executions.
    conservative: AtomicUsize,

    /// Waiting workers, and subtrees already posted for them. Mirrors of the
    /// fields inside `state`, published so [`Shared::donate`] can answer "is
    /// anyone waiting" without taking the lock.
    idle: AtomicUsize,
    queued: AtomicUsize,

    stop: std::sync::atomic::AtomicBool,

    /// Set when the run stopped at `max_duration`.
    timed_out: std::sync::atomic::AtomicBool,
}

struct QueueState {
    /// Subtrees posted for any worker to pick up.
    tasks: Vec<rt::Path>,

    /// Workers currently waiting for one.
    idle: usize,

    /// Set when the tree is exhausted, a ceiling was hit, or a model failed.
    done: bool,

    /// The first failed execution's failure, raised on the calling thread.
    failure: Option<Failure>,
}

impl Shared {
    /// A pool over `seed`, whose run has already spent `executions`.
    fn new(seed: rt::Path, executions: usize) -> Shared {
        Shared {
            state: Mutex::new(QueueState {
                tasks: vec![seed],
                idle: 0,
                done: false,
                failure: None,
            }),
            wake: Condvar::new(),
            executions: AtomicUsize::new(executions),
            pruned: AtomicUsize::new(0),
            conservative: AtomicUsize::new(0),
            idle: AtomicUsize::new(0),
            queued: AtomicUsize::new(1),
            stop: std::sync::atomic::AtomicBool::new(false),
            timed_out: std::sync::atomic::AtomicBool::new(false),
        }
    }

    /// Claim a subtree, blocking while peers still hold work that they might
    /// share. Returns `None` once the run is over — including when every
    /// worker is waiting at once, which is exactly the condition for the tree
    /// being exhausted.
    fn take(&self, workers: usize) -> Option<rt::Path> {
        let mut state = self.state.lock().unwrap();

        loop {
            if state.done {
                return None;
            }

            if let Some(task) = state.tasks.pop() {
                self.queued.store(state.tasks.len(), Relaxed);
                return Some(task);
            }

            state.idle += 1;
            self.idle.store(state.idle, Relaxed);

            if state.idle == workers {
                state.done = true;
                self.wake.notify_all();
                return None;
            }

            state = self.wake.wait(state).unwrap();
            state.idle -= 1;
            self.idle.store(state.idle, Relaxed);
        }
    }

    /// Offer idle peers part of this path's remaining subtree.
    ///
    /// The check is two relaxed loads rather than a lock acquire. Every worker
    /// runs it every `DONATE_INTERVAL` executions and almost always finds
    /// nobody waiting, so taking the pool mutex to ask would be paying for a
    /// lock per sixteen executions to be told there is nothing to do.
    ///
    /// Both counters are hints, and neither can be wrong in a way that
    /// matters: a peer missed because the load was stale waits one more
    /// interval, and work carved for a peer that has since found some of its
    /// own just goes to the pool, where the claim record settles who explores
    /// it exactly as it would have anyway.
    fn donate(&self, path: &mut rt::Path) {
        let wanted = self
            .idle
            .load(Relaxed)
            .saturating_sub(self.queued.load(Relaxed));

        if wanted == 0 || self.stopped() {
            return;
        }

        self.post(path.split_off(wanted));
    }

    /// Add tasks to the pool, waking a waiter for each.
    fn post(&self, tasks: Vec<rt::Path>) {
        if tasks.is_empty() {
            return;
        }

        let mut state = self.state.lock().unwrap();

        if state.done {
            return;
        }

        for task in tasks {
            state.tasks.push(task);
            self.wake.notify_one();
        }

        self.queued.store(state.tasks.len(), Relaxed);
    }

    fn stop(&self) {
        self.stop.store(true, Relaxed);

        let mut state = self.state.lock().unwrap();
        state.done = true;
        self.wake.notify_all();
    }

    fn stopped(&self) -> bool {
        self.stop.load(Relaxed)
    }

    fn fail(&self, failure: Failure) {
        self.stop.store(true, Relaxed);

        let mut state = self.state.lock().unwrap();
        state.done = true;

        if state.failure.is_none() {
            state.failure = Some(failure);
        }

        self.wake.notify_all();
    }

    fn into_failure(self) -> Option<Failure> {
        self.state.into_inner().unwrap().failure
    }
}

impl Default for Builder {
    fn default() -> Self {
        Self::new()
    }
}

/// Run all concurrent permutations of the provided closure.
///
/// Uses a default [`Builder`] which can be affected by environment variables.
pub fn model<F>(f: F)
where
    F: Fn() + Sync + Send + 'static,
{
    let subscriber = fmt::Subscriber::builder()
        .with_env_filter(EnvFilter::from_env("LOOM_LOG"))
        .with_test_writer()
        .without_time()
        .finish();

    subscriber::with_default(subscriber, || {
        Builder::new().check(f);
    });
}

#[cfg(feature = "checkpoint")]
mod checkpoint {
    use std::fs::File;
    use std::io::prelude::*;
    use std::path::Path;

    pub(crate) fn load_execution_path(fs_path: &Path) -> crate::rt::Path {
        let mut file = File::open(fs_path).unwrap();
        let mut contents = String::new();
        file.read_to_string(&mut contents).unwrap();
        serde_json::from_str(&contents).unwrap()
    }

    pub(crate) fn store_execution_path(path: &crate::rt::Path, fs_path: &Path) {
        let serialized = serde_json::to_string(path).unwrap();

        let mut file = File::create(fs_path).unwrap();
        file.write_all(serialized.as_bytes()).unwrap();
    }
}

#[cfg(not(feature = "checkpoint"))]
mod checkpoint {
    use std::path::Path;

    pub(crate) fn load_execution_path(_fs_path: &Path) -> crate::rt::Path {
        panic!("not compiled with `checkpoint` feature")
    }

    pub(crate) fn store_execution_path(_path: &crate::rt::Path, _fs_path: &Path) {
        panic!("not compiled with `checkpoint` feature")
    }
}
