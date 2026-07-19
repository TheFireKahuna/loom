//! Model concurrent programs.

use crate::rt::{self, Execution, Scheduler};
use std::any::Any;
use std::panic::{self, AssertUnwindSafe};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use tracing::{info, subscriber};
use tracing_subscriber::{fmt, EnvFilter};

const DEFAULT_MAX_THREADS: usize = 5;
const DEFAULT_MAX_BRANCHES: usize = 1_000;

/// How often a worker offers part of its remaining subtree to idle peers.
/// Each check is an uncontended mutex acquire, so this only has to be small
/// enough that the initial fan-out is not the bottleneck.
const DONATE_INTERVAL: usize = 16;

/// How deep the exhaustively-expanded, shardable region of the tree goes.
///
/// Every branch above this depth is expanded without partial-order reduction
/// so that subtrees can be handed between workers soundly (see
/// `Path::split_depth`), which costs some redundant exploration near the root;
/// every branch below it is reduced normally and belongs to one worker. Deeper
/// means more independent subtrees to hand out and more redundancy, so this
/// wants to be just deep enough to keep every worker fed.
///
/// 96 is the measured knee on a 16-thread host against the nt-sync futex
/// rigs: shallower starves workers on an unbalanced tree (48 → 33s), deeper
/// buys no more balance and pays for it in redundant exploration (192 → 28s,
/// with executions up from 3.5M to 6.0M).
const SPLIT_DEPTH: usize = 96;

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

    /// Maximum amount of time to spend on checking
    ///
    /// Defaults to `LOOM_MAX_DURATION` environment variable.
    pub max_duration: Option<Duration>,

    /// Maximum number of thread preemptions to explore
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

    /// How long a model may run as a plain reduced serial walk before the run
    /// restarts under a worker pool.
    ///
    /// Sharding expands the shardable region of the tree without partial-order
    /// reduction, which only pays off once there is enough tree to hand
    /// around. Zero shards immediately — mainly useful for testing the sharded
    /// path on a model too small to reach the probe on its own.
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

    /// Branch depth above which scheduling is expanded exhaustively so that
    /// subtrees can be handed between workers soundly.
    ///
    /// Deeper gives more independent subtrees to share out — better load
    /// balance on an unbalanced tree — at the cost of exploring more of the
    /// top of the tree than partial-order reduction would. Only consulted
    /// when a run actually shards.
    ///
    /// Defaults to the `LOOM_SPLIT_DEPTH` environment variable, else a value
    /// measured against the nt-sync futex models.
    pub split_depth: usize,
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
    pub fn check<F>(&self, f: F) -> Stats
    where
        F: Fn() + Sync + Send + 'static,
    {
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

        // Sharding is not free: the shardable region of the tree is expanded
        // without partial-order reduction, which is a good trade only once
        // there is enough tree to hand around. So every run starts as an
        // ordinary reduced serial walk, and only a model still going after
        // `PROBE` is big enough to be worth restarting under a worker pool.
        // Models that finish inside the probe — the large majority — explore
        // exactly what they always did.
        let probe = if workers == 1 {
            None
        } else {
            Some(start + self.probe)
        };

        let stats = match self.check_serial(&f, probe) {
            Ok(stats) => stats,
            Err(probed) => {
                let stats = self.check_parallel(&f, workers);

                Stats {
                    // The probe's executions were real work, even though its
                    // tree was thrown away; counting only the second phase
                    // would understate what the run cost.
                    executions: stats.executions + probed,
                    ..stats
                }
            }
        };

        drop(grant);

        // `LOOM_LOG` forces a serial walk, so it cannot report what a sharded
        // run explored. This can, and the execution count is the number that
        // says whether a reduction is doing anything.
        if std::env::var_os("LOOM_STATS").is_some() {
            eprintln!(
                "loom: {} executions, {} worker(s), bound={:?}, {:.2}s",
                stats.executions,
                stats.threads,
                self.preemption_bound,
                start.elapsed().as_secs_f64(),
            );
        }

        stats
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
        execution
    }

    /// Whether the run has hit a configured ceiling. Checked once per
    /// execution; `max_duration` reads the clock, so it is only consulted on
    /// the same cadence as work donation.
    fn limit_reached(&self, executions: usize, start: Instant) -> bool {
        if let Some(max) = self.max_permutations {
            if executions >= max {
                return true;
            }
        }

        if let Some(max) = self.max_duration {
            if executions % DONATE_INTERVAL == 0 && start.elapsed() >= max {
                return true;
            }
        }

        false
    }

    /// Walk the whole tree on this thread. Returns `Err` with the executions
    /// done so far if `deadline` passes first — the caller then knows the model
    /// is large enough to be worth restarting under a worker pool.
    fn check_serial<F>(&self, f: &Arc<F>, deadline: Option<Instant>) -> Result<Stats, usize>
    where
        F: Fn() + Sync + Send + 'static,
    {
        let mut i = 1;
        let mut _span = tracing::info_span!("iter", message = i).entered();

        let mut execution = self.new_execution();
        let mut scheduler = Scheduler::new(self.max_threads);

        if let Some(ref path) = self.checkpoint_file {
            if path.exists() {
                execution.path = checkpoint::load_execution_path(path);
                execution.path.set_max_branches(self.max_branches);
            }
        }

        let start = Instant::now();
        loop {
            if i % self.checkpoint_interval == 0 {
                info!(parent: None, "");
                info!(
                    parent: None,
                    " ================== Iteration {} ==================", i
                );
                info!(parent: None, "");

                if let Some(ref path) = self.checkpoint_file {
                    checkpoint::store_execution_path(&execution.path, path);
                }

                if let Some(max_permutations) = self.max_permutations {
                    if i >= max_permutations {
                        return Ok(self.stats(i - 1, 1));
                    }
                }

                if let Some(max_duration) = self.max_duration {
                    if start.elapsed() >= max_duration {
                        return Ok(self.stats(i - 1, 1));
                    }
                }
            }

            // Only a run that is deciding whether to shard reads the clock
            // here, and for it one read per execution is noise against the
            // execution itself. Checking on the donation cadence instead would
            // overshoot the probe by up to a full interval, and those
            // executions are thrown away along with the tree they built.
            if let Some(deadline) = deadline {
                if Instant::now() >= deadline {
                    return Err(i - 1);
                }
            }

            run_once(&mut scheduler, &mut execution, f);

            execution.check_for_leaks();

            i += 1;

            // Create the next iteration's `tracing` span before trying to step to the next
            // execution, as the `Execution` will capture the current span when
            // it's reset.
            _span = tracing::info_span!(parent: None, "iter", message = i).entered();
            if let Some(next) = execution.step() {
                execution = next;
            } else {
                info!(parent: None, "Completed in {} iterations", i - 1);
                return Ok(self.stats(i - 1, 1));
            }
        }
    }

    /// Explore the tree with `workers` threads over a shared pool of subtrees.
    ///
    /// One worker seeds with the whole tree; whenever peers go idle, a worker
    /// carves off its shallowest unexplored branches and posts them. Workers
    /// never share an `Execution`, a `Scheduler`, or a coroutine — only the
    /// task pool — so the model runs exactly as it does serially, one path at
    /// a time per worker.
    fn check_parallel<F>(&self, f: &Arc<F>, workers: usize) -> Stats
    where
        F: Fn() + Sync + Send + 'static,
    {
        let mut seed = self.new_execution().path;
        seed.set_split_depth(self.split_depth);

        let shared = Shared::new(seed);

        // Worker threads do not inherit the caller's `tracing` subscriber.
        let dispatch = tracing::dispatcher::get_default(|d| d.clone());
        let start = Instant::now();

        std::thread::scope(|scope| {
            for _ in 0..workers {
                let shared = &shared;
                let f = f.clone();
                let dispatch = dispatch.clone();

                scope.spawn(move || {
                    tracing::dispatcher::with_default(&dispatch, || {
                        // The `Scheduler` is built inside the guarded scope so
                        // that a model panic unwinds through its `Drop` with
                        // the thread still flagged as panicking, leaving
                        // suspended coroutines to the generator crate exactly
                        // as an unguarded serial run does.
                        let run = panic::catch_unwind(AssertUnwindSafe(|| {
                            self.worker(shared, &f, workers, start);
                        }));

                        if let Err(payload) = run {
                            shared.fail(payload);
                        }
                    });
                });
            }
        });

        let executions = shared.executions.load(Relaxed);

        if let Some(payload) = shared.into_failure() {
            panic::resume_unwind(payload);
        }

        info!(parent: None, "Completed in {} iterations", executions);
        self.stats(executions, workers)
    }

    fn worker<F>(&self, shared: &Shared, f: &Arc<F>, workers: usize, start: Instant)
    where
        F: Fn() + Sync + Send + 'static,
    {
        let mut scheduler = Scheduler::new(self.max_threads);

        while let Some(path) = shared.take(workers) {
            let mut execution = self.new_execution();
            execution.path = path;

            loop {
                run_once(&mut scheduler, &mut execution, f);

                execution.check_for_leaks();

                let done = shared.executions.fetch_add(1, Relaxed) + 1;

                if self.limit_reached(done, start) {
                    shared.stop();
                    return;
                }

                match execution.step() {
                    Some(next) => execution = next,
                    None => break,
                }

                if done % DONATE_INTERVAL == 0 {
                    shared.donate(&mut execution.path);

                    if shared.stopped() {
                        return;
                    }
                }
            }
        }
    }

    fn stats(&self, executions: usize, threads: usize) -> Stats {
        let _ = self;

        Stats {
            executions,
            threads,
        }
    }
}

fn run_once<F>(scheduler: &mut Scheduler, execution: &mut Execution, f: &Arc<F>)
where
    F: Fn() + Sync + Send + 'static,
{
    let f = f.clone();

    scheduler.run(execution, move || {
        f();

        // Run the main thread's `thread_local` destructors before
        // the lazy_statics tear down: a TLS destructor may still
        // read a static, never the reverse.
        rt::drop_locals();

        let lazy_statics = rt::execution(|execution| execution.lazy_statics.drop());

        // drop outside of execution
        drop(lazy_statics);

        rt::thread_done();
    });
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
    executions: AtomicUsize,
    stop: std::sync::atomic::AtomicBool,
}

struct QueueState {
    /// Subtrees posted for any worker to pick up.
    tasks: Vec<rt::Path>,

    /// Workers currently waiting for one.
    idle: usize,

    /// Set when the tree is exhausted, a ceiling was hit, or a model failed.
    done: bool,

    /// The first model panic seen, re-raised on the calling thread.
    failure: Option<Box<dyn Any + Send>>,
}

impl Shared {
    fn new(seed: rt::Path) -> Shared {
        Shared {
            state: Mutex::new(QueueState {
                tasks: vec![seed],
                idle: 0,
                done: false,
                failure: None,
            }),
            wake: Condvar::new(),
            executions: AtomicUsize::new(0),
            stop: std::sync::atomic::AtomicBool::new(false),
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
                return Some(task);
            }

            state.idle += 1;

            if state.idle == workers {
                state.done = true;
                self.wake.notify_all();
                return None;
            }

            state = self.wake.wait(state).unwrap();
            state.idle -= 1;
        }
    }

    /// Offer idle peers part of this path's remaining subtree.
    fn donate(&self, path: &mut rt::Path) {
        let wanted = {
            let state = self.state.lock().unwrap();

            if state.done {
                return;
            }

            state.idle.saturating_sub(state.tasks.len())
        };

        if wanted == 0 {
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

    fn fail(&self, payload: Box<dyn Any + Send>) {
        self.stop.store(true, Relaxed);

        let mut state = self.state.lock().unwrap();
        state.done = true;

        if state.failure.is_none() {
            state.failure = Some(payload);
        }

        self.wake.notify_all();
    }

    fn into_failure(self) -> Option<Box<dyn Any + Send>> {
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
