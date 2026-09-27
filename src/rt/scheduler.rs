#![allow(deprecated)]

use crate::rt::execution::Failure;
use crate::rt::Execution;

use generator::{self, Generator, Gn};
use scoped_tls::scoped_thread_local;
use std::cell::RefCell;
use std::collections::VecDeque;
use std::panic::{self, AssertUnwindSafe};

pub(crate) struct Scheduler {
    max_threads: usize,

    /// Stack size, in bytes, of a coroutine whose spawn names none.
    stack_size: usize,

    /// Coroutines pooled across iterations, indexed by loom thread id.
    ///
    /// After an iteration's closure returns, its coroutine parks back at
    /// the recv point (`yield_`) and the next iteration re-arms it with a
    /// fresh closure via `set_para`, so the per-iteration cost is a
    /// context switch instead of a stack mmap/munmap plus `done!()`'s
    /// panic-driven unwind. Termination happens in `Drop`.
    threads: Vec<PooledThread>,
}

struct PooledThread {
    gen: Thread,

    /// The stack size, in bytes, the coroutine was built with. A spawn
    /// requesting a different size cannot reuse this stack; the coroutine is
    /// retired and rebuilt.
    stack_size: usize,
}

type Thread = Generator<'static, Option<Box<dyn FnOnce()>>, ()>;

scoped_thread_local! {
    static STATE: Run<'_>
}

/// One execution's scheduler state, reachable from its coroutines.
struct Run<'a> {
    state: RefCell<State<'a>>,

    /// The most recent panic raised on one of this run's coroutines, recorded
    /// by the panic hook. Outside `state`: a panic raised inside the runtime
    /// holds that borrow.
    panic: RefCell<Option<PanicNote>>,

    /// Why the execution failed, once it has. The driver stops scheduling at
    /// the first resume that returns with this set.
    failure: RefCell<Option<Failure>>,
}

/// What the panic hook saw of a panic on a model thread: all of it that
/// survives when the unwind is abandoned before it reaches its catch.
struct PanicNote {
    message: Option<String>,
}

struct QueuedSpawn {
    f: Box<dyn FnOnce()>,
    stack_size: Option<usize>,
}

struct State<'a> {
    execution: &'a mut Execution,
    queued_spawn: &'a mut VecDeque<QueuedSpawn>,
}

impl Scheduler {
    /// Create an execution
    pub(crate) fn new(capacity: usize, stack_size: usize) -> Scheduler {
        Scheduler {
            max_threads: capacity,
            stack_size,
            threads: Vec::with_capacity(capacity),
        }
    }

    /// Access the execution
    pub(crate) fn with_execution<F, R>(f: F) -> R
    where
        F: FnOnce(&mut Execution) -> R,
    {
        Self::with_state(|state| f(state.execution))
    }

    /// Access the execution if there is one to access: `None` outside a model,
    /// and while the execution is already borrowed further up the stack.
    pub(crate) fn try_with_execution<F, R>(f: F) -> Option<R>
    where
        F: FnOnce(&mut Execution) -> R,
    {
        if !STATE.is_set() {
            return None;
        }
        STATE.with(|state| {
            let mut state = state.try_borrow_mut().ok()?;
            Some(f(state.execution))
        })
    }

    /// Perform a context switch
    pub(crate) fn switch() {
        use std::future::Future;
        use std::pin::Pin;
        use std::ptr;
        use std::task::{Context, RawWaker, RawWakerVTable, Waker};

        unsafe fn noop_clone(_: *const ()) -> RawWaker {
            unreachable!()
        }
        unsafe fn noop(_: *const ()) {}

        // Wrapping with an async block deals with the thread-local context
        // `std` uses to manage async blocks
        let mut switch = async { generator::yield_with(()) };
        let switch = unsafe { Pin::new_unchecked(&mut switch) };

        let raw_waker = RawWaker::new(
            ptr::null(),
            &RawWakerVTable::new(noop_clone, noop, noop, noop),
        );
        let waker = unsafe { Waker::from_raw(raw_waker) };
        let mut cx = Context::from_waker(&waker);

        assert!(switch.poll(&mut cx).is_ready());
    }

    pub(crate) fn spawn(stack_size: Option<usize>, f: Box<dyn FnOnce()>) {
        Self::with_state(|state| state.queued_spawn.push_back(QueuedSpawn { stack_size, f }));
    }

    /// Run one execution of `f` to completion.
    ///
    /// On failure the execution stops where it failed: every coroutine is
    /// abandoned suspended, and no user destructor runs in it again. The
    /// caller must not drop `execution` either — its thread locals and lazy
    /// statics are user values whose destructors would run against a failed
    /// execution — and must not run another on this OS thread: an abandoned
    /// unwind leaves the thread's panic count raised.
    pub(crate) fn run<F>(&mut self, execution: &mut Execution, f: F) -> Result<(), Failure>
    where
        F: FnOnce() + Send + 'static,
    {
        self.arm(0, Box::new(f), None);
        let mut used = 1;

        // The scoped-TLS state brackets the whole iteration, not each tick:
        // set/unset plus a fresh `RefCell` per branch is pure overhead when
        // the borrowed execution is the same one throughout. Inside the
        // closure the execution is only reachable through `run` — the
        // coroutines borrow it via `STATE` between resumes.
        let mut queued_spawn = VecDeque::new();
        let run = Run {
            state: RefCell::new(State {
                execution,
                queued_spawn: &mut queued_spawn,
            }),
            panic: RefCell::new(None),
            failure: RefCell::new(None),
        };

        let result = STATE.set(unsafe { transmute_lt(&run) }, || loop {
            let active = {
                let state = run.state.borrow();

                if state.execution.threads.is_complete() {
                    // Every loom thread has terminated, so every armed
                    // coroutine has finished its closure and parked back at
                    // its recv point, ready for the next iteration.
                    return Ok(());
                }

                state.execution.threads.active_id()
            };

            let gen = &mut self.threads[active.as_usize()].gen;

            // A panic that unwound to the coroutine's root: the thread's own
            // destructors have run, against a still-consistent runtime.
            if let Err(payload) = panic::catch_unwind(AssertUnwindSafe(|| gen.resume())) {
                std::hint::cold_path();
                return Err(Failure::Panic(payload));
            }

            if let Some(failure) = run.failure.borrow_mut().take() {
                std::hint::cold_path();
                return Err(failure);
            }

            loop {
                // Armed coroutines park at their recv point without running
                // user code, so `arm` is safe under the live `STATE`.
                let next = run.state.borrow_mut().queued_spawn.pop_front();

                let Some(QueuedSpawn { f, stack_size }) = next else {
                    break;
                };

                assert!(used < self.max_threads);

                self.arm(used, f, stack_size);
                used += 1;
            }
        });

        if result.is_err() {
            // Suspended coroutines are never resumed, so their frames are
            // never unwound; unstarted spawns own user values too.
            for th in self.threads.drain(..) {
                std::mem::forget(th);
            }
            drop(run);
            std::mem::forget(queued_spawn);
        }

        result
    }

    /// Record why the execution failed. The failing thread then switches
    /// out, and the driver resumes nothing further.
    pub(crate) fn fail(failure: Failure) {
        STATE.with(|run| {
            let mut slot = run.failure.borrow_mut();

            if slot.is_none() {
                *slot = Some(failure);
            }
        });
    }

    /// The active thread is unwinding a panic and has reached a tracked
    /// operation, from a destructor. Its execution has already failed, and
    /// the runtime cannot run the operation without the risk of failing
    /// again mid-unwind — a second panic, which aborts the process. So the
    /// thread stops here, suspended for good, and the model fails with the
    /// panic's message as the hook recorded it.
    #[cold]
    #[inline(never)]
    fn abandon_unwind() -> ! {
        let note = STATE.with(|run| run.panic.borrow_mut().take());

        let message = note.and_then(|note| note.message).unwrap_or_else(|| {
            "a model thread panicked, and its unwinding reached a tracked operation".to_string()
        });

        Self::fail(Failure::Panic(Box::new(message)));

        loop {
            Self::switch();
        }
    }

    /// Install, once per process, the panic hook that records a model
    /// thread's panic for [`abandon_unwind`](Self::abandon_unwind). It
    /// forwards every panic to the hook it replaces, so the panic is still
    /// reported at its own site.
    pub(crate) fn install_panic_hook() {
        static INSTALL: std::sync::Once = std::sync::Once::new();

        // `set_hook` refuses a panicking thread; a later check installs it.
        if std::thread::panicking() {
            return;
        }

        INSTALL.call_once(|| {
            let next = panic::take_hook();

            panic::set_hook(Box::new(move |info| {
                if STATE.is_set() && generator::is_generator() {
                    STATE.with(|run| {
                        if let Ok(mut note) = run.panic.try_borrow_mut() {
                            *note = Some(PanicNote {
                                message: info.payload_as_str().map(str::to_owned),
                            });
                        }
                    });
                }

                next(info);
            }));
        });
    }

    /// Hand `f` to the pooled coroutine at `index` (loom thread id),
    /// building or rebuilding the coroutine first if needed. On return
    /// the coroutine is parked one `resume` away from entering `f`,
    /// exactly like a freshly spawned thread.
    fn arm(&mut self, index: usize, f: Box<dyn FnOnce()>, stack_size: Option<usize>) {
        let stack_size = stack_size.unwrap_or(self.stack_size);

        match self.threads.get_mut(index) {
            Some(slot) if slot.stack_size == stack_size => {}
            Some(slot) => {
                retire(&mut slot.gen);
                *slot = PooledThread {
                    gen: park_thread(stack_size),
                    stack_size,
                };
            }
            None => {
                debug_assert_eq!(index, self.threads.len(), "[loom internal bug]");
                self.threads.push(PooledThread {
                    gen: park_thread(stack_size),
                    stack_size,
                });
            }
        }

        let gen = &mut self.threads[index].gen;
        gen.set_para(Some(f));
        gen.resume();
    }

    fn with_state<F, R>(f: F) -> R
    where
        F: FnOnce(&mut State<'_>) -> R,
    {
        if !STATE.is_set() {
            panic!("cannot access Loom execution state from outside a Loom model. \
            are you accessing a Loom synchronization primitive from outside a Loom test (a call to `model` or `check`)?")
        }

        // Only a model thread reaches here, and the driver never resumes one
        // while its OS thread unwinds: panicking means this thread unwinds.
        if std::thread::panicking() {
            Self::abandon_unwind();
        }

        STATE.with(|run| f(&mut run.state.borrow_mut()))
    }
}

impl Drop for Scheduler {
    fn drop(&mut self) {
        // A panicking model unwinds through `run` with coroutines still
        // suspended mid-closure; resuming one during the unwind would
        // re-enter the model outside an execution. Leave them to the
        // generator crate's own panicking-aware `Drop`.
        if std::thread::panicking() {
            return;
        }

        for th in &mut self.threads {
            retire(&mut th.gen);
        }
    }
}

/// Build a coroutine with a `stack_size`-byte stack and run it to its recv
/// point, where it waits for `arm` to hand it a closure.
fn park_thread(stack_size: usize) -> Thread {
    let body = || {
        loop {
            let f: Option<Option<Box<dyn FnOnce()>>> = generator::yield_(());

            if let Some(f) = f {
                generator::yield_with(());
                f.unwrap()();
            } else {
                // Retired: a plain return completes the coroutine
                // without `done!()`'s panic-driven stack unwind.
                return;
            }
        }
    };
    // The generator sizes stacks in words; an odd count makes it fill the
    // whole stack with a usage-tracking pattern.
    let words = (stack_size / std::mem::size_of::<usize>()) & !1;
    let mut g = Gn::new_opt(words, body);
    g.resume();
    g
}

/// Resume the parked coroutine with no closure: its recv loop sees
/// `None` and returns, completing the generator.
fn retire(gen: &mut Thread) {
    gen.resume();
    assert!(gen.is_done(), "[loom internal bug] coroutine still live");
}

unsafe fn transmute_lt<'a, 'b>(run: &'a Run<'b>) -> &'a Run<'static> {
    ::std::mem::transmute(run)
}
