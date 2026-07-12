#![allow(deprecated)]

use crate::rt::Execution;

use generator::{self, Generator, Gn};
use scoped_tls::scoped_thread_local;
use std::cell::RefCell;
use std::collections::VecDeque;

pub(crate) struct Scheduler {
    max_threads: usize,

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

    /// The stack size the coroutine was built with (`None` = generator
    /// crate default). A spawn requesting a different size cannot reuse
    /// this stack; the coroutine is retired and rebuilt.
    stack_size: Option<usize>,
}

type Thread = Generator<'static, Option<Box<dyn FnOnce()>>, ()>;

scoped_thread_local! {
    static STATE: RefCell<State<'_>>
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
    pub(crate) fn new(capacity: usize) -> Scheduler {
        Scheduler {
            max_threads: capacity,
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

    pub(crate) fn run<F>(&mut self, execution: &mut Execution, f: F)
    where
        F: FnOnce() + Send + 'static,
    {
        self.arm(0, Box::new(f), None);
        let mut used = 1;

        // The scoped-TLS state brackets the whole iteration, not each tick:
        // set/unset plus a fresh `RefCell` per branch is pure overhead when
        // the borrowed execution is the same one throughout. Inside the
        // closure the execution is only reachable through `state` — the
        // coroutines borrow it via `STATE` between resumes.
        let mut queued_spawn = VecDeque::new();
        let state = RefCell::new(State {
            execution,
            queued_spawn: &mut queued_spawn,
        });

        STATE.set(unsafe { transmute_lt(&state) }, || loop {
            let active = {
                let state = state.borrow();

                if state.execution.threads.is_complete() {
                    // Every loom thread has terminated, so every armed
                    // coroutine has finished its closure and parked back at
                    // its recv point, ready for the next iteration.
                    return;
                }

                state.execution.threads.active_id()
            };

            self.threads[active.as_usize()].gen.resume();

            loop {
                // Armed coroutines park at their recv point without running
                // user code, so `arm` is safe under the live `STATE`.
                let next = state.borrow_mut().queued_spawn.pop_front();

                let Some(QueuedSpawn { f, stack_size }) = next else {
                    break;
                };

                assert!(used < self.max_threads);

                self.arm(used, f, stack_size);
                used += 1;
            }
        });
    }

    /// Hand `f` to the pooled coroutine at `index` (loom thread id),
    /// building or rebuilding the coroutine first if needed. On return
    /// the coroutine is parked one `resume` away from entering `f`,
    /// exactly like a freshly spawned thread.
    fn arm(&mut self, index: usize, f: Box<dyn FnOnce()>, stack_size: Option<usize>) {
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
        STATE.with(|state| f(&mut state.borrow_mut()))
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

/// Build a coroutine and run it to its recv point, where it waits for
/// `arm` to hand it a closure.
fn park_thread(stack_size: Option<usize>) -> Thread {
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
    let mut g = match stack_size {
        Some(stack_size) => Gn::new_opt(stack_size, body),
        None => Gn::new(body),
    };
    g.resume();
    g
}

/// Resume the parked coroutine with no closure: its recv loop sees
/// `None` and returns, completing the generator.
fn retire(gen: &mut Thread) {
    gen.resume();
    assert!(gen.is_done(), "[loom internal bug] coroutine still live");
}

unsafe fn transmute_lt<'a, 'b>(state: &'a RefCell<State<'b>>) -> &'a RefCell<State<'static>> {
    ::std::mem::transmute(state)
}
