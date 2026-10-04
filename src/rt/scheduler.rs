#![allow(deprecated)]

use crate::rt::execution::Failure;
use crate::rt::world;
use crate::rt::Execution;

use generator::{self, Generator, Gn};
use scoped_tls::scoped_thread_local;
use std::cell::{Cell, RefCell};
use std::collections::VecDeque;
use std::panic::{self, AssertUnwindSafe};

pub(crate) struct Scheduler {
    max_threads: usize,

    /// Stack size, in bytes, of a coroutine whose spawn names none.
    stack_size: usize,

    /// The execution being driven, reachable from its coroutines.
    run: Run,

    /// Coroutines armed in the current execution.
    used: usize,

    /// A coroutine was rebuilt since this was last cleared: a snapshot taken
    /// before then names a stack that no longer exists.
    rebuilt: bool,

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

    /// Where the coroutine's stack pointer was when it last yielded, and the
    /// end of its stack's mapping: the live part a snapshot copies. The
    /// mapping runs past the stack's top, because the generator keeps the
    /// coroutine's context and closure there.
    sp: usize,
    end: usize,

    /// Whether the coroutine was running the code under test, rather than
    /// the runtime, when it last yielded: what its allocations resume as.
    in_model: bool,
}

type Thread = Generator<'static, Option<Box<dyn FnOnce()>>, ()>;

scoped_thread_local! {
    static STATE: Run
}

/// Stack pointer and stack top of the coroutine that yielded last.
#[thread_local]
static YIELDED: Cell<(usize, usize)> = Cell::new((0, 0));

/// One execution's scheduler state, reachable from its coroutines.
struct Run {
    state: RefCell<State>,

    /// The most recent panic raised on one of this run's coroutines, recorded
    /// by the panic hook. Outside `state`: a panic raised inside the runtime
    /// holds that borrow.
    panic: RefCell<Option<PanicNote>>,

    /// Why the execution failed, once it has. The driver stops scheduling at
    /// the first resume that returns with this set.
    failure: RefCell<Option<Failure>>,

    /// The coroutine that yielded asks the driver for a snapshot.
    snapshot: Cell<bool>,
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

struct State {
    /// The execution being driven; set by `start` for the whole execution.
    execution: *mut Execution,
    queued_spawn: VecDeque<QueuedSpawn>,
}

impl State {
    fn execution(&mut self) -> &mut Execution {
        // SAFETY: `start` points this at an execution that outlives every
        // drive of it, and the `RefCell` around `State` makes this the only
        // reference made through it at a time.
        unsafe { &mut *self.execution }
    }
}

/// How one stretch of driving an execution ended.
pub(crate) enum Drive {
    /// The execution finished, or failed.
    Done(Result<(), Failure>),

    /// A model thread asked for a snapshot; every coroutine is suspended, and
    /// `drive` continues the execution.
    Snapshot,
}

impl Scheduler {
    /// Create an execution
    pub(crate) fn new(capacity: usize, stack_size: usize) -> Scheduler {
        Scheduler {
            max_threads: capacity,
            stack_size,
            run: Run {
                state: RefCell::new(State {
                    execution: std::ptr::null_mut(),
                    queued_spawn: VecDeque::new(),
                }),
                panic: RefCell::new(None),
                failure: RefCell::new(None),
                snapshot: Cell::new(false),
            },
            used: 0,
            rebuilt: false,
            threads: Vec::with_capacity(capacity),
        }
    }

    /// A scheduler whose coroutines are all built up front, so the pool never
    /// grows under a snapshot: a restore would forget a stack built after it.
    pub(crate) fn new_pooled(capacity: usize, stack_size: usize) -> Scheduler {
        let mut scheduler = Scheduler::new(capacity, stack_size);
        for _ in 0..capacity {
            let gen = park_thread(stack_size);
            let (sp, top) = YIELDED.get();
            scheduler.threads.push(PooledThread {
                gen,
                stack_size,
                sp,
                end: mapping_end(top),
                in_model: false,
            });
        }
        scheduler
    }

    /// Run one coroutine to completion, so whatever the generator crate
    /// builds lazily on first use exists before a world does.
    pub(crate) fn warm_up() {
        let mut gen = park_thread(DEFAULT_WARM_STACK);
        retire(&mut gen);
    }

    /// Whether a coroutine was rebuilt since the last call.
    pub(crate) fn take_rebuilt(&mut self) -> bool {
        std::mem::take(&mut self.rebuilt)
    }

    /// The live stack range of every pooled coroutine, as `(low, high)`.
    pub(crate) fn stacks(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.threads
            .iter()
            .filter(|th| th.end != 0)
            .map(|th| (th.sp.saturating_sub(STACK_MARGIN), th.end))
    }

    /// Access the execution
    pub(crate) fn with_execution<F, R>(f: F) -> R
    where
        F: FnOnce(&mut Execution) -> R,
    {
        Self::with_state(|state| f(state.execution()))
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
        STATE.with(|run| {
            let mut state = run.state.try_borrow_mut().ok()?;
            let _runtime = world::runtime();
            Some(f(state.execution()))
        })
    }

    /// Perform a context switch
    pub(crate) fn switch() {
        // Not through `with_state`: a thread abandoning its unwind switches
        // here forever, and `with_state` would send it back to abandon.
        use std::future::Future;
        use std::pin::Pin;
        use std::ptr;
        use std::task::{Context, RawWaker, RawWakerVTable, Waker};

        unsafe fn noop_clone(_: *const ()) -> RawWaker {
            unreachable!()
        }
        unsafe fn noop(_: *const ()) {}

        YIELDED.set((stack_pointer(), stack_top()));

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

    /// Ask the driver for a snapshot here, unless this model thread is
    /// inside the execution already (a nested access), where it asks again at
    /// its next access.
    fn request_snapshot() {
        let free = STATE.with(|run| run.state.try_borrow_mut().is_ok());
        if !free || !generator::is_generator() {
            return;
        }

        crate::rt::snapshot::clear_due();
        STATE.with(|run| run.snapshot.set(true));
        Self::switch();
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
        self.start(execution, f);

        loop {
            match self.drive() {
                Drive::Done(result) => return result,
                Drive::Snapshot => {}
            }
        }
    }

    /// Begin an execution of `f` on `execution`, which must outlive every
    /// [`drive`](Self::drive) of it.
    pub(crate) fn start<F>(&mut self, execution: &mut Execution, f: F)
    where
        F: FnOnce() + Send + 'static,
    {
        self.run.state.get_mut().execution = execution;
        self.arm(0, Box::new(f), None);
        self.used = 1;
    }

    /// Drive the started execution until it ends or a model thread asks for
    /// a snapshot.
    pub(crate) fn drive(&mut self) -> Drive {
        // The scoped-TLS state brackets the whole stretch, not each tick:
        // set/unset plus a fresh `RefCell` per branch is pure overhead when
        // the borrowed execution is the same one throughout. The coroutines
        // borrow the execution via `STATE` between resumes.
        //
        // SAFETY: `run` is not moved or dropped while the drive lasts; the
        // loop below touches only `threads` and `used` through `self`.
        let run: &Run = unsafe { &*(&self.run as *const Run) };

        let result = STATE.set(run, || loop {
            let active = {
                let mut state = run.state.borrow_mut();
                let execution = state.execution();

                if execution.threads.is_complete() {
                    // Every loom thread has terminated, so every armed
                    // coroutine has finished its closure and parked back at
                    // its recv point, ready for the next iteration.
                    return Drive::Done(Ok(()));
                }

                execution.threads.active_id()
            };

            let thread = &mut self.threads[active.as_usize()];

            // A panic that unwound to the coroutine's root: the thread's own
            // destructors have run, against a still-consistent runtime.
            world::set_in_model(thread.in_model);
            let resumed = panic::catch_unwind(AssertUnwindSafe(|| thread.gen.resume()));
            thread.in_model = world::set_in_model(false);
            if let Err(payload) = resumed {
                std::hint::cold_path();
                return Drive::Done(Err(Failure::Panic(payload)));
            }

            thread.sp = YIELDED.get().0;

            if let Some(failure) = run.failure.borrow_mut().take() {
                std::hint::cold_path();
                return Drive::Done(Err(failure));
            }

            loop {
                // Armed coroutines park at their recv point without running
                // user code, so `arm` is safe under the live `STATE`.
                let next = run.state.borrow_mut().queued_spawn.pop_front();

                let Some(QueuedSpawn { f, stack_size }) = next else {
                    break;
                };

                assert!(self.used < self.max_threads);

                self.arm(self.used, f, stack_size);
                self.used += 1;
            }

            if run.snapshot.replace(false) {
                return Drive::Snapshot;
            }
        });

        if let Drive::Done(Err(_)) = result {
            // Suspended coroutines are never resumed, so their frames are
            // never unwound; unstarted spawns own user values too.
            for th in self.threads.drain(..) {
                std::mem::forget(th);
            }
            std::mem::forget(std::mem::take(&mut self.run.state.get_mut().queued_spawn));
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
                let gen = park_thread(stack_size);
                let (sp, top) = YIELDED.get();
                *slot = PooledThread {
                    gen,
                    stack_size,
                    sp,
                    end: mapping_end(top),
                    in_model: false,
                };
                self.rebuilt = true;
            }
            None => {
                debug_assert_eq!(index, self.threads.len(), "[loom internal bug]");
                let gen = park_thread(stack_size);
                let (sp, top) = YIELDED.get();
                self.threads.push(PooledThread {
                    gen,
                    stack_size,
                    sp,
                    end: mapping_end(top),
                    in_model: false,
                });
                self.rebuilt = true;
            }
        }

        let thread = &mut self.threads[index];
        thread.gen.set_para(Some(f));
        thread.gen.resume();
        thread.sp = YIELDED.get().0;
        // Parked at its recv point; its next resume runs the closure.
        thread.in_model = true;
    }

    fn with_state<F, R>(f: F) -> R
    where
        F: FnOnce(&mut State) -> R,
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

        if crate::rt::snapshot::is_due() {
            Self::request_snapshot();
        }

        let _runtime = world::runtime();
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
            YIELDED.set((stack_pointer(), stack_top()));
            let f: Option<Option<Box<dyn FnOnce()>>> = generator::yield_(());

            if let Some(f) = f {
                YIELDED.set((stack_pointer(), stack_top()));
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

/// Stack size of the coroutine `warm_up` runs.
const DEFAULT_WARM_STACK: usize = 64 * 1024;

/// How far below a yielding function's own frame the coroutine's saved
/// stack pointer may lie: the generator's yield and swap frames.
const STACK_MARGIN: usize = 4096;

/// An address in the calling function's frame.
#[inline(never)]
fn stack_pointer() -> usize {
    let local = 0u8;
    std::hint::black_box(&local) as *const u8 as usize
}

/// The top of the running stack: on a coroutine, its own (the generator
/// switches the TEB's stack bounds with the registers).
#[cfg(all(windows, target_arch = "x86_64"))]
fn stack_top() -> usize {
    let top: usize;
    // SAFETY: reads `NT_TIB::StackBase` of the current thread's TEB.
    unsafe { std::arch::asm!("mov {}, gs:[0x08]", out(reg) top, options(nostack, readonly, preserves_flags)) };
    top
}

/// The top of the running stack: on a coroutine, its own (the generator
/// switches the TEB's stack bounds with the registers).
#[cfg(all(windows, target_arch = "aarch64"))]
fn stack_top() -> usize {
    let top: usize;
    // SAFETY: reads `NT_TIB::StackBase` of the current thread's TEB, which
    // `x18` addresses on AArch64 Windows.
    unsafe { std::arch::asm!("ldr {}, [x18, #0x08]", out(reg) top, options(nostack, readonly, preserves_flags)) };
    top
}

#[cfg(not(all(windows, any(target_arch = "x86_64", target_arch = "aarch64"))))]
fn stack_top() -> usize {
    0
}

/// The end of the mapping holding a coroutine stack whose top is `top`, or 0
/// when unknown.
#[cfg(windows)]
fn mapping_end(top: usize) -> usize {
    #[repr(C)]
    struct MemoryBasicInformation {
        base: usize,
        allocation_base: usize,
        allocation_protect: u32,
        partition_id: u16,
        region_size: usize,
        state: u32,
        protect: u32,
        kind: u32,
    }

    #[link(name = "kernel32")]
    extern "system" {
        fn VirtualQuery(addr: usize, info: *mut MemoryBasicInformation, len: usize) -> usize;
    }

    if top == 0 {
        return 0;
    }

    let mut info = std::mem::MaybeUninit::<MemoryBasicInformation>::zeroed();
    // SAFETY: a valid out-buffer of the stated size.
    let filled = unsafe {
        VirtualQuery(top - 1, info.as_mut_ptr(), std::mem::size_of::<MemoryBasicInformation>())
    };
    assert_ne!(filled, 0, "[loom internal bug] a coroutine stack is not mapped");
    // SAFETY: filled by the call.
    let info = unsafe { info.assume_init() };
    info.base + info.region_size
}

#[cfg(not(windows))]
fn mapping_end(_top: usize) -> usize {
    0
}
