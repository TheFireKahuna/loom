//! Snapshots of an execution at branch points, so that the next execution
//! resumes from the deepest snapshot at or before the branch where it leaves
//! its predecessor's path, instead of replaying that prefix from the start.
//!
//! An execution's whole future is a function of its state at a branch point:
//! the worker's arena ([`super::world`]) — model state, the scheduler and its
//! coroutines' contexts, the user heap — plus the live part of each coroutine
//! stack. A snapshot copies exactly that. The exploration's record, the
//! [`Path`](super::Path), is not part of it: a restore keeps the current path
//! and rewinds only its [`Cursor`].
//!
//! A snapshot is taken at the entry of the runtime's next access after the
//! path passes its snapshot position (a spacing of branches on, or the
//! execution's own divergence branch), when the requesting model thread
//! switches to the driver and every coroutine is suspended. It is valid for any later
//! execution that diverges at or after the snapshot's position, because
//! nothing before that position depended on a branch at or beyond it.

use crate::rt::path::Cursor;
use crate::rt::world::{self, Arena};
use crate::rt::{Execution, Scheduler};

use std::cell::Cell;
use std::ptr::NonNull;
use std::sync::OnceLock;

/// The running execution passed its snapshot position.
#[thread_local]
static DUE: Cell<bool> = Cell::new(false);

pub(crate) fn set_due() {
    DUE.set(true);
}

pub(crate) fn clear_due() {
    DUE.set(false);
}

#[inline]
pub(crate) fn is_due() -> bool {
    DUE.get()
}

/// Whether snapshots can work here: the target is supported and the binary
/// installs [`crate::alloc::Model`].
pub(crate) fn available() -> bool {
    static AVAILABLE: OnceLock<bool> = OnceLock::new();

    *AVAILABLE.get_or_init(|| {
        // Any allocation outside a world marks the allocator as installed.
        drop(std::hint::black_box(Box::new(0u64)));
        cfg!(all(windows, target_arch = "x86_64"))
            && world::INSTALLED.load(std::sync::atomic::Ordering::Relaxed)
    })
}

/// Replay every resumed execution from the start as well and compare
/// (`LOOM_SNAPSHOT_CHECK`).
pub(crate) fn check() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("LOOM_SNAPSHOT_CHECK").is_some())
}

/// A 64-bit mix of `a` and `b`, for the observation digest.
pub(crate) fn mix(a: u64, b: u64) -> u64 {
    let mut x = a.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ b.rotate_left(29);
    x ^= x >> 31;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^ (x >> 27)
}

/// A worker's execution state, built in its own arena.
pub(crate) struct World {
    arena: Option<Arena>,
    inner: NonNull<Inner>,
}

pub(crate) struct Inner {
    pub(crate) execution: Execution,
    pub(crate) scheduler: Scheduler,
}

impl World {
    /// A world for this thread, or `None` when no arena is to be had.
    pub(crate) fn new(
        execution: impl FnOnce() -> Execution,
        max_threads: usize,
        stack_size: usize,
    ) -> Option<World> {
        // OS-thread state built lazily on first use: built inside the world,
        // it would point into memory a restore rewinds.
        // The generator crate installs a process-wide panic hook and an
        // overflow handler on its first coroutine, and keeps a per-thread
        // root context: start and finish one here.
        let _ = std::thread::current();
        let _ = std::io::stdout();
        Scheduler::warm_up();

        let arena = Arena::acquire()?;
        let inner = {
            let _route = arena.enter();
            Box::new(Inner {
                execution: execution(),
                scheduler: Scheduler::new_pooled(max_threads, stack_size),
            })
        };

        Some(World {
            arena: Some(arena),
            inner: NonNull::from(Box::leak(inner)),
        })
    }

    fn arena(&self) -> &Arena {
        self.arena.as_ref().expect("[loom internal bug] world without arena")
    }

    /// The execution and scheduler. Calls into either that can allocate run
    /// under [`routed`](Self::routed).
    pub(crate) fn inner(&mut self) -> &mut Inner {
        // SAFETY: allocated in `new`, freed only in `drop`.
        unsafe { self.inner.as_mut() }
    }

    /// Run `f` with this thread's fresh allocations in the world.
    pub(crate) fn routed<R>(&self, f: impl FnOnce() -> R) -> R {
        let _route = self.arena().enter();
        f()
    }

    /// Abandon the world after a failed execution: its objects own user
    /// values whose destructors must not run, and the failure's payload may
    /// live in it.
    pub(crate) fn leak(mut self) {
        let arena = self.arena.take().unwrap();
        std::mem::forget(self);
        arena.orphan();
    }
}

impl Drop for World {
    fn drop(&mut self) {
        {
            let _route = self.arena().enter();
            // SAFETY: allocated in `new`; nothing uses it after this.
            drop(unsafe { Box::from_raw(self.inner.as_ptr()) });
        }
        self.arena.take().unwrap().release();
    }
}

/// One snapshot: the arena's used range, each coroutine's live stack, and
/// the path cursor.
#[derive(Debug)]
struct Snapshot {
    cursor: Cursor,
    /// The arena's live ranges, concatenated, and each range's offset and
    /// length.
    heap: Vec<u8>,
    ranges: Vec<(usize, usize)>,
    stacks: Vec<(usize, Vec<u8>)>,
    /// Executions resumed from it.
    uses: u64,
}

/// The snapshots along a worker's current path, shallowest first.
#[derive(Debug)]
pub(crate) struct Snapshots {
    spacing: usize,
    taken: Vec<Snapshot>,
    spare: Vec<Vec<u8>>,
    spare_stacks: Vec<Vec<u8>>,
    spare_ranges: Vec<Vec<(usize, usize)>>,
}

/// Snapshot activity, for the `LOOM_STATS` report.
pub(crate) mod stats {
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

    pub(crate) static TAKEN: AtomicU64 = AtomicU64::new(0);
    pub(crate) static RESTORED: AtomicU64 = AtomicU64::new(0);
    pub(crate) static STARTED: AtomicU64 = AtomicU64::new(0);
    pub(crate) static BYTES: AtomicU64 = AtomicU64::new(0);
    pub(crate) static UNUSED: AtomicU64 = AtomicU64::new(0);
    pub(crate) static ON_TARGET: AtomicU64 = AtomicU64::new(0);
    pub(crate) static CHECKED: AtomicU64 = AtomicU64::new(0);

    pub(crate) fn add(counter: &AtomicU64, n: u64) {
        counter.fetch_add(n, Relaxed);
    }

    /// Print and clear, if anything was snapshotted.
    pub(crate) fn report(executions: u64) {
        let take = |c: &AtomicU64| c.swap(0, Relaxed);
        let (taken, restored, started) = (take(&TAKEN), take(&RESTORED), take(&STARTED));
        let bytes = take(&BYTES);
        let unused = take(&UNUSED);
        let on_target = take(&ON_TARGET);
        let checked = take(&CHECKED);
        if checked != 0 {
            eprintln!("loom snapshot: {checked} resumed executions matched their replay");
        }
        if taken == 0 {
            return;
        }
        let n = executions.max(1) as f64;
        eprintln!(
            "loom snapshot: {:.2} taken/exec ({:.0} KiB each), {:.2} restored/exec, \
             {:.3} started from the root/exec; {:.0}% dropped unused, {:.0}% at the divergence",
            taken as f64 / n,
            bytes as f64 / taken as f64 / 1024.0,
            restored as f64 / n,
            started as f64 / n,
            100.0 * unused as f64 / taken as f64,
            100.0 * on_target as f64 / taken as f64,
        );
    }
}

impl Snapshots {
    pub(crate) fn new(spacing: usize) -> Snapshots {
        Snapshots {
            spacing,
            taken: Vec::new(),
            spare: Vec::new(),
            spare_stacks: Vec::new(),
            spare_ranges: Vec::new(),
        }
    }

    pub(crate) fn spacing(&self) -> usize {
        self.spacing
    }

    fn recycle(&mut self, snapshot: Snapshot) {
        if snapshot.uses == 0 {
            stats::add(&stats::UNUSED, 1);
        }
        self.spare.push(snapshot.heap);
        self.spare_ranges.push(snapshot.ranges);
        self.spare_stacks
            .extend(snapshot.stacks.into_iter().map(|(_, bytes)| bytes));
    }

    /// Drop every snapshot.
    pub(crate) fn clear(&mut self) {
        while let Some(snapshot) = self.taken.pop() {
            self.recycle(snapshot);
        }
    }

    /// Drop the snapshots an execution diverging at `divergence` cannot use,
    /// and return the deepest one it can.
    pub(crate) fn restorable(&mut self, divergence: usize) -> Option<usize> {
        while let Some(top) = self.taken.last() {
            if top.cursor.pos() <= divergence {
                return Some(top.cursor.pos());
            }
            let top = self.taken.pop().unwrap();
            self.recycle(top);
        }
        None
    }

    fn buffer(spare: &mut Vec<Vec<u8>>, bytes: &[u8]) -> Vec<u8> {
        let mut buffer = spare.pop().unwrap_or_default();
        buffer.clear();
        buffer.extend_from_slice(bytes);
        buffer
    }

    /// Copy the world as it stands, every coroutine suspended.
    pub(crate) fn take(&mut self, world: &mut World) -> usize {
        let base = world.arena().base();
        let mut ranges = self.spare_ranges.pop().unwrap_or_default();
        ranges.clear();
        let mut heap = self.spare.pop().unwrap_or_default();
        heap.clear();
        world.arena().live(|offset, len| {
            // SAFETY: a live range is committed, and nothing runs on it.
            heap.extend_from_slice(unsafe { std::slice::from_raw_parts(base.add(offset), len) });
            ranges.push((offset, len));
        });

        let inner = world.inner();
        let live_stacks: Vec<(usize, usize)> = inner.scheduler.stacks().collect();
        let mut stacks = Vec::with_capacity(live_stacks.len());
        for (low, high) in live_stacks {
            // SAFETY: a suspended coroutine's stack between its saved stack
            // pointer (less the margin) and its top is committed.
            let bytes = unsafe { std::slice::from_raw_parts(low as *const u8, high - low) };
            stacks.push((low, Self::buffer(&mut self.spare_stacks, bytes)));
        }

        let cursor = inner.execution.path.cursor();
        let bytes = heap.len() + stacks.iter().map(|(_, b)| b.len()).sum::<usize>();
        self.taken.push(Snapshot {
            cursor,
            heap,
            ranges,
            stacks,
            uses: 0,
        });
        bytes
    }

    /// Rewind the world to the deepest snapshot, keeping the current path.
    pub(crate) fn restore(&mut self, world: &mut World) -> usize {
        let snapshot = self.taken.last_mut().expect("[loom internal bug] nothing to restore");
        snapshot.uses += 1;
        let snapshot = &*snapshot;
        let base = world.arena().base();
        let execution: *mut Execution = &mut world.inner().execution;

        // SAFETY: the path's own storage is outside the world, so the bytes
        // read here own it across the copy, which overwrites the field with a
        // stale copy that is never dropped. Every coroutine is suspended, and
        // the copies land exactly where they were taken from.
        unsafe {
            let path = std::ptr::read(&(*execution).path);
            let pruned = (*execution).pruned;

            let mut from = snapshot.heap.as_ptr();
            for &(offset, len) in &snapshot.ranges {
                std::ptr::copy_nonoverlapping(from, base.add(offset), len);
                from = from.add(len);
            }
            for (low, bytes) in &snapshot.stacks {
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), *low as *mut u8, bytes.len());
            }

            std::ptr::write(&mut (*execution).path, path);
            (*execution).pruned = pruned;
            (*execution).path.set_cursor(snapshot.cursor);
        }

        snapshot.cursor.pos()
    }
}
