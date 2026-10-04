//! Snapshots of an execution at branch points, so that the next execution
//! resumes from the deepest snapshot at or before the branch where it leaves
//! its predecessor's path, instead of replaying that prefix from the start.
//!
//! An execution's whole future is a function of its state at a branch point:
//! the worker's arena ([`super::world`]) — model state, the scheduler and its
//! coroutines' contexts, the user heap — plus the live part of each coroutine
//! stack. A snapshot images exactly that, page by page, holding only the
//! pages that differ from the snapshot below it on the path and sharing the
//! rest; a restore writes back only the cache lines the world changed since.
//! The exploration's record, the [`Path`](super::Path), is not part of it: a
//! restore keeps the current path and rewinds only its [`Cursor`].
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

use std::alloc::Layout;
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
        cfg!(all(windows, any(target_arch = "x86_64", target_arch = "aarch64")))
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
        self.arena
            .as_ref()
            .expect("[loom internal bug] world without arena")
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

    /// Fail if the code under test left anything it allocated live at the end
    /// of an execution: such a block is reachable only from state that
    /// outlives the execution, and the next restore would rewind it.
    ///
    /// On failure the world is abandoned as after a failed execution: the
    /// escaped blocks' owner may still free them, which must not reach a
    /// released arena.
    pub(crate) fn check_escapes(&mut self) {
        let live = self.arena().model_live();
        if live == 0 {
            return;
        }
        self.arena.take().unwrap().orphan();
        panic!(
            "loom: {live} block(s) the model allocated outlived its execution. An \
             execution's allocations are snapshotted and rewound, so state that outlives \
             one — a `static`, a `OnceLock`, a `std::thread_local!`, a registry, a leaked \
             box — must be built inside `loom::alloc::outside`; or run with \
             LOOM_SNAPSHOT=0, which replays every execution instead",
        );
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
        // Abandoned with everything in it (`check_escapes`).
        if self.arena.is_none() {
            return;
        }
        {
            let _route = self.arena().enter();
            // SAFETY: allocated in `new`; nothing uses it after this.
            drop(unsafe { Box::from_raw(self.inner.as_ptr()) });
        }
        self.arena.take().unwrap().release();
    }
}

/// Bytes per page of a snapshot image.
const PAGE: usize = 4096;

/// Pages the pool carves from the allocator at a time.
const CHUNK: usize = 16;

/// One page of saved bytes.
#[repr(C, align(4096))]
struct Page([u8; PAGE]);

/// Marks a live page whose saved copy is not yet chosen.
const PENDING: *const Page = std::ptr::dangling();

/// Where a snapshot keeps each page of one range: a page of its own or of a
/// snapshot below it, null where the range holds nothing.
type Table = Vec<*const Page>;

/// The pages a worker's snapshots hold, carved in chunks and recycled one by
/// one; the chunks go back to the allocator with the worker.
#[derive(Debug, Default)]
struct Pool {
    chunks: Vec<NonNull<Page>>,
    free: Vec<NonNull<Page>>,
}

impl Pool {
    fn layout() -> Layout {
        Layout::from_size_align(PAGE * CHUNK, PAGE).unwrap()
    }

    fn get(&mut self) -> NonNull<Page> {
        if let Some(page) = self.free.pop() {
            return page;
        }

        // SAFETY: a nonzero size.
        let chunk = NonNull::new(unsafe { std::alloc::alloc(Self::layout()) }.cast::<Page>())
            .unwrap_or_else(|| std::alloc::handle_alloc_error(Self::layout()));
        self.chunks.push(chunk);
        // SAFETY: every index is inside the chunk.
        self.free
            .extend((1..CHUNK).map(|i| unsafe { chunk.add(i) }));
        stats::max(&stats::HELD, (self.chunks.len() * CHUNK * PAGE) as u64);
        chunk
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        for &chunk in &self.chunks {
            // SAFETY: allocated in `get` with this layout.
            unsafe { std::alloc::dealloc(chunk.as_ptr().cast(), Self::layout()) };
        }
    }
}

/// One snapshot: the path cursor, and an image of the world as page tables
/// over the arena and each live coroutine stack.
///
/// A page byte-equal to the same page of the snapshot below this one is
/// shared with it rather than copied, so a snapshot holds only the pages its
/// stretch of the execution changed. The snapshots below outlive this one
/// (they are a stack), so a shared page outlives every table naming it.
#[derive(Debug)]
struct Snapshot {
    cursor: Cursor,
    /// Each arena zone's pages, from the zone's base.
    arena: [Table; world::ZONES],
    /// Each live coroutine stack's pages, from its top down, keyed by its top.
    stacks: Vec<(usize, Table)>,
    /// The pages this snapshot holds itself.
    own: Vec<NonNull<Page>>,
    /// Executions resumed from it.
    uses: u64,
    /// Under `LOOM_SNAPSHOT_CHECK`, a dense copy of the live ranges, which
    /// the world must equal after every restore of this snapshot.
    dense: Option<Vec<(usize, Vec<u8>)>>,
}

/// The snapshots along a worker's current path, shallowest first.
#[derive(Debug)]
pub(crate) struct Snapshots {
    spacing: usize,
    taken: Vec<Snapshot>,
    pool: Pool,
    spare_tables: Vec<Table>,
    spare_own: Vec<Vec<NonNull<Page>>>,
}

/// Snapshot activity, for the `LOOM_STATS` report.
pub(crate) mod stats {
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

    pub(crate) static TAKEN: AtomicU64 = AtomicU64::new(0);
    pub(crate) static RESTORED: AtomicU64 = AtomicU64::new(0);
    pub(crate) static STARTED: AtomicU64 = AtomicU64::new(0);
    /// Bytes snapshots copied.
    pub(crate) static BYTES: AtomicU64 = AtomicU64::new(0);
    /// Bytes snapshot images cover.
    pub(crate) static IMAGE: AtomicU64 = AtomicU64::new(0);
    /// Bytes restores compared, and the bytes they wrote.
    pub(crate) static SYNCED: AtomicU64 = AtomicU64::new(0);
    pub(crate) static WRITTEN: AtomicU64 = AtomicU64::new(0);
    /// The most pool memory one worker's snapshots held.
    pub(crate) static HELD: AtomicU64 = AtomicU64::new(0);
    pub(crate) static UNUSED: AtomicU64 = AtomicU64::new(0);
    pub(crate) static ON_TARGET: AtomicU64 = AtomicU64::new(0);
    pub(crate) static CHECKED: AtomicU64 = AtomicU64::new(0);

    pub(crate) fn add(counter: &AtomicU64, n: u64) {
        counter.fetch_add(n, Relaxed);
    }

    pub(crate) fn max(counter: &AtomicU64, n: u64) {
        counter.fetch_max(n, Relaxed);
    }

    /// Print and clear, if anything was snapshotted.
    pub(crate) fn report(executions: u64) {
        let take = |c: &AtomicU64| c.swap(0, Relaxed);
        let (taken, restored, started) = (take(&TAKEN), take(&RESTORED), take(&STARTED));
        let (bytes, image, synced, written) =
            (take(&BYTES), take(&IMAGE), take(&SYNCED), take(&WRITTEN));
        let held = take(&HELD);
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
        let kib = |b: u64, of: u64| b as f64 / of.max(1) as f64 / 1024.0;
        eprintln!(
            "loom snapshot: {:.2} taken/exec ({:.0} KiB image, {:.0} KiB copied), \
             {:.2} restored/exec ({:.0} KiB compared, {:.0} KiB written), \
             {:.3} started from the root/exec; {:.0}% dropped unused, {:.0}% at the divergence; \
             at most {:.0} KiB held by one worker",
            taken as f64 / n,
            kib(image, taken),
            kib(bytes, taken),
            restored as f64 / n,
            kib(synced, restored),
            kib(written, restored),
            started as f64 / n,
            100.0 * unused as f64 / taken as f64,
            100.0 * on_target as f64 / taken as f64,
            kib(held, 1),
        );
    }
}

/// Point every pending entry of `table` at a page byte-equal to the live page
/// `at` gives for its index: `parent`'s same page when it is, else a fresh
/// copy pushed to `own`. Returns the pages copied.
///
/// # Safety
/// Every pending entry's live page is committed and nothing writes it, and
/// `parent`'s entries are live pages of the pool.
unsafe fn capture(
    table: &mut Table,
    parent: Option<&Table>,
    at: impl Fn(usize) -> *const Page,
    pool: &mut Pool,
    own: &mut Vec<NonNull<Page>>,
) -> usize {
    let mut copied = 0;
    for (i, entry) in table.iter_mut().enumerate() {
        if *entry != PENDING {
            continue;
        }
        let live = at(i);
        let shared = parent
            .and_then(|parent| parent.get(i).copied())
            .filter(|page| !page.is_null() && (**page).0 == (*live).0);
        *entry = match shared {
            Some(page) => page,
            None => {
                let page = pool.get();
                std::ptr::copy_nonoverlapping(live, page.as_ptr(), 1);
                own.push(page);
                copied += 1;
                page.as_ptr()
            }
        };
    }
    copied
}

/// Make the page at `dst` equal to `src`, writing only the cache lines that
/// differ. Returns the lines written.
///
/// # Safety
/// Both are committed pages, and nothing else accesses `dst`.
unsafe fn sync(dst: *mut Page, src: *const Page) -> usize {
    let (dst, src) = (dst.cast::<[u64; 8]>(), src.cast::<[u64; 8]>());
    let mut written = 0;
    for line in 0..PAGE / 64 {
        let want = src.add(line).read();
        if dst.add(line).read() != want {
            dst.add(line).write(want);
            written += 1;
        }
    }
    written
}

/// The page of a stack whose top is `top`, counting down from it.
fn stack_page(top: usize, i: usize) -> *mut Page {
    (top - (i + 1) * PAGE) as *mut Page
}

impl Snapshots {
    pub(crate) fn new(spacing: usize) -> Snapshots {
        Snapshots {
            spacing,
            taken: Vec::new(),
            pool: Pool::default(),
            spare_tables: Vec::new(),
            spare_own: Vec::new(),
        }
    }

    pub(crate) fn spacing(&self) -> usize {
        self.spacing
    }

    fn recycle(&mut self, snapshot: Snapshot) {
        if snapshot.uses == 0 {
            stats::add(&stats::UNUSED, 1);
        }
        let mut own = snapshot.own;
        self.pool.free.extend(own.drain(..));
        self.spare_own.push(own);
        self.spare_tables.extend(snapshot.arena);
        self.spare_tables
            .extend(snapshot.stacks.into_iter().map(|(_, table)| table));
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

    fn table(&mut self, len: usize) -> Table {
        let mut table = self.spare_tables.pop().unwrap_or_default();
        table.clear();
        table.resize(len, std::ptr::null());
        table
    }

    /// Image the world as it stands, every coroutine suspended, sharing each
    /// page the snapshot below holds unchanged. Returns the bytes copied.
    pub(crate) fn take(&mut self, world: &mut World) -> usize {
        // The image's own bookkeeping must survive restores.
        world::outside(|| self.take_outside(world))
    }

    fn take_outside(&mut self, world: &mut World) -> usize {
        // Each zone's pages that hold anything: every page a live range
        // touches. A page wholly inside a hole holds nothing.
        let mut arena: [Table; world::ZONES] = std::array::from_fn(|_| self.table(0));
        for (zone, table) in arena.iter_mut().enumerate() {
            world.arena().live(zone, |offset, len| {
                if len == 0 {
                    return;
                }
                let (first, last) = (offset / PAGE, (offset + len - 1) / PAGE);
                if table.len() <= last {
                    table.resize(last + 1, std::ptr::null());
                }
                table[first..=last].fill(PENDING);
            });
        }

        let live_stacks: Vec<(usize, usize)> = world.inner().scheduler.stacks().collect();
        let mut stacks = Vec::with_capacity(live_stacks.len());
        for &(low, top) in &live_stacks {
            assert!(
                top % PAGE == 0,
                "[loom internal bug] a stack top off a page boundary"
            );
            let mut table = self.table((top - (low & !(PAGE - 1))) / PAGE);
            table.fill(PENDING);
            stacks.push((top, table));
        }

        let dense = check().then(|| dense(world, &live_stacks));

        let mut own = self.spare_own.pop().unwrap_or_default();
        let parent = self.taken.last();
        let mut copied = 0;
        // SAFETY: the live ranges are committed, a suspended coroutine's
        // stack from the margin below its saved stack pointer to its top is
        // committed, and nothing runs. The parent's pages are the pool's
        // until it is recycled, which is after this snapshot is.
        unsafe {
            for (zone, table) in arena.iter_mut().enumerate() {
                let base = world.arena().zone_base(zone);
                copied += capture(
                    table,
                    parent.map(|parent| &parent.arena[zone]),
                    |i| base.add(i * PAGE).cast(),
                    &mut self.pool,
                    &mut own,
                );
            }
            for (top, table) in &mut stacks {
                let top = *top;
                let parent = parent.and_then(|parent| {
                    parent
                        .stacks
                        .iter()
                        .find(|(t, _)| *t == top)
                        .map(|(_, table)| table)
                });
                copied += capture(
                    table,
                    parent,
                    |i| stack_page(top, i),
                    &mut self.pool,
                    &mut own,
                );
            }
        }

        let image = arena
            .iter()
            .flatten()
            .chain(stacks.iter().flat_map(|(_, t)| t))
            .filter(|p| !p.is_null())
            .count();
        stats::add(&stats::IMAGE, (image * PAGE) as u64);

        let cursor = world.inner().execution.path.cursor();
        self.taken.push(Snapshot {
            cursor,
            arena,
            stacks,
            own,
            uses: 0,
            dense,
        });
        copied * PAGE
    }

    /// Rewind the world to the deepest snapshot, keeping the current path.
    pub(crate) fn restore(&mut self, world: &mut World) -> usize {
        let snapshot = self
            .taken
            .last_mut()
            .expect("[loom internal bug] nothing to restore");
        snapshot.uses += 1;
        let snapshot = &*snapshot;
        let bases: [*mut u8; world::ZONES] = std::array::from_fn(|zone| world.arena().zone_base(zone));
        let execution: *mut Execution = &mut world.inner().execution;

        let mut synced = 0;
        let mut written = 0;
        // SAFETY: the path's own storage is outside the world, so the bytes
        // read here own it across the copy, which overwrites the field with a
        // stale copy that is never dropped. Every coroutine is suspended, and
        // every page the image names was committed when it was taken and is
        // never decommitted while the arena lives.
        unsafe {
            let path = std::ptr::read(&(*execution).path);
            let pruned = (*execution).pruned;

            for (table, base) in snapshot.arena.iter().zip(bases) {
                for (i, &page) in table.iter().enumerate() {
                    if !page.is_null() {
                        written += sync(base.add(i * PAGE).cast(), page);
                        synced += 1;
                    }
                }
            }
            for (top, table) in &snapshot.stacks {
                for (i, &page) in table.iter().enumerate() {
                    written += sync(stack_page(*top, i), page);
                    synced += 1;
                }
            }

            if let Some(dense) = &snapshot.dense {
                for (at, bytes) in dense {
                    assert!(
                        std::slice::from_raw_parts(*at as *const u8, bytes.len()) == &bytes[..],
                        "loom: a snapshot restore left the world differing from a dense copy \
                         taken with the snapshot, at {at:#x}",
                    );
                }
            }

            std::ptr::write(&mut (*execution).path, path);
            (*execution).pruned = pruned;
            (*execution).path.set_cursor(&snapshot.cursor);
        }

        stats::add(&stats::SYNCED, (synced * PAGE) as u64);
        stats::add(&stats::WRITTEN, (written * 64) as u64);
        snapshot.cursor.pos()
    }
}

/// The world's live ranges and live stacks, copied byte for byte.
fn dense(world: &World, stacks: &[(usize, usize)]) -> Vec<(usize, Vec<u8>)> {
    let mut ranges = Vec::new();
    for zone in 0..world::ZONES {
        let base = world.arena().zone_base(zone);
        // SAFETY: as `take`'s.
        world.arena().live(zone, |offset, len| {
            let at = base as usize + offset;
            ranges.push((
                at,
                unsafe { std::slice::from_raw_parts(at as *const u8, len) }.to_vec(),
            ));
        });
    }
    for &(low, top) in stacks {
        ranges.push((
            low,
            unsafe { std::slice::from_raw_parts(low as *const u8, top - low) }.to_vec(),
        ));
    }
    ranges
}
