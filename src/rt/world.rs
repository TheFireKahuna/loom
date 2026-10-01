//! Execution-owned memory: the arena a worker's model state, user heap and
//! coroutine bookkeeping live in, so that a snapshot of an execution is a
//! copy of one dense range plus the live part of each coroutine stack.
//!
//! Every arena is a fixed slot of one process-wide reservation, so whether a
//! pointer is world memory is one range compare, and which worker owns it is
//! a division. The allocator's metadata (bump offset, free lists) sits at the
//! arena's base: restoring the range restores the allocator with it.
//!
//! Routing is by thread: while a thread has entered its arena, every fresh
//! allocation the global allocator [`crate::alloc::Model`] sees comes from
//! the arena. A free goes wherever the pointer came from. A free of world
//! memory by a thread that does not own the arena aborts the process — it
//! means state the model created escaped into memory other threads share,
//! which a restore would corrupt.

use std::alloc::Layout;
use std::cell::Cell;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering::Relaxed};

/// Bytes of address space per arena.
const SLOT: usize = 1 << 30;

/// Arenas the process can hold at once: every exploring thread of every
/// check running in it. A thread that finds none free replays instead.
const SLOTS: usize = 256;
const WORDS: usize = SLOTS / 64;

/// Commit granularity as the bump offset grows.
const COMMIT: usize = 1 << 20;

/// Small blocks come in power-of-two classes from 16 bytes up to this;
/// larger ones are whole pages, carved first-fit from the holes freed ones
/// leave, so a snapshot can skip a hole and a block costs only its pages.
const SMALL_MAX: usize = 32 * 1024;
const CLASSES: usize = SMALL_MAX.trailing_zeros() as usize + 1;

const PAGE: usize = 4096;

/// Holes the header tracks; a freed large block beyond these is leaked
/// until the arena is released.
const HOLES: usize = 64;

/// The address-space calls the arenas need.
#[cfg(windows)]
mod os {
    const MEM_COMMIT: u32 = 0x1000;
    const MEM_RESERVE: u32 = 0x2000;
    const MEM_DECOMMIT: u32 = 0x4000;
    const MEM_RELEASE: u32 = 0x8000;
    const PAGE_READWRITE: u32 = 0x04;

    #[link(name = "kernel32")]
    extern "system" {
        fn VirtualAlloc(addr: *mut u8, size: usize, kind: u32, protect: u32) -> *mut u8;
        fn VirtualFree(addr: *mut u8, size: usize, kind: u32) -> i32;
    }

    /// Reserve `len` bytes of address space; null on failure.
    pub(super) fn reserve(len: usize) -> *mut u8 {
        // SAFETY: a fresh reservation; nothing else maps there.
        unsafe { VirtualAlloc(std::ptr::null_mut(), len, MEM_RESERVE, PAGE_READWRITE) }
    }

    /// Release a whole reservation made by `reserve`.
    pub(super) fn release(base: *mut u8) {
        // SAFETY: the caller's reservation, never handed out.
        unsafe { VirtualFree(base, 0, MEM_RELEASE) };
    }

    /// Commit `[addr, addr + len)` of a reservation; `false` on failure.
    pub(super) fn commit(addr: *mut u8, len: usize) -> bool {
        // SAFETY: inside a reservation the caller owns.
        !unsafe { VirtualAlloc(addr, len, MEM_COMMIT, PAGE_READWRITE) }.is_null()
    }

    /// Return `[addr, addr + len)` to reserved-only.
    pub(super) fn decommit(addr: *mut u8, len: usize) {
        // SAFETY: inside a reservation the caller owns, with nothing live.
        unsafe { VirtualFree(addr, len, MEM_DECOMMIT) };
    }
}

/// Snapshots are built on Windows' address-space calls; elsewhere no arena
/// is ever acquired, and every check replays.
#[cfg(not(windows))]
mod os {
    pub(super) fn reserve(_len: usize) -> *mut u8 {
        std::ptr::null_mut()
    }

    pub(super) fn release(_base: *mut u8) {}

    pub(super) fn commit(_addr: *mut u8, _len: usize) -> bool {
        false
    }

    pub(super) fn decommit(_addr: *mut u8, _len: usize) {}
}

/// Base of the process-wide reservation, or 0 before the first arena.
static BASE: AtomicUsize = AtomicUsize::new(0);

/// Slots in use, one bit each.
static TAKEN: [AtomicU64; WORDS] = [const { AtomicU64::new(0) }; WORDS];

/// Slots abandoned after a failed execution, one bit each. Their memory stays
/// committed for good, since a failure's payload may live there and is
/// dropped on another thread; a free of it does nothing.
static ORPHANED: [AtomicU64; WORDS] = [const { AtomicU64::new(0) }; WORDS];

fn is_orphaned(slot: usize) -> bool {
    ORPHANED[slot / 64].load(Relaxed) & 1 << (slot % 64) != 0
}

/// Claim a free slot.
fn claim() -> Option<usize> {
    for (word, taken) in TAKEN.iter().enumerate() {
        let mut bits = taken.load(Relaxed);
        while bits != u64::MAX {
            let bit = (!bits).trailing_zeros();
            match taken.compare_exchange(bits, bits | 1 << bit, Relaxed, Relaxed) {
                Ok(_) => return Some(word * 64 + bit as usize),
                Err(now) => bits = now,
            }
        }
    }
    None
}

/// Set once the global allocator has routed any request, which is how a
/// check learns that the binary installed [`crate::alloc::Model`].
pub(crate) static INSTALLED: AtomicBool = AtomicBool::new(false);

/// The arena fresh allocations on this thread go to, or null.
#[thread_local]
static ROUTE: Cell<*mut Header> = Cell::new(std::ptr::null_mut());

/// The arena this thread owns, or null.
#[thread_local]
static OWNED: Cell<*mut Header> = Cell::new(std::ptr::null_mut());

#[repr(C)]
struct Header {
    /// Offset of the first byte never handed out.
    bump: usize,
    /// Bytes committed from the base.
    committed: usize,
    /// Free small blocks per size class, linked through their first word.
    free: [usize; CLASSES],
    /// Freed large blocks as `(offset, len)`, unordered, never adjacent to
    /// each other or to the bump offset.
    holes: [(usize, usize); HOLES],
    hole_count: usize,
}

/// One worker's arena.
#[derive(Debug)]
pub(crate) struct Arena {
    base: *mut u8,
    slot: usize,
}

/// The process-wide reservation's base, made on first use; `None` if it
/// cannot be made.
fn reservation() -> Option<usize> {
    let base = BASE.load(Relaxed);
    if base != 0 {
        return Some(base);
    }

    let fresh = os::reserve(SLOT * SLOTS);
    if fresh.is_null() {
        return None;
    }

    match BASE.compare_exchange(0, fresh as usize, Relaxed, Relaxed) {
        Ok(_) => Some(fresh as usize),
        Err(won) => {
            os::release(fresh);
            Some(won)
        }
    }
}

/// Whether `ptr` lies in some arena.
#[inline]
pub(crate) fn is_world(ptr: *mut u8) -> bool {
    let base = BASE.load(Relaxed);
    base != 0 && (ptr as usize).wrapping_sub(base) < SLOT * SLOTS
}

impl Arena {
    /// Take a free slot and make it this thread's arena; `None` when there
    /// is no address space or no slot left for one.
    pub(crate) fn acquire() -> Option<Arena> {
        let base = reservation()?;
        let slot = claim()?;

        let base = (base + slot * SLOT) as *mut u8;
        if !os::commit(base, COMMIT) {
            TAKEN[slot / 64].fetch_and(!(1 << (slot % 64)), Relaxed);
            return None;
        }

        let header = base.cast::<Header>();
        // SAFETY: committed, exclusively ours.
        unsafe {
            header.write(Header {
                bump: std::mem::size_of::<Header>().next_multiple_of(64),
                committed: COMMIT,
                free: [0; CLASSES],
                holes: [(0, 0); HOLES],
                hole_count: 0,
            });
        }

        assert!(OWNED.get().is_null(), "loom: a thread owns one arena at a time");
        OWNED.set(header);

        Some(Arena { base, slot })
    }

    /// Route this thread's fresh allocations here until the guard drops.
    pub(crate) fn enter(&self) -> Route {
        Route(ROUTE.replace(self.base.cast()))
    }

    /// The arena's base.
    pub(crate) fn base(&self) -> *mut u8 {
        self.base
    }

    /// The byte ranges, as `(offset, len)` from the base, that hold
    /// anything: the allocator's metadata and every block it has handed out,
    /// less the holes freed large blocks left.
    pub(crate) fn live(&self, f: impl FnMut(usize, usize)) {
        // SAFETY: the header is committed for the arena's life.
        unsafe { (*self.base.cast::<Header>()).live(f) }
    }

    /// Give the slot back. Every object in the arena must be dead or leaked.
    pub(crate) fn release(self) {
        assert!(ROUTE.get() != self.base.cast(), "loom: releasing the routed arena");
        OWNED.set(std::ptr::null_mut());
        os::decommit(self.base, SLOT);
        TAKEN[self.slot / 64].fetch_and(!(1 << (self.slot % 64)), Relaxed);
    }
}

impl Arena {
    /// Abandon the arena with everything in it.
    pub(crate) fn orphan(self) {
        assert!(ROUTE.get() != self.base.cast(), "loom: orphaning the routed arena");
        OWNED.set(std::ptr::null_mut());
        ORPHANED[self.slot / 64].fetch_or(1 << (self.slot % 64), Relaxed);
    }
}

/// Fresh allocations go to an arena while this lives.
#[derive(Debug)]
pub(crate) struct Route(*mut Header);

impl Drop for Route {
    fn drop(&mut self) {
        ROUTE.set(self.0);
    }
}

/// Whether this thread's fresh allocations go to an arena.
pub(crate) fn is_routed() -> bool {
    !ROUTE.get().is_null()
}

/// Run `f` with routing off: for runtime state that must survive a restore.
pub(crate) fn outside<R>(f: impl FnOnce() -> R) -> R {
    let _back = leave();
    f()
}

/// Turn routing off until the guard drops.
pub(crate) fn leave() -> Route {
    Route(ROUTE.replace(std::ptr::null_mut()))
}

/// A layout's block: a small class index, or a large block's length.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Block {
    Small(usize),
    Large(usize),
}

fn block(layout: Layout) -> Block {
    let size = layout.size().max(layout.align()).max(16);
    if size <= SMALL_MAX && layout.align() <= PAGE {
        Block::Small(size.next_power_of_two().trailing_zeros() as usize)
    } else {
        Block::Large(size.next_multiple_of(PAGE))
    }
}

impl Header {
    /// Advance the bump offset to `end`, committing as needed.
    unsafe fn bump_to(&mut self, end: usize) -> bool {
        if end > SLOT {
            return false;
        }
        if end > self.committed {
            let base = (self as *mut Header).cast::<u8>();
            let grow = (end - self.committed).next_multiple_of(COMMIT);
            if !os::commit(base.add(self.committed), grow) {
                return false;
            }
            self.committed += grow;
        }
        self.bump = end;
        true
    }

    fn remove_hole(&mut self, i: usize) {
        self.hole_count -= 1;
        self.holes[i] = self.holes[self.hole_count];
    }

    /// Return `[offset, offset + len)` to the free space.
    fn free_range(&mut self, mut offset: usize, mut len: usize) {
        let mut i = 0;
        while i < self.hole_count {
            let (o, l) = self.holes[i];
            if o + l == offset {
                offset = o;
                len += l;
                self.remove_hole(i);
            } else if offset + len == o {
                len += l;
                self.remove_hole(i);
            } else {
                i += 1;
            }
        }
        if offset + len == self.bump {
            self.bump = offset;
        } else if self.hole_count < HOLES {
            self.holes[self.hole_count] = (offset, len);
            self.hole_count += 1;
        }
    }

    /// The byte ranges that hold anything, in address order: the bump range
    /// less the holes.
    pub(crate) fn live(&self, mut f: impl FnMut(usize, usize)) {
        let mut holes = self.holes;
        let holes = &mut holes[..self.hole_count];
        holes.sort_unstable();
        let mut at = 0;
        for &(o, l) in holes.iter() {
            f(at, o - at);
            at = o + l;
        }
        f(at, self.bump - at);
    }
}

/// # Safety
/// `header` is a live arena's header and this thread owns it.
unsafe fn alloc_in(header: *mut Header, layout: Layout) -> *mut u8 {
    let h = &mut *header;
    let base = header.cast::<u8>();

    match block(layout) {
        Block::Small(class) => {
            let head = h.free[class];
            if head != 0 {
                let block = base.add(head);
                h.free[class] = block.cast::<usize>().read();
                return block;
            }

            let size = 1usize << class;
            let offset = h.bump.next_multiple_of(size.min(PAGE).max(layout.align()));
            if !h.bump_to(offset + size) {
                return std::ptr::null_mut();
            }
            base.add(offset)
        }
        Block::Large(len) => {
            let align = layout.align().max(PAGE);
            for i in 0..h.hole_count {
                let (o, l) = h.holes[i];
                let at = o.next_multiple_of(align);
                if at + len > o + l {
                    continue;
                }
                h.remove_hole(i);
                if at > o {
                    h.free_range(o, at - o);
                }
                if at + len < o + l {
                    h.free_range(at + len, o + l - at - len);
                }
                return base.add(at);
            }

            let offset = h.bump.next_multiple_of(align);
            let pad = offset - h.bump;
            let bump = h.bump;
            if !h.bump_to(offset + len) {
                return std::ptr::null_mut();
            }
            if pad >= PAGE {
                h.free_range(bump, pad);
            }
            base.add(offset)
        }
    }
}

/// # Safety
/// `ptr` came from [`alloc_in`] on `header` with `layout`.
unsafe fn dealloc_in(header: *mut Header, ptr: *mut u8, layout: Layout) {
    let h = &mut *header;
    let offset = ptr as usize - header as usize;
    match block(layout) {
        Block::Small(class) => {
            ptr.cast::<usize>().write(h.free[class]);
            h.free[class] = offset;
        }
        Block::Large(len) => h.free_range(offset, len),
    }
}

/// # Safety
/// `ptr` came from [`alloc_in`] on `header` with `layout`.
unsafe fn realloc_in(header: *mut Header, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
    let new_layout = Layout::from_size_align_unchecked(new_size, layout.align());
    let (old, new) = (block(layout), block(new_layout));
    if old == new {
        return ptr;
    }

    // A large block at the end of the bump range grows or shrinks in place:
    // a doubling `Vec` leaves no hole behind.
    if let (Block::Large(old_len), Block::Large(new_len)) = (old, new) {
        let h = &mut *header;
        let offset = ptr as usize - header as usize;
        if offset + old_len == h.bump && h.bump_to(offset + new_len) {
            return ptr;
        }
    }

    let fresh = alloc_in(header, new_layout);
    if !fresh.is_null() {
        std::ptr::copy_nonoverlapping(ptr, fresh, layout.size().min(new_size));
        dealloc_in(header, ptr, layout);
    }
    fresh
}

fn escaped() -> ! {
    use std::io::Write;
    let _ = std::io::stderr().write_all(
        b"loom: model memory freed by a thread that does not own it; state an execution \
          created escaped into memory shared across workers, which a snapshot restore \
          cannot rewind\n",
    );
    std::process::abort()
}

/// [`crate::alloc::Model`]'s allocation.
///
/// # Safety
/// As [`std::alloc::GlobalAlloc::alloc`].
#[inline]
pub(crate) unsafe fn alloc<A: std::alloc::GlobalAlloc>(next: &A, layout: Layout) -> *mut u8 {
    let route = ROUTE.get();
    if !route.is_null() {
        return alloc_in(route, layout);
    }
    if !INSTALLED.load(Relaxed) {
        INSTALLED.store(true, Relaxed);
    }
    next.alloc(layout)
}

/// [`crate::alloc::Model`]'s zeroed allocation.
///
/// # Safety
/// As [`std::alloc::GlobalAlloc::alloc_zeroed`].
#[inline]
pub(crate) unsafe fn alloc_zeroed<A: std::alloc::GlobalAlloc>(next: &A, layout: Layout) -> *mut u8 {
    let route = ROUTE.get();
    if !route.is_null() {
        let ptr = alloc_in(route, layout);
        if !ptr.is_null() {
            ptr.write_bytes(0, layout.size());
        }
        return ptr;
    }
    next.alloc_zeroed(layout)
}

/// [`crate::alloc::Model`]'s free.
///
/// # Safety
/// As [`std::alloc::GlobalAlloc::dealloc`].
#[inline]
pub(crate) unsafe fn dealloc<A: std::alloc::GlobalAlloc>(next: &A, ptr: *mut u8, layout: Layout) {
    if is_world(ptr) {
        let owned = OWNED.get();
        let base = BASE.load(Relaxed);
        let slot = (ptr as usize - base) / SLOT;
        if owned as usize != base + slot * SLOT {
            if is_orphaned(slot) {
                return;
            }
            escaped();
        }
        return dealloc_in(owned, ptr, layout);
    }
    next.dealloc(ptr, layout)
}

/// [`crate::alloc::Model`]'s reallocation: a block stays on the side it was
/// allocated on, so memory outside the world never moves into it.
///
/// # Safety
/// As [`std::alloc::GlobalAlloc::realloc`].
#[inline]
pub(crate) unsafe fn realloc<A: std::alloc::GlobalAlloc>(
    next: &A,
    ptr: *mut u8,
    layout: Layout,
    new_size: usize,
) -> *mut u8 {
    if !is_world(ptr) {
        return next.realloc(ptr, layout, new_size);
    }

    let owned = OWNED.get();
    let base = BASE.load(Relaxed);
    let slot = (ptr as usize - base) / SLOT;
    if owned as usize != base + slot * SLOT {
        if is_orphaned(slot) {
            let fresh = next.alloc(Layout::from_size_align_unchecked(new_size, layout.align()));
            if !fresh.is_null() {
                std::ptr::copy_nonoverlapping(ptr, fresh, layout.size().min(new_size));
            }
            return fresh;
        }
        escaped();
    }
    realloc_in(owned, ptr, layout, new_size)
}
