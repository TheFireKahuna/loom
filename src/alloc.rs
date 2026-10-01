//! Memory allocation APIs

use crate::rt;

#[doc(no_inline)]
pub use std::alloc::Layout;

/// Allocate memory with the global allocator.
///
/// This is equivalent to the standard library's [`std::alloc::alloc`], but with
/// the addition of leak tracking for allocated objects. Loom's leak tracking
/// will not function for allocations not performed via this method.
///
/// This function forwards calls to the [`GlobalAlloc::alloc`] method
/// of the allocator registered with the `#[global_allocator]` attribute
/// if there is one, or the `std` crate’s default.
///
/// # Safety
///
/// See [`GlobalAlloc::alloc`].
///
/// [`GlobalAlloc::alloc`]: std::alloc::GlobalAlloc::alloc
#[track_caller]
pub unsafe fn alloc(layout: Layout) -> *mut u8 {
    let ptr = std::alloc::alloc(layout);
    rt::alloc(ptr, location!());
    ptr
}

/// Allocate zero-initialized memory with the global allocator.
///
/// This is equivalent to the standard library's [`std::alloc::alloc_zeroed`],
/// but with the addition of leak tracking for allocated objects. Loom's leak
/// tracking will not function for allocations not performed via this method.
///
/// This function forwards calls to the [`GlobalAlloc::alloc_zeroed`] method
/// of the allocator registered with the `#[global_allocator]` attribute
/// if there is one, or the `std` crate’s default.
///
/// # Safety
///
/// See [`GlobalAlloc::alloc_zeroed`].
///
/// [`GlobalAlloc::alloc_zeroed`]: std::alloc::GlobalAlloc::alloc_zeroed
#[track_caller]
pub unsafe fn alloc_zeroed(layout: Layout) -> *mut u8 {
    let ptr = std::alloc::alloc_zeroed(layout);
    rt::alloc(ptr, location!());
    ptr
}

/// Deallocate memory with the global allocator.
///
/// This is equivalent to the standard library's [`std::alloc::dealloc`],
/// but with the addition of leak tracking for allocated objects. Loom's leak
/// tracking may report false positives if allocations allocated with
/// [`loom::alloc::alloc`] or [`loom::alloc::alloc_zeroed`] are deallocated via
/// [`std::alloc::dealloc`] rather than by this function.
///
/// This function forwards calls to the [`GlobalAlloc::dealloc`] method
/// of the allocator registered with the `#[global_allocator]` attribute
/// if there is one, or the `std` crate’s default.
///
/// # Safety
///
/// See [`GlobalAlloc::dealloc`].
///
/// [`GlobalAlloc::dealloc`]: std::alloc::GlobalAlloc::dealloc
/// [`loom::alloc::alloc`]: crate::alloc::alloc
/// [`loom::alloc::alloc_zeroed`]: crate::alloc::alloc_zeroed
#[track_caller]
pub unsafe fn dealloc(ptr: *mut u8, layout: Layout) {
    rt::dealloc(ptr, location!());
    std::alloc::dealloc(ptr, layout)
}

/// Track allocations, detecting leaks
#[derive(Debug)]
pub struct Track<T> {
    value: T,
    /// Drop guard tracking the allocation's lifetime.
    _obj: rt::Allocation,
}

impl<T> Track<T> {
    /// Track a value for leaks
    #[track_caller]
    pub fn new(value: T) -> Track<T> {
        Track {
            value,
            _obj: rt::Allocation::new(location!()),
        }
    }

    /// Get a reference to the value
    pub fn get_ref(&self) -> &T {
        &self.value
    }

    /// Get a mutable reference to the value
    pub fn get_mut(&mut self) -> &mut T {
        &mut self.value
    }

    /// Stop tracking the value for leaks
    pub fn into_inner(self) -> T {
        self.value
    }
}

/// A global allocator that lets loom snapshot executions: install it with
/// `#[global_allocator]` over the allocator the binary would use anyway.
///
/// While a model runs, every allocation made on one of its exploring threads
/// comes from that thread's execution-owned arena, which loom copies at
/// branch points and copies back to resume a later execution from there
/// instead of replaying it from the start. Everything else passes through to
/// `A`. Without it installed, every execution replays from the start.
///
/// A model under it must not leave state it created reachable from outside
/// the model — a process-wide registry, a lazily built `static`, a
/// thread-local of the OS thread — because a restore rewinds the memory such
/// state points into. Freeing that memory from another thread aborts.
#[derive(Debug, Default)]
pub struct Model<A>(pub A);

unsafe impl<A: std::alloc::GlobalAlloc> std::alloc::GlobalAlloc for Model<A> {
    #[inline]
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        rt::world::alloc(&self.0, layout)
    }

    #[inline]
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        rt::world::alloc_zeroed(&self.0, layout)
    }

    #[inline]
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        rt::world::dealloc(&self.0, ptr, layout)
    }

    #[inline]
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        rt::world::realloc(&self.0, ptr, layout, new_size)
    }
}

/// Run `f` with its allocations outside the model's snapshotted memory: for
/// an oracle that deliberately outlives executions, such as a registry a
/// leak gate reads. A snapshot restore does not rewind what `f` builds, so
/// it must record only what stays true whichever execution resumes next, and
/// must not keep pointers into model state.
pub fn outside<R>(f: impl FnOnce() -> R) -> R {
    rt::world::outside(f)
}
