use crate::rt;

use std::marker::PhantomData;
use std::sync::atomic::Ordering;

/// The typed façade over a modelled cell.
///
/// `B` is the cell's in-memory representation. It carries no model semantics —
/// those live once in `rt::ModelOps` — only how the cell names its
/// registration, which is what decides its layout:
///
/// - [`rt::Atomic<T>`] (the default) caches a registration inline, so it is
///   wider than `T` and must be built by a constructor.
/// - [`rt::CellId`] / [`rt::CellId16`] are nothing but an identity word, so
///   they match `T`'s size and alignment and a zeroed region is a valid,
///   unregistered cell holding zero.
///
/// Everything below is numeric conversion, written once and shared by both.
#[derive(Debug)]
#[cfg_attr(feature = "zerocopy", derive(zerocopy::FromZeros))]
#[repr(transparent)]
pub(crate) struct Atomic<T, B = rt::Atomic<T>> {
    /// Atomic object
    state: B,
    _p: PhantomData<fn() -> T>,
}

// A materialized cell must be layout-identical to the integer it models, or it
// cannot be reinterpreted from a zeroed region. The façade adds only a
// `PhantomData`, so this holds exactly when the backing's does — asserted here
// because it is the façade that consumers name.
const _: () = {
    use std::mem::{align_of, size_of};

    assert!(size_of::<Atomic<u8, rt::Cell1>>() == size_of::<u8>());
    assert!(align_of::<Atomic<u8, rt::Cell1>>() == align_of::<u8>());
    assert!(size_of::<Atomic<u16, rt::Cell2>>() == size_of::<u16>());
    assert!(align_of::<Atomic<u16, rt::Cell2>>() == align_of::<u16>());
    assert!(size_of::<Atomic<u32, rt::Cell4>>() == size_of::<u32>());
    assert!(align_of::<Atomic<u32, rt::Cell4>>() == align_of::<u32>());
    assert!(size_of::<Atomic<u64, rt::Cell8>>() == size_of::<u64>());
    assert!(align_of::<Atomic<u64, rt::Cell8>>() == align_of::<u64>());
    assert!(size_of::<Atomic<u128, rt::Cell16>>() == size_of::<u128>());
    assert!(align_of::<Atomic<u128, rt::Cell16>>() == align_of::<u128>());
};

impl<T> Atomic<T>
where
    T: rt::Numeric,
{
    /// Creates a cell holding the value `init` represents — the typed façade
    /// converts, since `rt::Numeric::into_u128` is a trait method and not
    /// `const`-callable.
    ///
    /// One constructor for both contexts. At runtime it registers the cell
    /// with the execution at once, attributing the genesis store to the
    /// constructing thread so an unsynchronized publication of the cell itself
    /// is reported. In `const` evaluation there is no execution, so it defers
    /// registration to the cell's first access in each execution — the
    /// per-execution reset a `const`-initialized `static` needs. The creation
    /// site is recorded either way.
    #[track_caller]
    pub(crate) const fn new(init: u128) -> Atomic<T> {
        let created = std::panic::Location::caller();
        core::intrinsics::const_eval_select((init, created), Self::deferred, Self::eager)
    }

    /// The `const` arm of [`new`](Self::new).
    const fn deferred(init: u128, created: &'static std::panic::Location<'static>) -> Atomic<T> {
        Atomic {
            state: rt::Atomic::const_new(init, Some(created)),
            _p: PhantomData,
        }
    }

    /// The runtime arm of [`new`](Self::new).
    fn eager(init: u128, created: &'static std::panic::Location<'static>) -> Atomic<T> {
        Atomic {
            state: rt::Atomic::new(T::from_u128(init), captured(created)),
            _p: PhantomData,
        }
    }

    /// Creates a cell with registration deferred to first access whatever the
    /// context — the deferred genesis for a cell built at *runtime*, which
    /// [`new`](Self::new) would register eagerly. See [`rt::Atomic::const_new`].
    pub(crate) const fn const_new(init: u128) -> Atomic<T> {
        Atomic {
            state: rt::Atomic::const_new(init, None),
            _p: PhantomData,
        }
    }
}

impl<T> Atomic<*mut T> {
    /// The `const` arm of `AtomicPtr::new`: only a null pointer exists in
    /// `const` evaluation, and it is the one value the record can hold without
    /// the pointer-to-integer cast a `const fn` cannot perform.
    pub(crate) const fn deferred_null(
        v: *mut T,
        created: &'static std::panic::Location<'static>,
    ) -> Atomic<*mut T> {
        assert!(
            v.is_null(),
            "a `const`-built AtomicPtr must be null: no other pointer exists in const evaluation"
        );
        Atomic {
            state: rt::Atomic::const_new(0, Some(created)),
            _p: PhantomData,
        }
    }

    /// The runtime arm of `AtomicPtr::new`.
    pub(crate) fn eager_ptr(
        v: *mut T,
        created: &'static std::panic::Location<'static>,
    ) -> Atomic<*mut T> {
        Atomic {
            state: rt::Atomic::new(v, captured(created)),
            _p: PhantomData,
        }
    }
}

/// `created` as the execution records it: captured when location tracking is
/// on, disabled otherwise — what `location!()` does, for a site captured by a
/// `const fn` the macro cannot run in.
fn captured(created: &'static std::panic::Location<'static>) -> rt::Location {
    if rt::execution(|execution| execution.location) {
        rt::Location::from(created)
    } else {
        rt::Location::disabled()
    }
}

macro_rules! zeroed_ctor {
    ($($cell:ident),* $(,)?) => {$(
        impl<T> Atomic<T, rt::$cell> {
            /// The unregistered cell, which is also the all-zeroes pattern.
            pub(crate) const fn zeroed() -> Self {
                Atomic {
                    state: rt::$cell::ZEROED,
                    _p: PhantomData,
                }
            }
        }
    )*};
}

zeroed_ctor!(Cell1, Cell2, Cell4, Cell8, Cell16);

impl<T, B> Atomic<T, B>
where
    T: rt::Numeric,
    B: rt::ModelOps,
{
    /// Formats the cell as `core` formats its atomics — through `show` on the
    /// current value — but reads that value with no model effect
    /// ([`rt::ModelOps::peek`]) where `core` performs a `Relaxed` load. A load
    /// would be a modelled step and a scheduling point, so a `Debug` call in
    /// a rig would change the schedules explored. Outside a model the value is
    /// not observable, and the cell prints as `name { .. }`.
    pub(crate) fn fmt_peek(
        &self,
        f: &mut std::fmt::Formatter<'_>,
        name: &str,
        show: fn(&T, &mut std::fmt::Formatter<'_>) -> std::fmt::Result,
    ) -> std::fmt::Result {
        match self.state.peek() {
            Some(raw) => show(&T::from_u128(raw), f),
            None => f.debug_struct(name).finish_non_exhaustive(),
        }
    }

    #[track_caller]
    pub(crate) unsafe fn unsync_load(&self) -> T {
        T::from_u128(self.state.unsync_load(location!()))
    }

    #[track_caller]
    pub(crate) fn load(&self, order: Ordering) -> T {
        T::from_u128(self.state.load_masked(location!(), rt::FULL_MASK, order))
    }

    /// Sub-word load: reads only the bits under `mask` (other bits zero). The
    /// masked lane is independently coherent, so this may return a stale lane
    /// value while another lane is seen fresh — the fidelity a single welded
    /// ring cannot express (the typed lane views' `load` is the stronger,
    /// whole-cell-projection alternative). A load spanning more than one
    /// region still returns a single consistent (non-torn) snapshot.
    #[track_caller]
    pub(crate) fn load_masked(&self, mask: T, order: Ordering) -> T {
        T::from_u128(self.state.load_masked(location!(), mask.into_u128(), order))
    }

    /// Whole-cell-coherent lane load — the model of a typed lane view's
    /// `load()` (see [`rt::Atomic::load_coherent_lane`]). Reads only `mask`'s
    /// bits (others zero) as a projection coherent with the whole
    /// single-copy-atomic cell, yet DPOR-scoped to `mask` so it commutes with
    /// disjoint-lane traffic.
    #[track_caller]
    pub(crate) fn load_coherent_lane(&self, mask: T, order: Ordering) -> T {
        T::from_u128(
            self.state
                .load_coherent_lane(location!(), mask.into_u128(), order),
        )
    }

    #[track_caller]
    pub(crate) fn store(&self, value: T, order: Ordering) {
        self.state
            .store_masked(location!(), rt::FULL_MASK, value.into_u128(), order)
    }

    /// Sub-word store: writes only the bits under `mask`, leaving the rest of
    /// the cell untouched. Models an aligned lane store inside a wider
    /// single-copy-atomic cell (spec carve-out #5) — a *weak* store to that
    /// lane, independently coherent from the other lanes.
    #[track_caller]
    pub(crate) fn store_masked(&self, mask: T, value: T, order: Ordering) {
        self.state
            .store_masked(location!(), mask.into_u128(), value.into_u128(), order)
    }

    #[track_caller]
    pub(crate) fn with_mut<R>(&mut self, f: impl FnOnce(&mut T) -> R) -> R {
        self.state.with_mut(location!(), |raw| {
            let mut value = T::from_u128(*raw);
            let r = f(&mut value);
            *raw = value.into_u128();
            r
        })
    }

    /// Read-modify-write
    ///
    /// Always reads the most recent write
    #[track_caller]
    pub(crate) fn rmw<F>(&self, f: F, order: Ordering) -> T
    where
        F: FnOnce(T) -> T,
    {
        self.try_rmw::<_, ()>(order, order, None, |v| Ok(f(v)))
            .unwrap()
    }

    /// `f` succeeds exactly when the value equals `expected`, or always for
    /// `None` (`rt::ModelOps::rmw_preserving`).
    #[track_caller]
    fn try_rmw<F, E>(
        &self,
        success: Ordering,
        failure: Ordering,
        expected: Option<T>,
        f: F,
    ) -> Result<T, E>
    where
        F: FnOnce(T) -> Result<T, E>,
    {
        self.state
            .rmw_masked(
                location!(),
                rt::FULL_MASK,
                expected.map(T::into_u128),
                success,
                failure,
                |num| f(T::from_u128(num)).map(T::into_u128),
            )
            .map(T::from_u128)
    }

    #[track_caller]
    pub(crate) fn swap(&self, val: T, order: Ordering) -> T {
        self.rmw(|_| val, order)
    }

    /// Conditional read-modify-write: one modelled atomic step that applies
    /// `f` to the most recent value and either commits its `Ok` result or
    /// leaves the cell untouched on `Err`. This is the modelling primitive
    /// for *sub-word* atomics on a wider cell (mixed-size access on one
    /// 16-byte single-copy-atomic object): the caller expresses "compare only
    /// these bits / write only these bits, preserving the rest verbatim" in
    /// `f`, and the whole operation is a single linearization point exactly
    /// like the hardware sub-word op it models.
    /// Read-modify-write over only the bits under `mask` (one modelled step).
    /// `f` receives the composed current value of the masked lane(s) — the
    /// masked bits meaningful, the rest zero — and returns the new full value;
    /// only the masked bits are written. The modelling primitive for a
    /// *sub-word* RMW / masked CAS on a wider single-copy-atomic cell (spec
    /// carve-out #5): the masked lane is independently coherent from the rest.
    ///
    /// `f` succeeds exactly when the masked bits equal `expected`'s, or always
    /// for `None` (`rt::ModelOps::rmw_preserving`).
    #[track_caller]
    pub(crate) fn rmw_masked<F, E>(
        &self,
        mask: T,
        expected: Option<T>,
        success: Ordering,
        failure: Ordering,
        f: F,
    ) -> Result<T, E>
    where
        F: FnOnce(T) -> Result<T, E>,
    {
        self.state
            .rmw_masked(
                location!(),
                mask.into_u128(),
                expected.map(T::into_u128),
                success,
                failure,
                |cur| f(T::from_u128(cur)).map(T::into_u128),
            )
            .map(T::from_u128)
    }

    /// Read-modify-write that consults every bit but may change only those
    /// under `write_mask` (one modelled step). The modelling primitive for a
    /// **preserving** wide CAS: a `cmpxchg16b` that compares all sixteen bytes
    /// yet leaves some aligned lane at the value it read.
    ///
    /// `f` receives the whole current value — so the caller's compare is over
    /// every bit, exactly as the instruction's is — and its result's bits
    /// outside `write_mask` must equal the ones it was given; the commit
    /// asserts it. In exchange the preserved lane is modelled as read, not
    /// written: it gains no store and commutes with its own readers.
    ///
    /// Nothing may acquire through a preserved lane — see
    /// `rt::ModelOps::rmw_preserving`, which traps rather than let the lost
    /// edge pass unnoticed.
    ///
    /// `f` succeeds exactly when the whole value equals `expected`, or always
    /// for `None`.
    #[track_caller]
    pub(crate) fn rmw_preserving<F, E>(
        &self,
        write_mask: T,
        expected: Option<T>,
        success: Ordering,
        failure: Ordering,
        f: F,
    ) -> Result<T, E>
    where
        F: FnOnce(T) -> Result<T, E>,
    {
        self.state
            .rmw_preserving(
                location!(),
                rt::FULL_MASK,
                write_mask.into_u128(),
                expected.map(T::into_u128),
                success,
                failure,
                |cur| f(T::from_u128(cur)).map(T::into_u128),
            )
            .map(T::from_u128)
    }

    #[track_caller]
    pub(crate) fn compare_and_swap(&self, current: T, new: T, order: Ordering) -> T {
        use self::Ordering::*;

        let failure = match order {
            Relaxed | Release => Relaxed,
            Acquire | AcqRel => Acquire,
            _ => SeqCst,
        };

        match self.compare_exchange(current, new, order, failure) {
            Ok(v) => v,
            Err(v) => v,
        }
    }

    #[track_caller]
    pub(crate) fn compare_exchange(
        &self,
        current: T,
        new: T,
        success: Ordering,
        failure: Ordering,
    ) -> Result<T, T> {
        self.try_rmw(success, failure, Some(current), |actual| {
            if actual == current {
                Ok(new)
            } else {
                Err(actual)
            }
        })
    }

    /// May fail spuriously even when the value equals `current`
    /// (`rt::ModelOps::compare_exchange_weak`).
    #[track_caller]
    pub(crate) fn compare_exchange_weak(
        &self,
        current: T,
        new: T,
        success: Ordering,
        failure: Ordering,
    ) -> Result<T, T> {
        self.state
            .compare_exchange_weak(
                location!(),
                rt::FULL_MASK,
                current.into_u128(),
                success,
                failure,
                |_| new.into_u128(),
            )
            .map(T::from_u128)
            .map_err(T::from_u128)
    }

    /// [`Self::compare_exchange_weak`] over the bits under `mask`: they are
    /// compared against `current`'s and replaced by `new` of the value read.
    #[track_caller]
    pub(crate) fn compare_exchange_weak_masked(
        &self,
        mask: T,
        current: T,
        new: impl FnOnce(T) -> T,
        success: Ordering,
        failure: Ordering,
    ) -> Result<T, T> {
        self.state
            .compare_exchange_weak(
                location!(),
                mask.into_u128(),
                current.into_u128(),
                success,
                failure,
                |cur| new(T::from_u128(cur)).into_u128(),
            )
            .map(T::from_u128)
            .map_err(T::from_u128)
    }

    #[track_caller]
    pub(crate) fn try_update<F>(
        &self,
        set_order: Ordering,
        fetch_order: Ordering,
        mut f: F,
    ) -> Result<T, T>
    where
        F: FnMut(T) -> Option<T>,
    {
        let mut prev = self.load(fetch_order);
        while let Some(next) = f(prev) {
            match self.compare_exchange_weak(prev, next, set_order, fetch_order) {
                Ok(x) => return Ok(x),
                Err(next_prev) => prev = next_prev,
            }
        }
        Err(prev)
    }
}
