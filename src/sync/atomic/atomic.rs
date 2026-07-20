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
    pub(crate) fn new(value: T, location: rt::Location) -> Atomic<T> {
        let state = rt::Atomic::new(value, location);

        Atomic {
            state,
            _p: PhantomData,
        }
    }

    /// `const` constructor — see [`rt::Atomic::const_new`]. `init` is the
    /// `u128` representation of the initial value; callers know the concrete
    /// type and convert with a `const`-callable cast, because
    /// `rt::Numeric::into_u128` is a trait method and cannot be one.
    ///
    /// No `location!()`: `Location::caller()` is not `const`-callable either,
    /// so a cell built here reports no creation site. The cost is confined to
    /// diagnostics — every *access* still tracks its own location.
    pub(crate) const fn const_new(init: u128) -> Atomic<T> {
        Atomic {
            state: rt::Atomic::const_new(init),
            _p: PhantomData,
        }
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
        self.try_rmw::<_, ()>(order, order, |v| Ok(f(v))).unwrap()
    }

    #[track_caller]
    fn try_rmw<F, E>(&self, success: Ordering, failure: Ordering, f: F) -> Result<T, E>
    where
        F: FnOnce(T) -> Result<T, E>,
    {
        self.state
            .rmw_masked(location!(), rt::FULL_MASK, success, failure, |num| {
                f(T::from_u128(num)).map(T::into_u128)
            })
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
    #[track_caller]
    pub(crate) fn rmw_masked<F, E>(
        &self,
        mask: T,
        success: Ordering,
        failure: Ordering,
        f: F,
    ) -> Result<T, E>
    where
        F: FnOnce(T) -> Result<T, E>,
    {
        self.state
            .rmw_masked(location!(), mask.into_u128(), success, failure, |cur| {
                f(T::from_u128(cur)).map(T::into_u128)
            })
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
        self.try_rmw(success, failure, |actual| {
            if actual == current {
                Ok(new)
            } else {
                Err(actual)
            }
        })
    }

    #[track_caller]
    pub(crate) fn fetch_update<F>(
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
            match self.compare_exchange(prev, next, set_order, fetch_order) {
                Ok(x) => return Ok(x),
                Err(next_prev) => prev = next_prev,
            }
        }
        Err(prev)
    }
}
