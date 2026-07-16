use crate::rt;

use std::sync::atomic::Ordering;

#[derive(Debug)]
pub(crate) struct Atomic<T> {
    /// Atomic object
    state: rt::Atomic<T>,
}

impl<T> Atomic<T>
where
    T: rt::Numeric,
{
    pub(crate) fn new(value: T, location: rt::Location) -> Atomic<T> {
        let state = rt::Atomic::new(value, location);

        Atomic { state }
    }

    #[track_caller]
    pub(crate) unsafe fn unsync_load(&self) -> T {
        self.state.unsync_load(location!())
    }

    #[track_caller]
    pub(crate) fn load(&self, order: Ordering) -> T {
        self.state.load(location!(), order)
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

    #[track_caller]
    pub(crate) fn store(&self, value: T, order: Ordering) {
        self.state.store(location!(), value, order)
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
        self.state.with_mut(location!(), f)
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
        self.state.rmw(location!(), success, failure, f)
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
