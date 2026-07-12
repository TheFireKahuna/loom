//! An atomic cell
//!
//! See the CDSChecker paper for detailed explanation.
//!
//! # Modification order implications (figure 7)
//!
//! - Read-Read Coherence:
//!
//!   On `load`, all stores are iterated, finding stores that were read by
//!   actions in the current thread's causality. These loads happen-before the
//!   current load. The `modification_order` of these happen-before loads are
//!   joined into the current load's `modification_order`.
//!
//! - Write-Read Coherence:
//!
//!   On `load`, all stores are iterated, finding stores that happens-before the
//!   current thread's causality. The `modification_order` of these stores are
//!   joined into the current load's `modification_order`.
//!
//! - Read-Write Coherence:
//!
//!   On `store`, find all existing stores that were read in the current
//!   thread's causality. Join these stores' `modification_order` into the new
//!   store's modification order.
//!
//! - Write-Write Coherence:
//!
//!   The `modification_order` is initialized to the thread's causality. Any
//!   store that happened in the thread causality will be earlier in the
//!   modification order.
//!
//! # Sequential consistency
//!
//! `SeqCst` operations — accesses *and* fences — are ordered by a single total
//! order S (C++20 [atomics.order]). S is modelled as the order in which those
//! operations commit in the current schedule, tracked by an integer position
//! handed out per SC store and per SC fence (`thread::Set::next_sc_pos`); a
//! store also carries its position in `Store::sc_rank`. Because permutation
//! testing explores every schedule, every S consistent with happens-before is
//! explored. The rules below key on the cell (per-location) and never create
//! happens-before between SeqCst operations.
//!
//! A store is *SC-ranked* (`Store::sc_rank` is `Some`) when it participates in
//! S as a write. That is so in two ways:
//!
//! - it was itself a `SeqCst` store, ranked at its own commit; or
//! - it was sequenced before a `SeqCst` fence that has since executed, which
//!   promotes it into S at the fence's position (`promote_sc_writes`; C++20
//!   [atomics.order] p5/p7). Promotion is what lets a `fence(SeqCst)` in one
//!   thread order against an SC *access* in another.
//!
//! Rules:
//!
//! - Seq-cst/MO Consistency:
//!
//!   The SC-ranked stores to a cell are totally ordered by S, and modification
//!   order must agree with S. On `store`, a SeqCst store joins the
//!   `modification_order` of every SC-ranked store already committed to the
//!   cell, so they form an mo-chain in commit order (`State::store`).
//!
//! - Seq-cst Read Restriction:
//!
//!   A load obeys the SC read rule within a *scope* — how far into S it must
//!   respect. A `SeqCst` load's scope is all of S; a load sequenced after a
//!   `SeqCst` fence has the fence's position as its scope (the fence-read rules
//!   p4/p6); any other load is unconstrained. The load may not return a store
//!   that is modification-order-before an SC-ranked store to the cell whose
//!   rank lies within scope (`match_load_to_stores`). Only genuine `mo_before`
//!   edges gate the exclusion, so a store promoted late (mo-early yet given a
//!   high rank, e.g. a cell's initial store under a `SeqCst` fence in its
//!   creating thread) can never masquerade as a newer witness. Enforcing this
//!   forbids the store-buffering, IRIW and read-write-causality outcomes that
//!   plain acquire/release permits — spelled with SC accesses, SC fences, or a
//!   mix of the two — while mo-incomparable concurrent stores stay readable so
//!   no legal weak behavior is lost.
//!
//!   The read half of a SeqCst RMW (and a failed SeqCst compare-exchange)
//!   needs no check: it reads an mo-maximal store, which is never mo-before any
//!   other store to the cell.
//!
//! `fence(SeqCst)` participates in two cooperating mechanisms. Its position in
//! S (above) supplies the access↔fence interaction — promotion and the
//! fence-read scope. A separate causality frontier
//! (`thread::Set::seq_cst_fence`) supplies fence↔fence ordering (p6/p7 among
//! fences) by propagating happens-before. The two are independent: the S rules
//! are read restrictions that create no happens-before, while the frontier
//! carries the happens-before that ordered fence pairs require.
//!
//! - RMW/MO Consistency: Subsumed by Write-Write Coherence?
//!
//! - RMW Atomicity:
//!
//!   An RMW's write is *immediately* after the store it read in modification
//!   order — no other store may sit between them. Two obligations follow:
//!
//!   1. The RMW may only read a modification-order-maximal store
//!      (`match_rmw_to_stores`).
//!   2. Any store that is modification-order-after the RMW's read store is
//!      modification-order-after the RMW's write. Each RMW write records the
//!      identity of the store it read (`Store::rmw_read`), and every time a
//!      store's `modification_order` is (re)computed — at creation and on
//!      every load-coherence join — `close_rmw_atomicity` runs the
//!      implication to fixpoint. Without this closure, a plain store racing
//!      a committed RMW lands mo-*incomparable* to the RMW's write, and the
//!      write survives as a permanently readable "zombie" candidate that no
//!      real machine can still expose (C11 forces it mo-before the racing
//!      store).
//!
//! # Modification-order representation
//!
//! `Store::modification_order` is a join of genuine causality snapshots:
//! the storing thread's causality, the vectors of stores known mo-before it,
//! and (for a `SeqCst` store) the vectors of the SC-ranked stores already
//! committed to the same cell (SC/mo consistency). Because vector clocks are
//! transitively closed, "store `a` is known mo-before
//! store `b`" is decided by the single-lane marker test
//! `b.modification_order[a.creator] >= a.tick` (`mo_before`): the lane can
//! only reach `a`'s creation tick by having joined a snapshot that causally
//! contains `a`'s creation, and every such join site corresponds to a real
//! C11 mo edge. This subsumes the old whole-vector dominance comparison
//! (`mo_a < mo_b` implies the marker fires, never the reverse) and
//! additionally catches causality-only ancestry — a store whose *creator*
//! transitively heard of `a` without ever reading it — which dominance
//! missed whenever `a`'s vector had grown through coherence joins the
//! descendant never saw.
//!
//! # Fence modification order implications (figure 9)
//!
//! - SC Fences Restrict RF:
//! - SC Fences Restrict RF (Collapsed Store):
//! - SC Fences Restrict RF (Collapsed Load):
//! - SC Fences Impose MO:
//! - SC Fences Impose MO (Collapsed 1st Store):
//! - SC Fences Impose MO (Collapsed 2st Store):
//!
//!
//! # Fence Synchronization implications (figure 10)
//!
//! - Fence Synchronization
//! - Fence Synchronization (Collapsed Store)
//! - Fence Synchronization (Collapsed Load)

use crate::rt::execution::Execution;
use crate::rt::location::{self, Location, LocationSet};
use crate::rt::object;
use crate::rt::{
    self, thread, Access, Numeric, Synchronize, VersionVec, MAX_ATOMIC_HISTORY, MAX_THREADS,
};

use std::cmp;
use std::marker::PhantomData;
use std::sync::atomic::Ordering;
use std::u16;

use tracing::trace;

#[derive(Debug)]
pub(crate) struct Atomic<T> {
    state: object::Ref<State>,
    _p: PhantomData<fn() -> T>,
}

#[derive(Debug)]
pub(super) struct State {
    /// Where the atomic was created
    created_location: Location,

    /// Transitive closure of all atomic loads from the cell.
    loaded_at: VersionVec,

    /// Location for the *last* time a thread atomically loaded from the cell.
    loaded_locations: LocationSet,

    /// Transitive closure of all **unsynchronized** loads from the cell.
    unsync_loaded_at: VersionVec,

    /// Location for the *last* time a thread read **synchronized** from the cell.
    unsync_loaded_locations: LocationSet,

    /// Transitive closure of all atomic stores to the cell.
    stored_at: VersionVec,

    /// Location for the *last* time a thread atomically stored to the cell.
    stored_locations: LocationSet,

    /// Version of the most recent **unsynchronized** mutable access to the
    /// cell.
    ///
    /// This includes the initialization of the cell as well as any calls to
    /// `get_mut`.
    unsync_mut_at: VersionVec,

    /// Location for the *last* time a thread `with_mut` from the cell.
    unsync_mut_locations: LocationSet,

    /// `true` when in a `with_mut` closure. If this is set, there can be no
    /// access to the cell.
    is_mutating: bool,

    /// Last time each thread accessed the atomic. This tracks the dependent
    /// accesses for the DPOR algorithm.
    ///
    /// Per-thread, not a single shared slot (fork fix): with one slot, a
    /// thread's own access overwrites the record of every peer's access — a
    /// spawned thread whose prefix is `load; compare_exchange` records its
    /// own load as the cell's last access, so its CAS is only ever checked
    /// against that (trivially happens-before) and the conflict with a
    /// peer's earlier plain load is never seen. The reorder DPOR owes for
    /// that conflict (the child prefix scheduled ahead of the peer's load)
    /// was then silently never explored.
    ///
    /// Boxed (with `last_non_load_access` and `stores`) to keep `State`
    /// small: the object store's `Entry` enum is sized by its largest
    /// variant, so an inline `State` (~1 KB) taxes every object slot's
    /// insert/clear/memmove with its full width. The indirection is paid
    /// once per atomic per iteration; the slot traffic is per operation.
    last_access: Box<[Option<Access>; MAX_THREADS]>,

    /// Last time each thread accessed the atomic with a store or rmw
    /// operation.
    last_non_load_access: Box<[Option<Access>; MAX_THREADS]>,

    /// Currently tracked stored values. This is the `MAX_ATOMIC_HISTORY` most
    /// recent stores to the atomic cell in loom execution order.
    stores: Box<[Store; MAX_ATOMIC_HISTORY]>,

    /// The total number of stores to the cell.
    cnt: u16,
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub(super) enum Action {
    /// Atomic load
    Load,

    /// Atomic store
    Store,

    /// Atomic read-modify-write
    Rmw,
}

#[derive(Debug)]
struct Store {
    /// The stored value. All atomic types can be converted to `u128`.
    value: u128,

    /// The causality of the thread when it stores the value.
    happens_before: VersionVec,

    /// Tracks the modification order: a join of the causality snapshots of
    /// this store and every store known to be modification-order-before it.
    /// Order is queried through the single-lane marker test (`mo_before`) —
    /// see the module docs.
    modification_order: VersionVec,

    /// Absolute store count at creation (`State::cnt`); identifies the store
    /// across ring eviction within one execution.
    id: u16,

    /// Lane index of the storing thread. `(creator, tick())` is the store's
    /// unique creation stamp — the coordinate the marker test reads.
    creator: usize,

    /// When this store is the write half of an RMW, the creation stamp of
    /// the store the RMW read. Snapshotted (not a slot index) so the
    /// atomicity closure keeps working after the read store is evicted from
    /// the ring.
    rmw_read: Option<RmwRead>,

    /// Manages causality transfers between threads
    sync: Synchronize,

    /// Tracks when each thread first saw value
    first_seen: FirstSeen,

    /// This store's rank in the SC total order S, or `None` if it does not
    /// participate in S. A store is SC-ranked either because it was itself a
    /// `SeqCst` store (ranked at its own commit, `State::store`) or because it
    /// was sequenced before a `SeqCst` fence that has since executed and
    /// promoted it (ranked at the fence's position, `promote_sc_writes`;
    /// C++20 [atomics.order] p5/p7). Two SC-ranked stores to this cell are
    /// ordered in S by their positions, and S agrees with modification order,
    /// so `a.sc_rank < b.sc_rank` implies `a` is mo-before `b` — the fact the
    /// SC read rule uses to exclude superseded writes without needing a
    /// materialized mo edge (see `match_load_to_stores`).
    sc_rank: Option<u32>,
}

/// Creation stamp of the store an RMW write read — the persistent record of
/// the "nothing may split this pair" obligation.
#[derive(Debug, Copy, Clone)]
struct RmwRead {
    /// `Store::id` of the read store, to exclude the read store itself from
    /// the closure (it is mo-*before* its own RMW successor).
    read_id: u16,

    /// `Store::creator` of the read store.
    creator: usize,

    /// `Store::tick()` of the read store.
    tick: u16,
}

impl Store {
    /// The creating thread's clock component at creation — with `creator`,
    /// the store's unique creation stamp.
    fn tick(&self) -> u16 {
        self.happens_before.lane(self.creator)
    }
}

/// True when store `a` is known modification-order-before store `b`.
///
/// Single-lane marker test: `b`'s modification order joins only genuine
/// causality snapshots, each joined along a real mo edge, so its `a.creator`
/// lane reaches `a`'s creation tick iff some mo-ancestor of `b` (or `b`'s own
/// creation) causally contains `a`'s creation — a real C11 mo edge in every
/// case. Strictly more complete than whole-vector dominance and immune to
/// the "vectors grew apart after the join" imprecision (see module docs).
fn mo_before(a: &Store, b: &Store) -> bool {
    a.id != b.id && b.modification_order.lane(a.creator) >= a.tick()
}

#[derive(Debug)]
struct FirstSeen([u16; MAX_THREADS]);

/// Implements atomic fence behavior
pub(crate) fn fence(ordering: Ordering) {
    rt::synchronize(|execution| match ordering {
        Ordering::Acquire => fence_acq(execution),
        Ordering::Release => fence_rel(execution),
        Ordering::AcqRel => fence_acqrel(execution),
        Ordering::SeqCst => fence_seqcst(execution),
        Ordering::Relaxed => panic!("there is no such thing as a relaxed fence"),
        order => unimplemented!("unimplemented ordering {:?}", order),
    });
}

fn fence_acq(execution: &mut Execution) {
    // Find all stores for all atomic objects and, if they have been read by
    // the current thread, establish an acquire synchronization.
    //
    // "Read by the current thread" is literal (C11 fence synchronization:
    // some atomic operation *sequenced before this fence* must read the
    // store) — a store that is merely in the thread's causality because some
    // OTHER thread's relaxed read of it happens-before us does not qualify;
    // syncing with those would over-approximate and hide real reorderings.
    // A store this thread itself created also touches `first_seen`, which is
    // harmless here: its release view is already contained in (or, for an
    // RMW, legitimately acquired through) this thread's causality.
    for state in execution.objects.iter_mut::<State>() {
        // Iterate all the stores
        for store in state.stores_mut() {
            if !store.first_seen.is_touched_by(execution.threads.active_id()) {
                continue;
            }

            store
                .sync
                .sync_load(&mut execution.threads, Ordering::Acquire);
        }
    }
}

fn fence_rel(execution: &mut Execution) {
    // take snapshot of cur view and record as rel view
    let active = execution.threads.active_mut();
    active.released = active.causality;
}

fn fence_acqrel(execution: &mut Execution) {
    fence_acq(execution);
    fence_rel(execution);
}

fn fence_seqcst(execution: &mut Execution) {
    fence_acqrel(execution);
    execution.threads.seq_cst_fence();

    // Join this fence into the single SC total order S. It takes the next
    // position and promotes every store sequenced before it — i.e. every store
    // this thread created — into S at that position (C++20 [atomics.order]
    // p5/p7). This is what lets a fence in one thread order against an SC
    // *access* in another: a promoted store is then an ordinary SC-ranked
    // witness for the SC read rule, and the fence's position bounds the
    // fence-read scope of later loads in this thread (p4/p6). Independent of
    // the `seq_cst_fence` causality frontier above, which handles fence↔fence.
    let pos = execution.threads.begin_sc_fence();
    let creator = execution.threads.active_id().as_usize();
    for state in execution.objects.iter_mut::<State>() {
        state.promote_sc_writes(creator, pos);
    }
}

impl<T: Numeric> Atomic<T> {
    /// Create a new, atomic cell initialized with the provided value
    pub(crate) fn new(value: T, location: Location) -> Atomic<T> {
        rt::execution(|execution| {
            let state = State::new(&mut execution.threads, value.into_u128(), location);
            let state = execution.objects.insert(state);

            trace!(?state, "Atomic::new");

            Atomic {
                state,
                _p: PhantomData,
            }
        })
    }

    /// Loads a value from the atomic cell.
    pub(crate) fn load(&self, location: Location, ordering: Ordering) -> T {
        self.branch(Action::Load, location);

        super::synchronize(|execution| {
            let state = self.state.get_mut(&mut execution.objects);

            // If necessary, generate the list of stores to permute through.
            //
            // A `SeqCst` load participates in the SC total order S; a load past
            // a `SeqCst` fence is bounded by the fence's position in S. Which
            // stores either may return is restricted inside
            // `match_load_to_stores` (keyed off `ordering` and the thread's
            // fence position) — see there. It is a read-side rule only: no
            // happens-before is created between SC operations, so nearby
            // relaxed accesses keep their full legal weak behavior.
            if execution.path.is_traversed() {
                let mut seed = [0; MAX_ATOMIC_HISTORY];

                let n = state.match_load_to_stores(&execution.threads, &mut seed[..], ordering);

                execution.path.push_load(&seed[..n]);
            }

            // Get the store to return from this load.
            let index = execution.path.branch_load();

            trace!(state = ?self.state, ?ordering, "Atomic::load");

            T::from_u128(state.load(&mut execution.threads, index, location, ordering))
        })
    }

    /// Loads a value from the atomic cell without performing synchronization
    pub(crate) fn unsync_load(&self, location: Location) -> T {
        rt::execution(|execution| {
            let state = self.state.get_mut(&mut execution.objects);

            state
                .unsync_loaded_locations
                .track(location, &execution.threads);

            // An unsync load counts as a "read" access
            state.track_unsync_load(&execution.threads);

            trace!(state = ?self.state, "Atomic::unsync_load");

            // Return the value
            let index = index(state.cnt - 1);
            T::from_u128(state.stores[index].value)
        })
    }

    /// Stores a value into the atomic cell.
    pub(crate) fn store(&self, location: Location, val: T, ordering: Ordering) {
        self.branch(Action::Store, location);

        super::synchronize(|execution| {
            let state = self.state.get_mut(&mut execution.objects);

            state.stored_locations.track(location, &execution.threads);

            // An atomic store counts as a read access to the underlying memory
            // cell.
            state.track_store(&execution.threads);

            trace!(state = ?self.state, ?ordering, "Atomic::store");

            // Do the store. A `SeqCst` store's SC/mo-consistency edges are
            // established inside `State::store`.
            state.store(
                &mut execution.threads,
                Synchronize::new(),
                val.into_u128(),
                ordering,
            );
        })
    }

    pub(crate) fn rmw<F, E>(
        &self,
        location: Location,
        success: Ordering,
        failure: Ordering,
        f: F,
    ) -> Result<T, E>
    where
        F: FnOnce(T) -> Result<T, E>,
    {
        self.branch(Action::Rmw, location);

        super::synchronize(|execution| {
            let state = self.state.get_mut(&mut execution.objects);

            // If necessary, generate the list of stores to permute through
            if execution.path.is_traversed() {
                let mut seed = [0; MAX_ATOMIC_HISTORY];

                let n = state.match_rmw_to_stores(&mut seed[..]);
                execution.path.push_load(&seed[..n]);
            }

            // Get the store to use for the read portion of the rmw operation.
            let index = execution.path.branch_load();

            trace!(state = ?self.state, ?success, ?failure, "Atomic::rmw");

            // The read half of an SC RMW needs no SC restriction: an RMW only
            // reads a modification-order-maximal store (`match_rmw_to_stores`),
            // which can never be mo-before another store to the cell, so the SC
            // read rule is satisfied automatically. The write half routes
            // through `State::store` with the `success` ordering, where the
            // store takes its SC position and SC/mo consistency is enforced.
            state
                .rmw(
                    &mut execution.threads,
                    index,
                    location,
                    success,
                    failure,
                    |num| f(T::from_u128(num)).map(T::into_u128),
                )
                .map(T::from_u128)
        })
    }

    /// Access a mutable reference to value most recently stored.
    ///
    /// `with_mut` must happen-after all stores to the cell.
    pub(crate) fn with_mut<R>(&mut self, location: Location, f: impl FnOnce(&mut T) -> R) -> R {
        let value = super::execution(|execution| {
            let state = self.state.get_mut(&mut execution.objects);

            state
                .unsync_mut_locations
                .track(location, &execution.threads);
            // Verify the mutation may happen
            state.track_unsync_mut(&execution.threads);
            state.is_mutating = true;

            trace!(state = ?self.state, "Atomic::with_mut");

            // Return the value of the most recent store
            let index = index(state.cnt - 1);
            T::from_u128(state.stores[index].value)
        });

        struct Reset<T: Numeric>(T, object::Ref<State>);

        impl<T: Numeric> Drop for Reset<T> {
            fn drop(&mut self) {
                super::execution(|execution| {
                    let state = self.1.get_mut(&mut execution.objects);

                    // Make sure the state is as expected
                    assert!(state.is_mutating);
                    state.is_mutating = false;

                    // The value may have been mutated, so it must be placed
                    // back.
                    let index = index(state.cnt - 1);
                    state.stores[index].value = T::into_u128(self.0);

                    if !std::thread::panicking() {
                        state.track_unsync_mut(&execution.threads);
                    }
                });
            }
        }

        // Unset on exit
        let mut reset = Reset(value, self.state);
        f(&mut reset.0)
    }

    fn branch(&self, action: Action, location: Location) {
        let r = self.state;
        r.branch_action(action, location);
        assert!(
            r.ref_eq(self.state),
            "Internal state mutated during branch. This is \
                usually due to a bug in the algorithm being tested writing in \
                an invalid memory location."
        );
    }
}

// ===== impl State =====

impl State {
    fn new(threads: &mut thread::Set, value: u128, location: Location) -> State {
        let mut state = State {
            created_location: location,
            loaded_at: VersionVec::new(),
            loaded_locations: LocationSet::new(),
            unsync_loaded_at: VersionVec::new(),
            unsync_loaded_locations: LocationSet::new(),
            stored_at: VersionVec::new(),
            stored_locations: LocationSet::new(),
            unsync_mut_at: VersionVec::new(),
            unsync_mut_locations: LocationSet::new(),
            is_mutating: false,
            last_access: Default::default(),
            last_non_load_access: Default::default(),
            stores: Default::default(),
            cnt: 0,
        };

        // All subsequent accesses must happen-after.
        state.track_unsync_mut(threads);

        // Store the initial thread
        //
        // The actual order shouldn't matter as operation on the atomic
        // **should** already include the thread causality resulting in the
        // creation of this atomic cell.
        //
        // This is verified using `cell`.
        state.store(threads, Synchronize::new(), value, Ordering::Release);

        state
    }

    fn load(
        &mut self,
        threads: &mut thread::Set,
        index: usize,
        location: Location,
        ordering: Ordering,
    ) -> u128 {
        self.loaded_locations.track(location, threads);
        // Validate memory safety
        self.track_load(threads);

        // Apply coherence rules
        self.apply_load_coherence(threads, index);

        let store = &mut self.stores[index];

        store.first_seen.touch(threads);
        store.sync.sync_load(threads, ordering);
        store.value
    }

    fn store(
        &mut self,
        threads: &mut thread::Set,
        mut sync: Synchronize,
        value: u128,
        ordering: Ordering,
    ) {
        let index = index(self.cnt);
        let live = self.live_stores();
        let id = self.cnt;
        let creator = threads.active_id().as_usize();

        // Increment the count
        self.cnt += 1;

        // The modification order is initialized to the thread's current
        // causality. All reads / writes that happen before this store are
        // ordered before the store.
        let happens_before = threads.active().causality;

        // Starting with the thread's causality covers WRITE-WRITE coherence
        let mut modification_order = happens_before;

        // Whether this store participates in the modelled SC total order S,
        // and, if so, its position in S (= commit order among SC operations).
        let sc = is_seq_cst(ordering);
        let sc_rank = if sc { Some(threads.next_sc_pos()) } else { None };

        // Apply coherence rules
        for i in 0..live {
            let store_i = &self.stores[i];

            // READ-WRITE coherence: stores this thread has read are
            // mo-before the new store.
            //
            // WRITE-WRITE coherence: stores in this thread's causality are
            // mo-before it too. Their creation stamps are already inside
            // `happens_before`, but their vectors carry mo edges (coherence
            // and RMW-atomicity joins) the raw causality does not — joining
            // them keeps known ancestry transitive.
            //
            // SC/MO consistency: the SC-ranked stores to a cell are totally
            // ordered by S, and mo must agree with S. Every SC-ranked store
            // already committed — whether an SC store or one a fence promoted
            // (`promote_sc_writes`) — is therefore mo-before this SeqCst store.
            // This is a per-location, S-only mo edge — joined into
            // `modification_order`, never into causality (the S edge orders the
            // writes without manufacturing happens-before). Timing is exact:
            // only stores already SC-ranked when this store commits are joined,
            // matching that only they precede it in S.
            if store_i.first_seen.is_seen_by_current(threads)
                || happens_before.lane(store_i.creator) >= store_i.tick()
                || (sc && store_i.sc_rank.is_some())
            {
                let mo = store_i.modification_order;
                modification_order.join(&mo);
            }
        }

        // RMW Atomicity: everything mo-after an RMW's read store is mo-after
        // the RMW's write.
        self.close_rmw_atomicity(&mut modification_order, id);

        sync.sync_store(threads, ordering);

        let mut first_seen = FirstSeen::new();
        first_seen.touch(threads);

        // Track the store
        self.stores[index] = Store {
            value,
            happens_before,
            modification_order,
            id,
            creator,
            rmw_read: None,
            sync,
            first_seen,
            sc_rank,
        };
    }

    fn rmw<E>(
        &mut self,
        threads: &mut thread::Set,
        index: usize,
        location: Location,
        success: Ordering,
        failure: Ordering,
        f: impl FnOnce(u128) -> Result<u128, E>,
    ) -> Result<u128, E> {
        self.loaded_locations.track(location, threads);

        // Track the load is happening in order to ensure correct
        // synchronization to the underlying cell.
        self.track_load(threads);

        // Apply coherence rules.
        self.apply_load_coherence(threads, index);

        self.stores[index].first_seen.touch(threads);

        let prev = self.stores[index].value;

        match f(prev) {
            Ok(next) => {
                self.stored_locations.track(location, threads);
                // Track a store operation happened
                self.track_store(threads);

                // Perform load synchronization using the `success` ordering.
                self.stores[index].sync.sync_load(threads, success);

                // Capture the read store's creation stamp *before* the write
                // half runs: if the ring is full and the read store is the
                // oldest live store, the new store lands in its slot.
                let rmw_read = RmwRead {
                    read_id: self.stores[index].id,
                    creator: self.stores[index].creator,
                    tick: self.stores[index].tick(),
                };

                // Store the new value, initializing with the `sync` value from
                // the load. This is our (hacky) way to establish a release
                // sequence.
                let sync = self.stores[index].sync;
                self.store(threads, sync, next, success);

                // RMW Atomicity: mark the write half with what it read, so
                // every future store mo-after the read store gets closed to
                // mo-after this write (`close_rmw_atomicity`).
                self.stores[self::index(self.cnt - 1)].rmw_read = Some(rmw_read);

                Ok(prev)
            }
            Err(e) => {
                // A failed compare-exchange is a load. With `SeqCst` failure
                // ordering it is an SC read, but it reads the store chosen by
                // `match_rmw_to_stores` (modification-order-maximal), which is
                // never mo-before another store to the cell, so the SC read
                // rule holds with no extra work.
                self.stores[index].sync.sync_load(threads, failure);
                Err(e)
            }
        }
    }

    fn apply_load_coherence(&mut self, threads: &mut thread::Set, index: usize) {
        for i in 0..self.live_stores() {
            // Skip if the is current.
            if index == i {
                continue;
            }

            // READ-READ coherence
            if self.stores[i].first_seen.is_seen_by_current(threads) {
                let mo = self.stores[i].modification_order;
                self.stores[index].modification_order.join(&mo);
            }

            // WRITE-READ coherence
            if self.stores[i].happens_before < threads.active().causality {
                let mo = self.stores[i].modification_order;
                self.stores[index].modification_order.join(&mo);
            }
        }

        // RMW Atomicity: the joins above may have taught the read store that
        // it is mo-after some RMW's read store — close it to mo-after that
        // RMW's write as well. (`VersionVec` is `Copy`; work on a scratch
        // copy to keep the borrows disjoint.)
        let self_id = self.stores[index].id;
        let mut mo = self.stores[index].modification_order;
        self.close_rmw_atomicity(&mut mo, self_id);
        self.stores[index].modification_order = mo;
    }

    /// Run the RMW-atomicity implication to fixpoint on `mo`, the
    /// modification order of the store identified by `self_id` (use the
    /// about-to-be-created store's id at creation — it is not in the ring
    /// yet, so nothing matches it).
    ///
    /// For every live RMW write `w` that read store `x`: if `mo` already
    /// contains `x` (single-lane marker on `x`'s creation stamp) but not yet
    /// `w`, then — because nothing may sit between `x` and `w` — the target
    /// store is mo-after `w`; join `w`'s vector. Iterated because RMW writes
    /// chain (`w` may itself be some other RMW's read store).
    ///
    /// Exclusions: `w` itself (a store is not mo-after itself), and the read
    /// store `x` (it is mo-*before* its own RMW successor; without the
    /// `read_id` check, `x`'s own vector trivially contains its own stamp
    /// and the closure would wrongly order `x` after `w`).
    fn close_rmw_atomicity(&self, mo: &mut VersionVec, self_id: u16) {
        let live = self.live_stores();

        loop {
            let mut changed = false;

            for i in 0..live {
                let w = &self.stores[i];

                if w.id == self_id {
                    continue;
                }

                let read = match w.rmw_read {
                    Some(read) => read,
                    None => continue,
                };

                if read.read_id == self_id {
                    continue;
                }

                let after_read = mo.lane(read.creator) >= read.tick;
                let after_write = mo.lane(w.creator) >= w.tick();

                if after_read && !after_write {
                    mo.join(&w.modification_order);
                    changed = true;
                }
            }

            if !changed {
                return;
            }
        }
    }

    /// Track an atomic load
    fn track_load(&mut self, threads: &thread::Set) {
        assert!(!self.is_mutating, "atomic cell is in `with_mut` call");

        let current = &threads.active().causality;

        if let Some(mut_at) = current.ahead(&self.unsync_mut_at) {
            location::panic("Causality violation: Concurrent load and mut accesses.")
                .location("created", self.created_location)
                .thread("with_mut", mut_at, self.unsync_mut_locations[mut_at])
                .thread("load", threads.active_id(), self.loaded_locations[threads])
                .fire();
        }

        self.loaded_at.join(current);
    }

    /// Track an unsynchronized load
    fn track_unsync_load(&mut self, threads: &thread::Set) {
        assert!(!self.is_mutating, "atomic cell is in `with_mut` call");

        let current = &threads.active().causality;

        if let Some(mut_at) = current.ahead(&self.unsync_mut_at) {
            location::panic("Causality violation: Concurrent `unsync_load` and mut accesses.")
                .location("created", self.created_location)
                .thread("with_mut", mut_at, self.unsync_mut_locations[mut_at])
                .thread(
                    "unsync_load",
                    threads.active_id(),
                    self.unsync_loaded_locations[threads],
                )
                .fire();
        }

        if let Some(stored) = current.ahead(&self.stored_at) {
            location::panic("Causality violation: Concurrent `unsync_load` and atomic store.")
                .location("created", self.created_location)
                .thread("atomic store", stored, self.stored_locations[stored])
                .thread(
                    "unsync_load",
                    threads.active_id(),
                    self.unsync_loaded_locations[threads],
                )
                .fire();
        }

        self.unsync_loaded_at.join(current);
    }

    /// Track an atomic store
    fn track_store(&mut self, threads: &thread::Set) {
        assert!(!self.is_mutating, "atomic cell is in `with_mut` call");

        let current = &threads.active().causality;

        if let Some(mut_at) = current.ahead(&self.unsync_mut_at) {
            location::panic("Causality violation: Concurrent atomic store and mut accesses.")
                .location("created", self.created_location)
                .thread("with_mut", mut_at, self.unsync_mut_locations[mut_at])
                .thread(
                    "atomic store",
                    threads.active_id(),
                    self.stored_locations[threads],
                )
                .fire();
        }

        if let Some(loaded) = current.ahead(&self.unsync_loaded_at) {
            location::panic(
                "Causality violation: Concurrent atomic store and `unsync_load` accesses.",
            )
            .location("created", self.created_location)
            .thread("unsync_load", loaded, self.unsync_loaded_locations[loaded])
            .thread(
                "atomic store",
                threads.active_id(),
                self.stored_locations[threads],
            )
            .fire();
        }

        self.stored_at.join(current);
    }

    /// Track an unsynchronized mutation
    fn track_unsync_mut(&mut self, threads: &thread::Set) {
        assert!(!self.is_mutating, "atomic cell is in `with_mut` call");

        let current = &threads.active().causality;

        if let Some(loaded) = current.ahead(&self.loaded_at) {
            location::panic("Causality violation: Concurrent atomic load and unsync mut accesses.")
                .location("created", self.created_location)
                .thread("atomic load", loaded, self.loaded_locations[loaded])
                .thread(
                    "with_mut",
                    threads.active_id(),
                    self.unsync_mut_locations[threads],
                )
                .fire();
        }

        if let Some(loaded) = current.ahead(&self.unsync_loaded_at) {
            location::panic(
                "Causality violation: Concurrent `unsync_load` and unsync mut accesses.",
            )
            .location("created", self.created_location)
            .thread("unsync_load", loaded, self.unsync_loaded_locations[loaded])
            .thread(
                "with_mut",
                threads.active_id(),
                self.unsync_mut_locations[threads],
            )
            .fire();
        }

        if let Some(stored) = current.ahead(&self.stored_at) {
            location::panic(
                "Causality violation: Concurrent atomic store and unsync mut accesses.",
            )
            .location("created", self.created_location)
            .thread("atomic store", stored, self.stored_locations[stored])
            .thread(
                "with_mut",
                threads.active_id(),
                self.unsync_mut_locations[threads],
            )
            .fire();
        }

        if let Some(mut_at) = current.ahead(&self.unsync_mut_at) {
            location::panic("Causality violation: Concurrent unsync mut accesses.")
                .location("created", self.created_location)
                .thread("with_mut one", mut_at, self.unsync_mut_locations[mut_at])
                .thread(
                    "with_mut two",
                    threads.active_id(),
                    self.unsync_mut_locations[threads],
                )
                .fire();
        }

        self.unsync_mut_at.join(current);
    }

    /// Find all stores that could be returned by an atomic load.
    ///
    /// A load obeying the C++20 SC read rule ([atomics.order]) may not return a
    /// store that is modification-order-before some SC-ranked store to this
    /// cell that lies within the load's SC *scope* — all of S for a `SeqCst`
    /// load, or the position of the most recent `SeqCst` fence for a load
    /// sequenced after one (the fence-read rules p4/p6). See the `sc_scope`
    /// comment below and the module SC notes. The rule is per-location and
    /// exact: mo-incomparable concurrent stores stay readable, so legal weak
    /// behaviors of nearby relaxed accesses are preserved.
    fn match_load_to_stores(
        &self,
        threads: &thread::Set,
        dst: &mut [u8],
        ordering: Ordering,
    ) -> usize {
        let mut n = 0;
        let live = self.live_stores();

        // The SC read rule reaches this load through a *scope* — how far into
        // the SC total order S the load must respect (C++20 [atomics.order]):
        //
        // - A `SeqCst` load participates in S directly; its scope is all of S
        //   (`u32::MAX`). It may not read a store mo-before any SC-ranked store
        //   to this cell (the SC read rule for accesses).
        //
        // - A non-SC load sequenced after a `SeqCst` fence is bounded by that
        //   fence's position (p4/p6): it may not read a store mo-before an
        //   SC-ranked store that is *as-early-as-or-before that fence in S*
        //   (`sc_rank <= limit`). The most recent fence's position is used — a
        //   later fence reaches further into S and subsumes all earlier ones.
        //
        // - Any other load is unconstrained by SC (`None`).
        //
        // Enforced by the fold in the coherence loop below: a candidate is
        // dropped once some in-scope SC-ranked store is found mo-after it. Only
        // genuine `mo_before` edges are consulted — never a rank comparison —
        // so a store promoted late (given a high `sc_rank` by a fence though it
        // is mo-early, e.g. a cell's initial store) can never masquerade as a
        // newer witness and wrongly supersede a mo-later write. mo-incomparable
        // concurrent stores stay readable, so no legal weak behavior is lost.
        let sc_scope = if is_seq_cst(ordering) {
            Some(u32::MAX)
        } else {
            threads.active_sc_fence_pos()
        };

        // We only need to consider loads as old as the **most** recent load
        // seen by each thread in the current causality.
        //
        // This probably isn't the smartest way to implement this, but someone
        // else can figure out how to improve on it if it turns out to be a
        // bottleneck.
        //
        // Add all stores **unless** a newer store has already been seen by the
        // current thread's causality.
        'outer: for i in 0..live {
            let store_i = &self.stores[i];

            for j in 0..live {
                let store_j = &self.stores[j];

                if i == j {
                    continue;
                }

                if mo_before(store_i, store_j) {
                    // SC read rule: `store_i` is mo-before `store_j`; if
                    // `store_j` is SC-ranked within this load's scope, `store_i`
                    // is superseded in S and may not be read.
                    if let Some(limit) = sc_scope {
                        if store_j.sc_rank.is_some_and(|r| r <= limit) {
                            continue 'outer;
                        }
                    }

                    if store_j.first_seen.is_seen_by_current(threads) {
                        // Store `j` is newer, so don't store the current one.
                        continue 'outer;
                    }

                    if store_i.first_seen.is_seen_before_yield(threads) {
                        // Saw this load before the previous yield. In order to
                        // advance the model, don't return it again.
                        continue 'outer;
                    }
                }
            }

            // The load may return this store
            dst[n] = i as u8;
            n += 1;
        }

        n
    }

    /// Promote every live store this thread created (hence sequenced before an
    /// executing `SeqCst` fence) into the SC total order S at the fence's
    /// position `pos` (C++20 [atomics.order] p5/p7). A store already SC-ranked
    /// keeps its own — necessarily earlier — position. After this, a later SC
    /// load, or a load past a fence that follows `pos` in S, sees the promoted
    /// write through the ordinary SC read rule.
    pub(super) fn promote_sc_writes(&mut self, creator: usize, pos: u32) {
        for store in self.stores_mut() {
            if store.creator == creator && store.sc_rank.is_none() {
                store.sc_rank = Some(pos);
            }
        }
    }

    fn match_rmw_to_stores(&self, dst: &mut [u8]) -> usize {
        let mut n = 0;
        let live = self.live_stores();

        // Unlike `match_load_to_stores`, rmw operations only load "newest"
        // stores, in terms of modification order: an RMW's write is
        // immediately mo-after its read, so a store with any known mo
        // successor is not a legal read. Stores that remain mo-incomparable
        // are all offered — the exploration branches over the possible total
        // extensions.
        'outer: for i in 0..live {
            let store_i = &self.stores[i];

            for j in 0..live {
                let store_j = &self.stores[j];

                if i == j {
                    continue;
                }

                if mo_before(store_i, store_j) {
                    // There is a newer store.
                    continue 'outer;
                }
            }

            // The load may return this store
            dst[n] = i as u8;
            n += 1;
        }

        n
    }

    /// Number of `stores` slots holding a real store.
    ///
    /// The ring fills positions `0..cnt` in order and wraps once full, so
    /// positions at and past `min(cnt, MAX_ATOMIC_HISTORY)` are the
    /// zeroed `Default` — their all-MAX `first_seen` matches no thread
    /// and their zero `modification_order` joins as a no-op, so skipping
    /// them never changes a result, only the work.
    fn live_stores(&self) -> usize {
        cmp::min(self.cnt as usize, MAX_ATOMIC_HISTORY)
    }

    fn stores_mut(&mut self) -> impl DoubleEndedIterator<Item = &mut Store> {
        let (start, end) = range(self.cnt);
        let (two, one) = self.stores[..end].split_at_mut(start);

        one.iter_mut().chain(two.iter_mut())
    }

    /// Calls `f` with every thread's last dependent access.
    ///
    /// A load depends on each thread's last store/rmw; a store/rmw depends on
    /// each thread's last access of any kind. Accesses by the querying thread
    /// itself are included — they are program-ordered before the current
    /// operation, so the caller's happens-before check filters them.
    pub(super) fn for_each_dependent_access<'a>(
        &'a self,
        action: Action,
        mut f: impl FnMut(&'a Access),
    ) {
        let slots: &[Option<Access>; MAX_THREADS] = match action {
            Action::Load => &self.last_non_load_access,
            _ => &self.last_access,
        };

        for access in slots.iter().flatten() {
            f(access);
        }
    }

    /// Sets the thread's last dependent access
    pub(super) fn set_last_access(
        &mut self,
        action: Action,
        thread_id: thread::Id,
        path_id: usize,
        version: &VersionVec,
    ) {
        let index = thread_id.as_usize();

        // Always set `last_access`
        Access::set_or_create(&mut self.last_access[index], path_id, version);

        match action {
            Action::Load => {}
            _ => {
                // Stores / RMWs
                Access::set_or_create(&mut self.last_non_load_access[index], path_id, version);
            }
        }
    }
}

// ===== impl Store =====

impl Default for Store {
    fn default() -> Store {
        Store {
            value: 0,
            happens_before: VersionVec::new(),
            modification_order: VersionVec::new(),
            // Dead-slot id: real ids are assigned from `cnt` starting at 0
            // and the ring evicts old ids long before the counter could
            // reach `u16::MAX`.
            id: u16::MAX,
            creator: 0,
            rmw_read: None,
            sync: Synchronize::new(),
            first_seen: FirstSeen::new(),
            sc_rank: None,
        }
    }
}

// ===== impl FirstSeen =====

impl FirstSeen {
    fn new() -> FirstSeen {
        FirstSeen([u16::max_value(); MAX_THREADS])
    }

    fn touch(&mut self, threads: &thread::Set) {
        if self.0[threads.active_id().as_usize()] == u16::max_value() {
            self.0[threads.active_id().as_usize()] = threads.active_atomic_version();
        }
    }

    fn is_seen_by_current(&self, threads: &thread::Set) -> bool {
        self.is_seen_in(&threads.active().causality, threads.execution_id())
    }

    /// True if the given thread has itself loaded from (or created) the
    /// store, at any point.
    fn is_touched_by(&self, thread_id: thread::Id) -> bool {
        self.0[thread_id.as_usize()] != u16::MAX
    }

    /// True if some thread's first sight of the store is contained in `view`.
    fn is_seen_in(&self, view: &VersionVec, execution_id: crate::rt::execution::Id) -> bool {
        for (thread_id, version) in view.versions(execution_id) {
            match self.0[thread_id.as_usize()] {
                u16::MAX => {}
                v if v <= version => return true,
                _ => {}
            }
        }

        false
    }

    fn is_seen_before_yield(&self, threads: &thread::Set) -> bool {
        let thread_id = threads.active_id();

        let last_yield = match threads.active().last_yield {
            Some(v) => v,
            None => return false,
        };

        match self.0[thread_id.as_usize()] {
            u16::MAX => false,
            v => v <= last_yield,
        }
    }
}

fn is_seq_cst(order: Ordering) -> bool {
    order == Ordering::SeqCst
}

fn range(cnt: u16) -> (usize, usize) {
    let start = index(cnt.saturating_sub(MAX_ATOMIC_HISTORY as u16));
    let mut end = index(cmp::min(cnt, MAX_ATOMIC_HISTORY as u16));

    if end == 0 {
        end = MAX_ATOMIC_HISTORY;
    }

    assert!(
        start <= end,
        "[loom internal bug] cnt = {}; start = {}; end = {}",
        cnt,
        start,
        end
    );

    (start, end)
}

fn index(cnt: u16) -> usize {
    cnt as usize % MAX_ATOMIC_HISTORY
}
