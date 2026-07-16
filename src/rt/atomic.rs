//! An atomic cell
//!
//! See the CDSChecker paper for detailed explanation.
//!
//! # Sub-word sub-locations (mixed-size atomics)
//!
//! One `AtomicU128`/`AtomicU64` cell may be accessed at sub-word granularity —
//! an aligned lane written on its own while the rest of the word is untouched
//! (spec carve-out #5: `lse2` 16-byte single-copy atomicity + per-byte
//! coherence). C11 does not model mixed-size access to one object, so the cell
//! is modelled as a set of **regions**: pairwise-disjoint bit-masks, each a
//! self-contained store history with its own modification order, exactly the
//! per-location machinery below scoped to a lane.
//!
//! - A region carries the full per-cell ring (`Region`: `stores`, `cnt`, all
//!   the coherence/SC logic). Two regions with disjoint masks order
//!   independently — a lane-A load may return an older lane-A store after a
//!   newer lane-B store is seen, which per-byte hardware coherence permits and
//!   a single welded ring wrongly forbids.
//! - Regions are **discovered by refining split**: a cell starts as one
//!   full-width region and an op whose mask cuts a region splits it so the
//!   mask becomes a union of whole regions (`State::ensure_partition`). A
//!   full-width op (`mask == u128::MAX`) never splits — a cell only ever
//!   touched full-width stays one region and behaves exactly as a single ring.
//! - A **full-width op spans every region as one linearization point**: a full
//!   load composes the per-region readable choices; a full store/RMW writes
//!   every region sharing one SC position. Reading the newest-per-region at one
//!   step *is* the single-copy-atomic snapshot, while the histories stay
//!   independent between wide ops.
//! - **Cell coherence across loads** (`filter_seen_op_floors`): a load of one
//!   region may never return a store that is modification-order-before the
//!   sibling (same `op_id`) of a *multi-region* op the loading thread has
//!   already observed through any other region of the cell. Single-copy
//!   atomicity makes a wide op one indivisible event on the one cell: having
//!   seen any part of it, a thread reading a sibling lane older than it would
//!   travel backwards through that event — the witnessed futex failure was a
//!   broadcaster whose masked queue-lane load missed a committed 128-bit push
//!   CAS it had already observed through the value lane. Separate masked ops
//!   share no `op_id`, so genuinely independent lanes stay independently
//!   coherent (the per-lane staleness the region model exists to express),
//!   and mo-*incomparable* racing stores stay readable — only genuine
//!   travelled-backwards candidates are excluded.
//! - **DPOR dependence stays cell-wide** (the whole cell is one exploration
//!   object): disjoint-lane ops are still treated as dependent, so this is a
//!   fidelity change only — it never prunes a schedule, only widens the set of
//!   readable values. The mask-intersection pruning is a separate, later step.
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
//! explored. The rules below key on the region (per-location) and never create
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
//! A full-width SeqCst store spans every region at **one** S position (the op
//! allocates a single `sc_rank` and hands it to each region), matching that the
//! wide write is a single event in S.
//!
//! Rules:
//!
//! - Seq-cst/MO Consistency:
//!
//!   The SC-ranked stores to a region are totally ordered by S, and
//!   modification order must agree with S. On `store`, a SeqCst store joins the
//!   `modification_order` of every SC-ranked store already committed to the
//!   region, so they form an mo-chain in commit order (`Region::store`).
//!
//! - Seq-cst Read Restriction:
//!
//!   A load obeys the SC read rule within a *scope* — how far into S it must
//!   respect. A `SeqCst` load's scope is all of S; a load sequenced after a
//!   `SeqCst` fence has the fence's position as its scope (the fence-read rules
//!   p4/p6); any other load is unconstrained. The load may not return a store
//!   that is modification-order-before an SC-ranked store to the region whose
//!   rank lies within scope (`match_load_to_stores`). Only genuine `mo_before`
//!   edges gate the exclusion, so a store promoted late (mo-early yet given a
//!   high rank, e.g. a region's initial store under a `SeqCst` fence in its
//!   creating thread) can never masquerade as a newer witness. Enforcing this
//!   forbids the store-buffering, IRIW and read-write-causality outcomes that
//!   plain acquire/release permits — spelled with SC accesses, SC fences, or a
//!   mix of the two — while mo-incomparable concurrent stores stay readable so
//!   no legal weak behavior is lost.
//!
//!   The read half of a SeqCst RMW (and a failed SeqCst compare-exchange)
//!   needs no check: it reads an mo-maximal store, which is never mo-before any
//!   other store to the region.
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
//! committed to the same region (SC/mo consistency). Because vector clocks are
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

use crate::rt::execution::Execution;
use crate::rt::location::{self, Location, LocationSet};
use crate::rt::object;
use crate::rt::{
    self, thread, Access, Numeric, Synchronize, VersionVec, MAX_ATOMIC_HISTORY, MAX_THREADS,
};

use smallvec::SmallVec;
use std::cmp;
use std::marker::PhantomData;
use std::sync::atomic::Ordering;
use std::u16;

use tracing::trace;

/// Mask of a full-width access: every bit of the 128-bit cell.
const FULL_MASK: u128 = u128::MAX;

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

    /// The sub-word regions of the cell: pairwise-disjoint masks, each a
    /// self-contained store history. Starts as one full-width region and is
    /// refined by `ensure_partition` when a masked op cuts a region. A cell
    /// only ever touched full-width keeps a single region — identical to the
    /// pre-sub-location single ring.
    regions: Vec<Region>,

    /// Monotonic per-cell store-op counter. Each store op takes the next id
    /// (`next_op_id`) and stamps every region it writes with it, so a wide
    /// op's siblings share one `op_id` (single-copy atomicity) while masked
    /// ops to different lanes get distinct ids (independence). The genesis
    /// store is id 0.
    op_clock: u64,

    /// Bitset (one bit per thread id) of threads that have *touched a store's
    /// `first_seen`* in this cell — i.e. loaded, stored, or rmw'd it, plus the
    /// creating thread for the genesis store. A `fence(Acquire)` synchronizes
    /// only with stores the active thread has touched, and a `fence(SeqCst)`
    /// only promotes stores the active thread created; both are impossible in
    /// a cell whose bit for the active thread is clear, so such cells are
    /// skipped wholesale instead of scanning their stores. Maintained wherever
    /// `first_seen.touch` runs (see `track_load`/`track_store`).
    touched_by: u32,
}

/// One sub-word region of a cell: a bit-mask and the store history over just
/// those bits. Every method here is the per-location coherence/SC machinery
/// scoped to the region's ring.
#[derive(Debug)]
struct Region {
    /// The bits of the cell this region owns. Regions of one cell partition
    /// the whole 128-bit width.
    mask: u128,

    /// Currently tracked stored values (the region's bits; other bits of a
    /// `Store::value` are don't-cares, masked off on compose). The
    /// `MAX_ATOMIC_HISTORY` most recent stores in loom execution order.
    stores: Box<[Store; MAX_ATOMIC_HISTORY]>,

    /// The total number of stores to the region.
    cnt: u16,

    /// Last time each thread accessed **this region**. Tracks the dependent
    /// accesses for the DPOR algorithm.
    ///
    /// Per-thread, not a single shared slot (fork fix): with one slot, a
    /// thread's own access overwrites the record of every peer's — a spawned
    /// thread whose prefix is `load; compare_exchange` records its own load as
    /// the last access, so its CAS is only ever checked against that (trivially
    /// happens-before) and the conflict with a peer's earlier plain load is
    /// never seen; the reorder DPOR owes for it was then silently unexplored.
    ///
    /// Per-region, not cell-wide, so mask-intersection pruning is *sound*: a
    /// thread's lane-A access must not shadow its earlier lane-B access, or a
    /// later lane-B op would filter it out and miss the conflict. Splitting a
    /// region clones these records into both halves (`split_off`).
    last_access: Box<[Option<Access>; MAX_THREADS]>,

    /// Last time each thread accessed this region with a store or rmw.
    last_non_load_access: Box<[Option<Access>; MAX_THREADS]>,
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub(super) enum Action {
    /// Atomic load of the bits under the mask
    Load(u128),

    /// Atomic store to the bits under the mask
    Store(u128),

    /// Atomic read-modify-write of the bits under the mask
    Rmw(u128),
}

impl Action {
    /// The bits this action touches. Two atomic ops on one cell are
    /// DPOR-dependent only when their masks intersect — disjoint lanes commute.
    fn mask(self) -> u128 {
        match self {
            Action::Load(m) | Action::Store(m) | Action::Rmw(m) => m,
        }
    }

    fn is_load(self) -> bool {
        matches!(self, Action::Load(_))
    }
}

#[derive(Debug, Clone)]
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

    /// Absolute store count at creation (`Region::cnt`); identifies the store
    /// across ring eviction within one execution.
    id: u16,

    /// Identity of the store **operation** that created this store, shared by
    /// every region a wide (multi-region) op wrote in one step and unique to a
    /// masked op. A full-width load keeps wide ops single-copy-atomic by
    /// reading a store's siblings all-or-none: two regions must agree on
    /// whether they see op `op_id` (`State::load_masked` consistency filter).
    /// Two *separate* masked ops to different lanes carry different `op_id`s
    /// and stay independently coherent.
    op_id: u64,

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
    /// `SeqCst` store (ranked at its own commit) or because it was sequenced
    /// before a `SeqCst` fence that has since executed and promoted it (ranked
    /// at the fence's position, `promote_sc_writes`; C++20 [atomics.order]
    /// p5/p7). Two SC-ranked stores to this region are ordered in S by their
    /// positions, and S agrees with modification order, so `a.sc_rank <
    /// b.sc_rank` implies `a` is mo-before `b` — the fact the SC read rule uses
    /// to exclude superseded writes without needing a materialized mo edge (see
    /// `match_load_to_stores`).
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

/// Per-thread "version at which this thread first saw the store", padded to
/// `VersionVec::LANES` so `is_seen_in` is a single branchless all-lane compare
/// against a `VersionVec`'s lane array. The padding lanes
/// `MAX_THREADS..LANES` are held at `u16::MAX` (matches no view lane, which is
/// `<= MAX_THREADS`-bounded and structurally zero in padding), so they are
/// inert in every comparison — exactly like the real lanes of a thread that
/// has not seen the store.
#[derive(Debug, Clone)]
struct FirstSeen([u16; VersionVec::LANES]);

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
    let active_bit = 1u32 << execution.threads.active_id().as_usize();
    for state in execution.objects.iter_mut::<State>() {
        // No store in this cell was ever touched by the active thread, so
        // `is_touched_by` below is false for all of them — skip the scan.
        if state.touched_by & active_bit == 0 {
            continue;
        }
        // Iterate every region's stores
        for store in state.all_stores_mut() {
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
    let creator_bit = 1u32 << creator;
    for state in execution.objects.iter_mut::<State>() {
        // Promotion only affects stores whose `creator` is this thread, and
        // every such store set this thread's `touched_by` bit — so a cell
        // without the bit has nothing to promote.
        if state.touched_by & creator_bit == 0 {
            continue;
        }
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
        T::from_u128(self.load_masked(location, FULL_MASK, ordering))
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

            // Compose the most recent value across every region.
            T::from_u128(state.newest_value())
        })
    }

    /// Stores a value into the atomic cell.
    pub(crate) fn store(&self, location: Location, val: T, ordering: Ordering) {
        self.store_masked(location, FULL_MASK, val.into_u128(), ordering)
    }

    /// Read-modify-write over the full cell.
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
        self.rmw_masked(location, FULL_MASK, success, failure, |num| {
            f(T::from_u128(num)).map(T::into_u128)
        })
        .map(T::from_u128)
    }

    /// Loads only the bits under `mask` (other bits returned as zero). A full
    /// load passes `FULL_MASK` and composes every region.
    ///
    /// The op registers a single cell-wide DPOR branch, then branches the
    /// value selection **per region**: each covered region contributes one
    /// `push_load`/`branch_load` pair, so exploration walks the cross-product
    /// of the lanes' readable sets — independent per-lane staleness — while the
    /// op stays one linearization point.
    pub(crate) fn load_masked(&self, location: Location, mask: u128, ordering: Ordering) -> u128 {
        self.ensure_partition(mask);
        self.branch(Action::Load(mask), location);

        super::synchronize(|execution| {
            let state = self.state.get_mut(&mut execution.objects);

            state.loaded_locations.track(location, &execution.threads);
            // Validate memory safety (cell-wide).
            state.track_load(&execution.threads);

            trace!(state = ?self.state, ?ordering, ?mask, "Atomic::load_masked");

            let covered = state.covered(mask);

            // A load spanning more than one region must return a single
            // consistent snapshot: wide (multi-region) ops are seen all-or-none
            // so a 128-bit load never tears one. `resolved` carries the wide-op
            // visibility fixed by the regions already read; each region's
            // readable set is filtered to agree with it. Two *separate* masked
            // ops to different lanes share no `op_id`, so this never couples
            // independent lanes — they stay free to be read in either order.
            let multi = covered.len() > 1;
            let mut resolved: SmallVec<[(u64, bool); 8]> = SmallVec::new();
            let mut result = 0u128;

            for ri in covered {
                // If necessary, generate the list of stores to permute through
                // for this region.
                //
                // A `SeqCst` load participates in the SC total order S; a load
                // past a `SeqCst` fence is bounded by the fence's position. The
                // readable set is restricted inside `match_load_to_stores`.
                if execution.path.is_traversed() {
                    let mut seed = [0; MAX_ATOMIC_HISTORY];
                    let mut n = state.regions[ri].match_load_to_stores(
                        &execution.threads,
                        &mut seed[..],
                        ordering,
                    );

                    // Cell coherence: never read this region from behind a
                    // whole-cell op the thread already observed elsewhere.
                    n = state.filter_seen_op_floors(ri, &execution.threads, &mut seed[..], n);

                    if multi {
                        // Keep only candidates consistent with the wide-op
                        // visibility earlier regions committed to.
                        let mut w = 0;
                        for r in 0..n {
                            if state.regions[ri].is_consistent(seed[r] as usize, &resolved) {
                                seed[w] = seed[r];
                                w += 1;
                            }
                        }
                        assert!(
                            w > 0,
                            "[loom internal bug] no consistent store for a wide load"
                        );
                        n = w;
                    }

                    execution.path.push_load(&seed[..n]);
                }

                let index = execution.path.branch_load();
                if multi {
                    state.regions[ri].record_resolutions(index, &mut resolved);
                }
                let mask_ri = state.regions[ri].mask;
                let v = state.regions[ri].load(&mut execution.threads, index, ordering);
                result |= v & mask_ri;
            }

            result
        })
    }

    /// Stores `val`'s masked bits, leaving the rest of the cell untouched. A
    /// full store passes `FULL_MASK` and writes every region as one event
    /// (one shared SC position).
    pub(crate) fn store_masked(&self, location: Location, mask: u128, val: u128, ordering: Ordering) {
        self.ensure_partition(mask);
        self.branch(Action::Store(mask), location);

        super::synchronize(|execution| {
            let state = self.state.get_mut(&mut execution.objects);

            state.stored_locations.track(location, &execution.threads);
            // An atomic store counts as a read access to the underlying memory
            // cell (cell-wide).
            state.track_store(&execution.threads);

            trace!(state = ?self.state, ?ordering, ?mask, "Atomic::store_masked");

            // A SeqCst store is one event in S even when it spans regions: one
            // position, handed to each region so they share it. Likewise one
            // op id, so a wide store's siblings stay single-copy-atomic.
            let sc_rank = if is_seq_cst(ordering) {
                Some(execution.threads.next_sc_pos())
            } else {
                None
            };
            let op_id = state.next_op_id();

            for ri in state.covered(mask) {
                state.regions[ri].store(
                    &mut execution.threads,
                    Synchronize::new(),
                    val,
                    ordering,
                    sc_rank,
                    op_id,
                );
            }
        })
    }

    /// Read-modify-write over just the bits under `mask`. `f` receives the
    /// composed current value of the covered regions (masked bits meaningful,
    /// others zero) and returns the new full value; only the covered regions'
    /// bits are written, all as one linearization point sharing one SC
    /// position. A full RMW passes `FULL_MASK`.
    pub(crate) fn rmw_masked<F, E>(
        &self,
        location: Location,
        mask: u128,
        success: Ordering,
        failure: Ordering,
        f: F,
    ) -> Result<u128, E>
    where
        F: FnOnce(u128) -> Result<u128, E>,
    {
        self.ensure_partition(mask);
        self.branch(Action::Rmw(mask), location);

        super::synchronize(|execution| {
            let state = self.state.get_mut(&mut execution.objects);

            state.loaded_locations.track(location, &execution.threads);
            // Track the load is happening in order to ensure correct
            // synchronization to the underlying cell (cell-wide).
            state.track_load(&execution.threads);

            trace!(state = ?self.state, ?success, ?failure, ?mask, "Atomic::rmw_masked");

            // Read the current value: each covered region's RMW reads a
            // modification-order-maximal store (`match_rmw_to_stores`).
            let mut current = 0u128;
            let mut reads: SmallVec<[(usize, usize); 4]> = SmallVec::new();

            for ri in state.covered(mask) {
                if execution.path.is_traversed() {
                    let mut seed = [0; MAX_ATOMIC_HISTORY];
                    let n = state.regions[ri].match_rmw_to_stores(&mut seed[..]);
                    execution.path.push_load(&seed[..n]);
                }

                let index = execution.path.branch_load();
                let mask_ri = state.regions[ri].mask;
                let v = state.regions[ri].rmw_read(&mut execution.threads, index);
                current |= v & mask_ri;
                reads.push((ri, index));
            }

            match f(current) {
                Ok(next) => {
                    state.stored_locations.track(location, &execution.threads);
                    // Track a store operation happened (cell-wide).
                    state.track_store(&execution.threads);

                    let sc_rank = if is_seq_cst(success) {
                        Some(execution.threads.next_sc_pos())
                    } else {
                        None
                    };
                    let op_id = state.next_op_id();

                    for (ri, index) in reads {
                        state.regions[ri].rmw_commit(
                            &mut execution.threads,
                            index,
                            next,
                            success,
                            sc_rank,
                            op_id,
                        );
                    }

                    Ok(current)
                }
                Err(e) => {
                    // A failed compare-exchange is a load. With `SeqCst`
                    // failure ordering it is an SC read, but it read the
                    // mo-maximal store per region, which is never mo-before
                    // another store, so the SC read rule holds with no extra
                    // work.
                    for (ri, index) in reads {
                        state.regions[ri].rmw_fail(&mut execution.threads, index, failure);
                    }
                    Err(e)
                }
            }
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

            // Compose the most recent value across every region.
            T::from_u128(state.newest_value())
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
                    // back into every region (masked to each region's bits).
                    let val = T::into_u128(self.0);
                    for region in &mut state.regions {
                        let index = index(region.cnt - 1);
                        region.stores[index].value = val;
                    }

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

    /// Refine the region partition to `mask` **before** the DPOR branch, so
    /// `set_last_access` records the access on the final per-lane regions — the
    /// mask-intersection pruning needs each op's record scoped to exactly the
    /// lanes it touched, or a coarse-then-split record would leak the access
    /// into a disjoint lane and spuriously couple them. A full-width mask never
    /// splits, so it is skipped.
    fn ensure_partition(&self, mask: u128) {
        if mask == FULL_MASK {
            return;
        }
        rt::execution(|execution| {
            self.state
                .get_mut(&mut execution.objects)
                .ensure_partition(mask);
        });
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
            regions: vec![Region::new(FULL_MASK)],
            op_clock: 0,
            // The genesis store's `first_seen` is touched by the creating
            // thread, so seed its bit.
            touched_by: 1 << threads.active_id().as_usize(),
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
        state.regions[0].store(threads, Synchronize::new(), value, Ordering::Release, None, 0);

        state
    }

    /// Allocate the next store-op id for this cell. A wide op passes the same
    /// id to every region it writes (siblings); a masked op gets a fresh id.
    fn next_op_id(&mut self) -> u64 {
        self.op_clock += 1;
        self.op_clock
    }

    /// Refine the region partition so `mask` is a union of whole regions:
    /// split every region the mask cuts (part inside, part outside) into its
    /// inside and outside halves. A full-width mask never cuts anything.
    ///
    /// A split clones the region's history into both halves — up to now those
    /// bits moved together (coherent), so both halves inherit the same past
    /// and diverge only as future masked ops touch one but not the other.
    fn ensure_partition(&mut self, mask: u128) {
        if mask == FULL_MASK {
            return;
        }

        let mut i = 0;
        while i < self.regions.len() {
            let rm = self.regions[i].mask;
            let inside = rm & mask;
            let outside = rm & !mask;

            if inside != 0 && outside != 0 {
                // The region straddles the mask boundary: keep the inside part
                // in place and split off the outside part. Neither half
                // straddles this mask afterwards, so advancing is correct.
                let split = self.regions[i].split_off(inside);
                self.regions.push(split);
            }

            i += 1;
        }
    }

    /// Indices of the regions covered by `mask` (those whose bits intersect
    /// it). After `ensure_partition(mask)` every such region is fully inside
    /// the mask.
    fn covered(&self, mask: u128) -> SmallVec<[usize; 4]> {
        self.regions
            .iter()
            .enumerate()
            .filter(|(_, r)| r.mask & mask != 0)
            .map(|(i, _)| i)
            .collect()
    }

    /// Cell coherence for sub-word loads (module docs): drop every candidate
    /// in `seed[..n]` that is modification-order-before a region-`ri` store
    /// belonging to a multi-region op the active thread has already observed
    /// through *another* region of this cell. Returns the retained count.
    ///
    /// A region-`ri` store `f` is a **floor** when some other region holds a
    /// store with the same `op_id` whose `first_seen` is inside the active
    /// thread's causality — the thread has provably seen that whole-cell op
    /// (its own earlier lane read, or a synchronized-with observation), so a
    /// candidate mo-before `f` would read this lane from behind one
    /// indivisible single-copy-atomic event. Only genuine `mo_before` edges
    /// exclude: a racing store mo-incomparable to `f` stays readable (per-byte
    /// coherence permits either order until something orders them), and
    /// same-region observations need no floor — the plain coherence rules in
    /// `match_load_to_stores` already cover them.
    ///
    /// The result can never be empty: a floor `f` is itself never mo-before
    /// `f`, and every store excluded by the "seen a newer store" rule is
    /// superseded by a store mo-after it, which also satisfies every floor.
    ///
    /// The RMW read path needs no such filter: `match_rmw_to_stores` offers
    /// only modification-order-maximal stores, and a candidate mo-before a
    /// floor has a known mo successor — it was never offered.
    fn filter_seen_op_floors(
        &self,
        ri: usize,
        threads: &thread::Set,
        seed: &mut [u8],
        n: usize,
    ) -> usize {
        // A cell never touched by a masked op has one region — no siblings.
        if self.regions.len() <= 1 {
            return n;
        }

        let region = &self.regions[ri];
        let mut w = 0;

        'candidate: for k in 0..n {
            let c = &region.stores[seed[k] as usize];

            for f_idx in 0..region.live_stores() {
                let f = &region.stores[f_idx];
                if !mo_before(c, f) {
                    continue;
                }
                // `c` is mo-before `f`: exclude `c` iff `f` is the sibling of
                // an op the thread has seen through another region.
                for (rj, other) in self.regions.iter().enumerate() {
                    if rj == ri {
                        continue;
                    }
                    for j in 0..other.live_stores() {
                        let d = &other.stores[j];
                        if d.op_id == f.op_id && d.first_seen.is_seen_by_current(threads) {
                            continue 'candidate;
                        }
                    }
                }
            }

            seed[w] = seed[k];
            w += 1;
        }

        assert!(
            w > 0,
            "[loom internal bug] cell-coherence floor filter emptied a readable set"
        );
        w
    }

    /// Compose the newest value across every region (each region contributes
    /// the newest store of its own bits).
    fn newest_value(&self) -> u128 {
        let mut value = 0u128;
        for region in &self.regions {
            let index = index(region.cnt - 1);
            value |= region.stores[index].value & region.mask;
        }
        value
    }

    /// Promote every live store this thread created into S at `pos`, across
    /// every region (`Region::promote_sc_writes`).
    pub(super) fn promote_sc_writes(&mut self, creator: usize, pos: u32) {
        for region in &mut self.regions {
            region.promote_sc_writes(creator, pos);
        }
    }

    /// Every store across every region (for `fence_acq`).
    fn all_stores_mut(&mut self) -> impl Iterator<Item = &mut Store> {
        self.regions.iter_mut().flat_map(|r| r.stores_mut())
    }

    /// Track an atomic load
    fn track_load(&mut self, threads: &thread::Set) {
        assert!(!self.is_mutating, "atomic cell is in `with_mut` call");

        // This op will `touch` a store's `first_seen` (a load reads one; an rmw
        // reads then writes). Record the active thread so acquire fences can
        // skip this cell if no thread of theirs ever touched it.
        self.touched_by |= 1 << threads.active_id().as_usize();

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

        // This op creates a store whose `first_seen` is touched by the active
        // thread; record it so seqcst fences can skip promoting cells this
        // thread never wrote (and acquire fences can skip it entirely).
        self.touched_by |= 1 << threads.active_id().as_usize();

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

    /// Calls `f` with every thread's last dependent access **that shares bits
    /// with this op** (mask-intersection DPOR pruning): only regions the op's
    /// mask touches are consulted, so disjoint-lane ops generate no backtrack
    /// point against each other. A load depends on each thread's last
    /// store/rmw; a store/rmw depends on each thread's last access of any kind.
    /// Accesses by the querying thread itself are included — they are
    /// program-ordered before the current op, so the caller's happens-before
    /// check filters them.
    ///
    /// A wide op's access is recorded in every region it wrote, so a peer op is
    /// reported once per shared region; `Path::backtrack` and the DPOR clock
    /// join are idempotent, so the redundancy costs work, never correctness.
    pub(super) fn for_each_dependent_access<'a>(
        &'a self,
        action: Action,
        mut f: impl FnMut(&'a Access),
    ) {
        let mask = action.mask();
        let is_load = action.is_load();

        for region in &self.regions {
            if region.mask & mask != 0 {
                region.for_each_dependent_access(is_load, &mut f);
            }
        }
    }

    /// Sets the thread's last dependent access in every region the op touches.
    pub(super) fn set_last_access(
        &mut self,
        action: Action,
        thread_id: thread::Id,
        path_id: usize,
        version: &VersionVec,
    ) {
        let mask = action.mask();
        let is_load = action.is_load();
        let index = thread_id.as_usize();

        for region in &mut self.regions {
            if region.mask & mask != 0 {
                region.set_last_access(is_load, index, path_id, version);
            }
        }
    }
}

// ===== impl Region =====

impl Region {
    fn new(mask: u128) -> Region {
        Region {
            mask,
            stores: Default::default(),
            cnt: 0,
            last_access: Default::default(),
            last_non_load_access: Default::default(),
        }
    }

    /// Keep the `keep_mask` bits of this region in place; split the remaining
    /// bits into a new region that inherits a full copy of the history **and
    /// the DPOR access records**. Both halves start perfectly coherent
    /// (identical stores) and carry the same past accesses — a peer op that
    /// conflicted with the pre-split region conflicts with whichever half it
    /// still overlaps — and diverge only as future masked ops touch one but
    /// not the other.
    fn split_off(&mut self, keep_mask: u128) -> Region {
        let other_mask = self.mask & !keep_mask;
        self.mask &= keep_mask;

        Region {
            mask: other_mask,
            stores: self.stores.clone(),
            cnt: self.cnt,
            last_access: self.last_access.clone(),
            last_non_load_access: self.last_non_load_access.clone(),
        }
    }

    /// Report this region's dependent accesses (a load depends on this thread's
    /// last store/rmw; a store/rmw on any last access).
    fn for_each_dependent_access<'a>(&'a self, is_load: bool, f: &mut impl FnMut(&'a Access)) {
        let slots = if is_load {
            &self.last_non_load_access
        } else {
            &self.last_access
        };

        for access in slots.iter().flatten() {
            f(access);
        }
    }

    /// Record a thread's last access to this region.
    fn set_last_access(
        &mut self,
        is_load: bool,
        thread_id: usize,
        path_id: usize,
        version: &VersionVec,
    ) {
        Access::set_or_create(&mut self.last_access[thread_id], path_id, version);

        if !is_load {
            Access::set_or_create(&mut self.last_non_load_access[thread_id], path_id, version);
        }
    }

    fn load(&mut self, threads: &mut thread::Set, index: usize, ordering: Ordering) -> u128 {
        // Apply coherence rules
        self.apply_load_coherence(threads, index);

        let store = &mut self.stores[index];

        store.first_seen.touch(threads);
        store.sync.sync_load(threads, ordering);
        store.value
    }

    /// True if reading store `c_index` sees op `op_id` in this region: either
    /// the read store *is* that op's store here, or it is modification-order
    /// after it. Used to keep a wide op single-copy-atomic across regions.
    fn sees_op(&self, c_index: usize, op_id: u64) -> Option<bool> {
        let c = &self.stores[c_index];
        for i in 0..self.live_stores() {
            let d = &self.stores[i];
            if d.op_id == op_id {
                return Some(c.id == d.id || mo_before(d, c));
            }
        }
        // This region was not written by that op — no constraint.
        None
    }

    /// True if reading store `c_index` is consistent with the wide-op
    /// visibility already committed by earlier regions of a multi-region load:
    /// for every committed `(op_id, seen)` this region shares, the read must
    /// agree on whether it sees that op (all-or-none — single-copy atomicity).
    fn is_consistent(&self, c_index: usize, resolved: &[(u64, bool)]) -> bool {
        for &(op_id, seen) in resolved {
            if let Some(here) = self.sees_op(c_index, op_id) {
                if here != seen {
                    return false;
                }
            }
        }
        true
    }

    /// Record the wide-op visibility this region's chosen store `c_index`
    /// implies, so later regions of the same multi-region load stay consistent
    /// with it. Every live op in this region is resolved by the read's mo
    /// position relative to it.
    fn record_resolutions(&self, c_index: usize, resolved: &mut SmallVec<[(u64, bool); 8]>) {
        let c = &self.stores[c_index];
        for i in 0..self.live_stores() {
            let d = &self.stores[i];
            let op_id = d.op_id;
            let seen = c.id == d.id || mo_before(d, c);
            match resolved.iter_mut().find(|(o, _)| *o == op_id) {
                Some((_, s)) => *s = seen,
                None => resolved.push((op_id, seen)),
            }
        }
    }

    fn store(
        &mut self,
        threads: &mut thread::Set,
        mut sync: Synchronize,
        value: u128,
        ordering: Ordering,
        sc_rank: Option<u32>,
        op_id: u64,
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

        // Whether this store participates in the modelled SC total order S. The
        // position is allocated once per op (shared across the regions a wide
        // store spans) and handed in.
        let sc = sc_rank.is_some();

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
            // SC/MO consistency: the SC-ranked stores to a region are totally
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
            op_id,
            creator,
            rmw_read: None,
            sync,
            first_seen,
            sc_rank,
        };
    }

    /// The read half of an RMW: apply load coherence and return the read
    /// value. The caller composes it across regions; `rmw_commit` or
    /// `rmw_fail` follows.
    fn rmw_read(&mut self, threads: &mut thread::Set, index: usize) -> u128 {
        // Apply coherence rules.
        self.apply_load_coherence(threads, index);

        self.stores[index].first_seen.touch(threads);

        self.stores[index].value
    }

    /// The write half of a successful RMW: synchronize with the read store and
    /// append the new value, recording the read store so `close_rmw_atomicity`
    /// keeps nothing between the pair.
    fn rmw_commit(
        &mut self,
        threads: &mut thread::Set,
        index: usize,
        next: u128,
        success: Ordering,
        sc_rank: Option<u32>,
        op_id: u64,
    ) {
        // Perform load synchronization using the `success` ordering.
        self.stores[index].sync.sync_load(threads, success);

        // Capture the read store's creation stamp *before* the write half
        // runs: if the ring is full and the read store is the oldest live
        // store, the new store lands in its slot.
        let rmw_read = RmwRead {
            read_id: self.stores[index].id,
            creator: self.stores[index].creator,
            tick: self.stores[index].tick(),
        };

        // Store the new value, initializing with the `sync` value from the
        // load. This is our (hacky) way to establish a release sequence.
        let sync = self.stores[index].sync;
        self.store(threads, sync, next, success, sc_rank, op_id);

        // RMW Atomicity: mark the write half with what it read, so every
        // future store mo-after the read store gets closed to mo-after this
        // write (`close_rmw_atomicity`).
        self.stores[self::index(self.cnt - 1)].rmw_read = Some(rmw_read);
    }

    /// The failed-compare-exchange path: a load synchronizing with `failure`.
    fn rmw_fail(&mut self, threads: &mut thread::Set, index: usize, failure: Ordering) {
        self.stores[index].sync.sync_load(threads, failure);
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

    /// Find all stores that could be returned by an atomic load of this region.
    ///
    /// A load obeying the C++20 SC read rule ([atomics.order]) may not return a
    /// store that is modification-order-before some SC-ranked store to this
    /// region that lies within the load's SC *scope* — all of S for a `SeqCst`
    /// load, or the position of the most recent `SeqCst` fence for a load
    /// sequenced after one (the fence-read rules p4/p6). The rule is
    /// per-location and exact: mo-incomparable concurrent stores stay readable.
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
        //   to this region (the SC read rule for accesses).
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
        // is mo-early, e.g. a region's initial store) can never masquerade as a
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
    /// keeps its own — necessarily earlier — position.
    fn promote_sc_writes(&mut self, creator: usize, pos: u32) {
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
            // Dead-slot op id: never matches a live op (live ids start at 1;
            // the genesis store is 0). Skipped via `live_stores` anyway.
            op_id: u64::MAX,
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
        FirstSeen([u16::max_value(); VersionVec::LANES])
    }

    fn touch(&mut self, threads: &thread::Set) {
        if self.0[threads.active_id().as_usize()] == u16::max_value() {
            self.0[threads.active_id().as_usize()] = threads.active_atomic_version();
        }
    }

    fn is_seen_by_current(&self, threads: &thread::Set) -> bool {
        self.is_seen_in(&threads.active().causality)
    }

    /// True if the given thread has itself loaded from (or created) the
    /// store, at any point.
    fn is_touched_by(&self, thread_id: thread::Id) -> bool {
        self.0[thread_id.as_usize()] != u16::MAX
    }

    /// True if some thread's first sight of the store is contained in `view`.
    ///
    /// Branchless all-lane form: for every lane, the store is seen through that
    /// thread iff its first-seen version is real (`!= u16::MAX`) and `<=` the
    /// view's version for that thread. Padding lanes hold `u16::MAX` on the
    /// left and `0` on the right, so they never contribute — identical result
    /// to the old per-thread scan over `0..MAX_THREADS`, minus the branches.
    fn is_seen_in(&self, view: &VersionVec) -> bool {
        let lanes = view.lanes();
        let mut mask = 0u32;
        for i in 0..VersionVec::LANES {
            let fs = self.0[i];
            mask |= ((fs != u16::MAX) as u32) & ((fs <= lanes[i]) as u32);
        }
        mask != 0
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
