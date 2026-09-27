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
//! - **Typed lane loads are whole-cell-coherent lane reads**
//!   (`load_coherent_lane`, the model of a `sync::atomic` lane view's
//!   `load()`): the lane reads its *own* region(s) only, but its readable set
//!   is narrowed so it can never travel behind a whole-cell (multi-region) op
//!   the thread has already observed through another region by an
//!   observation both x86-64 and AArch64 order before the load
//!   (`filter_seen_op_floors`, `LoadView::is_seen`). The lanes are distinct
//!   locations, so an unordered observation — a relaxed read, an own store not
//!   followed by a `SeqCst` fence — may be satisfied after the lane load on
//!   AArch64 and floors nothing. `load_masked` itself stays independently
//!   coherent per lane — the weaker, per-byte-coherence model — for consumers
//!   that want it.
//! - **DPOR dependence is mask-scoped** (`Action::Load(mask)`): a lane load is
//!   dependent only with ops whose mask it intersects, so it commutes with
//!   disjoint-lane traffic (a value-lane load no longer serializes against
//!   every queue-lane CAS). The coherence floor above reads only already-fixed
//!   causality, so it adds no cross-lane dependence — the one op that can move
//!   the lane's own value, a wide store, carries `FULL_MASK` and is already
//!   dependent on the lane's region. This composes with the mask-intersection
//!   store pruning: loads were the last op still re-coupling decoupled lanes.
//! - **A wide RMW declares reads and writes separately**
//!   (`ModelOps::rmw_preserving`, `Action::Rmw { read, write }`): a
//!   `cmpxchg16b` that installs two lanes of a three-lane word compares all
//!   sixteen bytes but leaves the third at the value it read, and saying so is
//!   what keeps that lane independent — the op is a *reader* there, appends no
//!   store, and commutes with the lane's own readers. Whole-cell coherence
//!   survives it ([`PreservedOp`] reconstructs the floor the elided identity
//!   write supplied); the release into the carried lane does not, and
//!   `State::check_preserved_scope` traps rather than let that pass silently.
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
//! `fence(SeqCst)` participates in two cooperating mechanisms, neither of
//! which creates happens-before — C++20 SC fences have none of their own, so
//! a data race across two fences stays a race:
//!
//! - Its position in S supplies the fence↔access rules. Promotion (above)
//!   ranks, at the fence's position, every store that happens before the
//!   fence or that an operation happening before it read (p4.2). The fence's
//!   position bounds the *scope* of every operation the fence happens before
//!   — carried along synchronization in `ScView::fence_pos` — which may not be
//!   coherence-ordered before an SC-ranked store within it (p4.3): a load may
//!   not read under one, a store or RMW is placed modification-order-after
//!   each.
//! - A coherence frontier (`thread::Set::seq_cst_fence`, `ScView::frontier`)
//!   supplies fence↔fence ordering (p4.4): whatever happens before a fence
//!   earlier in S counts, for coherence alone, as happening before whatever
//!   this fence happens before (`Thread::coherence_view`).
//!
//! A fence is an operation on S itself, DPOR-dependent with every other SC
//! fence and SC access, so every order of them S could take is explored.
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
    self, thread, Access, Numeric, Path, Synchronize, VersionVec, MAX_ATOMIC_HISTORY, MAX_THREADS,
};

use rustc_hash::FxHashMap;
use smallvec::SmallVec;
use std::cmp;
use std::marker::PhantomData;
use std::sync::atomic::Ordering;
use std::u16;

use tracing::trace;

/// Mask of a full-width access: every bit of the 128-bit cell.
pub(crate) const FULL_MASK: u128 = u128::MAX;

/// Source of `Atomic::cell_id` values. Process-global and monotone, so one
/// identity is valid across every execution and every parallel worker. Zero is
/// reserved for "not yet minted", so the counter starts at one.
static NEXT_CELL_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// A constructed cell's backing: a registration cached inline.
///
/// `pub` for the same reason the materialized cells are — it is the *default*
/// backing of the public lane views, so it appears in their signature. Opaque
/// outside the crate: no public constructor, no public field.
#[derive(Debug)]
pub struct Atomic<T> {
    /// This cell's registration in the execution's object store.
    ///
    /// `Some` — registered eagerly by [`Atomic::new`], which is what every
    /// runtime construction uses. The ref is captured once and read directly
    /// by every op, exactly as it was before deferred registration existed.
    ///
    /// `None` — the cell was built by [`Atomic::const_new`] in a `const`
    /// context, where there is no execution to register with. It resolves
    /// through `Execution::deferred_atomics` on its first access of each
    /// execution (see [`Atomic::state`]).
    state: Option<object::Ref<State>>,

    /// Identity of a deferred cell, minted on first access and `0` until then.
    ///
    /// A real atomic, not a `Cell`: a `const` constructor exists to serve
    /// `static`s, and a `static` is shared by every parallel exploration
    /// worker at once (`model::check_parallel` spawns real OS threads). Minted
    /// by compare-exchange so racing workers agree on a single identity —
    /// were they instead to each cache their own execution's `object::Ref`
    /// here, the loser of every race would re-register and fragment one cell's
    /// history across several `State` objects.
    ///
    /// Identity rather than address is what makes moving a deferred cell
    /// sound: `[AtomicUsize::const_new(0); N]` built in a `const` context and
    /// then moved keeps its history, and a later cell reusing a freed address
    /// mints a fresh id instead of inheriting a stale registration. That is
    /// also why no `Drop` is needed to evict — a stale entry is unreachable by
    /// construction, and dies at the next `reset_iteration`.
    ///
    /// Unused (and left `0`) for eagerly registered cells.
    cell_id: std::sync::atomic::AtomicU64,

    /// Initial value of a deferred cell, held until first access registers it.
    ///
    /// Carried as `u128` rather than `T` for two reasons: `Numeric::into_u128`
    /// is a trait method and so not callable from a `const fn`, and storing a
    /// bare `T` would cost `Atomic<*mut T>` the automatic `Send`/`Sync` that
    /// `PhantomData<fn() -> T>` grants it. Unread when `state` is `Some`.
    init: u128,

    /// The creation site of a deferred cell, recorded when its registration
    /// lands. `None` for an eagerly registered cell, whose `State` carries it
    /// from construction, and for a deferred cell built with no caller.
    created: Option<&'static std::panic::Location<'static>>,

    _p: PhantomData<fn() -> T>,
}

/// A cell that can name its registration in the *current* execution.
///
/// The C11 model below — regions, modification order, SC promotion, RMW
/// atomicity — is written once against a resolved [`object::Ref<State>`] and
/// is reached only through this trait. What differs between cell
/// representations is how identity is established, never what the operations
/// mean: an eagerly registered cell hands back a cached ref, a deferred one
/// resolves through `Execution::deferred_atomics`.
///
/// `resolve` is called *outside* any `rt::execution` borrow, and may be called
/// more than once per operation — `branch` re-resolves on purpose, to check
/// that the algorithm under test has not written through an invalid pointer
/// into the cell's own memory.
pub(super) trait Resolve {
    fn resolve(&self) -> object::Ref<State>;

    /// This cell's registration in `execution` if it has one, else the value
    /// an unregistered cell holds. Never registers: an observer that must not
    /// perturb the execution looks the cell up through this.
    fn registered(&self, execution: &Execution) -> Result<object::Ref<State>, u128>;

    /// Resolve for the access about to run, settling any page state it depends
    /// on. Called once the access is scheduled, with nothing between it and the
    /// access; `write` marks an access that certainly writes.
    #[inline]
    fn resolve_for_access(&self, _write: bool) -> object::Ref<State> {
        self.resolve()
    }
}

impl<T: Numeric> Resolve for Atomic<T> {
    #[inline]
    fn resolve(&self) -> object::Ref<State> {
        self.state()
    }

    fn registered(&self, execution: &Execution) -> Result<object::Ref<State>, u128> {
        if let Some(state) = self.state {
            return Ok(state);
        }
        match self.cell_id.load(std::sync::atomic::Ordering::Relaxed) {
            0 => Err(self.init),
            id => execution.deferred_atomics.get(&id).copied().ok_or(self.init),
        }
    }
}

/// A cell materialized from published memory: its identity is *where it is*.
///
/// The cell carries no value and no identity word — nothing but `T`'s size and
/// alignment — so **a zeroed region is a valid array of them**, at every width
/// down to a single byte. That is what lets memory obtained the way production
/// obtains it (demand-committed VA carved into records) be modelled as it
/// actually is, rather than the checker build substituting a differently shaped
/// allocation it is able to construct.
///
/// # Why address, when [`Atomic::const_new`] mints an identity
///
/// The two constructors serve cells with different lifecycles, and the keying
/// follows the lifecycle rather than being a global choice:
///
/// - A `const`-constructed cell can be built in a `const` context and then
///   **moved** (`[C::const_new(..); N]` relocated into an owner). Its identity
///   has to travel with it, so it lives in the cell.
/// - A materialized cell is carved from a published region and **cannot move**
///   — relocating the region invalidates every cell in it, and production VA is
///   reserved once and never released. Its address *is* its identity.
///
/// Address keying is also the only scheme that works here at all. A minted
/// identity needs bits in the cell, and a materialized cell re-mints every
/// execution because its backing is freshly zeroed: a `u32` cell exhausts a
/// 32-bit id space after a few deep rigs, and a `u16` cell after a fraction of
/// one. Deriving identity from position costs no bits and accumulates nothing.
///
/// # Cost
///
/// Every operation resolves through `Execution::vm`: there is nowhere
/// to cache a ref.
macro_rules! materialized_cell {
    ($name:ident, $align:literal, $bytes:literal) => {
        #[doc = concat!("A materialized cell ", stringify!($bytes), " bytes wide.")]
        #[derive(Debug)]
        // All-zeroes is the unregistered cell, which is the whole premise of
        // the type — so the proof is mechanized rather than asserted, and
        // travels to any record a consumer builds out of these.
        #[cfg_attr(feature = "zerocopy", derive(zerocopy::FromZeros))]
        #[repr(C, align($align))]
        // `pub` only so a materialized cell's lane views can name their
        // backing in a public signature. Opaque: no public constructor, no
        // public field, nothing to do with one but pass it along.
        pub struct $name {
            // Never read and never written *by the model* — the value lives in
            // the execution's object store. The cell exists to occupy `T`'s
            // layout and to have an address. `UnsafeCell` because the bytes do
            // change underneath a shared reference: the code under test zeroes
            // records on claim.
            _bytes: std::cell::UnsafeCell<[u8; $bytes]>,
        }

        // SAFETY: the invariant is that these bytes are never accessed. Every
        // read and write the model performs goes to the object store, reached
        // by the cell's address; nothing here dereferences `_bytes`. Sharing
        // the cell across threads therefore cannot race on anything this type
        // owns.
        unsafe impl Sync for $name {}

        impl $name {
            /// The unregistered state, which is also the all-zeroes pattern.
            pub(crate) const ZEROED: $name = $name {
                _bytes: std::cell::UnsafeCell::new([0; $bytes]),
            };
        }

        impl Resolve for $name {
            fn resolve(&self) -> object::Ref<State> {
                rt::execution(|execution| {
                    resolve_materialized(execution, self as *const $name as usize)
                })
            }

            fn resolve_for_access(&self, write: bool) -> object::Ref<State> {
                rt::execution(|execution| {
                    let addr = self as *const $name as usize;
                    let state = resolve_materialized(execution, addr);
                    fault_materialized(execution, addr, write);
                    state
                })
            }

            fn registered(&self, execution: &Execution) -> Result<object::Ref<State>, u128> {
                let addr = self as *const $name as usize;
                execution.vm.cells.get(&addr).map(|c| c.state).ok_or(0)
            }
        }
    };
}

materialized_cell!(Cell1, 1, 1);
materialized_cell!(Cell2, 2, 2);
materialized_cell!(Cell4, 4, 4);
materialized_cell!(Cell8, 8, 8);
materialized_cell!(Cell16, 16, 16);

/// Bytes per page: the granule a `MEM_RESET` discard decides at.
const PAGE: usize = 4096;

/// A set of events, one tick per thread lane and zero for none; each lane keeps
/// its earliest event. A view has *reached* the set when it happens-after any
/// one of them — for vector clocks, `view[lane] >= tick` is exactly
/// "happens-after the event stamped `(lane, tick)`".
type Epochs = [u16; MAX_THREADS];

fn epoch_add(set: &mut Epochs, lane: usize, tick: u16) {
    let t = &mut set[lane];
    if *t == 0 || tick < *t {
        *t = tick;
    }
}

fn epoch_reached(set: &Epochs, view: &VersionVec) -> bool {
    let mut hit = false;
    for (lane, &tick) in set.iter().enumerate() {
        hit |= tick != 0 && view.lane(lane) >= tick;
    }
    hit
}

/// The address space the materialized cells live in, for one execution.
#[derive(Debug, Default)]
pub(crate) struct Vm {
    /// Registered cells, by address.
    cells: FxHashMap<usize, VmCell>,

    /// The registered addresses, sorted, for the range verbs.
    order: Vec<usize>,

    /// Committed ranges: disjoint, sorted by base.
    committed: Vec<Committed>,

    /// Ranges decommitted and not committed since, so an access through a stale
    /// address is named for what it is.
    decommitted: Vec<(usize, usize)>,

    /// Pages under an outstanding `MEM_RESET`, sorted by base.
    reset_pages: Vec<ResetPage>,

    /// Whether any reset ran this execution: until one does, no access has
    /// page state to settle.
    any_reset: bool,

    /// The object every page verb orders on — the address-space lock. Commits
    /// are reads of it, so they commute with each other; decommit and reset
    /// are writes, so the search explores both orders against any other verb.
    token: Option<object::Ref<State>>,
}

#[derive(Debug)]
struct VmCell {
    state: object::Ref<State>,

    /// The cell's `op_clock` when its page was reset, while that reset may
    /// still discard it: neither discarded nor written since.
    reset: Option<u64>,

    /// Op id of the zero store the latest discard appended, or 0. A later write
    /// lands on the zero page, so it is ordered after that store whether or not
    /// its writer has seen it.
    discard_op: u64,
}

#[derive(Debug)]
struct Committed {
    lo: usize,
    hi: usize,

    /// Every commit covering the range. The zero contents a commit guarantees
    /// are established for whoever returned from any of them, and for no one
    /// else: an access must happen-after at least one.
    commits: Epochs,

    location: Location,
}

/// A page under an outstanding reset. Discards and cancels are per page, but
/// each cell resolves on its own access, so the page couples its cells only
/// where happens-before fixes the order: a thread that happens-after a discard
/// of the page reads zero from every cell of it, and one that happens-after a
/// write to it reads what the page retained.
#[derive(Debug)]
struct ResetPage {
    base: usize,
    discards: Epochs,
    cancels: Epochs,
}

impl ResetPage {
    fn new(base: usize) -> ResetPage {
        ResetPage {
            base,
            discards: [0; MAX_THREADS],
            cancels: [0; MAX_THREADS],
        }
    }
}

impl Vm {
    /// Forget the execution, keeping the allocations.
    pub(crate) fn clear(&mut self) {
        self.cells.clear();
        self.order.clear();
        self.committed.clear();
        self.decommitted.clear();
        self.reset_pages.clear();
        self.any_reset = false;
        self.token = None;
    }

    /// Positions in `order` of the registered cells inside `[lo, hi)`.
    fn cells_in(&self, lo: usize, hi: usize) -> std::ops::Range<usize> {
        self.order.partition_point(|&a| a < lo)..self.order.partition_point(|&a| a < hi)
    }

    fn committed_at(&self, addr: usize) -> Option<&Committed> {
        let i = self.committed.partition_point(|r| r.hi <= addr);
        self.committed.get(i).filter(|r| r.lo <= addr)
    }

    /// Split the committed range straddling `at`, if any, so a range boundary
    /// falls there.
    fn split_committed(&mut self, at: usize) {
        let i = self.committed.partition_point(|r| r.hi <= at);
        if let Some(r) = self.committed.get_mut(i) {
            if r.lo < at {
                let tail = Committed {
                    lo: at,
                    hi: r.hi,
                    commits: r.commits,
                    location: r.location,
                };
                r.hi = at;
                self.committed.insert(i + 1, tail);
            }
        }
    }

    /// Panic naming why `addr` has no committed memory under it.
    fn uncommitted(&self, addr: usize, what: &str) -> ! {
        if self.decommitted.iter().any(|&(lo, hi)| lo <= addr && addr < hi) {
            panic!(
                "use after decommit: {what} at {addr:#x}, in memory declared decommitted with \
                 `loom::sync::atomic::materialized::unpublish` and not committed since. \
                 Decommitted memory faults; the decommit asserted no thread could still \
                 reach it."
            );
        }
        panic!(
            "materialized atomic at {addr:#x} is not in any published region.\n\
             A materialized cell takes its identity from where it lives, so the \
             memory holding it must be declared with \
             `loom::sync::atomic::materialized::publish(ptr, len)` — by whichever \
             thread published it, before any thread reaches a cell inside it."
        );
    }

    /// The leak check `Builder::check_committed_leaks` asks for: every range
    /// committed in the execution was decommitted by its end.
    pub(crate) fn check_for_leaks(&self) {
        if let Some(r) = self.committed.first() {
            location::panic(format!(
                "materialized memory [{:#x}, {:#x}) is still committed at the end of the \
                 execution ({} committed range(s) in all); commit charge is never returned.",
                r.lo,
                r.hi,
                self.committed.len(),
            ))
            .location("committed", r.location)
            .fire();
        }
    }
}

/// The page-verb object, registered on first use in the execution.
fn vm_token() -> object::Ref<State> {
    rt::execution(|execution| {
        if let Some(token) = execution.vm.token {
            return token;
        }
        let token = execution.objects.insert_with(State::shell, State::recycle);
        token
            .get_mut(&mut execution.objects)
            .init_deferred(0, None);
        execution.vm.token = Some(token);
        token
    })
}

/// Declare that the calling thread has committed `len` bytes of memory at
/// `base`: every byte reads zero until written, and an idempotent commit of
/// memory already committed changes nothing in it.
///
/// Required, not advisory: a materialized cell takes its identity from where it
/// lives, so a cell outside every committed range has no identity and its first
/// access panics. That is deliberate — it makes the declaration impossible to
/// forget, where a permissive fallback would silently model the memory as
/// having preceded the execution and could never report a reader that reached
/// it unsynchronized.
///
/// A commit synchronizes with nothing. It establishes the zero contents for the
/// committing thread, so an access must happen-after *some* commit covering the
/// cell — its own, or a committer's it synchronized with through the program's
/// own release. A second commit of live memory therefore lets its caller use the
/// memory without ordering it after the first committer's other writes.
#[track_caller]
pub(crate) fn publish(base: usize, len: usize) {
    let location = location!();
    let (lo, hi) = (base, base + len);

    vm_token().branch_action(Action::Load(FULL_MASK), location);

    super::synchronize(|execution| {
        trace!(base, len, "atomic::publish");

        let lane = execution.threads.active_id().as_usize();
        let tick = execution.threads.active().causality.lane(lane);
        let vm = &mut execution.vm;

        let mut i = 0;
        while i < vm.decommitted.len() {
            let (a, b) = vm.decommitted[i];
            if b <= lo || hi <= a {
                i += 1;
                continue;
            }
            vm.decommitted.swap_remove(i);
            if a < lo {
                vm.decommitted.push((a, lo));
            }
            if hi < b {
                vm.decommitted.push((hi, b));
            }
        }

        // Covered parts gain this commit; the gaps become committed by it alone.
        vm.split_committed(lo);
        vm.split_committed(hi);
        let mut i = vm.committed.partition_point(|r| r.hi <= lo);
        let mut at = lo;
        while at < hi {
            match vm.committed.get_mut(i) {
                Some(r) if r.lo == at => {
                    epoch_add(&mut r.commits, lane, tick);
                    at = r.hi;
                }
                next => {
                    let end = next.map_or(hi, |r| r.lo.min(hi));
                    let mut commits = [0; MAX_THREADS];
                    commits[lane] = tick;
                    vm.committed.insert(
                        i,
                        Committed {
                            lo: at,
                            hi: end,
                            commits,
                            location,
                        },
                    );
                    at = end;
                }
            }
            i += 1;
        }

        // A cell registers only in committed memory and leaves at decommit, so
        // every registered cell here lies in a part that was covered.
        for addr in &vm.order[vm.cells_in(lo, hi)] {
            epoch_add(
                &mut vm.cells[addr].state.get_mut(&mut execution.objects).genesis,
                lane,
                tick,
            );
        }
    })
}

/// Withdraw the declaration over `[base, base + len)`: the mapping itself is
/// gone, not merely its contents.
///
/// The inverse of [`publish`], and the model of a `MEM_DECOMMIT` / `munmap`. It
/// asserts that no thread can still reach the range, and checks both halves of
/// that: it is a non-atomic write to every registered cell in the range, so an
/// access by any thread the decommit does not happen-after is reported; and the
/// cells lose their registrations and the range leaves the committed set, so an
/// access after it panics as a use after decommit rather than reading the value
/// the cell last held. Between them every schedule of a concurrent access is
/// reported, and the page-verb order against a concurrent commit is explored.
///
/// Not [`reset`]: that verb keeps the mapping and admits a concurrent reader.
/// A later [`publish`] over the range re-registers its cells at zero, which is
/// what re-committing decommitted pages really hands back.
#[track_caller]
pub(crate) fn unpublish(base: usize, len: usize) {
    let location = location!();
    let (lo, hi) = (base, base + len);

    vm_token().branch_action(Action::Store(FULL_MASK), location);

    super::synchronize(|execution| {
        trace!(base, len, "atomic::unpublish");

        let vm = &mut execution.vm;
        let range = vm.cells_in(lo, hi);
        for addr in &vm.order[range.clone()] {
            vm.cells[addr]
                .state
                .get_mut(&mut execution.objects)
                .track_decommit(&execution.threads, location);
        }

        // The registrations go: they are keyed by address, so a cell that
        // survived here would be adopted by a later commit of the same range
        // together with its whole store history.
        for addr in vm.order.drain(range) {
            vm.cells.remove(&addr);
        }

        vm.split_committed(lo);
        vm.split_committed(hi);
        let first = vm.committed.partition_point(|r| r.hi <= lo);
        let last = vm.committed.partition_point(|r| r.lo < hi);
        vm.committed.drain(first..last);

        vm.decommitted.push((lo, hi));
        vm.reset_pages.retain(|p| p.base + PAGE <= lo || hi <= p.base);
    })
}

// The property the materialized representation exists to provide. If these ever
// stop holding, a cell can no longer be reinterpreted from a zeroed region and
// the representation is pointless — so they are asserted, not documented.
const _: () = {
    use std::mem::{align_of, size_of};

    assert!(size_of::<Cell1>() == 1 && align_of::<Cell1>() == 1);
    assert!(size_of::<Cell2>() == 2 && align_of::<Cell2>() == 2);
    assert!(size_of::<Cell4>() == 4 && align_of::<Cell4>() == 4);
    assert!(size_of::<Cell8>() == 8 && align_of::<Cell8>() == 8);
    assert!(size_of::<Cell16>() == 16 && align_of::<Cell16>() == 16);
};

/// Model a bulk zeroing of `[base, base + len)` by a thread that owns the range
/// exclusively.
///
/// A materialized cell keeps its value in the object store, not in the bytes it
/// occupies, so a `memset` over raw memory is invisible to the model. Code that
/// recycles a record by zeroing it before any typed reference exists — the
/// claim-time body zero of a pool — has to say so, or the reset silently does
/// not happen and every cell keeps its previous lifecycle's value.
///
/// Exclusive, and checked: this is a non-atomic write, so it is tracked exactly
/// as `with_mut` is and a peer that has not synchronized-with the caller is
/// reported. Cells in the range that are not yet registered are already zero.
/// Writing the bytes dirties their pages, cancelling any pending reset of them.
pub(crate) fn zero_exclusive(base: usize, len: usize, location: Location) {
    rt::execution(|execution| {
        trace!(base, len, "atomic::zero_exclusive");

        let (lo, hi) = (base, base + len);
        cancel_resets(execution, lo, hi);

        let vm = &mut execution.vm;
        for addr in &vm.order[vm.cells_in(lo, hi)] {
            let cell = vm.cells.get_mut(addr).expect("registered cell");
            cell.reset = None;
            let state = cell.state.get_mut(&mut execution.objects);

            state
                .unsync_mut_locations
                .track(location, &execution.threads);
            // The same check `with_mut` performs: a concurrent reader or writer
            // of a range being bulk-zeroed is a race, and the exclusivity the
            // caller claims is what makes the write legal.
            state.track_unsync_mut(&execution.threads);

            // Overwrite in place rather than appending a store: a non-atomic
            // write is not a modification-order event, and no reader may
            // legally still be looking at the old value.
            for region in &mut state.regions {
                let index = index(region.cnt - 1);
                region.stores[index].value = 0;
            }
        }
    })
}

/// A write by the active thread lands on every page of `[lo, hi)`: record it
/// as cancelling each page's pending reset.
fn cancel_resets(execution: &mut Execution, lo: usize, hi: usize) {
    // Stamped past the thread's last release, so only a view that synchronized
    // with a later one counts as having seen the write.
    let id = execution.threads.active_id();
    execution.threads.active_mut().causality.inc(id);
    let lane = execution.threads.active_id().as_usize();
    let tick = execution.threads.active().causality.lane(lane);

    let first = execution.vm.reset_pages.partition_point(|p| p.base + PAGE <= lo);
    for page in &mut execution.vm.reset_pages[first..] {
        if page.base >= hi {
            break;
        }
        epoch_add(&mut page.cancels, lane, tick);
    }
}

/// Model the `MEM_RESET` verb over `[base, base + len)`: the contents are no
/// longer of interest, but the mapping stays and a concurrent reader is
/// *admissible* rather than a bug.
///
/// `MEM_RESET` marks the pages clean without zeroing them. Until a page is
/// written again the kernel may discard it at any moment, after which it reads
/// zero; a write dirties it and cancels the discard. So each page reads its old
/// contents until a discard that may never come, then zero for good — decided
/// per page, not at the reset. The cells are marked reset here and resolve on
/// their next access ([`fault_materialized`]); nothing is stored.
///
/// A reset of an atomic a peer is reading is benign: the peer reads old or
/// zero. It races a non-atomic access like a store does, because the checker
/// explores no order between a non-atomic access and anything else, and the
/// caller claiming exclusivity over bytes a peer is resetting is the bug.
///
/// One DPOR step per registered cell, so each is dependent with every access
/// to its cell; the page-verb step orders it against commits and decommits.
pub(crate) fn reset(base: usize, len: usize, location: Location) {
    let (lo, hi) = (base, base + len);

    vm_token().branch_action(Action::Store(FULL_MASK), location);

    let targets: SmallVec<[(usize, object::Ref<State>); 8]> = rt::execution(|execution| {
        trace!(base, len, "atomic::reset");

        let vm = &mut execution.vm;
        let mut at = lo;
        while at < hi {
            match vm.committed_at(at) {
                Some(r) => at = r.hi,
                None => vm.uncommitted(at, "reset"),
            }
        }

        vm.any_reset = true;
        let mut page = lo & !(PAGE - 1);
        while page < hi {
            match vm.reset_pages.binary_search_by_key(&page, |p| p.base) {
                Ok(i) => vm.reset_pages[i] = ResetPage::new(page),
                Err(i) => vm.reset_pages.insert(i, ResetPage::new(page)),
            }
            page += PAGE;
        }

        vm.order[vm.cells_in(lo, hi)]
            .iter()
            .map(|addr| (*addr, vm.cells[addr].state))
            .collect()
    });

    for (addr, state_ref) in targets {
        state_ref.branch_action(Action::Store(FULL_MASK), location);

        super::synchronize(|execution| {
            // Between the steps a peer may decommit the cell. Recommitted since,
            // it is a fresh zero cell with nothing to discard.
            let vm = &mut execution.vm;
            let Some(cell) = vm.cells.get_mut(&addr) else {
                if vm.committed_at(addr).is_none() {
                    vm.uncommitted(addr, "reset");
                }
                return;
            };
            if !cell.state.ref_eq(state_ref) {
                return;
            }

            let state = state_ref.get_mut(&mut execution.objects);
            state.stored_locations.track(location, &execution.threads);
            state.track_store(&execution.threads);
            cell.reset = Some(state.op_clock);
        });
    }
}

/// Resolve — and on first access of this execution, create — the registration
/// of the materialized cell at `addr`.
///
/// Keyed by the address itself rather than by the committed range: a range may
/// be committed more than once in an execution (`commit` is idempotent, and the
/// pool re-commits the fringe pages a chunk shares with its neighbours as a
/// matter of course), and re-committing must not hand a live cell a fresh
/// registration and drop its history.
fn resolve_materialized(execution: &mut Execution, addr: usize) -> object::Ref<State> {
    let vm = &mut execution.vm;
    if let Some(cell) = vm.cells.get(&addr) {
        return cell.state;
    }

    let Some(committed) = vm.committed_at(addr) else {
        vm.uncommitted(addr, "materialized atomic access");
    };
    let genesis = (committed.commits, committed.location);

    let state = execution.objects.insert_with(State::shell, State::recycle);
    state
        .get_mut(&mut execution.objects)
        .init_deferred(0, Some(genesis));
    // Not reset even on a reset page: never written, it reads zero either
    // way, and it has no stores for a discard to order.
    vm.cells.insert(
        addr,
        VmCell {
            state,
            reset: None,
            discard_op: 0,
        },
    );
    let at = vm.order.partition_point(|&a| a < addr);
    vm.order.insert(at, addr);

    trace!(?state, addr, "atomic::resolve_materialized");

    state
}

/// Settle what a reset left pending on the cell at `addr`, at the access about
/// to run — after its scheduling point, so the page state is the state at the
/// access. `write` marks an access that certainly writes (a store, `with_mut`);
/// an RMW may fail, so its write is found by its store on the next access.
fn fault_materialized(execution: &mut Execution, addr: usize, write: bool) {
    let vm = &mut execution.vm;
    if !vm.any_reset {
        return;
    }
    let Some(cell) = vm.cells.get_mut(&addr) else {
        return;
    };
    if cell.reset.is_none() && (!write || cell.discard_op == 0) {
        return;
    }
    std::hint::cold_path();

    let state = cell.state.get_mut(&mut execution.objects);

    let Some(reset_clock) = cell.reset else {
        // A write onto the zero page a discard left: ordered after the
        // discard's store, which the writer may not have seen.
        for region in &mut state.regions {
            let live = region.live_stores();
            if let Some(z) = region.stores[..live]
                .iter_mut()
                .find(|s| s.op_id == cell.discard_op)
            {
                z.first_seen.touch(&execution.threads);
            }
        }
        return;
    };

    let Ok(p) = vm
        .reset_pages
        .binary_search_by_key(&(addr & !(PAGE - 1)), |p| p.base)
    else {
        // The page record went with a decommit of part of the page.
        cell.reset = None;
        return;
    };
    let page = &mut vm.reset_pages[p];

    // A store since the reset dirtied the page.
    if state.op_clock > reset_clock {
        for region in &state.regions {
            for s in &region.stores[..region.live_stores()] {
                if s.op_id > reset_clock {
                    epoch_add(&mut page.cancels, s.creator, s.tick());
                }
            }
        }
        cell.reset = None;
        return;
    }

    let id = execution.threads.active_id();
    execution.threads.active_mut().causality.inc(id);
    let lane = execution.threads.active_id().as_usize();
    let view = execution.threads.active().causality;
    let tick = view.lane(lane);

    let cancelled = epoch_reached(&page.cancels, &view);
    let discard = if epoch_reached(&page.discards, &view) {
        true
    } else if cancelled {
        false
    } else {
        execution.path.branch_spurious()
    };

    if discard {
        epoch_add(&mut page.discards, lane, tick);
        let op_id = state.next_op_id();
        state.touched_by |= 1 << lane;
        for region in &mut state.regions {
            region.store_mo_max(&mut execution.threads, 0, op_id);
        }
        cell.discard_op = op_id;
    }
    if write {
        epoch_add(&mut page.cancels, lane, tick);
    }
    if discard || write || cancelled {
        cell.reset = None;
    }
}

/// A cell's process-global identity, minting one if it has none.
///
/// `compare_exchange` rather than `fetch_add`-and-store so that workers racing
/// on a shared cell settle on exactly one identity; a loser discards the id it
/// minted (ids are cheap and need only be unique, not dense) and adopts the
/// winner's.
fn mint_id(slot: &std::sync::atomic::AtomicU64) -> u64 {
    use std::sync::atomic::Ordering::Relaxed;

    match slot.load(Relaxed) {
        0 => {
            let fresh = NEXT_CELL_ID.fetch_add(1, Relaxed);
            match slot.compare_exchange(0, fresh, Relaxed, Relaxed) {
                Ok(_) => fresh,
                Err(won) => won,
            }
        }
        id => id,
    }
}

/// Resolve — and on first access of this execution, create — the registration
/// of the cell with identity `id`, whose value before any store is `init`.
fn register(
    id: u64,
    init: u128,
    created: Option<&'static std::panic::Location<'static>>,
) -> object::Ref<State> {
    rt::execution(|execution| {
        if let Some(&state) = execution.deferred_atomics.get(&id) {
            return state;
        }

        let state = execution.objects.insert_with(State::shell, State::recycle);
        // A `const`-constructed cell is in the binary image, not in memory a
        // thread published, so its genesis precedes the execution.
        let cell = state.get_mut(&mut execution.objects);
        cell.init_deferred(init, None);
        if execution.location {
            if let Some(created) = created {
                cell.created_location = Location::from(created);
            }
        }
        execution.deferred_atomics.insert(id, state);

        trace!(?state, id, "atomic::register");

        state
    })
}

/// The C11 model, written once over any [`Resolve`]able cell.
///
/// These are the width-agnostic (`u128`) operations: every typed entry point
/// — `Atomic<T>`'s `load`/`store`/`rmw`, the lane views, and any other cell
/// representation — reduces to one of these. Keeping a single implementation
/// is what stops two representations from drifting apart on the parts that
/// actually carry the semantics: region partitioning, modification order, SC
/// promotion, and RMW atomicity.
pub(crate) trait ModelOps {
    fn load_masked(&self, location: Location, mask: u128, ordering: Ordering) -> u128;

    fn load_coherent_lane(&self, location: Location, mask: u128, ordering: Ordering) -> u128;

    fn store_masked(&self, location: Location, mask: u128, val: u128, ordering: Ordering);

    /// Read-modify-write over the bits under `read_mask`, of which only those
    /// also under `write_mask` may change. `write_mask ⊆ read_mask`.
    ///
    /// The two differ for a **preserving** RMW: a wide compare-and-swap that
    /// consults bits it writes back verbatim — a `cmpxchg16b` installing two
    /// lanes of a three-lane word compares all sixteen bytes but leaves the
    /// third lane at the value it read. Declaring that is what lets the model
    /// treat the preserved lane as read rather than written: the op commutes
    /// with that lane's readers, and the lane gains no store, no candidate and
    /// no modification order of its own. The claim is checked, not trusted —
    /// the commit asserts the preserved bits came back identical.
    ///
    /// What it costs is stated at [`PreservedOp`] and guarded by
    /// [`State::check_preserved_scope`].
    fn rmw_preserving<F, E>(
        &self,
        location: Location,
        read_mask: u128,
        write_mask: u128,
        success: Ordering,
        failure: Ordering,
        f: F,
    ) -> Result<u128, E>
    where
        F: FnOnce(u128) -> Result<u128, E>;

    /// Read-modify-write over just the bits under `mask` — the ordinary form,
    /// which may change every bit it reads.
    fn rmw_masked<F, E>(
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
        self.rmw_preserving(location, mask, mask, success, failure, f)
    }

    /// Read the composed newest value with no synchronization.
    fn unsync_load(&self, location: Location) -> u128;

    /// Access the newest value mutably. Must happen-after all stores.
    fn with_mut<R>(&mut self, location: Location, f: impl FnOnce(&mut u128) -> R) -> R;

    /// The composed newest value, for an observer outside the model — `Debug`.
    ///
    /// No branch, no access record, no registration: the execution is exactly
    /// as it was, so formatting a cell never changes which schedules are
    /// explored. The value is the newest in modification order, which the
    /// calling thread need not be able to load. `None` outside a model.
    fn peek(&self) -> Option<u128>;
}

#[derive(Debug)]
pub(super) struct State {
    /// Where the atomic was created
    created_location: Location,

    /// The commits covering a materialized cell's memory; every access must
    /// happen-after one of them. Empty for any other cell.
    genesis: Epochs,

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
    /// creating thread for the genesis store. A `fence(Acquire)` can only
    /// have read a preserved lane of a cell whose bit for the active thread is
    /// set, so it skips the others' guard check. Maintained wherever
    /// `first_seen.touch` runs (see `track_load`/`track_store`).
    touched_by: u32,

    /// Region carcasses from previous epochs of this cell. A reincarnated
    /// cell collapses its partition back to one full-width region; the split
    /// halves park here and `ensure_partition` reuses their allocations when
    /// this epoch re-splits. A cell that re-splits identically every
    /// execution — the common case — reaches steady state with zero region
    /// allocation.
    spares: Vec<Region>,
}

/// The wide-op visibility a multi-region load has fixed so far: one
/// `(op_id, seen)` per op the regions already resolved speak for.
///
/// The walk carries a single buffer and treats it as a stack — a candidate
/// pushes what it implies, is tested, and winds back — so a rejected prefix
/// costs no copy. Sized inline for the structural worst case, one entry per
/// live store of every region of a 32-bit-laned 128-bit cell, so that
/// push/truncate never reaches the allocator on the lookahead's hot path.
type Resolved = SmallVec<[(u64, bool); MAX_ATOMIC_HISTORY * 4]>;

/// The reader-side state a multi-region load filters its candidates against.
///
/// A wide load resolves one region at a time, and `Region::load` *changes* what
/// the regions still to come may read: `sync_load` joins the store's release
/// view into the reader's causality (so a store the reader has now provably
/// passed stops being readable), and `first_seen.touch` marks the store seen
/// (so it can floor a sibling lane through `op_seen_through_other_region`).
/// Both effects are real coherence - a writer's release orders its earlier lane
/// write ahead of the released one, so a snapshot that takes the released store
/// must not take a stale value of the earlier lane.
///
/// The walk therefore cannot filter a later region against the state at load
/// entry; it must filter against the state the earlier regions will have
/// established. `LoadView` is that projection: the causality and the extra
/// seen-marks a prefix of choices implies. Carrying it makes the forward
/// lookahead (`has_consistent_completion`) predict exactly the readable sets
/// the walk will go on to compute, which is what makes the walk total.
#[derive(Clone)]
struct LoadView {
    /// The reader's causality, projected forward over the prefix.
    causality: VersionVec,

    /// `(region, slot)` pairs the load has committed to reading. `Region::load`
    /// touches `first_seen` for each, which no `causality` join reproduces.
    touched: SmallVec<[(usize, usize); 4]>,

    /// The reader's happens-before alone, projected like `causality`. The
    /// floor reads this one: a `SeqCst`-fence frontier orders no observation
    /// before the load.
    hb: VersionVec,

    /// The reading thread's lane.
    me: usize,

    /// Whether the load is `SeqCst`: an own `SeqCst` store before it is then
    /// ordered before it on both targets (`xchg`; `STLR` then `LDAR`).
    seq_cst: bool,

    /// Own-clock versions of the reader's latest acquire-or-stronger fence and
    /// latest `SeqCst` fence (`Thread::acq_fence_version`, `sc_fence_version`).
    acq_fence: u16,
    own_sc_fence: u16,
}

impl LoadView {
    /// The state as of the load, once it has read the `(region, slot)` pairs
    /// in `touched`.
    fn entry(threads: &thread::Set, ordering: Ordering, touched: &[(usize, usize)]) -> LoadView {
        LoadView {
            causality: threads.active().coherence_view(),
            hb: threads.active().causality,
            touched: SmallVec::from_slice(touched),
            me: threads.active_id().as_usize(),
            seq_cst: is_seq_cst(ordering),
            acq_fence: threads.active().acq_fence_version,
            own_sc_fence: threads.active().sc_fence_version,
        }
    }

    /// The state after this load additionally reads slot `ci` of region `ri`.
    fn extend(&self, region: &Region, ri: usize, ci: usize, ordering: Ordering) -> LoadView {
        let mut next = self.clone();
        if acquires(ordering) {
            next.causality.join(region.stores[ci].sync.released_view());
            next.hb.join(region.stores[ci].sync.released_view());
        }
        next.touched.push((ri, ci));
        next
    }

    /// Has the reader observed slot `gi` of region `rj` in a way that orders the
    /// observation before this load on both x86-64 and AArch64? Only then does
    /// the observation floor a sibling lane: the lanes are distinct locations,
    /// so an unordered observation may be satisfied after this load.
    ///
    /// - Another thread's observation reached this one through a release/acquire
    ///   edge, which orders it (bob on AArch64, TSO on x86).
    /// - An own read of a peer's store is ordered once it acquired or an acquire
    ///   fence followed it; a relaxed read is not (AArch64 reorders loads).
    /// - An own store is ordered only by a later `SeqCst` fence of this thread,
    ///   or when it was a `SeqCst` store and this load is `SeqCst`: both targets
    ///   otherwise let a load pass an earlier store to other bytes.
    /// - A peer's store this same load already committed to reading is one
    ///   single-copy-atomic event with it. An own store read here may have been
    ///   forwarded, which AArch64 permits to split the snapshot.
    fn is_seen(&self, store: &Store, rj: usize, gi: usize) -> bool {
        let me_bit = 1u32 << self.me;
        let seen = store.first_seen.seen_threads(&self.hb);
        if seen & !me_bit != 0 {
            return true;
        }
        if store.creator == self.me {
            return self.own_sc_fence > store.tick() || (store.seq_cst && self.seq_cst);
        }
        let ordered_read = store.first_seen.1 & me_bit != 0
            || store.first_seen.0[self.me] < self.acq_fence;
        (seen & me_bit != 0 && ordered_read) || self.touched.contains(&(rj, gi))
    }
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

    /// Wide ops that compared these bits and wrote them back verbatim
    /// ([`PreservedOp`]) — the store-less stand-ins for the identity writes
    /// they replace. A ring of the same depth as `stores`, for the same
    /// reason: a record whose window has passed constrains no candidate the
    /// ring can still offer.
    preserved: [PreservedOp; MAX_ATOMIC_HISTORY],

    /// Total number of preserving ops over this region. Non-zero is what
    /// makes the region *elision-tainted* — the state
    /// [`State::check_preserved_scope`] guards.
    preserved_cnt: u16,

    /// Threads that have read these bits, since a preserving op landed, with
    /// an operation covering none of the regions that op wrote. Those reads
    /// are sound on their own (they acquired nothing), but a later
    /// `fence(Acquire)` in the same thread would consume a release the elided
    /// identity write no longer offers, so the fence traps on this set.
    unrouted_readers: u32,

    /// Threads that have read these bits at all, and the subset that did so
    /// while acquiring.
    ///
    /// A preserving op and a read of the lane it carries are, by construction,
    /// DPOR-independent — that is the whole point — so the search is entitled
    /// to explore one order between them and never the other. The guard
    /// therefore cannot live only on the read: it runs from **both** ends, and
    /// these are what let the op ask what has already read its carried lane.
    readers: u32,
    acquiring_readers: u32,
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub(super) enum Action {
    /// Atomic load of the bits under the mask
    Load(u128),

    /// Atomic store to the bits under the mask
    Store(u128),

    /// Atomic read-modify-write: `read` are the bits the operation consults,
    /// `write` the bits it may change, and `write ⊆ read`. They differ only
    /// for a **preserving** RMW (`ModelOps::rmw_preserving`) — a wide CAS that
    /// compares bits it writes back verbatim. On the preserved bits the
    /// operation is a reader, and DPOR treats it as one.
    Rmw { read: u128, write: u128 },
}

impl Action {
    /// The bits this action reads.
    fn read_mask(self) -> u128 {
        match self {
            Action::Load(m) => m,
            Action::Store(_) => 0,
            Action::Rmw { read, .. } => read,
        }
    }

    /// The bits this action may change.
    fn write_mask(self) -> u128 {
        match self {
            Action::Load(_) => 0,
            Action::Store(m) => m,
            Action::Rmw { write, .. } => write,
        }
    }

    /// The bits this action touches at all — the regions it must resolve.
    fn mask(self) -> u128 {
        self.read_mask() | self.write_mask()
    }

    /// DPOR dependence between two operations on this cell — the ordinary
    /// read/write dependence over the bit masks, the same pairing
    /// `for_each_dependent_access` reports, decided from the actions alone
    /// without the access records.
    ///
    /// Equivalent to the old "overlapping bits and at least one writer"
    /// wherever an RMW reads and writes the same bits; a preserving RMW's
    /// preserved lane commutes with that lane's readers, exactly as a load
    /// of it would.
    pub(super) fn conflicts_with(self, other: Action) -> bool {
        (self.write_mask() & other.read_mask())
            | (self.read_mask() & other.write_mask())
            | (self.write_mask() & other.write_mask())
            != 0
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

    /// Whether the store was itself a `SeqCst` store (or RMW write half) —
    /// unlike `sc_rank`, never set by promotion. A `SeqCst` load after it in
    /// the same thread is ordered after it on both targets.
    seq_cst: bool,
}

/// Creation stamp of the store an RMW write read — the persistent record of
/// the "nothing may split this pair" obligation.
#[derive(Debug, Copy, Clone, Default)]
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

    /// Whether this store is SC-ranked within `scope`, an operation's SC
    /// scope (`thread::Set::active_sc_scope`): that operation may not be
    /// coherence-ordered before it.
    fn in_sc_scope(&self, scope: Option<u32>) -> bool {
        matches!((self.sc_rank, scope), (Some(rank), Some(limit)) if rank <= limit)
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

/// [`mo_before`] for a store recorded only as a creation stamp: `stamp` names
/// store `a`, and this is `mo_before(a, b)` verbatim. The stamp form is what
/// lets the relation outlive `a`'s eviction from the ring.
fn stamp_mo_before(stamp: &RmwRead, b: &Store) -> bool {
    stamp.read_id != b.id && b.modification_order.lane(stamp.creator) >= stamp.tick
}

/// A wide op that **compared** this region's bits and wrote them back
/// verbatim, recorded without a store of its own
/// (`ModelOps::rmw_preserving`).
///
/// Write `s` for the store the op read here and `s'` for the identity write it
/// would have appended. RMW atomicity puts `s'` modification-order-immediately
/// after `s` with nothing insertable between, and `s'` carries `s`'s value, so
/// the pair is one run of equal bits. Everything below follows from that.
///
/// **The coherence floor is exact in value.** Seeing the op through a written
/// region must stop this lane reading older than what the op left here. With
/// `s'` the rule reads "drop candidates mo-before `s'`" — which drops `s`
/// too; here it reads "drop candidates mo-before `s`", keeping `s`. The two
/// admit the *same values*, because `s` and `s'` hold the same bits; they
/// differ only in which coherence node the reader lands on. That difference is
/// the residual below, not a difference in what can be read.
///
/// **All-or-none visibility is vacuous here, and the record stays out of it**
/// ([`Region::try_resolve`]). A wide op's regions must be seen all-or-none to
/// forbid a torn snapshot; a lane the op wrote back verbatim cannot tear.
///
/// **Being seen *through* this lane resolves strictly** ([`OpPin::Preserved`],
/// via [`State::op_seen_through_other_region`]): only a candidate strictly
/// mo-after `s` witnesses the op. Reading `s` itself is ambiguous once `s` and
/// `s'` are one node — it is what a reader that ran *before* the op sees, and
/// what a reader of `s'` would have seen — and answering "witnessed" would
/// floor a sibling lane for a reader that legitimately preceded the op,
/// forbidding a real behavior. Answering "not witnessed" only declines to
/// floor, which admits behaviors rather than removing them.
///
/// **The residual.** The record does not carry `s'`'s release: an acquiring
/// read of this lane can no longer synchronize-with the op through these bits
/// (it still can through the bits the op wrote). That is the one behavior
/// elision costs, and [`State::check_preserved_scope`] makes reaching it a
/// hard error rather than a silent under-exploration.
///
/// **The floor is also a cost, which is why elision is a trade and not a pure
/// win.** The identity write is a coherence ratchet: it sits mo-latest, so a
/// thread that observes the op through a written region has this lane's
/// readable set floored to the newest node. Carrying the lane moves that floor
/// one node earlier, handing back one candidate of staleness per op for the
/// search to explore. Against that stands what carriage buys — no store
/// appended here, and no dependence with this lane's readers. Which wins is a
/// property of the *lane*, not of the operation: on a lane whose only writes
/// are these identity writes the ratchet was pruning nothing real and carriage
/// is a clear win, while on a lane with genuine traffic of its own the two
/// terms cancel. Measure per call site; do not assume.
#[derive(Debug, Copy, Clone, Default)]
struct PreservedOp {
    /// The op's `Store::op_id` in the regions it *did* write. Zero in an
    /// unused ring slot, which no real op id ever takes (`op_clock` starts
    /// at one, and the genesis store's id 0 belongs to no preserving op).
    op_id: u64,

    /// Creation stamp of the store the op read here — its modification-order
    /// pin, standing in for the identity write's own position.
    read: RmwRead,

    /// The bits the op *did* write, so a reader that also covers one of those
    /// regions is known to still have a route to the op's release.
    write_mask: u128,

    /// Lane index of the thread that ran the op. It cannot lose a release it
    /// published itself — its own causality already contains it — so it is
    /// never an unrouted reader of its own preserved lane.
    creator: usize,
}

/// Where one store operation sits in one region's modification order — see
/// [`Region::pin_of_op`].
#[derive(Debug, Copy, Clone)]
enum OpPin {
    /// The op wrote this region; the slot holding its store.
    Wrote(usize),

    /// The op preserved this region; the creation stamp of the store it read.
    Preserved(RmwRead),
}

/// Per-thread "version at which this thread first saw the store", padded to
/// `VersionVec::LANES` so `is_seen_in` is a single branchless all-lane compare
/// against a `VersionVec`'s lane array. The padding lanes
/// `MAX_THREADS..LANES` are held at `u16::MAX` (matches no view lane, which is
/// `<= MAX_THREADS`-bounded and structurally zero in padding), so they are
/// inert in every comparison — exactly like the real lanes of a thread that
/// has not seen the store.
///
/// The second field is one bit per thread whose read of the store acquired,
/// which orders it before everything the thread does next.
#[derive(Debug, Clone)]
struct FirstSeen([u16; VersionVec::LANES], u32);

/// Implements atomic fence behavior
#[track_caller]
pub(crate) fn fence(ordering: Ordering) {
    if ordering == Ordering::SeqCst {
        rt::branch_sc_fence(location!());
    }

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
    // Synchronize with the release of every store an atomic operation
    // sequenced before this fence read ([atomics.fences] p4). Each read
    // accumulated its store's release view into `acquirable` as it happened
    // (`Synchronize::sync_load`), so a store since evicted from its cell's
    // history still counts. Only this thread's own reads qualify: a store
    // merely in its causality through another thread's relaxed read does not.
    let active = execution.threads.active_id().as_usize();
    let active_bit = 1u32 << active;
    for state in execution.objects.iter_mut::<State>() {
        if state.touched_by & active_bit == 0 {
            continue;
        }
        // A relaxed read of a preserved lane takes no causality, so it is
        // sound where it stands; this fence is where it would have taken the
        // elided identity write's release, and cannot (`PreservedOp`).
        assert!(
            !state.has_unrouted_reader(active),
            "fence(Acquire) over a cell this thread read only through a preserved lane.\n\
             `rmw_preserving` elides the identity write there, so the fence cannot draw the \
             preserving operation's release. Use `rmw_masked` for that operation, or read a \
             lane it writes."
        );
    }

    // lanes: the fence orders every earlier read of this thread before its
    // later accesses (the lane coherence floor, `LoadView::is_seen`).
    let version = execution.threads.active_atomic_version();
    let active = execution.threads.active_mut();
    active.acq_fence_version = version;
    let acquirable = active.acquirable;
    active.acquire(&acquirable);
}

fn fence_rel(execution: &mut Execution) {
    // Every later store carries the thread's view as of this fence.
    let active = execution.threads.active_mut();
    active.released = active.view();
}

fn fence_acqrel(execution: &mut Execution) {
    fence_acq(execution);
    fence_rel(execution);
}

fn fence_seqcst(execution: &mut Execution) {
    fence_acq(execution);
    // lanes: the fence orders every earlier access of this thread, stores
    // included, before its later ones (`LoadView::is_seen`).
    let version = execution.threads.active_atomic_version();
    execution.threads.active_mut().sc_fence_version = version;

    // Commit the fence into S (its position, and the coherence frontier of the
    // fences before it), then promote into S at its position every store that
    // happens before it or that an operation happening before it read: an SC
    // operation later in S may not be coherence-ordered before any of them
    // (C++20 [atomics.order] p4.2). A store already SC-ranked keeps its
    // earlier position.
    let pos = execution.threads.seq_cst_fence();
    let causality = execution.threads.active().causality;
    for state in execution.objects.iter_mut::<State>() {
        state.promote_sc_writes(&causality, pos);
    }

    // The release half last, so the stores after the fence carry its S
    // position and frontier to whoever synchronizes with them.
    fence_rel(execution);
}

// `Numeric` is private and stays that way: the bound seals this impl even
// though `Atomic` is nameable (as `ConstructedCell`) for signatures.
#[allow(private_bounds)]
impl<T: Numeric> Atomic<T> {
    /// Create a new, atomic cell initialized with the provided value
    pub(crate) fn new(value: T, location: Location) -> Atomic<T> {
        rt::execution(|execution| {
            let state = execution.objects.insert_with(State::shell, State::recycle);
            state.get_mut(&mut execution.objects).init(
                &mut execution.threads,
                value.into_u128(),
                location,
            );

            trace!(?state, "Atomic::new");

            Atomic {
                state: Some(state),
                cell_id: std::sync::atomic::AtomicU64::new(0),
                init: 0,
                created: None,
                _p: PhantomData,
            }
        })
    }

    /// Create a cell in a `const` context, initialized to `init` (the value's
    /// `u128` representation — the caller knows the concrete type and converts
    /// with a `const`-callable cast, since `Numeric::into_u128` is a trait
    /// method and cannot be one).
    ///
    /// Registration is *deferred* to the cell's first access in each execution
    /// rather than performed here, which is what makes this constructor
    /// `const`: registering means allocating a slot in the live execution's
    /// object store, and a `const` context has no execution. Nothing observes
    /// the difference — a cell no thread has touched has no stores for anyone
    /// to read, so whether it is registered is not a property of the model.
    ///
    /// Deferring also *is* the per-execution reset that a `const`-initialized
    /// `static` needs: `Execution::deferred_atomics` is cleared each
    /// iteration, so the next one re-registers the cell at `init` instead of
    /// inheriting its predecessor's stores.
    ///
    /// The genesis store of a deferred cell carries an **empty** causality
    /// rather than the constructing thread's (`State::init_deferred`), which
    /// is the truth for the case this constructor serves: a `const`-initialized
    /// value is in the binary image, so every thread trivially happens-after
    /// it. The cost is that a cell built here does not carry the
    /// initialization-race check that [`Atomic::new`]'s thread-attributed
    /// genesis provides, which is why `new` keeps that genesis and every
    /// runtime construction keeps using it.
    pub(crate) const fn const_new(
        init: u128,
        created: Option<&'static std::panic::Location<'static>>,
    ) -> Atomic<T> {
        Atomic {
            state: None,
            cell_id: std::sync::atomic::AtomicU64::new(0),
            init,
            created,
            _p: PhantomData,
        }
    }

    /// This cell's registration in the *current* execution, registering it
    /// first if this is a deferred cell's first access here.
    ///
    /// Every operation resolves through this rather than reading the field, so
    /// a deferred cell is indistinguishable from an eager one from its first
    /// touch onward. Must be called *outside* an `rt::execution` borrow — the
    /// deferred path takes one itself, and the borrow is not reentrant.
    #[inline]
    fn state(&self) -> object::Ref<State> {
        match self.state {
            Some(state) => state,
            None => self.register_deferred(),
        }
    }

    /// Resolve — and on first access of this execution, create — the
    /// registration of a deferred cell.
    fn register_deferred(&self) -> object::Ref<State> {
        register(mint_id(&self.cell_id), self.init, self.created)
    }

}

impl<C: Resolve + ?Sized> ModelOps for C {
    /// Loads only the bits under `mask` (other bits returned as zero). A full
    /// load passes `FULL_MASK` and composes every region.
    ///
    /// The op registers a single cell-wide DPOR branch, then branches the
    /// value selection **per region**: each covered region contributes one
    /// `push_load`/`branch_load` pair, so exploration walks the cross-product
    /// of the lanes' readable sets — independent per-lane staleness — while the
    /// op stays one linearization point.
    fn load_masked(&self, location: Location, mask: u128, ordering: Ordering) -> u128 {
        let state_ref = self.resolve();
        ensure_partition(state_ref, mask);
        branch(self, state_ref, Action::Load(mask), is_seq_cst(ordering), location);

        super::synchronize(|execution| {
            let state = state_ref.get_mut(&mut execution.objects);

            state.loaded_locations.track(location, &execution.threads);
            // Validate memory safety (cell-wide).
            state.track_load(&execution.threads);
            state.check_preserved_scope(mask, acquires(ordering), &execution.threads);

            trace!(state = ?state_ref, ?ordering, ?mask, "Atomic::load_masked");

            let covered = state.covered(mask);
            // `apply_floor = false`: `load_masked` is documented
            // independently-coherent per lane — each masked lane orders on its
            // own history alone. The stronger whole-cell-coherent projection is
            // the typed lane views' `load_coherent_lane`.
            state.compose_load(
                &mut execution.path,
                &mut execution.threads,
                &covered,
                ordering,
                false,
            )
        })
    }

    /// Loads the bits under `mask` as a **whole-cell-coherent lane
    /// projection** — the model of a typed lane view's `load()` (`sync::atomic`
    /// lane views). Identical to [`Self::load_masked`] in every respect but
    /// two, both deliberate:
    ///
    /// 1. **DPOR scope.** The dependence branch is `Action::Load(mask)`, scoped
    ///    to the lane, so a lane load commutes with disjoint-lane traffic — a
    ///    value-lane load no longer serializes against every queue-lane CAS.
    ///    (`load_masked` shares this; the whole-cell `load` does not.)
    ///
    /// 2. **Cell coherence.** The readable set is additionally narrowed by
    ///    [`State::filter_seen_op_floors`] so the lane can never travel behind a
    ///    whole-cell (multi-region) op the active thread has already observed
    ///    through a region *outside* this load, by an observation ordered
    ///    before this load ([`LoadView::is_seen`]). This is the property a
    ///    plain per-lane masked load does not carry.
    ///
    /// The floor consults sibling regions but reads only *already-fixed*
    /// causality (stores this thread has seen or written before this op), so it
    /// creates no new cross-lane DPOR dependence: it is a function of state a
    /// concurrent peer cannot change, and the only op that can move the lane's
    /// own value — a wide store — carries `FULL_MASK` and is already dependent
    /// on this region. Reading only the lane's own regions is what preserves
    /// the pruning (2) grants (module docs, "Sub-word sub-locations").
    fn load_coherent_lane(&self, location: Location, mask: u128, ordering: Ordering) -> u128 {
        let state_ref = self.resolve();
        ensure_partition(state_ref, mask);
        branch(self, state_ref, Action::Load(mask), is_seq_cst(ordering), location);

        super::synchronize(|execution| {
            let state = state_ref.get_mut(&mut execution.objects);

            state.loaded_locations.track(location, &execution.threads);
            state.track_load(&execution.threads);
            state.check_preserved_scope(mask, acquires(ordering), &execution.threads);

            trace!(state = ?state_ref, ?ordering, ?mask, "Atomic::load_coherent_lane");

            let covered = state.covered(mask);
            state.compose_load(
                &mut execution.path,
                &mut execution.threads,
                &covered,
                ordering,
                true,
            )
        })
    }

    /// Stores `val`'s masked bits, leaving the rest of the cell untouched. A
    /// full store passes `FULL_MASK` and writes every region as one event
    /// (one shared SC position).
    fn store_masked(&self, location: Location, mask: u128, val: u128, ordering: Ordering) {
        let state_ref = self.resolve();
        ensure_partition(state_ref, mask);
        branch(self, state_ref, Action::Store(mask), is_seq_cst(ordering), location);

        super::synchronize(|execution| {
            let state = state_ref.get_mut(&mut execution.objects);

            state.stored_locations.track(location, &execution.threads);
            // An atomic store counts as a read access to the underlying memory
            // cell (cell-wide).
            state.track_store(&execution.threads);

            trace!(state = ?state_ref, ?ordering, ?mask, "Atomic::store_masked");

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
    fn rmw_preserving<F, E>(
        &self,
        location: Location,
        read_mask: u128,
        write_mask: u128,
        success: Ordering,
        failure: Ordering,
        f: F,
    ) -> Result<u128, E>
    where
        F: FnOnce(u128) -> Result<u128, E>,
    {
        assert!(
            write_mask & !read_mask == 0,
            "rmw_preserving: write mask {:#034x} is not contained in read mask {:#034x} \
             — an operation cannot change bits it does not consult",
            write_mask,
            read_mask,
        );

        let state_ref = self.resolve();
        // Both masks must be unions of whole regions: the read mask so the
        // value is composed from exactly the bits consulted, the write mask so
        // no region is half-written and half-preserved.
        ensure_partition(state_ref, read_mask);
        ensure_partition(state_ref, write_mask);
        branch(
            self,
            state_ref,
            Action::Rmw {
                read: read_mask,
                write: write_mask,
            },
            is_seq_cst(success) || is_seq_cst(failure),
            location,
        );

        super::synchronize(|execution| {
            let state = state_ref.get_mut(&mut execution.objects);

            state.loaded_locations.track(location, &execution.threads);
            // Track the load is happening in order to ensure correct
            // synchronization to the underlying cell (cell-wide).
            state.track_load(&execution.threads);
            // Either arm's ordering may acquire, and which one runs is not
            // known until `f` has been applied.
            state.check_preserved_scope(
                read_mask,
                acquires(success) || acquires(failure),
                &execution.threads,
            );

            trace!(state = ?state_ref, ?success, ?failure, ?read_mask, ?write_mask, "Atomic::rmw_preserving");

            // Read the current value: each covered region's RMW reads a
            // modification-order-maximal store (`match_rmw_to_stores`).
            let mut current = 0u128;
            let mut reads: SmallVec<[(usize, usize); 4]> = SmallVec::new();

            for ri in state.covered(read_mask) {
                if execution.path.is_traversed() {
                    let mut seed = [0; MAX_ATOMIC_HISTORY];
                    let n = state.regions[ri].match_rmw_to_stores(&mut seed[..]);
                    execution.path.push_load(&seed[..n]);
                }

                let index = execution.path.branch_load();
                let mask_ri = state.regions[ri].mask;
                let v = state.regions[ri].rmw_read(
                    &mut execution.threads,
                    index,
                    is_seq_cst(success) || is_seq_cst(failure),
                );
                current |= v & mask_ri;
                reads.push((ri, index));
            }

            match f(current) {
                Ok(next) => {
                    // Unconditional even when `write_mask` is empty: the
                    // hardware op owns the line and writes it, so the races
                    // these track — against `with_mut` and `unsync_load` — are
                    // real whatever the model does with the preserved bits.
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
                        let mask_ri = state.regions[ri].mask;

                        if mask_ri & write_mask != 0 {
                            state.regions[ri].rmw_commit(
                                &mut execution.threads,
                                index,
                                next,
                                success,
                                sc_rank,
                                op_id,
                            );
                            continue;
                        }

                        // The preservation claim, enforced rather than
                        // trusted: the whole elision rests on these bits
                        // coming back exactly as they were read, which is
                        // what the caller's own compare over them
                        // guarantees.
                        assert_eq!(
                            next & mask_ri,
                            current & mask_ri,
                            "rmw_preserving changed bits outside its write mask \
                             (region {:#034x}) — the preserved lane is not preserved",
                            mask_ri,
                        );

                        state.regions[ri].preserve_commit(
                            &mut execution.threads,
                            index,
                            success,
                            op_id,
                            write_mask,
                        );
                        state.check_preserved_against_prior_reads(
                            ri,
                            write_mask,
                            execution.threads.active_id().as_usize(),
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

    fn unsync_load(&self, location: Location) -> u128 {
        let state_ref = self.resolve_for_access(false);
        rt::execution(|execution| {
            let state = state_ref.get_mut(&mut execution.objects);

            state
                .unsync_loaded_locations
                .track(location, &execution.threads);

            // An unsync load counts as a "read" access
            state.track_unsync_load(&execution.threads);

            trace!(state = ?state_ref, "Atomic::unsync_load");

            // Compose the most recent value across every region.
            state.newest_value()
        })
    }

    fn with_mut<R>(&mut self, location: Location, f: impl FnOnce(&mut u128) -> R) -> R {
        let state_ref = self.resolve_for_access(true);
        let value = super::execution(|execution| {
            let state = state_ref.get_mut(&mut execution.objects);

            state
                .unsync_mut_locations
                .track(location, &execution.threads);
            // Verify the mutation may happen
            state.track_unsync_mut(&execution.threads);
            state.is_mutating = true;

            trace!(state = ?state_ref, "Atomic::with_mut");

            // Compose the most recent value across every region.
            state.newest_value()
        });

        struct Reset(u128, object::Ref<State>);

        impl Drop for Reset {
            fn drop(&mut self) {
                super::execution(|execution| {
                    let state = self.1.get_mut(&mut execution.objects);

                    // Make sure the state is as expected
                    assert!(state.is_mutating);
                    state.is_mutating = false;

                    // The value may have been mutated, so it must be placed
                    // back into every region (masked to each region's bits).
                    let val = self.0;
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
        let mut reset = Reset(value, state_ref);
        f(&mut reset.0)
    }

    fn peek(&self) -> Option<u128> {
        rt::Scheduler::try_with_execution(|execution| match self.registered(execution) {
            Ok(state) => state.get(&execution.objects).newest_value(),
            Err(unregistered) => unregistered,
        })
    }
}

/// Register the operation's DPOR branch, then check the algorithm under test
/// has not written through an invalid pointer into the cell's own memory.
/// `sc` is whether the operation takes a place in S (`SeqCst`), which makes
/// it dependent with every SC fence.
fn branch<C: Resolve + ?Sized>(
    cell: &C,
    state_ref: object::Ref<State>,
    action: Action,
    sc: bool,
    location: Location,
) {
    state_ref.branch_ordered_action(action, sc, location);
    // Re-resolve rather than compare against the ref we were handed: the point
    // of the check is that the algorithm under test has not written through an
    // invalid pointer into *this cell's own memory*, so the second read must go
    // back to the cell. For a deferred cell that means re-reading `cell_id` — a
    // corrupted identity resolves to a different registration and trips the
    // same assert.
    assert!(
        state_ref.ref_eq(cell.resolve_for_access(matches!(action, Action::Store(_)))),
        "Internal state mutated during branch. This is \
            usually due to a bug in the algorithm being tested writing in \
            an invalid memory location."
    );
}

/// Refine the region partition to `mask` **before** the DPOR branch, so
/// `set_last_access` records the access on the final per-lane regions — the
/// mask-intersection pruning needs each op's record scoped to exactly the lanes
/// it touched, or a coarse-then-split record would leak the access into a
/// disjoint lane and spuriously couple them. A full-width mask never splits, so
/// it is skipped.
fn ensure_partition(state_ref: object::Ref<State>, mask: u128) {
    if mask == FULL_MASK {
        return;
    }
    rt::execution(|execution| {
        state_ref
            .get_mut(&mut execution.objects)
            .ensure_partition(mask);
    });
}

// ===== impl State =====

impl State {
    /// An uninitialized cell shape: one full-width empty region. `init` must
    /// run before any access. Split from `init` so a carcass can supply the
    /// shape (`recycle`) with its allocations intact.
    fn shell() -> State {
        State {
            created_location: Location::default(),
            genesis: [0; MAX_THREADS],
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
            touched_by: 0,
            spares: Vec::new(),
        }
    }

    /// Return a carcass to exactly the shape `shell` constructs, keeping its
    /// allocations. Split regions park on `spares`; region 0 is reset to one
    /// empty full-width region. Stale ring contents are never zeroed: every
    /// ring read is bounded by `live_stores()` (`cnt`, reset here) and a push
    /// overwrites its slot whole.
    fn recycle(&mut self) {
        let extra = self.regions.drain(1..);
        self.spares.extend(extra);
        self.regions[0].reset(FULL_MASK);
    }

    /// Initialize a shell (fresh or recycled) as `Atomic::new` requires.
    /// Writes every non-storage field, so a recycled cell is extensionally
    /// identical to a fresh one.
    fn init(&mut self, threads: &mut thread::Set, value: u128, location: Location) {
        self.created_location = location;
        self.genesis = [0; MAX_THREADS];
        self.loaded_at = VersionVec::new();
        self.loaded_locations = LocationSet::new();
        self.unsync_loaded_at = VersionVec::new();
        self.unsync_loaded_locations = LocationSet::new();
        self.stored_at = VersionVec::new();
        self.stored_locations = LocationSet::new();
        self.unsync_mut_at = VersionVec::new();
        self.unsync_mut_locations = LocationSet::new();
        self.is_mutating = false;
        self.op_clock = 0;
        // The genesis store's `first_seen` is touched by the creating
        // thread, so seed its bit.
        self.touched_by = 1 << threads.active_id().as_usize();

        // All subsequent accesses must happen-after.
        self.track_unsync_mut(threads);

        // Store the initial thread
        //
        // The actual order shouldn't matter as operation on the atomic
        // **should** already include the thread causality resulting in the
        // creation of this atomic cell.
        //
        // This is verified using `cell`.
        self.regions[0].store(threads, Synchronize::new(), value, Ordering::Release, None, 0);
    }

    /// Initialize a shell for a `const`-constructed cell, whose genesis store
    /// precedes the execution rather than belonging to any thread in it.
    ///
    /// Differs from [`State::init`] in exactly the three places thread
    /// attribution shows up, and nowhere else:
    ///
    /// - **No `track_unsync_mut`.** `init`'s call cannot fire any of its panic
    ///   arms (every `*_at` vector was just zeroed), so its only effect there
    ///   is to seed `unsync_mut_at` with the constructing thread's causality —
    ///   the edge that makes a later unsynchronized access a reported race.
    ///   A `const`-initialized value is in the binary image before any thread
    ///   runs, so there is no such edge to record and leaving the vector empty
    ///   is the accurate model, not a relaxation of one.
    /// - **`touched_by` stays `0`.** No thread has seen the genesis store, so
    ///   acquire fences and SC promotion correctly skip this cell until one
    ///   does.
    /// - **The genesis store carries an empty causality** and `creator` lane 0
    ///   at tick 0 (`Region::store_pre_execution`), which makes it
    ///   unconditionally modification-order-first — the correct C11 reading of
    ///   an initialization, and strictly more accurate than a genesis whose
    ///   `tick` is some thread's clock.
    fn init_deferred(&mut self, value: u128, committed: Option<(Epochs, Location)>) {
        self.created_location = committed.map_or_else(Location::default, |(_, l)| l);
        // A cell materialized in committed memory must be reached through a
        // commit: every `track_*` reports an access that happens-after none of
        // the commits covering it. A `const` cell keeps the empty set — it is
        // in the binary image before any thread runs.
        self.genesis = committed.map_or([0; MAX_THREADS], |(c, _)| c);
        self.loaded_at = VersionVec::new();
        self.loaded_locations = LocationSet::new();
        self.unsync_loaded_at = VersionVec::new();
        self.unsync_loaded_locations = LocationSet::new();
        self.stored_at = VersionVec::new();
        self.stored_locations = LocationSet::new();
        self.unsync_mut_at = VersionVec::new();
        self.unsync_mut_locations = LocationSet::new();
        self.is_mutating = false;
        self.op_clock = 0;
        self.touched_by = 0;

        // The genesis *store* stays pre-execution even when committed: it
        // carries no causality, so an acquiring reader inherits nothing it did
        // not earn. A commit is the constraint above, not a happens-before
        // edge.
        self.regions[0].store_pre_execution(value);
    }

    /// Allocate the next store-op id for this cell. A wide op passes the same
    /// id to every region it writes (siblings); a masked op gets a fresh id.
    fn next_op_id(&mut self) -> u64 {
        self.op_clock += 1;
        self.op_clock
    }

    /// Guard the one behavior a preserving RMW gives up (see [`PreservedOp`]):
    /// the elided identity write cannot be read-from, so an acquiring read
    /// confined to the preserved bits can no longer synchronize-with the op.
    ///
    /// A read is *routed* around a preserving op X when it also reaches a
    /// region X wrote — there X's store is present and its release still
    /// available, so nothing is lost. A thread that ran X itself is routed
    /// trivially. An unrouted read is lost causality, and the two ways to
    /// consume it are handled separately:
    ///
    /// - **acquiring now** — the edge would be taken at this operation, so
    ///   this is the loss, and it traps;
    /// - **acquiring later, through a fence** — the read itself takes nothing,
    ///   so it is sound on its own; the thread is recorded on the region and
    ///   `fence_acq` traps if it ever reaches the fence.
    ///
    /// This is only the read end. The op end is
    /// [`Self::check_preserved_against_prior_reads`], and both are needed: the
    /// two are DPOR-independent, so the search may explore either order and
    /// only that order.
    fn check_preserved_scope(&mut self, op_mask: u128, acquiring: bool, threads: &thread::Set) {
        let reader = threads.active_id().as_usize();
        let bit = 1u32 << reader;

        for region in &mut self.regions {
            if region.mask & op_mask == 0 {
                continue;
            }

            // Recorded for the op end, whether or not this cell has ever seen
            // a preserving op — the op that will ask has not run yet.
            region.readers |= bit;
            if acquiring {
                region.acquiring_readers |= bit;
            }

            for i in 0..region.live_preserved() {
                let p = region.preserved[i];
                if op_mask & p.write_mask != 0 || p.creator == reader {
                    continue;
                }

                assert!(
                    !acquiring,
                    "acquiring read of a preserved lane (region {:#034x}) that cannot reach \
                     the preserving operation's own bits ({:#034x}).\n\
                     `rmw_preserving` elides the identity write on the preserved lane, so \
                     there is no store here to read-from and no release to acquire. The \
                     declaration that the lane publishes nothing is wrong for this cell — \
                     use `rmw_masked` for that operation, or stop acquiring here.",
                    region.mask,
                    p.write_mask,
                );

                region.unrouted_readers |= bit;
            }
        }
    }

    /// The op end of [`Self::check_preserved_scope`]: reads of the carried
    /// lane that already happened, which the op is about to render unroutable.
    ///
    /// `ri` is the region being preserved and `write_mask` the bits the op
    /// does write. A prior reader is routed if it also read some region under
    /// `write_mask` — for the acquire test it must have acquired there, since
    /// that is what would carry the release; for the fence test any read will
    /// do, because a fence collects from every store the thread touched.
    fn check_preserved_against_prior_reads(&mut self, ri: usize, write_mask: u128, creator: usize) {
        let mine = 1u32 << creator;
        let readers = self.regions[ri].readers & !mine;
        let acquirers = self.regions[ri].acquiring_readers & !mine;

        if readers == 0 {
            return;
        }

        let mut routed = 0u32;
        let mut acq_routed = 0u32;
        for region in &self.regions {
            if region.mask & write_mask != 0 {
                routed |= region.readers;
                acq_routed |= region.acquiring_readers;
            }
        }

        assert!(
            acquirers & !acq_routed == 0,
            "preserving RMW over a lane (region {:#034x}) another thread has already \
             acquire-read without reaching the bits this operation writes ({:#034x}).\n\
             `rmw_preserving` elides the identity write on the preserved lane, so that \
             reader has no store of this operation to have read-from. Use `rmw_masked` \
             here, or stop acquiring through the carried lane.",
            self.regions[ri].mask,
            write_mask,
        );

        self.regions[ri].unrouted_readers |= readers & !routed;
    }

    /// True when the active thread holds a read of some preserved lane of this
    /// cell that an acquire fence would try, and fail, to draw causality from
    /// (`check_preserved_scope`).
    fn has_unrouted_reader(&self, thread: usize) -> bool {
        let bit = 1u32 << thread;
        self.regions.iter().any(|r| r.unrouted_readers & bit != 0)
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
                let spare = self.spares.pop();
                let split = self.regions[i].split_off(inside, spare);
                self.regions.push(split);
            }

            i += 1;
        }
    }

    /// Indices of the regions covered by `mask` (those whose bits intersect
    /// it). After `ensure_partition(mask)` every such region is fully inside
    /// the mask.
    fn covered(&self, mask: u128) -> SmallVec<[usize; 4]> {
        // Regions partition the cell, so a full-width mask covers all of them:
        // skip the per-region intersection test on the common path.
        if mask == FULL_MASK {
            return (0..self.regions.len()).collect();
        }

        self.regions
            .iter()
            .enumerate()
            .filter(|(_, r)| r.mask & mask != 0)
            .map(|(i, _)| i)
            .collect()
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

    /// Compose the value an atomic load returns across the regions the mask
    /// covers, taking the one per-region *value* branch each (`push_load` /
    /// `branch_load`) — the DPOR *dependence* branch is the caller's single
    /// `branch(Action::Load(mask))`. Shared by `load_masked` (`apply_floor =
    /// false`) and `load_coherent_lane` (`apply_floor = true`).
    ///
    /// A multi-region load must return a single consistent snapshot: wide
    /// (multi-region) ops are seen all-or-none so a load never tears one.
    /// `resolved` carries the wide-op visibility fixed by the regions already
    /// read; each region's readable set is filtered to agree with it. Two
    /// *separate* masked ops to different lanes share no `op_id`, so this never
    /// couples independent lanes — they stay free to be read in either order.
    ///
    /// With `apply_floor`, each region's readable set is additionally narrowed
    /// by `filter_seen_op_floors` to the whole-cell-coherent projection the
    /// typed lane views promise.
    fn compose_load(
        &mut self,
        path: &mut Path,
        threads: &mut thread::Set,
        covered: &[usize],
        ordering: Ordering,
        apply_floor: bool,
    ) -> u128 {
        let multi = covered.len() > 1;
        let mut resolved = Resolved::new();
        let mut result = 0u128;
        // The `(region, slot)` pairs this load has read so far.
        let mut read: SmallVec<[(usize, usize); 4]> = SmallVec::new();

        for (k, &ri) in covered.iter().enumerate() {
            // If necessary, generate the list of stores to permute through for
            // this region.
            //
            // A `SeqCst` load participates in the SC total order S; a load past
            // a `SeqCst` fence is bounded by the fence's position. The readable
            // set is restricted inside `match_load_to_stores`.
            if path.is_traversed() {
                // The regions before this one have already been read, so the
                // live state *is* the projection for this step: their causality
                // joins and `first_seen` touches are already recorded. A touch
                // does not say it came from this load, so `touched` still
                // names what the load read, exactly as the lookahead does.
                let view = LoadView::entry(threads, ordering, &read);

                let mut seed = [0; MAX_ATOMIC_HISTORY];
                let mut n = self.regions[ri].match_load_to_stores(
                    threads,
                    &view.causality,
                    &mut seed[..],
                    ordering,
                );

                // Whole-cell coherence for a typed lane load: drop candidates
                // that would travel behind a wide op already observed through
                // another region. `load_masked` skips this (per-lane coherent).
                if apply_floor {
                    n = self.filter_seen_op_floors(ri, &view, &mut seed[..], n);
                }

                if multi {
                    // Keep only candidates consistent with the wide-op
                    // visibility earlier regions committed to *and* completable
                    // by the regions still to come. The forward half is what
                    // makes the walk total: regions resolve in `covered` order,
                    // but a later region's readable set can be pinned by plain
                    // coherence to a store that sees wide op X, which forces
                    // *every* region of this one single-copy-atomic load to see
                    // X. Filtering on `resolved` alone lets an earlier region
                    // commit to not-seeing-X, and the pinned region then has
                    // nothing left — a dead end the walk cannot back out of,
                    // reported as an internal bug. Offering only completable
                    // prefixes prunes exactly those; every whole-cell snapshot
                    // that some assignment realizes stays reachable.
                    let rest = &covered[k + 1..];
                    let mut w = 0;
                    for r in 0..n {
                        let ci = seed[r] as usize;

                        // The prefix is a stack: extend it for this candidate,
                        // test, wind back. Nothing is committed here — the walk
                        // extends `resolved` for real only once `branch_load`
                        // has chosen.
                        let mark = resolved.len();
                        if !self.regions[ri].try_resolve(ci, &mut resolved) {
                            continue;
                        }
                        let complete = rest.is_empty() || {
                            let next_view = view.extend(&self.regions[ri], ri, ci, ordering);
                            self.has_consistent_completion(
                                rest,
                                threads,
                                ordering,
                                apply_floor,
                                &mut resolved,
                                &next_view,
                            )
                        };
                        resolved.truncate(mark);

                        if !complete {
                            continue;
                        }
                        seed[w] = seed[r];
                        w += 1;
                    }
                    // Now a genuine claim about the cell, not about walk order:
                    // no assignment of stores to `covered` forms a consistent
                    // whole-cell snapshot.
                    assert!(
                        w > 0,
                        "[loom internal bug] no consistent store for a wide load"
                    );
                    n = w;
                }

                path.push_load(&seed[..n]);
            }

            let index = path.branch_load();
            if multi {
                // The chosen candidate is one the filter above passed — a
                // replayed path replays a seed that same filter recorded — so
                // it agrees with the prefix by construction.
                assert!(
                    self.regions[ri].try_resolve(index, &mut resolved),
                    "[loom internal bug] wide load committed to an inconsistent store"
                );
            }
            let mask_ri = self.regions[ri].mask;
            let v = self.regions[ri].load(threads, index, ordering);
            result |= v & mask_ri;
            if apply_floor && multi {
                read.push((ri, index));
            }
        }

        result
    }

    /// Can `rest` — the regions of a multi-region load not yet resolved — be
    /// assigned stores consistent with `resolved`? Depth-first over the
    /// regions in order, so a prefix is rejected only when *no* completion
    /// exists. Exact rather than per-region approximate: a conflict can span
    /// two later regions, and both sets are bounded (regions per cell, and
    /// `MAX_ATOMIC_HISTORY` candidates each).
    ///
    /// Reads nothing the caller has not already fixed — `match_load_to_stores`
    /// and `filter_seen_op_floors` both take `&self`/`&thread::Set` — so the
    /// lookahead cannot perturb the execution it is predicting.
    fn has_consistent_completion(
        &self,
        rest: &[usize],
        threads: &thread::Set,
        ordering: Ordering,
        apply_floor: bool,
        resolved: &mut Resolved,
        view: &LoadView,
    ) -> bool {
        let Some((&rj, tail)) = rest.split_first() else {
            return true;
        };

        let mut seed = [0; MAX_ATOMIC_HISTORY];
        let mut n = self.regions[rj].match_load_to_stores(
            threads,
            &view.causality,
            &mut seed[..],
            ordering,
        );
        if apply_floor {
            n = self.filter_seen_op_floors(rj, view, &mut seed[..], n);
        }

        for r in 0..n {
            let ci = seed[r] as usize;

            let mark = resolved.len();
            if !self.regions[rj].try_resolve(ci, resolved) {
                continue;
            }
            if tail.is_empty() {
                resolved.truncate(mark);
                return true;
            }
            let next_view = view.extend(&self.regions[rj], rj, ci, ordering);
            let complete = self.has_consistent_completion(
                tail,
                threads,
                ordering,
                apply_floor,
                resolved,
                &next_view,
            );
            resolved.truncate(mark);

            if complete {
                return true;
            }
        }

        false
    }

    /// Whole-cell coherence for a typed lane load (module docs, "Typed lane
    /// loads"): drop every candidate in `seed[..n]` for region `ri` that is
    /// modification-order-before a region-`ri` store whose whole-cell op the
    /// active thread has already observed through *another* region of this
    /// cell. Returns the retained count.
    ///
    /// A region-`ri` store `f` is a **floor** when its op `f.op_id` is observed
    /// through some other region `rj` (`op_seen_through_other_region`) — the
    /// thread has provably passed that one indivisible single-copy-atomic event
    /// in `rj`, so reading this lane from before `f` would travel backwards
    /// through it. Only genuine `mo_before` edges exclude: a racing store
    /// mo-incomparable to `f` stays readable (per-byte coherence permits either
    /// order until something orders them), and same-region observations need no
    /// floor — the plain coherence rules in `match_load_to_stores` cover them.
    ///
    /// The result can never be empty: a floor `f` is never mo-before itself,
    /// and any candidate dropped by the "saw a newer store" rule in
    /// `match_load_to_stores` is superseded by a store mo-after it, which
    /// clears every floor too. The RMW read path needs no such filter —
    /// `match_rmw_to_stores` offers only mo-maximal stores, and a candidate
    /// mo-before a floor has a known mo successor, so it was never offered.
    fn filter_seen_op_floors(
        &self,
        ri: usize,
        view: &LoadView,
        seed: &mut [u8],
        n: usize,
    ) -> usize {
        // A cell never touched by a masked op has one region — no siblings.
        if self.regions.len() <= 1 {
            return n;
        }

        let region = &self.regions[ri];
        let mut w = 0;

        // `op_seen_through_other_region` is an O(regions x stores) sweep whose
        // answer depends only on the floor slot (`f.op_id` is a function of
        // `f_idx` — one region holds one store per op). The candidate loop
        // re-asks it for the same floors up to `n` times, so resolve each slot
        // at most once. `None` = not yet asked; the floor is only consulted
        // when `mo_before` holds, so this stays lazy.
        let live = region.live_stores();
        let mut seen_memo: [Option<bool>; MAX_ATOMIC_HISTORY] = [None; MAX_ATOMIC_HISTORY];

        // The same memo for the floors a preserving op contributes. Its floor
        // is the store it read: the identity write it replaces sat immediately
        // after that store, so "mo-before the identity write" and "strictly
        // mo-before the store it read" name the same candidates
        // (`PreservedOp`). The store itself stays readable — it is the node the
        // identity write's readers now land on.
        let live_pre = region.live_preserved();
        let mut pre_memo: [Option<bool>; MAX_ATOMIC_HISTORY] = [None; MAX_ATOMIC_HISTORY];

        'candidate: for k in 0..n {
            let c = &region.stores[seed[k] as usize];

            for f_idx in 0..live {
                let f = &region.stores[f_idx];
                if !mo_before(c, f) {
                    continue;
                }

                let seen = match seen_memo[f_idx] {
                    Some(seen) => seen,
                    None => {
                        let seen = self.op_seen_through_other_region(ri, f.op_id, view);
                        seen_memo[f_idx] = Some(seen);
                        seen
                    }
                };

                if seen {
                    continue 'candidate;
                }
            }

            for p_idx in 0..live_pre {
                let p = &region.preserved[p_idx];

                // A floor the ring no longer holds cannot be compared against;
                // it lapses with the store, exactly as an evicted store's own
                // floor does.
                let Some(s_idx) = region.slot_of_store(p.read.read_id) else {
                    continue;
                };
                if !mo_before(c, &region.stores[s_idx]) {
                    continue;
                }

                let seen = match pre_memo[p_idx] {
                    Some(seen) => seen,
                    None => {
                        let seen = self.op_seen_through_other_region(ri, p.op_id, view);
                        pre_memo[p_idx] = Some(seen);
                        seen
                    }
                };

                if seen {
                    continue 'candidate;
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

    /// True when the active thread has observed whole-cell op `op_id` through
    /// some region other than `ri` — i.e. that region holds a store `g` which
    /// is op `op_id`'s sibling there or modification-order-after it, and which
    /// the thread observed in a way ordered before this load
    /// ([`LoadView::is_seen`]).
    ///
    /// The thread may have *read* the wide op through another lane (`g` is
    /// that op's sibling, or a later store it read) or *written* that lane
    /// past it (`g` is its own store). Either way the floor stands only when
    /// the observation is ordered before this load on both targets; a
    /// genuinely-concurrent op, or an unordered observation, never floors.
    fn op_seen_through_other_region(&self, ri: usize, op_id: u64, view: &LoadView) -> bool {
        for (rj, other) in self.regions.iter().enumerate() {
            if rj == ri {
                continue;
            }

            // The op's position in `other` is a property of `(other, op_id)`,
            // not of the candidate `g`: resolve it once per region rather than
            // re-searching the ring for every `g`. A region the op neither
            // wrote nor preserved holds no constraint at all and is skipped
            // whole.
            let Some(pin) = other.pin_of_op(op_id) else {
                continue;
            };

            for g_idx in 0..other.live_stores() {
                let g = &other.stores[g_idx];

                // Structural test first — an id compare and a single-lane
                // marker read — ahead of `is_seen`'s all-lane compare and
                // touched-list scan. Both are pure, so the order is free.
                if other.pin_sees(&pin, g) && view.is_seen(g, rj, g_idx) {
                    return true;
                }
            }
        }
        false
    }

    /// Promote every live store seen in `view` into S at `pos`, across every
    /// region (`Region::promote_sc_writes`).
    pub(super) fn promote_sc_writes(&mut self, view: &VersionVec, pos: u32) {
        for region in &mut self.regions {
            region.promote_sc_writes(view, pos);
        }
    }

    /// Report an access to a materialized cell that happens-after none of the
    /// commits of its memory.
    fn check_genesis(&self, threads: &thread::Set, access: &str, location: Location) {
        if self.genesis == [0; MAX_THREADS]
            || epoch_reached(&self.genesis, &threads.active().causality)
        {
            return;
        }
        std::hint::cold_path();

        let committer = self.genesis.iter().position(|&t| t != 0).unwrap_or(0);
        location::panic(
            "Causality violation: access to committed memory without synchronizing-with its commit.",
        )
        .thread("commit", committer, self.created_location)
        .thread(access, threads.active_id(), location)
        .fire();
    }

    /// Track an atomic load
    fn track_load(&mut self, threads: &thread::Set) {
        assert!(!self.is_mutating, "atomic cell is in `with_mut` call");
        self.check_genesis(threads, "load", self.loaded_locations[threads]);

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
        self.check_genesis(threads, "unsync_load", self.unsync_loaded_locations[threads]);

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
        self.check_genesis(threads, "atomic store", self.stored_locations[threads]);

        // This op creates a store whose `first_seen` is touched by the active
        // thread; record it so acquire fences can skip this cell if no thread
        // of theirs ever touched it.
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
        self.check_genesis(threads, "with_mut", self.unsync_mut_locations[threads]);

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

    /// Track a decommit of the cell's memory: a non-atomic write that every
    /// prior access, by any thread, must happen-before.
    fn track_decommit(&mut self, threads: &thread::Set, location: Location) {
        assert!(!self.is_mutating, "atomic cell is in `with_mut` call");

        let current = &threads.active().causality;
        let prior = [
            ("atomic load", &self.loaded_at, &self.loaded_locations),
            ("atomic store", &self.stored_at, &self.stored_locations),
            ("unsync_load", &self.unsync_loaded_at, &self.unsync_loaded_locations),
            ("with_mut", &self.unsync_mut_at, &self.unsync_mut_locations),
        ];
        for (access, at, locations) in prior {
            if let Some(thread) = current.ahead(at) {
                location::panic(
                    "Causality violation: decommit of memory a thread accessed without \
                     happening-before the decommit.",
                )
                .location("created", self.created_location)
                .thread(access, thread, locations[thread])
                .thread("decommit", threads.active_id(), location)
                .fire();
            }
        }
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
    ///
    /// The op's kind is decided **per region**, not once: a preserving RMW is a
    /// writer in the regions its `write` mask covers and a reader in the ones
    /// only its `read` mask does, so its preserved lane raises no backtrack
    /// point against that lane's readers.
    pub(super) fn for_each_dependent_access<'a>(
        &'a self,
        action: Action,
        mut f: impl FnMut(&'a Access),
    ) {
        let mask = action.mask();
        let write = action.write_mask();

        for region in &self.regions {
            if region.mask & mask != 0 {
                region.for_each_dependent_access(region.mask & write == 0, &mut f);
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
        let write = action.write_mask();
        let index = thread_id.as_usize();

        for region in &mut self.regions {
            if region.mask & mask != 0 {
                region.set_last_access(region.mask & write == 0, index, path_id, version);
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
            preserved: Default::default(),
            preserved_cnt: 0,
            unrouted_readers: 0,
            readers: 0,
            acquiring_readers: 0,
        }
    }

    /// Reset to an empty region owning `mask`, keeping the allocations. The
    /// ring is not zeroed: `cnt = 0` puts every slot outside `live_stores()`
    /// and a push overwrites its slot whole, so stale bytes are unreachable.
    fn reset(&mut self, mask: u128) {
        self.mask = mask;
        self.cnt = 0;
        *self.last_access = Default::default();
        *self.last_non_load_access = Default::default();
        self.preserved_cnt = 0;
        self.unrouted_readers = 0;
        self.readers = 0;
        self.acquiring_readers = 0;
    }

    /// Keep the `keep_mask` bits of this region in place; split the remaining
    /// bits into a new region that inherits a full copy of the history, **the
    /// DPOR access records and the preserving-op records**. Both halves start
    /// perfectly coherent
    /// (identical stores) and carry the same past accesses — a peer op that
    /// conflicted with the pre-split region conflicts with whichever half it
    /// still overlaps — and diverge only as future masked ops touch one but
    /// not the other.
    ///
    /// `spare` recycles a previous epoch's region: its allocations receive
    /// the copies (`clone_from` clones contents in place).
    fn split_off(&mut self, keep_mask: u128, spare: Option<Region>) -> Region {
        let other_mask = self.mask & !keep_mask;
        self.mask &= keep_mask;

        match spare {
            Some(mut region) => {
                region.mask = other_mask;
                region.cnt = self.cnt;
                region.stores.clone_from(&self.stores);
                region.last_access.clone_from(&self.last_access);
                region
                    .last_non_load_access
                    .clone_from(&self.last_non_load_access);
                region.preserved = self.preserved;
                region.preserved_cnt = self.preserved_cnt;
                region.unrouted_readers = self.unrouted_readers;
                region.readers = self.readers;
                region.acquiring_readers = self.acquiring_readers;
                region
            }
            None => Region {
                mask: other_mask,
                stores: self.stores.clone(),
                cnt: self.cnt,
                last_access: self.last_access.clone(),
                last_non_load_access: self.last_non_load_access.clone(),
                preserved: self.preserved,
                preserved_cnt: self.preserved_cnt,
                unrouted_readers: self.unrouted_readers,
                readers: self.readers,
                acquiring_readers: self.acquiring_readers,
            },
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
        debug_assert!(index < self.live_stores(), "load of dead slot");
        // Apply coherence rules
        let sc_scope = threads.active_sc_scope(is_seq_cst(ordering));
        self.apply_load_coherence(threads, index, sc_scope);

        let store = &mut self.stores[index];

        store.first_seen.touch(threads);
        if acquires(ordering) {
            store.first_seen.acquire(threads);
        }
        store.sync.sync_load(threads, ordering);
        store.value
    }

    /// The slot holding op `op_id`'s store in this region, or `None` when the
    /// op never wrote these bits (it constrains nothing here).
    ///
    /// A store op appends exactly one store per region it covers, so the slot
    /// is unique — which is what lets a caller resolve the op *once* for a
    /// region instead of re-searching the ring per candidate.
    fn slot_of_op(&self, op_id: u64) -> Option<usize> {
        (0..self.live_stores()).find(|&i| self.stores[i].op_id == op_id)
    }

    /// Where op `op_id` sits in this region's modification order, or `None`
    /// when the op neither wrote nor preserved these bits and so constrains
    /// nothing here.
    ///
    /// An op reaches a region in exactly one of two ways, and both pin it to
    /// one point of the order — its own store, or the store it read and wrote
    /// back verbatim. [`PreservedOp`] is where the two pins are shown to
    /// select the same candidates.
    fn pin_of_op(&self, op_id: u64) -> Option<OpPin> {
        if let Some(slot) = self.slot_of_op(op_id) {
            return Some(OpPin::Wrote(slot));
        }

        self.preserved_ops()
            .find(|p| p.op_id == op_id)
            .map(|p| OpPin::Preserved(p.read))
    }

    /// Does reading store `c` imply having seen the op pinned at `pin`?
    ///
    /// - **Wrote** — `c` is the op's own store, or modification-order after it.
    /// - **Preserved** — `c` is *strictly* modification-order after the store
    ///   the op read. Strict is the deliberate side of an ambiguity the
    ///   elision creates, argued at [`PreservedOp`]: that store is also what a
    ///   reader running before the op sees, and calling it a witness would
    ///   floor such a reader's sibling lanes for an op it never observed.
    fn pin_sees(&self, pin: &OpPin, c: &Store) -> bool {
        match pin {
            OpPin::Wrote(slot) => {
                let d = &self.stores[*slot];
                c.id == d.id || mo_before(d, c)
            }
            OpPin::Preserved(read) => stamp_mo_before(read, c),
        }
    }

    /// The live slot holding the store with `id`, if the ring still has it.
    fn slot_of_store(&self, id: u16) -> Option<usize> {
        (0..self.live_stores()).find(|&i| self.stores[i].id == id)
    }

    /// Number of `preserved` slots holding a real record — the ring discipline
    /// of `live_stores`, for the same reason.
    fn live_preserved(&self) -> usize {
        cmp::min(self.preserved_cnt as usize, MAX_ATOMIC_HISTORY)
    }

    /// The live preserving-op records over these bits. Slot order, like
    /// `live_stores`: the ring's rotation is immaterial because every consumer
    /// asks each record an independent question.
    fn preserved_ops(&self) -> impl Iterator<Item = &PreservedOp> {
        self.preserved[..self.live_preserved()].iter()
    }

    /// Record a wide op that compared these bits and wrote them back verbatim.
    ///
    /// `index` is the slot the op read here; its creation stamp is captured
    /// now, so the record survives that store's eviction exactly as
    /// `Store::rmw_read` does. The op's `success` ordering performs its load
    /// synchronization just as a committing RMW's does — the op *did* read
    /// these bits — but no store is appended, so this region gains no
    /// coherence node, no candidate, and no modification order.
    fn preserve_commit(
        &mut self,
        threads: &mut thread::Set,
        index: usize,
        success: Ordering,
        op_id: u64,
        write_mask: u128,
    ) {
        debug_assert!(index < self.live_stores(), "preserve_commit of dead slot");
        self.stores[index].sync.sync_load(threads, success);
        if acquires(success) {
            self.stores[index].first_seen.acquire(threads);
        }

        let read = RmwRead {
            read_id: self.stores[index].id,
            creator: self.stores[index].creator,
            tick: self.stores[index].tick(),
        };

        self.preserved[self::index(self.preserved_cnt)] = PreservedOp {
            op_id,
            read,
            write_mask,
            creator: threads.active_id().as_usize(),
        };
        self.preserved_cnt += 1;
    }

    /// Check candidate `c_index` against the wide-op visibility `resolved`
    /// already fixes and, if they agree, extend `resolved` with the visibility
    /// this region's read newly implies.
    ///
    /// The rule is single-copy atomicity: a wide op's stores are seen
    /// all-or-none, so every region of one load must agree on whether it sees
    /// op X. Reading store `c` sees op X here iff `c` *is* X's store in this
    /// region or is modification-order after it. Ops only this region holds are
    /// unconstrained and simply join `resolved` for the regions still to come.
    ///
    /// One scan of the ring does both halves: every live store here is either
    /// an op `resolved` already speaks for (check it) or one it does not
    /// (record it).
    ///
    /// A **preserving** op reaches this region without a store and imposes
    /// nothing here, deliberately. All-or-none exists to forbid a torn
    /// snapshot — half an op's bytes with half a peer's — and a preserved lane
    /// has no such half: the op wrote the value it read, so "before the op"
    /// and "after the op" are the same bits. There is nothing for the regions
    /// to disagree about. Constraining it would instead be actively wrong: it
    /// would demand a candidate strictly mo-after the store the op read, and
    /// on a quiescent lane no such store exists, so a snapshot that saw the op
    /// through a written region would have no completion at all.
    ///
    /// On disagreement `resolved` is restored to its entry length, so a
    /// rejected candidate leaves nothing behind and the caller may try the next
    /// one against the same prefix.
    fn try_resolve(&self, c_index: usize, resolved: &mut Resolved) -> bool {
        let mark = resolved.len();
        let c = &self.stores[c_index];

        for i in 0..self.live_stores() {
            let d = &self.stores[i];
            let seen = c.id == d.id || mo_before(d, c);

            // Copied out, so the search's borrow ends before the push.
            let fixed = resolved
                .iter()
                .find(|(o, _)| *o == d.op_id)
                .map(|&(_, s)| s);

            match fixed {
                Some(s) if s != seen => {
                    resolved.truncate(mark);
                    return false;
                }
                Some(_) => {}
                None => resolved.push((d.op_id, seen)),
            }
        }

        true
    }

    /// Plant the genesis store of a `const`-constructed cell: an
    /// initialization that precedes the execution and belongs to no thread.
    ///
    /// Every field that `store` would derive from the active thread is instead
    /// the empty/neutral value, and the effects `store` has on thread state
    /// (`sync_store`, `first_seen.touch`) are simply absent — there is no
    /// thread to have released or seen anything.
    ///
    /// `creator: 0` with an empty `happens_before` gives `tick() == 0`, so
    /// `mo_before(genesis, x)` holds for every other store `x` and
    /// `mo_before(x, genesis)` for none: the initialization is
    /// modification-order-first, unconditionally and by construction. That is
    /// what C11 says an initialization is, and it is why lane 0 being a real
    /// thread's lane is harmless — a real store's `tick()` is its creator's
    /// own clock, which `thread::Set` has already advanced past 0.
    fn store_pre_execution(&mut self, value: u128) {
        debug_assert_eq!(
            self.cnt, 0,
            "pre-execution genesis into a region that already has stores"
        );

        self.stores[0] = Store {
            value,
            happens_before: VersionVec::new(),
            modification_order: VersionVec::new(),
            id: 0,
            op_id: 0,
            creator: 0,
            rmw_read: None,
            sync: Synchronize::new(),
            // Untouched: no thread has seen this store yet, including the one
            // that will read it first.
            first_seen: FirstSeen::new(),
            sc_rank: None,
            seq_cst: false,
        };
        self.cnt = 1;
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

        // The modification order is initialized to the thread's current
        // causality. All reads / writes that happen before this store are
        // ordered before the store.
        let happens_before = threads.active().causality;

        // Starting with the thread's coherence view covers WRITE-WRITE
        // coherence, including against what `SeqCst` fences ordered before
        // this store (`Thread::coherence_view`).
        let mut modification_order = threads.active().coherence_view();

        // The SC scope this store is placed under: all of S for a store that
        // participates in S (its position is allocated once per op, shared
        // across the regions a wide store spans, and handed in), else the
        // latest SC fence that happens before it.
        let sc_scope = threads.active_sc_scope(sc_rank.is_some());

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
            // (`promote_sc_writes`) — is therefore mo-before this SeqCst store,
            // and every one ranked within the scope of a fence happening before
            // this store is mo-before it too (C++20 [atomics.order] p4.3).
            // This is a per-location, S-only mo edge — joined into
            // `modification_order`, never into causality (the S edge orders the
            // writes without manufacturing happens-before). Timing is exact:
            // only stores already SC-ranked when this store commits are joined,
            // matching that only they precede it in S.
            if store_i.first_seen.is_seen_by_current(threads)
                || happens_before.lane(store_i.creator) >= store_i.tick()
                || store_i.in_sc_scope(sc_scope)
            {
                let mo = store_i.modification_order;
                modification_order.join(&mo);
            }
        }

        // RMW Atomicity: everything mo-after an RMW's read store is mo-after
        // the RMW's write. Runs against the pre-push ring: `cnt` is not yet
        // incremented, so the slot this store will occupy is outside
        // `live_stores()` and its previous contents — dead sentinel or a
        // reincarnated cell's stale carcass — are structurally unreadable.
        // (`cnt` incremented early here once relied on the sentinel's
        // `rmw_read: None` to keep this scan benign.)
        self.close_rmw_atomicity(&mut modification_order, id);

        sync.sync_store(threads, ordering);

        let mut first_seen = FirstSeen::new();
        first_seen.touch(threads);

        // Track the store: write the slot whole, then publish it by count.
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
            seq_cst: is_seq_cst(ordering),
        };
        self.cnt += 1;
    }

    /// The read half of an RMW: apply load coherence and return the read
    /// value. The caller composes it across regions; `rmw_commit` or
    /// `rmw_fail` follows.
    fn rmw_read(&mut self, threads: &mut thread::Set, index: usize, seq_cst: bool) -> u128 {
        debug_assert!(index < self.live_stores(), "rmw_read of dead slot");
        // Apply coherence rules.
        let sc_scope = threads.active_sc_scope(seq_cst);
        self.apply_load_coherence(threads, index, sc_scope);

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
        debug_assert!(index < self.live_stores(), "rmw_commit of dead slot");
        // Perform load synchronization using the `success` ordering.
        self.stores[index].sync.sync_load(threads, success);
        if acquires(success) {
            self.stores[index].first_seen.acquire(threads);
        }

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

    /// Append a store modification-order-after every live store of the region,
    /// attributed to the active thread and releasing nothing.
    fn store_mo_max(&mut self, threads: &mut thread::Set, value: u128, op_id: u64) {
        let happens_before = threads.active().causality;
        let mut modification_order = threads.active().coherence_view();
        for store in &self.stores[..self.live_stores()] {
            modification_order.join(&store.modification_order);
        }

        let mut first_seen = FirstSeen::new();
        first_seen.touch(threads);

        self.stores[index(self.cnt)] = Store {
            value,
            happens_before,
            modification_order,
            id: self.cnt,
            op_id,
            creator: threads.active_id().as_usize(),
            rmw_read: None,
            sync: Synchronize::new(),
            first_seen,
            sc_rank: None,
            seq_cst: false,
        };
        self.cnt += 1;
    }

    /// The failed-compare-exchange path: a load synchronizing with `failure`.
    fn rmw_fail(&mut self, threads: &mut thread::Set, index: usize, failure: Ordering) {
        debug_assert!(index < self.live_stores(), "rmw_fail of dead slot");
        self.stores[index].sync.sync_load(threads, failure);
        if acquires(failure) {
            self.stores[index].first_seen.acquire(threads);
        }
    }

    /// `sc_scope` is the read's SC scope (`thread::Set::active_sc_scope`).
    fn apply_load_coherence(
        &mut self,
        threads: &mut thread::Set,
        index: usize,
        sc_scope: Option<u32>,
    ) {
        for i in 0..self.live_stores() {
            // Skip if the is current.
            if index == i {
                continue;
            }

            // SC scope: the read may not be coherence-ordered before an
            // SC-ranked store within its scope, so the store it reads is
            // mo-after every such store (C++20 [atomics.order] p4.1-p4.3).
            // The candidate filters kept only stores not already mo-before
            // one; this fixes the order against those left incomparable.
            if self.stores[i].in_sc_scope(sc_scope) {
                let mo = self.stores[i].modification_order;
                self.stores[index].modification_order.join(&mo);
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
        view: &VersionVec,
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
        //
        // `is_seen_in` is an all-lane compare that depends only on `j`,
        // yet the pair loop can re-ask it for the same store once per `i`.
        // Resolve each store at most once, and only if some `mo_before` edge
        // actually reaches it — a lazy memo does no work the pair loop did not
        // already require.
        let mut seen_memo: [Option<bool>; MAX_ATOMIC_HISTORY] = [None; MAX_ATOMIC_HISTORY];

        'outer: for i in 0..live {
            let store_i = &self.stores[i];

            // Depends only on `i`; the `mo_before` guard below still gates
            // whether it is consulted at all.
            let seen_before_yield_i = store_i.first_seen.is_seen_before_yield(threads);

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

                    let seen_j = match seen_memo[j] {
                        Some(seen) => seen,
                        None => {
                            let seen = store_j.first_seen.is_seen_in(view);
                            seen_memo[j] = Some(seen);
                            seen
                        }
                    };

                    if seen_j {
                        // Store `j` is newer, so don't store the current one.
                        continue 'outer;
                    }

                    if seen_before_yield_i {
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

    /// Promote into the SC total order S, at an executing `SeqCst` fence's
    /// position `pos`, every live store seen in `view`, the fence's causality:
    /// those that happen before the fence, and those an operation happening
    /// before it read — `first_seen` holds both (C++20 [atomics.order] p4.2).
    /// A store already SC-ranked keeps its own — necessarily earlier —
    /// position.
    fn promote_sc_writes(&mut self, view: &VersionVec, pos: u32) {
        for store in self.stores_mut() {
            if store.sc_rank.is_none() && store.first_seen.is_seen_in(view) {
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
            seq_cst: false,
        }
    }
}

// ===== impl FirstSeen =====

impl FirstSeen {
    fn new() -> FirstSeen {
        FirstSeen([u16::max_value(); VersionVec::LANES], 0)
    }

    /// Record that the active thread read the store acquiring. The caller has
    /// touched the store first.
    fn acquire(&mut self, threads: &thread::Set) {
        self.1 |= 1 << threads.active_id().as_usize();
    }

    fn touch(&mut self, threads: &thread::Set) {
        if self.0[threads.active_id().as_usize()] == u16::max_value() {
            self.0[threads.active_id().as_usize()] = threads.active_atomic_version();
        }
    }

    /// Seen in the active thread's coherence view (`Thread::coherence_view`).
    fn is_seen_by_current(&self, threads: &thread::Set) -> bool {
        self.is_seen_in(&threads.active().coherence_view())
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

    /// The threads whose first sight of the store is contained in `view`, one
    /// bit each — `is_seen_in` split by thread.
    fn seen_threads(&self, view: &VersionVec) -> u32 {
        let lanes = view.lanes();
        let mut mask = 0u32;
        for i in 0..VersionVec::LANES {
            let fs = self.0[i];
            mask |= (((fs != <u16>::MAX) & (fs <= lanes[i])) as u32) << i;
        }
        mask
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

/// True when a load of this ordering joins the store's release view into the
/// reader's causality (`Synchronize::sync_load`).
fn acquires(order: Ordering) -> bool {
    matches!(order, Ordering::Acquire | Ordering::AcqRel | Ordering::SeqCst)
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
