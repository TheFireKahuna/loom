use crate::rt::execution;
use crate::rt::object::Operation;
use crate::rt::synchronize::{ScView, Synchronize};
use crate::rt::vv::VersionVec;

use std::{any::Any, fmt, ops};

use super::Location;
pub(crate) struct Thread {
    pub id: Id,

    /// An order-free digest of what the thread observed — the store each read
    /// took, the release each lock acquisition followed, each condvar wake
    /// and fired timeout — under `LOOM_SNAPSHOT_CHECK`, which compares it
    /// between an execution resumed from a snapshot and its replay.
    pub(crate) observed: std::cell::Cell<u64>,

    /// If the thread is runnable, blocked, or terminated.
    pub state: State,

    /// True if the thread is in a critical section
    pub critical: bool,

    /// The operation the thread is about to take
    pub(super) operation: Option<Operation>,

    /// Tracks observed causality: happens-before, and nothing else. Every
    /// data-race check reads it, so no `SeqCst` fence effect may enter it.
    pub causality: VersionVec,

    /// The `SeqCst`-fence constraints on this thread's current operation
    /// (`coherence_view`, `Set::active_sc_fence_pos`). Held fixed for the
    /// length of one operation: what the operation itself acquires lands in
    /// `sc_acquired` and takes effect from the next one, so every region of a
    /// wide load is filtered against one scope.
    pub(crate) sc: ScView,

    /// `SeqCst`-fence constraints acquired since the current operation began;
    /// folded into `sc` by `begin_op`. Published views include it.
    sc_acquired: ScView,

    /// The thread's view as of its latest release fence, which every later
    /// store carries (C++20 [atomics.fences] p2).
    pub(crate) released: Synchronize,

    /// The release views of every store this thread has read without
    /// acquiring: what its next acquire fence joins ([atomics.fences] p4).
    /// Accumulated at the read, so it survives the store leaving its cell's
    /// history.
    pub(crate) acquirable: Synchronize,

    /// lanes: own-clock version of this thread's latest acquire-or-stronger
    /// fence and latest `SeqCst` fence, `0` for none (every operation's
    /// version is at least `1`). What the lane coherence floor reads to tell
    /// which of the thread's own earlier accesses its next one is ordered after.
    pub(crate) acq_fence_version: u16,
    pub(crate) sc_fence_version: u16,

    /// lanes: own-clock version of this thread's latest atomic RMW, `0` for
    /// none — a full barrier on some targets (`rt::atomic::LANE_FLOOR`).
    pub(crate) rmw_version: u16,

    /// lanes: own-clock versions of every full barrier this thread has run —
    /// each `SeqCst` fence, and each RMW where the target makes one a barrier
    /// — ascending. What tells a peer which barrier first followed a store.
    pub(crate) barriers: smallvec::SmallVec<[u16; 8]>,

    /// `std::thread::park`'s token.
    park_token: bool,

    /// Join of the views of every `unpark` since the token was last consumed:
    /// consuming it is the acquire half of their synchronization.
    park_view: Synchronize,

    /// Blocked in `park` waiting for the token; only `unpark` wakes it.
    parked: bool,

    /// Whether this thread's `park` has already returned spuriously this
    /// execution. One spurious return per thread bounds every park loop.
    park_spurred: bool,

    /// A step woke this thread from an untimed block on an event it waits
    /// for — a notification, a wake, an unpark, a message — and the
    /// scheduler has not yet seen it (`rt::dpor`): the thread's next step can
    /// run only after the step that woke it. A timed block's next step can
    /// also be its timeout firing, earlier, so it is not set for one; nor
    /// for a lock's release, since what orders a lock's acquirers is their
    /// acquires.
    pub(crate) woken: bool,

    /// Whether a timed wait of this thread has already timed out while
    /// another thread could run, this execution. One such early timeout per
    /// thread bounds every timed-wait loop; a timeout with nothing else
    /// runnable is always available.
    timeout_spent: bool,

    /// Tracks DPOR relations
    pub dpor_vv: VersionVec,

    /// `dpor_vv` as it stood before the current operation joined its own
    /// dependences: what the operation is ordered after by everything but
    /// itself. `None` under a preemption bound, where the search does not
    /// promise every reordering of unordered operations.
    pub dpor_prior: Option<VersionVec>,

    /// Version at which the thread last yielded
    pub last_yield: Option<u16>,

    /// Number of times the thread yielded
    pub yield_count: usize,

    /// The symmetry group (`thread::symmetric`) the thread belongs to and its
    /// rank in it, or `None` for an ordinary thread.
    pub symmetry: Option<Symmetry>,

    /// Whether the thread has been scheduled at least once. What releases the
    /// symmetry pin on later-ranked members of its group.
    pub started: bool,

    locals: LocalMap,

    /// `tracing` span used to associate diagnostics with the current thread.
    span: tracing::Span,
}

#[derive(Debug)]
pub(crate) struct Set {
    /// Unique execution identifier
    execution_id: execution::Id,

    /// Set of threads
    threads: Vec<Thread>,

    /// Currently scheduled thread.
    ///
    /// `None` signifies that no thread is runnable.
    active: Option<usize>,

    /// Join of the causality of every `SeqCst` fence committed so far. A fence
    /// takes it into its thread's `ScView::frontier` (never its causality):
    /// the events it holds precede, in S, whatever that fence happens before.
    sc_fence_frontier: VersionVec,

    /// Next position to hand out in the single total order S over `SeqCst`
    /// operations (C++20 [atomics.order]). Every SC store and every SC fence
    /// takes the next value at the point it commits in the explored schedule,
    /// so S *is* commit order; permutation testing covers every S consistent
    /// with happens-before. SC loads consult S but need no position of their
    /// own — nothing in the model refers back to a load's place in S.
    ///
    /// This is the one order shared by SC accesses (`rt::atomic`) and SC
    /// fences: an SC fence tags the stores sequenced before it with its own
    /// position (`begin_sc_fence`), which is how a fence in one thread comes to
    /// sit in S against an SC access in another.
    sc_clock: u32,

    /// Symmetry groups minted so far this execution; the next one's id.
    symmetry_groups: usize,

    /// `tracing` span used as the parent for new thread spans.
    iteration_span: tracing::Span,
}

/// A thread's place in one `thread::symmetric` call: the call's group, and
/// the thread's spawn position in it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Symmetry {
    pub(crate) group: usize,
    pub(crate) rank: usize,
}

#[derive(Eq, PartialEq, Hash, Copy, Clone)]
pub(crate) struct Id {
    execution_id: execution::Id,
    id: usize,
}

impl Id {
    /// Returns an integer ID unique to this current execution (for use in
    /// [`thread::ThreadId`]'s `Debug` impl)
    pub(crate) fn public_id(&self) -> usize {
        self.id
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum State {
    Runnable,
    Blocked {
        #[allow(dead_code)]
        location: Location,
        /// A timed block (`park_timeout`, `Condvar::wait_timeout`, `sleep`)
        /// can end on its own — its timeout fires, which
        /// `Execution::schedule` offers as the thread's next transition.
        timed: bool,
    },
    Yield,
    Terminated,
}

/// Locals, keyed by the `LocalKey`'s address, in creation order.
///
/// Creation order is load-bearing twice over: destructors run in *reverse*
/// creation order (`take_next_local`), and the order must be deterministic
/// across iterations — destructors run tracked ops, so a randomized (hash)
/// order would make one iteration's path diverge from its replay.
type LocalMap = Vec<(LocalKeyId, LocalValue)>;

#[derive(Eq, PartialEq, Hash, Copy, Clone)]
struct LocalKeyId(usize);

struct LocalValue(Option<Box<dyn Any>>);

impl Thread {
    fn new(id: Id, parent_span: &tracing::Span) -> Thread {
        Thread {
            id,
            span: tracing::info_span!(parent: parent_span.id(), "thread", id = id.id),
            state: State::Runnable,
            critical: false,
            operation: None,
            causality: VersionVec::new(),
            sc: ScView::new(),
            sc_acquired: ScView::new(),
            released: Synchronize::new(),
            acquirable: Synchronize::new(),
            acq_fence_version: 0,
            sc_fence_version: 0,
            rmw_version: 0,
            barriers: smallvec::SmallVec::new(),
            park_token: false,
            park_view: Synchronize::new(),
            parked: false,
            park_spurred: false,
            timeout_spent: false,
            woken: false,
            dpor_vv: VersionVec::new(),
            dpor_prior: None,
            observed: std::cell::Cell::new(0),
            last_yield: None,
            yield_count: 0,
            symmetry: None,
            started: false,
            locals: Vec::new(),
        }
    }

    pub(crate) fn is_runnable(&self) -> bool {
        matches!(self.state, State::Runnable { .. })
    }

    pub(crate) fn set_runnable(&mut self) {
        self.state = State::Runnable;
    }

    pub(crate) fn set_blocked(&mut self, location: Location, timed: bool) {
        self.state = State::Blocked { location, timed };
    }

    pub(crate) fn is_blocked(&self) -> bool {
        matches!(self.state, State::Blocked { .. })
    }

    pub(crate) fn is_blocked_timed(&self) -> bool {
        matches!(self.state, State::Blocked { timed: true, .. })
    }

    pub(crate) fn is_yield(&self) -> bool {
        matches!(self.state, State::Yield)
    }

    pub(crate) fn set_yield(&mut self) {
        self.state = State::Yield;
        self.see_time_pass();
        self.yield_count += 1;
    }

    /// Time passed for this thread while it waited on no one in particular
    /// (a spin, a sleep): a store it had already seen superseded is no longer
    /// returned to it (`rt::atomic`'s yield rule), so a poll loop ends once
    /// the store it polls for lands.
    pub(crate) fn see_time_pass(&mut self) {
        self.last_yield = Some(self.causality[self.id]);
    }

    pub(crate) fn is_terminated(&self) -> bool {
        matches!(self.state, State::Terminated)
    }

    pub(crate) fn set_terminated(&mut self) {
        self.state = State::Terminated;
    }

    /// Take the most recently created local that has not yet been
    /// destroyed, leaving its tombstone in place: later accesses to the key
    /// error with `AccessError` ("already destroyed"), like `std` during
    /// TLS teardown. Reverse creation order matches the usual destructor
    /// convention — a later local's destructor may still read an earlier
    /// one.
    pub(crate) fn take_next_local(&mut self) -> Option<Box<dyn Any>> {
        self.locals
            .iter_mut()
            .rev()
            .find_map(|(_, local)| local.0.take())
    }

    /// Make a thread blocked on a modeled primitive runnable again. A wake
    /// carries no synchronization: the primitive supplies its own.
    pub(crate) fn wake(&mut self) {
        if self.is_blocked() {
            self.woken = !self.is_blocked_timed();
            self.set_runnable();
        }
    }

    /// The thread's current view, as a release publishes it.
    pub(crate) fn view(&self) -> Synchronize {
        let mut sc = self.sc;
        sc.join(&self.sc_acquired);
        Synchronize::from_views(self.causality, sc)
    }

    /// Acquire `view`: its happens-before joins causality now; its `SeqCst`
    /// part joins from the next operation (`sc`).
    pub(crate) fn acquire(&mut self, view: &Synchronize) {
        self.causality.join(view.happens_before());
        self.sc_acquired.join(view.sc());
    }

    /// The view coherence is decided against: causality, plus every event the
    /// `SeqCst` fences ordered before this point (C++20 [atomics.order] p4.4).
    /// A superset of causality, never used for race detection.
    pub(crate) fn coherence_view(&self) -> VersionVec {
        let mut view = self.causality;
        view.join(&self.sc.frontier);
        view
    }

    /// Consume the park token if it is available, acquiring every `unpark`
    /// that contributed to it. Returns whether it was.
    pub(crate) fn take_park_token(&mut self) -> bool {
        if !self.park_token {
            return false;
        }
        self.park_token = false;
        let view = std::mem::replace(&mut self.park_view, Synchronize::new());
        self.acquire(&view);
        true
    }

    /// Block in `park` until `unpark` makes the token available.
    pub(crate) fn set_parked(&mut self, location: Location, timed: bool) {
        self.parked = true;
        self.set_blocked(location, timed);
    }

    /// End a park its timeout ended, so a later `unpark` only sets the token.
    pub(crate) fn clear_parked(&mut self) {
        self.parked = false;
    }

    /// Whether the thread's timeout may fire at a point where `others_runnable`:
    /// always when nothing else can run, else only while its early timeout
    /// is unspent.
    pub(crate) fn may_time_out(&self, others_runnable: bool) -> bool {
        self.is_blocked_timed() && (!others_runnable || !self.timeout_spent)
    }

    /// The timeout of the thread's timed block fires: it runs again, having
    /// spent its early timeout if another thread could have run instead.
    pub(crate) fn fire_timeout(&mut self, others_runnable: bool) {
        debug_assert!(self.is_blocked_timed(), "[loom internal bug] no timed block");
        self.set_runnable();
        self.timeout_spent |= others_runnable;
    }

    /// Whether this execution's one spurious `park` return is still unspent.
    pub(crate) fn may_spur_park(&self) -> bool {
        !self.park_spurred
    }

    pub(crate) fn spend_park_spur(&mut self) {
        self.park_spurred = true;
    }
}

impl fmt::Debug for Thread {
    // Manual debug impl is necessary because thread locals are represented as
    // `dyn Any`, which does not implement `Debug`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Thread")
            .field("id", &self.id)
            .field("state", &self.state)
            .field("critical", &self.critical)
            .field("operation", &self.operation)
            .field("causality", &self.causality)
            .field("sc", &self.sc)
            .field("released", &self.released)
            .field("dpor_vv", &self.dpor_vv)
            .field("last_yield", &self.last_yield)
            .field("yield_count", &self.yield_count)
            .field("locals", &format_args!("[..locals..]"))
            .finish()
    }
}

impl Set {
    /// Fold an observation into the active thread's digest, keyed by the
    /// thread and its operation index.
    pub(crate) fn observe(&self, key: u64) {
        if !super::snapshot::check() {
            return;
        }
        if let Some(i) = self.active {
            let t = &self.threads[i];
            let own = t.dpor_vv[Id::new(self.execution_id, i)] as u64;
            t.observed
                .set(t.observed.get().wrapping_add(super::snapshot::mix(own | (i as u64) << 20, key)));
        }
    }

    /// The execution's observations so far.
    pub(crate) fn observed(&self) -> u64 {
        self.threads.iter().fold(0u64, |h, t| h.wrapping_add(t.observed.get()))
    }

    /// Create an empty thread set.
    ///
    /// The set may contain up to `max_threads` threads.
    pub(crate) fn new(execution_id: execution::Id, max_threads: usize) -> Set {
        let mut threads = Vec::with_capacity(max_threads);
        // Capture the current iteration's span to be used as each thread
        // span's parent.
        let iteration_span = tracing::Span::current();
        // Push initial thread
        threads.push(Thread::new(Id::new(execution_id, 0), &iteration_span));

        Set {
            execution_id,
            threads,
            active: Some(0),
            sc_fence_frontier: VersionVec::new(),
            sc_clock: 0,
            symmetry_groups: 0,
            iteration_span,
        }
    }

    /// Create a new thread
    pub(crate) fn new_thread(&mut self, symmetry: Option<Symmetry>) -> Id {
        assert!(self.threads.len() < self.max());

        // Get the identifier for the thread about to be created
        let id = self.threads.len();

        // Spawning synchronizes the spawner with the new thread; the caller
        // joins causality, the `SeqCst` part travels with it here.
        let spawner_sc = self.active.map(|active| *self.threads[active].view().sc());

        // Push the thread onto the stack
        self.threads.push(Thread::new(
            Id::new(self.execution_id, id),
            &self.iteration_span,
        ));

        if let Some(sc) = spawner_sc {
            self.threads[id].sc = sc;
        }

        self.threads[id].symmetry = symmetry;

        Id::new(self.execution_id, id)
    }

    /// Mint the group of one `thread::symmetric` call. Relabeling holds only
    /// among the members of one call, so each call pins its own.
    pub(crate) fn new_symmetry_group(&mut self) -> usize {
        let group = self.symmetry_groups;
        self.symmetry_groups += 1;
        group
    }

    /// Mask of threads pinned by symmetry: a `thread::symmetric` thread may
    /// not be scheduled while a lower-ranked member of its own group has yet
    /// to be. The scheduler shows a pinned thread as disabled, which is what
    /// restricts the walk to one representative per relabeling of each group
    /// (see `thread::symmetric` for the soundness contract).
    ///
    /// Each group's lowest unscheduled rank is never pinned, so every group
    /// always has a schedulable member and no deadlock can be introduced.
    pub(crate) fn symmetry_pinned_mask(&self) -> u16 {
        let mut mask = 0u16;

        for (i, th) in self.threads.iter().enumerate() {
            let Some(sym) = th.symmetry.filter(|_| !th.started) else {
                continue;
            };

            let behind = self.threads.iter().any(|other| {
                !other.started
                    && other
                        .symmetry
                        .is_some_and(|o| o.group == sym.group && o.rank < sym.rank)
            });

            if behind {
                mask |= 1 << i;
            }
        }

        mask
    }

    pub(crate) fn max(&self) -> usize {
        self.threads.capacity()
    }

    pub(crate) fn is_active(&self) -> bool {
        self.active.is_some()
    }

    pub(crate) fn is_complete(&self) -> bool {
        if self.active.is_none() {
            // All threads should be terminated
            for thread in &self.threads {
                assert!(
                    thread.is_terminated(),
                    "thread not terminated; {:#?}",
                    thread
                );
            }

            true
        } else {
            false
        }
    }

    pub(crate) fn active_id(&self) -> Id {
        Id::new(self.execution_id, self.active.unwrap())
    }

    pub(crate) fn active(&self) -> &Thread {
        &self.threads[self.active.unwrap()]
    }

    pub(crate) fn set_active(&mut self, id: Option<Id>) {
        // Disabled spans (no subscriber) have no id; skip the dispatcher's
        // TLS lookup entirely rather than paying it on every branch.
        let exit_span = self.active().span.id();
        let enter_span = id.and_then(|id| self.threads.get(id.id)?.span.id());

        if exit_span.is_some() || enter_span.is_some() {
            tracing::dispatcher::get_default(|subscriber| {
                if let Some(span_id) = exit_span.clone() {
                    subscriber.exit(&span_id)
                }

                if let Some(span_id) = enter_span.clone() {
                    subscriber.enter(&span_id);
                }
            });
        }
        self.active = id.map(Id::as_usize);
    }

    pub(crate) fn active_mut(&mut self) -> &mut Thread {
        &mut self.threads[self.active.unwrap()]
    }

    /// Get the active thread and second thread
    pub(crate) fn active2_mut(&mut self, other: Id) -> (&mut Thread, &mut Thread) {
        let active = self.active.unwrap();
        let other = other.id;

        if other >= active {
            let (l, r) = self.threads.split_at_mut(other);

            (&mut l[active], &mut r[0])
        } else {
            let (l, r) = self.threads.split_at_mut(active);

            (&mut r[0], &mut l[other])
        }
    }

    /// Begin an operation of the active thread: advance its clock, and bring
    /// the `SeqCst` constraints it acquired during its previous operation
    /// into effect.
    pub(crate) fn begin_op(&mut self) {
        let id = self.active_id();
        let active = self.active_mut();
        active.causality.inc(id);
        let acquired = std::mem::replace(&mut active.sc_acquired, ScView::new());
        active.sc.join(&acquired);
    }

    pub(crate) fn active_atomic_version(&self) -> u16 {
        let id = self.active_id();
        self.active().causality[id]
    }

    /// Wake thread `id` from a block on a modeled primitive (`Thread::wake`).
    pub(crate) fn wake(&mut self, id: Id) {
        self.threads[id.id].wake();
    }

    /// `std::thread::Thread::unpark` by the active thread: make `id`'s park
    /// token available, publishing the unparker's view into it — the release
    /// half of unpark→park synchronization, which `take_park_token`
    /// completes — and wake `id` if it is parked.
    pub(crate) fn unpark(&mut self, id: Id) {
        let view = self.active().view();
        let th = &mut self.threads[id.id];

        th.park_view.join(&view);
        th.park_token = true;

        if th.parked {
            th.parked = false;
            th.woken = !th.is_blocked_timed();
            th.set_runnable();
        }
    }

    /// Insert a point of sequential consistency
    /// TODO
    /// - Deprecate SeqCst accesses and allow SeqCst fences only. The semantics of SeqCst accesses
    ///   is complex and difficult to implement correctly. On the other hand, SeqCst fence has
    ///   well-understood and clear semantics in the absence of SeqCst accesses, and can be used
    ///   for enforcing the read-after-write (RAW) ordering which is probably what the user want to
    ///   achieve with SeqCst.
    /// - Revisit the other uses of this function. They probably don't require sequential
    ///   consistency. E.g. see https://en.cppreference.com/w/cpp/named_req/Mutex
    ///
    /// References
    /// - The "scfix" paper, which proposes a memory model called RC11 that fixes SeqCst
    ///   semantics. of C11. https://plv.mpi-sws.org/scfix/
    /// - Some fixes from the "scfix" paper has been incorporated into C/C++20:
    ///   http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2018/p0668r5.html
    /// - The "promising semantics" paper, which propose an intuitive semantics of SeqCst fence in
    ///   the absence of SC accesses. https://sf.snu.ac.kr/promise-concurrency/
    pub(crate) fn seq_cst(&mut self) {
        // Intentionally a no-op. `SeqCst` *accesses* are modelled in
        // `rt::atomic` by the C++20 SC read rule (`match_load_to_stores`) and
        // per-location SC/mo consistency (`State::store`) — a read restriction
        // only. Joining causality here would manufacture happens-before that
        // SC accesses do not have. `fence(SeqCst)` is handled separately, by
        // `seq_cst_fence`: an S position and a coherence frontier, neither of
        // which creates happens-before either. Callers reach this as a
        // "sequential consistency point" for the lock primitives, whose
        // surrounding acquire/release edges already carry the ordering they
        // need.
    }

    /// Commit the active thread's `SeqCst` fence into S and return its
    /// position.
    ///
    /// Every fence already committed precedes it in S, so everything that
    /// happens before any of them is, for coherence, ordered before whatever
    /// this fence happens before (C++20 [atomics.order] p4.4): the frontier
    /// joins the thread's `ScView`, never its causality — SC fences create no
    /// happens-before. The fence then adds its own causality to the frontier
    /// for the fences after it.
    pub(crate) fn seq_cst_fence(&mut self) -> u32 {
        let pos = self.next_sc_pos();
        let active = &mut self.threads[self.active.unwrap()];

        active.sc.frontier.join(&self.sc_fence_frontier);
        active.sc.fence_pos = Some(pos);
        self.sc_fence_frontier.join(&active.causality);

        pos
    }

    /// Allocate the next position in the SC total order S and hand it to the
    /// caller. Called once per SC store (from `rt::atomic::State::store`).
    pub(crate) fn next_sc_pos(&mut self) -> u32 {
        let pos = self.sc_clock;
        self.sc_clock += 1;
        pos
    }

    /// S-position of the latest `SeqCst` fence that happens before the active
    /// thread's current operation, if any: its scope under the fence rules
    /// (C++20 [atomics.order] p4.3).
    pub(crate) fn active_sc_fence_pos(&self) -> Option<u32> {
        self.active().sc.fence_pos
    }

    /// How far into S the active thread's current operation is bound by the
    /// SC rules: all of it for an operation in S (`seq_cst`), else the latest
    /// SC fence happening before it, else not at all.
    pub(crate) fn active_sc_scope(&self, seq_cst: bool) -> Option<u32> {
        if seq_cst {
            Some(u32::MAX)
        } else {
            self.active_sc_fence_pos()
        }
    }

    pub(crate) fn clear(&mut self, execution_id: execution::Id) {
        self.iteration_span = tracing::Span::current();
        self.threads.clear();
        self.threads
            .push(Thread::new(Id::new(execution_id, 0), &self.iteration_span));

        self.execution_id = execution_id;
        self.active = Some(0);
        self.sc_fence_frontier = VersionVec::new();
        self.sc_clock = 0;
        self.symmetry_groups = 0;
    }

    pub(crate) fn iter(&self) -> impl ExactSizeIterator<Item = (Id, &Thread)> + '_ {
        let execution_id = self.execution_id;
        self.threads
            .iter()
            .enumerate()
            .map(move |(id, thread)| (Id::new(execution_id, id), thread))
    }

    pub(crate) fn iter_mut(&mut self) -> impl ExactSizeIterator<Item = (Id, &mut Thread)> + '_ {
        let execution_id = self.execution_id;
        self.threads
            .iter_mut()
            .enumerate()
            .map(move |(id, thread)| (Id::new(execution_id, id), thread))
    }

    /// Split the set of threads into the active thread and an iterator of all
    /// other threads.
    pub(crate) fn split_active(&mut self) -> (&mut Thread, impl Iterator<Item = &mut Thread>) {
        let active = self.active.unwrap();
        let (one, two) = self.threads.split_at_mut(active);
        let (active, two) = two.split_at_mut(1);

        let iter = one.iter_mut().chain(two.iter_mut());

        (&mut active[0], iter)
    }

    pub(crate) fn local<T: 'static>(
        &mut self,
        key: &'static crate::thread::LocalKey<T>,
    ) -> Option<Result<&T, AccessError>> {
        let key = LocalKeyId::new(key);

        self.active_mut()
            .locals
            .iter()
            .find(|(k, _)| *k == key)
            .map(|(_, local_value)| local_value.get())
    }

    pub(crate) fn local_init<T: 'static>(
        &mut self,
        key: &'static crate::thread::LocalKey<T>,
        value: T,
    ) {
        let key = LocalKeyId::new(key);
        let locals = &mut self.active_mut().locals;

        assert!(locals.iter().all(|(k, _)| *k != key));
        locals.push((key, LocalValue::new(value)));
    }
}

impl ops::Index<Id> for Set {
    type Output = Thread;

    fn index(&self, index: Id) -> &Thread {
        &self.threads[index.id]
    }
}

impl ops::IndexMut<Id> for Set {
    fn index_mut(&mut self, index: Id) -> &mut Thread {
        &mut self.threads[index.id]
    }
}

impl Id {
    pub(crate) fn new(execution_id: execution::Id, id: usize) -> Id {
        Id { execution_id, id }
    }

    pub(crate) fn as_usize(self) -> usize {
        self.id
    }
}

impl From<Id> for usize {
    fn from(src: Id) -> usize {
        src.id
    }
}

impl fmt::Display for Id {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.id.fmt(fmt)
    }
}

impl fmt::Debug for Id {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(fmt, "Id({})", self.id)
    }
}

impl LocalKeyId {
    fn new<T>(key: &'static crate::thread::LocalKey<T>) -> Self {
        Self(key as *const _ as usize)
    }
}

impl LocalValue {
    fn new<T: 'static>(value: T) -> Self {
        Self(Some(Box::new(value)))
    }

    fn get<T: 'static>(&self) -> Result<&T, AccessError> {
        self.0
            .as_ref()
            .ok_or(AccessError { _private: () })
            .map(|val| {
                val.downcast_ref::<T>()
                    .expect("local value must downcast to expected type")
            })
    }
}

/// An error returned by [`LocalKey::try_with`](struct.LocalKey.html#method.try_with).
pub struct AccessError {
    _private: (),
}

impl fmt::Debug for AccessError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AccessError").finish()
    }
}

impl fmt::Display for AccessError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt("already destroyed", f)
    }
}
