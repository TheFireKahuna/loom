use crate::rt::{execution, object, thread, MAX_ATOMIC_HISTORY, MAX_THREADS};

use std::sync::atomic::{
    AtomicU16, AtomicU32, Ordering::AcqRel, Ordering::Acquire, Ordering::Relaxed,
};
use std::sync::Arc;

#[cfg(feature = "checkpoint")]
use serde::{Deserialize, Serialize};

/// Every thread slot a branch can hold.
const ALL_THREADS: u16 = ((1u32 << MAX_THREADS) - 1) as u16;

/// Ownership of the alternatives at one frozen branch, shared by every task
/// descended from the split that froze it.
///
/// A branch that has been frozen has a fixed choice, so its index stops being
/// recycled into some other part of the tree and `(prefix, branch)` becomes a
/// stable name. That is what makes a backtrack mark landing here *deliverable*
/// rather than lost: DPOR marks propagate upward, and a worker deep in a
/// donated subtree routinely proves that some thread had to run at an ancestor
/// it no longer owns. Writing that into its private copy would tell nobody.
/// Writing it here tells everyone.
///
/// Both halves of `word` only ever gain bits, so `fetch_or` is the whole
/// protocol: it is the join of a lattice, the previous value says who won, and
/// no two callers need to agree on an order.
#[derive(Debug)]
pub(crate) struct Frozen {
    /// Threads a mark may still open here — exactly those that were `Skip`
    /// when the branch froze. Nothing else is an alternative DPOR can create:
    /// `Disabled` and `Yield` are never marked, and the rest are already open.
    markable: u16,

    /// Threads that were not `Disabled` here. `Schedule::backtrack` falls back
    /// to opening every candidate when its target cannot itself be scheduled,
    /// and a task's private copy of this branch has been overwritten, so the
    /// test has to read the state the branch actually had.
    enabled: u16,

    /// `open` in the low half, `claimed` in the high half.
    ///
    /// Open means DPOR has proven this alternative must be explored; claimed
    /// means somebody has taken responsibility for exploring it. One word so a
    /// single load sees both.
    word: ClaimWord,

    /// Threads whose open bit was set only by the preemption bound's
    /// conservative backtrack points (BPOR Algorithm 3, Line 9), never by a
    /// standard DPOR mark. Measurement only (`Builder::stats`); a standard
    /// mark clears the bit, and the clear/set race with a concurrent peer is
    /// tolerated — attribution is an upper bound, not part of exploration.
    conservative: AtomicU16,
}

/// The claim word on a line of its own, away from the immutable fields beside
/// it and the `Arc` refcount behind it. Those are read on every replay through
/// the frozen prefix; this is written whenever a mark lands. Sharing a line
/// would let the rare write invalidate the frequent read.
#[derive(Debug)]
#[repr(align(64))]
struct ClaimWord(AtomicU32);

impl Frozen {
    fn new(markable: u16, enabled: u16, open: u16, claimed: u16, conservative: u16) -> Frozen {
        Frozen {
            markable,
            enabled,
            word: ClaimWord(AtomicU32::new((open as u32) | ((claimed as u32) << 16))),
            conservative: AtomicU16::new(conservative),
        }
    }

    /// A branch nothing can be claimed at or marked on: not being explored, or
    /// not a schedule. `step()` moves past it exactly as a serial walk does.
    fn sealed() -> Frozen {
        Frozen::new(0, 0, 0, 0, 0)
    }

    /// Prove these threads must be explored here. Idempotent, and safe to race
    /// with any other caller.
    fn open(&self, mask: u16, conservative: bool) {
        if mask != 0 {
            if conservative {
                self.conservative.fetch_or(mask, Relaxed);
            } else {
                self.conservative.fetch_and(!mask, Relaxed);
            }

            self.word.0.fetch_or(mask as u32, AcqRel);
        }
    }

    /// Whether this alternative is (still) only conservatively justified.
    fn is_conservative(&self, thread: usize) -> bool {
        self.conservative.load(Relaxed) & (1u16 << thread) != 0
    }

    /// Take responsibility for the next alternative nobody holds, or `None`
    /// when this branch is entirely spoken for.
    ///
    /// The retry is not a wait: every pass either wins a bit or observes one
    /// permanently claimed by somebody else, so it runs at most once per
    /// thread slot and always makes progress.
    fn claim_next(&self) -> Option<usize> {
        loop {
            let word = self.word.0.load(Acquire);
            let free = (word as u16) & !((word >> 16) as u16);

            if free == 0 {
                return None;
            }

            let thread = free.trailing_zeros() as usize;
            let bit = 1u32 << (16 + thread);

            if self.word.0.fetch_or(bit, AcqRel) & bit == 0 {
                return Some(thread);
            }
        }
    }

    /// Take responsibility for one specific alternative, reporting whether
    /// this caller is the one that got it.
    fn claim(&self, thread: usize) -> bool {
        let bit = 1u32 << (16 + thread);

        self.word.0.fetch_or(bit, AcqRel) & bit == 0
    }

    /// Alternatives proven necessary here so far. Monotone — a set bit is a
    /// standing promise that some task fully explores that alternative, so a
    /// reader may defer to any bit it observes (`rt::sleep`).
    fn open_mask(&self) -> u16 {
        self.word.0.load(Acquire) as u16
    }
}

/// An execution path
#[derive(Debug, Clone)]
#[cfg_attr(feature = "checkpoint", derive(Serialize, Deserialize))]
pub(crate) struct Path {
    preemption_bound: Option<u8>,

    /// Current execution's position in the branches vec.
    ///
    /// When the execution starts, this is zero, but `branches` might not be
    /// empty.
    ///
    /// In order to perform an exhaustive search, the execution is seeded with a
    /// set of branches.
    pos: usize,

    /// List of all branches in the execution.
    ///
    /// A branch is of type `Schedule`, `Load`, or `Spurious`
    branches: object::Store<Entry>,

    /// If true, exploring is enabled at start
    exploring: bool,

    /// If true, the user decided to skip the current execution branch. We do
    /// not do any further exploration here.
    skipping: bool,

    /// How to reset the `exploring` state
    exploring_on_start: bool,

    /// Shared claim state for this path's frozen prefix, one entry per branch.
    ///
    /// `frozen.len()` is the floor: branches below it have a fixed choice here
    /// and are shared with every task descended from the same split, so their
    /// alternatives come from [`Frozen`] rather than from this path's private
    /// copy. Branches at or above it are this path's alone and take ordinary
    /// DPOR with no atomics involved at all.
    ///
    /// Empty for a serial run — nothing is ever frozen, so nothing is shared.
    #[cfg_attr(feature = "checkpoint", serde(skip))]
    frozen: Vec<Arc<Frozen>>,

    /// Deepest branch [`Path::split_off`] may freeze.
    ///
    /// This bounds how much of the tree carries shared bookkeeping, and
    /// nothing else — freezing a branch changes who explores its alternatives,
    /// never which alternatives exist. So it only has to be deep enough to
    /// expose more independent subtrees than there are workers.
    split_depth: usize,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "checkpoint", derive(Serialize, Deserialize))]
pub(crate) struct Schedule {
    /// Number of times the thread leading to this branch point has been
    /// pre-empted.
    preemptions: u8,

    /// The thread that was active first
    initial_active: Option<u8>,

    /// State of each thread
    threads: [Thread; MAX_THREADS],

    /// The previous schedule branch
    prev: Option<object::Ref<Schedule>>,

    exploring: bool,

    /// Threads whose `Pending` here was opened only by the preemption bound's
    /// conservative backtrack points (BPOR Algorithm 3, Line 9). A standard
    /// DPOR mark clears the bit: the alternative is then required regardless
    /// of the bound. Measurement only (`Builder::stats`).
    conservative: u16,

    /// The branch's current choice was conservative-only (see above) at the
    /// moment it was activated. What execution attribution reads.
    entered_conservative: bool,

    /// The seed switched here because the previous thread *yielded*, not
    /// because it blocked or finished. Alternatives still cost no preemption
    /// (loom's yield stance: a voluntary switch is free), but a spent bound
    /// must keep refusing marks here — yield seams recur every spin
    /// iteration, so ungated they make cyclic state spaces inexhaustible.
    /// This is the fairness-bound seam of Coons et al., carried as a flag
    /// rather than a second bound.
    yield_seam: bool,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "checkpoint", derive(Serialize, Deserialize))]
pub(crate) struct Load {
    /// All possible values
    values: [u8; MAX_ATOMIC_HISTORY],

    /// Current value
    pos: u8,

    /// Number of values in list
    len: u8,

    exploring: bool,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "checkpoint", derive(Serialize, Deserialize))]
pub(crate) struct Spurious {
    spur: bool,
    exploring: bool,
}

objects! {
    #[derive(Debug, Clone)]
    #[cfg_attr(feature = "checkpoint", derive(Serialize, Deserialize))]
    Entry,
    Schedule(Schedule),
    Load(Load),
    Spurious(Spurious),
}

#[derive(Debug, Eq, PartialEq, Clone, Copy)]
#[cfg_attr(feature = "checkpoint", derive(Serialize, Deserialize))]
pub(crate) enum Thread {
    /// The thread is currently disabled
    Disabled,

    /// The thread should not be explored
    Skip,

    /// The thread is in a yield state.
    Yield,

    /// The thread is waiting to be explored
    Pending,

    /// The thread is currently being explored
    Active,

    /// The thread has been explored
    Visited,
}

macro_rules! assert_path_len {
    ($branches:expr) => {{
        assert!(
            // if we are panicking, we may be performing a branch due to a
            // `Drop` impl (e.g., for `Arc`, or for a user type that does an
            // atomic operation in its `Drop` impl). if that's the case,
            // asserting this again will double panic. therefore, short-circuit
            // the assertion if the thread is panicking.
            $branches.len() < $branches.capacity() || std::thread::panicking(),
            "Model exceeded maximum number of branches. This is often caused \
             by an algorithm requiring the processor to make progress, e.g. \
             spin locks.",
        );
    }};
}

impl Path {
    /// Create a new, blank, configured to branch at most `max_branches` times
    /// and at most `preemption_bound` thread preemptions.
    pub(crate) fn new(max_branches: usize, preemption_bound: Option<u8>, exploring: bool) -> Path {
        assert!(
            MAX_THREADS <= 16,
            "[loom internal bug] thread sets are a u16 mask over threads"
        );

        Path {
            preemption_bound,
            pos: 0,
            branches: object::Store::with_capacity(max_branches),
            exploring,
            skipping: false,
            exploring_on_start: exploring,
            frozen: Vec::new(),
            split_depth: 0,
        }
    }

    /// Branches below this are frozen: their choice here is fixed, and their
    /// alternatives are claimed through [`Frozen`] rather than explored
    /// directly.
    fn floor(&self) -> usize {
        self.frozen.len()
    }

    pub(crate) fn set_split_depth(&mut self, depth: usize) {
        self.split_depth = depth;
    }

    pub(crate) fn explore_state(&mut self) {
        if !self.skipping {
            assert!(!self.exploring, "not in critical state");
            self.exploring = true;
        }
    }

    pub(crate) fn critical(&mut self) {
        if !self.skipping {
            assert!(self.exploring, "not in exploring state");
            self.exploring = false;
        }
    }

    pub(crate) fn skip_branch(&mut self) {
        self.exploring = false;
        self.skipping = true;
    }

    /// Whether the rest of this execution is a non-exploring scout — either
    /// the user skipped it, or a sleep set proved it redundant.
    pub(super) fn is_skipping(&self) -> bool {
        self.skipping
    }

    pub(crate) fn set_max_branches(&mut self, max_branches: usize) {
        self.branches
            .reserve_exact(max_branches - self.branches.len());
    }

    /// Returns `true` if the execution has reached a point where the known path
    /// has been traversed and has reached a new branching point.
    pub(super) fn is_traversed(&self) -> bool {
        self.pos == self.branches.len()
    }

    pub(super) fn pos(&self) -> usize {
        self.pos
    }

    /// Push a new atomic-load branch
    pub(super) fn push_load(&mut self, seed: &[u8]) {
        assert_path_len!(self.branches);

        let load_ref = self.branches.insert(Load {
            values: [0; MAX_ATOMIC_HISTORY],
            pos: 0,
            len: 0,
            exploring: self.exploring,
        });

        let load = load_ref.get_mut(&mut self.branches);

        for (i, &store) in seed.iter().enumerate() {
            assert!(
                store < MAX_ATOMIC_HISTORY as u8,
                "[loom internal bug] store = {}; max = {}",
                store,
                MAX_ATOMIC_HISTORY
            );
            assert!(
                i < MAX_ATOMIC_HISTORY,
                "[loom internal bug] i = {}; max = {}",
                i,
                MAX_ATOMIC_HISTORY
            );

            load.values[i] = store;
            load.len += 1;
        }
    }

    /// Returns the atomic write to read
    pub(super) fn branch_load(&mut self) -> usize {
        assert!(!self.is_traversed(), "[loom internal bug]");

        let load = object::Ref::from_usize(self.pos)
            .downcast::<Load>(&self.branches)
            .expect("Reached unexpected exploration state. Is the model fully deterministic?")
            .get(&self.branches);

        self.pos += 1;

        load.values[load.pos as usize] as usize
    }

    /// Branch on spurious notifications
    pub(super) fn branch_spurious(&mut self) -> bool {
        if self.is_traversed() {
            assert_path_len!(self.branches);

            self.branches.insert(Spurious {
                spur: false,
                exploring: self.exploring,
            });
        }

        let spurious = object::Ref::from_usize(self.pos)
            .downcast::<Spurious>(&self.branches)
            .expect("Reached unexpected exploration state. Is the model fully deterministic?")
            .get(&self.branches)
            .spur;

        self.pos += 1;
        spurious
    }

    /// Returns the thread identifier to schedule, and the mask of sibling
    /// alternatives this execution may defer to at the branch (`rt::sleep`):
    /// the canonically-lower open alternatives, minus the bound's
    /// conservative ones. One rule for owned and frozen branches alike — at a
    /// frozen branch the private copy's thread states are degenerate, so the
    /// open set comes from the claim record instead.
    pub(super) fn branch_thread(
        &mut self,
        execution_id: execution::Id,
        seed: impl ExactSizeIterator<Item = Thread>,
    ) -> (Option<thread::Id>, u16) {
        if self.is_traversed() {
            assert_path_len!(self.branches);

            // Find the last thread scheduling branch in the path
            let prev = self.last_schedule();

            // Entering a new exploration space.
            //
            // Initialize a  new branch. The initial field values don't matter
            // as they will be updated below.
            let schedule_ref = self.branches.insert(Schedule {
                preemptions: 0,
                initial_active: None,
                threads: [Thread::Disabled; MAX_THREADS],
                prev,
                exploring: self.exploring,
                conservative: 0,
                entered_conservative: false,
                yield_seam: false,
            });

            // Get a reference to the branch in the object store.
            let schedule = schedule_ref.get_mut(&mut self.branches);

            assert!(seed.len() <= MAX_THREADS, "[loom internal bug]");

            // Currently active thread
            let mut active = None;

            for (i, v) in seed.enumerate() {
                // Initialize thread states
                schedule.threads[i] = v;

                if v.is_active() {
                    assert!(
                        active.is_none(),
                        "[loom internal bug] only one thread should start as active"
                    );
                    active = Some(i as u8);
                }
            }

            // Ensure at least one thread is active, otherwise toggle a yielded
            // thread.
            if active.is_none() {
                for (i, th) in schedule.threads.iter_mut().enumerate() {
                    if *th == Thread::Yield {
                        *th = Thread::Active;
                        active = Some(i as u8);
                        break;
                    }
                }
            }

            let mut initial_active = active;

            // The thread whose transition ran into this branch: the previous
            // branch's choice, or at the root the main thread, whose
            // un-branched prefix is what executed before any branch existed.
            let displaced = match prev {
                Some(prev) => prev.get(&self.branches).active_thread_index(),
                None => Some(0),
            };

            let mut yield_seam = false;

            if initial_active != displaced {
                // The seed switched threads, so the displaced thread stopped
                // being runnable — every alternative here is free
                // (Definition 2.5: no enabled thread is being preempted; at
                // the root the seed's pick has yet to run a transition).
                // Unless it *yielded*: that switch is still free by loom's
                // yield stance, but the seam is flagged so a spent bound
                // keeps refusing marks here (see `Schedule::yield_seam`).
                yield_seam = displaced
                    .map(|d| schedule_ref.get(&self.branches).threads[d as usize] == Thread::Yield)
                    .unwrap_or(false);
                initial_active = None;
            }

            let preemptions = prev
                .map(|prev| prev.get(&self.branches).preemptions())
                .unwrap_or(0);

            debug_assert!(
                self.preemption_bound.is_none() || Some(preemptions) <= self.preemption_bound,
                "[loom internal bug] max = {:?}; curr = {}",
                self.preemption_bound,
                preemptions,
            );

            let schedule = schedule_ref.get_mut(&mut self.branches);
            schedule.initial_active = initial_active;
            schedule.preemptions = preemptions;
            schedule.yield_seam = yield_seam;
        }

        let index = self.pos;

        let schedule_ref = object::Ref::from_usize(self.pos)
            .downcast::<Schedule>(&self.branches)
            .expect("Reached unexpected exploration state. Is the model fully deterministic?");

        let schedule = schedule_ref.get(&self.branches);

        self.pos += 1;

        let active = schedule.threads.iter().position(|th| th.is_active());

        // Zero under a preemption bound: a covering linearization can spend
        // up to two more preemptions than the execution it covers (the
        // commutation seam inserts a switch and a switch-back), so the
        // deferred-to subtree may be bound-truncated exactly where the pruned
        // class would have lived. Not a theoretical scruple — the fuzz corpus
        // finds behavior loss for bounded deference within seconds.
        let covered = match active {
            Some(chosen) if self.preemption_bound.is_none() => {
                let coverable = match self.frozen.get(index) {
                    Some(frozen) => frozen.open_mask(),
                    None => schedule
                        .threads
                        .iter()
                        .enumerate()
                        .filter(|&(_, th)| {
                            matches!(th, Thread::Pending | Thread::Active | Thread::Visited)
                        })
                        .fold(0u16, |mask, (i, _)| mask | (1u16 << i)),
                };

                coverable & ((1u16 << chosen) - 1)
            }
            _ => 0,
        };

        (
            active.map(|i| thread::Id::new(execution_id, i)),
            covered,
        )
    }

    pub(super) fn backtrack(&mut self, mut point: usize, thread_id: thread::Id) {
        let prev = loop {
            if let Some(schedule_ref) =
                object::Ref::from_usize(point).downcast::<Schedule>(&self.branches)
            {
                if schedule_ref.get(&self.branches).exploring {
                    let prev = schedule_ref.get(&self.branches).prev;

                    self.mark(schedule_ref, thread_id, false);
                    break prev;
                }
            }

            if point == 0 {
                return;
            }

            point -= 1;
        };

        let mut curr = if let Some(curr) = prev {
            curr
        } else {
            return;
        };

        if self.preemption_bound.is_some() {
            loop {
                // Preemption bounded DPOR requires conservatively adding
                // another backtrack point to cover cases missed by the bounds.
                if let Some(prev) = curr.get(&self.branches).prev {
                    let active_a = curr.get(&self.branches).active_thread_index();
                    let active_b = prev.get(&self.branches).active_thread_index();

                    if active_a != active_b && curr.get(&self.branches).exploring {
                        self.mark(curr, thread_id, true);
                        return;
                    }

                    curr = prev;
                } else {
                    if curr.get(&self.branches).exploring {
                        // This is the very first schedule
                        self.mark(curr, thread_id, true);
                    }
                    return;
                }
            }
        }
    }

    /// Open the alternatives DPOR proved necessary at `schedule`, given that
    /// `thread_id` raced with an access recorded there.
    ///
    /// Above the floor this branch is the path's own and the mark is a plain
    /// state change. At or below it the branch is shared, so the mark goes to
    /// the claim record instead — where it stays visible to whichever task
    /// reaches the branch next, instead of being written into a private copy
    /// that every other holder of this branch has already overwritten.
    fn mark(&mut self, schedule: object::Ref<Schedule>, thread_id: thread::Id, conservative: bool) {
        let thread_id = thread_id.as_usize();

        if thread_id >= MAX_THREADS {
            return;
        }

        // The bound is a property of the prefix, so a frozen branch's private
        // copy still carries the right value for it.
        if let Some(bound) = self.preemption_bound {
            let branch = schedule.get(&self.branches);

            assert!(
                branch.preemptions <= bound,
                "[loom internal bug] actual = {}, bound = {}",
                branch.preemptions,
                bound
            );

            // BPOR's gate is on an exploration's own cost (Algorithm 2,
            // Line 12), not on point creation. At a branch whose seed
            // switched threads (`initial_active == None`) the previous
            // thread stopped being runnable, so scheduling any alternative
            // is free (Definition 2.5) and stays within a spent bound. A
            // saturated prefix rules out only alternatives that would
            // preempt the naturally-continuing thread — plus yield seams,
            // whose free switches recur every spin iteration and must stay
            // budget-cut for cyclic state spaces to exhaust.
            if branch.preemptions == bound
                && (branch.initial_active.is_some() || branch.yield_seam)
            {
                return;
            }
        }

        let bit = 1u16 << thread_id;

        if let Some(frozen) = self.frozen.get(schedule.index()) {
            // A disabled target cannot itself be scheduled here, so DPOR falls
            // back to opening every candidate at this branch.
            let candidates = if frozen.enabled & bit != 0 { bit } else { ALL_THREADS };

            frozen.open(candidates & frozen.markable, conservative);
            return;
        }

        let schedule = schedule.get_mut(&mut self.branches);
        let candidates = if schedule.threads[thread_id].is_enabled() {
            bit
        } else {
            ALL_THREADS
        };

        for (i, th) in schedule.threads.iter_mut().enumerate() {
            if candidates & (1 << i) == 0 {
                continue;
            }

            if *th == Thread::Skip {
                *th = Thread::Pending;

                if conservative {
                    schedule.conservative |= 1 << i;
                } else {
                    schedule.conservative &= !(1 << i);
                }
            } else if !conservative {
                // A standard mark reaching an alternative that was opened (or
                // even taken) conservatively upgrades it: ordinary DPOR now
                // requires it, so nothing under it is bound-attributable.
                schedule.conservative &= !(1 << i);

                if *th == Thread::Active {
                    schedule.entered_conservative = false;
                }
            }
        }
    }

    /// Reset the path to prepare for the next exploration of the model.
    ///
    /// This function will also trim the object store, dropping any objects that
    /// are created in pruned sections of the path.
    pub(super) fn step(&mut self) -> bool {
        // Reset the position to zero, the path will start traversing from the
        // beginning
        self.pos = 0;

        // Reset exploring / critical / skip
        self.exploring = self.exploring_on_start;
        self.skipping = false;

        // Set the final branch to try the next option. If all options have been
        // traversed, pop the final branch and try again w/ the one under it.
        //
        // This is depth-first tree traversal.
        for last in (0..self.branches.len()).rev() {
            let last = object::Ref::from_usize(last);

            // Remove all objects that were created **after** this branch
            self.branches.truncate(last);

            // A frozen branch's alternatives are shared, so the next one comes
            // from the claim record rather than from this path's copy. Winning
            // one here is the same act as picking up a posted task — the
            // difference is only that this worker was already standing on the
            // branch, so it costs one atomic instead of a trip through the
            // pool.
            if last.index() < self.frozen.len() {
                let Some(thread) = self.frozen[last.index()].claim_next() else {
                    // Nothing left here for anybody. Dropping the record is
                    // safe precisely because of that: this path can no longer
                    // be the one that owes work at this branch.
                    continue;
                };

                // Deeper frozen branches described alternatives under the
                // choice being left behind, and were just observed empty.
                self.frozen.truncate(last.index() + 1);

                let entered_conservative = self.frozen[last.index()].is_conservative(thread);

                let schedule = last
                    .downcast::<Schedule>(&self.branches)
                    .expect("[loom internal bug] claimed a branch that is not a schedule")
                    .get_mut(&mut self.branches);

                for (i, th) in schedule.threads.iter_mut().enumerate() {
                    *th = if i == thread {
                        Thread::Active
                    } else {
                        Thread::Visited
                    };
                }

                schedule.entered_conservative = entered_conservative;

                return true;
            }

            if let Some(schedule_ref) = last.downcast::<Schedule>(&self.branches) {
                let schedule = schedule_ref.get_mut(&mut self.branches);

                if !schedule.exploring {
                    continue;
                }

                // The alternative just explored is finished; its subtree from
                // this branch is fully walked.
                if let Some(thread) = schedule.threads.iter_mut().find(|th| th.is_active()) {
                    *thread = Thread::Visited;
                }

                // Find a pending thread and transition it to active.
                let mut rem = None;

                for (i, th) in schedule.threads.iter_mut().enumerate() {
                    if !th.is_pending() {
                        continue;
                    }

                    *th = Thread::Active;
                    rem = Some(i);
                    break;
                }

                if let Some(i) = rem {
                    schedule.entered_conservative = schedule.conservative & (1 << i) != 0;
                    return true;
                }
            } else if let Some(load_ref) = last.downcast::<Load>(&self.branches) {
                let load = load_ref.get_mut(&mut self.branches);

                if !load.exploring {
                    continue;
                }

                load.pos += 1;

                if load.pos < load.len {
                    return true;
                }
            } else if let Some(spurious_ref) = last.downcast::<Spurious>(&self.branches) {
                let spurious = spurious_ref.get_mut(&mut self.branches);

                if !spurious.exploring {
                    continue;
                }

                if !spurious.spur {
                    spurious.spur = true;
                    return true;
                }
            } else {
                unreachable!();
            }
        }

        false
    }

    fn last_schedule(&self) -> Option<object::Ref<Schedule>> {
        self.branches.iter_ref::<Schedule>().rev().next()
    }

    /// Whether the execution just traversed sits under at least one schedule
    /// choice that only the preemption bound's conservative backtrack points
    /// opened (`Builder::stats`). An upper bound on what a bound-aware
    /// optimal DPOR could avoid exploring: everything counted here exists
    /// only because of BPOR Algorithm 3's Line 9.
    pub(crate) fn conservative_attributed(&self) -> bool {
        self.branches
            .iter_ref::<Schedule>()
            .any(|r| r.get(&self.branches).entered_conservative)
    }

    /// Carve up to `wanted` subtrees off this path for idle peers.
    ///
    /// Nothing is *given away* here in the sense the name suggests. Every
    /// alternative a task explores — donated or not, this path's own or a
    /// peer's — is acquired by one `fetch_or` on the branch that owns it, and
    /// this only reaches for the ones nobody holds yet. That is the whole
    /// reason a sharded search now walks the same tree a serial one does:
    /// there is no second mechanism that has to agree with the first about
    /// which alternatives exist.
    ///
    /// Two sources, shallowest first, because shallow alternatives root the
    /// largest subtrees and a donation should carry real work rather than a
    /// leaf:
    ///
    /// 1. Branches already frozen, where a mark has since opened an
    ///    alternative. This path would have picked those up on its way back
    ///    up; an idle peer can have them now instead.
    /// 2. Fresh branches, frozen on the spot. Freezing fixes the choice here,
    ///    which is what lets the branch be named by more than one task at once.
    pub(crate) fn split_off(&mut self, wanted: usize) -> Vec<Path> {
        let mut tasks = Vec::new();

        for point in 0..self.floor() {
            while tasks.len() < wanted {
                let Some(thread) = self.frozen[point].claim_next() else {
                    break;
                };

                tasks.push(self.task_at(point, Fixed::Thread(thread)));
            }
        }

        while tasks.len() < wanted {
            let Some(index) = self.pick_split() else { break };

            for point in self.floor()..=index {
                let entry = object::Ref::from_usize(point);
                let mut handing_out = Vec::new();

                let frozen = if let Some(sched) = entry.downcast::<Schedule>(&self.branches) {
                    let sched = sched.get(&self.branches);

                    // A branch that is not being explored is one a serial walk
                    // steps straight past, so nothing here is work for anyone.
                    if !sched.exploring {
                        Frozen::sealed()
                    } else {
                        let (mut markable, mut enabled, mut open, mut claimed) = (0, 0, 0, 0);

                        for (i, th) in sched.threads.iter().enumerate() {
                            let bit = 1u16 << i;

                            if th.is_enabled() {
                                enabled |= bit;
                            }

                            match th {
                                // Still an alternative a mark can create.
                                Thread::Skip => markable |= bit,
                                // Open, and about to be claimed by its task.
                                Thread::Pending => {
                                    open |= bit;
                                    handing_out.push(Fixed::Thread(i));
                                }
                                // Open, and this path is already on it or done
                                // with it.
                                Thread::Active | Thread::Visited => {
                                    open |= bit;
                                    claimed |= bit;
                                }
                                Thread::Yield | Thread::Disabled => {}
                            }
                        }

                        Frozen::new(markable, enabled, open, claimed, sched.conservative)
                    }
                } else if let Some(load) = entry.downcast::<Load>(&self.branches) {
                    let load = load.get(&self.branches);

                    // A load's value list is fixed when the branch is created,
                    // so unlike a schedule it can never gain an alternative
                    // later. Handing out the remainder here is exhaustive, and
                    // the record stays sealed.
                    if load.exploring {
                        handing_out.extend((load.pos + 1..load.len).map(Fixed::Value));
                    }

                    Frozen::sealed()
                } else if let Some(spurious) = entry.downcast::<Spurious>(&self.branches) {
                    let spurious = spurious.get(&self.branches);

                    if spurious.exploring && !spurious.spur {
                        handing_out.push(Fixed::Spurious);
                    }

                    Frozen::sealed()
                } else {
                    unreachable!()
                };

                // Freeze before spawning: a task's frozen prefix has to cover
                // the branch it fixes, and it takes that from this path's.
                self.frozen.push(Arc::new(frozen));
                debug_assert_eq!(self.floor(), point + 1, "[loom internal bug]");

                for fixed in handing_out {
                    if let Fixed::Thread(i) = fixed {
                        let won = self.frozen[point].claim(i);

                        debug_assert!(won, "[loom internal bug] fresh branch already claimed");
                    }

                    tasks.push(self.task_at(point, fixed));
                }
            }
        }

        tasks
    }

    /// A task that replays this path as far as `point`, takes `fixed` there,
    /// and explores everything below.
    fn task_at(&self, point: usize, fixed: Fixed) -> Path {
        let capacity = self.branches.capacity();
        let entry = object::Ref::from_usize(point);

        let mut task = self.clone();
        task.pos = 0;
        task.branches.truncate(entry);

        // `point` is frozen for the new task too — its choice there is
        // `fixed`. It keeps the *same* record, so a mark arriving there from
        // either side is settled once, between all of them.
        task.frozen.truncate(point + 1);

        debug_assert_eq!(task.frozen.len(), task.branches.len(), "[loom internal bug]");

        // `Vec::clone` allocates to fit, which would make the
        // `assert_path_len!` guard fire early on the new task.
        if task.branches.capacity() < capacity {
            let additional = capacity - task.branches.len();
            task.branches.reserve_exact(additional);
        }

        match fixed {
            Fixed::Thread(thread) => {
                let entered_conservative = task.frozen[point].is_conservative(thread);

                let schedule = entry
                    .downcast::<Schedule>(&task.branches)
                    .expect("[loom internal bug] not a schedule branch")
                    .get_mut(&mut task.branches);

                // Closing the siblings costs nothing now: at a frozen branch
                // it is the record, not this array, that says what is still an
                // alternative here.
                for (i, th) in schedule.threads.iter_mut().enumerate() {
                    *th = if i == thread {
                        Thread::Active
                    } else {
                        Thread::Visited
                    };
                }

                schedule.entered_conservative = entered_conservative;
            }
            Fixed::Value(value) => {
                entry
                    .downcast::<Load>(&task.branches)
                    .expect("[loom internal bug] not a load branch")
                    .get_mut(&mut task.branches)
                    .pos = value;
            }
            Fixed::Spurious => {
                entry
                    .downcast::<Spurious>(&task.branches)
                    .expect("[loom internal bug] not a spurious branch")
                    .get_mut(&mut task.branches)
                    .spur = true;
            }
        }

        task
    }

    /// The shallowest owned branch with an alternative to hand out, within the
    /// shardable region.
    fn pick_split(&self) -> Option<usize> {
        (self.floor()..self.branches.len().min(self.split_depth)).find(|&index| {
            let entry = object::Ref::from_usize(index);

            if let Some(schedule) = entry.downcast::<Schedule>(&self.branches) {
                let schedule = schedule.get(&self.branches);

                schedule.exploring && schedule.threads.iter().any(Thread::is_pending)
            } else if let Some(load) = entry.downcast::<Load>(&self.branches) {
                let load = load.get(&self.branches);

                load.exploring && load.len > load.pos + 1
            } else if let Some(spurious) = entry.downcast::<Spurious>(&self.branches) {
                let spurious = spurious.get(&self.branches);

                spurious.exploring && !spurious.spur
            } else {
                false
            }
        })
    }
}

/// The one choice a spawned task takes at the branch it was created for.
enum Fixed {
    Thread(usize),
    Value(u8),
    Spurious,
}

impl Schedule {
    /// Returns the index of the currently active thread
    fn active_thread_index(&self) -> Option<u8> {
        self.threads
            .iter()
            .enumerate()
            .find(|(_, th)| th.is_active())
            .map(|(index, _)| index as u8)
    }

    /// Compute the number of preemptions for the current state of the branch
    fn preemptions(&self) -> u8 {
        if self.initial_active.is_some() && self.initial_active != self.active_thread_index() {
            return self.preemptions + 1;
        }

        self.preemptions
    }

}

impl Thread {
    fn is_pending(&self) -> bool {
        *self == Thread::Pending
    }

    fn is_active(&self) -> bool {
        *self == Thread::Active
    }

    fn is_enabled(&self) -> bool {
        !self.is_disabled()
    }

    fn is_disabled(&self) -> bool {
        *self == Thread::Disabled
    }
}
