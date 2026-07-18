use crate::rt::{execution, object, thread, MAX_ATOMIC_HISTORY, MAX_THREADS};

#[cfg(feature = "checkpoint")]
use serde::{Deserialize, Serialize};

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

    /// Whether sleep-set reduction is applied (see [`Schedule::sleep_done`]).
    sleep_sets: bool,

    /// Sleep set to install at the *next* schedule branch this traversal
    /// reaches, computed by `Execution::schedule` from the transition it
    /// just committed to. Staged here because the branch it belongs to has
    /// not been visited yet.
    sleep_in: u16,

    /// Lowest branch index this path owns. `step()` never advances or
    /// truncates below it, which is what makes a path clone a self-contained
    /// unit of work: the subtree rooted here and nothing else (see
    /// [`Path::split_off`]).
    floor: usize,

    /// Branch depth above which scheduling is expanded exhaustively rather
    /// than by DPOR, and the only region [`split_off`] may hand out.
    ///
    /// This is what makes sharding sound. DPOR marks propagate upward, so a
    /// worker exploring a donated subtree can prove that some thread had to run
    /// at an ancestor — a branch frozen in its private copy, whose owner will
    /// never see the mark. Expanding those branches exhaustively (subject to
    /// the preemption bound, which still applies) means every such thread is
    /// already open, so there is no mark left to lose. Below this depth,
    /// branches belong to exactly one task and ordinary DPOR applies.
    ///
    /// Zero for a serial run: with one worker nothing is ever frozen, so the
    /// full reduction applies at every depth.
    split_depth: usize,

    /// Backtrack points this path discovered below its floor, as
    /// `(branch, thread)`.
    ///
    /// DPOR marks propagate *upward*: exploring a subtree can prove that some
    /// thread had to be scheduled at an ancestor branch. A path that does not
    /// own that ancestor cannot mark it — the owner is off exploring its own
    /// alternatives and will never see the mark — so the point is recorded
    /// here and turned into a task of its own (see [`Path::escape`]). Losing
    /// one silently drops every interleaving behind it, which is exactly the
    /// coverage a sharded run would otherwise miss.
    escapes: Vec<(u32, u8)>,
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

    /// Threads whose subtree *from this branch* the DFS has already finished
    /// (or handed to another worker). Grown by `step()` as it retires each
    /// alternative; persists across iterations because this branch's prefix
    /// does not change while it is being explored.
    sleep_done: u16,

    /// Threads carried into this branch from its predecessor: put to sleep
    /// higher up, and independent of every transition taken since. Recomputed
    /// each traversal, since it is a function of the prefix rather than of
    /// the exploration so far.
    sleep_in: u16,
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
    pub(crate) fn new(
        max_branches: usize,
        preemption_bound: Option<u8>,
        exploring: bool,
        sleep_sets: bool,
    ) -> Path {
        assert!(
            MAX_THREADS <= 16,
            "[loom internal bug] sleep sets are a u16 mask over threads"
        );

        Path {
            preemption_bound,
            pos: 0,
            branches: object::Store::with_capacity(max_branches),
            exploring,
            skipping: false,
            exploring_on_start: exploring,
            sleep_sets,
            sleep_in: 0,
            floor: 0,
            split_depth: 0,
            escapes: Vec::new(),
        }
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

    /// The effective sleep set at the schedule branch stored at `point` —
    /// what the DFS has retired here, plus what was carried in. A branch that
    /// is not a `Schedule` (or a path with sleep sets off) sleeps nothing.
    pub(super) fn sleep_at(&self, point: usize) -> u16 {
        if !self.sleep_sets || point >= self.branches.len() {
            return 0;
        }

        object::Ref::from_usize(point)
            .downcast::<Schedule>(&self.branches)
            .map(|s| self.sleep_of(s))
            .unwrap_or(0)
    }

    fn sleep_of(&self, schedule: object::Ref<Schedule>) -> u16 {
        if !self.sleep_sets {
            return 0;
        }

        let schedule = schedule.get(&self.branches);
        schedule.sleep_done | schedule.sleep_in
    }

    /// Stage the sleep set for the next schedule branch this traversal
    /// reaches. Called once per `Execution::schedule`, after it has committed
    /// to the transition that filtered the set.
    pub(super) fn set_sleep_in(&mut self, sleep: u16) {
        self.sleep_in = if self.sleep_sets { sleep } else { 0 };
    }

    pub(super) fn sleep_sets(&self) -> bool {
        self.sleep_sets
    }

    /// Note backtrack points that landed below this path's floor. Deduplicated
    /// because the prefix below the floor is frozen for this path's whole
    /// lifetime, so `(branch, thread)` names the same subtree every time.
    fn record_escapes(&mut self, point: usize, owned: bool, marked: u16) {
        if owned || marked == 0 {
            return;
        }

        for i in 0..MAX_THREADS {
            if marked & (1 << i) == 0 {
                continue;
            }

            let escape = (point as u32, i as u8);

            if !self.escapes.contains(&escape) {
                self.escapes.push(escape);
            }
        }
    }

    pub(crate) fn has_escapes(&self) -> bool {
        !self.escapes.is_empty()
    }

    /// Turn each escaped backtrack point into a path that explores exactly the
    /// subtree behind it: the frozen prefix, that one thread scheduled, and
    /// nothing else at that branch — every sibling is closed, because the
    /// branch's owner is still responsible for those.
    pub(crate) fn drain_escapes(&mut self) -> Vec<Path> {
        let escapes = std::mem::take(&mut self.escapes);
        let capacity = self.branches.capacity();

        escapes
            .into_iter()
            .map(|(point, thread)| {
                let entry = object::Ref::from_usize(point as usize);

                let mut task = self.clone();
                task.floor = point as usize;
                task.pos = 0;
                task.sleep_in = 0;
                task.escapes = Vec::new();
                task.branches.truncate(entry);

                if task.branches.capacity() < capacity {
                    let additional = capacity - task.branches.len();
                    task.branches.reserve_exact(additional);
                }

                let schedule = entry
                    .downcast::<Schedule>(&task.branches)
                    .expect("[loom internal bug] escape does not name a schedule")
                    .get_mut(&mut task.branches);

                for (i, th) in schedule.threads.iter_mut().enumerate() {
                    *th = if i == thread as usize {
                        Thread::Active
                    } else {
                        Thread::Visited
                    };
                }

                task
            })
            .collect()
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

    /// Returns the thread identifier to schedule
    pub(super) fn branch_thread(
        &mut self,
        execution_id: execution::Id,
        seed: impl ExactSizeIterator<Item = Thread>,
    ) -> Option<thread::Id> {
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
                sleep_done: 0,
                sleep_in: 0,
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

            if let Some(prev) = prev {
                if initial_active != prev.get(&self.branches).active_thread_index() {
                    initial_active = None;
                }
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

            // Shallow branches may be handed to another worker, whose subtree
            // can no longer mark them; open every candidate now so there is
            // nothing left for a later mark to add. The preemption bound still
            // gates this — it bounds the search itself, not the reduction.
            let exhaustive = self.pos < self.split_depth
                && self
                    .preemption_bound
                    .is_none_or(|bound| preemptions < bound);

            let schedule = schedule_ref.get_mut(&mut self.branches);
            schedule.initial_active = initial_active;
            schedule.preemptions = preemptions;

            if exhaustive {
                for th in &mut schedule.threads {
                    if *th == Thread::Skip {
                        *th = Thread::Pending;
                    }
                }
            }
        }

        let schedule_ref = object::Ref::from_usize(self.pos)
            .downcast::<Schedule>(&self.branches)
            .expect("Reached unexpected exploration state. Is the model fully deterministic?");

        // Refresh the carried-in sleep set: `sleep_done` is a property of how
        // far the DFS has gotten here and persists, but `sleep_in` is a
        // function of the prefix and is recomputed every traversal.
        schedule_ref.get_mut(&mut self.branches).sleep_in = self.sleep_in;

        let schedule = schedule_ref.get(&self.branches);

        self.pos += 1;

        schedule
            .threads
            .iter()
            .enumerate()
            .find(|&(_, th)| th.is_active())
            .map(|(i, _)| thread::Id::new(execution_id, i))
    }

    pub(super) fn backtrack(&mut self, mut point: usize, thread_id: thread::Id) {
        let prev = loop {
            if let Some(schedule_ref) =
                object::Ref::from_usize(point).downcast::<Schedule>(&self.branches)
            {
                let sleep = self.sleep_of(schedule_ref);
                let owned = point >= self.floor;
                let schedule = schedule_ref.get_mut(&mut self.branches);

                if schedule.exploring {
                    let marked =
                        schedule.backtrack(thread_id, self.preemption_bound, sleep, owned);
                    let prev = schedule.prev;

                    self.record_escapes(point, owned, marked);
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
                        let sleep = self.sleep_of(curr);
                        let owned = curr.index() >= self.floor;
                        let marked = curr.get_mut(&mut self.branches).backtrack(
                            thread_id,
                            self.preemption_bound,
                            sleep,
                            owned,
                        );

                        self.record_escapes(curr.index(), owned, marked);
                        return;
                    }

                    curr = prev;
                } else {
                    if curr.get(&self.branches).exploring {
                        // This is the very first schedule
                        let sleep = self.sleep_of(curr);
                        let owned = curr.index() >= self.floor;
                        let marked = curr.get_mut(&mut self.branches).backtrack(
                            thread_id,
                            self.preemption_bound,
                            sleep,
                            owned,
                        );

                        self.record_escapes(curr.index(), owned, marked);
                    }
                    return;
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

        // The first schedule branch carries nothing in; every later one has
        // its `sleep_in` recomputed as the traversal reaches it.
        self.sleep_in = 0;

        // Set the final branch to try the next option. If all options have been
        // traversed, pop the final branch and try again w/ the one under it.
        //
        // This is depth-first tree traversal.
        //
        // `floor` bounds it from below: a path handed over by `split_off` owns
        // only the subtree rooted at its floor, and retiring that root ends the
        // path rather than escaping into a sibling another worker holds.
        for last in (self.floor..self.branches.len()).rev() {
            let last = object::Ref::from_usize(last);

            // Remove all objects that were created **after** this branch
            self.branches.truncate(last);

            if let Some(schedule_ref) = last.downcast::<Schedule>(&self.branches) {
                let sleep = self.sleep_of(schedule_ref);
                let sleep_sets = self.sleep_sets;
                let schedule = schedule_ref.get_mut(&mut self.branches);

                if !schedule.exploring {
                    continue;
                }

                // Transition the active thread to visited. Its subtree from
                // this branch is now fully explored, which is exactly the
                // condition for putting it to sleep for the alternatives that
                // follow: any interleaving they could reach by running it
                // later is a reordering of one just covered.
                if let Some((i, thread)) = schedule
                    .threads
                    .iter_mut()
                    .enumerate()
                    .find(|(_, th)| th.is_active())
                {
                    *thread = Thread::Visited;

                    if sleep_sets {
                        schedule.sleep_done |= 1 << i;
                    }
                }

                let sleep = sleep | schedule.sleep_done;

                // Find a pending thread and transition it to active, retiring
                // any that have since gone to sleep.
                let mut rem = false;

                for (i, th) in schedule.threads.iter_mut().enumerate() {
                    if !th.is_pending() {
                        continue;
                    }

                    if sleep_sets && sleep & (1 << i) != 0 {
                        *th = Thread::Visited;
                        continue;
                    }

                    *th = Thread::Active;
                    rem = true;
                    break;
                }

                if rem {
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

    /// Hand half of the shallowest branch that still has unexplored
    /// alternatives to a second path, so another worker can explore it
    /// concurrently. `None` when nothing is left to share.
    ///
    /// The shallowest branch is picked deliberately: its alternatives root the
    /// largest subtrees, so a single donation carries real work rather than a
    /// leaf. Both sides come out with a strictly smaller share, so repeated
    /// donation terminates.
    ///
    /// Each side records the other's share as slept-through: the two subtrees
    /// are disjoint and both are explored before the model run completes, so
    /// an interleaving one side prunes as a reordering is one the other side
    /// actually walks. `Spurious` branches are not split — they are two-way,
    /// and the scan simply moves past them to the next schedule or load.
    pub(crate) fn split_off(&mut self) -> Option<Path> {
        let capacity = self.branches.capacity();
        let limit = self.branches.len().min(self.split_depth);

        for index in self.floor..limit {
            let entry = object::Ref::from_usize(index);

            let taken = if let Some(sched) = entry.downcast::<Schedule>(&self.branches) {
                if !sched.get(&self.branches).exploring {
                    continue;
                }

                let pending: Vec<usize> = sched
                    .get(&self.branches)
                    .threads
                    .iter()
                    .enumerate()
                    .filter(|(_, th)| th.is_pending())
                    .map(|(i, _)| i)
                    .collect();

                if pending.is_empty() {
                    continue;
                }

                Taken::Threads(pending[pending.len() - pending.len().div_ceil(2)..].to_vec())
            } else if let Some(load) = entry.downcast::<Load>(&self.branches) {
                let load = load.get(&self.branches);

                if !load.exploring {
                    continue;
                }

                // Values still to try, beyond the one being explored now.
                let rem = (load.len - load.pos - 1) as usize;

                if rem == 0 {
                    continue;
                }

                let split = load.len as usize - rem.div_ceil(2);
                Taken::Values(split)
            } else {
                continue;
            };

            let mut thief = self.clone();
            thief.floor = index;
            thief.pos = 0;
            thief.sleep_in = 0;
            thief.branches.truncate(entry);

            // `Vec::clone` allocates to fit, which would make the
            // `assert_path_len!` guard fire early on the thief.
            if thief.branches.capacity() < capacity {
                let additional = capacity - thief.branches.len();
                thief.branches.reserve_exact(additional);
            }

            match taken {
                Taken::Threads(taken) => {
                    let mine = entry.downcast::<Schedule>(&self.branches).unwrap();
                    let theirs = entry.downcast::<Schedule>(&thief.branches).unwrap();

                    let mine = mine.get_mut(&mut self.branches);
                    for &i in &taken {
                        mine.threads[i] = Thread::Visited;
                        mine.sleep_done |= 1 << i;
                    }

                    let theirs = theirs.get_mut(&mut thief.branches);
                    let mut active = false;

                    for (i, th) in theirs.threads.iter_mut().enumerate() {
                        if taken.contains(&i) {
                            *th = if active {
                                Thread::Pending
                            } else {
                                active = true;
                                Thread::Active
                            };
                            continue;
                        }

                        // Close every sibling, `Skip` ones included: this
                        // branch stays the donor's, so a mark that lands here
                        // later must open a subtree there and only there.
                        // Leaving them skippable would let both sides open the
                        // same one.
                        *th = Thread::Visited;
                        theirs.sleep_done |= 1 << i;
                    }

                    debug_assert!(active, "[loom internal bug] donated no thread");
                }
                Taken::Values(split) => {
                    let mine = entry.downcast::<Load>(&self.branches).unwrap();
                    let theirs = entry.downcast::<Load>(&thief.branches).unwrap();

                    let mine = mine.get_mut(&mut self.branches);
                    let end = mine.len;
                    mine.len = split as u8;

                    let theirs = theirs.get_mut(&mut thief.branches);
                    theirs.values.copy_within(split..end as usize, 0);
                    theirs.pos = 0;
                    theirs.len = end - split as u8;
                }
            }

            return Some(thief);
        }

        None
    }
}

/// The share `split_off` carves out of one branch.
enum Taken {
    /// Schedule branch: these thread indices become the thief's to explore.
    Threads(Vec<usize>),
    /// Load branch: values from this index on become the thief's.
    Values(usize),
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

    /// Mark `thread_id` for exploration from this branch, returning the mask
    /// of threads that changed to pending.
    ///
    /// `sleep` names the threads whose subtree here is already accounted for;
    /// marking one would re-derive interleavings equivalent to ones already
    /// walked, so it is skipped — this is where sleep sets do their pruning.
    ///
    /// When `owned` is false this branch belongs to another worker: the mask
    /// is computed but not applied, and the caller routes it to a task of its
    /// own instead.
    fn backtrack(
        &mut self,
        thread_id: thread::Id,
        preemption_bound: Option<u8>,
        sleep: u16,
        owned: bool,
    ) -> u16 {
        assert!(self.exploring);

        if let Some(bound) = preemption_bound {
            assert!(
                self.preemptions <= bound,
                "[loom internal bug] actual = {}, bound = {}",
                self.preemptions,
                bound
            );

            if self.preemptions == bound {
                return 0;
            }
        }

        let thread_id = thread_id.as_usize();

        if thread_id >= self.threads.len() {
            return 0;
        }

        // A disabled target cannot itself be scheduled here, so DPOR falls
        // back to opening every candidate at this branch.
        let candidates: &[usize] = if self.threads[thread_id].is_enabled() {
            &[thread_id]
        } else {
            &[0, 1, 2, 3, 4]
        };

        debug_assert!(MAX_THREADS <= 5, "[loom internal bug] widen `candidates`");

        let mut marked = 0;

        for &i in candidates {
            if i >= self.threads.len() || sleep & (1 << i) != 0 {
                continue;
            }

            if self.threads[i] != Thread::Skip {
                continue;
            }

            marked |= 1 << i;

            if owned {
                self.threads[i] = Thread::Pending;
            }
        }

        marked
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
