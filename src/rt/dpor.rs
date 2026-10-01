//! Dynamic partial-order reduction for the unbounded search: source sets
//! and wakeup trees (Abdulla, Aronis, Jonsson & Sagonas, *Source Sets: A
//! Foundation for Optimal Dynamic Partial Order Reduction*, JACM 64(4), 2017).
//!
//! An execution is a sequence of *events*, one per schedule branch: the step
//! the chosen thread takes there, which runs its pending operation and the
//! thread's local code up to its next one. Every load-value and spurious
//! branch the step takes belongs to it and is part of its identity (its
//! *key*): a thread's step that read another store is another event.
//!
//! Each execution logs its events with their DPOR clocks and, at each step,
//! the earlier events its operation raced: dependent and concurrent. Once
//! the execution ends, every race of an event `e'` this execution explored
//! first is reversed at the branch its earlier event `e` was chosen at, when
//! `e` is an immediate predecessor of `e'` and nothing but `e'`'s own
//! operation orders `e'` after `e`. Happens-before here includes the order a
//! wake imposes (`Event::requires`): a step a notification woke can never
//! run before the notification, whatever the two operations touch.
//!
//! The reversal is the sequence `v = notdep(e).e'`: the events between `e`
//! and `e'` not ordered after `e`, then `e'`. It is owed nothing when an
//! *initial* of `v` — a thread whose first event in `v` follows no other
//! event of `v` — is asleep at the branch, explored there, or a child of
//! its wakeup tree, which `v` then descends through by that child's key:
//! every execution `v` leads to starts with that thread's step. Otherwise
//! `v` joins the wakeup tree as a new child, starting with an initial that
//! can run there. A branch explores its children in the tree's order, each
//! following its subtree, and an explored child sleeps for the ones after
//! it. A thread that merely commutes with all of `v` covers nothing: `v`
//! stops at `e'`, and the continuation may still need that thread after an
//! event it depends on.
//!
//! A branch frozen for sharding is claimed by many workers and keeps no tree:
//! a reversal there opens an initial of `v` in the claim record unless one is
//! open already, the source-set rule (the paper's Algorithm 1). Where every
//! initial of `v` is blocked at the branch, only `e` or what follows it can
//! unblock them, so no execution runs `e'` before `e`; where they spin, a
//! runnable thread that commutes with all of `v` steps first. The scout tail
//! a sleep set cuts reverses no race: its continuations are covered where
//! its sleepers fell asleep, and their races are reversed there.

use crate::rt::access::Access;
use crate::rt::object::Operation;
use crate::rt::{thread, Path, VersionVec, MAX_THREADS};

#[cfg(feature = "checkpoint")]
use serde::{Deserialize, Serialize};

/// A wakeup tree, or the subtree under one of its children: the
/// alternatives a branch still owes, in the order it explores them, each
/// with the continuation that reverses the race it was inserted for.
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "checkpoint", derive(Serialize, Deserialize))]
pub(crate) struct Wakeup {
    kids: Vec<Kid>,
}

/// One child of a wakeup tree: a thread's step, and what follows it.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "checkpoint", derive(Serialize, Deserialize))]
pub(crate) struct Kid {
    thread: u8,

    /// The continuation after the step, per key the step can take. A key
    /// with no entry continues unconstrained — the ordinary search, which
    /// explores every continuation — and so does every key when this is
    /// empty: the child is a leaf.
    subs: Vec<(Box<[u64]>, Wakeup)>,
}

impl Wakeup {
    pub(crate) fn is_empty(&self) -> bool {
        self.kids.is_empty()
    }

    pub(crate) fn clear(&mut self) {
        self.kids.clear();
    }

    /// Pass `f` the start of every heap block the tree owns.
    pub(crate) fn storage(&self, f: &mut impl FnMut(*const u8)) {
        if self.kids.capacity() != 0 {
            f(self.kids.as_ptr().cast());
        }
        for kid in &self.kids {
            if kid.subs.capacity() != 0 {
                f(kid.subs.as_ptr().cast());
            }
            for (key, sub) in &kid.subs {
                if !key.is_empty() {
                    f(key.as_ptr().cast());
                }
                sub.storage(f);
            }
        }
    }

    /// The children's threads, in exploration order.
    pub(crate) fn threads(&self) -> impl Iterator<Item = usize> + '_ {
        self.kids.iter().map(|kid| kid.thread as usize)
    }

    pub(crate) fn contains(&self, thread: usize) -> bool {
        self.kids.iter().any(|kid| kid.thread as usize == thread)
    }

    /// Add `thread` as a leaf child, after the others.
    pub(crate) fn push_leaf(&mut self, thread: usize) {
        debug_assert!(!self.contains(thread), "[loom internal bug] duplicate child");
        self.kids.push(Kid {
            thread: thread as u8,
            subs: Vec::new(),
        });
    }

    /// Drop `thread`'s child: its subtree is explored.
    pub(crate) fn remove(&mut self, thread: usize) {
        self.kids.retain(|kid| kid.thread as usize != thread);
    }

    /// Keep only `thread`'s child.
    pub(crate) fn retain_only(&mut self, thread: usize) {
        self.kids.retain(|kid| kid.thread as usize == thread);
    }

    /// Keep only the children `keep` names.
    pub(crate) fn retain_mask(&mut self, keep: u16) {
        self.kids.retain(|kid| keep & (1 << kid.thread) != 0);
    }

    /// The continuation `thread`'s step owes after taking `key`, removed from
    /// the tree; empty when there is none.
    pub(crate) fn take(&mut self, thread: usize, key: &[u64]) -> Wakeup {
        let Some(kid) = self.kids.iter_mut().find(|kid| kid.thread as usize == thread) else {
            return Wakeup::default();
        };

        match kid.subs.iter().position(|(k, _)| **k == *key) {
            Some(i) => kid.subs.swap_remove(i).1,
            None => Wakeup::default(),
        }
    }

    /// Insert `w` (event indices into `rev`'s log) below this node: descend
    /// through the child whose step is an initial of what remains of `w`,
    /// taking the same key, and where none is, add the rest as a new chain
    /// after the existing children. Children in `skip` are passed over.
    ///
    /// Covered, with nothing added, when the descent reaches a leaf or a key
    /// with no continuation: the ordinary search below explores every
    /// execution that step leads to. A child whose thread merely commutes
    /// with all of `w` is not descended through. That would place `w` after
    /// its step, which reaches the same trace only when the thread can
    /// still go first in the execution `w` was cut from, and the cut does
    /// not show whether it can.
    fn insert(&mut self, w: &mut Vec<u32>, skip: u16, rev: &Reversal<'_>) {
        for kid in &mut self.kids {
            let q = kid.thread as usize;
            if skip & (1 << q) != 0 {
                continue;
            }

            let Some(i) = rev.first(w, q).filter(|&i| rev.is_initial(w, i)) else {
                continue;
            };

            let key = rev.log.key(w[i]);
            let Some((_, sub)) = kid.subs.iter_mut().find(|(k, _)| **k == *key) else {
                return;
            };
            w.remove(i);

            if !sub.is_empty() && !w.is_empty() {
                sub.insert(w, 0, rev);
            }
            return;
        }

        self.kids.push(Kid::chain(w, rev));
    }
}

impl Kid {
    /// `w` as a chain of single children, the last a leaf.
    fn chain(w: &[u32], rev: &Reversal<'_>) -> Kid {
        let (&first, rest) = w.split_first().expect("[loom internal bug] empty chain");

        let subs = if rest.is_empty() {
            Vec::new()
        } else {
            let key: Box<[u64]> = rev.log.key(first).into();
            vec![(
                key,
                Wakeup {
                    kids: vec![Kid::chain(rest, rev)],
                },
            )]
        };

        Kid {
            thread: rev.log.events[first as usize].thread,
            subs,
        }
    }
}

/// One race found in an execution: an earlier event's access, by the path
/// index of its schedule branch, and its DPOR clock.
#[derive(Debug, Clone)]
struct Race {
    pos: usize,
    version: VersionVec,
}

/// One step of the execution.
#[derive(Debug, Clone)]
pub(crate) struct Event {
    thread: u8,

    /// Path index of the schedule branch the step was chosen at.
    pos: usize,

    /// The operation the step ran; `None` for a step of local code only.
    op: Option<Operation>,

    /// Dependent with every other step, past what `op` says: a timeout
    /// firing, whose enabledness hangs on what else can run, or a step whose
    /// operation reached accesses of other objects (page events).
    opaque: bool,

    /// The thread's DPOR clock when the step ended.
    clock: VersionVec,

    /// Threads asleep at the branch.
    asleep: u16,

    /// The step that enabled this one, when it could run only after it:
    /// the step that woke its thread from a wait, or produced the
    /// notification it consumed (the later, if both and they differ). An
    /// order the dependence relation does not see — a woken step may touch
    /// nothing the waker touched — but every execution keeps.
    requires: Option<u32>,

    /// The step's races (`Log::races`).
    races: (u32, u32),

    /// The step's key (`Log::keys`).
    key: (u32, u32),
}

impl Event {
    /// The version an event ordered after this one holds in this thread's
    /// lane. A step of local code advances no clock: what follows it in its
    /// thread is the next operation.
    fn own(&self) -> u16 {
        self.clock.lane(self.thread as usize) + self.op.is_none() as u16
    }
}

/// The events of the execution under way.
#[derive(Debug, Default)]
pub(crate) struct Log {
    events: Vec<Event>,
    races: Vec<Race>,
    keys: Vec<u64>,

    /// Whether the last event's step is still running.
    open: bool,

    /// The first event of the execution's scout tail (`Execution::skip`),
    /// whose races are not reversed.
    scout: Option<usize>,

    /// Per thread, the step that woke it, for its next step to require.
    woken: [Option<u32>; MAX_THREADS],

    /// Per event, its clock joined with everything it requires, transitively
    /// (`Event::requires`): happens-before with the enabling order added.
    /// Computed when the races are reversed.
    ext: Vec<VersionVec>,
}

impl Log {
    pub(crate) fn clear(&mut self) {
        self.events.clear();
        self.races.clear();
        self.keys.clear();
        self.open = false;
        self.scout = None;
        self.woken = [None; MAX_THREADS];
    }

    /// The execution explores nothing from the next step on.
    pub(crate) fn scout(&mut self) {
        if self.scout.is_none() {
            self.scout = Some(self.events.len());
        }
    }

    /// Open the step `thread` takes at the branch at `pos`.
    pub(crate) fn begin(&mut self, thread: usize, pos: usize, op: Option<Operation>, asleep: u16) {
        debug_assert!(!self.open, "[loom internal bug] a step is already open");

        let races = self.races.len() as u32;
        let keys = self.keys.len() as u32;

        self.events.push(Event {
            thread: thread as u8,
            pos,
            op,
            opaque: false,
            clock: VersionVec::new(),
            asleep,
            requires: self.woken[thread].take(),
            races: (races, races),
            key: (keys, keys),
        });
        self.open = true;
    }

    /// The step that just ended woke `thread` (`Thread::woken`).
    pub(crate) fn woke(&mut self, thread: usize) {
        self.woken[thread] = self.events.len().checked_sub(1).map(|i| i as u32);
    }

    /// The running step, if any, by index.
    pub(crate) fn current(&self) -> Option<u32> {
        self.open.then(|| self.events.len() as u32 - 1)
    }

    /// The running step consumes what step `enabler` produced, so it runs
    /// after it in every execution (`Event::requires`).
    pub(crate) fn require(&mut self, enabler: Option<u32>) {
        if let (true, Some(enabler)) = (self.open, enabler) {
            let event = self.events.last_mut().unwrap();
            event.requires = Some(event.requires.map_or(enabler, |r| r.max(enabler)));
        }
    }

    /// The open step raced the access recorded at branch `pos` with clock
    /// `version`.
    pub(crate) fn race(&mut self, pos: usize, version: &VersionVec) {
        if self.open {
            self.races.push(Race {
                pos,
                version: *version,
            });
            self.events.last_mut().unwrap().races.1 = self.races.len() as u32;
        }
    }

    /// The open step is dependent with every other.
    pub(crate) fn make_opaque(&mut self) {
        if self.open {
            self.events.last_mut().unwrap().opaque = true;
        }
    }

    /// Close the open step, if any, with its thread's clock and the key its
    /// branches took.
    pub(crate) fn end(&mut self, clock: &VersionVec, key: &[u64]) {
        if !self.open {
            return;
        }

        self.open = false;
        self.keys.extend_from_slice(key);

        let end = self.keys.len() as u32;
        let event = self.events.last_mut().unwrap();
        event.clock = *clock;
        event.key.1 = end;
    }

    /// The log as a comparable record, less the clocks the reversals derive
    /// from it.
    pub(crate) fn record(&self) -> String {
        format!(
            "{:?} {:?} {:?} {} {:?} {:?}",
            self.events, self.races, self.keys, self.open, self.scout, self.woken
        )
    }

    fn key(&self, event: u32) -> &[u64] {
        let (a, b) = self.events[event as usize].key;
        &self.keys[a as usize..b as usize]
    }

    /// Whether `g` happens before `f`, `g` earlier in the execution: through
    /// program order, dependence, or what a step requires.
    fn hb(&self, g: u32, f: u32) -> bool {
        let lane = self.events[g as usize].thread as usize;
        self.events[g as usize].thread == self.events[f as usize].thread
            || self.ext[f as usize].lane(lane) >= self.events[g as usize].own()
    }

    /// Fill `ext`. A required step of local code only, with no clock of its
    /// own to join, adds nothing.
    fn extend_clocks(&mut self) {
        self.ext.clear();
        let mut last = [None::<usize>; MAX_THREADS];

        for k in 0..self.events.len() {
            let event = &self.events[k];
            let mut clock = event.clock;

            if let Some(prev) = last[event.thread as usize] {
                clock.join(&self.ext[prev]);
            }
            if let Some(r) = event.requires {
                if self.events[r as usize].op.is_some() {
                    clock.join(&self.ext[r as usize]);
                }
            }

            last[event.thread as usize] = Some(k);
            self.ext.push(clock);
        }
    }

    /// Whether `e'` (index `k`) is ordered after `e` other than by its own
    /// operation: then no execution runs it first, and the race is not one
    /// to reverse.
    fn follows(&self, e: u32, k: usize) -> bool {
        let thread = self.events[k].thread;
        let lane = self.events[e as usize].thread as usize;
        let own = self.events[e as usize].own();

        let prev = self.events[..k].iter().rposition(|f| f.thread == thread);
        prev.is_some_and(|p| self.ext[p].lane(lane) >= own)
            || self.events[k].requires.is_some_and(|r| {
                r == e || (self.events[r as usize].op.is_some() && self.ext[r as usize].lane(lane) >= own)
            })
    }

    /// The event chosen at the branch that holds path index `pos`: the last
    /// one whose branch is at or before it.
    fn at(&self, pos: usize) -> Option<u32> {
        self.events
            .partition_point(|event| event.pos <= pos)
            .checked_sub(1)
            .map(|i| i as u32)
    }

    /// Reverse the races of every event at or after path index `fresh`: the
    /// steps this execution explored first, short of its scout tail. Each
    /// reversal goes to `reverse` with the branch it belongs at.
    pub(crate) fn reverse_races(&mut self, fresh: usize, mut reverse: impl FnMut(usize, &Reversal<'_>)) {
        self.extend_clocks();
        let this = &*self;
        this.reverse_from(fresh, &mut reverse);
    }

    fn reverse_from(&self, fresh: usize, reverse: &mut impl FnMut(usize, &Reversal<'_>)) {
        let first = self.events.partition_point(|event| event.pos < fresh);
        let last = self.scout.unwrap_or(self.events.len());
        let mut seq = Vec::new();
        let mut racers: Vec<(u32, VersionVec)> = Vec::new();

        for k in first..last {
            let target = &self.events[k];
            let (a, b) = target.races;

            if a == b {
                continue;
            }

            // The earlier events, each once, with the clock of its access.
            racers.clear();
            for race in &self.races[a as usize..b as usize] {
                let Some(e) = self.at(race.pos) else { continue };
                if e as usize >= k {
                    continue;
                }
                match racers.iter_mut().find(|(r, _)| *r == e) {
                    Some((_, version)) => version.join(&race.version),
                    None => racers.push((e, race.version)),
                }
            }

            for &(e, ref version) in &racers {
                let racer = &self.events[e as usize];
                let lane = racer.thread as usize;

                // A race only with an immediate predecessor: one ordered
                // before another of `e'`'s races is not reversible alone.
                let shadowed = racers.iter().any(|&(o, ref other)| {
                    o != e && other.lane(lane) >= version.lane(lane)
                });
                if shadowed {
                    continue;
                }

                if self.follows(e, k) {
                    continue;
                }

                seq.clear();
                for f in e + 1..k as u32 {
                    if !self.hb(e, f) {
                        seq.push(f);
                    }
                }
                seq.push(k as u32);

                let rev = Reversal::new(self, e, &seq);
                reverse(racer.pos, &rev);
            }
        }
    }
}

/// One race to reverse: the sequence `v = notdep(e).e'` at the branch `e`
/// was chosen at, with what the branch's checks need of the execution.
pub(crate) struct Reversal<'a> {
    log: &'a Log,

    /// `v`, as event indices.
    seq: &'a [u32],

    /// The thread of `e`: the branch's current choice.
    racer: usize,

    /// The thread of `e'`.
    target: usize,

    /// Threads asleep at the branch.
    asleep: u16,

    /// Each thread's next step at the branch, when the execution took one.
    next: [Option<u32>; MAX_THREADS],
}

impl<'a> Reversal<'a> {
    fn new(log: &'a Log, e: u32, seq: &'a [u32]) -> Reversal<'a> {
        let racer = &log.events[e as usize];
        let mut next = [None; MAX_THREADS];
        let mut missing = (1u32 << MAX_THREADS) - 1;

        for f in e..log.events.len() as u32 {
            let t = log.events[f as usize].thread as usize;
            if missing & (1 << t) != 0 {
                next[t] = Some(f);
                missing &= !(1 << t);
                if missing == 0 {
                    break;
                }
            }
        }

        Reversal {
            log,
            seq,
            racer: racer.thread as usize,
            target: log.events[*seq.last().unwrap() as usize].thread as usize,
            asleep: racer.asleep,
            next,
        }
    }

    pub(crate) fn seq(&self) -> &[u32] {
        self.seq
    }

    pub(crate) fn racer(&self) -> usize {
        self.racer
    }

    pub(crate) fn target(&self) -> usize {
        self.target
    }

    pub(crate) fn asleep(&self) -> u16 {
        self.asleep
    }

    /// `q`'s next step at the branch, when the execution took one.
    pub(crate) fn next_of(&self, q: usize) -> Option<u32> {
        self.next[q]
    }

    pub(crate) fn thread_of(&self, event: u32) -> usize {
        self.log.events[event as usize].thread as usize
    }

    /// Index in `w` of `q`'s first event.
    fn first(&self, w: &[u32], q: usize) -> Option<usize> {
        w.iter().position(|&f| self.thread_of(f) == q)
    }

    /// Whether `w[i]` happens after no other event of `w`.
    fn is_initial(&self, w: &[u32], i: usize) -> bool {
        w[..i].iter().all(|&g| !self.log.hb(g, w[i]))
    }

    /// Whether `q`'s next step at the branch commutes with every event of
    /// `w`, by their operations: a step dependent with everything (`opaque`)
    /// commutes with nothing. Only meaningful for a thread with no event in
    /// `w`.
    fn independent_of_all(&self, q: usize, w: &[u32]) -> bool {
        let Some(next) = self.next[q] else {
            return false;
        };
        let a = &self.log.events[next as usize];

        !a.opaque
            && w.iter().all(|&f| {
                let b = &self.log.events[f as usize];
                !b.opaque
                    && match (&a.op, &b.op) {
                        (Some(x), Some(y)) => !x.conflicts_with(y),
                        _ => true,
                    }
            })
    }

    /// The threads whose first event in `v` happens after no other: `I(v)`.
    pub(crate) fn initials(&self) -> u16 {
        let mut seen = 0u16;
        let mut initials = 0u16;

        for (i, &f) in self.seq.iter().enumerate() {
            let t = self.thread_of(f);
            if seen & (1 << t) == 0 {
                seen |= 1 << t;
                if self.is_initial(self.seq, i) {
                    initials |= 1 << t;
                }
            }
        }

        initials
    }

    /// A thread among `usable` with no event in `v` whose next step commutes
    /// with all of it.
    pub(crate) fn runnable_weak_initial(&self, usable: u16) -> Option<usize> {
        (0..MAX_THREADS).find(|&y| {
            usable & (1 << y) != 0
                && self.first(self.seq, y).is_none()
                && self.independent_of_all(y, self.seq)
        })
    }

    /// `v` reordered to start with `q`'s first event, an initial.
    pub(crate) fn starting_with(&self, q: usize) -> Vec<u32> {
        let i = self.first(self.seq, q).expect("[loom internal bug] not in v");
        let mut w = Vec::with_capacity(self.seq.len());
        w.push(self.seq[i]);
        w.extend(self.seq.iter().enumerate().filter(|&(j, _)| j != i).map(|(_, &f)| f));
        w
    }

    /// Whether a child of the branch's wakeup tree `tree` is an initial of
    /// `v`, descending through it to insert the rest (`Wakeup::insert`);
    /// `skip` names the branch's current choice.
    pub(crate) fn covered_by_tree(&self, tree: &mut Wakeup, skip: u16) -> bool {
        let initials = self.initials();
        if !tree.threads().any(|q| skip & (1 << q) == 0 && initials & (1 << q) != 0) {
            return false;
        }

        tree.insert(&mut self.seq.to_vec(), skip, self);
        true
    }

    /// Append `w` (whose first event's thread can be scheduled at the
    /// branch) to the branch's wakeup tree as a new child.
    pub(crate) fn append(&self, tree: &mut Wakeup, w: &[u32]) {
        tree.kids.push(Kid::chain(w, self));
    }
}

/// A race the running step's operation found outside the scheduler's scan,
/// against the access `access` (page events): marked at once under a
/// preemption bound, logged for reversal without one.
pub(crate) fn race(path: &mut Path, log: &mut Log, access: &Access, thread: thread::Id) {
    if path.is_bounded() {
        path.backtrack(access.path_id(), thread);
    } else {
        log.race(access.path_id(), access.version());
    }
}
