use crate::rt;
use crate::rt::{thread, Access, Execution, Location, VersionVec, MAX_THREADS};

use std::fmt;
use std::marker::PhantomData;

use tracing::trace;

#[cfg(feature = "checkpoint")]
use serde::{Deserialize, Serialize};

/// Stores objects
///
/// Executions are epochs over an almost-identical object graph: iteration
/// i+1 re-creates the objects iteration i dropped, in the same order, because
/// creation order is deterministic given the schedule prefix and consecutive
/// executions share most of it. So the store never frees at an epoch
/// boundary — `begin_epoch` resets a cursor and the old entries become
/// carcasses, reincarnated in place slot-by-slot as the next execution
/// re-creates the same objects. Steady state allocates nothing.
///
/// `entries[..live]` are the current epoch's objects; `entries[live..]` are
/// carcasses awaiting reuse. Every read path is bounded by `live` — a carcass
/// must be invisible (a fence scanning a stale atomic's stores would absorb
/// synchronization from a previous execution).
#[derive(Debug, Clone)]
#[cfg_attr(feature = "checkpoint", derive(Serialize, Deserialize))]
pub(super) struct Store<T = Entry> {
    /// Stored state for all objects, current epoch then carcasses.
    entries: Vec<T>,

    /// Number of live (current-epoch) entries. Stores that never call
    /// `begin_epoch` (the path's branch store) keep `live == entries.len()`.
    live: usize,

    /// Access records of the objects that have no entry (`Ref::SC_ORDER`,
    /// `Ref::park`). Per-epoch like the entries.
    #[cfg_attr(feature = "checkpoint", serde(skip))]
    virtual_accesses: Box<VirtualAccesses>,
}

/// DPOR records of the objects that exist only as orderings, with no modeled
/// state of their own. Records are per thread, like an atomic's: an object
/// whose operations are not all mutually dependent must not let one thread's
/// access shadow another's.
#[derive(Debug, Clone, Default)]
struct VirtualAccesses {
    /// Each thread's last `SeqCst` fence.
    sc_fences: [Option<Access>; MAX_THREADS],

    /// Each thread's last `SeqCst` atomic access, of any cell.
    sc_accesses: [Option<Access>; MAX_THREADS],

    /// Each thread's last `park` (on its own park object).
    parks: [Option<Access>; MAX_THREADS],

    /// `unparks[target][unparker]`: each thread's last `unpark` of `target`.
    unparks: [[Option<Access>; MAX_THREADS]; MAX_THREADS],
}

pub(super) trait Object: Sized {
    type Entry;

    /// Convert an object into an entry
    fn into_entry(self) -> Self::Entry;

    /// Convert an entry ref into an object ref
    fn get_ref(entry: &Self::Entry) -> Option<&Self>;

    /// Convert a mutable entry ref into a mutable object ref
    fn get_mut(entry: &mut Self::Entry) -> Option<&mut Self>;
}

/// References an object in the store.
///
/// The reference tracks the type it references. Using `()` indicates the type
/// is unknown.
#[derive(Eq, PartialEq)]
#[cfg_attr(feature = "checkpoint", derive(Serialize, Deserialize))]
pub(super) struct Ref<T = ()> {
    /// Index in the store
    index: usize,

    _p: PhantomData<T>,
}

// TODO: mov to separate file
#[derive(Debug, Copy, Clone)]
pub(super) struct Operation {
    obj: Ref,
    action: Action,
    location: Location,

    /// The operation takes a place in the SC total order S — a `SeqCst`
    /// access or fence — whose position against every SC fence is
    /// observable, whatever object each touches.
    sc: bool,
}

// TODO: move to separate file
#[derive(Debug, Copy, Clone, PartialEq)]
pub(super) enum Action {
    /// Action on an Arc object
    Arc(rt::arc::Action),

    /// Action on an atomic object
    Atomic(rt::atomic::Action),

    /// Action on a channel
    Channel(rt::mpsc::Action),

    /// Action on a RwLock
    RwLock(rt::rwlock::Action),

    /// `park` on the parking thread's own park object: consumes the token.
    Park,

    /// `unpark` on the target's park object: makes the token available.
    /// Unparks commute with one another; each is dependent with `Park`.
    Unpark,

    /// Generic action with no specialized dependencies on access.
    Opaque,
}

macro_rules! objects {
    ( $(#[$attrs:meta])* $e:ident, $( $name:ident($ty:path), )* ) => {

        $(#[$attrs])*
        pub(super) enum $e {

            $(
                $name($ty),
            )*
        }

        $(
            impl crate::rt::object::Object for $ty {
                type Entry = $e;

                fn into_entry(self) -> Entry {
                    $e::$name(self)
                }

                fn get_ref(entry: &Entry) -> Option<&$ty> {
                    match entry {
                        $e::$name(obj) => Some(obj),
                        _ => None,
                    }
                }

                fn get_mut(entry: &mut Entry) -> Option<&mut $ty> {
                    match entry {
                        $e::$name(obj) => Some(obj),
                        _ => None,
                    }
                }
            }
        )*
    };
}

objects! {
    #[derive(Debug)]
    // Many of the common variants of this enum are quite large --- only `Entry`
    // and `Alloc` are significantly smaller than most other variants.
    #[allow(clippy::large_enum_variant)]
    Entry,

    // State tracking allocations. Used for leak detection.
    Alloc(rt::alloc::State),

    // State associated with a modeled `Arc`.
    Arc(rt::arc::State),

    // State associated with an atomic cell
    Atomic(rt::atomic::State),

    // State associated with a mutex.
    Mutex(rt::mutex::State),

    // State associated with a modeled condvar.
    Condvar(rt::condvar::State),

    // State associated with a modeled thread notifier.
    Notify(rt::notify::State),

    // State associated with an RwLock
    RwLock(rt::rwlock::State),

    // State associated with a modeled channel.
    Channel(rt::mpsc::State),

    // Tracks access to a memory cell
    Cell(rt::cell::State),
}

impl<T> Store<T> {
    /// Create a new, empty, object store
    pub(super) fn with_capacity(capacity: usize) -> Store<T> {
        Store {
            entries: Vec::with_capacity(capacity),
            live: 0,
            virtual_accesses: Box::default(),
        }
    }

    pub(super) fn len(&self) -> usize {
        self.live
    }

    pub(super) fn capacity(&self) -> usize {
        self.entries.capacity()
    }

    pub(super) fn reserve_exact(&mut self, additional: usize) {
        self.entries.reserve_exact(additional);
    }

    /// Insert an object into the store, overwriting a carcass in place when
    /// one is available (any variant — assignment drops the old entry without
    /// touching the `Vec`).
    pub(super) fn insert<O>(&mut self, item: O) -> Ref<O>
    where
        O: Object<Entry = T>,
    {
        let index = self.live;
        if index < self.entries.len() {
            self.entries[index] = item.into_entry();
        } else {
            self.entries.push(item.into_entry());
        }
        self.live += 1;

        Ref {
            index,
            _p: PhantomData,
        }
    }

    /// Insert an object, reincarnating a same-variant carcass in place.
    ///
    /// `reuse` must leave the carcass in exactly the state `make` constructs —
    /// same fields, same reachable history — reusing its allocations.
    ///
    /// A carcass of a different variant is overwritten where it lies, and the
    /// tail behind it is left alone. Truncating there instead would be a
    /// memory heuristic paid for at exactly the wrong time: object creation
    /// order is only deterministic for eagerly constructed cells, and a
    /// deferred one is numbered by *first access*, so a mismatch is an
    /// ordinary consequence of exploring a different schedule rather than
    /// evidence the object graph changed shape. Dropping the tail on each such
    /// mismatch would discard every surviving carcass — and with them
    /// `atomic::State::spares` and every inner allocation — turning
    /// `reuse_objects` into a per-iteration rebuild. Keeping them costs only
    /// the memory of a wrong-variant entry until `live` reaches it again, and
    /// they stay invisible meanwhile: every read path is bounded by `live`
    /// (`iter_ref`, `iter_mut`, `check_for_leaks`), and `insert`/`insert_with`
    /// each reset whatever entry they land on. Plain `insert` has always
    /// overwritten any variant in place for the same reason.
    pub(super) fn insert_with<O>(
        &mut self,
        make: impl FnOnce() -> O,
        reuse: impl FnOnce(&mut O),
    ) -> Ref<O>
    where
        O: Object<Entry = T>,
    {
        let index = self.live;
        if index < self.entries.len() {
            match O::get_mut(&mut self.entries[index]) {
                Some(obj) => reuse(obj),
                None => self.entries[index] = make().into_entry(),
            }
        } else {
            self.entries.push(make().into_entry());
        }
        self.live += 1;

        Ref {
            index,
            _p: PhantomData,
        }
    }

    pub(crate) fn truncate<O>(&mut self, obj: Ref<O>) {
        let target = obj.index + 1;
        self.entries.truncate(target);
        self.live = self.live.min(target);
    }

    pub(crate) fn clear(&mut self) {
        self.entries.clear();
        self.live = 0;
        *self.virtual_accesses = VirtualAccesses::default();
    }

    /// Start a new epoch: every entry becomes a carcass available for
    /// in-place reincarnation by this epoch's `insert`/`insert_with` calls.
    /// Nothing is dropped here.
    pub(crate) fn begin_epoch(&mut self) {
        self.live = 0;
        *self.virtual_accesses = VirtualAccesses::default();
    }

    pub(super) fn iter_ref<O>(&self) -> impl DoubleEndedIterator<Item = Ref<O>> + '_
    where
        O: Object<Entry = T>,
    {
        self.entries[..self.live]
            .iter()
            .enumerate()
            .filter(|(_, e)| O::get_ref(e).is_some())
            .map(|(index, _)| Ref {
                index,
                _p: PhantomData,
            })
    }

    pub(super) fn iter_mut<'a, O>(&'a mut self) -> impl DoubleEndedIterator<Item = &'a mut O>
    where
        O: Object<Entry = T> + 'a,
    {
        let live = self.live;
        self.entries[..live].iter_mut().filter_map(O::get_mut)
    }
}

impl Store {
    /// Calls `f` with every dependent access of the operation.
    ///
    /// Atomics track dependent accesses per thread (see `atomic::State` —
    /// a single shared slot lets a thread's own access shadow a peer's,
    /// silently dropping the DPOR reorder owed to that conflict); the other
    /// object types keep their single last-access slot and yield it here.
    pub(super) fn for_each_dependent_access(&self, operation: Operation, mut f: impl FnMut(&Access)) {
        let virt = &*self.virtual_accesses;

        if operation.obj == Ref::SC_ORDER {
            // A fence is dependent with every SC fence and SC access.
            virt.sc_fences
                .iter()
                .chain(&virt.sc_accesses)
                .flatten()
                .for_each(f);
            return;
        }

        if let Some(target) = operation.obj.park_target() {
            match operation.action {
                Action::Park => virt.unparks[target].iter().flatten().for_each(f),
                Action::Unpark => virt.parks[target].iter().for_each(f),
                _ => unreachable!("park object touched by {:?}", operation.action),
            }
            return;
        }

        // An SC access is dependent with every SC fence, in addition to its
        // own object's accesses.
        if operation.sc {
            virt.sc_fences.iter().flatten().for_each(&mut f);
        }

        match &self.entries[operation.obj.index] {
            Entry::Atomic(entry) => entry.for_each_dependent_access(operation.action.into(), f),
            Entry::Arc(entry) => {
                if let Some(access) = entry.last_dependent_access(operation.action.into()) {
                    f(access);
                }
            }
            Entry::Mutex(entry) => {
                if let Some(access) = entry.last_dependent_access() {
                    f(access);
                }
            }
            Entry::Condvar(entry) => {
                if let Some(access) = entry.last_dependent_access() {
                    f(access);
                }
            }
            Entry::Notify(entry) => {
                if let Some(access) = entry.last_dependent_access() {
                    f(access);
                }
            }
            Entry::RwLock(entry) => {
                if let Some(access) = entry.last_dependent_access() {
                    f(access);
                }
            }
            Entry::Channel(entry) => {
                if let Some(access) = entry.last_dependent_access(operation.action.into()) {
                    f(access);
                }
            }
            obj => panic!(
                "object is not branchable {:?}; ref = {:?}",
                obj, operation.obj
            ),
        }
    }

    pub(super) fn set_last_access(
        &mut self,
        operation: Operation,
        thread_id: thread::Id,
        path_id: usize,
        dpor_vv: &VersionVec,
    ) {
        let virt = &mut *self.virtual_accesses;
        let thread = thread_id.as_usize();

        if operation.obj == Ref::SC_ORDER {
            Access::set_or_create(&mut virt.sc_fences[thread], path_id, dpor_vv);
            return;
        }

        if let Some(target) = operation.obj.park_target() {
            let record = match operation.action {
                Action::Park => &mut virt.parks[target],
                Action::Unpark => &mut virt.unparks[target][thread],
                _ => unreachable!("park object touched by {:?}", operation.action),
            };
            Access::set_or_create(record, path_id, dpor_vv);
            return;
        }

        if operation.sc {
            Access::set_or_create(&mut virt.sc_accesses[thread], path_id, dpor_vv);
        }

        match &mut self.entries[operation.obj.index] {
            Entry::Arc(entry) => entry.set_last_access(operation.action.into(), path_id, dpor_vv),
            Entry::Atomic(entry) => {
                entry.set_last_access(operation.action.into(), thread_id, path_id, dpor_vv)
            }
            Entry::Mutex(entry) => entry.set_last_access(path_id, dpor_vv),
            Entry::Condvar(entry) => entry.set_last_access(path_id, dpor_vv),
            Entry::Notify(entry) => entry.set_last_access(path_id, dpor_vv),
            Entry::RwLock(entry) => entry.set_last_access(path_id, dpor_vv),
            Entry::Channel(entry) => {
                entry.set_last_access(operation.action.into(), path_id, dpor_vv)
            }
            _ => panic!("object is not branchable"),
        }
    }

    /// Panics if any leaks were detected
    pub(crate) fn check_for_leaks(&self) {
        for (index, entry) in self.entries[..self.live].iter().enumerate() {
            match entry {
                Entry::Alloc(entry) => entry.check_for_leaks(index),
                Entry::Arc(entry) => entry.check_for_leaks(index),
                Entry::Channel(entry) => entry.check_for_leaks(index),
                _ => {}
            }
        }
    }
}

impl Ref {
    /// The SC total order S, an object with no entry: an SC fence operates on
    /// it (`Operation::sc_fence`).
    pub(super) const SC_ORDER: Ref = Ref {
        index: usize::MAX,
        _p: PhantomData,
    };

    /// First index of the per-thread park objects, which have no entry.
    const PARK_BASE: usize = usize::MAX - MAX_THREADS;

    /// Thread `thread`'s park object, which its `park` and every `unpark` of
    /// it operate on.
    pub(super) fn park(thread: thread::Id) -> Ref {
        Ref {
            index: Ref::PARK_BASE + thread.as_usize(),
            _p: PhantomData,
        }
    }

    /// The thread whose park object this is, if it is one.
    fn park_target(self) -> Option<usize> {
        (Ref::PARK_BASE..usize::MAX)
            .contains(&self.index)
            .then(|| self.index - Ref::PARK_BASE)
    }
}

impl<T> Ref<T> {
    /// Erase the type marker
    pub(super) fn erase(self) -> Ref<()> {
        Ref {
            index: self.index,
            _p: PhantomData,
        }
    }

    pub(super) fn ref_eq(self, other: Ref<T>) -> bool {
        self.index == other.index
    }

    /// Position in the store. For path branches this is the branch index,
    /// which is what identifies a point in the execution path.
    pub(super) fn index(self) -> usize {
        self.index
    }
}

impl<T: Object> Ref<T> {
    /// Get a reference to the object associated with this reference from the store
    pub(super) fn get(self, store: &Store<T::Entry>) -> &T {
        debug_assert!(self.index < store.live, "[loom internal bug] ref to carcass");
        T::get_ref(&store.entries[self.index])
            .expect("[loom internal bug] unexpected object stored at reference")
    }

    /// Get a mutable reference to the object associated with this reference
    /// from the store
    pub(super) fn get_mut(self, store: &mut Store<T::Entry>) -> &mut T {
        debug_assert!(self.index < store.live, "[loom internal bug] ref to carcass");
        T::get_mut(&mut store.entries[self.index])
            .expect("[loom internal bug] unexpected object stored at reference")
    }
}

impl Ref {
    /// Convert a store index `usize` into a ref
    pub(super) fn from_usize(index: usize) -> Ref {
        Ref {
            index,
            _p: PhantomData,
        }
    }

    pub(super) fn downcast<T>(self, store: &Store<T::Entry>) -> Option<Ref<T>>
    where
        T: Object,
    {
        T::get_ref(&store.entries[self.index]).map(|_| Ref {
            index: self.index,
            _p: PhantomData,
        })
    }
}

impl<T> Clone for Ref<T> {
    fn clone(&self) -> Ref<T> {
        Ref {
            index: self.index,
            _p: PhantomData,
        }
    }
}

impl<T> Copy for Ref<T> {}

impl<T> fmt::Debug for Ref<T> {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        use std::any::type_name;

        write!(fmt, "Ref<{}>({})", type_name::<T>(), self.index)
    }
}

// TODO: These fns shouldn't be on Ref
impl<T: Object<Entry = Entry>> Ref<T> {
    // TODO: rename `branch_disable`
    pub(super) fn branch_acquire(self, is_locked: bool, location: Location) {
        super::branch(|execution| {
            trace!(obj = ?self, ?is_locked, "Object::branch_acquire");

            self.set_action(execution, Action::Opaque, location);

            if is_locked {
                // The mutex is currently blocked, cannot make progress
                execution.threads.active_mut().set_blocked(location, false);
            }
        })
    }

    pub(super) fn branch_action(
        self,
        action: impl Into<Action> + std::fmt::Debug,
        location: Location,
    ) {
        self.branch_ordered_action(action, false, location)
    }

    /// `branch_action` for an operation that may take a place in S: `sc` is
    /// whether it does (a `SeqCst` atomic access).
    pub(super) fn branch_ordered_action(
        self,
        action: impl Into<Action> + std::fmt::Debug,
        sc: bool,
        location: Location,
    ) {
        super::branch(|execution| {
            trace!(obj = ?self, ?action, ?sc, "Object::branch_action");

            self.set_action(execution, action.into(), location);
            if sc {
                if let Some(operation) = &mut execution.threads.active_mut().operation {
                    operation.sc = true;
                }
            }
        })
    }

    pub(super) fn branch_disable(
        self,
        action: impl Into<Action> + std::fmt::Debug,
        disable: bool,
        location: Location,
    ) {
        super::branch(|execution| {
            trace!(obj = ?self, ?action, ?disable, "Object::branch_disable");

            self.set_action(execution, action.into(), location);

            if disable {
                // Cannot make progress.
                execution.threads.active_mut().set_blocked(location, false);
            }
        })
    }

    pub(super) fn branch_opaque(self, location: Location) {
        self.branch_action(Action::Opaque, location)
    }

    fn set_action(self, execution: &mut Execution, action: Action, location: Location) {
        assert!(
            T::get_ref(&execution.objects.entries[self.index]).is_some(),
            "failed to get object for ref {:?}",
            self
        );

        execution.threads.active_mut().operation = Some(Operation {
            obj: self.erase(),
            action,
            location,
            sc: false,
        });
    }
}

impl Operation {
    /// A `SeqCst` fence: an operation on S itself.
    pub(super) fn sc_fence(location: Location) -> Operation {
        Operation {
            obj: Ref::SC_ORDER,
            action: Action::Opaque,
            location,
            sc: true,
        }
    }

    /// `park` by `thread`.
    pub(super) fn park(thread: thread::Id, location: Location) -> Operation {
        Operation {
            obj: Ref::park(thread),
            action: Action::Park,
            location,
            sc: false,
        }
    }

    /// `unpark` of `target`.
    pub(super) fn unpark(target: thread::Id, location: Location) -> Operation {
        Operation {
            obj: Ref::park(target),
            action: Action::Unpark,
            location,
            sc: false,
        }
    }

    /// Whether running `self` can change the dependent access records
    /// `other` is checked against: they share an object, or both sit in S
    /// and one of them is a fence.
    pub(super) fn may_affect(&self, other: &Operation) -> bool {
        self.obj == other.obj || self.sc_linked(other)
    }

    /// Dependence through S alone: an SC fence against any SC operation.
    fn sc_linked(&self, other: &Operation) -> bool {
        self.sc && other.sc && (self.obj == Ref::SC_ORDER || other.obj == Ref::SC_ORDER)
    }

    pub(super) fn object(&self) -> Ref {
        self.obj
    }

    pub(super) fn action(&self) -> Action {
        self.action
    }

    /// DPOR dependence with another operation, decided from the operations
    /// alone. Same-object only; atomics refine by lane mask and access kind;
    /// every other object type is conservatively dependent. For the sleep-set
    /// wake rule an over-approximation only wakes threads earlier — pruning
    /// less, never unsoundly more.
    pub(super) fn conflicts_with(&self, other: &Operation) -> bool {
        if self.obj != other.obj {
            return self.sc_linked(other);
        }

        match (self.action, other.action) {
            (Action::Atomic(a), Action::Atomic(b)) => a.conflicts_with(b),
            (Action::Unpark, Action::Unpark) => false,
            _ => true,
        }
    }

    pub(super) fn location(&self) -> Location {
        self.location
    }
}

impl From<Action> for rt::arc::Action {
    fn from(action: Action) -> Self {
        match action {
            Action::Arc(action) => action,
            _ => unreachable!(),
        }
    }
}

impl From<Action> for rt::atomic::Action {
    fn from(action: Action) -> Self {
        match action {
            Action::Atomic(action) => action,
            _ => unreachable!(),
        }
    }
}

impl From<Action> for rt::mpsc::Action {
    fn from(action: Action) -> Self {
        match action {
            Action::Channel(action) => action,
            _ => unreachable!(),
        }
    }
}

impl From<rt::arc::Action> for Action {
    fn from(action: rt::arc::Action) -> Self {
        Action::Arc(action)
    }
}

impl From<rt::atomic::Action> for Action {
    fn from(action: rt::atomic::Action) -> Self {
        Action::Atomic(action)
    }
}

impl From<rt::mpsc::Action> for Action {
    fn from(action: rt::mpsc::Action) -> Self {
        Action::Channel(action)
    }
}

impl From<rt::rwlock::Action> for Action {
    fn from(action: rt::rwlock::Action) -> Self {
        Action::RwLock(action)
    }
}

impl PartialEq<rt::rwlock::Action> for Action {
    fn eq(&self, other: &rt::rwlock::Action) -> bool {
        let other: Action = (*other).into();
        *self == other
    }
}
