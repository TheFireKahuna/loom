use crate::rt::synchronize::Synchronize;
use crate::rt::thread;
use std::{any::Any, collections::HashMap};

pub(crate) struct Set {
    /// Registered statics. The map is retained across iterations and reused —
    /// only its contents are per-execution — so a model that registers statics
    /// does not allocate and free a fresh table every iteration.
    statics: HashMap<StaticKeyId, StaticValue>,

    /// Statics whose initializer is running, each with the threads blocked
    /// until it is done.
    running: HashMap<StaticKeyId, Vec<thread::Id>>,

    /// False between `drop` and the next `reset`: the execution is tearing
    /// down and the statics are gone even though the table still exists.
    live: bool,
}

#[derive(Eq, PartialEq, Hash, Copy, Clone)]
pub(crate) struct StaticKeyId(usize);

/// What a thread reaching a static must do.
pub(crate) enum Claim<'a> {
    /// Use the value.
    Done(&'a mut StaticValue),
    /// Block until the running initializer is done; the thread is queued.
    Wait,
    /// Run the initializer: the static is now running on this thread.
    Init,
}

pub(crate) struct StaticValue {
    pub(crate) sync: Synchronize,
    v: Box<dyn Any>,
}

impl Set {
    /// Create an empty statics set.
    pub(crate) fn new() -> Set {
        Set {
            statics: HashMap::new(),
            running: HashMap::new(),
            live: true,
        }
    }

    pub(crate) fn reset(&mut self) {
        // A live set is only tolerable when it is empty (a never-run
        // execution being re-armed at a worker task boundary): live and
        // non-empty means an execution ended without dropping its statics.
        assert!(
            !self.live || self.statics.is_empty(),
            "lazy_static was not dropped during execution"
        );
        debug_assert!(self.statics.is_empty(), "`drop` left statics behind");
        // A failed execution can end with an initializer still running.
        self.running.clear();
        self.live = true;
    }

    /// Hand the execution's statics out to be dropped by the caller — outside
    /// the execution borrow, since a value's `Drop` may re-enter the runtime.
    /// The table itself stays here, emptied and allocated, for the next
    /// iteration.
    pub(crate) fn drop(&mut self) -> Vec<StaticValue> {
        assert!(self.live, "lazy_statics were dropped twice in one execution");
        self.live = false;
        self.statics.drain().map(|(_, value)| value).collect()
    }

    /// `thread` reaches `key`: the value if it is initialized, otherwise the
    /// initializer to run or to wait for.
    pub(crate) fn claim<T: 'static>(
        &mut self,
        key: &'static crate::lazy_static::Lazy<T>,
        thread: thread::Id,
    ) -> Claim<'_> {
        assert!(self.live, "attempted to access lazy_static during shutdown");
        let id = StaticKeyId::new(key);
        if let Some(value) = self.statics.get_mut(&id) {
            return Claim::Done(value);
        }
        match self.running.get_mut(&id) {
            Some(waiters) => {
                waiters.push(thread);
                Claim::Wait
            }
            None => {
                self.running.insert(id, Vec::new());
                Claim::Init
            }
        }
    }

    /// The claimed initializer of `key` returned `value`: the static is
    /// initialized, and the threads waiting on it are handed back to wake.
    pub(crate) fn init_static<T: 'static>(
        &mut self,
        key: &'static crate::lazy_static::Lazy<T>,
        value: StaticValue,
    ) -> (&mut StaticValue, Vec<thread::Id>) {
        assert!(self.live, "attempted to access lazy_static during shutdown");
        let id = StaticKeyId::new(key);
        let waiters = self
            .running
            .remove(&id)
            .expect("told to init static, but it was not claimed");
        let v = self.statics.entry(id);

        if let std::collections::hash_map::Entry::Occupied(_) = v {
            unreachable!("told to init static, but it was already init'd");
        }

        (v.or_insert(value), waiters)
    }
}

impl StaticKeyId {
    fn new<T>(key: &'static crate::lazy_static::Lazy<T>) -> Self {
        Self(key as *const _ as usize)
    }
}

impl StaticValue {
    pub(crate) fn new<T: 'static>(value: T) -> Self {
        Self {
            sync: Synchronize::new(),
            v: Box::new(value),
        }
    }

    pub(crate) fn get<T: 'static>(&self) -> &T {
        self.v
            .downcast_ref::<T>()
            .expect("lazy value must downcast to expected type")
    }
}
