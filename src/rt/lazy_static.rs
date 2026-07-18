use crate::rt::synchronize::Synchronize;
use std::{any::Any, collections::HashMap};

pub(crate) struct Set {
    /// Registered statics. The map is retained across iterations and reused —
    /// only its contents are per-execution — so a model that registers statics
    /// does not allocate and free a fresh table every iteration.
    statics: HashMap<StaticKeyId, StaticValue>,

    /// False between `drop` and the next `reset`: the execution is tearing
    /// down and the statics are gone even though the table still exists.
    live: bool,
}

#[derive(Eq, PartialEq, Hash, Copy, Clone)]
pub(crate) struct StaticKeyId(usize);

pub(crate) struct StaticValue {
    pub(crate) sync: Synchronize,
    v: Box<dyn Any>,
}

impl Set {
    /// Create an empty statics set.
    pub(crate) fn new() -> Set {
        Set {
            statics: HashMap::new(),
            live: true,
        }
    }

    pub(crate) fn reset(&mut self) {
        assert!(!self.live, "lazy_static was not dropped during execution");
        debug_assert!(self.statics.is_empty(), "`drop` left statics behind");
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

    pub(crate) fn get_static<T: 'static>(
        &mut self,
        key: &'static crate::lazy_static::Lazy<T>,
    ) -> Option<&mut StaticValue> {
        assert!(self.live, "attempted to access lazy_static during shutdown");
        self.statics.get_mut(&StaticKeyId::new(key))
    }

    pub(crate) fn init_static<T: 'static>(
        &mut self,
        key: &'static crate::lazy_static::Lazy<T>,
        value: StaticValue,
    ) -> &mut StaticValue {
        assert!(self.live, "attempted to access lazy_static during shutdown");
        let v = self.statics.entry(StaticKeyId::new(key));

        if let std::collections::hash_map::Entry::Occupied(_) = v {
            unreachable!("told to init static, but it was already init'd");
        }

        v.or_insert(value)
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
