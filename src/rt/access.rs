use crate::rt::VersionVec;

#[derive(Debug, Clone)]
pub(crate) struct Access {
    path_id: usize,
    dpor_vv: VersionVec,
}

impl Access {
    pub(crate) fn new(path_id: usize, version: &VersionVec) -> Access {
        Access {
            path_id,
            dpor_vv: *version,
        }
    }

    pub(crate) fn set(&mut self, path_id: usize, version: &VersionVec) {
        self.path_id = path_id;
        self.dpor_vv = *version;
    }

    pub(crate) fn set_or_create(access: &mut Option<Self>, path_id: usize, version: &VersionVec) {
        if let Some(access) = access.as_mut() {
            access.set(path_id, version);
        } else {
            *access = Some(Access::new(path_id, version));
        }
    }

    /// Location in the path
    pub(crate) fn path_id(&self) -> usize {
        self.path_id
    }

    pub(crate) fn version(&self) -> &VersionVec {
        &self.dpor_vv
    }

    pub(crate) fn happens_before(&self, version: &VersionVec) -> bool {
        self.dpor_vv.is_le(version)
    }
}

/// A lock's operation, as its DPOR record sees it.
#[derive(Debug, Copy, Clone)]
pub(crate) enum LockOp {
    /// A blocking acquire.
    Acquire,
    /// A non-blocking attempt, which acquires when the lock is free.
    Try,
    /// A guard's release.
    Unlock,
}

/// A lock's DPOR record. A blocking acquire races only the acquires before
/// it: an unlock enables it, and no schedule runs it first. An attempt races
/// acquires and unlocks both, since either changes its answer, and an unlock
/// races the attempts it would answer.
#[derive(Debug, Default)]
pub(crate) struct LockAccesses {
    acquire: Option<Access>,
    unlock: Option<Access>,
    attempt: Option<Access>,
}

impl LockAccesses {
    pub(crate) fn for_each_dependent(&self, op: LockOp, f: impl FnMut(&Access)) {
        let (a, b) = match op {
            LockOp::Acquire => (&self.acquire, &None),
            LockOp::Try => (&self.acquire, &self.unlock),
            LockOp::Unlock => (&self.attempt, &None),
        };
        a.iter().chain(b).for_each(f);
    }

    pub(crate) fn record(&mut self, op: LockOp, path_id: usize, version: &VersionVec) {
        match op {
            LockOp::Acquire => Access::set_or_create(&mut self.acquire, path_id, version),
            LockOp::Try => {
                Access::set_or_create(&mut self.acquire, path_id, version);
                Access::set_or_create(&mut self.attempt, path_id, version);
            }
            LockOp::Unlock => Access::set_or_create(&mut self.unlock, path_id, version),
        }
    }
}
