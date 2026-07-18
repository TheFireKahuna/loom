use crate::rt::{thread, MAX_THREADS};

#[cfg(feature = "checkpoint")]
use serde::{Deserialize, Serialize};
use std::cmp;
use std::ops;

/// Version lanes: `MAX_THREADS` rounded up to a whole 128-bit SIMD register
/// (8 × u16), so `join` / `partial_cmp` / `ahead` compile to single vector
/// ops instead of five-lane scalar loops. The padding lanes
/// `MAX_THREADS..LANES` are structurally zero: every write goes through a
/// `thread::Id` index (`< MAX_THREADS`) or `join` (lane-max of two zeros),
/// so they are inert in every comparison and invisible to `versions()`.
const LANES: usize = (MAX_THREADS + 7) & !7;

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
#[cfg_attr(feature = "checkpoint", derive(Serialize, Deserialize))]
#[repr(align(16))]
pub(crate) struct VersionVec {
    versions: [u16; LANES],
}

impl VersionVec {
    /// Number of physical lanes (`MAX_THREADS` rounded up to a SIMD register).
    /// Padding lanes `MAX_THREADS..LANES` are structurally zero — see the
    /// module-level invariant.
    pub(crate) const LANES: usize = LANES;

    pub(crate) fn new() -> VersionVec {
        VersionVec {
            versions: [0; LANES],
        }
    }

    /// The raw lane array, including the always-zero padding lanes. Exposed so
    /// callers holding a parallel `[u16; LANES]` (e.g. `FirstSeen`) can do a
    /// single branchless all-lane comparison instead of a per-thread scalar
    /// scan.
    pub(crate) fn lanes(&self) -> &[u16; LANES] {
        &self.versions
    }

    pub(crate) fn inc(&mut self, id: thread::Id) {
        self.versions[id.as_usize()] += 1;
    }

    /// Read a single lane by raw index.
    ///
    /// Vector clocks are transitively closed, so for any operation `op`
    /// stamped `(thread, tick)`, `v.lane(thread) >= tick` holds iff `v`
    /// contains `op`'s *entire* causal snapshot — a one-lane read decides
    /// full snapshot containment. `rt::atomic` uses this as its
    /// modification-order marker test.
    pub(crate) fn lane(&self, lane: usize) -> u16 {
        self.versions[lane]
    }

    pub(crate) fn join(&mut self, other: &VersionVec) {
        for i in 0..LANES {
            self.versions[i] = cmp::max(self.versions[i], other.versions[i]);
        }
    }

    /// True when `self <= other` in every lane.
    ///
    /// The one-sided form of `partial_cmp`, which computes the `ge` reduction
    /// as well and discards it. This is the happens-before test on the DPOR
    /// scan (`Access::happens_before`), the hottest comparison in the runtime,
    /// where the `ge` half is never consulted.
    pub(crate) fn is_le(&self, other: &VersionVec) -> bool {
        let mut le = true;
        for i in 0..LANES {
            le &= self.versions[i] <= other.versions[i];
        }
        le
    }

    /// Returns the thread ID, if any, that is ahead of the current version.
    pub(crate) fn ahead(&self, other: &VersionVec) -> Option<usize> {
        // Branchless lane compare + first-set-bit, rather than an early-out
        // loop, so the whole check is one vector op. Padding lanes are 0 on
        // both sides and can never set a bit.
        let mut mask = 0u32;
        for i in 0..LANES {
            mask |= ((self.versions[i] < other.versions[i]) as u32) << i;
        }

        if mask == 0 {
            None
        } else {
            Some(mask.trailing_zeros() as usize)
        }
    }
}

impl cmp::PartialOrd for VersionVec {
    fn partial_cmp(&self, other: &VersionVec) -> Option<cmp::Ordering> {
        use cmp::Ordering::*;

        // Two vectorizable all-lane reductions instead of a stateful scalar
        // scan: `self <= other` in every lane, and `self >= other` in every
        // lane, decide the partial order exactly as the lane loop did.
        let mut le = true;
        let mut ge = true;

        for i in 0..LANES {
            le &= self.versions[i] <= other.versions[i];
            ge &= self.versions[i] >= other.versions[i];
        }

        match (le, ge) {
            (true, true) => Some(Equal),
            (true, false) => Some(Less),
            (false, true) => Some(Greater),
            (false, false) => None,
        }
    }
}

impl ops::Index<thread::Id> for VersionVec {
    type Output = u16;

    fn index(&self, index: thread::Id) -> &u16 {
        self.versions.index(index.as_usize())
    }
}

impl ops::IndexMut<thread::Id> for VersionVec {
    fn index_mut(&mut self, index: thread::Id) -> &mut u16 {
        self.versions.index_mut(index.as_usize())
    }
}
