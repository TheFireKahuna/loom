use crate::rt::{thread, VersionVec};

use std::sync::atomic::Ordering::{self, *};

/// A synchronization point between two threads.
///
/// Threads synchronize with this point using any of the available orderings. On
/// loads, the thread's causality is updated using the synchronization point's
/// stored causality. On stores, the synchronization point's causality is
/// updated with the threads.
///
/// The point carries two views: `happens_before`, which is happens-before
/// proper, and `sc`, what `SeqCst` fences contribute to the events it reaches.
/// Both travel along every synchronizes-with edge, but only `happens_before`
/// ever becomes a thread's causality.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Synchronize {
    happens_before: VersionVec,
    sc: ScView,
}

/// The `SeqCst`-fence constraints on the events a view reaches, kept apart
/// from happens-before: an SC fence orders S and restricts coherence (C++20
/// [atomics.order] p4.3/p4.4) but creates no happens-before, so none of this
/// may reach a data-race check.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ScView {
    /// Join of the causality of every SC fence that precedes, in S, an SC fence
    /// happening before this point. Coherence treats these events as though
    /// they happened before this point (p4.4): nothing here may be read over
    /// or written under.
    pub(crate) frontier: VersionVec,

    /// S-position of the latest SC fence happening before this point: the scope
    /// within which SC-ranked writes restrict it (p4.3).
    pub(crate) fence_pos: Option<u32>,
}

impl ScView {
    pub(crate) fn new() -> ScView {
        ScView {
            frontier: VersionVec::new(),
            fence_pos: None,
        }
    }

    pub(crate) fn join(&mut self, other: &ScView) {
        self.frontier.join(&other.frontier);
        self.fence_pos = self.fence_pos.max(other.fence_pos);
    }
}

impl Synchronize {
    pub fn new() -> Self {
        Synchronize {
            happens_before: VersionVec::new(),
            sc: ScView::new(),
        }
    }

    /// A synchronization point holding exactly `happens_before` and `sc`.
    pub(crate) fn from_views(happens_before: VersionVec, sc: ScView) -> Self {
        Synchronize { happens_before, sc }
    }

    pub(crate) fn happens_before(&self) -> &VersionVec {
        &self.happens_before
    }

    pub(crate) fn sc(&self) -> &ScView {
        &self.sc
    }

    pub(crate) fn join(&mut self, other: &Synchronize) {
        self.happens_before.join(&other.happens_before);
        self.sc.join(&other.sc);
    }

    /// The release view this synchronization point publishes — what an
    /// acquiring load joins into the reader's causality (`sync_acq`). Exposed
    /// so a multi-region load can *project* the causality its own earlier
    /// regions will establish, without performing the reads.
    pub fn released_view(&self) -> &VersionVec {
        &self.happens_before
    }

    pub fn sync_load(&mut self, threads: &mut thread::Set, order: Ordering) {
        match order {
            Relaxed | Release => {
                // A later acquire fence synchronizes with this point through
                // this read ([atomics.fences] p4), whether or not the store is
                // still in its cell's history by then.
                threads.active_mut().acquirable.join(self);
            }
            Acquire | AcqRel => {
                self.sync_acq(threads);
            }
            SeqCst => {
                self.sync_acq(threads);
                threads.seq_cst();
            }
            order => unimplemented!("unimplemented ordering {:?}", order),
        }
    }

    pub fn sync_store(&mut self, threads: &mut thread::Set, order: Ordering) {
        self.join(&threads.active().released);
        match order {
            Relaxed | Acquire => {
                // Nothing happens!
            }
            Release | AcqRel => {
                self.sync_rel(threads);
            }
            SeqCst => {
                self.sync_rel(threads);
                threads.seq_cst();
            }
            order => unimplemented!("unimplemented ordering {:?}", order),
        }
    }

    fn sync_acq(&mut self, threads: &mut thread::Set) {
        threads.active_mut().acquire(self);
    }

    fn sync_rel(&mut self, threads: &thread::Set) {
        self.join(&threads.active().view());
    }
}
