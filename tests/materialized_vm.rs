#![deny(warnings)]

//! The page verbs over materialized cells: commit (`publish`), decommit
//! (`unpublish`) and `MEM_RESET` (`reset`), checked against the contract each
//! models rather than against a convenient worst case.

use loom::cell::UnsafeCell;
use loom::sync::atomic::materialized::{publish, reset, unpublish, AtomicU64};
use loom::sync::atomic::Ordering::{Acquire, Relaxed, Release};
use loom::sync::Arc;
use loom::thread;

use std::collections::BTreeSet;
use std::sync::Mutex;

/// Page-aligned, so tests can place cells on one page or on two.
#[repr(C, align(4096))]
struct Pages([u64; 1024]);

/// Zeroed pages reinterpreted as cells. Leaked rather than dropped: a cell's
/// identity is its address, and freeing the allocation between executions would
/// let a later execution's allocation alias it.
struct Region {
    base: *mut u64,
    pages: usize,
}

unsafe impl Send for Region {}
unsafe impl Sync for Region {}

const PAGE_CELLS: usize = 4096 / 8;

impl Region {
    fn new(pages: usize) -> Region {
        let v: Vec<Pages> = (0..pages).map(|_| Pages([0; 1024])).collect();
        let base = Box::leak(v.into_boxed_slice()).as_mut_ptr() as *mut u64;
        Region { base, pages }
    }

    fn committed(pages: usize) -> Region {
        let r = Region::new(pages);
        r.commit();
        r
    }

    fn bytes(&self) -> usize {
        self.pages * 4096
    }

    fn commit(&self) {
        publish(self.base as *const u8, self.bytes());
    }

    fn decommit(&self) {
        unpublish(self.base as *const u8, self.bytes());
    }

    fn reset(&self) {
        reset(self.base as *mut u8, self.bytes());
    }

    fn cell(&self, i: usize) -> &AtomicU64 {
        assert!(i < self.pages * PAGE_CELLS);
        // SAFETY: zeroed, `u64`-aligned, inside the leaked allocation, and a
        // materialized cell is layout-identical to `u64`.
        unsafe { &*(self.base.add(i) as *const AtomicU64) }
    }
}

/// Collects one outcome per execution, and the whole set once the model is
/// done — the oracle for "this behaviour is reachable" and "that one is not".
struct Outcomes<T: Ord>(Mutex<BTreeSet<T>>);

impl<T: Ord + Clone> Outcomes<T> {
    const fn new() -> Self {
        Outcomes(Mutex::new(BTreeSet::new()))
    }

    fn record(&self, t: T) {
        self.0.lock().unwrap().insert(t);
    }

    fn take(&self) -> BTreeSet<T> {
        std::mem::take(&mut *self.0.lock().unwrap())
    }
}

// ===== decommit =====

/// A reader with no happens-before to the decommit races it, whichever of the
/// two runs first. The reader touches a *different* cell than the decommitter
/// did, so nothing but the decommit itself relates them.
#[test]
#[should_panic(expected = "decommit")]
fn unsynchronized_reader_vs_decommit_is_reported() {
    loom::model(|| {
        let r = Arc::new(Region::committed(1));
        r.cell(0).store(0, Relaxed);
        let r1 = r.clone();
        let reader = thread::spawn(move || {
            r1.cell(1).load(Relaxed);
        });
        let r2 = r.clone();
        let decommitter = thread::spawn(move || {
            r2.cell(0).store(7, Relaxed);
            r2.decommit();
        });
        reader.join().unwrap();
        decommitter.join().unwrap();
    });
}

/// An access after the decommit names what it is.
#[test]
#[should_panic(expected = "use after decommit")]
fn access_after_decommit_is_named() {
    loom::model(|| {
        let r = Region::committed(1);
        r.cell(0).store(1, Relaxed);
        r.decommit();
        r.cell(0).load(Relaxed);
    });
}

/// A decommit that happens-after every access, followed by a recommit, hands
/// back fresh zeroed cells: nothing is reported and nothing leaks across.
#[test]
fn synchronized_decommit_and_recommit_is_clean() {
    loom::model(|| {
        let r = Arc::new(Region::committed(1));
        let r1 = r.clone();
        let t = thread::spawn(move || {
            r1.cell(0).store(5, Release);
        });
        t.join().unwrap();
        r.decommit();
        r.commit();
        assert_eq!(r.cell(0).load(Relaxed), 0);
    });
}

/// A thread re-commits a range, uses it, and does not order either against a
/// peer's decommit of the same range: in the order commit, decommit, use the
/// use faults. The two commits are platform-serialized, so the search must
/// explore both orders of the page verbs.
#[test]
#[should_panic(expected = "decommit")]
fn recommit_racing_decommit_is_reported() {
    loom::model(|| {
        let r = Arc::new(Region::committed(1));
        let r1 = r.clone();
        let user = thread::spawn(move || {
            r1.commit();
            r1.cell(3).store(1, Relaxed);
        });
        let r2 = r.clone();
        let reclaimer = thread::spawn(move || {
            r2.decommit();
        });
        user.join().unwrap();
        reclaimer.join().unwrap();
    });
}

// ===== commit =====

/// An idempotent commit of a live range synchronizes nothing with the first
/// committer: data the first committer wrote before committing is raced by a
/// read the second makes after its own commit.
#[test]
#[should_panic(expected = "Causality violation")]
fn recommit_does_not_synchronize_with_the_first_committer() {
    struct Shared {
        region: Region,
        data: UnsafeCell<u64>,
    }
    unsafe impl Sync for Shared {}
    unsafe impl Send for Shared {}

    loom::model(|| {
        let s = Arc::new(Shared {
            region: Region::new(1),
            data: UnsafeCell::new(0),
        });
        let a = s.clone();
        let first = thread::spawn(move || {
            a.data.with_mut(|p| unsafe { *p = 1 });
            a.region.commit();
        });
        let b = s.clone();
        let second = thread::spawn(move || {
            b.region.commit();
            b.data.with(|p| unsafe { *p });
        });
        first.join().unwrap();
        second.join().unwrap();
    });
}

/// ...but the second committer may use the memory it committed: the zero
/// contents a commit guarantees are established for every committer, whoever
/// committed first.
#[test]
fn recommitter_may_use_the_memory() {
    loom::model(|| {
        let r = Arc::new(Region::new(1));
        let a = r.clone();
        let first = thread::spawn(move || {
            a.commit();
            a.cell(0).store(1, Relaxed);
        });
        let b = r.clone();
        let second = thread::spawn(move || {
            b.commit();
            let v = b.cell(0).load(Relaxed);
            assert!(v == 0 || v == 1);
        });
        first.join().unwrap();
        second.join().unwrap();
    });
}

/// A thread reaching published memory without synchronizing-with any committer
/// is still reported, with a message that says so.
#[test]
#[should_panic(expected = "without synchronizing-with its commit")]
fn access_unsynchronized_with_every_committer_is_reported() {
    loom::model(|| {
        let r = Arc::new(Region::new(1));
        let a = r.clone();
        let flag = Arc::new(loom::sync::atomic::AtomicUsize::new(0));
        let f = flag.clone();
        let committer = thread::spawn(move || {
            a.commit();
            f.store(1, Relaxed);
        });
        if flag.load(Relaxed) == 1 {
            r.cell(0).load(Relaxed);
        }
        committer.join().unwrap();
    });
}

// ===== reset =====

/// A consumer that assumes a reset page reads zero is wrong: `MEM_RESET` does
/// not guarantee zeros, and a page the kernel has not yet discarded still reads
/// its old contents — even to the thread that reset it.
#[test]
#[should_panic(expected = "reset page read back its retained value")]
fn assuming_zero_after_reset_is_reported() {
    loom::model(|| {
        let r = Region::committed(1);
        r.cell(0).store(5, Relaxed);
        r.reset();
        let v = r.cell(0).load(Relaxed);
        assert!(v == 0, "reset page read back its retained value {v}");
    });
}

/// Per page, a reset is either discarded at some point or never: reads before
/// any write see the old contents, then zero from the discard on, and never
/// the old contents again.
#[test]
fn reset_reads_are_old_then_zero_never_back() {
    static SEEN: Outcomes<(u64, u64)> = Outcomes::new();
    loom::model(|| {
        let r = Region::committed(1);
        r.cell(0).store(5, Relaxed);
        r.reset();
        let a = r.cell(0).load(Relaxed);
        let b = r.cell(0).load(Relaxed);
        SEEN.record((a, b));
    });
    let seen = SEEN.take();
    assert_eq!(
        seen,
        BTreeSet::from([(0, 0), (5, 0), (5, 5)]),
        "a reset page reads old until discarded, then zero for good"
    );
}

/// The discard decision is per page: once a thread has seen one cell of a page
/// discarded, every other cell of that page reads zero to it.
#[test]
fn reset_discards_a_whole_page_at_once() {
    static SEEN: Outcomes<(u64, u64)> = Outcomes::new();
    loom::model(|| {
        let r = Region::committed(2);
        r.cell(0).store(5, Relaxed);
        r.cell(1).store(6, Relaxed);
        r.reset();
        let a = r.cell(0).load(Relaxed);
        let b = r.cell(1).load(Relaxed);
        SEEN.record((a, b));
    });
    let seen = SEEN.take();
    assert!(!seen.contains(&(0, 6)), "one page discarded half: {seen:?}");
    assert!(seen.contains(&(5, 0)), "discard between the reads unexplored: {seen:?}");
    assert!(seen.contains(&(5, 6)) && seen.contains(&(0, 0)), "{seen:?}");
}

/// Pages decide independently.
#[test]
fn reset_pages_decide_independently() {
    static SEEN: Outcomes<(u64, u64)> = Outcomes::new();
    loom::model(|| {
        let r = Region::committed(2);
        r.cell(0).store(5, Relaxed);
        r.cell(PAGE_CELLS).store(6, Relaxed);
        r.reset();
        let a = r.cell(0).load(Relaxed);
        let b = r.cell(PAGE_CELLS).load(Relaxed);
        SEEN.record((a, b));
    });
    let seen = SEEN.take();
    assert!(seen.contains(&(0, 6)), "pages decided together: {seen:?}");
}

/// A write after the reset dirties the page and cancels the discard: the cell
/// keeps what was written, and a cell of the same page read after the write
/// is stable from then on.
#[test]
fn write_after_reset_cancels_the_discard() {
    static SEEN: Outcomes<(u64, u64, u64)> = Outcomes::new();
    loom::model(|| {
        let r = Region::committed(1);
        r.cell(0).store(5, Relaxed);
        r.cell(1).store(6, Relaxed);
        r.reset();
        r.cell(0).store(7, Relaxed);
        let a = r.cell(0).load(Relaxed);
        let b1 = r.cell(1).load(Relaxed);
        let b2 = r.cell(1).load(Relaxed);
        SEEN.record((a, b1, b2));
    });
    let seen = SEEN.take();
    assert_eq!(
        seen,
        BTreeSet::from([(7, 0, 0), (7, 6, 6)]),
        "the written page is settled: either discarded before the write or kept"
    );
}

/// A write after a peer's discard lands on the zero page: it is ordered after
/// the discard even though the writer never saw it, so the discard can never
/// resurface over it.
#[test]
fn write_after_a_peers_discard_is_ordered_after_it() {
    loom::model(|| {
        let r = Arc::new(Region::committed(1));
        r.cell(0).store(5, Relaxed);
        r.reset();
        let r1 = r.clone();
        let reader = thread::spawn(move || {
            r1.cell(0).load(Relaxed);
        });
        let r2 = r.clone();
        let writer = thread::spawn(move || {
            r2.cell(0).store(9, Relaxed);
        });
        reader.join().unwrap();
        writer.join().unwrap();
        assert_eq!(r.cell(0).load(Relaxed), 9, "a discard overtook a later write");
    });
}

/// A reader racing the reset is admissible and sees old or zero; a reader
/// acquiring a write made after the reset is not owed zero by it.
#[test]
fn reset_racing_an_atomic_reader_is_admissible() {
    static SEEN: Outcomes<u64> = Outcomes::new();
    loom::model(|| {
        let r = Arc::new(Region::committed(1));
        r.cell(0).store(5, Relaxed);
        let r1 = r.clone();
        let t = thread::spawn(move || {
            SEEN.record(r1.cell(0).load(Acquire));
        });
        r.reset();
        t.join().unwrap();
    });
    assert_eq!(SEEN.take(), BTreeSet::from([0, 5]));
}

/// A reset of decommitted memory is a use after decommit.
#[test]
#[should_panic(expected = "use after decommit")]
fn reset_after_decommit_is_named() {
    loom::model(|| {
        let r = Region::committed(1);
        r.cell(0).store(1, Relaxed);
        r.decommit();
        r.reset();
    });
}

// ===== leak check =====

/// Opt-in: a range still committed when the execution ends is reported.
#[test]
#[should_panic(expected = "still committed")]
fn committed_range_at_exit_is_reported_when_asked() {
    let mut b = loom::model::Builder::new();
    b.check_committed_leaks = true;
    b.check(|| {
        let r = Region::committed(1);
        r.cell(0).store(1, Relaxed);
    });
}

/// ...and clean once the range is decommitted.
#[test]
fn decommitted_range_at_exit_is_clean() {
    let mut b = loom::model::Builder::new();
    b.check_committed_leaks = true;
    b.check(|| {
        let r = Region::committed(1);
        r.cell(0).store(1, Relaxed);
        r.decommit();
    });
}
