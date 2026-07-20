#![deny(warnings)]

//! Cells reinterpreted from raw zeroed memory rather than constructed.
//!
//! The headline claim is that a zeroed region *is* a valid array of cells, so
//! these tests obtain their cells the way the code this exists for does — by
//! casting a buffer — never by calling a constructor.

use loom::sync::atomic::materialized::{publish, AtomicU128, AtomicU64};
use loom::sync::atomic::Ordering::{AcqRel, Acquire, Relaxed, Release, SeqCst};
use loom::sync::Arc;
use loom::thread;

/// A zeroed buffer reinterpreted as cells — the pattern under test.
struct Region {
    backing: Vec<u64>,
}

impl Region {
    fn zeroed(cells: usize) -> Region {
        Region {
            backing: vec![0u64; cells],
        }
    }

    fn cells(&self) -> &[AtomicU64] {
        // SAFETY: `materialized::AtomicU64` is `repr(transparent)` over a
        // single identity word and is asserted to match `u64`'s size and
        // alignment, and the all-zeroes pattern is a valid unregistered cell.
        // The buffer is zeroed and `u64`-aligned, so every element is one.
        unsafe {
            std::slice::from_raw_parts(
                self.backing.as_ptr() as *const AtomicU64,
                self.backing.len(),
            )
        }
    }
}

#[test]
fn zeroed_memory_is_a_usable_cell_array() {
    loom::model(|| {
        let region = Arc::new(Region::zeroed(2));

        // Never written, only reinterpreted: a zeroed cell reads as zero.
        assert_eq!(region.cells()[0].load(Relaxed), 0);
        assert_eq!(region.cells()[1].load(Relaxed), 0);

        let w = region.clone();
        let t = thread::spawn(move || {
            w.cells()[0].store(1, Relaxed);
        });

        region.cells()[1].store(2, Relaxed);
        t.join().unwrap();

        assert_eq!(region.cells()[0].load(Relaxed), 1);
        assert_eq!(region.cells()[1].load(Relaxed), 2);
    });
}

#[test]
fn distinct_slots_are_distinct_cells() {
    loom::model(|| {
        let region = Region::zeroed(4);
        let cells = region.cells();

        for (i, cell) in cells.iter().enumerate() {
            cell.store(i as u64 + 1, Relaxed);
        }

        // Identity is minted per cell; writing one must not disturb another.
        for (i, cell) in cells.iter().enumerate() {
            assert_eq!(cell.load(Relaxed), i as u64 + 1);
        }
    });
}

/// A `static` region survives across executions, so its cells must re-register
/// each one rather than inherit the previous execution's stores.
static SHARED: [AtomicU64; 2] = [AtomicU64::ZEROED, AtomicU64::ZEROED];

#[test]
fn static_region_resets_between_executions() {
    static EXECUTIONS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

    loom::model(|| {
        // Whatever the previous execution left behind must be invisible.
        assert_eq!(SHARED[0].load(Relaxed), 0);
        assert_eq!(SHARED[1].load(Relaxed), 0);

        // Conflicting accesses to *one* cell, so DPOR explores both orders —
        // a reset is only meaningful across more than one execution. (Writes
        // to two different slots would be independent and prune to one.)
        let t = thread::spawn(|| {
            SHARED[0].store(7, Relaxed);
        });

        let seen = SHARED[0].load(Relaxed);
        assert!(seen == 0 || seen == 7);

        SHARED[1].fetch_add(9, Relaxed);
        t.join().unwrap();

        EXECUTIONS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    });

    assert!(
        EXECUTIONS.load(std::sync::atomic::Ordering::Relaxed) > 1,
        "test went vacuous: reset is only meaningful across several executions"
    );
}

#[test]
fn release_acquire_message_passing_holds() {
    loom::model(|| {
        let region = Arc::new(Region::zeroed(2));
        let w = region.clone();

        let t = thread::spawn(move || {
            w.cells()[0].store(42, Relaxed);
            w.cells()[1].store(1, Release);
        });

        if region.cells()[1].load(Acquire) == 1 {
            assert_eq!(region.cells()[0].load(Relaxed), 42);
        }

        t.join().unwrap();
    });
}

/// The guard against every other test in this file passing vacuously: if the
/// cells were inert, or all shared one registration, the stale read below could
/// not be produced and this would fail to panic.
#[test]
#[should_panic]
fn relaxed_message_passing_exposes_the_stale_read() {
    loom::model(|| {
        let region = Arc::new(Region::zeroed(2));
        let w = region.clone();

        let t = thread::spawn(move || {
            w.cells()[0].store(42, Relaxed);
            w.cells()[1].store(1, Relaxed);
        });

        if region.cells()[1].load(Relaxed) == 1 {
            assert_eq!(region.cells()[0].load(Relaxed), 42);
        }

        t.join().unwrap();
    });
}

#[test]
fn rmw_atomicity_holds() {
    loom::model(|| {
        let region = Arc::new(Region::zeroed(1));
        let w = region.clone();

        let t = thread::spawn(move || {
            w.cells()[0].fetch_add(1, AcqRel);
        });

        region.cells()[0].fetch_add(1, AcqRel);
        t.join().unwrap();

        // No interleaving may lose an increment.
        assert_eq!(region.cells()[0].load(SeqCst), 2);
    });
}

static PUBLISHED_RACY: [AtomicU64; 1] = [AtomicU64::ZEROED];
static PUBLISHED_CLEAN: [AtomicU64; 1] = [AtomicU64::ZEROED];

fn region_bytes(region: &[AtomicU64]) -> (*const u8, usize) {
    (
        region.as_ptr() as *const u8,
        std::mem::size_of_val(region),
    )
}

/// The point of declaring publication: a thread that reads the memory without
/// synchronizing-with whoever handed it out is reported, exactly as it would be
/// for a constructed cell.
#[test]
#[should_panic(expected = "Concurrent load and mut accesses")]
fn access_unsynchronized_with_the_publisher_is_reported() {
    loom::model(|| {
        let publisher = thread::spawn(|| {
            let (ptr, len) = region_bytes(&PUBLISHED_RACY);
            publish(ptr, len);
            PUBLISHED_RACY[0].store(1, Relaxed);
        });

        // Never synchronized with `publisher` — it is a sibling, so spawning
        // it gave this thread none of its causality.
        let reader = thread::spawn(|| {
            PUBLISHED_RACY[0].load(Relaxed);
        });

        publisher.join().unwrap();
        reader.join().unwrap();
    });
}

/// ...and the converse, or the check would be useless: a reader that *has*
/// synchronized with the publisher is clean in every execution.
#[test]
fn access_synchronized_with_the_publisher_is_clean() {
    loom::model(|| {
        let publisher = thread::spawn(|| {
            let (ptr, len) = region_bytes(&PUBLISHED_CLEAN);
            publish(ptr, len);
            PUBLISHED_CLEAN[0].store(1, Relaxed);
        });

        // Joining synchronizes with the publisher.
        publisher.join().unwrap();

        assert_eq!(PUBLISHED_CLEAN[0].load(Relaxed), 1);
    });
}

#[test]
fn wide_cells_materialize_and_partition_by_lane() {
    loom::model(|| {
        let mut backing = vec![0u128; 1];
        // SAFETY: as `Region::cells`, at 128-bit width — `AtomicU128` is
        // asserted to match `u128`'s size and alignment, and the buffer is
        // zeroed and `u128`-aligned.
        let cell = unsafe { &*(backing.as_mut_ptr() as *const AtomicU128) };

        assert_eq!(cell.load(Relaxed), 0);

        let low: u128 = u64::MAX as u128;
        let high: u128 = !low;

        cell.store_masked(low, 0xabcd, Relaxed);
        cell.store_masked(high, 0x1234 << 64, Relaxed);

        assert_eq!(cell.load_masked(low, Relaxed), 0xabcd);
        assert_eq!(cell.load_masked(high, Relaxed), 0x1234 << 64);
    });
}
