#![deny(warnings)]

//! Cells reinterpreted from raw zeroed memory rather than constructed.
//!
//! The headline claim is that a zeroed region *is* a valid array of cells, so
//! these tests obtain their cells the way the code this exists for does — by
//! casting a buffer — never by calling a constructor.

use loom::sync::atomic::materialized::{
    publish, reset, zero_exclusive, AtomicU128, AtomicU16, AtomicU32, AtomicU64, AtomicU8,
};
use loom::sync::atomic::Ordering::{AcqRel, Acquire, Relaxed, Release, SeqCst};
use loom::sync::Arc;
use loom::thread;

/// A zeroed buffer reinterpreted as cells — the pattern under test.
struct Region {
    backing: Vec<u64>,
}

impl Region {
    /// Allocate and declare. A materialized cell's identity is its address, so
    /// the declaration is what gives the cells inside identities at all.
    fn zeroed(cells: usize) -> Region {
        let backing = vec![0u64; cells];
        publish(
            backing.as_ptr() as *const u8,
            std::mem::size_of_val(&backing[..]),
        );
        Region { backing }
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
/// each one rather than inherit the previous execution's stores. Its address is
/// stable, so this is the case where address keying is most obviously exposed
/// to leaking state between executions — the per-iteration table clear is what
/// prevents it.
static SHARED: [AtomicU64; 2] = [AtomicU64::ZEROED, AtomicU64::ZEROED];

fn publish_static() {
    publish(
        SHARED.as_ptr() as *const u8,
        std::mem::size_of_val(&SHARED),
    );
}

#[test]
fn static_region_resets_between_executions() {
    static EXECUTIONS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

    loom::model(|| {
        publish_static();

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
        // Declare only the *tail* up front, so no schedule can reach a cell
        // before some declaration exists — otherwise the reader races the "not
        // in any published region" panic and this test would not be measuring
        // what it claims to. Cell 0 is deliberately left undeclared here: it is
        // the one the publisher below brings into existence.
        let (ptr, len) = region_bytes(&PUBLISHED_RACY);
        let cell = std::mem::size_of_val(&PUBLISHED_RACY[0]);
        publish(unsafe { ptr.add(cell) }, len - cell);

        // Publishing genuinely new memory, so cell 0's genesis is *this*
        // thread's causality — and this thread is only a sibling of the reader.
        let publisher = thread::spawn(move || {
            publish(ptr, cell);
            PUBLISHED_RACY[0].store(1, Relaxed);
        });

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

/// The width range address keying exists to reach. A minted identity needs bits
/// these cells do not have — a `u16` cell has 65535 ids for a whole process,
/// and a materialized cell re-registers every execution.
#[test]
fn narrow_cells_materialize_and_stay_distinct() {
    loom::model(|| {
        // `u64`-backed so the sub-word casts below are all naturally aligned.
        let backing = vec![0u64; 2];
        let base = backing.as_ptr() as *const u8;
        publish(base, std::mem::size_of_val(&backing[..]));

        // SAFETY: each cell is asserted to match its integer's size and
        // alignment, the buffer is zeroed, and each offset below is a multiple
        // of the width read there. The three cells occupy disjoint bytes.
        let (byte, word, dword) = unsafe {
            (
                &*(base as *const AtomicU8),
                &*(base.add(2) as *const AtomicU16),
                &*(base.add(4) as *const AtomicU32),
            )
        };

        assert_eq!(byte.load(Relaxed), 0);
        assert_eq!(word.load(Relaxed), 0);
        assert_eq!(dword.load(Relaxed), 0);

        byte.store(0xab, Relaxed);
        word.store(0xbeef, Relaxed);
        dword.store(0xdead_beef, Relaxed);

        // Distinct addresses are distinct cells — no aliasing across widths.
        assert_eq!(byte.load(Relaxed), 0xab);
        assert_eq!(word.load(Relaxed), 0xbeef);
        assert_eq!(dword.load(Relaxed), 0xdead_beef);
    });
}

/// The declaration is required, not advisory — the guard against a consumer
/// silently getting the weaker, publication-blind genesis.
#[test]
#[should_panic(expected = "not in any published region")]
fn undeclared_memory_is_rejected() {
    loom::model(|| {
        let backing = vec![0u64; 1];
        // SAFETY: layout is right; the *declaration* is what is missing, which
        // is the point of the test.
        let cell = unsafe { &*(backing.as_ptr() as *const AtomicU64) };
        cell.load(Relaxed);
    });
}

/// The shape a consumer actually builds: a record laid out over raw memory,
/// whose zero-validity is *proved* by the derive rather than asserted in a
/// comment. The `zerocopy` feature exists so this derive can see through to the
/// cells — the orphan rule puts the impl out of the consumer's reach.
#[cfg(feature = "zerocopy")]
#[derive(zerocopy::FromZeros)]
#[repr(C, align(64))]
struct Record {
    head: AtomicU64,
    generation: AtomicU32,
    flags: AtomicU16,
    tag: AtomicU8,
    _pad: [u8; 49],
}

#[cfg(feature = "zerocopy")]
#[test]
fn a_record_of_materialized_cells_composes_and_is_zero_valid() {
    // The record is the production shape, not a checker-only stand-in — which
    // is the entire point of the module.
    assert_eq!(std::mem::size_of::<Record>(), 64);
    assert_eq!(std::mem::align_of::<Record>(), 64);
    assert_eq!(std::mem::offset_of!(Record, head), 0);
    assert_eq!(std::mem::offset_of!(Record, generation), 8);
    assert_eq!(std::mem::offset_of!(Record, flags), 12);
    assert_eq!(std::mem::offset_of!(Record, tag), 14);

    // A `Vec` would only be element-aligned; the records need 64, so the
    // backing carries the alignment in its own type.
    #[repr(C, align(64))]
    struct Backing([u8; 128]);

    loom::model(|| {
        // Two records carved out of one zeroed, declared region.
        let backing = Backing([0; 128]);
        let base = &backing as *const Backing as *const u8;
        publish(base, std::mem::size_of::<Backing>());

        // SAFETY: `Record: FromZeros` (derived above) says the all-zero pattern
        // is a valid `Record`; the backing is zeroed and its type carries
        // 64-byte alignment, so both records are aligned. This is the
        // reinterpretation the module licenses.
        let records = unsafe { std::slice::from_raw_parts(base as *const Record, 2) };

        for r in records {
            assert_eq!(r.head.load(Relaxed), 0);
            assert_eq!(r.generation.load(Relaxed), 0);
            assert_eq!(r.flags.load(Relaxed), 0);
            assert_eq!(r.tag.load(Relaxed), 0);
        }

        records[0].head.store(0xdead_beef, Relaxed);
        records[1].generation.store(7, Relaxed);

        // Fields of distinct records are distinct cells.
        assert_eq!(records[0].head.load(Relaxed), 0xdead_beef);
        assert_eq!(records[1].head.load(Relaxed), 0);
        assert_eq!(records[0].generation.load(Relaxed), 0);
        assert_eq!(records[1].generation.load(Relaxed), 7);
    });
}

/// A recycled record is zeroed as raw memory, before any typed reference to it
/// exists. The cells' values live in the model, not in those bytes, so the
/// zeroing has to be declared — otherwise it silently does not happen and every
/// cell keeps its previous lifecycle's value.
#[test]
fn bulk_zero_resets_the_cells_in_its_range() {
    loom::model(|| {
        let region = Region::zeroed(4);
        let cells = region.cells();

        for (i, cell) in cells.iter().enumerate() {
            cell.store(i as u64 + 1, Relaxed);
        }

        // Zero only the first two cells, the way a pool recycles one record out
        // of several.
        zero_exclusive(
            cells.as_ptr() as *mut u8,
            2 * std::mem::size_of::<u64>(),
        );

        assert_eq!(cells[0].load(Relaxed), 0);
        assert_eq!(cells[1].load(Relaxed), 0);
        // Outside the range, untouched — a bulk zero must not reach past its
        // own record.
        assert_eq!(cells[2].load(Relaxed), 3);
        assert_eq!(cells[3].load(Relaxed), 4);
    });
}

/// The exclusivity is a real precondition, not a comment: a peer reading the
/// range while it is being zeroed is the race the check exists to catch.
#[test]
#[should_panic(expected = "Causality violation")]
fn bulk_zero_racing_a_reader_is_reported() {
    loom::model(|| {
        let region = Arc::new(Region::zeroed(2));
        let peer = region.clone();

        let t = thread::spawn(move || {
            peer.cells()[0].load(Relaxed);
        });

        // A bulk zero takes no DPOR branch — it is an exclusive, non-atomic
        // write, like `with_mut` — so something before it has to open the
        // scheduling point, and that something must *conflict* with the
        // reader. A store to a different cell would be independent of the
        // reader's load and the interleaving would never be explored at all.
        region.cells()[0].store(1, Relaxed);

        zero_exclusive(
            region.cells().as_ptr() as *mut u8,
            std::mem::size_of::<u64>(),
        );

        t.join().unwrap();
    });
}

/// The distinguishing property, and the reason `reset` is not `zero_exclusive`
/// with a different name: this is the *identical* rig to
/// `bulk_zero_racing_a_reader_is_reported`, and it must come out clean. A stale
/// walker reading a span through its own atomics while the reset lands is
/// admissible — T3 depends on it.
#[test]
fn reset_racing_a_reader_is_admissible() {
    loom::model(|| {
        let region = Arc::new(Region::zeroed(2));
        let peer = region.clone();

        let t = thread::spawn(move || {
            // Either value is legal: the reader may cross the reset or not.
            let seen = peer.cells()[0].load(Acquire);
            assert!(seen == 0 || seen == 1);
        });

        region.cells()[0].store(1, Release);
        reset(
            region.cells().as_ptr() as *mut u8,
            std::mem::size_of::<u64>(),
        );

        t.join().unwrap();
    });
}

#[test]
fn reset_discards_only_its_own_range() {
    loom::model(|| {
        let region = Region::zeroed(4);
        let cells = region.cells();

        for (i, cell) in cells.iter().enumerate() {
            cell.store(i as u64 + 1, Relaxed);
        }

        reset(cells.as_ptr() as *mut u8, 2 * std::mem::size_of::<u64>());

        assert_eq!(cells[0].load(Relaxed), 0);
        assert_eq!(cells[1].load(Relaxed), 0);
        assert_eq!(cells[2].load(Relaxed), 3);
        assert_eq!(cells[3].load(Relaxed), 4);
    });
}

/// A pointer cell materializes too — the shape `Slot.wait_address` needs.
#[test]
fn pointer_cells_materialize() {
    use loom::sync::atomic::materialized::AtomicPtr;

    loom::model(|| {
        let backing = vec![0u64; 2];
        let base = backing.as_ptr() as *const u8;
        publish(base, std::mem::size_of_val(&backing[..]));

        // SAFETY: `AtomicPtr` is asserted to match `*mut _`'s size and
        // alignment, the buffer is zeroed and pointer-aligned.
        let cell = unsafe { &*(base as *const AtomicPtr<u32>) };

        assert!(cell.load(Relaxed).is_null());

        let mut target = 5u32;
        cell.store(&raw mut target, Release);
        assert!(!cell.load(Acquire).is_null());
    });
}

#[test]
fn wide_cells_materialize_and_partition_by_lane() {
    loom::model(|| {
        let mut backing = vec![0u128; 1];
        publish(
            backing.as_ptr() as *const u8,
            std::mem::size_of_val(&backing[..]),
        );
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

        // The typed lane views work on a materialized cell too — the shape
        // `EpisodeMeta`'s hot path needs, where nearly every access is a lane
        // rather than the whole 16 bytes.
        assert_eq!(cell.lane_u64(0).load(Relaxed), 0xabcd);
        assert_eq!(cell.lane_u64(8).load(Relaxed), 0x1234);
        assert_eq!(cell.lane_u32(0).load(Relaxed), 0xabcd);

        cell.lane_u32(4).store(0x99, Relaxed);
        assert_eq!(cell.lane_u32(4).load(Relaxed), 0x99);
        // A lane store leaves its siblings alone.
        assert_eq!(cell.lane_u32(0).load(Relaxed), 0xabcd);
        assert_eq!(cell.lane_u64(8).load(Relaxed), 0x1234);
    });
}
