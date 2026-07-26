#![deny(warnings, rust_2018_idioms)]

//! Preserving wide read-modify-writes — a 16-byte CAS that compares every bit
//! but writes some aligned lane back at the value it read.
//!
//! The model has to be told which bits an operation may *change*, separately
//! from which it *consults*, or a `cmpxchg16b` looks like a writer of the
//! whole cell and every lane is coupled to every other. `rmw_preserving` is
//! that declaration, and these tests pin the three things it claims:
//!
//! 1. **It buys independence.** The carried lane's readers commute with the
//!    operation, so the search is strictly smaller than the same rig spelled
//!    as a full-width CAS — while reaching the same behaviors.
//! 2. **It keeps whole-cell coherence.** Seeing the operation through a lane
//!    it wrote still floors the lane it carried: a reader cannot take the
//!    installed value and then a stale owner.
//! 3. **What it gives up, it gives up loudly.** The carried lane has no store
//!    of the operation to read-from, so nothing may acquire through it, and
//!    the attempt traps instead of quietly exploring less.

use loom::sync::atomic::{fence, AtomicU128, Ordering::*};
use loom::thread;

use std::collections::BTreeSet;
use std::sync::{Arc, Mutex};

// The ntlib futex word, by byte offset into the little-endian cell:
// `[ value:32 | owner:32 ][ gen:24 | top:20 | old:20 ]`.
const S_OFF: usize = 0;
const O_OFF: usize = 8;
const V_OFF: usize = 12;

const O_MASK: u128 = (u32::MAX as u128) << 64;

const OWNER: u32 = 0xABCD;
const OWNER_BITS: u128 = (OWNER as u128) << 64;

/// The word after a wake: value lane installed, group S advanced.
const PUBLISHED: u128 = (1u128 << 96) | 1;

#[derive(Clone, Default)]
struct Log(Arc<Mutex<BTreeSet<String>>>);

impl Log {
    fn record(&self, what: impl std::fmt::Display) {
        self.0.lock().unwrap().insert(what.to_string());
    }

    fn take(&self) -> BTreeSet<String> {
        self.0.lock().unwrap().clone()
    }
}

/// Run `model` to exhaustion; return every behavior it exhibited and the
/// number of executions it took.
fn explore<M>(model: M) -> (BTreeSet<String>, usize)
where
    M: Fn(&Log) + Send + Sync + 'static,
{
    let log = Log::default();
    let inner = log.clone();

    let mut b = loom::model::Builder::new();
    // Pin everything the environment could otherwise leak in — the execution
    // count is an assertion here, not a diagnostic.
    b.stats = true;
    b.log = false;
    b.max_permutations = None;
    b.max_duration = None;
    b.preemption_bound = None;

    let stats = b.check(move || model(&inner));
    (log.take(), stats.executions)
}

/// The rig the declaration exists for: a wake installs the value lane and
/// advances group S in one 16-byte CAS, carrying the owner lane through
/// untouched, while another thread reads that owner lane.
///
/// `preserving` picks the spelling. Both do the same thing to the same bits;
/// only the model's picture of them differs.
fn fused_wake(preserving: bool) -> impl Fn(&Log) + Send + Sync + 'static {
    move |log: &Log| {
        let x = Arc::new(AtomicU128::new(OWNER_BITS));

        let w = {
            let x = x.clone();
            thread::spawn(move || {
                let r = if preserving {
                    x.compare_exchange_preserving(O_MASK, OWNER_BITS, PUBLISHED, AcqRel, Relaxed)
                } else {
                    x.compare_exchange(OWNER_BITS, OWNER_BITS | PUBLISHED, AcqRel, Relaxed)
                };
                assert!(r.is_ok(), "uncontended CAS must succeed");
            })
        };

        // A reader of the carried lane, twice — the traffic the declaration
        // decouples the CAS from.
        let r = {
            let x = x.clone();
            let log = log.clone();
            thread::spawn(move || {
                let a = x.lane_u32(O_OFF).load(Relaxed);
                let b = x.lane_u32(O_OFF).load(Relaxed);
                log.record(format_args!("owner {a:#x},{b:#x}"));
            })
        };

        // ...and a reader of a lane the CAS does write.
        let v = x.lane_u32(V_OFF).load(Relaxed);
        let s = x.lane_u64(S_OFF).load(Relaxed);
        log.record(format_args!("value {v} s {s}"));

        w.join().unwrap();
        r.join().unwrap();

        assert_eq!(x.load(Relaxed), OWNER_BITS | PUBLISHED);
    }
}

/// The declaration reaches exactly the same behaviors, in strictly fewer
/// executions. Both halves matter: the reduction is the point, and the
/// behavior equality is what makes it a reduction rather than a hole.
#[test]
fn preserving_reaches_the_same_behaviors_in_fewer_executions() {
    let (full_seen, full_execs) = explore(fused_wake(false));
    let (pres_seen, pres_execs) = explore(fused_wake(true));

    assert_eq!(
        full_seen, pres_seen,
        "declaring the owner lane preserved changed which behaviors are \
         reachable; it must only change how the model accounts for them"
    );
    assert!(
        pres_execs < full_execs,
        "preserving explored {pres_execs} executions against the full-width \
         CAS's {full_execs} — the carried lane's readers are still coupled to \
         the operation"
    );

    println!("full-width CAS: {full_execs} executions; preserving: {pres_execs}");
}

/// Whole-cell coherence survives the elision: a reader that takes the value
/// lane the wide CAS installed may not then take an owner lane older than the
/// one that same CAS carried through. The cell is 16-byte single-copy-atomic,
/// so that pair never existed on the line.
#[test]
fn seeing_a_preserving_op_floors_the_lane_it_carried() {
    let (seen, _) = explore(|log: &Log| {
        let x = Arc::new(AtomicU128::new(0));

        let w = {
            let x = x.clone();
            thread::spawn(move || {
                // Stamp the owner lane, then carry it through a wide CAS that
                // installs the value lane.
                x.lane_u32(O_OFF).store(OWNER, Relaxed);
                x.compare_exchange_preserving(O_MASK, OWNER_BITS, 1 << 96, Relaxed, Relaxed)
                    .expect("uncontended CAS must succeed");
            })
        };

        let v = x.lane_u32(V_OFF).load(Relaxed);
        let o = x.lane_u32(O_OFF).load(Relaxed);

        w.join().unwrap();
        log.record(format_args!("v={v} o={o:#x}"));
    });

    // The three states the line actually passes through.
    assert!(seen.contains("v=0 o=0x0"), "{seen:?}");
    assert!(seen.contains("v=0 o=0xabcd"), "{seen:?}");
    assert!(seen.contains("v=1 o=0xabcd"), "{seen:?}");

    assert!(
        !seen.contains("v=1 o=0x0"),
        "took the value lane the wide CAS installed and then an owner lane \
         older than the one it carried — the coherence floor the elided \
         identity write used to supply is gone: {seen:?}"
    );
}

/// The preserved lane is genuinely carried, not merely declared so: the bits
/// that come back are the ones that were read, whatever the caller passed as
/// the replacement.
#[test]
fn the_carried_lane_comes_from_the_cell_not_from_the_argument() {
    loom::model(|| {
        let x = AtomicU128::new(OWNER_BITS);

        // `new` names a different owner; the preserved lane ignores it.
        x.compare_exchange_preserving(O_MASK, OWNER_BITS, PUBLISHED | (0xBEEF << 64), Relaxed, Relaxed)
            .expect("uncontended CAS must succeed");

        assert_eq!(x.load(Relaxed), OWNER_BITS | PUBLISHED);
    });
}

/// The preservation claim is checked, not trusted. `fetch_modify_preserving`
/// is the general form, where a caller can get it wrong.
#[test]
#[should_panic(expected = "the preserved lane is not preserved")]
fn changing_a_carried_lane_traps() {
    loom::model(|| {
        let x = AtomicU128::new(OWNER_BITS);

        // Declares the owner lane carried, then moves it.
        x.fetch_modify_preserving(!O_MASK, |cur| (cur & !O_MASK) | (0xBEEF << 64), Relaxed);
    });
}

/// Runs `f` on a thread that is spawned and joined, so the caller's code
/// before the call is ordered before it and the code after is ordered after.
/// The two trap tests need a *different* thread than the reader — a thread
/// never loses a release it published itself — and need the order fixed,
/// because the whole point of the declaration is that the two are
/// DPOR-independent and the search is free to explore one order only.
fn carried_wake(x: &Arc<AtomicU128>) {
    let x = x.clone();
    thread::spawn(move || {
        x.compare_exchange_preserving(O_MASK, OWNER_BITS, PUBLISHED, AcqRel, Relaxed)
            .expect("uncontended CAS must succeed");
    })
    .join()
    .unwrap();
}

/// The one behavior elision costs: there is no store of the operation on the
/// carried lane, so nothing can synchronize-with it there. Reaching for that
/// edge is an error, not a quietly smaller search.
///
/// The declaration is that the lane is not a publication channel, so it is an
/// error whichever side runs first. This is the read side.
#[test]
#[should_panic(expected = "acquiring read of a preserved lane")]
fn acquiring_through_a_carried_lane_traps() {
    loom::model(|| {
        let x = Arc::new(AtomicU128::new(OWNER_BITS));
        carried_wake(&x);
        let _ = x.lane_u32(O_OFF).load(Acquire);
    });
}

/// ...and this is the operation side, which is not redundant: the two are
/// independent, so a search that explores only "read first" would never reach
/// the check above.
#[test]
#[should_panic(expected = "another thread has already acquire-read")]
fn carrying_a_lane_another_thread_acquires_traps() {
    loom::model(|| {
        let x = Arc::new(AtomicU128::new(OWNER_BITS));
        let _ = x.lane_u32(O_OFF).load(Acquire);
        carried_wake(&x);
    });
}

/// The deferred form of the same edge: a relaxed read of the carried lane
/// takes nothing and is sound where it stands, but the fence that would have
/// collected the elided identity write's release cannot.
#[test]
#[should_panic(expected = "read only through a preserved lane")]
fn fencing_after_a_carried_lane_read_traps() {
    loom::model(|| {
        let x = Arc::new(AtomicU128::new(OWNER_BITS));
        let _ = x.lane_u32(O_OFF).load(Relaxed);
        carried_wake(&x);
        fence(Acquire);
    });
}

/// A thread cannot lose a release it published itself, so its own preserved
/// lane is not a trap for it — reading and fencing over the lane it just
/// carried is fine.
#[test]
fn a_thread_may_acquire_through_a_lane_it_carried_itself() {
    loom::model(|| {
        let x = AtomicU128::new(OWNER_BITS);

        x.compare_exchange_preserving(O_MASK, OWNER_BITS, PUBLISHED, AcqRel, Relaxed)
            .expect("uncontended CAS must succeed");

        assert_eq!(x.lane_u32(O_OFF).load(Acquire), OWNER);
        fence(Acquire);
    });
}

/// A reader that also covers a lane the operation wrote keeps its route to the
/// operation's release, so acquiring is unaffected there.
#[test]
fn a_reader_routed_through_a_written_lane_may_still_acquire() {
    loom::model(|| {
        let x = Arc::new(AtomicU128::new(OWNER_BITS));

        let w = {
            let x = x.clone();
            thread::spawn(move || {
                x.compare_exchange_preserving(O_MASK, OWNER_BITS, PUBLISHED, AcqRel, Relaxed)
                    .expect("uncontended CAS must succeed");
            })
        };

        // The whole-cell snapshot covers the lanes the CAS wrote.
        let _ = x.load(Acquire);
        w.join().unwrap();
    });
}
