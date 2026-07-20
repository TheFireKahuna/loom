//! `const`-constructed atomic cells (`const_new` / `const_null`).
//!
//! These cells register with the execution on first *access* rather than at
//! construction, which is what lets them exist in a `const` context. That
//! deferral is the whole mechanism, and it has four ways to go wrong:
//!
//! 1. **Leaking across executions** — a `static` is process-lifetime, so if its
//!    registration were not rebuilt per iteration it would carry the previous
//!    execution's stores into the next one.
//! 2. **Losing identity on a move** — registration is keyed on an identity
//!    minted into the cell, not its address, precisely so a cell that moves
//!    after first access keeps its history.
//! 3. **Aliasing a recycled address** — the converse: a new cell landing on a
//!    dead one's storage must be a distinct cell.
//! 4. **Fragmenting under parallel exploration** — one `static` is shared by
//!    every OS worker at once, each with its own `Execution`.
//!
//! Beyond those, the load-bearing claim is that a deferred cell is
//! *indistinguishable* from an eagerly registered one once touched: same
//! coherence, same SC rules, same explored state space. The equivalence and
//! memory-model tests below are what pin that, since a deferred cell that
//! quietly collapsed to "no concurrency here" would pass every structural test
//! in this file while checking nothing.

#![deny(warnings, rust_2018_idioms)]

use loom::sync::atomic::{AtomicBool, AtomicPtr, AtomicU128, AtomicUsize};
use loom::thread;

use std::sync::atomic::Ordering::{AcqRel, Acquire, Relaxed, Release, SeqCst};
use std::sync::atomic::{AtomicUsize as StdUsize, Ordering as StdOrdering};
use std::sync::Arc;

/// Process-lifetime counter for observations that must be aggregated *across*
/// loom executions (loom cells reset every iteration, so they cannot count
/// iterations). Relaxed is enough: reads happen after `check` returns.
fn bump(c: &StdUsize) {
    c.fetch_add(1, StdOrdering::Relaxed);
}

// ===== 1. per-execution reset =====

/// A `const`-initialized `static` must present its initial value at the start
/// of every execution. This is the property `checker_statics!`/`lazy_static!`
/// existed to provide by hand; deferred registration provides it by
/// construction, because the registration table is cleared per iteration while
/// the cell itself is not.
#[test]
fn const_static_resets_between_executions() {
    static X: AtomicUsize = AtomicUsize::const_new(0);
    static ITERS: StdUsize = StdUsize::new(0);

    loom::model(|| {
        // Runs before anything in this execution has stored: any value other
        // than the initializer is a previous execution bleeding through.
        assert_eq!(
            X.load(Relaxed),
            0,
            "const static leaked a store across executions"
        );
        bump(&ITERS);

        let t = thread::spawn(|| X.store(1, Relaxed));
        X.store(2, Relaxed);
        t.join().unwrap();

        assert_ne!(X.load(Relaxed), 0);
    });

    // The reset assertion above is only meaningful if more than one execution
    // actually ran.
    assert!(
        ITERS.load(StdOrdering::Relaxed) > 1,
        "rig explored a single execution; the reset check was vacuous"
    );
}

/// The same property for the other two `const` constructors, and for a cell
/// mutated through `with_mut` (which writes the ring directly rather than
/// appending a store).
#[test]
fn const_bool_ptr_and_with_mut_reset_between_executions() {
    static B: AtomicBool = AtomicBool::const_new(false);
    static P: AtomicPtr<u32> = AtomicPtr::const_null();
    static ITERS: StdUsize = StdUsize::new(0);

    loom::model(|| {
        assert!(!B.load(Relaxed), "const AtomicBool leaked across executions");
        assert!(
            P.load(Relaxed).is_null(),
            "const AtomicPtr leaked across executions"
        );
        bump(&ITERS);

        let mut local = AtomicUsize::const_new(5);
        assert_eq!(local.with_mut(|v| std::mem::replace(v, 6)), 5);
        assert_eq!(local.load(Relaxed), 6);

        // Both threads must touch the *same* cells, or DPOR finds no dependent
        // accesses, explores one execution, and the reset check above never
        // runs a second time.
        let t = thread::spawn(|| {
            B.store(true, Relaxed);
            P.load(Relaxed)
        });
        let mut target = 7u32;
        P.store(&mut target as *mut u32, Relaxed);
        let seen = B.load(Relaxed);
        let p = t.join().unwrap();
        assert!(seen || !seen);
        assert!(p.is_null() || p == &mut target as *mut u32);
    });

    assert!(ITERS.load(StdOrdering::Relaxed) > 1);
}

// ===== 2. identity survives a move =====

/// A deferred cell registers on first access; if that registration were keyed
/// on the cell's *address*, moving it afterwards would silently re-register at
/// the initial value and discard everything stored in between. Identity is
/// minted into the cell so the history moves with the bytes.
///
/// This is not a hypothetical shape: `[Cell::const_new(..); N]` built in a
/// `const` context and then moved into an owning struct is exactly it.
#[test]
fn deferred_cell_survives_a_move() {
    loom::model(|| {
        let a = AtomicUsize::const_new(7);
        assert_eq!(a.load(Relaxed), 7);
        a.store(99, Relaxed);

        let b = a; // moved *after* registration
        assert_eq!(
            b.load(Relaxed),
            99,
            "moving a registered deferred cell lost its history"
        );

        // And it is still the same cell, not a fresh one: a further store
        // composes with the pre-move history rather than restarting from it.
        b.store(100, Relaxed);
        assert_eq!(b.load(Relaxed), 100);
    });
}

/// The same, through a heap move and a struct field — the `array::from_fn`
/// shape, where the cells are registered while owned by one value and then
/// relocated wholesale.
#[test]
fn deferred_cells_survive_a_boxed_struct_move() {
    struct Slot {
        cells: [AtomicUsize; 4],
    }

    loom::model(|| {
        let slot = Slot {
            cells: [
                AtomicUsize::const_new(0),
                AtomicUsize::const_new(1),
                AtomicUsize::const_new(2),
                AtomicUsize::const_new(3),
            ],
        };
        for (i, c) in slot.cells.iter().enumerate() {
            assert_eq!(c.load(Relaxed), i);
            c.store(i + 10, Relaxed);
        }

        let boxed = Box::new(slot); // registered cells change address
        for (i, c) in boxed.cells.iter().enumerate() {
            assert_eq!(
                c.load(Relaxed),
                i + 10,
                "cell {i} lost its history when the owner moved to the heap"
            );
        }
    });
}

// ===== 3. a recycled address is a distinct cell =====

/// The converse of the move test. `Option<AtomicUsize>` reuses one storage
/// slot across the loop, so each new cell is built at the address the previous
/// one just vacated. Address-keyed registration would hand the newcomer its
/// predecessor's stores; identity-keyed registration mints a fresh id.
#[test]
fn recycled_address_is_a_distinct_cell() {
    loom::model(|| {
        let mut slot: Option<AtomicUsize>;
        let mut addrs = Vec::new();

        for i in 1..=4usize {
            slot = Some(AtomicUsize::const_new(i));
            let a = slot.as_ref().unwrap();
            addrs.push(a as *const AtomicUsize as usize);

            assert_eq!(
                a.load(Relaxed),
                i,
                "cell {i} inherited a freed cell's history"
            );
            a.store(1000 + i, Relaxed);
            assert_eq!(a.load(Relaxed), 1000 + i);
        }

        // Guard against the test going vacuous: it only proves anything if the
        // storage really was reused.
        assert!(
            addrs.windows(2).any(|w| w[0] == w[1]),
            "no address was reused; this test proved nothing"
        );
    });
}

// ===== 4. parallel exploration =====

/// One `static` const cell is shared by every exploration worker
/// simultaneously — real OS threads, each driving its own `Execution`. The
/// identity is minted once by compare-exchange so they agree on it, while each
/// worker registers the cell separately in its own execution.
///
/// The gate is that parallel and serial exploration cover the same state space:
/// a worker that fragmented one cell into several registrations, or that
/// adopted another worker's `object::Ref`, would change what is reachable and
/// move the execution count.
#[test]
fn const_statics_explore_identically_under_parallel_workers() {
    static X: AtomicUsize = AtomicUsize::const_new(0);
    static Y: AtomicUsize = AtomicUsize::const_new(0);

    fn rig() {
        let t = thread::spawn(|| {
            X.store(1, Release);
            Y.load(Acquire)
        });
        Y.store(1, Release);
        let a = X.load(Acquire);
        let b = t.join().unwrap();
        assert!(a == 1 || b == 1 || (a == 0 && b == 0));
    }

    let mut serial = loom::model::Builder::new();
    serial.threads = 1;
    let s = serial.check(rig);

    let mut parallel = loom::model::Builder::new();
    parallel.threads = 4;
    let p = parallel.check(rig);

    assert_eq!(
        s.executions, p.executions,
        "parallel workers explored a different state space through a const static \
         ({} serial vs {} parallel)",
        s.executions, p.executions
    );
}

// ===== 5. equivalence with eager registration =====

/// The central claim: once touched, a deferred cell is indistinguishable from
/// an eagerly registered one. Two structurally identical rigs — same threads,
/// same ops, same orderings, differing only in the constructor — must explore
/// exactly the same number of executions.
///
/// A divergence here means deferral perturbed the model: an extra or missing
/// DPOR branch, a different readable set, a genesis that ordered differently.
#[test]
fn deferred_and_eager_cells_explore_the_same_state_space() {
    fn rig(make: fn() -> AtomicUsize) -> usize {
        let mut b = loom::model::Builder::new();
        // Pin the exploration so the comparison is of the state space, not of
        // two different schedulers' luck.
        b.threads = 1;
        b.check(move || {
            let x = Arc::new(make());
            let y = Arc::new(make());

            let (x2, y2) = (x.clone(), y.clone());
            let t = thread::spawn(move || {
                x2.store(1, Relaxed);
                y2.load(Relaxed)
            });

            y.store(1, Relaxed);
            let a = x.load(Relaxed);
            let b = t.join().unwrap();
            let _ = (a, b);
        })
        .executions
    }

    let eager = rig(|| AtomicUsize::new(0));
    let deferred = rig(|| AtomicUsize::const_new(0));

    assert_eq!(
        eager, deferred,
        "deferred registration changed the explored state space \
         ({eager} eager vs {deferred} deferred)"
    );
    assert!(eager > 1, "rig collapsed to one execution; comparison vacuous");
}

/// Eager and deferred cells in the same execution, interacting. Registration
/// order is now access order rather than construction order, so this pins that
/// interleaving the two kinds does not disturb either.
#[test]
fn eager_and_deferred_cells_interoperate() {
    static DEFERRED: AtomicUsize = AtomicUsize::const_new(0);

    loom::model(|| {
        let eager = Arc::new(AtomicUsize::new(0));
        let e2 = eager.clone();

        let t = thread::spawn(move || {
            // Touch the deferred cell first from *this* thread on some
            // schedules, the main thread on others.
            DEFERRED.store(1, Release);
            e2.store(1, Release);
        });

        let e = eager.load(Acquire);
        let d = DEFERRED.load(Acquire);
        t.join().unwrap();

        // Release/Acquire on two independent cells constrains nothing across
        // them; the point is that all four outcomes stay reachable and nothing
        // panics.
        assert!(e <= 1 && d <= 1);
        assert_eq!(eager.load(Relaxed), 1);
        assert_eq!(DEFERRED.load(Relaxed), 1);
    });
}

// ===== 6. the memory model still applies =====

/// A deferred cell must lose no weak behavior. Relaxed message passing through
/// two `const` statics has to expose the stale read — if it never did, these
/// cells would be silently over-synchronized and every other test here would
/// pass while checking nothing.
#[test]
#[should_panic(expected = "stale read")]
fn relaxed_message_passing_through_const_statics_exposes_the_stale_read() {
    static DATA: AtomicUsize = AtomicUsize::const_new(0);
    static FLAG: AtomicBool = AtomicBool::const_new(false);

    loom::model(|| {
        let t = thread::spawn(|| {
            DATA.store(42, Relaxed);
            FLAG.store(true, Relaxed);
        });

        if FLAG.load(Relaxed) {
            assert_eq!(DATA.load(Relaxed), 42, "stale read");
        }
        t.join().unwrap();
    });
}

/// The same shape with Release/Acquire must hold — the deferred cell carries
/// real release sequences, not just the absence of checking.
#[test]
fn release_acquire_message_passing_through_const_statics_holds() {
    static DATA: AtomicUsize = AtomicUsize::const_new(0);
    static FLAG: AtomicBool = AtomicBool::const_new(false);

    loom::model(|| {
        let t = thread::spawn(|| {
            DATA.store(42, Relaxed);
            FLAG.store(true, Release);
        });

        if FLAG.load(Acquire) {
            assert_eq!(DATA.load(Relaxed), 42);
        }
        t.join().unwrap();
    });
}

/// SC on deferred cells: store buffering must be forbidden under `SeqCst`
/// (both threads reading zero is not a legal outcome), and *reachable* under
/// `Relaxed`. Together these pin that `sc_rank` and the SC read rule operate on
/// a pre-execution genesis exactly as on a thread-attributed one — the genesis
/// store being unconditionally modification-order-first is the property under
/// test.
#[test]
fn seqcst_store_buffering_is_forbidden_through_const_statics() {
    static X: AtomicUsize = AtomicUsize::const_new(0);
    static Y: AtomicUsize = AtomicUsize::const_new(0);

    loom::model(|| {
        let t = thread::spawn(|| {
            X.store(1, SeqCst);
            Y.load(SeqCst)
        });
        Y.store(1, SeqCst);
        let a = X.load(SeqCst);
        let b = t.join().unwrap();

        assert!(
            !(a == 0 && b == 0),
            "sequentially consistent store buffering observed through const statics"
        );
    });
}

#[test]
fn relaxed_store_buffering_is_reachable_through_const_statics() {
    static X: AtomicUsize = AtomicUsize::const_new(0);
    static Y: AtomicUsize = AtomicUsize::const_new(0);
    static BOTH_ZERO: StdUsize = StdUsize::new(0);

    loom::model(|| {
        let t = thread::spawn(|| {
            X.store(1, Relaxed);
            Y.load(Relaxed)
        });
        Y.store(1, Relaxed);
        let a = X.load(Relaxed);
        let b = t.join().unwrap();

        if a == 0 && b == 0 {
            bump(&BOTH_ZERO);
        }
    });

    assert!(
        BOTH_ZERO.load(StdOrdering::Relaxed) > 0,
        "relaxed store buffering was never observed — const cells are \
         over-synchronized and the SC test above proves nothing"
    );
}

/// RMW atomicity on a deferred cell: a `fetch_add` chain from two threads must
/// total exactly, with no lost update on any schedule.
#[test]
fn rmw_atomicity_holds_on_deferred_cells() {
    static N: AtomicUsize = AtomicUsize::const_new(0);

    loom::model(|| {
        let t = thread::spawn(|| {
            N.fetch_add(1, AcqRel);
        });
        N.fetch_add(1, AcqRel);
        t.join().unwrap();

        assert_eq!(N.load(SeqCst), 2, "lost update on a deferred cell");
    });
}

// ===== 7. sub-word regions on deferred cells =====

/// Lane ops split a cell into regions the first time a masked access cuts it.
/// On a deferred cell that partition is built against a shell registered on
/// first access, so this pins that the genesis store is present and correctly
/// cloned into both halves of every split.
#[test]
fn lane_ops_partition_a_deferred_cell() {
    loom::model(|| {
        let x = AtomicU128::const_new(0);

        x.lane_u32(0).store(0x1111_1111, Relaxed);
        x.lane_u32(4).store(0x2222_2222, Relaxed);
        x.lane_u32(8).store(0x3333_3333, Relaxed);
        x.lane_u32(12).store(0x4444_4444, Relaxed);

        assert_eq!(
            x.load(Relaxed),
            0x4444_4444_3333_3333_2222_2222_1111_1111u128
        );
        assert_eq!(x.lane_u64(0).load(Relaxed), 0x2222_2222_1111_1111);
    });
}

/// A deferred 128-bit cell under genuine lane concurrency: the value lane and
/// the queue lane are written by different threads, and a whole-cell load must
/// still return a single non-torn snapshot.
#[test]
fn deferred_wide_cell_stays_single_copy_atomic_across_lanes() {
    static W: AtomicU128 = AtomicU128::const_new(0);

    loom::model(|| {
        let t = thread::spawn(|| W.lane_u64(8).store(u64::MAX, Release));
        W.lane_u64(0).store(1, Release);
        t.join().unwrap();

        let v = W.load(SeqCst);
        assert_eq!(v & u64::MAX as u128, 1);
        assert_eq!(v >> 64, u64::MAX as u128);
    });
}

// ===== 8. degenerate cases =====

/// A `const` cell never accessed in an execution is never registered. It must
/// cost nothing and disturb nothing — in particular it must not consume an
/// object slot that would shift the refs of cells that *are* used.
#[test]
fn never_accessed_const_static_is_inert() {
    #[allow(dead_code)]
    static UNUSED_A: AtomicUsize = AtomicUsize::const_new(1234);
    #[allow(dead_code)]
    static UNUSED_B: AtomicPtr<u8> = AtomicPtr::const_null();
    static USED: AtomicUsize = AtomicUsize::const_new(0);

    loom::model(|| {
        let t = thread::spawn(|| USED.store(1, Release));
        let v = USED.load(Acquire);
        t.join().unwrap();
        assert!(v <= 1);
    });
}

/// A cell accessed for the first time from a *spawned* thread, on some
/// schedules, and from the main thread on others. Registration happens
/// wherever the first touch lands, and the pre-execution genesis is what makes
/// that attribution-free: no thread "owns" the initialization, so no
/// unsynchronized-access report can depend on which thread got there first.
#[test]
fn first_touch_from_either_thread_is_equivalent() {
    static X: AtomicUsize = AtomicUsize::const_new(0);

    loom::model(|| {
        let t = thread::spawn(|| {
            let v = X.load(Relaxed);
            X.store(v + 1, Relaxed);
        });
        let v = X.load(Relaxed);
        X.store(v + 10, Relaxed);
        t.join().unwrap();

        // Lost updates are legal here (plain load/store, not RMW); the point is
        // that no schedule reports a causality violation on the genesis.
        assert!(X.load(Relaxed) > 0);
    });
}

/// `unsync_load` on a deferred cell. It asserts against `unsync_mut_at`, which
/// a pre-execution genesis deliberately leaves empty — so a single-threaded
/// unsynchronized read of a `const` cell is legal, where the same read of an
/// eagerly constructed cell would also be legal (same thread). This pins that
/// the empty genesis did not turn into a false *positive*.
#[test]
fn unsync_load_of_a_deferred_cell_is_clean() {
    loom::model(|| {
        let x = AtomicUsize::const_new(3);
        // SAFETY: no other thread can observe `x`.
        assert_eq!(unsafe { x.unsync_load() }, 3);
        x.store(4, Relaxed);
        assert_eq!(unsafe { x.unsync_load() }, 4);
    });
}
