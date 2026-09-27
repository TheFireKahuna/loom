#![deny(warnings)]

//! The `core` atomic surface loom mirrors as compositions of modelled
//! operations: `update`, `fetch_not`, the strict-provenance pointer RMWs, the
//! constructors and formatting, `compiler_fence`, and the `Atomic<T>` name.

use std::collections::BTreeSet;
use std::sync::{Arc as StdArc, Mutex as StdMutex};

use loom::sync::atomic::{
    compiler_fence, Atomic, AtomicBool, AtomicPtr, AtomicU32, AtomicU64, AtomicUsize,
    Ordering::*,
};
use loom::sync::Arc;
use loom::thread;

type Outcomes<T> = StdArc<StdMutex<BTreeSet<T>>>;

fn outcomes<T>() -> Outcomes<T> {
    StdArc::new(StdMutex::new(BTreeSet::new()))
}

fn builder() -> loom::model::Builder {
    let mut b = loom::model::Builder::new();
    b.threads = 1;
    b.max_permutations = None;
    b.max_duration = None;
    b.log = false;
    b.budgeted = false;
    b.stats = true;
    b
}

#[test]
fn update_is_atomic_and_returns_previous() {
    let seen = outcomes();
    let s = seen.clone();
    loom::model(move || {
        let n = Arc::new(AtomicU32::new(0));
        let b = Arc::new(AtomicBool::new(false));
        let t = {
            let (n, b) = (n.clone(), b.clone());
            thread::spawn(move || (n.update(AcqRel, Acquire, |x| x + 1), b.update(AcqRel, Acquire, |x| !x)))
        };
        let mine = (n.update(AcqRel, Acquire, |x| x + 1), b.update(AcqRel, Acquire, |x| !x));
        let theirs = t.join().unwrap();

        assert_eq!(n.load(Relaxed), 2);
        assert!(!b.load(Relaxed));
        let mut prev = [mine.0, theirs.0];
        prev.sort();
        assert_eq!(prev, [0, 1]);
        assert_ne!(mine.1, theirs.1);
        s.lock().unwrap().insert(mine.0);
    });
    // Both linearizations of the two updates are explored.
    assert_eq!(*seen.lock().unwrap(), BTreeSet::from([0, 1]));
}

#[test]
fn fetch_not_is_one_rmw() {
    loom::model(|| {
        let b = Arc::new(AtomicBool::new(false));
        let t = {
            let b = b.clone();
            thread::spawn(move || b.fetch_not(Relaxed))
        };
        let mine = b.fetch_not(Relaxed);
        let theirs = t.join().unwrap();
        assert_ne!(mine, theirs, "two RMWs never read the same store");
        assert!(!b.load(Relaxed));
    });
}

#[test]
fn ptr_update_moves_the_pointer() {
    loom::model(|| {
        let mut buf = [10u32, 11, 12, 13];
        let base = buf.as_mut_ptr();
        let p = AtomicPtr::new(base);
        let prev = p.update(Relaxed, Relaxed, |q| q.wrapping_add(3));
        assert_eq!(prev, base);
        assert_eq!(unsafe { *p.load(Relaxed) }, 13);
    });
}

#[test]
fn ptr_add_sub_race_and_preserve_provenance() {
    let seen = outcomes();
    let s = seen.clone();
    loom::model(move || {
        let buf: &'static mut [u64; 8] = Box::leak(Box::new([0, 1, 2, 3, 4, 5, 6, 7]));
        let base = buf.as_mut_ptr();
        let p = Arc::new(AtomicPtr::new(base));
        let t = {
            let p = p.clone();
            thread::spawn(move || p.fetch_ptr_add(2, Relaxed) as usize)
        };
        let mine = p.fetch_byte_add(8, Relaxed);
        let theirs = t.join().unwrap() as *mut u64;

        // Both read-modify-writes land: 2 elements + 8 bytes = 3 elements.
        let end = p.load(Relaxed);
        assert_eq!(end, base.wrapping_add(3));
        // Every pointer returned and stored still reads the buffer.
        assert_eq!(unsafe { *end }, 3);
        assert_eq!(unsafe { *mine }, (mine as usize - base as usize) as u64 / 8);
        s.lock().unwrap().insert(unsafe { *mine });
        let _ = theirs;

        assert_eq!(p.fetch_ptr_sub(1, Relaxed), end);
        assert_eq!(p.fetch_byte_sub(16, Relaxed), base.wrapping_add(2));
        assert_eq!(p.load(Relaxed), base);
        unsafe { drop(Box::from_raw(buf)) };
    });
    // Our add read either the initial pointer or the other thread's result.
    assert_eq!(*seen.lock().unwrap(), BTreeSet::from([0, 2]));
}

#[test]
fn ptr_tag_bits_race() {
    loom::model(|| {
        let slot: &'static mut u64 = Box::leak(Box::new(42));
        let base: *mut u64 = slot;
        let p = Arc::new(AtomicPtr::new(base));
        let t = {
            let p = p.clone();
            thread::spawn(move || {
                p.fetch_or(0b01, Relaxed);
            })
        };
        p.fetch_or(0b10, Relaxed);
        t.join().unwrap();

        let tagged = p.load(Relaxed);
        assert_eq!(tagged.addr(), base.addr() | 0b11);
        assert_eq!(p.fetch_xor(0b01, Relaxed), tagged);
        assert_eq!(p.fetch_and(!0b11, Relaxed).addr(), base.addr() | 0b10);
        let untagged = p.load(Relaxed);
        assert_eq!(untagged, base);
        assert_eq!(unsafe { *untagged }, 42);
        unsafe { drop(Box::from_raw(untagged)) };
    });
}

#[test]
fn ptr_construction_and_formatting() {
    loom::model(|| {
        let null = AtomicPtr::<u8>::null();
        assert!(null.load(Relaxed).is_null());

        let mut x = 7u8;
        let px: *mut u8 = &mut x;
        let p = AtomicPtr::from(px);
        assert_eq!(p.load(Relaxed), px);
        assert_eq!(format!("{:p}", p), format!("{:p}", px));
        assert_eq!(format!("{:?}", p), format!("{:?}", px));
    });
}

#[test]
fn debug_prints_the_value() {
    loom::model(|| {
        let n = AtomicU64::new(5);
        assert_eq!(format!("{n:?}"), "5");
        n.store(9, Relaxed);
        assert_eq!(format!("{n:?}"), "9");
        assert_eq!(format!("{:?}", AtomicBool::new(true)), "true");
        assert_eq!(format!("{:?}", loom::sync::atomic::AtomicI8::new(-3)), "-3");
    });
}

static DEFERRED: AtomicUsize = AtomicUsize::new(3);

#[test]
fn debug_of_an_unregistered_cell_shows_its_initial_value() {
    loom::model(|| {
        assert_eq!(format!("{DEFERRED:?}"), "3");
        DEFERRED.store(4, Relaxed);
        assert_eq!(format!("{DEFERRED:?}"), "4");
    });
}

#[test]
fn debug_outside_a_model_is_opaque() {
    assert_eq!(format!("{DEFERRED:?}"), "AtomicUsize { .. }");
}

/// Formatting is no modelled step: the same rig explores exactly as many
/// executions with a `Debug` between every operation as without.
#[test]
fn debug_does_not_perturb_exploration() {
    fn rig(format: bool) -> usize {
        builder()
            .check(move || {
                let x = Arc::new(AtomicUsize::new(0));
                let t = {
                    let x = x.clone();
                    thread::spawn(move || {
                        x.store(1, Relaxed);
                        if format {
                            let _ = format!("{x:?}");
                        }
                        x.store(2, Relaxed);
                    })
                };
                if format {
                    let _ = format!("{x:?}");
                }
                let _ = x.load(Relaxed);
                if format {
                    let _ = format!("{x:?}");
                }
                let _ = x.load(Relaxed);
                t.join().unwrap();
            })
            .executions
    }
    assert_eq!(rig(true), rig(false));
}

#[test]
fn compiler_fence_is_a_no_op() {
    loom::model(|| {
        compiler_fence(Acquire);
        compiler_fence(Release);
        compiler_fence(AcqRel);
        compiler_fence(SeqCst);
    });
}

#[test]
#[should_panic(expected = "there is no such thing as a relaxed fence")]
fn relaxed_compiler_fence_panics() {
    compiler_fence(Relaxed);
}

#[test]
fn generic_atomic_names_the_cells() {
    fn same<T>(_: &T, _: &T) {}
    loom::model(|| {
        let a: Atomic<u32> = AtomicU32::new(1);
        same(&a, &AtomicU32::new(0));
        let b = Atomic::<u64>::new(2);
        let c = Atomic::<bool>::new(true);
        let d = Atomic::<*mut u8>::null();
        let e = Atomic::<i128>::new(-1);
        assert_eq!(a.load(Relaxed) as u64 + b.load(Relaxed), 3);
        assert!(c.load(Relaxed));
        assert!(d.load(Relaxed).is_null());
        assert_eq!(e.load(Relaxed), -1);
    });
}

