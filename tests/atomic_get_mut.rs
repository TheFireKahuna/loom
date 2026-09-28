#![deny(warnings, rust_2018_idioms)]

use loom::sync::atomic::{AtomicBool, AtomicI8, AtomicPtr, AtomicU128, AtomicUsize};
use loom::sync::Arc;
use loom::thread;

use std::sync::atomic::Ordering::*;

#[test]
fn get_mut_writes_reach_every_later_access() {
    loom::model(|| {
        let mut a = AtomicUsize::new(1);
        assert_eq!(*a.get_mut(), 1);
        *a.get_mut() += 41;
        assert_eq!(format!("{a:?}"), "42");
        assert_eq!(a.fetch_add(1, Relaxed), 42);
        *a.get_mut() = 7;
        assert_eq!(a.into_inner(), 7);

        let mut b = AtomicBool::new(false);
        *b.get_mut() = true;
        assert!(b.load(Relaxed));

        let mut i = AtomicI8::new(-3);
        *i.get_mut() -= 1;
        assert_eq!(i.load(Relaxed), -4);

        let mut x = 5u32;
        let mut p = AtomicPtr::new(std::ptr::null_mut());
        *p.get_mut() = &mut x;
        assert_eq!(p.load(Relaxed), &raw mut x);

        let mut w = AtomicU128::new(0);
        *w.get_mut() = u128::MAX;
        assert_eq!(w.lane_u64(8).load(Relaxed), u64::MAX);
    });
}

// The value a borrow leaves is the store every thread that happens after the
// borrow reads, whichever of them touches the cell first.
#[test]
fn get_mut_value_is_the_borrowers_store() {
    loom::model(|| {
        let mut a = AtomicUsize::new(0);
        *a.get_mut() = 1;
        let a = Arc::new(a);
        let ts: Vec<_> = (0..2)
            .map(|_| {
                let a = a.clone();
                thread::spawn(move || a.load(Relaxed))
            })
            .collect();
        for t in ts {
            assert_eq!(t.join().unwrap(), 1);
        }
    });
}
