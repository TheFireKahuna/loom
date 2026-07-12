#![deny(warnings, rust_2018_idioms)]

macro_rules! test_int {
    ($name:ident, $int:ty, $atomic:ty) => {
        mod $name {
            use loom::sync::atomic::*;
            use std::sync::atomic::Ordering::SeqCst;

            // High and low halves are both non-trivial so the 128-bit types
            // get full-width coverage; the low 64 bits are what the narrower
            // types have always been tested with.
            const NUM_A: u128 = (7594829094997581893 << 64) | 11641914933775430211;
            const NUM_B: u128 = (17298086858409381443 << 64) | 13209405719799650717;

            #[test]
            fn xor() {
                loom::model(|| {
                    let a: $int = NUM_A as $int;
                    let b: $int = NUM_B as $int;

                    let atomic = <$atomic>::new(a);
                    let prev = atomic.fetch_xor(b, SeqCst);

                    assert_eq!(a, prev, "prev did not match");
                    assert_eq!(a ^ b, atomic.load(SeqCst), "load failed");
                });
            }

            #[test]
            fn max() {
                loom::model(|| {
                    let a: $int = NUM_A as $int;
                    let b: $int = NUM_B as $int;

                    let atomic = <$atomic>::new(a);
                    let prev = atomic.fetch_max(b, SeqCst);

                    assert_eq!(a, prev, "prev did not match");
                    assert_eq!(a.max(b), atomic.load(SeqCst), "load failed");
                });
            }

            #[test]
            fn min() {
                loom::model(|| {
                    let a: $int = NUM_A as $int;
                    let b: $int = NUM_B as $int;

                    let atomic = <$atomic>::new(a);
                    let prev = atomic.fetch_min(b, SeqCst);

                    assert_eq!(a, prev, "prev did not match");
                    assert_eq!(a.min(b), atomic.load(SeqCst), "load failed");
                });
            }

            #[test]
            fn compare_exchange() {
                loom::model(|| {
                    let a: $int = NUM_A as $int;
                    let b: $int = NUM_B as $int;

                    let atomic = <$atomic>::new(a);
                    assert_eq!(Err(a), atomic.compare_exchange(b, a, SeqCst, SeqCst));
                    assert_eq!(Ok(a), atomic.compare_exchange(a, b, SeqCst, SeqCst));

                    assert_eq!(b, atomic.load(SeqCst));
                });
            }

            #[test]
            #[ignore]
            fn compare_exchange_weak() {
                loom::model(|| {
                    let a: $int = NUM_A as $int;
                    let b: $int = NUM_B as $int;

                    let atomic = <$atomic>::new(a);
                    assert_eq!(Err(a), atomic.compare_exchange_weak(b, a, SeqCst, SeqCst));
                    assert_eq!(Ok(a), atomic.compare_exchange_weak(a, b, SeqCst, SeqCst));

                    assert_eq!(b, atomic.load(SeqCst));
                });
            }

            #[test]
            fn fetch_update() {
                loom::model(|| {
                    let a: $int = NUM_A as $int;
                    let b: $int = NUM_B as $int;

                    let atomic = <$atomic>::new(a);
                    assert_eq!(Ok(a), atomic.fetch_update(SeqCst, SeqCst, |_| Some(b)));
                    assert_eq!(Err(b), atomic.fetch_update(SeqCst, SeqCst, |_| None));
                    assert_eq!(b, atomic.load(SeqCst));
                });
            }
        }
    };
}

test_int!(atomic_u8, u8, AtomicU8);
test_int!(atomic_u16, u16, AtomicU16);
test_int!(atomic_u32, u32, AtomicU32);
test_int!(atomic_usize, usize, AtomicUsize);

test_int!(atomic_i8, i8, AtomicI8);
test_int!(atomic_i16, i16, AtomicI16);
test_int!(atomic_i32, i32, AtomicI32);
test_int!(atomic_isize, isize, AtomicIsize);

#[cfg(target_pointer_width = "64")]
test_int!(atomic_u64, u64, AtomicU64);

#[cfg(target_pointer_width = "64")]
test_int!(atomic_i64, i64, AtomicI64);

test_int!(atomic_u128, u128, AtomicU128);

test_int!(atomic_i128, i128, AtomicI128);

/// Tests that specifically exercise the upper 64 bits of the 128-bit atomics,
/// which a lossy internal representation would silently drop.
mod atomic_128_full_width {
    use loom::sync::atomic::{AtomicI128, AtomicU128};
    use loom::thread;
    use std::sync::atomic::Ordering::{Acquire, Release, SeqCst};
    use std::sync::Arc;

    const HIGH_BIT: u128 = 1 << 127;
    const LOW_HALF: u128 = u64::MAX as u128;

    #[test]
    fn fetch_add_carries_across_bit_64() {
        loom::model(|| {
            let atomic = AtomicU128::new(LOW_HALF);
            assert_eq!(atomic.fetch_add(1, SeqCst), LOW_HALF);
            assert_eq!(atomic.load(SeqCst), 1 << 64);
        });
    }

    #[test]
    fn fetch_sub_borrows_across_bit_64() {
        loom::model(|| {
            let atomic = AtomicU128::new(1 << 64);
            assert_eq!(atomic.fetch_sub(1, SeqCst), 1 << 64);
            assert_eq!(atomic.load(SeqCst), LOW_HALF);
        });
    }

    #[test]
    fn fetch_add_wraps_at_max() {
        loom::model(|| {
            let atomic = AtomicU128::new(u128::MAX);
            assert_eq!(atomic.fetch_add(1, SeqCst), u128::MAX);
            assert_eq!(atomic.load(SeqCst), 0);
        });
    }

    #[test]
    fn compare_exchange_distinguishes_high_words() {
        loom::model(|| {
            // Identical low 64 bits; only the high word differs.
            let a = (1 << 64) | 42;
            let b = (2 << 64) | 42;

            let atomic = AtomicU128::new(a);
            assert_eq!(Err(a), atomic.compare_exchange(b, HIGH_BIT, SeqCst, SeqCst));
            assert_eq!(Ok(a), atomic.compare_exchange(a, b, SeqCst, SeqCst));
            assert_eq!(b, atomic.load(SeqCst));
        });
    }

    #[test]
    fn negative_i128_round_trips() {
        loom::model(|| {
            let atomic = AtomicI128::new(i128::MIN);
            assert_eq!(atomic.swap(-1, SeqCst), i128::MIN);
            assert_eq!(atomic.load(SeqCst), -1);
            assert_eq!(atomic.fetch_min(i128::MIN, SeqCst), -1);
            assert_eq!(atomic.load(SeqCst), i128::MIN);
        });
    }

    #[test]
    fn store_release_load_acquire_across_threads() {
        loom::model(|| {
            let atomic = Arc::new(AtomicU128::new(0));
            let flagged = HIGH_BIT | 1;

            let a = atomic.clone();
            let th = thread::spawn(move || {
                a.store(flagged, Release);
            });

            let observed = atomic.load(Acquire);
            assert!(
                observed == 0 || observed == flagged,
                "torn or invented value: {observed:#034x}"
            );
            th.join().unwrap();
        });
    }

    #[test]
    fn concurrent_rmw_keeps_both_words() {
        loom::model(|| {
            // Each thread increments its own 64-bit half; if the model lost
            // either half, one of the increments would vanish.
            let atomic = Arc::new(AtomicU128::new(0));

            let a = atomic.clone();
            let th = thread::spawn(move || {
                a.fetch_add(1 << 64, SeqCst);
            });
            atomic.fetch_add(1, SeqCst);
            th.join().unwrap();

            assert_eq!(atomic.load(SeqCst), (1 << 64) | 1);
        });
    }

    #[test]
    fn with_mut_and_into_inner_preserve_full_width() {
        loom::model(|| {
            let mut atomic = AtomicU128::new(HIGH_BIT);
            atomic.with_mut(|v| {
                assert_eq!(*v, HIGH_BIT);
                *v |= 1;
            });
            assert_eq!(atomic.load(SeqCst), HIGH_BIT | 1);
            assert_eq!(atomic.into_inner(), HIGH_BIT | 1);
        });
    }
}
