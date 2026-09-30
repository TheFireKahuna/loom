use loom::sync::{Arc, RwLock, TryLockResult};
use loom::thread;

use std::rc::Rc;
use std::sync::TryLockError;

#[test]
fn rwlock_read_one() {
    loom::model(|| {
        let lock = Arc::new(RwLock::new(1));
        let c_lock = lock.clone();

        let n = lock.read().unwrap();
        assert_eq!(*n, 1);

        thread::spawn(move || {
            let r = c_lock.read();
            assert!(r.is_ok());
        })
        .join()
        .unwrap();
    });
}

#[test]
fn rwlock_read_two_write_one() {
    loom::model(|| {
        let lock = Arc::new(RwLock::new(1));

        for _ in 0..2 {
            let lock = lock.clone();

            thread::spawn(move || {
                let _l = lock.read().unwrap();

                thread::yield_now();
            });
        }

        let _l = lock.write().unwrap();
        thread::yield_now();
    });
}

#[test]
fn rwlock_write_three() {
    loom::model(|| {
        let lock = Arc::new(RwLock::new(1));

        for _ in 0..2 {
            let lock = lock.clone();
            thread::spawn(move || {
                let _l = lock.write().unwrap();

                thread::yield_now();
            });
        }

        let _l = lock.write().unwrap();
        thread::yield_now();
    });
}

#[test]
fn rwlock_write_then_try_write() {
    loom::model(|| {
        let lock = Arc::new(RwLock::new(1));

        let _l1 = lock.write().unwrap();

        assert!(matches!(
            lock.try_write(),
            TryLockResult::Err(TryLockError::WouldBlock)
        ));
    });
}

#[test]
fn rwlock_write_then_try_read() {
    loom::model(|| {
        let lock = Arc::new(RwLock::new(1));

        let _l1 = lock.write().unwrap();

        assert!(matches!(
            lock.try_read(),
            TryLockResult::Err(TryLockError::WouldBlock)
        ));
    });
}

#[test]
fn rwlock_read_then_try_write() {
    loom::model(|| {
        let lock = Arc::new(RwLock::new(1));

        let _l1 = lock.write().unwrap();

        assert!(matches!(
            lock.try_write(),
            TryLockResult::Err(TryLockError::WouldBlock)
        ));
    });
}

#[test]
fn rwlock_try_read() {
    loom::model(|| {
        let lock = RwLock::new(1);

        match lock.try_read() {
            Ok(n) => assert_eq!(*n, 1),
            Err(_) => unreachable!(),
        };
    });
}

#[test]
fn rwlock_write() {
    loom::model(|| {
        let lock = RwLock::new(1);

        let mut n = lock.write().unwrap();
        *n = 2;

        assert!(lock.try_read().is_err());
    });
}

#[test]
fn rwlock_try_write() {
    loom::model(|| {
        let lock = RwLock::new(1);

        let n = lock.read().unwrap();
        assert_eq!(*n, 1);

        assert!(lock.try_write().is_err());
    });
}

#[test]
fn rwlock_into_inner() {
    loom::model(|| {
        let lock = Rc::new(RwLock::new(0));

        let ths: Vec<_> = (0..2)
            .map(|_| {
                let lock = lock.clone();

                thread::spawn(move || {
                    *lock.write().unwrap() += 1;
                })
            })
            .collect();

        for th in ths {
            th.join().unwrap();
        }

        let lock = Rc::try_unwrap(lock).unwrap().into_inner().unwrap();
        assert_eq!(lock, 2);
    })
}

// An unlock is a step of its own: a peer's `try_read` or `try_write` may run
// while a critical section with no modelled op inside it still holds the lock.
#[test]
fn try_locks_see_empty_critical_sections_held() {
    use std::sync::atomic::{AtomicBool, Ordering::SeqCst};

    static READ_FAILED: AtomicBool = AtomicBool::new(false);
    static WRITE_FAILED: AtomicBool = AtomicBool::new(false);
    loom::model(|| {
        let l = Arc::new(RwLock::new(()));
        let l2 = l.clone();
        let t = thread::spawn(move || drop(l2.write().unwrap()));
        if l.try_read().is_err() {
            READ_FAILED.store(true, SeqCst);
        }
        t.join().unwrap();

        let l2 = l.clone();
        let t = thread::spawn(move || drop(l2.read().unwrap()));
        if l.try_write().is_err() {
            WRITE_FAILED.store(true, SeqCst);
        }
        t.join().unwrap();
    });
    assert!(READ_FAILED.load(SeqCst), "try_read never saw the write lock held");
    assert!(WRITE_FAILED.load(SeqCst), "try_write never saw the read lock held");
}

// A thread's second read guard is counted: dropping it leaves the first one
// holding the lock, so a writer cannot enter while it is live.
#[test]
fn recursive_read_guard_keeps_the_lock() {
    loom::model(|| {
        let l = Arc::new(RwLock::new(0u32));
        let g1 = l.read().unwrap();
        drop(l.read().unwrap());
        let l2 = l.clone();
        let t = thread::spawn(move || *l2.write().unwrap() += 1);
        thread::yield_now();
        assert!(l.try_write().is_err());
        let _v = *g1;
        drop(g1);
        t.join().unwrap();
    });
}
