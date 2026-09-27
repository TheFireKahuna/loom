#![deny(warnings)]

//! `thread::scope` and the free thread functions. A scoped thread's effects
//! must happen-before the scope's return; a tracked access witnesses it.

use std::time::Duration;

use loom::cell::UnsafeCell;
use loom::sync::atomic::{AtomicUsize, Ordering::*};
use loom::thread;

/// A tracked cell shared across threads, so that loom — not the type system —
/// judges whether its accesses are ordered.
struct Racy<T>(UnsafeCell<T>);

// SAFETY: every access goes through loom's tracking, which reports the
// unordered ones this impl lets the tests attempt.
unsafe impl<T: Send> Sync for Racy<T> {}

impl<T> std::ops::Deref for Racy<T> {
    type Target = UnsafeCell<T>;
    fn deref(&self) -> &UnsafeCell<T> {
        &self.0
    }
}

fn racy<T>(v: T) -> Racy<T> {
    Racy(UnsafeCell::new(v))
}

#[test]
fn scope_joins_every_thread_before_returning() {
    loom::model(|| {
        let mut left = 0;
        let right = racy(0);
        let counter = AtomicUsize::new(0);

        let joined = thread::scope(|s| {
            let h = s.spawn(|| {
                left += 1;
                counter.fetch_add(1, Relaxed);
                7
            });
            s.spawn(|| {
                right.with_mut(|p| unsafe { *p = 1 });
                counter.fetch_add(1, Relaxed);
            });
            h.join().unwrap()
        });

        // The unjoined thread's write happens-before the scope's return: an
        // unsynchronized read here would be reported.
        assert_eq!(joined, 7);
        assert_eq!(left, 1);
        assert_eq!(right.with(|p| unsafe { *p }), 1);
        assert_eq!(counter.load(Relaxed), 2);
    });
}

#[test]
fn scope_waits_for_nested_spawns() {
    loom::model(|| {
        let cell = racy(0);
        thread::scope(|s| {
            s.spawn(|| {
                s.spawn(|| cell.with_mut(|p| unsafe { *p = 1 }));
            });
        });
        assert_eq!(cell.with(|p| unsafe { *p }), 1);
    });
}

#[test]
fn scoped_builder_names_the_thread() {
    loom::model(|| {
        thread::scope(|s| {
            let h = thread::Builder::new()
                .name("worker".into())
                .spawn_scoped(s, || thread::current().name().map(str::to_owned))
                .unwrap();
            assert_eq!(h.thread().name(), Some("worker"));
            assert_eq!(h.join().unwrap().as_deref(), Some("worker"));
        });
    });
}

#[test]
#[should_panic(expected = "scoped boom")]
fn scoped_thread_panic_fails_the_model() {
    loom::model(|| {
        thread::scope(|s| {
            s.spawn(|| panic!("scoped boom"));
        });
    });
}

#[test]
fn park_timeout_resolves_without_an_unparker() {
    loom::model(|| {
        thread::park_timeout(Duration::from_millis(1));
        thread::sleep(Duration::from_millis(1));
    });
}

#[test]
fn current_id_matches_current() {
    loom::model(|| {
        let t = thread::spawn(|| thread::current_id() == thread::current().id());
        assert!(t.join().unwrap());
        assert_eq!(thread::current_id(), thread::current().id());
    });
}

