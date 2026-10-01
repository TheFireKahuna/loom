#![deny(warnings)]

//! A block the model allocates that outlives its execution is reported at
//! the execution's end, before any snapshot restore can rewind it under its
//! owner; execution-scoped state and `loom::alloc::outside` are not.

use std::cell::RefCell;
use std::sync::{Mutex as StdMutex, OnceLock};

use loom::sync::atomic::{AtomicUsize, Ordering::SeqCst};
use loom::sync::Arc;
use loom::thread;

#[global_allocator]
static ALLOC: loom::alloc::Model<std::alloc::System> = loom::alloc::Model(std::alloc::System);

fn racing_pair(f: impl Fn() + Sync + Send + 'static) {
    let f = std::sync::Arc::new(f);
    loom::model(move || {
        let n = Arc::new(AtomicUsize::new(0));
        let n2 = n.clone();
        let f2 = f.clone();
        let t = thread::spawn(move || {
            n2.fetch_add(1, SeqCst);
            f2();
        });
        n.fetch_add(1, SeqCst);
        f();
        t.join().unwrap();
    });
}

#[test]
#[should_panic(expected = "outlived its execution")]
fn a_static_grown_in_the_model_is_reported() {
    static SEEN: OnceLock<StdMutex<Vec<u64>>> = OnceLock::new();
    racing_pair(|| SEEN.get_or_init(Default::default).lock().unwrap().push(1));
}

#[test]
#[should_panic(expected = "outlived its execution")]
fn a_std_thread_local_grown_in_the_model_is_reported() {
    std::thread_local! {
        static LOG: RefCell<Vec<u64>> = const { RefCell::new(Vec::new()) };
    }
    racing_pair(|| LOG.with(|log| log.borrow_mut().push(1)));
}

#[test]
fn a_static_grown_outside_is_not_reported() {
    static SEEN: OnceLock<StdMutex<Vec<u64>>> = OnceLock::new();
    racing_pair(|| {
        loom::alloc::outside(|| SEEN.get_or_init(Default::default).lock().unwrap().push(1))
    });
}

#[test]
fn execution_scoped_state_is_not_reported() {
    loom::lazy_static! {
        static ref SHARED: loom::sync::Mutex<Vec<u64>> = loom::sync::Mutex::new(Vec::new());
    }
    loom::thread_local! {
        static LOCAL: RefCell<Vec<u64>> = RefCell::new(Vec::new());
    }
    racing_pair(|| {
        let boxed = Box::new([0u64; 64]);
        let mut v: Vec<Box<u64>> = (0..8).map(Box::new).collect();
        v.push(boxed.iter().sum::<u64>().into());
        SHARED.lock().unwrap().push(v.len() as u64);
        LOCAL.with(|l| l.borrow_mut().extend(v.iter().map(|b| **b)));
    });
}
