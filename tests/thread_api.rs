#![deny(warnings, rust_2018_idioms)]
use loom::sync::mpsc::channel;
use loom::thread;

#[test]
fn initial_thread() {
    loom::model(|| {
        thread::current().id(); // can call id()
        assert_eq!(None, thread::current().name());
    });
}

#[test]
fn many_joins() {
    loom::model(|| {
        let mut handles = vec![];
        let mutex = loom::sync::Arc::new(loom::sync::Mutex::new(()));
        let lock = mutex.lock().unwrap();

        for _ in 1..3 {
            let mutex = mutex.clone();
            handles.push(thread::spawn(move || {
                mutex.lock().unwrap();
            }));
        }

        std::mem::drop(lock);

        for handle in handles.into_iter() {
            let _ = handle.join();
        }
    })
}

#[test]
fn alt_join() {
    loom::model(|| {
        use loom::sync::{Arc, Mutex};

        let arcmut: Arc<Mutex<Option<thread::JoinHandle<()>>>> = Arc::new(Mutex::new(None));
        let lock = arcmut.lock().unwrap();

        let arcmut2 = arcmut.clone();

        let th1 = thread::spawn(|| {});
        let th2 = thread::spawn(move || {
            arcmut2.lock().unwrap();
            let _ = th1.join();
        });
        let th3 = thread::spawn(move || {});
        std::mem::drop(lock);
        let _ = th3.join();
        let _ = th2.join();
    })
}

#[test]
fn threads_have_unique_ids() {
    loom::model(|| {
        let (tx, rx) = channel();
        let th1 = thread::spawn(move || tx.send(thread::current().id()));
        let thread_id_1 = rx.recv().unwrap();

        assert_eq!(th1.thread().id(), thread_id_1);
        assert_ne!(thread::current().id(), thread_id_1);
        let _ = th1.join();

        let (tx, rx) = channel();
        let th2 = thread::spawn(move || tx.send(thread::current().id()));
        let thread_id_2 = rx.recv().unwrap();
        assert_eq!(th2.thread().id(), thread_id_2);
        assert_ne!(thread::current().id(), thread_id_2);
        assert_ne!(thread_id_1, thread_id_2);
        let _ = th2.join();
    })
}

#[test]
fn thread_names() {
    loom::model(|| {
        let (tx, rx) = channel();
        let th = thread::spawn(move || tx.send(thread::current().name().map(|s| s.to_string())));
        assert_eq!(None, rx.recv().unwrap());
        assert_eq!(None, th.thread().name());
        let _ = th.join();

        let (tx, rx) = channel();
        let th = thread::Builder::new()
            .spawn(move || tx.send(thread::current().name().map(|s| s.to_string())))
            .unwrap();
        assert_eq!(None, rx.recv().unwrap());
        assert_eq!(None, th.thread().name());
        let _ = th.join();

        let (tx, rx) = channel();
        let th = thread::Builder::new()
            .name("foobar".to_string())
            .spawn(move || tx.send(thread::current().name().map(|s| s.to_string())))
            .unwrap();
        assert_eq!(Some("foobar".to_string()), rx.recv().unwrap());
        assert_eq!(Some("foobar"), th.thread().name());

        let _ = th.join();
    })
}

#[test]
fn thread_stack_size() {
    const STACK_SIZE: usize = 1 << 16;
    loom::model(|| {
        let body = || {
            // Allocate a large array on the stack.
            std::hint::black_box(&mut [0usize; STACK_SIZE]);
        };
        thread::Builder::new()
            .stack_size(
                // Include space for function calls in addition to the array.
                2 * STACK_SIZE * std::mem::size_of::<usize>(),
            )
            .spawn(body)
            .unwrap()
            .join()
            .unwrap()
    })
}

#[test]
fn park_unpark_loom() {
    loom::model(|| {
        println!("unpark");
        thread::current().unpark();
        println!("park");
        thread::park();
        println!("it did not deadlock");
    });
}

#[test]
fn park_unpark_std() {
    println!("unpark");
    std::thread::current().unpark();
    println!("park");
    std::thread::park();
    println!("it did not deadlock");
}

#[test]
fn is_finished_races_the_thread_and_join_still_waits() {
    use std::collections::HashSet;
    use std::sync::{Arc as StdArc, Mutex as StdMutex};
    let seen = StdArc::new(StdMutex::new(HashSet::new()));
    let seen_ = seen.clone();
    loom::model(move || {
        let t = loom::thread::spawn(|| 7);
        let finished = t.is_finished();
        assert_eq!(t.join().unwrap(), 7);
        seen_.lock().unwrap().insert(finished);
    });
    assert_eq!(*seen.lock().unwrap(), HashSet::from([false, true]));
}

// `is_finished` turns true once the main function returns, which is before the
// thread's TLS destructors run.
#[test]
fn is_finished_before_tls_destructors() {
    use loom::sync::{Arc, Mutex};
    use std::sync::atomic::{AtomicBool, Ordering::SeqCst};

    struct Dtor(Arc<Mutex<bool>>);
    impl Drop for Dtor {
        fn drop(&mut self) {
            *self.0.lock().unwrap() = true;
        }
    }
    loom::thread_local! {
        static SLOT: std::cell::RefCell<Option<Dtor>> = std::cell::RefCell::new(None);
    }
    static EARLY: AtomicBool = AtomicBool::new(false);

    loom::model(|| {
        let ran = Arc::new(Mutex::new(false));
        let ran2 = ran.clone();
        let t = loom::thread::spawn(move || SLOT.with(|s| *s.borrow_mut() = Some(Dtor(ran2))));
        if t.is_finished() && !*ran.lock().unwrap() {
            EARLY.store(true, SeqCst);
        }
        t.join().unwrap();
        assert!(*ran.lock().unwrap());
    });
    assert!(EARLY.load(SeqCst), "is_finished never preceded the TLS destructor");
}

#[test]
fn scoped_is_finished_races_the_thread() {
    use std::collections::HashSet;
    use std::sync::Mutex as StdMutex;

    static SEEN: StdMutex<Option<HashSet<bool>>> = StdMutex::new(None);
    loom::model(|| {
        let finished = loom::thread::scope(|s| {
            let t = s.spawn(|| 7);
            let finished = t.is_finished();
            assert_eq!(t.join().unwrap(), 7);
            finished
        });
        SEEN.lock().unwrap().get_or_insert_with(HashSet::new).insert(finished);
    });
    assert_eq!(SEEN.lock().unwrap().take().unwrap(), HashSet::from([false, true]));
}
