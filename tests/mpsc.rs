use loom::sync::mpsc::channel;
use loom::thread;

#[test]
fn basic_sequential_usage() {
    loom::model(|| {
        let (s, r) = channel();
        s.send(5).unwrap();
        let val = r.recv().unwrap();
        assert_eq!(val, 5);
    });
}

#[test]
fn basic_parallel_usage() {
    loom::model(|| {
        let (s, r) = channel();
        thread::spawn(move || {
            s.send(5).unwrap();
        });
        let val = r.recv().unwrap();
        assert_eq!(val, 5);
    });
}

#[test]
fn commutative_senders() {
    loom::model(|| {
        let (s, r) = channel();
        let s2 = s.clone();
        thread::spawn(move || {
            s.send(5).unwrap();
        });
        thread::spawn(move || {
            s2.send(6).unwrap();
        });
        let mut val = r.recv().unwrap();
        val += r.recv().unwrap();
        assert_eq!(val, 11);
    });
}

fn ignore_result<A, B>(_: Result<A, B>) {}

#[test]
#[should_panic]
fn non_commutative_senders1() {
    loom::model(|| {
        let (s, r) = channel();
        let s2 = s.clone();
        thread::spawn(move || {
            ignore_result(s.send(5));
        });
        thread::spawn(move || {
            ignore_result(s2.send(6));
        });
        let val = r.recv().unwrap();
        assert_eq!(val, 5);
        ignore_result(r.recv());
    });
}

#[test]
#[should_panic]
fn non_commutative_senders2() {
    loom::model(|| {
        let (s, r) = channel();
        let s2 = s.clone();
        thread::spawn(move || {
            ignore_result(s.send(5));
        });
        thread::spawn(move || {
            ignore_result(s2.send(6));
        });
        let val = r.recv().unwrap();
        assert_eq!(val, 6);
        ignore_result(r.recv());
    });
}

#[test]
fn drop_receiver() {
    loom::model(|| {
        let (s, r) = channel();
        s.send(1).unwrap();
        s.send(2).unwrap();
        assert_eq!(r.recv().unwrap(), 1);
    });
}

// `recv` on an empty channel whose senders are all gone fails rather than
// blocking; a message sent before the last sender dropped is still received.
#[test]
fn recv_fails_once_every_sender_is_gone() {
    loom::model(|| {
        let (s, r) = channel::<u32>();
        let s2 = s.clone();
        let t = thread::spawn(move || {
            s2.send(1).unwrap();
            drop(s2);
        });
        drop(s);
        assert_eq!(r.recv(), Ok(1));
        assert!(r.recv().is_err());
        assert_eq!(r.try_recv(), Err(std::sync::mpsc::TryRecvError::Disconnected));
        t.join().unwrap();
    });
}

// A send after the receiver is dropped returns the message and leaves nothing
// in the channel to leak.
#[test]
fn send_fails_once_the_receiver_is_gone() {
    use std::sync::atomic::{AtomicBool, Ordering::SeqCst};

    static FAILED: AtomicBool = AtomicBool::new(false);
    loom::model(|| {
        let (s, r) = channel::<u32>();
        let t = thread::spawn(move || {
            if let Err(e) = s.send(1) {
                assert_eq!(e.0, 1);
                FAILED.store(true, SeqCst);
            }
        });
        drop(r);
        t.join().unwrap();
    });
    assert!(FAILED.load(SeqCst), "no send ever followed the receiver's drop");
}

// `try_recv` racing a send may see the message or find the channel empty.
#[test]
fn try_recv_races_a_send() {
    use std::sync::Mutex;

    static SEEN: Mutex<Vec<bool>> = Mutex::new(Vec::new());
    loom::model(|| {
        let (s, r) = channel::<u32>();
        let t = thread::spawn(move || s.send(1).unwrap());
        let seen = r.try_recv().is_ok();
        SEEN.lock().unwrap().push(seen);
        t.join().unwrap();
    });
    let seen = SEEN.lock().unwrap();
    assert!(seen.contains(&true), "try_recv never saw the racing send");
    assert!(seen.contains(&false), "try_recv never missed the racing send");
}

// `try_recv` racing the last sender's drop may report the channel empty or
// disconnected.
#[test]
fn try_recv_races_the_last_sender_drop() {
    use std::sync::mpsc::TryRecvError;
    use std::sync::Mutex;

    static SEEN: Mutex<Vec<TryRecvError>> = Mutex::new(Vec::new());
    loom::model(|| {
        let (s, r) = channel::<u32>();
        let t = thread::spawn(move || drop(s));
        let seen = r.try_recv().unwrap_err();
        SEEN.lock().unwrap().push(seen);
        t.join().unwrap();
    });
    let seen = SEEN.lock().unwrap();
    assert!(seen.contains(&TryRecvError::Empty));
    assert!(seen.contains(&TryRecvError::Disconnected));
}
