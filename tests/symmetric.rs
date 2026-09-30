//! `thread::symmetric` walks one representative per relabeling of a group,
//! and only of a group: whatever it prunes must be reachable by relabeling.

use std::collections::BTreeSet;
use std::sync::{Arc, Mutex};

use loom::sync::atomic::{AtomicUsize, Ordering::SeqCst};
use loom::thread;

fn outcomes<F>(bound: Option<usize>, f: F) -> BTreeSet<usize>
where
    F: Fn() -> usize + Send + Sync + 'static,
{
    let seen: Arc<Mutex<BTreeSet<usize>>> = Default::default();
    let out = seen.clone();
    let mut builder = loom::model::Builder::new();
    builder.threads = 1;
    builder.preemption_bound = bound;
    builder.check(move || {
        let r = f();
        out.lock().unwrap().insert(r);
    });
    let seen = seen.lock().unwrap().clone();
    seen
}

/// Two `symmetric` calls in one model are two groups: the second group's
/// first transition is not pinned behind the first group's. g loading x
/// before f stores it needs no preemption beyond the one plain spawns pay.
#[test]
fn two_groups_pin_independently() {
    let model = |symmetric: bool| {
        move || {
            let x = Arc::new(AtomicUsize::new(0));
            let r = Arc::new(AtomicUsize::new(9));
            let (x1, x2, r2) = (x.clone(), x.clone(), r.clone());
            let f = move || x1.store(1, SeqCst);
            let g = move || r2.store(x2.load(SeqCst), SeqCst);
            let handles = if symmetric {
                let mut handles = thread::symmetric(1, f);
                handles.extend(thread::symmetric(1, g));
                handles
            } else {
                vec![thread::spawn(f), thread::spawn(g)]
            };
            for handle in handles {
                handle.join().unwrap();
            }
            r.load(SeqCst)
        }
    };

    for bound in [None, Some(0), Some(1), Some(2)] {
        assert_eq!(
            outcomes(bound, model(false)),
            outcomes(bound, model(true)),
            "bound = {bound:?}"
        );
    }
}
