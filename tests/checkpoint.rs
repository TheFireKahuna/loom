#![cfg(feature = "checkpoint")]

//! A checkpoint resumes the walk it was written by, under that walk's
//! settings: a resumed path explores under the Builder's preemption bound
//! or not at all.

use loom::model::Builder;
use loom::sync::atomic::AtomicUsize;
use loom::sync::Arc;
use loom::thread;

use std::panic::{self, AssertUnwindSafe};
use std::path::PathBuf;
use std::sync::atomic::Ordering::Relaxed;

fn model() {
    let x = Arc::new(AtomicUsize::new(0));

    let ths: Vec<_> = (0..2)
        .map(|_| {
            let x = x.clone();
            thread::spawn(move || {
                for _ in 0..3 {
                    x.fetch_add(1, Relaxed);
                }
            })
        })
        .collect();

    for _ in 0..3 {
        x.fetch_add(1, Relaxed);
    }

    for th in ths {
        th.join().unwrap();
    }
}

fn checkpoint_file(name: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!("loom-checkpoint-{}-{name}.json", std::process::id()));
    let _ = std::fs::remove_file(&path);
    path
}

/// Write a checkpoint from a walk bounded at `bound`.
fn write_checkpoint(path: &PathBuf, bound: Option<usize>) {
    let mut builder = Builder::new();
    builder.checkpoint_file = Some(path.clone());
    builder.checkpoint_interval = 5;
    builder.max_permutations = Some(7);
    builder.preemption_bound = bound;
    builder.check(model);
    assert!(path.exists(), "no checkpoint written");
}

#[test]
fn a_checkpoint_resumes_under_the_bound_it_was_written_with() {
    let path = checkpoint_file("same");
    write_checkpoint(&path, Some(2));

    let mut builder = Builder::new();
    builder.checkpoint_file = Some(path.clone());
    builder.preemption_bound = Some(2);
    let stats = builder.check(model);
    let _ = std::fs::remove_file(&path);

    assert!(stats.executions > 0);
}

#[test]
fn a_checkpoint_written_under_another_bound_is_refused() {
    let path = checkpoint_file("other");
    write_checkpoint(&path, Some(2));

    let mut builder = Builder::new();
    builder.checkpoint_file = Some(path.clone());
    builder.preemption_bound = None;
    let result = panic::catch_unwind(AssertUnwindSafe(|| builder.check(model)));
    let _ = std::fs::remove_file(&path);

    let payload = result.expect_err("resumed a bounded checkpoint without a bound");
    let message = payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_default();
    assert!(message.contains("preemption_bound"), "{message}");
}
