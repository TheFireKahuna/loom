#![deny(warnings, rust_2018_idioms)]

use loom::sync::atomic::{AtomicUsize, Ordering::Relaxed};

/// A thread's clock lane is exhausted loudly, never wrapped: a wrap would
/// invert every happens-before test against that lane.
#[test]
#[should_panic(expected = "synchronizing operations in one execution")]
fn clock_lane_exhaustion_fails_loudly() {
    let mut builder = loom::model::Builder::new();
    builder.max_branches = 1 << 20;
    builder.check(|| {
        let a = AtomicUsize::new(0);
        for i in 0..=u16::MAX as usize {
            a.store(i, Relaxed);
        }
    });
}
