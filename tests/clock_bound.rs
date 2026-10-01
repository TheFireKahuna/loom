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
        // Spread over two locations, so no one location's store count is
        // exhausted before the thread's lane.
        let cells = [AtomicUsize::new(0), AtomicUsize::new(0)];
        for i in 0..=u16::MAX as usize {
            cells[i & 1].store(i, Relaxed);
        }
    });
}

/// A location's store count is exhausted loudly, never wrapped: a wrap would
/// alias a new store's identity with an older one's.
#[test]
#[should_panic(expected = "stores to one atomic location in one execution")]
fn location_store_exhaustion_fails_loudly() {
    let mut builder = loom::model::Builder::new();
    builder.max_branches = 1 << 20;
    builder.check(|| {
        let a = AtomicUsize::new(0);
        for i in 0..=u16::MAX as usize {
            a.store(i, Relaxed);
        }
    });
}
