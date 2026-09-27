//! The generic name `Atomic<T>` for the modelled atomic types.

use super::*;

mod sealed {
    pub trait Sealed {}
}

/// A primitive type that has a modelled atomic counterpart.
///
/// Mirrors `core::sync::atomic::AtomicPrimitive`: implemented for exactly the
/// primitives `core` implements it for, and sealed. [`Cell`](Self::Cell) is
/// the constructed loom cell that [`Atomic<T>`] names.
pub trait AtomicPrimitive: Sized + Copy + sealed::Sealed {
    /// The loom atomic type holding a `Self`.
    type Cell;
}

/// A memory location which can be safely modified from multiple threads.
///
/// Mock of `core::sync::atomic::Atomic<T>`: `Atomic<u32>` is
/// [`AtomicU32`], `Atomic<*mut T>` is [`AtomicPtr<T>`], and so on. Unlike
/// `core`'s, this is an alias through [`AtomicPrimitive`] rather than a type
/// of its own, so a constructor call needs the type named —
/// `Atomic::<u32>::new(0)` — since inference cannot run a projection
/// backwards.
pub type Atomic<T> = <T as AtomicPrimitive>::Cell;

macro_rules! atomic_primitive {
    ($($(#[$cfg:meta])* $prim:ty => $cell:ty;)*) => {$(
        $(#[$cfg])*
        impl sealed::Sealed for $prim {}

        $(#[$cfg])*
        impl AtomicPrimitive for $prim {
            type Cell = $cell;
        }
    )*};
}

atomic_primitive! {
    bool => AtomicBool;
    u8 => AtomicU8;
    i8 => AtomicI8;
    u16 => AtomicU16;
    i16 => AtomicI16;
    u32 => AtomicU32;
    i32 => AtomicI32;
    #[cfg(target_has_atomic = "64")]
    u64 => AtomicU64;
    #[cfg(target_has_atomic = "64")]
    i64 => AtomicI64;
    u128 => AtomicU128;
    i128 => AtomicI128;
    usize => AtomicUsize;
    isize => AtomicIsize;
}

impl<T> sealed::Sealed for *mut T {}

impl<T> AtomicPrimitive for *mut T {
    type Cell = AtomicPtr<T>;
}
