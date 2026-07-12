/// Numeric-like type can be represented by a `u128`.
///
/// Used by `Atomic` to store values. `u128` is wide enough to hold every
/// atomic payload loom models (integers up to 128 bits, pointers, and
/// booleans) without loss.
pub(crate) trait Numeric: Sized + Copy + PartialEq {
    /// Convert a value into `u128` representation
    fn into_u128(self) -> u128;

    /// Convert a `u128` representation into the value
    fn from_u128(src: u128) -> Self;
}

macro_rules! impl_num {
    ( $($t:ty),* ) => {
        $(
            impl Numeric for $t {
                fn into_u128(self) -> u128 {
                    self as u128
                }

                fn from_u128(src: u128) -> $t {
                    src as $t
                }
            }
        )*
    };
}

impl_num!(u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize);

impl<T> Numeric for *mut T {
    fn into_u128(self) -> u128 {
        self as usize as u128
    }

    fn from_u128(src: u128) -> *mut T {
        src as usize as *mut T
    }
}

impl Numeric for bool {
    fn into_u128(self) -> u128 {
        if self {
            1
        } else {
            0
        }
    }

    fn from_u128(src: u128) -> bool {
        src != 0
    }
}
