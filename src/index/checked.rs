use super::*;

/// A trait that defines what it means for an index to be a checked index.
/// In other words implementers are indices that are checked for validity in some way.
pub trait CheckedIndex<T>:
    From<T>
    + Into<Option<T>>
    + Add<Output = Self>
    + Add<T, Output = Self>
    + Sub<Output = Self>
    + Sub<T, Output = Self>
    + Mul<T, Output = Self>
    + Div<T, Output = Self>
    + Rem<T, Output = Self>
{
    /// Construct an invalid index.
    fn invalid() -> Self;

    /// Check if the index is valid.
    fn is_valid(&self) -> bool;

    /// Convert the index into its underlying integer type. Panic if index is invalid.
    fn unwrap(self) -> T;

    /// Convert the index into its underlying integer type. In case of failure, report the
    /// given message.
    fn expect(self, msg: &str) -> T;

    /// Map the underlying integer type. This allows restricting operations to valid indices only.
    fn map<F: FnOnce(T) -> T>(self, f: F) -> Self;

    /// Map the underlying integer type. If the index is invalid, output the provided default.
    fn map_or<U, F: FnOnce(T) -> U>(self, default: U, f: F) -> U;

    /// Produce the raw inner type without checking.
    fn into_inner(self) -> T;
}

impl CheckedIndex<usize> for Index {
    #[inline]
    fn invalid() -> Self {
        Index::INVALID
    }

    #[inline]
    fn is_valid(&self) -> bool {
        *self != Self::INVALID
    }

    #[inline]
    fn unwrap(self) -> usize {
        self.expect("Unhandled Invalid Index.")
    }

    #[inline]
    fn expect(self, msg: &str) -> usize {
        if self.is_valid() {
            self.0
        } else {
            panic!("{}", msg)
        }
    }

    #[inline]
    fn map<F: FnOnce(usize) -> usize>(self, f: F) -> Index {
        if self.is_valid() {
            Index::new(f(self.0 as usize))
        } else {
            self
        }
    }

    #[inline]
    fn map_or<U, F: FnOnce(usize) -> U>(self, default: U, f: F) -> U {
        if self.is_valid() {
            f(self.0 as usize)
        } else {
            default
        }
    }

    #[inline]
    fn into_inner(self) -> usize {
        self.0
    }
}
