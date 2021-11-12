//! This module defines an index type that can be invalid, although it has the same size as
//! usize. This allows collections of `usize` integers to be reinterpreted as collections of
//! `Index` types.
//! For indexing into mesh topologies use types defined in the `mesh::topology` module.
use std::fmt;
use std::ops::{Add, AddAssign, Div, Mul, Rem, Sub};

mod checked;

pub use self::checked::*;

/// A possibly invalid unsigned index.
/// The maximum `usize` integer represents an invalid index.
/// This index type is ideal for storage.
/// Overflow is not handled by this type. Instead we rely on Rust's internal overflow panics during
/// debug builds.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[repr(transparent)]
pub struct Index(usize);

// SAFETY: Index is transparent and usize is Pod and Zeroable.
unsafe impl bytemuck::Pod for Index {}
unsafe impl bytemuck::Zeroable for Index {}

impl Index {
    /// Invalid index instance.
    pub const INVALID: Index = Index(std::usize::MAX);

    /// Create a valid index from a usize type. This constructor does the necessary check
    /// for debug builds only.
    #[inline]
    pub fn new(i: usize) -> Index {
        debug_assert!(Index::fits(i));
        Index(i)
    }

    /// Convert this `Index` into `Option<usize>`, which is a larger struct.
    #[inline]
    pub fn into_option(self) -> Option<usize> {
        self.into()
    }

    /// Manipulate the inner representation of the index. This method avoids the additional check
    /// used in `map`. Use this to opt out of automatic index checking.
    #[inline]
    pub fn map_inner<F: FnOnce(usize) -> usize>(self, f: F) -> Index {
        Index::new(f(self.0))
    }

    /// Checked `and_then` over inner index. This allows operations on valid indices only.
    #[inline]
    pub fn and_then<F: FnOnce(usize) -> Index>(self, f: F) -> Index {
        if self.is_valid() {
            f(self.0 as usize)
        } else {
            self
        }
    }

    /// Apply a function to the inner `usize` index. The index remains unchanged if invalid.
    #[inline]
    pub fn apply<F: FnOnce(&mut usize)>(&mut self, f: F) {
        if self.is_valid() {
            f(&mut self.0);
        }
    }

    /// Apply a function to the inner `usize` index.
    #[inline]
    pub fn if_valid<F: FnOnce(usize)>(self, f: F) {
        if self.is_valid() {
            f(self.0);
        }
    }

    /// Get the raw `usize` representation of this Index. This may be useful for performance
    /// critical code or debugging.
    #[inline]
    pub fn into_inner(self) -> usize {
        self.0
    }

    /// Check that the given index fits inside the internal Index representation. This is used
    /// internally to check various conversions and arithmetics for debug builds.
    #[inline]
    fn fits(i: usize) -> bool {
        i != std::usize::MAX
    }
}

macro_rules! impl_from_unsigned {
    ($u:ty) => {
        /// Create a valid index from a `usize` type. This converter does the necessary bounds
        /// check for debug builds only.
        impl From<$u> for Index {
            #[inline]
            fn from(i: $u) -> Self {
                Index::new(i as usize)
            }
        }
    };
}

impl_from_unsigned!(usize);
impl_from_unsigned!(u64);
impl_from_unsigned!(u32);
impl_from_unsigned!(u16);
impl_from_unsigned!(u8);

macro_rules! impl_from_signed {
    ($i:ty) => {
        /// Create an index from a signed integer type. If the given argument is negative, the
        /// created index will be invalid.
        impl From<$i> for Index {
            #[inline]
            fn from(i: $i) -> Self {
                if i < 0 {
                    Index::invalid()
                } else {
                    Index(i as usize)
                }
            }
        }
    };
}

impl_from_signed!(isize);
impl_from_signed!(i64);
impl_from_signed!(i32);
impl_from_signed!(i16);
impl_from_signed!(i8);

impl From<Index> for Option<usize> {
    #[inline]
    fn from(val: Index) -> Self {
        if val.is_valid() {
            Some(val.0 as usize)
        } else {
            None
        }
    }
}

impl From<Option<usize>> for Index {
    #[inline]
    fn from(i: Option<usize>) -> Index {
        match i {
            Some(i) => Index::new(i),
            None => Index::INVALID,
        }
    }
}

impl IntoIterator for Index {
    type Item = usize;
    type IntoIter = std::option::IntoIter<usize>;
    fn into_iter(self) -> Self::IntoIter {
        self.into_option().into_iter()
    }
}

impl Add<usize> for Index {
    type Output = Index;

    #[inline]
    fn add(self, rhs: usize) -> Index {
        self.map(|x| x + rhs)
    }
}

impl AddAssign<usize> for Index {
    #[inline]
    fn add_assign(&mut self, rhs: usize) {
        self.apply(|x| *x += rhs)
    }
}

impl Add<Index> for usize {
    type Output = Index;

    #[inline]
    fn add(self, rhs: Index) -> Index {
        rhs + self
    }
}

impl Add for Index {
    type Output = Index;

    #[inline]
    fn add(self, rhs: Index) -> Index {
        // Note: add with overflow is checked by Rust for debug builds.
        self.and_then(|x| rhs.map(|y| x + y))
    }
}

impl Sub<usize> for Index {
    type Output = Index;

    #[inline]
    fn sub(self, rhs: usize) -> Index {
        self.map(|x| x - rhs)
    }
}

impl Sub<Index> for usize {
    type Output = Index;

    #[inline]
    fn sub(self, rhs: Index) -> Index {
        rhs.map(|x| self - x)
    }
}

impl Sub for Index {
    type Output = Index;

    #[inline]
    fn sub(self, rhs: Index) -> Index {
        // Note: subtract with overflow is checked by Rust for debug builds.
        self.and_then(|x| rhs.map(|y| x - y))
    }
}

impl Mul<usize> for Index {
    type Output = Index;

    #[inline]
    fn mul(self, rhs: usize) -> Index {
        self.map(|x| x * rhs)
    }
}

impl Mul<Index> for usize {
    type Output = Index;

    #[inline]
    fn mul(self, rhs: Index) -> Index {
        rhs * self
    }
}

// It often makes sense to divide an index by a non-zero integer
impl Div<usize> for Index {
    type Output = Index;

    #[inline]
    fn div(self, rhs: usize) -> Index {
        if rhs != 0 {
            self.map(|x| x / rhs)
        } else {
            Index::invalid()
        }
    }
}

impl Rem<usize> for Index {
    type Output = Index;

    #[inline]
    fn rem(self, rhs: usize) -> Index {
        if rhs != 0 {
            self.map(|x| x % rhs)
        } else {
            Index::invalid()
        }
    }
}

impl PartialEq<usize> for Index {
    #[inline]
    fn eq(&self, other: &usize) -> bool {
        self.map_or(false, |x| x == *other)
    }
}

impl PartialOrd<usize> for Index {
    #[inline]
    fn partial_cmp(&self, other: &usize) -> Option<std::cmp::Ordering> {
        self.map_or(None, |x| x.partial_cmp(other))
    }
}

impl Default for Index {
    fn default() -> Self {
        Self::invalid()
    }
}

impl fmt::Display for Index {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

// Since we can't use a generic parameter when implementing traits for structs from
// other packages, we have to use a macro here and anticipate the types of things
// would need to index.
macro_rules! impl_index_for {
    (__impl, $value_type:ty, $idx_type:ty) => {
        impl ::std::ops::Index<$idx_type> for Vec<$value_type> {
            type Output = $value_type;
            fn index(&self, index: $idx_type) -> &$value_type {
                &self[usize::from(index)]
            }
        }
        impl ::std::ops::IndexMut<$idx_type> for Vec<$value_type> {
            fn index_mut(&mut self, index: $idx_type) -> &mut $value_type {
                &mut self[usize::from(index)]
            }
        }
    };
    (Vec<$value_type:ty> by $($idx_type:ty),+) => {
        $(impl_index_for!(__impl, $value_type, $idx_type);)*
    };
    // If the collection contains a generic type. We have to wrap the Vec into a custom collection
    // type. The following rules will do this for us automatically
    (__impl_newtype, $collection_type:ident (Vec<$value_type:ty>) by $idx_type:ty) => {
        impl ::std::ops::Index<$idx_type> for $collection_type {
            type Output = $value_type;
            fn index(&self, index: $idx_type) -> &$value_type {
                &self.0[usize::from(index)]
            }
        }
        impl ::std::ops::IndexMut<$idx_type> for $collection_type {
            fn index_mut(&mut self, index: $idx_type) -> &mut $value_type {
                &mut self.0[usize::from(index)]
            }
        }
    };
    (__impl_newtype, $collection_type:ident (Vec<$value_type:ty>) by $idx_type:ty
     where $t:ident: $($traits:tt)*) => {
        impl<$t: $($traits)*> ::std::ops::Index<$idx_type> for $collection_type<$t> {
            type Output = $value_type;
            fn index(&self, index: $idx_type) -> &$value_type {
                &self.0[usize::from(index)]
            }
        }
        impl<$t: $($traits)*> ::std::ops::IndexMut<$idx_type> for $collection_type<$t> {
            fn index_mut(&mut self, index: $idx_type) -> &mut $value_type {
                &mut self.0[usize::from(index)]
            }
        }
    };
    // Custom dynamically sized collection with a generic parameter.
    ($collection_type:ident (Vec<$value_type:ty>) by $($idx_type:ty),+
     where $t:ident: $($traits:tt)*) => {
        #[derive(Clone, Debug, PartialEq, NewCollectionType)]
        #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
        pub struct $collection_type<$t: $($traits)*>(Vec<$value_type>);

        $(impl_index_for!(__impl_newtype, $collection_type(Vec<$value_type>) by $idx_type
                          where $t: $traits);)*
    };
    // Custom dynamically sized collection, no generics
    ($collection_type:ident (Vec<$value_type:ty>) by $($idx_type:ty),+) => {
        #[derive(Clone, Debug, PartialEq, NewCollectionType)]
        #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
        pub struct $collection_type(Vec<$value_type>);

        $(impl_index_for!(__impl_newtype, $collection_type(Vec<$value_type>) by $idx_type);)*
    };
    // Custom statically sized collection, no generics
    ($collection_type:ident ([$value_type:ty; $n:expr]) by $($idx_type:ty),+) => {
        #[derive(Clone, Debug, PartialEq, NewCollectionType)]
        #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
        pub struct $collection_type([$value_type;$n]);

        $(impl_index_for!(__impl_newtype, $collection_type(Vec<$value_type>) by $idx_type);)*
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn index_test() {
        let i = Index::new(2);
        let j = Index::new(4);
        let k = Index::invalid();
        assert_eq!(i + j, Index::new(6));
        assert_eq!(j - i, Index::new(2));
        assert_eq!(j * 3, Index::new(12));
        assert_eq!(j % 3, Index::new(1));
        assert_eq!(j / 3, Index::new(1));
        assert_eq!(i - k, Index::invalid());
        assert_eq!(i + k, Index::invalid());
        assert_eq!(k * 2, Index::invalid());
        assert_eq!(k / 2, Index::invalid());
        assert_eq!(k % 2, Index::invalid());
    }
}
