use std::any::{Any, TypeId};
use std::marker::PhantomData;
use std::slice;

use dync::{dync_mod, from_dyn, into_dyn, BoxValue, Slice, SliceMut, SmallValue, VecDyn};
//use fnv::FnvHashSet as HashSet;
use hashbrown::HashSet;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::index::*;
use crate::mesh::topology::*;

use super::Error;

pub use std::sync::Arc as Irc;

/// A module defining traits for values stored at a mesh attribute.
#[allow(missing_docs)]
#[dync_mod]
mod value_traits {
    /// A basic value that can be stored as an attribute in a mesh type.
    pub trait AttributeValue: Clone + PartialEq + std::fmt::Debug + Send + Sync + 'static {}
    impl<T> AttributeValue for T where T: Clone + PartialEq + std::fmt::Debug + Send + Sync + 'static {}

    /// A value that can be stored as an indirect attribute in a mesh type.
    ///
    /// This value is cached inside a `HashSet`, so it requires additional constraints beyond
    /// those imposed on `AttributeValue`.
    pub trait AttributeValueHash: AttributeValue + Eq + std::hash::Hash {}
    impl<T> AttributeValueHash for T where T: AttributeValue + Eq + std::hash::Hash {}
}

pub use self::value_traits::*;

/// A slice of attribute values belonging to a particular attribute.
pub type DataSlice<'a> = Slice<'a, AttributeValueVTable>;
/// A slice of attribute values belonging to a particular indirect attribute.
pub type HDataSlice<'a> = Slice<'a, AttributeValueHashVTable>;
/// A mutable slice of attribute values belonging to a particular attribute.
pub type DataSliceMut<'a> = SliceMut<'a, AttributeValueVTable>;
/// A mutable slice of attribute values belonging to a particular indirect attribute.
pub type HDataSliceMut<'a> = SliceMut<'a, AttributeValueHashVTable>;
/// A vector of values stored by a direct attribute.
pub type DataVec = VecDyn<AttributeValueVTable>;
/// A vector of values stored by an indirect attribute.
pub type HDataVec = VecDyn<AttributeValueHashVTable>;
/// An owned value stored at an attribute at some index.
pub type Value = BoxValue<AttributeValueVTable>;
/// A reference to a value stored at an attribute at some index.
pub type ValueRef<'a> = dync::ValueRef<'a, AttributeValueVTable>;
/// An reference to a value stored at an indirect attribute at some index.
pub type HValue = SmallValue<AttributeValueHashVTable>;
/// A reference to a value stored at an indirect attribute at some index.
pub type HValueRef<'a> = dync::ValueRef<'a, AttributeValueHashVTable>;
/// A mutable reference to a value stored at an indirect attribute at some index.
pub type HValueMut<'a> = dync::ValueMut<'a, AttributeValueHashVTable>;

/// A collection of indirect attribute values cached for improved memory usage.
pub type AttribValueCache = HashSet<HValue>;

// Common implementations for VecDyn data.
macro_rules! impl_data_base {
    ($vec_type:ty) => {
        /// Get the type data stored within this attribute
        #[inline]
        pub fn check<T: Any>(&self) -> Result<&Self, Error> {
            self.buf
                .check_ref::<T>()
                .map(|_| self)
                .ok_or_else(|| Error::type_mismatch_from_buf::<T, _>(&self.buf))
        }

        /// Get the mutable typed data stored within this attribute
        #[inline]
        pub fn check_mut<T: Any>(&mut self) -> Result<&mut Self, Error> {
            match self.buf.check_mut::<T>() {
                Some(_) => Ok(self),
                None => Err(Error::type_mismatch_from_buf::<T, _>(&self.buf)),
            }
        }

        /// Get the type data stored within this attribute
        #[inline]
        pub fn element_type_id(&self) -> TypeId {
            self.buf.element_type_id()
        }

        /// Get the number of bytes per element stored in this attribute.
        #[inline]
        pub fn element_size(&self) -> usize {
            self.buf.element_size()
        }

        /// Number of elements stored by this attribute. This is the same as the number of elements in
        /// the associated topology.
        #[inline]
        pub fn len(&self) -> usize {
            self.buf.len()
        }

        /// Number of bytes stored by this attribute. This is the same as the number of elements
        /// multiplied by the size of each element.
        #[inline]
        pub fn byte_len(&self) -> usize {
            self.buf.len() * self.buf.element_size()
        }

        /// Check if there are any values in this attribute.
        #[inline]
        pub fn is_empty(&self) -> bool {
            self.buf.is_empty()
        }

        /// Get a `const` reference to the `i`'th attribute value.
        ///
        /// # Safety
        ///
        /// Calling this method with an out-of-bounds index is [*undefined behavior*] even if the output is
        /// not used. For a safe alternative use the `get_ref` method.
        ///
        /// [*undefined behavior*]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
        #[inline]
        pub unsafe fn get_unchecked_ref<T: Any>(&self, i: usize) -> &T {
            self.buf.get_unchecked_ref(i)
        }

        /// Get a mutable reference to the `i`'th attribute value.
        ///
        /// # Safety
        ///
        /// Calling this method with an out-of-bounds index is [*undefined behavior*] even if the output is
        /// not used. For a safe alternative use the `get_mut` method.
        ///
        /// [*undefined behavior*]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
        #[inline]
        pub unsafe fn get_unchecked_mut<T: Any>(&mut self, i: usize) -> &mut T {
            self.buf.get_unchecked_mut(i)
        }

        /// Get a reference to the internal value vector.
        #[inline]
        pub fn data_ref(&self) -> &$vec_type {
            &self.buf
        }

        /// Get a mutable reference to the internal value vector.
        #[inline]
        pub fn data_mut(&mut self) -> &mut $vec_type {
            &mut self.buf
        }

        /// Convert this attribute into the underlying buffer. This consumes the attribute.
        #[inline]
        pub fn into_data(self) -> $vec_type {
            self.buf
        }

        /// Extend this attribute by `n` elements. Effectively, this function appends the default
        /// element `n` number of times to this attribute.
        #[inline]
        pub fn extend_by(&mut self, n: usize) {
            let Self {
                buf,
                default_element,
                ..
            } = self;
            for _ in 0..n {
                buf.push_cloned(default_element.as_ref());
            }
        }

        /// Rotates elements of this attribute in-place to the left.
        ///
        /// Rotate this attribute in-place such that the first `mid` elements of the underlying buffer
        /// move to the end while the last `self.len() - mid` elements move to the front. After
        /// calling `rotate_left`, the element previously at index `mid` will become the first element
        /// in the slice.
        #[inline]
        pub fn rotate_left(&mut self, mid: usize) {
            self.buf.rotate_left(mid);
        }

        /// Rotates elements of this attribute in-place to the right.
        ///
        /// Rotate this attribute in-place such that the first `self.len() - k` elements of the
        /// underlying buffer move to the end while the last `k` elements move to the front. After
        /// calling `rotate_right`, the element previously at index `self.len() - k` will become the
        /// first element in the slice.
        #[inline]
        pub fn rotate_right(&mut self, k: usize) {
            self.buf.rotate_right(k);
        }
    };
}

/// Mesh attribute data.
///
/// This stores unique values shared among mesh elements via smart pointers.
/// This type doesn't store the location of the attribute.
#[derive(Clone, Debug, PartialEq)]
pub struct IndirectData {
    buf: HDataVec,
    default_element: HValue,
}

impl IndirectData {
    impl_data_base!(HDataVec);

    /// Construct an attribute with a given size.
    pub fn with_size<T: AttributeValueHash>(n: usize, def: T) -> Self {
        let default_element = Irc::new(def);
        IndirectData {
            buf: HDataVec::with_size(n, Irc::clone(&default_element)),
            default_element: HValue::new(default_element),
        }
    }

    /// Get the value pointer from the value set corresponding to the given value and insert it in
    /// to the values set if it doesn't already exist.
    fn get_or_insert<T: AttributeValueHash>(set: &mut AttribValueCache, elem: T) -> Irc<T> {
        let elem = HValue::new(Irc::new(elem));
        if let Some(elem) = set.get(&elem) {
            Irc::clone(elem.as_ref().downcast().unwrap())
        } else {
            assert!(set.insert(elem.clone()));
            elem.downcast().unwrap()
        }
    }

    /// Construct an attribute from a given `Vec<T>` of data.
    pub fn from_vec<T: AttributeValueHash + Default>(
        vec: Vec<T>,
        cache: &mut AttribValueCache,
    ) -> Self {
        let default_element = Irc::new(T::default());
        let buf: Vec<_> = vec
            .into_iter()
            .map(|elem| Self::get_or_insert(cache, elem))
            .collect();

        IndirectData {
            buf: HDataVec::from_vec(buf),
            default_element: HValue::new(default_element),
        }
    }

    /// Construct an attribute from a given slice of data, by copying the data.
    #[inline]
    pub fn from_slice<T: AttributeValueHash + Default>(
        buf: &[T],
        cache: &mut AttribValueCache,
    ) -> Self {
        Self::from_vec(buf.to_vec(), cache)
    }

    /// Construct a new empty attribute with the same values and default element as `self`.
    pub fn duplicate_empty(&self) -> Self {
        IndirectData {
            buf: HDataVec::with_type_from(&self.buf),
            default_element: self.default_element.clone(),
        }
    }

    /// Construct a new attribute with the same values and default element as `self`.
    pub fn duplicate_with(
        &self,
        dup_data: impl FnOnce(HDataSlice) -> VecDyn<dyn HasAttributeValue>,
    ) -> Self {
        IndirectData {
            buf: from_dyn![VecDyn<dyn HasAttributeValue as AttributeValueHashVTable>](dup_data(
                self.data_ref().as_slice(),
            )),
            default_element: self.default_element.clone(),
        }
    }

    /// Construct a new attribute with the same values and default element as `self`.
    ///
    /// The attribute is first initialized with the default value by allocating `len` default
    /// elements. Then the newly created buffer is expected to be modified by the `init` function.
    /// The output `HDataVec` must be composed of values from the original `HDataVec` or the
    /// default element.
    ///
    /// The `init` function is only allowed to clone data from the second argument into the first.
    /// Adding new data will cause this attribute to go out of sync with the cache.
    pub fn duplicate_with_len(
        &self,
        len: usize,
        init: impl FnOnce(HDataSliceMut, HDataSlice),
    ) -> Self {
        let mut attrib = self.duplicate_empty();
        attrib.extend_by(len);
        init(attrib.data_mut().as_mut_slice(), self.data_ref().as_slice());
        attrib
    }

    /// Convert the data stored by this attribute into a vector of the same size.
    #[inline]
    pub fn clone_into_vec<T: AttributeValueHash>(&self) -> Result<Vec<T>, Error> {
        let result = Vec::with_capacity(self.len());
        self.buf
            .iter_as::<Irc<T>>()
            .unwrap()
            .fold(Some(result), |mut acc, rc| {
                if let Some(acc) = acc.as_mut() {
                    acc.push((**rc).clone());
                }
                acc
            })
            .ok_or_else(|| Error::type_mismatch_from_buf::<Irc<T>, _>(&self.buf))
    }

    /// Produce an iterator over the underlying data elements.
    #[inline]
    pub fn iter<T: Any>(&self) -> Result<impl Iterator<Item = &T>, Error> {
        self.buf
            .iter_as::<Irc<T>>()
            .map(|iter| iter.map(|rc| &**rc))
            .ok_or_else(|| Error::type_mismatch_from_buf::<Irc<T>, _>(&self.buf))
    }

    /// Iterate over all the value in this attribute and update them as needed.
    ///
    /// This function takes a closure which takes an index and a smart pointer to the stored value and
    /// produces an optional new value. The new value is then used to update the attribute using
    /// the provided cache.
    pub fn update_with<T, F>(
        &mut self,
        mut f: F,
        cache: &mut AttribValueCache,
    ) -> Result<&mut Self, Error>
    where
        T: AttributeValueHash,
        F: FnMut(usize, &Irc<T>) -> Option<Irc<T>>,
    {
        let id = self.buf.element_type_id();
        for (i, val) in self.buf.iter_mut().enumerate() {
            let rc = val
                .downcast::<Irc<T>>()
                .ok_or_else(|| Error::type_mismatch_id::<Irc<T>>(id))?;
            if let Some(new_rc) = f(i, &*rc) {
                let new_value = HValue::new(new_rc);
                if let Some(existing) = cache.get(&new_value) {
                    HValueMut::new(rc).clone_from_other(existing.as_ref())?;
                } else if new_value == self.default_element {
                    HValueMut::new(rc).clone_from_other(self.default_element.as_ref())?;
                } else {
                    HValueMut::new(rc).clone_from_other(new_value.as_ref())?;
                    assert!(cache.insert(new_value));
                }
            }
        }
        Ok(self)
    }

    /// Set the value of a particular element.
    pub fn set_at<'a, T>(
        &'a mut self,
        i: usize,
        new_value: T,
        cache: &'a mut AttribValueCache,
    ) -> Result<&'a mut Self, Error>
    where
        T: AttributeValueHash,
    {
        self.set_value_at(i, &HValue::new(Irc::new(new_value)), cache)
    }

    /// Set the value of a particular element.
    pub fn set_value_at<'a>(
        &'a mut self,
        i: usize,
        new_value: &HValue,
        cache: &'a mut AttribValueCache,
    ) -> Result<&'a mut Self, Error> {
        let mut value_out = self.buf.get_mut(i);
        if let Some(existing) = cache.get(new_value) {
            value_out.clone_from_other(existing.as_ref())?;
        } else if new_value == &self.default_element {
            value_out.clone_from_other(self.default_element.as_ref())?;
        } else {
            value_out.clone_from_other(new_value.as_ref())?;
            assert!(cache.insert((*new_value).clone()));
        }
        Ok(self)
    }

    /// Push a value onto the underlying data buffer.
    pub fn push_cloned(
        &mut self,
        new_value_ref: HValueRef,
        cache: &mut AttribValueCache,
    ) -> Result<&mut Self, Error> {
        let expected = self.buf.element_type_id();
        let actual = new_value_ref.value_type_id();
        let err = || Error::TypeMismatch { expected, actual };

        let new_value = new_value_ref.clone_small_value();
        if let Some(existing) = cache.get(&new_value) {
            self.buf.push_cloned(existing.as_ref()).ok_or_else(err)?;
        } else if new_value == self.default_element {
            self.buf
                .push_cloned(self.default_element.as_ref())
                .ok_or_else(err)?;
        } else {
            self.buf.push_cloned(new_value.as_ref()).ok_or_else(err)?;
            assert!(cache.insert(new_value));
        }
        Ok(self)
    }

    /// Produce a slice to the underlying data referenced by smart pointers.
    #[inline]
    pub fn as_rc_slice<T: Any>(&self) -> Result<&[Irc<T>], Error> {
        self.buf
            .as_slice_as()
            .ok_or_else(|| Error::type_mismatch_from_buf::<T, _>(&self.buf))
    }

    /// Produce a mutable slice to the underlying data referenced by smart pointers.
    #[inline]
    pub fn as_mut_rc_slice<T: Any>(&mut self) -> Result<&mut [Irc<T>], Error> {
        let element_id = self.buf.element_type_id();
        self.buf
            .as_mut_slice_as()
            .ok_or_else(|| Error::type_mismatch_id::<Irc<T>>(element_id))
    }

    /// Get a reference to the default element.
    #[inline]
    pub fn default_element(&self) -> HValueRef {
        self.default_element.as_ref()
    }
}

/// Attribute data stored directly with each associated element.
///
/// This type doesn't store the location of the attribute.
#[derive(Clone, Debug, PartialEq)]
pub struct DirectData {
    buf: DataVec,
    default_element: Value,
}

/// This type wraps a `DataVec` to store attribute data.
impl DirectData {
    impl_data_base!(DataVec);

    /// Construct an attribute with a given size.
    pub fn with_size<T: AttributeValue>(n: usize, def: T) -> Self {
        DirectData {
            buf: DataVec::with_size(n, def.clone()),
            default_element: Value::new(def),
        }
    }

    /// Construct an attribute from a given `Vec<T>` of data reusing the space already
    /// allocated by the `Vec`.
    pub fn from_vec<T: AttributeValue + Default>(vec: Vec<T>) -> Self {
        DirectData {
            buf: DataVec::from_vec(vec),
            default_element: Value::new(T::default()),
        }
    }

    /// Construct an attribute from a given `DataVec` of data reusing the space already
    /// allocated.
    ///
    /// # Safety
    ///
    /// The given `def` must be a valid binary representation of an element stored in the given
    /// `DataVec`.
    pub unsafe fn from_raw_data(buf: DataVec, default_element: Value) -> Self {
        DirectData {
            buf,
            default_element,
        }
    }

    /// Construct an attribute from a given slice of data, by copying the data.
    #[inline]
    pub fn from_slice<T: AttributeValue + Default>(buf: &[T]) -> Self {
        Self::from_vec(buf.to_vec())
    }

    /// Construct a new empty attribute with the same buffer type and default element as `self`.
    pub fn duplicate_empty(&self) -> Self {
        DirectData {
            buf: DataVec::with_type_from(&self.buf),
            default_element: self.default_element.clone(),
        }
    }

    /// Construct a new attribute with the same buffer type and default element as `self`.
    ///
    /// The data within the newly created attribute is expected to be initialized with the given
    /// function `init`, which takes the output `DataVec` for the new attribute and the existing
    /// `DataSlice` from `self`.
    pub fn duplicate_with(
        &self,
        dup_data: impl FnOnce(DataSlice) -> VecDyn<dyn HasAttributeValue>,
    ) -> Self {
        DirectData {
            buf: from_dyn![VecDyn<dyn HasAttributeValue as AttributeValueVTable>](dup_data(
                self.data_ref().as_slice(),
            )),
            default_element: self.default_element.clone(),
        }
    }

    /// Construct a new attribute with the same buffer type, default element as `self`.
    ///
    /// The attribute is first initialized with the default value by allocating `len` default
    /// elements. Then the newly created buffer is expected to be modified by the `init` function.
    pub fn duplicate_with_len(
        &self,
        len: usize,
        init: impl FnOnce(DataSliceMut, DataSlice),
    ) -> Self {
        let mut attrib = self.duplicate_empty();
        attrib.extend_by(len);
        init(attrib.data_mut().as_mut_slice(), self.data_ref().as_slice());
        attrib
    }

    /// Produce a slice to the underlying data.
    #[inline]
    pub fn as_slice<T: Any>(&self) -> Result<&[T], Error> {
        self.buf
            .as_slice_as()
            .ok_or_else(|| Error::type_mismatch_from_buf::<T, _>(&self.buf))
    }

    /// Produce a mutable slice to the underlying data.
    #[inline]
    pub fn as_mut_slice<T: Any>(&mut self) -> Result<&mut [T], Error> {
        let element_id = self.buf.element_type_id();
        self.buf
            .as_mut_slice_as()
            .ok_or_else(|| Error::type_mismatch_id::<T>(element_id))
    }

    /// Convert the data stored by this attribute into a vector of the same size.
    #[inline]
    pub fn clone_into_vec<T: Any + Clone>(&self) -> Result<Vec<T>, Error> {
        self.buf
            .clone_into_vec()
            .ok_or_else(|| Error::type_mismatch_from_buf::<T, _>(&self.buf))
    }

    /// Push a value onto the underlying data buffer.
    pub fn push_cloned(&mut self, new_value_ref: ValueRef) -> Result<&mut Self, Error> {
        let expected = self.buf.element_type_id();
        let actual = new_value_ref.value_type_id();
        self.data_mut()
            .push_cloned(new_value_ref)
            .ok_or(Error::TypeMismatch { expected, actual })?;
        Ok(self)
    }

    /// Produce an iterator over the underlying data elements.
    #[inline]
    pub fn iter<'a, T: Any + 'a>(&'a self) -> Result<slice::Iter<T>, Error> {
        self.buf
            .iter_as::<T>()
            .ok_or_else(|| Error::type_mismatch_from_buf::<T, _>(&self.buf))
    }

    /// Produce a mutable iterator over the underlying data elements.
    #[inline]
    pub fn iter_mut<'a, T: Any + 'a>(&'a mut self) -> Result<slice::IterMut<T>, Error> {
        let element_id = self.buf.element_type_id();
        self.buf
            .iter_mut_as::<T>()
            .ok_or_else(|| Error::type_mismatch_id::<T>(element_id))
    }

    /// Convert this attribute into a typed vector of `T`. This consumes the attribute.
    #[inline]
    pub fn into_vec<T: AttributeValue>(self) -> Result<Vec<T>, Error> {
        let element_id = self.buf.element_type_id();
        self.buf
            .into_vec()
            .ok_or_else(|| Error::type_mismatch_id::<T>(element_id))
    }

    /// Get a reference to the defult element as a byte slice.
    #[inline]
    pub fn default_element(&self) -> ValueRef {
        self.default_element.as_ref()
    }
}

/// Mesh attribute data.
///
/// An attribute could store attribute values directly with the corresponding mesh elements, or
/// indirectly in a secondary storage indexed at each mesh element.
#[derive(Clone, Debug, PartialEq)]
pub enum AttributeData {
    /// A direct attribute stores values located directly (and uniquely) for each mesh element.
    Direct(DirectData),
    /// An indirect attribute stores values located on the heap an referenced using smart pointers
    /// (so non-uniquely) for each mesh element.
    ///
    /// Additionally these values are cached in an external cache data structure to make clones
    /// cheaper and reduce memory usage.
    Indirect(IndirectData),
}

impl AttributeData {
    /// Returns true if the underlying attribute is a direct attribute and false otherwise.
    pub fn is_direct(&self) -> bool {
        matches!(self, AttributeData::Direct(_))
    }

    /// Returns true if the underlying attribute is an indirect attribute and false otherwise.
    pub fn is_indirect(&self) -> bool {
        matches!(self, AttributeData::Indirect(_))
    }

    /// Construct a direct attribute with a given size.
    pub fn direct_with_size<T: AttributeValue>(n: usize, def: T) -> Self {
        AttributeData::Direct(DirectData::with_size(n, def))
    }

    /// Construct an indirect attribute with a given size.
    pub fn indirect_with_size<T: AttributeValueHash>(n: usize, def: T) -> Self {
        AttributeData::Indirect(IndirectData::with_size(n, def))
    }

    /// Construct a direct attribute from a given `Vec<T>` of data reusing the space already
    /// allocated by the `Vec`.
    pub fn direct_from_vec<T: AttributeValue + Default>(vec: Vec<T>) -> Self {
        AttributeData::Direct(DirectData::from_vec(vec))
    }

    /// Construct an indirect attribute from a given `Vec<T>` of data, while saving any repeated
    /// values in the given cache.
    pub fn indirect_from_vec<T: AttributeValueHash + Default>(
        vec: Vec<T>,
        cache: &mut AttribValueCache,
    ) -> Self {
        AttributeData::Indirect(IndirectData::from_vec(vec, cache))
    }

    /// Construct an indirect attribute from a given `IndirectData` instance. It is assumed that
    /// the included data is already cached correctly in the associated cache.
    pub fn indirect_from_data(data: IndirectData) -> Self {
        AttributeData::Indirect(data)
    }

    /// Construct an attribute from a given slice of data, by copying the data.
    #[inline]
    pub fn direct_from_slice<T: AttributeValue + Default>(data: &[T]) -> Self {
        Self::direct_from_vec(data.to_vec())
    }

    /// Construct an attribute from a given slice of data, by copying the data.
    #[inline]
    pub fn indirect_from_slice<T: AttributeValueHash + Default>(
        data: &[T],
        cache: &mut AttribValueCache,
    ) -> Self {
        Self::indirect_from_vec(data.to_vec(), cache)
    }

    // Helper function to map through the variants.
    fn map(
        &self,
        direct: impl FnOnce(&DirectData) -> DirectData,
        indirect: impl FnOnce(&IndirectData) -> IndirectData,
    ) -> Self {
        self.map_to(
            |d| AttributeData::Direct(direct(d)),
            |i| AttributeData::Indirect(indirect(i)),
        )
    }

    /// A utility function to map over the direct and indirect attribute variants given the two
    /// closures.
    pub fn map_to<'a, O>(
        &'a self,
        direct: impl FnOnce(&'a DirectData) -> O,
        indirect: impl FnOnce(&'a IndirectData) -> O,
    ) -> O {
        match self {
            AttributeData::Direct(data) => direct(data),
            AttributeData::Indirect(data) => indirect(data),
        }
    }

    /// A utility function to mutably map over the direct and indirect attribute variants given the
    /// two closures.
    pub fn map_mut_to<'a, O>(
        &'a mut self,
        direct: impl FnOnce(&'a mut DirectData) -> O,
        indirect: impl FnOnce(&'a mut IndirectData) -> O,
    ) -> O {
        match self {
            AttributeData::Direct(data) => direct(data),
            AttributeData::Indirect(data) => indirect(data),
        }
    }

    /// Produce a slice to the underlying direct attribute data.
    #[inline]
    pub fn as_slice<T: Any>(&self) -> Result<&[T], Error> {
        self.map_to(|d| d.as_slice(), |_| Err(Error::KindMismatchFoundIndirect))
    }

    /// Produce a mutable slice to the underlying direct attribute data.
    #[inline]
    pub fn as_mut_slice<T: Any>(&mut self) -> Result<&mut [T], Error> {
        self.map_mut_to(
            |d| d.as_mut_slice(),
            |_| Err(Error::KindMismatchFoundIndirect),
        )
    }

    /// Construct a new empty attribute with the same buffer type, default element and topology as
    /// `self`.
    pub fn duplicate_empty(&self) -> Self {
        self.map(|d| d.duplicate_empty(), |i| i.duplicate_empty())
    }

    /// Construct a new attribute with the same buffer type, default element and topology type as
    /// `self`.
    pub fn duplicate_with(
        &self,
        dup_data: impl FnOnce(&mut VecDyn<dyn HasAttributeValue>, Slice<dyn HasAttributeValue>),
    ) -> Self {
        match self {
            AttributeData::Direct(d) => AttributeData::Direct(d.duplicate_with(|input| {
                let vec_drop = VecDyn::with_type_from(input.reborrow());
                let mut vec_dyn = into_dyn![VecDyn<dyn HasAttributeValue>](vec_drop);
                dup_data(&mut vec_dyn, into_dyn![Slice<dyn HasAttributeValue>](input));
                vec_dyn
            })),
            AttributeData::Indirect(i) => AttributeData::Indirect(i.duplicate_with(|input| {
                let vec_drop = VecDyn::with_type_from(input.reborrow());
                let mut vec_dyn = into_dyn![VecDyn<dyn HasAttributeValue>](vec_drop);
                dup_data(&mut vec_dyn, into_dyn![Slice<dyn HasAttributeValue>](input));
                vec_dyn
            })),
        }
    }

    /// Construct a new attribute with the same buffer type, default element and topology type as
    /// `self`.
    ///
    /// The attribute is first initialized with the default value by allocating `len` default
    /// elements. Then the newly created buffer is expected to be modified by the `init` function.
    pub fn duplicate_with_len(
        &self,
        len: usize,
        init: impl FnOnce(DataSliceMut, DataSlice),
    ) -> Self {
        match self {
            AttributeData::Direct(d) => AttributeData::Direct(d.duplicate_with_len(len, init)),
            AttributeData::Indirect(i) => AttributeData::Indirect(
                i.duplicate_with_len(len, |new, old| init(new.upcast(), old.upcast())),
            ),
        }
    }

    /// Get the type data stored within this attribute
    #[inline]
    pub fn check<T: Any>(&self) -> Result<&Self, Error> {
        self.map_to(
            |d| d.check::<T>().map(|_| self),
            |i| i.check::<T>().map(|_| self),
        )
    }

    /// Get the mutable typed data stored within this attribute
    #[inline]
    pub fn check_mut<T: Any>(&mut self) -> Result<&mut Self, Error> {
        match self {
            AttributeData::Direct(d) => match d.check_mut::<T>() {
                Ok(_) => Ok(self),
                Err(e) => Err(e),
            },
            AttributeData::Indirect(i) => match i.check_mut::<T>() {
                Ok(_) => Ok(self),
                Err(e) => Err(e),
            },
        }
    }

    /// Produce an iterator over the underlying data elements regardless of kind.
    ///
    /// This incurs an additional `Box` deref when iterating.
    #[inline]
    pub fn iter<'a, T: Any>(&'a self) -> Result<Box<dyn Iterator<Item = &'a T> + 'a>, Error> {
        self.map_to(
            |d| {
                d.iter::<T>().map(|iter| {
                    let b: Box<dyn Iterator<Item = &T>> = Box::new(iter);
                    b
                })
            },
            |i| {
                i.iter::<T>().map(|iter| {
                    let b: Box<dyn Iterator<Item = &T>> = Box::new(iter);
                    b
                })
            },
        )
    }

    /// Produce an iterator over the underlying data elements for a direct attribute.
    #[inline]
    pub fn direct_iter<T: Any>(&self) -> Result<slice::Iter<T>, Error> {
        self.map_to(|d| d.iter::<T>(), |_| Err(Error::KindMismatchFoundIndirect))
    }

    /// Produce an iterator over the underlying data elements for an indirect attribute.
    #[inline]
    pub fn indirect_iter<T: Any>(&self) -> Result<impl Iterator<Item = &T>, Error> {
        self.map_to(|_| Err(Error::KindMismatchFoundDirect), |i| i.iter::<T>())
    }

    /// Produce a mutable iterator over the underlying data elements for a direct attribute.
    #[inline]
    pub fn direct_iter_mut<T: Any>(&mut self) -> Result<slice::IterMut<T>, Error> {
        self.map_mut_to(
            |d| d.iter_mut::<T>(),
            |_| Err(Error::KindMismatchFoundIndirect),
        )
    }

    /// Iterate over all the value in this attribute and update them as needed.
    ///
    /// This function takes a closure which takes an index and a smart pointer to the stored value and
    /// produces an optional new value. The new value is then used to update the attribute using
    /// the provided cache.
    #[inline]
    pub fn indirect_update_with<T: AttributeValueHash>(
        &mut self,
        f: impl FnMut(usize, &Irc<T>) -> Option<Irc<T>>,
        cache: &mut AttribValueCache,
    ) -> Result<&mut Self, Error> {
        match self {
            AttributeData::Indirect(i) => match i.update_with::<T, _>(f, cache) {
                Ok(_) => Ok(self),
                Err(e) => Err(e),
            },
            _ => Err(Error::KindMismatchFoundDirect),
        }
    }

    /// Get the type data stored within this attribute
    #[inline]
    pub fn element_type_id(&self) -> TypeId {
        self.map_to(|d| d.element_type_id(), |i| i.element_type_id())
    }

    /// Convert the data stored by this attribute into a vector of the same size.
    #[inline]
    pub fn clone_into_vec<T: AttributeValueHash>(&self) -> Result<Vec<T>, Error> {
        self.map_to(|d| d.clone_into_vec::<T>(), |i| i.clone_into_vec::<T>())
    }

    /// Convert the data stored by this direct attribute into a vector of the same size.
    #[inline]
    pub fn direct_clone_into_vec<T: Any + Clone>(&self) -> Result<Vec<T>, Error> {
        self.map_to(
            |d| d.clone_into_vec::<T>(),
            |_| Err(Error::KindMismatchFoundIndirect),
        )
    }

    /// Number of elements stored by this attribute. This is the same as the number of elements in
    /// the associated topology.
    #[inline]
    pub fn len(&self) -> usize {
        self.map_to(|d| d.len(), |i| i.len())
    }

    /// Check if there are any values in this attribute.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.map_to(|d| d.is_empty(), |i| i.is_empty())
    }

    /// Get a reference to the internal data as a `DataSlice`.
    #[inline]
    pub fn data_slice(&self) -> DataSlice {
        self.map_to(
            |d| d.data_ref().as_slice(),
            |i| i.data_ref().as_slice().upcast(),
        )
    }

    /// Get a mutable reference to the internal data as a `DataSliceMut`.
    #[inline]
    pub fn data_mut_slice(&mut self) -> DataSliceMut {
        self.map_mut_to(
            |d| d.data_mut().as_mut_slice(),
            |i| i.data_mut().as_mut_slice().upcast(),
        )
    }

    /// Get a reference to the internal indirect data as an `HDataVec`.
    #[inline]
    pub fn indirect_data(&self) -> Result<&IndirectData, Error> {
        self.map_to(|_| Err(Error::KindMismatchFoundDirect), Ok)
    }

    /// Get a mutable reference to the internal indirect data as an `HDataVec`.
    #[inline]
    pub fn indirect_data_mut(&mut self) -> Result<&mut IndirectData, Error> {
        self.map_mut_to(|_| Err(Error::KindMismatchFoundDirect), Ok)
    }

    /// Get a reference to the internal direct data as a `DirectData` struct.
    #[inline]
    pub fn direct_data(&self) -> Result<&DirectData, Error> {
        self.map_to(Ok, |_| Err(Error::KindMismatchFoundDirect))
    }

    /// Get a mutable reference to the internal direct data as a `DirectData` struct.
    #[inline]
    pub fn direct_data_mut(&mut self) -> Result<&mut DirectData, Error> {
        self.map_mut_to(Ok, |_| Err(Error::KindMismatchFoundDirect))
    }

    /// Convert this attribute into the underlying vector.
    ///
    /// This consumes the attribute.
    #[inline]
    pub fn into_data(self) -> DataVec {
        match self {
            AttributeData::Direct(d) => d.into_data(),
            AttributeData::Indirect(i) => i.into_data().upcast(),
        }
    }

    /// Extend this attribute by `n` elements. Effectively, this function appends the default
    /// element `n` number of times to this attribute.
    #[inline]
    pub fn extend_by(&mut self, n: usize) {
        self.map_mut_to(|d| d.extend_by(n), |i| i.extend_by(n))
    }

    /// Rotate this attribute in-place such that the first `mid` elements of the underlying buffer
    /// move to the end while the last `self.len() - mid` elements move to the front. After
    /// calling `rotate_left`, the element previously at index `mid` will become the first element
    /// in the slice.
    #[inline]
    pub fn rotate_left(&mut self, mid: usize) {
        self.map_mut_to(|d| d.rotate_left(mid), |i| i.rotate_left(mid))
    }

    /// Rotate this attribute in-place such that the first `self.len() - k` elements of the
    /// underlying buffer move to the end while the last `k` elements move to the front. After
    /// calling `rotate_right`, the element previously at index `self.len() - k` will become the
    /// first element in the slice.
    #[inline]
    pub fn rotate_right(&mut self, k: usize) {
        self.map_mut_to(|d| d.rotate_right(k), |i| i.rotate_right(k))
    }

    /// Get a reference to the default element as a byte slice.
    #[inline]
    pub fn default_element(&self) -> ValueRef {
        match self {
            AttributeData::Direct(d) => d.default_element(),
            AttributeData::Indirect(i) => i.default_element().upcast(),
        }
    }
}

/// Mesh attribute with an associated topology `I`.
///
/// This stores values that can be attached to mesh elements.
#[derive(Clone, Debug, PartialEq)]
pub struct Attribute<I> {
    /// Underlying attribute data.
    ///
    /// This can be used to manipulate attribute values or their references directly.
    pub data: AttributeData,
    phantom: PhantomData<I>,
}

/// This type wraps a `DataVec` to store attribute data. Having the type parameter `I` allows
/// the compiler verify that attributes are being indexed correctly.
impl<I> Attribute<I> {
    /// Construct a direct attribute with a given size.
    pub fn direct_with_size<T: AttributeValue>(n: usize, def: T) -> Self {
        Attribute {
            data: AttributeData::direct_with_size(n, def),
            phantom: PhantomData,
        }
    }

    /// Construct an indirect attribute with a given size.
    pub fn indirect_with_size<T: AttributeValueHash>(n: usize, def: T) -> Self {
        Attribute {
            data: AttributeData::indirect_with_size(n, def),
            phantom: PhantomData,
        }
    }

    /// Construct a direct attribute from a given `Vec<T>` of data reusing the space already
    /// allocated by the `Vec`.
    pub fn direct_from_vec<T: AttributeValue + Default>(vec: Vec<T>) -> Self {
        Attribute {
            data: AttributeData::direct_from_vec(vec),
            phantom: PhantomData,
        }
    }

    /// Construct an indirect attribute from a given `Vec<T>` of data, while saving any repeated
    /// values in the given cache.
    pub fn indirect_from_vec<T: AttributeValueHash + Default>(
        vec: Vec<T>,
        cache: &mut AttribValueCache,
    ) -> Self {
        Attribute {
            data: AttributeData::indirect_from_vec(vec, cache),
            phantom: PhantomData,
        }
    }

    /// Construct an indirect attribute from a given `IndirectData` instance. It is assumed that
    /// the included data is already cached correctly in the associated cache.
    pub fn indirect_from_data(data: IndirectData) -> Self {
        Attribute {
            data: AttributeData::indirect_from_data(data),
            phantom: PhantomData,
        }
    }

    /// Produce a slice to the underlying direct attribute data.
    #[inline]
    pub fn as_slice<T: Any>(&self) -> Result<&[T], Error> {
        self.data.as_slice()
    }

    /// Produce a mutable slice to the underlying direct attribute data.
    #[inline]
    pub fn as_mut_slice<T: Any>(&mut self) -> Result<&mut [T], Error> {
        self.data.as_mut_slice()
    }

    /// Construct a new empty attribute with the same buffer type, default element and topology as
    /// `self`.
    #[inline]
    pub fn duplicate_empty(&self) -> Self {
        self.promote_empty()
    }

    /// Construct a new attribute with the same buffer type, default element and topology type as
    /// `self`.
    ///
    /// The data within the newly created attribute is expected to be initialized with the given
    /// function `init`, which takes the output `DataSliceMut` for the new attribute and the existing
    /// `DataSlice` from `self`.
    #[inline]
    pub fn duplicate_with(
        &self,
        duplicate_data: impl FnOnce(&mut VecDyn<dyn HasAttributeValue>, Slice<dyn HasAttributeValue>),
    ) -> Self {
        self.promote_with(duplicate_data)
    }

    /// Construct a new attribute with the same buffer type, default element and topology type as
    /// `self`.
    ///
    /// The attribute is first initialized with the default value by allocating `len` default
    /// elements. Then the newly created buffer is expected to be modified by the `init` function.
    #[inline]
    pub fn duplicate_with_len(
        &self,
        len: usize,
        init: impl FnOnce(DataSliceMut, DataSlice),
    ) -> Self {
        self.promote_with_len(len, init)
    }

    /// Construct a new empty attribute with the same buffer type and default element as `self`.
    ///
    /// In contrast to `duplicate_empty` this function allows the new attribute to correspond with a
    /// different topology.
    #[inline]
    pub fn promote_empty<J>(&self) -> Attribute<J> {
        Attribute {
            data: self.data.duplicate_empty(),
            phantom: PhantomData,
        }
    }

    /// Construct a new attribute with the same data and default element as
    /// `self`, but corresponding to a different topology.
    pub fn promote<J>(&self) -> Attribute<J> {
        Attribute {
            data: self.data.duplicate_with(|dst, src| {
                // TODO: add an `extend` implementation to VecDyn
                for elem in src.iter() {
                    dst.push_cloned(elem);
                }
            }),
            phantom: PhantomData,
        }
    }

    /// Construct a new attribute with the same buffer type and default element as `self`.
    #[inline]
    pub fn promote_with<J>(
        &self,
        promote_data: impl FnOnce(&mut VecDyn<dyn HasAttributeValue>, Slice<dyn HasAttributeValue>),
    ) -> Attribute<J> {
        Attribute {
            data: self.data.duplicate_with(promote_data),
            phantom: PhantomData,
        }
    }

    /// Construct a new attribute with the same buffer type and default element as `self`.
    ///
    /// The attribute is first initialized with the default value by allocating `len` default
    /// elements. Then the newly created buffer is expected to be modified by the `init` function.
    pub fn promote_with_len<J>(
        &self,
        len: usize,
        init: impl FnOnce(DataSliceMut, DataSlice),
    ) -> Attribute<J> {
        Attribute {
            data: self.data.duplicate_with_len(len, init),
            phantom: PhantomData,
        }
    }

    /// Construct a direct attribute from a given slice of data, by copying the data.
    #[inline]
    pub fn direct_from_slice<T: AttributeValue + Default>(data: &[T]) -> Self {
        Self::direct_from_vec(data.to_vec())
    }

    /// Construct an indirect attribute from a given slice of data, by copying the data.
    #[inline]
    pub fn indirect_from_slice<T: AttributeValueHash + Default>(
        data: &[T],
        cache: &mut AttribValueCache,
    ) -> Self {
        Self::indirect_from_vec(data.to_vec(), cache)
    }

    /// Get the type data stored within this attribute
    #[inline]
    pub fn check<T: Any>(&self) -> Result<&Self, Error> {
        self.data.check::<T>().map(|_| self)
    }

    /// Get the mutable typed data stored within this attribute
    #[inline]
    pub fn check_mut<T: Any>(&mut self) -> Result<&mut Self, Error> {
        match self.data.check_mut::<T>() {
            Ok(_) => Ok(self),
            Err(e) => Err(e),
        }
    }

    /// Produce an iterator over the underlying data elements.
    #[inline]
    pub fn iter<'a, T: Any>(&'a self) -> Result<Box<dyn Iterator<Item = &'a T> + 'a>, Error> {
        self.data.iter::<T>()
    }

    /// Produce an iterator over the underlying data elements for a direct attribute.
    #[inline]
    pub fn direct_iter<T: Any>(&self) -> Result<slice::Iter<T>, Error> {
        self.data.direct_iter()
    }

    /// Produce an iterator over the underlying data elements for an indirect attribute.
    #[inline]
    pub fn indirect_iter<T: Any>(&self) -> Result<impl Iterator<Item = &T>, Error> {
        self.data.indirect_iter()
    }

    /// Produce a mutable iterator over the underlying data elements for a direct attribute.
    #[inline]
    pub fn direct_iter_mut<T: Any>(&mut self) -> Result<slice::IterMut<T>, Error> {
        self.data.direct_iter_mut()
    }

    /// Iterate over all the value in this attribute and update them as needed.
    ///
    /// This function takes a closure which takes an index and a smart pointer to the stored value and
    /// produces an optional new value. The new value is then used to update the attribute using
    /// the provided cache.
    #[inline]
    pub fn indirect_update_with<T, F>(
        &mut self,
        f: F,
        cache: &mut AttribValueCache,
    ) -> Result<&mut Self, Error>
    where
        T: AttributeValueHash,
        F: FnMut(usize, &Irc<T>) -> Option<Irc<T>>,
    {
        match self.data.indirect_update_with(f, cache) {
            Ok(_) => Ok(self),
            Err(e) => Err(e),
        }
    }

    /// Convert the data stored by this attribute into a vector of the same size.
    #[inline]
    pub fn clone_into_vec<T: AttributeValueHash>(&self) -> Result<Vec<T>, Error> {
        self.data.clone_into_vec::<T>()
    }

    /// Convert the data stored by this direct attribute into a vector of the same size.
    #[inline]
    pub fn direct_clone_into_vec<T: Any + Clone>(&self) -> Result<Vec<T>, Error> {
        self.data.direct_clone_into_vec::<T>()
    }

    /// Number of elements stored by this attribute. This is the same as the number of elements in
    /// the associated topology.
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if there are any values in this attribute.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get a reference to the internal data as a `DataSlice`.
    #[inline]
    pub fn data_slice(&self) -> DataSlice {
        self.data.data_slice()
    }

    /// Get a mutable reference to the internal data as a `DataSliceMut`.
    #[inline]
    pub fn data_mut_slice(&mut self) -> DataSliceMut {
        self.data.data_mut_slice()
    }

    /// Convert this attribute into the underlying buffer. This consumes the attribute.
    #[inline]
    pub fn into_data(self) -> DataVec {
        self.data.into_data()
    }

    /// Extend this attribute by `n` elements. Effectively, this function appends the default
    /// element `n` number of times to this attribute.
    #[inline]
    pub fn extend_by(&mut self, n: usize) {
        self.data.extend_by(n);
    }

    /// Rotate this attribute in-place such that the first `mid` elements of the underlying buffer
    /// move to the end while the last `self.len() - mid` elements move to the front. After
    /// calling `rotate_left`, the element previously at index `mid` will become the first element
    /// in the slice.
    #[inline]
    pub fn rotate_left(&mut self, mid: usize) {
        self.data.rotate_left(mid);
    }

    /// Rotate this attribute in-place such that the first `self.len() - k` elements of the
    /// underlying buffer move to the end while the last `k` elements move to the front. After
    /// calling `rotate_right`, the element previously at index `self.len() - k` will become the
    /// first element in the slice.
    #[inline]
    pub fn rotate_right(&mut self, k: usize) {
        self.data.rotate_right(k);
    }

    /// Get a reference to the default element.
    #[inline]
    pub fn default_element(&self) -> ValueRef {
        self.data.default_element()
    }
}

/*
 * Implement typed indexing into an attribute.
 * This is a costly operation, but it could be useful for debugging.
 */

macro_rules! impl_attribute_get {
    ($type:ty) => {
        impl Attribute<$type> {
            /// Get `i`'th attribute value.
            #[inline]
            pub fn get<T: Any + Copy, I: Into<$type>>(&self, i: I) -> Result<T, Error> {
                let element_id = self.data.element_type_id();
                Index::from(i.into())
                    .map_or(None, move |x| {
                        self.data
                            .map_to(
                                |d| d.as_slice().map(|s| s[x]),
                                |i| i.as_rc_slice().map(|s| *s[x]),
                            )
                            .ok()
                    })
                    .ok_or(Error::type_mismatch_id::<T>(element_id))
            }

            /// Get a `const` reference to the `i`'th attribute value.
            #[inline]
            pub fn get_ref<T: Any, I: Into<$type>>(&self, i: I) -> Result<&T, Error> {
                let element_id = self.data.element_type_id();
                Index::from(i.into())
                    .map_or(None, move |x| {
                        self.data
                            .map_to(
                                |d| d.as_slice().map(|s| &s[x]),
                                |i| i.as_rc_slice().map(|s| &*s[x]),
                            )
                            .ok()
                    })
                    .ok_or(Error::type_mismatch_id::<T>(element_id))
            }

            /// Get a mutable reference to the `i`'th direct attribute value.
            ///
            /// This function works only on direct attributes. Indirect attributes cannot be
            /// modified via mutable references, since they employ a special caching mechanism
            /// which aliases each stored element.
            #[inline]
            pub fn get_mut<T: Any, I: Into<$type>>(&mut self, i: I) -> Result<&mut T, Error> {
                let element_id = self.data.element_type_id();
                Index::from(i.into())
                    .map_or(None, move |x| {
                        self.data
                            .map_mut_to(
                                |d| d.as_mut_slice().map(|s| &mut s[x]),
                                |_| Err(Error::KindMismatchFoundIndirect),
                            )
                            .ok()
                    })
                    .ok_or(Error::type_mismatch_id::<T>(element_id))
            }
        }
    };
}

impl_attribute_get!(MeshIndex);
impl_attribute_get!(VertexIndex);
impl_attribute_get!(EdgeIndex);
impl_attribute_get!(FaceIndex);
impl_attribute_get!(CellIndex);
impl_attribute_get!(EdgeVertexIndex);
impl_attribute_get!(FaceVertexIndex);
impl_attribute_get!(FaceEdgeIndex);
impl_attribute_get!(CellVertexIndex);
impl_attribute_get!(CellEdgeIndex);
impl_attribute_get!(CellFaceIndex);
impl_attribute_get!(VertexEdgeIndex);
impl_attribute_get!(VertexFaceIndex);
impl_attribute_get!(VertexCellIndex);
impl_attribute_get!(EdgeFaceIndex);
impl_attribute_get!(EdgeCellIndex);
impl_attribute_get!(FaceCellIndex);

/// An intrinsic attribute type. This differs from `Attribute<I>` in that it is explicitly typed
/// and it is intended to be used for attributes that are "intrinsic" to the specific mesh type.
/// For instance, the position attribute is intrinsic to polygonal or tetrahedral meshes and point
/// clouds. Intrinsic attributes define the geometry of the mesh type.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct IntrinsicAttribute<T, I> {
    data: Vec<T>,
    phantom: PhantomData<I>,
}

impl<T, I> IntrinsicAttribute<T, I> {
    /// Construct an attribute with a given size.
    pub fn with_size(n: usize, def: T) -> Self
    where
        T: Clone,
    {
        IntrinsicAttribute {
            data: vec![def; n],
            phantom: PhantomData,
        }
    }

    /// Construct an attribute from a given `Vec<T>` of data reusing the space aready
    /// allocated by the `Vec`.
    pub fn from_vec(vec: Vec<T>) -> Self {
        IntrinsicAttribute {
            data: vec,
            phantom: PhantomData,
        }
    }

    //    /// Construct an attribute from a given `DataVec` of data reusing the space aready
    //    /// allocated.
    //    #[cfg(feature = "io")]
    //    pub fn from_io_buffer(data: IOBuffer) -> Option<Self>
    //    where
    //        T: Any,
    //    {
    //        data.into_vec::<T>().map(|vec| IntrinsicAttribute {
    //            data: vec,
    //            phantom: PhantomData,
    //        })
    //    }

    /// Construct an attribute from a given slice of data, by copying the data.
    #[inline]
    pub fn from_slice(data: &[T]) -> Self
    where
        T: Clone,
    {
        Self::from_vec(data.to_vec())
    }

    /// Produce a slice to the underlying data.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        self.data.as_slice()
    }

    /// Produce a mutable slice to the underlying data.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.data.as_mut_slice()
    }

    /// Move the contents of this attribute into a `Vec`. This is identical to using the `Into`
    /// trait.
    #[inline]
    pub fn into_vec(self) -> Vec<T> {
        self.data
    }

    /// Get the internal `Vec` storing the attribute data.
    ///
    /// Use this very carefully because it allows the user to modify the size of the internal
    /// vector which may violate intrinsic properties of the mesh that this attribute is part of.
    #[inline]
    pub fn as_mut_vec(&mut self) -> &mut Vec<T> {
        &mut self.data
    }

    /// Convert the data stored by this attribute into a vector of the same size.
    #[inline]
    pub fn clone_into_vec(&self) -> Vec<T>
    where
        T: Clone,
    {
        self.data.clone()
    }

    /// Convert the data stored by this attribute into a vector of the same size. This function is
    /// similar to `clone_into_vec` but assumes that elements are `Copy`. It may also be more performant
    /// than `clone_into_vec`.
    #[inline]
    pub fn copy_into_vec(&self) -> Vec<T>
    where
        T: Copy,
    {
        let mut vec = Vec::with_capacity(self.len());
        vec.extend(self.as_slice());
        vec
    }

    /// Produce an iterator over the underlying data elements.
    #[inline]
    pub fn iter(&self) -> slice::Iter<T> {
        self.data.iter()
    }

    /// Produce a parallel iterator over the underlying data elements.
    #[cfg(feature = "rayon")]
    #[inline]
    pub fn par_iter(&self) -> rayon::slice::Iter<T>
    where
        T: Sync,
    {
        use rayon::iter::IntoParallelRefIterator;
        self.data.par_iter()
    }

    /// Produce a mutable iterator over the underlying data elements.
    #[inline]
    pub fn iter_mut(&mut self) -> slice::IterMut<T> {
        self.data.iter_mut()
    }

    /// Produce a mutable parallel iterator over the underlying data elements.
    #[cfg(feature = "rayon")]
    #[inline]
    pub fn par_iter_mut(&mut self) -> rayon::slice::IterMut<T>
    where
        T: Sync + Send,
    {
        use rayon::iter::IntoParallelRefMutIterator;
        self.data.par_iter_mut()
    }

    /// Number of elements stored by this attribute. This is the same as the number of elements in
    /// the associated topology.
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if there are any values in this attribute.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

impl<T, I> From<Vec<T>> for IntrinsicAttribute<T, I> {
    #[inline]
    fn from(vec: Vec<T>) -> Self {
        Self::from_vec(vec)
    }
}

impl<T, I> From<IntrinsicAttribute<T, I>> for Vec<T> {
    #[inline]
    fn from(val: IntrinsicAttribute<T, I>) -> Self {
        val.into_vec()
    }
}

impl<T, I: Into<usize>, J: Into<I>> std::ops::Index<J> for IntrinsicAttribute<T, I> {
    type Output = T;
    fn index(&self, index: J) -> &T {
        &self.data[index.into().into()]
    }
}
impl<T, I: Into<usize>, J: Into<I>> std::ops::IndexMut<J> for IntrinsicAttribute<T, I> {
    fn index_mut(&mut self, index: J) -> &mut T {
        &mut self.data[index.into().into()]
    }
}

impl<T, I> std::iter::IntoIterator for IntrinsicAttribute<T, I> {
    type Item = T;
    type IntoIter = std::vec::IntoIter<T>;
    fn into_iter(self) -> Self::IntoIter {
        self.into_vec().into_iter()
    }
}

impl<T, I> std::iter::FromIterator<T> for IntrinsicAttribute<T, I> {
    fn from_iter<J>(iter: J) -> Self
    where
        J: IntoIterator<Item = T>,
    {
        Self::from_vec(Vec::from_iter(iter))
    }
}

#[cfg(feature = "rayon")]
impl<T: Send, I> rayon::iter::IntoParallelIterator for IntrinsicAttribute<T, I> {
    type Item = T;
    type Iter = rayon::vec::IntoIter<T>;
    fn into_par_iter(self) -> Self::Iter {
        self.into_vec().into_par_iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn indirect_set_value_at() {
        let mut cache = AttribValueCache::default();
        let mut data = IndirectData::with_size(3, String::from("default"));
        let val = HValue::new(Irc::new(String::from("default")));
        assert_eq!(&data.default_element, &val);
        data.set_at(1, String::from("default"), &mut cache).unwrap();
        assert!(cache.is_empty());
        data.set_at(1, String::from("New Value"), &mut cache)
            .unwrap();
        assert_eq!(cache.len(), 1);
    }
}
