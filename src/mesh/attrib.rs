#![warn(missing_docs)]

//! # Attribute API.
//!
//! This module defines the API used to store and access attributes in meshes. Attributes are
//! stored inside `HashMap` data structures, and are accessed through the variety of getters
//! described below. The most efficient attribute access is through iterators which are standard
//! slice iterators, so slice operations work on attribute data. Alternatively, individual
//! attribute values can be accessed using `get`, `get_ref` and `get_mut`, but each call to such a
//! getter icurs a cost for runtime type checking. Since the type of the attribute being stored is
//! determined at runtime, you must provide the expected type and location of the attribute when
//! retrieving values. This is done using a "turbofish"; for example to retrieve a 3 float array
//! attribute stored at vertices we need to specify `::<[f32; 3], VertexIndex>` to the appropriate
//! accessor. See the `Attrib` trait for more details.
//!
//! ## New Mesh Types
//! When defining new meshes, simply "derive" the `Attrib` trait. The custom derive will look for
//! a field with type `AttribDict<_>` where `_` corresponds to an index type identifying the
//! location where the attribute lives. For example `AttribDict<VertexIndex>` is a `HashMap` that
//! stores attributes at vertices of the mesh. Any topology index can be used there. See the
//! implementations of `TetMesh` and `TriMesh` for more details.

use std::any::{Any, TypeId};
//use std::collections::HashMap;
use hashbrown::HashMap;
use std::slice;

use dync::{traits::HasDrop, VecDyn};

use crate::mesh::topology::*;

mod attribute;
mod bytes;
mod index;

// Expose the entry API for our AttribDict type.
//pub use std::collections::hash_map::Entry;
pub use hashbrown::hash_map::Entry;

pub use attribute::*;
pub use bytes::*;
pub use index::*;

/// Attribute dictionary for a specific topology `I`.
pub type AttribDict<I> = HashMap<String, Attribute<I>>;

// TODO: explore the use of a single attribute dictionary for all topologies, where the location of
// the attribute is determined by an enum. For example:
// ```
// pub enum Location {
//     Mesh,
//     Vertex,
//     Edge,
//     Face,
//     Cell,
//     EdgeVertex,
//     FaceVertex,
//     FaceEdge,
//     CellVertex,
//     CellEdge,
//     CellFace,
//     VertexEdge,
//     VertexFace,
//     VertexCell,
//     EdgeFace,
//     EdgeCell,
//     FaceCell,
// }
// pub type AttribDict = HashMap<(Location, String), AttributeData>;
// ```
// This approach should have the following tradeoffs
// 1. Slightly faster compiles at the cost of slightly slower attribute access.
// 2. Smaller mesh types sinces they require only a single AttribDict.
// 3. No compile-time checking of attribute types.

///// Attribute Collector. This structure queries attributes from the mesh and collects references to
///// the data slices for further processing. This allows users to batch borrow mutable slices.
//pub struct AttribCollect {
//    refs: Vec<Attribute>
//
//}

/// Attribute interfaces for meshes. In order to derive this trait the mesh must have a field
/// called `attributes` with type `AttribDict`.
pub trait Attrib
where
    Self: std::marker::Sized,
{
    /// Get the size of the attribute at the appropriate mesh location determined by `I`.
    fn attrib_size<I: AttribIndex<Self>>(&self) -> usize;

    /// Read only access to the attribute dictionary.
    fn attrib_dict<I: AttribIndex<Self>>(&self) -> &AttribDict<I>;

    /// Read and write access to the attribute dictionary.
    fn attrib_dict_mut<I: AttribIndex<Self>>(&mut self) -> &mut AttribDict<I>;

    /// Read and write access to the attribute dictionary along with a cache for indirect attribute
    /// values.
    fn attrib_dict_and_cache_mut<I: AttribIndex<Self>>(
        &mut self,
    ) -> (&mut AttribDict<I>, Option<&mut AttribValueCache>);

    /// Add an attribute at the appropriate location with a given default.
    fn add_attrib<'a, T, I: AttribIndex<Self>>(
        &mut self,
        name: &'a str,
        def: T,
    ) -> Result<&mut Attribute<I>, Error>
    where
        T: AttributeValue,
    {
        let n = self.attrib_size::<I>();
        match self.attrib_dict_mut().entry(name.to_owned()) {
            Entry::Occupied(_) => Err(Error::AlreadyExists(name.to_owned())),
            Entry::Vacant(entry) => Ok(entry.insert(Attribute::direct_with_size(n, def))),
        }
    }

    /// Construct an attribute from a given data `Vec<T>`. `data` must have
    /// exactly the right size for the attribute to be added successfully.
    fn add_attrib_data<'a, T, I: AttribIndex<Self>>(
        &mut self,
        name: &'a str,
        data: Vec<T>,
    ) -> Result<&mut Attribute<I>, Error>
    where
        T: AttributeValue + Default,
    {
        let expected_size = self.attrib_size::<I>();
        let given_size = data.len();
        if given_size != expected_size {
            Err(Error::WrongSize {
                expected_size,
                given_size,
            })
        } else {
            match self.attrib_dict_mut().entry(name.to_owned()) {
                Entry::Occupied(_) => Err(Error::AlreadyExists(name.to_owned())),
                Entry::Vacant(entry) => Ok(entry.insert(Attribute::direct_from_vec(data))),
            }
        }
    }

    /// Sets the attribute to the specified default value whether or not it
    /// already exists.
    fn set_attrib<'a, T, I: AttribIndex<Self>>(
        &mut self,
        name: &'a str,
        def: T,
    ) -> Result<&mut Attribute<I>, Error>
    where
        T: AttributeValue,
    {
        let n = self.attrib_size::<I>();
        Ok(match self.attrib_dict_mut().entry(name.to_owned()) {
            Entry::Occupied(mut entry) => {
                entry.insert(Attribute::direct_with_size(n, def));
                entry.into_mut()
            }
            Entry::Vacant(entry) => entry.insert(Attribute::direct_with_size(n, def)),
        })
    }

    /// Set an attribute to the given data slice. `data` must have exactly the
    /// right size for the attribute to be set successfully.
    fn set_attrib_data<'a, T, I: AttribIndex<Self>>(
        &mut self,
        name: &'a str,
        data: &[T],
    ) -> Result<&mut Attribute<I>, Error>
    where
        T: AttributeValue + Default,
    {
        let expected_size = self.attrib_size::<I>();
        let given_size = data.len();
        if given_size != expected_size {
            Err(Error::WrongSize {
                expected_size,
                given_size,
            })
        } else {
            Ok(match self.attrib_dict_mut().entry(name.to_owned()) {
                Entry::Occupied(mut entry) => {
                    entry.insert(Attribute::direct_from_slice(data));
                    entry.into_mut()
                }
                Entry::Vacant(entry) => entry.insert(Attribute::direct_from_slice(data)),
            })
        }
    }

    /// Add an indirect attribute at the appropriate location with a given default.
    fn add_indirect_attrib<'a, T, I: AttribIndex<Self>>(
        &mut self,
        name: &'a str,
        def: T,
    ) -> Result<(&mut Attribute<I>, &mut AttribValueCache), Error>
    where
        T: AttributeValueHash,
    {
        let n = self.attrib_size::<I>();
        if let (dict, Some(cache)) = self.attrib_dict_and_cache_mut() {
            match dict.entry(name.to_owned()) {
                Entry::Occupied(_) => Err(Error::AlreadyExists(name.to_owned())),
                Entry::Vacant(entry) => {
                    Ok((entry.insert(Attribute::indirect_with_size(n, def)), cache))
                }
            }
        } else {
            Err(Error::MissingAttributeValueCache)
        }
    }

    /// Sets the indirect attribute to the specified default value whether or not it
    /// already exists.
    fn set_indirect_attrib<'a, T, I: AttribIndex<Self>>(
        &mut self,
        name: &'a str,
        def: T,
    ) -> Result<(&mut Attribute<I>, &mut AttribValueCache), Error>
    where
        T: AttributeValueHash,
    {
        let n = self.attrib_size::<I>();
        if let (dict, Some(cache)) = self.attrib_dict_and_cache_mut() {
            Ok((
                match dict.entry(name.to_owned()) {
                    Entry::Occupied(mut entry) => {
                        entry.insert(Attribute::indirect_with_size(n, def));
                        entry.into_mut()
                    }
                    Entry::Vacant(entry) => entry.insert(Attribute::indirect_with_size(n, def)),
                },
                cache,
            ))
        } else {
            Err(Error::MissingAttributeValueCache)
        }
    }

    /// Construct an indirect attribute from a given `IndirectData`. `data` must have
    /// exactly the right size for the attribute to be added successfully.
    fn add_indirect_attrib_data<'a, I: AttribIndex<Self>>(
        &mut self,
        name: &'a str,
        data: IndirectData,
    ) -> Result<(&mut Attribute<I>, &mut AttribValueCache), Error> {
        let expected_size = self.attrib_size::<I>();
        let given_size = data.len();
        if given_size != expected_size {
            Err(Error::WrongSize {
                expected_size,
                given_size,
            })
        } else if let (dict, Some(cache)) = self.attrib_dict_and_cache_mut() {
            match dict.entry(name.to_owned()) {
                Entry::Occupied(_) => Err(Error::AlreadyExists(name.to_owned())),
                Entry::Vacant(entry) => {
                    Ok((entry.insert(Attribute::indirect_from_data(data)), cache))
                }
            }
        } else {
            Err(Error::MissingAttributeValueCache)
        }
    }

    /// Set an indirect attribute to the given `IndirectData` instance. `data` must have
    /// exactly the right size for the attribute to be set successfully.
    fn set_indirect_attrib_data<'a, I: AttribIndex<Self>>(
        &mut self,
        name: &'a str,
        data: IndirectData,
    ) -> Result<(&mut Attribute<I>, &mut AttribValueCache), Error> {
        let expected_size = self.attrib_size::<I>();
        let given_size = data.len();
        if given_size != expected_size {
            Err(Error::WrongSize {
                expected_size,
                given_size,
            })
        } else if let (dict, Some(cache)) = self.attrib_dict_and_cache_mut() {
            Ok((
                match dict.entry(name.to_owned()) {
                    Entry::Occupied(mut entry) => {
                        entry.insert(Attribute::indirect_from_data(data));
                        entry.into_mut()
                    }
                    Entry::Vacant(entry) => entry.insert(Attribute::indirect_from_data(data)),
                },
                cache,
            ))
        } else {
            Err(Error::MissingAttributeValueCache)
        }
    }

    /// Makes a copy of an existing attribute.
    ///
    /// Return a mutable reference to the new attribute if successful.
    fn duplicate_attrib<'a, 'b, T, I: AttribIndex<Self>>(
        &mut self,
        name: &'a str,
        new_name: &'b str,
    ) -> Result<&mut Attribute<I>, Error>
    where
        T: Any + Clone,
    {
        let dup_attrib = self.attrib(name)?.clone();
        match self.attrib_check::<T, I>(new_name) {
            Ok(_) => Err(Error::AlreadyExists(new_name.to_owned())),
            Err(Error::DoesNotExist(_)) => Ok(self
                .attrib_dict_mut()
                .entry(new_name.to_owned())
                .or_insert(dup_attrib)),
            Err(err) => Err(err),
        }
    }

    /// Remove an attribute from the attribute dictionary.
    ///
    /// From there you can use methods defined on the returned attribute, and if desired return
    /// it back to the dictionary with `insert_attrib`.
    fn remove_attrib<I: AttribIndex<Self>>(&mut self, name: &str) -> Result<Attribute<I>, Error> {
        match self.attrib_dict_mut().remove(name) {
            Some(attrib) => Ok(attrib),
            None => Err(Error::DoesNotExist(name.to_owned())),
        }
    }

    /// Inserts an attribute into the dictionary with the usual `HashMap` semantics.
    ///
    /// This means that if an attribute with the same name exists, it will be returned and the
    /// current table entry is updated with the new value. If it doesn't already exist, `None` is
    /// returned.
    fn insert_attrib<I: AttribIndex<Self>>(
        &mut self,
        name: &str,
        attrib: Attribute<I>,
    ) -> Result<Option<Attribute<I>>, Error> {
        let expected_size = self.attrib_size::<I>();
        let given_size = attrib.len();
        if given_size != expected_size {
            Err(Error::WrongSize {
                expected_size,
                given_size,
            })
        } else {
            Ok(self.attrib_dict_mut().insert(name.to_owned(), attrib))
        }
    }

    /// Retrieve the attribute with the given name and if it doesn't exist, add a new one and set
    /// it to a given default value. In either case the mutable reference to the attribute is
    /// returned.
    fn attrib_or_add<'a, T, I: AttribIndex<Self>>(
        &mut self,
        name: &'a str,
        def: T,
    ) -> Result<&mut Attribute<I>, Error>
    where
        T: AttributeValue,
    {
        let n = self.attrib_size::<I>();
        match self.attrib_dict_mut().entry(name.to_owned()) {
            Entry::Occupied(entry) => entry.into_mut().check_mut::<T>(),
            Entry::Vacant(entry) => Ok(entry.insert(Attribute::direct_with_size(n, def))),
        }
    }

    /// Retrieve the attribute with the given name and if it doesn't exist, set its data to
    /// what's in the given slice. In either case the mutable reference to the attribute is
    /// returned.
    fn attrib_or_add_data<'a, T, I: AttribIndex<Self>>(
        &mut self,
        name: &'a str,
        data: &[T],
    ) -> Result<&mut Attribute<I>, Error>
    where
        T: AttributeValue + Default,
    {
        let expected_size = self.attrib_size::<I>();
        let given_size = data.len();
        if given_size != expected_size {
            Err(Error::WrongSize {
                expected_size,
                given_size,
            })
        } else {
            match self.attrib_dict_mut().entry(name.to_owned()) {
                Entry::Occupied(entry) => entry.into_mut().check_mut::<T>(),
                Entry::Vacant(entry) => Ok(entry.insert(Attribute::direct_from_slice(data))),
            }
        }
    }

    /// Retrieve the indirect attribute with the given name and if it doesn't exist, add a new one
    /// and set it to a given default value. In either case the mutable reference to the attribute
    /// is returned.
    fn attrib_or_add_indirect<'a, T, I: AttribIndex<Self>>(
        &mut self,
        name: &'a str,
        def: T,
    ) -> Result<(&mut Attribute<I>, &mut AttribValueCache), Error>
    where
        T: AttributeValueHash,
    {
        let n = self.attrib_size::<I>();
        if let (dict, Some(cache)) = self.attrib_dict_and_cache_mut() {
            match dict.entry(name.to_owned()) {
                Entry::Occupied(entry) => entry.into_mut().check_mut::<T>().map(|a| (a, cache)),
                Entry::Vacant(entry) => {
                    Ok((entry.insert(Attribute::indirect_with_size(n, def)), cache))
                }
            }
        } else {
            Err(Error::MissingAttributeValueCache)
        }
    }

    /// Get the attribute iterator for a direct attribute.
    fn direct_attrib_iter<'a, 'b, T, I: 'b + AttribIndex<Self>>(
        &'b self,
        name: &'a str,
    ) -> Result<slice::Iter<T>, Error>
    where
        T: Any + Clone,
    {
        self.attrib::<I>(name)?.direct_iter::<T>()
    }

    /// Get the attribute mutable iterator for a direct attribute.
    ///
    /// This is essentially an alias for `attrib_mut(name).unwrap().direct_iter_mut::<T>()`.
    fn attrib_iter_mut<'a, 'b, T, I: 'b + AttribIndex<Self>>(
        &'b mut self,
        name: &'a str,
    ) -> Result<slice::IterMut<T>, Error>
    where
        T: Any + Clone,
    {
        self.attrib_mut::<I>(name)?.direct_iter_mut::<T>()
    }

    /// Get the iterator for an attribute no matter what kind.
    fn attrib_iter<'b, T, I: 'b + AttribIndex<Self>>(
        &'b self,
        name: &str,
    ) -> Result<Box<dyn Iterator<Item = &'b T> + 'b>, Error>
    where
        T: Any + Clone,
    {
        self.attrib::<I>(name)?.iter::<T>()
    }

    /// Update indirect attribute entries with the given closure.
    ///
    /// Return a mutable reference to `Self` on success.
    fn indirect_attrib_update_with<'a, 'b, T, I, F>(
        &'b mut self,
        name: &'a str,
        f: F,
    ) -> Result<(&'b mut Attribute<I>, &'b mut AttribValueCache), Error>
    where
        T: AttributeValueHash,
        I: 'b + AttribIndex<Self>,
        F: FnMut(usize, &Irc<T>) -> Option<Irc<T>>,
    {
        let (dict, cache) = self.attrib_dict_and_cache_mut();
        if let Some(cache) = cache {
            match dict.get_mut(name) {
                Some(attrib) => attrib
                    .indirect_update_with::<T, _>(f, cache)
                    .map(|a| (a, cache)),
                None => Err(Error::DoesNotExist(name.to_owned())),
            }
        } else {
            Err(Error::MissingAttributeValueCache)
        }
    }

    /// Return `true` if the given attribute exists at the given location, and
    /// `false` otherwise, even if the specified attribute location is invalid
    /// for the mesh.
    fn attrib_exists<I: AttribIndex<Self>>(&self, name: &str) -> bool {
        self.attrib_dict::<I>().contains_key(&name.to_owned())
    }

    /// Determine if the given attribute is valid and exists at the given
    /// location.
    fn attrib_check<'a, T: Any, I: AttribIndex<Self>>(
        &self,
        name: &'a str,
    ) -> Result<&Attribute<I>, Error> {
        self.attrib::<I>(name)?.check::<T>()
    }

    /// Expose the underlying direct attribute as a slice.
    fn attrib_as_slice<'a, 'b, T: 'static, I: 'b + AttribIndex<Self>>(
        &'b self,
        name: &'a str,
    ) -> Result<&'b [T], Error> {
        self.attrib::<I>(name)?.as_slice()
    }

    /// Expose the underlying direct attribute as a mutable slice.
    fn attrib_as_mut_slice<'a, 'b, T: 'static, I: 'b + AttribIndex<Self>>(
        &'b mut self,
        name: &'a str,
    ) -> Result<&'b mut [T], Error> {
        self.attrib_mut::<I>(name)?.as_mut_slice()
    }

    /// Clone attribute data into a `Vec<T>`.
    ///
    /// This works for direct and indirect attributes. Note that indirect attributes can be
    /// expensive to clone.
    fn attrib_clone_into_vec<'a, 'b, T, I: 'b + AttribIndex<Self>>(
        &'b self,
        name: &'a str,
    ) -> Result<Vec<T>, Error>
    where
        T: AttributeValueHash,
    {
        self.attrib::<I>(name)?.clone_into_vec()
    }

    /// Clone direct attribute data into a `Vec<T>`.
    fn direct_attrib_clone_into_vec<'a, 'b, T, I: 'b + AttribIndex<Self>>(
        &'b self,
        name: &'a str,
    ) -> Result<Vec<T>, Error>
    where
        T: AttributeValue,
    {
        self.attrib::<I>(name)?.direct_clone_into_vec()
    }

    /// Borrow the raw attribute from the attribute dictionary. From there you can
    /// use methods defined on the attribute itself.
    fn attrib<'a, I: AttribIndex<Self>>(&self, name: &'a str) -> Result<&Attribute<I>, Error> {
        match self.attrib_dict().get(name) {
            Some(attrib) => Ok(attrib),
            None => Err(Error::DoesNotExist(name.to_owned())),
        }
    }

    /// Get the raw mutable attribute from the attribute dictionary. From there
    /// you can use methods defined on the attribute itself.
    fn attrib_mut<'a, I: AttribIndex<Self>>(
        &mut self,
        name: &'a str,
    ) -> Result<&mut Attribute<I>, Error> {
        match self.attrib_dict_mut().get_mut(name) {
            Some(attrib) => Ok(attrib),
            None => Err(Error::DoesNotExist(name.to_owned())),
        }
    }
}

/// Promote attributes from one topology to another.
pub trait AttribPromote<SI, TI>
where
    Self: Sized,
    SI: AttribIndex<Self>,
    TI: AttribIndex<Self>,
{
    /// Promote the given attribute from source topology `SI` to target topology `TI`.
    ///
    /// A mutable reference to the resulting attribute is returned upon success.
    ///
    /// Collisions are handled using the given `combine` closure which takes two attribute
    /// components of type `T` and combines them into a single `T` to be written to the target
    /// topology attribute.
    ///
    /// If an attribute with the same name already exists at the target topology, it will be
    /// combined with the promoted attribute. If that attribute has the wrong type, an error is
    /// returned.
    fn attrib_promote<'a, U, F>(
        &mut self,
        name: &'a str,
        combine: F,
    ) -> Result<&Attribute<TI>, Error>
    where
        U: Clone + 'static,
        F: for<'b> FnMut(&'b mut U, &'b U);
}

impl<T: crate::Real> AttribPromote<FaceVertexIndex, VertexIndex> for crate::mesh::TriMesh<T> {
    fn attrib_promote<'a, U, F>(
        &mut self,
        name: &'a str,
        mut combine: F,
    ) -> Result<&Attribute<VertexIndex>, Error>
    where
        U: Any + Clone,
        F: for<'b> FnMut(&'b mut U, &'b U),
    {
        // Before removing, ensure that we have the right attribute.
        self.attrib_check::<U, FaceVertexIndex>(name)?;

        let attrib = self.remove_attrib::<FaceVertexIndex>(name)?;
        let num_verts = self.num_vertices();

        // TODO: Attribute dict mutable borrow is preventing us from using the topo functions
        //       below.  Figure out a better solution.
        let crate::mesh::TriMesh {
            vertex_attributes,
            indices,
            ..
        } = self;

        // Add attribute to the appropriate dictionary.
        match vertex_attributes.entry(name.to_owned()) {
            Entry::Occupied(mut entry) => {
                let other = entry.get_mut().as_mut_slice::<U>()?;
                // Combine with promoted attribute
                for (fv_idx, fv_val) in attrib.direct_iter::<U>()?.enumerate() {
                    let vtx_idx = indices[fv_idx / 3][fv_idx % 3];
                    combine(&mut other[vtx_idx], fv_val);
                }
                Ok(entry.into_mut())
            }
            Entry::Vacant(entry) => {
                let new_attrib: Attribute<VertexIndex> =
                    attrib.promote_with_len(num_verts, move |mut new, orig| {
                        let new = new.as_slice::<U>().unwrap(); // already checked
                        let mut seen = vec![false; new.len()];
                        for (fv_idx, fv_val) in orig.iter_as::<U>().unwrap().enumerate() {
                            let vtx_idx = indices[fv_idx / 3][fv_idx % 3];
                            if !seen[vtx_idx] {
                                new[vtx_idx] = fv_val.clone();
                                seen[vtx_idx] = true;
                            } else {
                                // Already initialized, combine with previously written value:
                                combine(&mut new[vtx_idx], fv_val);
                            }
                        }
                    });
                Ok(entry.insert(new_attrib))
            }
        }
    }
}

/// Error type specific to retrieving attributes from the attribute dictionary.
#[derive(Debug, PartialEq)]
pub enum Error {
    /// Attribute being added already exists.
    AlreadyExists(String),
    /// Attribute exists but the specified type is inaccurate.
    TypeMismatch {
        /// `TypeId` of the expected attribute type.
        expected: TypeId,
        /// `TypeId` of the actual attribute type.
        actual: TypeId,
    },
    /// Couldn't find the attribute with the given name and location.
    DoesNotExist(String),
    /// Given attribute size does not match expected attribute size.
    WrongSize {
        /// Attribute size of the underlying topology.
        expected_size: usize,
        /// Given data size, which must match the size of the underlying topology.
        given_size: usize,
    },
    /// Trying to access an indirect attribute as a direct attribute or vice versa.
    /// Expected an indirect attribute, found a direct attribute.
    KindMismatchFoundDirect,
    /// Trying to access an indirect attribute as a direct attribute or vice versa.
    /// Expected a direct attribute, found an indirect attribute.
    KindMismatchFoundIndirect,
    /// Missing attribute value cache for indirect attributes.
    MissingAttributeValueCache,
}

impl Error {
    /// Convenience function for building the `TypeMismatch` variant.
    fn type_mismatch_from_buf<T: Any, V: HasDrop>(data: &VecDyn<V>) -> Self {
        Self::type_mismatch_id::<T>(data.element_type_id())
    }
    /// Same as above but for specified `TypeId`.
    fn type_mismatch_id<T: Any>(actual: TypeId) -> Self {
        Error::TypeMismatch {
            expected: TypeId::of::<T>(),
            actual,
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        None
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Error::AlreadyExists(attrib_name) => {
                write!(f, "An attribute named \"{}\" already exists", attrib_name)
            }
            Error::TypeMismatch { expected, actual } => write!(
                f,
                "Type mismatch.\nExpected: {:?},\nActual: {:?}",
                expected, actual
            ),
            Error::DoesNotExist(attrib_name) => {
                write!(f, "The attribute \"{}\" does not exist", attrib_name)
            }
            Error::WrongSize {
                expected_size,
                given_size,
            } => write!(
                f,
                "Given attribute size: {}, does not match expected size: {}",
                given_size, expected_size
            ),
            Error::KindMismatchFoundDirect => write!(
                f,
                "Expected an indirect attribute, found a direct attribute",
            ),
            Error::KindMismatchFoundIndirect => write!(
                f,
                "Expected a direct attribute, found an indirect attribute",
            ),
            Error::MissingAttributeValueCache => write!(
                f,
                "Missing an attribute value cache needed for indirect attributes"
            ),
        }
    }
}

impl From<dync::Error> for Error {
    fn from(dync_err: dync::Error) -> Self {
        match dync_err {
            dync::Error::ValueTooLarge => {
                // ValueTooLarge can only be thrown when the value cannot fit into a usize. Since
                // We use usize sized pointers, this will never be thrown.
                unreachable!()
            }
            dync::Error::MismatchedTypes { expected, actual } => {
                Error::TypeMismatch { expected, actual }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::TriMesh;
    use crate::mesh::VertexPositions;
    use num_traits::Zero;

    /// Test that an attribute with a default value can be added, removed and reinserted.
    #[test]
    fn basic_test() {
        use math::Vector3;

        let pts = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ];
        let indices = vec![[0, 1, 2], [1, 3, 2]];

        let mut trimesh = TriMesh::new(pts, indices);

        {
            let nml_attrib = trimesh
                .add_attrib::<_, VertexIndex>("N", Vector3::<f64>::zero())
                .unwrap();
            assert_eq!(
                nml_attrib.get::<Vector3<f64>, _>(VertexIndex::from(1)),
                Ok(Vector3::zero())
            );
        }

        // check that the attribute indeed exists
        assert!(trimesh
            .attrib_check::<Vector3<f64>, VertexIndex>("N")
            .is_ok());

        // delete the attribute
        let mut nml_attrib = trimesh.remove_attrib::<VertexIndex>("N").unwrap();

        // check that the attribute was deleted
        assert!(!trimesh.attrib_exists::<VertexIndex>("N"));

        *nml_attrib.get_mut(2).unwrap() = Vector3::from([1.0, 2.0, 3.0]); // modify an element

        assert!(trimesh.insert_attrib("N", nml_attrib).unwrap().is_none());
        assert!(trimesh.attrib_exists::<VertexIndex>("N"));
        assert_eq!(
            trimesh
                .attrib::<VertexIndex>("N")
                .unwrap()
                .get::<Vector3<f64>, _>(2)
                .unwrap(),
            Vector3::from([1.0, 2.0, 3.0])
        );
    }

    /// Test setting of attributes
    #[test]
    fn set_attrib_test() {
        use math::Vector3;
        use num_traits::Zero;

        let pts = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ];
        let indices = vec![[0, 1, 2], [1, 3, 2]];

        let mut trimesh = TriMesh::new(pts, indices);

        // Attrib doesn't exist yet
        for nml in trimesh
            .set_attrib::<_, VertexIndex>("N", Vector3::<f64>::zero())
            .unwrap()
            .iter::<Vector3<f64>>()
            .unwrap()
        {
            assert_eq!(*nml, Vector3::zero());
        }
        // Attrib already exist, we expect it to be overwritten
        for nml in trimesh
            .set_attrib::<_, VertexIndex>("N", Vector3::from([1.0f64; 3]))
            .unwrap()
            .iter::<Vector3<f64>>()
            .unwrap()
        {
            assert_eq!(*nml, Vector3::from([1.0; 3]));
        }

        // Attrib already exist, we expect it to be overwritten
        let verts = trimesh.vertex_positions().to_vec();
        trimesh
            .set_attrib_data::<_, VertexIndex>("N", verts.as_slice())
            .unwrap();
        for (nml, p) in trimesh
            .attrib_iter::<[f64; 3], VertexIndex>("N")
            .unwrap()
            .zip(trimesh.vertex_positions())
        {
            assert_eq!(*nml, *p);
        }

        // Attrib doesn't exist yet
        let verts = trimesh.vertex_positions().to_vec();
        trimesh
            .set_attrib_data::<_, VertexIndex>("ref", verts.as_slice())
            .unwrap();
        for (r, p) in trimesh
            .attrib_iter::<[f64; 3], VertexIndex>("ref")
            .unwrap()
            .zip(trimesh.vertex_positions())
        {
            assert_eq!(*r, *p);
        }
    }

    /// Test attrib_or_add* methods.
    #[test]
    fn attrib_or_add_test() -> Result<(), Error> {
        let pts = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ];
        let indices = vec![[0, 1, 2], [1, 3, 2]];

        let mut trimesh = TriMesh::new(pts, indices);

        // Attrib doesn't exist yet
        for nml in trimesh
            .attrib_or_add::<_, VertexIndex>("N", [0.0f64; 3])?
            .iter::<[f64; 3]>()?
        {
            assert_eq!(*nml, [0.0; 3]);
        }

        // Attrib already exists, we expect it to be left intact
        for nml in trimesh
            .attrib_or_add::<_, VertexIndex>("N", [1.0f64; 3])?
            .iter::<[f64; 3]>()?
        {
            assert_eq!(*nml, [0.0; 3]);
        }

        // Attrib already exists, we expect it to be left intact
        let verts: Vec<_> = trimesh.vertex_positions().to_vec();
        for nml in trimesh
            .attrib_or_add_data::<_, VertexIndex>("N", verts.as_slice())?
            .iter::<[f64; 3]>()?
        {
            assert_eq!(*nml, [0.0; 3]);
        }

        // Attrib doesn't yet exist
        let verts = trimesh.vertex_positions().to_vec();
        trimesh.attrib_or_add_data::<_, VertexIndex>("ref", verts.as_slice())?;
        for (r, p) in trimesh
            .attrib_iter::<[f64; 3], VertexIndex>("ref")?
            .zip(trimesh.vertex_positions())
        {
            assert_eq!(*r, *p);
        }

        Ok(())
    }

    /// Test that multidimensional attributes can be added to the mesh from a `Vec`.
    /// Also test the `attrib_clone_into_vec` and `attrib_as_slice` functions for multidimensional
    /// attributes.
    #[test]
    fn multidim_test() {
        let pts = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ];
        let indices = vec![[0, 1, 2], [1, 3, 2]];

        let mut trimesh = TriMesh::new(pts, indices);

        let data = vec![[0i8, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]];
        {
            let attrib = trimesh
                .add_attrib_data::<_, VertexIndex>("attrib1", data.clone())
                .unwrap();
            for i in 0..data.len() {
                assert_eq!(attrib.get::<[i8; 3], _>(VertexIndex::from(i)), Ok(data[i]));
            }
        }
        assert_eq!(
            trimesh
                .attrib_clone_into_vec::<[i8; 3], VertexIndex>("attrib1")
                .unwrap(),
            data
        );
        assert_eq!(
            trimesh
                .attrib_as_slice::<[i8; 3], VertexIndex>("attrib1")
                .unwrap(),
            data.as_slice()
        );

        assert!(trimesh
            .attrib_check::<[i8; 3], VertexIndex>("attrib1")
            .is_ok());
        assert!(trimesh.remove_attrib::<VertexIndex>("attrib1").is_ok());
        assert!(!trimesh.attrib_exists::<VertexIndex>("attrib1"));
    }

    /// Test miscallenous attribute manipulation, like duplicating an attribute.
    #[test]
    fn misc_test() {
        let pts = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ];
        let indices = vec![[0, 1, 2], [1, 3, 2]];

        let mut trimesh = TriMesh::new(pts, indices);

        let data = vec![[0i8, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]];
        {
            trimesh
                .add_attrib_data::<_, VertexIndex>("attrib1", data.clone())
                .unwrap();
        }

        {
            trimesh
                .duplicate_attrib::<[i8; 3], VertexIndex>("attrib1", "attrib2")
                .unwrap();
        }

        // Check that the duplicate attribute exists and has the right type.
        trimesh
            .attrib_check::<[i8; 3], VertexIndex>("attrib2")
            .unwrap();

        for (i, val) in trimesh
            .attrib_iter::<[i8; 3], VertexIndex>("attrib2")
            .unwrap()
            .enumerate()
        {
            assert_eq!(*val, data[i]);
        }

        assert!(trimesh.remove_attrib::<VertexIndex>("attrib1").is_ok());
        assert!(trimesh.remove_attrib::<VertexIndex>("attrib2").is_ok());
        assert!(!trimesh.attrib_exists::<VertexIndex>("attrib1"));
        assert!(!trimesh.attrib_exists::<VertexIndex>("attrib2"));
    }

    /// Test promoting an attribute from face-vertex to vertex.
    #[test]
    fn attrib_promote() {
        let pts = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ];
        let indices = vec![[0, 1, 2], [1, 3, 2]];

        let mut trimesh = TriMesh::new(pts, indices);

        let data = vec![[0i8, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11]];
        {
            trimesh
                .add_attrib_data::<_, FaceVertexIndex>("attrib1", data.clone())
                .unwrap();
        }

        {
            // This should not remove "attrib1" since it has the wrong type. (regression test)
            let first_attempt = trimesh.attrib_promote::<[f32; 2], _>("attrib1", |_, _| {
                panic!("Wrong attribute promote executed");
            });

            assert!(first_attempt.is_err());

            // Actual promote function
            trimesh
                .attrib_promote::<[i8; 2], _>("attrib1", |a, b| {
                    a[0] += b[0];
                    a[1] += b[1];
                })
                .unwrap();
        }

        // Check that the promoted attribute exists and has the right type.
        trimesh
            .attrib_check::<[i8; 2], VertexIndex>("attrib1")
            .unwrap();

        assert_eq!(
            trimesh
                .attrib_as_slice::<[i8; 2], VertexIndex>("attrib1")
                .unwrap(),
            &[[0i8, 1], [8, 10], [14, 16], [8, 9]][..]
        );
    }
}
