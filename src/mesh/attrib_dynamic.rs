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

use topology::*;
use index::*;
use mesh::buffer::DataBuffer;
use std::collections::{hash_map, HashMap};
use std::any::{Any, TypeId};
use std::slice;
use std::marker::PhantomData;

/// The place on the mesh where an attribute is stored.
pub enum AttribLocation {
    Mesh,
    Vertex,
    Edge,
    Face,
    Cell,
    EdgeVertex,
    FaceVertex,
    FaceEdge,
    CellVertex,
    CellEdge,
    CellFace,
    VertexEdge,
    VertexFace,
    VertexCell,
    EdgeFace,
    EdgeCell,
    FaceCell,
}

/// Attribute map. This stores all the attributes on a mesh.
pub type AttribDict = HashMap<(AttribLocation, String), Attribute>;

/// Mesh attribute type. This stores values that can be attached to mesh elements.
#[derive(Clone, Debug, PartialEq)]
pub struct Attribute {
    data: DataBuffer,
    phantom: PhantomData<I>,
}

/// This type wraps a `DataBuffer` to store attribute data. Having the type parameter `I` allows
/// the compiler verify that attributes are being indexed correctly.
impl<I> Attribute<I> {
    /// Construct an attribute with a given size at the given location.
    pub fn with_size<T: Any + Clone>(n: usize, def: T) -> Self {
        Attribute {
            data: DataBuffer::with_size(n, def),
            phantom: PhantomData,
        }
    }

    /// Construct an attribute from a given `Vec<T>` of data reusing the space aready
    /// allocated by the `Vec`.
    pub fn from_vec<T: Any + Clone>(vec: Vec<T>) -> Self {
        Attribute {
            data: DataBuffer::from_vec(vec),
            phantom: PhantomData,
        }
    }

    /// Construct an attribute from a given `DataBuffer` of data reusing the space aready
    /// allocated.
    pub fn from_data_buffer<T: Any>(data: DataBuffer) -> Self {
        Attribute {
            data,
            phantom: PhantomData,
        }
    }

    /// Get the type data stored within this attribute
    #[inline]
    pub fn check<T: Any>(&mut self) -> Result<&mut Self, Error> {
        match self.data.check_mut::<T>() {
            Some(_) => Ok(self),
            None => Err(Error::TypeMismatch),
        }
    }

    /// Get the type data stored within this attribute
    #[inline]
    pub fn element_type_id(&self) -> TypeId {
        self.data.element_type_id()
    }

    /// Construct an attribute from a given slice of data, by copying the data.
    #[inline]
    pub fn from_slice<T: Any + Clone>(data: &[T]) -> Self {
        Self::from_vec(data.to_vec())
    }

    /// Produce a slice to the underlying data.
    #[inline]
    pub fn as_slice<T: Any>(&self) -> Result<&[T], Error> {
        self.data.as_slice().ok_or(Error::TypeMismatch)
    }

    /// Produce a mutable slice to the underlying data.
    #[inline]
    pub fn as_mut_slice<T: Any>(&mut self) -> Result<&mut [T], Error> {
        self.data.as_mut_slice().ok_or(Error::TypeMismatch)
    }

    /// Convert the data stored by this attribute into a vector of the same size.
    #[inline]
    pub fn into_vec<T: Any + Clone>(&self) -> Result<Vec<T>, Error> {
        self.data.into_vec().ok_or(Error::TypeMismatch)
    }

    /// Produce an iterator over the underlying data elements.
    #[inline]
    pub fn iter<'a, T: Any + 'a>(&'a self) -> Result<slice::Iter<T>, Error> {
        self.data.iter::<T>().ok_or(Error::TypeMismatch)
    }

    /// Produce a mutable iterator over the underlying data elements.
    #[inline]
    pub fn iter_mut<'a, T: Any + 'a>(&'a mut self) -> Result<slice::IterMut<T>, Error> {
        self.data.iter_mut::<T>().ok_or(Error::TypeMismatch)
    }

    /// Number of elements stored by this attribute. This is the same as the number of elements in
    /// the associated topology.
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Get `i`'th attribute value.
    #[inline]
    pub unsafe fn get_unchecked<T: Any + Copy>(&self, i: usize) -> T {
        self.data.get_unchecked(i)
    }

    /// Get a `const` reference to the `i`'th attribute value.
    #[inline]
    pub unsafe fn get_unchecked_ref<T: Any>(&self, i: usize) -> &T {
        self.data.get_unchecked_ref(i)
    }

    /// Get a mutable reference to the `i`'th attribute value.
    #[inline]
    pub unsafe fn get_unchecked_mut<T: Any>(&mut self, i: usize) -> &mut T {
        self.data.get_unchecked_mut(i)
    }
}

/// This trait provides an interface for the implementer of `Attrib` to access attributes
/// associated with a specific topology within a mesh.
pub trait AttribIndex<M>
where
    Self: ::std::marker::Sized + Clone,
{
    /// Get the size of the attribute at the appropriate mesh location determined by `I`.
    fn attrib_size(mesh: &M) -> usize;

    /// Read only access to the attribute dictionary.
    fn attrib_dict(mesh: &M) -> &AttribDict<Self>;

    /// Read and write access to the attribute dictionary.
    fn attrib_dict_mut(mesh: &mut M) -> &mut AttribDict<Self>;
}

macro_rules! impl_attrib_index {
    ($topo_attrib:ident, $type:ty, $topo_num:ident) => {
        impl Attribute<$type> {
            /// Get `i`'th attribute value.
            #[inline]
            pub fn get<T: Any + Copy, I: Into<$type>>(&self, i: I) -> Result<T, Error> {
                Index::from(i.into())
                    .map_or(None, move |x| self.data.get(x)).ok_or(Error::TypeMismatch)
            }

            /// Get a `const` reference to the `i`'th attribute value.
            #[inline]
            pub fn get_ref<T: Any, I: Into<$type>>(&self, i: I) -> Result<&T, Error> {
                Index::from(i.into())
                    .map_or(None, move |x| self.data.get_ref(x)).ok_or(Error::TypeMismatch)
            }

            /// Get a mutable reference to the `i`'th attribute value.
            #[inline]
            pub fn get_mut<T: Any, I: Into<$type>>(&mut self, i: I) -> Result<&mut T, Error> {
                Index::from(i.into())
                    .map_or(None, move |x| self.data.get_mut(x)).ok_or(Error::TypeMismatch)
            }
        }

        /// Topology specific attribute implementation trait. This trait exists to allow one to
        /// access mesh attributes using the `AttribIndex` trait, and should never be used
        /// explicitly.
        pub trait $topo_attrib {
            /// Mesh implementation of the attribute size getter.
            fn impl_attrib_size(&self) -> usize;
            /// Mesh implementation of the attribute dictionary getter.
            fn impl_attrib_dict(&self) -> &AttribDict<$type>;
            /// Mesh implementation of the attribute dictionary mutable getter.
            fn impl_attrib_dict_mut(&mut self) -> &mut AttribDict<$type>;
        }

		impl<M: $topo_attrib> AttribIndex<M> for $type {
            #[inline]
			fn attrib_size(mesh: &M) -> usize {
				mesh.impl_attrib_size()
			}

            #[inline]
			fn attrib_dict(mesh: &M) -> &AttribDict<Self> {
				mesh.impl_attrib_dict()
			}

            #[inline]
			fn attrib_dict_mut(mesh: &mut M) -> &mut AttribDict<Self> {
				mesh.impl_attrib_dict_mut()
			}
		}
    }
}

impl_attrib_index!(MeshAttrib, MeshIndex, num_meshes);
impl_attrib_index!(VertexAttrib, VertexIndex, num_verts);
impl_attrib_index!(EdgeAttrib, EdgeIndex, num_edges);
impl_attrib_index!(FaceAttrib, FaceIndex, num_faces);
impl_attrib_index!(CellAttrib, CellIndex, num_cells);
impl_attrib_index!(EdgeVertexAttrib, EdgeVertexIndex, num_edge_verts);
impl_attrib_index!(FaceVertexAttrib, FaceVertexIndex, num_face_verts);
impl_attrib_index!(FaceEdgeAttrib, FaceEdgeIndex, num_face_edges);
impl_attrib_index!(CellVertexAttrib, CellVertexIndex, num_cell_verts);
impl_attrib_index!(CellEdgeAttrib, CellEdgeIndex, num_cell_edges);
impl_attrib_index!(CellFaceAttrib, CellFaceIndex, num_cell_faces);
impl_attrib_index!(VertexEdgeAttrib, VertexEdgeIndex, num_vert_edges);
impl_attrib_index!(VertexFaceAttrib, VertexFaceIndex, num_vert_faces);
impl_attrib_index!(VertexCellAttrib, VertexCellIndex, num_vert_cell);
impl_attrib_index!(EdgeFaceAttrib, EdgeFaceIndex, num_edge_faces);
impl_attrib_index!(EdgeCellAttrib, EdgeCellIndex, num_edge_cells);
impl_attrib_index!(FaceCellAttrib, FaceCellIndex, num_face_cells);

/// Attribute interfaces for meshes. In order to derive this trait the mesh must have a field
/// called `attributes` with type `AttribDict`.
pub trait Attrib
where
    Self: ::std::marker::Sized,
{
    /// Get the size of the attribute at the appropriate mesh location determined by `I`.
    fn attrib_size<I: AttribIndex<Self>>(&self) -> usize;

    /// Read only access to the attribute dictionary.
    fn attrib_dict<I: AttribIndex<Self>>(&self) -> &AttribDict<I>;

    /// Read and write access to the attribute dictionary.
    fn attrib_dict_mut<I: AttribIndex<Self>>(&mut self) -> &mut AttribDict<I>;

    /// Add an attribute at the appropriate location with a given default.
    fn add_attrib<'a, T, I: AttribIndex<Self>>(
        &mut self,
        name: &'a str,
        def: T,
    ) -> Result<&mut Attribute<I>, Error>
    where
        T: Any + Clone,
    {
        let n = self.attrib_size::<I>();
        match self.attrib_dict_mut().entry(name.to_owned()) {
            hash_map::Entry::Occupied(_) => Err(Error::AlreadyExists(name.to_owned())),
            hash_map::Entry::Vacant(entry) => Ok(entry.insert(Attribute::with_size(n, def))),
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
        T: Any + Clone,
    {
        if data.len() != self.attrib_size::<I>() {
            Err(Error::WrongSize(data.len()))
        } else {
            match self.attrib_dict_mut().entry(name.to_owned()) {
                hash_map::Entry::Occupied(_) => Err(Error::AlreadyExists(name.to_owned())),
                hash_map::Entry::Vacant(entry) => Ok(entry.insert(Attribute::from_vec(data))),
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
        T: Any + Clone,
    {
        let n = self.attrib_size::<I>();
        Ok(match self.attrib_dict_mut().entry(name.to_owned()) {
            hash_map::Entry::Occupied(mut entry) => {
                entry.insert(Attribute::with_size(n, def));
                entry.into_mut()
            }
            hash_map::Entry::Vacant(entry) => entry.insert(Attribute::with_size(n, def)),
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
        T: Any + Clone,
    {
        if self.attrib_size::<I>() != data.len() {
            Err(Error::WrongSize(data.len()))
        } else {
            Ok(match self.attrib_dict_mut().entry(name.to_owned()) {
                hash_map::Entry::Occupied(mut entry) => {
                    entry.insert(Attribute::from_slice(data));
                    entry.into_mut()
                }
                hash_map::Entry::Vacant(entry) => entry.insert(Attribute::from_slice(data)),
            })
        }
    }

    /// Makes a copy of an existing attribute. Return a mutable reference to the
    /// new attribute if successful.
    fn duplicate_attrib<'a, 'b, T, I: AttribIndex<Self>>(
        &mut self,
        name: &'a str,
        new_name: &'b str,
    ) -> Result<&mut Attribute<I>, Error>
    where
        T: Any + Clone,
    {
        let dup_attrib = self.attrib(name)?.clone();
        match self.attrib_check::<I>(new_name) {
            Ok(_) => Err(Error::AlreadyExists(new_name.to_owned())),
            Err(Error::DoesNotExist(_)) => Ok(self.attrib_dict_mut()
                .entry(new_name.to_owned())
                .or_insert(dup_attrib)),
            Err(err) => Err(err),
        }
    }

    /// Remove an attribute.
    fn remove_attrib<'a, I: AttribIndex<Self>>(&mut self, name: &'a str) -> Result<(), Error> {
        self.attrib_check::<I>(name).map(|_| {
            self.attrib_dict_mut::<I>()
                .remove(&name.to_owned())
                .unwrap();
        })
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
        T: Any + Clone,
    {
        let n = self.attrib_size::<I>();
        match self.attrib_dict_mut().entry(name.to_owned()) {
            hash_map::Entry::Occupied(entry) => entry.into_mut().check::<T>(),
            hash_map::Entry::Vacant(entry) => Ok(entry.insert(Attribute::with_size(n, def))),
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
        T: Any + Clone,
    {
        if self.attrib_size::<I>() != data.len() {
            Err(Error::WrongSize(data.len()))
        } else {
            match self.attrib_dict_mut().entry(name.to_owned()) {
                hash_map::Entry::Occupied(entry) => entry.into_mut().check::<T>(),
                hash_map::Entry::Vacant(entry) => Ok(entry.insert(Attribute::from_slice(data))),
            }
        }
    }

    /// Get the attribute iterator. This is essentially a safe version of
    /// `attrib(loc, name).unwrap().iter::<T>()`.
    fn attrib_iter<'a, 'b, T, I: 'b + AttribIndex<Self>>(
        &'b self,
        name: &'a str,
    ) -> Result<slice::Iter<T>, Error>
    where
        T: 'static + Clone,
    {
        self.attrib::<I>(name)?.iter::<T>()
    }

    /// Get the attribute mutable iterator. This is essentially a safe version of
    /// `attrib_mut(name).unwrap().iter_mut::<T>()`.
    fn attrib_iter_mut<'a, 'b, T, I: 'b + AttribIndex<Self>>(
        &'b mut self,
        name: &'a str,
    ) -> Result<slice::IterMut<T>, Error>
    where
        T: 'static + Clone,
    {
        self.attrib_mut::<I>(name)?.iter_mut::<T>()
    }

    /// Return `true` if the given attribute exists at the given location, and
    /// `false` otherwise, even if the specified attribute location is invalid
    /// for the mesh.
    fn attrib_exists<'a, I: AttribIndex<Self>>(&self, name: &'a str) -> bool {
        self.attrib_dict::<I>().contains_key(&name.to_owned())
    }

    /// Determine if the given attribute is valid and exists at the given
    /// location.
    fn attrib_check<'a, I: AttribIndex<Self>>(&self, name: &'a str) -> Result<(), Error> {
        if self.attrib_dict::<I>().contains_key(&name.to_owned()) {
            Ok(())
        } else {
            Err(Error::DoesNotExist(name.to_owned()))
        }
    }

    /// Expose the underlying attribute as a slice.
    fn attrib_as_slice<'a, 'b, T: 'static, I: 'b + AttribIndex<Self>>(
        &'b self,
        name: &'a str,
    ) -> Result<&'b [T], Error> {
        self.attrib::<I>(name)?.as_slice()
    }

    /// Expose the underlying attribute as a mutable slice.
    fn attrib_as_mut_slice<'a, 'b, T: 'static, I: 'b + AttribIndex<Self>>(
        &'b mut self,
        name: &'a str,
    ) -> Result<&'b mut [T], Error> {
        self.attrib_mut::<I>(name)?.as_mut_slice()
    }

    /// Clone attribute data into a `Vec<T>`.
    fn attrib_into_vec<'a, 'b, T, I: 'b + AttribIndex<Self>>(
        &'b mut self,
        name: &'a str,
    ) -> Result<Vec<T>, Error>
    where
        T: 'static + Clone,
    {
        self.attrib::<I>(name)?.into_vec()
    }

    /// Get the raw attribute from the attribute dictionary. From there you can
    /// use methods defined on the attribute itself.
    fn attrib<'a, I: AttribIndex<Self>>(&self, name: &'a str) -> Result<&Attribute<I>, Error> {
        match self.attrib_dict().get(&name.to_owned()) {
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
        match self.attrib_dict_mut().get_mut(&name.to_owned()) {
            Some(attrib) => Ok(attrib),
            None => Err(Error::DoesNotExist(name.to_owned())),
        }
    }
}

/// Error type specific to retrieving attributes from the attribute dictionary.
#[derive(Debug, PartialEq)]
pub enum Error {
    /// Attribute being added already exists.
    AlreadyExists(String),
    /// Attribute exists but the specified type is inaccurate.
    TypeMismatch,
    /// Couln't find the attribute with the given name and location.
    DoesNotExist(String),
    /// Vec size must match number of attributes allowed in the given location.
    WrongSize(usize),
}

#[cfg(test)]
mod tests {
    use super::*;
    use mesh::TriMesh;
    use reinterpret::*;

    /// Test that an attribute with a default value can be added and removed.
    #[test]
    fn basic_test() {
        use math::Vector3;

        let pts = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ];
        let indices = vec![0, 1, 2, 1, 3, 2];

        let mut trimesh = TriMesh::new(pts, indices);

        {
            let nml_attrib = trimesh
                .add_attrib::<_, VertexIndex>("N", Vector3::<f64>::zeros())
                .unwrap();
            assert_eq!(
                nml_attrib.get::<Vector3<f64>, _>(VertexIndex::from(1)),
                Ok(Vector3::zeros())
            );
        }

        // check that the attribute indeed exists
        trimesh.attrib_check::<VertexIndex>("N").ok().unwrap();

        // delete the attribute
        trimesh.remove_attrib::<VertexIndex>("N").ok();

        // check that the attribute was deleted
        trimesh.attrib_check::<VertexIndex>("N").err().unwrap();
    }

    /// Test setting of attributes
    #[test]
    fn set_attrib_test() {
        use math::Vector3;

        let pts = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ];
        let indices = vec![0, 1, 2, 1, 3, 2];

        let mut trimesh = TriMesh::new(pts, indices);

        // Attrib doesn't exist yet
        for nml in trimesh
            .set_attrib::<_, VertexIndex>("N", Vector3::<f64>::zeros())
            .unwrap()
            .iter::<Vector3<f64>>()
            .unwrap()
        {
            assert_eq!(*nml, Vector3::zeros());
        }
        // Attrib already exist, we expect it to be overwritten
        for nml in trimesh
            .set_attrib::<_, VertexIndex>("N", Vector3::<f64>::ones())
            .unwrap()
            .iter::<Vector3<f64>>()
            .unwrap()
        {
            assert_eq!(*nml, Vector3::ones());
        }

        // Attrib already exist, we expect it to be overwritten
        let verts: Vec<Vector3<f64>> = reinterpret_slice(trimesh.vertex_positions()).to_vec();
        trimesh
            .set_attrib_data::<_, VertexIndex>("N", verts.as_slice())
            .unwrap();
        for (nml, p) in trimesh
            .attrib_iter::<Vector3<f64>, VertexIndex>("N")
            .unwrap()
            .zip(trimesh.vertex_positions())
        {
            assert_eq!(*nml, Vector3(*p));
        }

        // Attrib doesn't exist yet
        let verts = trimesh.vertex_positions().to_vec();
        trimesh
            .set_attrib_data::<_, VertexIndex>("ref", verts.as_slice())
            .unwrap();
        for (r, p) in trimesh
            .attrib_iter::<[f64;3], VertexIndex>("ref")
            .unwrap()
            .zip(trimesh.vertex_positions())
        {
            assert_eq!(*r, *p);
        }
    }

    /// Test attrib_or_add* methods.
    #[test]
    fn attrib_or_add_test() {
        use math::Vector3;

        let pts = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ];
        let indices = vec![0, 1, 2, 1, 3, 2];

        let mut trimesh = TriMesh::new(pts, indices);

        // Attrib doesn't exist yet
        for nml in trimesh
            .attrib_or_add::<_, VertexIndex>("N", Vector3::<f64>::zeros())
            .unwrap()
            .iter::<Vector3<f64>>()
            .unwrap()
        {
            assert_eq!(*nml, Vector3::zeros());
        }

        // Attrib already exists, we expect it to be left intact
        for nml in trimesh
            .attrib_or_add::<_, VertexIndex>("N", Vector3::<f64>::ones())
            .unwrap()
            .iter::<Vector3<f64>>()
            .unwrap()
        {
            assert_eq!(*nml, Vector3::zeros());
        }

        // Attrib already exists, we expect it to be left intact
        let verts: Vec<Vector3<f64>> = reinterpret_slice(trimesh.vertex_positions()).to_vec();
        for nml in trimesh
            .attrib_or_add_data::<_, VertexIndex>("N", verts.as_slice())
            .unwrap()
            .iter::<Vector3<f64>>()
            .unwrap()
        {
            assert_eq!(*nml, Vector3::zeros());
        }

        // Attrib doesn't yet exist
        let verts = trimesh.vertex_positions().to_vec();
        trimesh
            .attrib_or_add_data::<_, VertexIndex>("ref", verts.as_slice())
            .unwrap();
        for (r, p) in trimesh
            .attrib_iter::<[f64;3], VertexIndex>("ref")
            .unwrap()
            .zip(trimesh.vertex_positions())
        {
            assert_eq!(*r, *p);
        }
    }

    /// Test that multidimensional attributes can be added to the mesh from a `Vec`.
    /// Also test the `attrib_into_vec` and `attrib_as_slice` functions for multidimensional
    /// attributes.
    #[test]
    fn multidim_test() {
        let pts = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ];
        let indices = vec![0, 1, 2, 1, 3, 2];

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
                .attrib_into_vec::<[i8; 3], VertexIndex>("attrib1")
                .unwrap(),
            data
        );
        assert_eq!(
            trimesh
                .attrib_as_slice::<[i8; 3], VertexIndex>("attrib1")
                .unwrap(),
            data.as_slice()
        );

        trimesh.attrib_check::<VertexIndex>("attrib1").ok().unwrap();
        trimesh.remove_attrib::<VertexIndex>("attrib1").ok();
        trimesh
            .attrib_check::<VertexIndex>("attrib1")
            .err()
            .unwrap();
    }
}
