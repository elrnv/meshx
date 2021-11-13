//! This module implements the mechanism that allows users to index attributes for specific
//! topological locations using a type parameter like `VertexIndex`.

use super::{AttribDict, AttribValueCache};
use crate::mesh::topology::*;

/// This trait provides an interface for the implementer of `Attrib` to access attributes
/// associated with a specific topology within a mesh.
pub trait AttribIndex<M>
where
    Self: std::marker::Sized + Clone,
{
    /// Get the size of the attribute at the appropriate mesh location determined by `I`.
    fn attrib_size(mesh: &M) -> usize;

    /// Read only access to the attribute dictionary.
    fn attrib_dict(mesh: &M) -> &AttribDict<Self>;

    /// Read and write access to the attribute dictionary.
    fn attrib_dict_mut(mesh: &mut M) -> &mut AttribDict<Self>;

    /// Read and write access to the attribute dictionary as well as an optional cache for indirect
    /// attribute values.
    fn attrib_dict_and_cache_mut(
        mesh: &mut M,
    ) -> (&mut AttribDict<Self>, Option<&mut AttribValueCache>);
}

macro_rules! impl_attrib_index {
    ($topo_attrib:ident, $type:ty) => {
        /// Topology specific attribute implementation trait.
        ///
        /// This trait exists to allow one to access mesh attributes from `Attrib` using the
        /// `AttribIndex` trait. Additionally users can use this trait to filter meshes that have
        /// attributes at specific topological locations.
        pub trait $topo_attrib {
            /// Mesh implementation of the attribute size getter.
            fn topo_attrib_size(&self) -> usize;
            /// Mesh implementation of the attribute dictionary getter.
            fn topo_attrib_dict(&self) -> &AttribDict<$type>;
            /// Mesh implementation of the attribute dictionary mutable getter.
            fn topo_attrib_dict_mut(&mut self) -> &mut AttribDict<$type>;
            /// Mesh implementation of the attribute dictionary and cache mutable getter.
            fn topo_attrib_dict_and_cache_mut(
                &mut self,
            ) -> (&mut AttribDict<$type>, Option<&mut AttribValueCache>);
        }

        impl<M: $topo_attrib> AttribIndex<M> for $type {
            #[inline]
            fn attrib_size(mesh: &M) -> usize {
                mesh.topo_attrib_size()
            }

            #[inline]
            fn attrib_dict(mesh: &M) -> &AttribDict<Self> {
                mesh.topo_attrib_dict()
            }

            #[inline]
            fn attrib_dict_mut(mesh: &mut M) -> &mut AttribDict<Self> {
                mesh.topo_attrib_dict_mut()
            }

            #[inline]
            fn attrib_dict_and_cache_mut(
                mesh: &mut M,
            ) -> (&mut AttribDict<Self>, Option<&mut AttribValueCache>) {
                mesh.topo_attrib_dict_and_cache_mut()
            }
        }
    };
}

impl_attrib_index!(MeshAttrib, MeshIndex);
impl_attrib_index!(VertexAttrib, VertexIndex);
impl_attrib_index!(EdgeAttrib, EdgeIndex);
impl_attrib_index!(FaceAttrib, FaceIndex);
impl_attrib_index!(CellAttrib, CellIndex);
impl_attrib_index!(EdgeVertexAttrib, EdgeVertexIndex);
impl_attrib_index!(FaceVertexAttrib, FaceVertexIndex);
impl_attrib_index!(FaceEdgeAttrib, FaceEdgeIndex);
impl_attrib_index!(CellVertexAttrib, CellVertexIndex);
impl_attrib_index!(CellEdgeAttrib, CellEdgeIndex);
impl_attrib_index!(CellFaceAttrib, CellFaceIndex);
impl_attrib_index!(VertexEdgeAttrib, VertexEdgeIndex);
impl_attrib_index!(VertexFaceAttrib, VertexFaceIndex);
impl_attrib_index!(VertexCellAttrib, VertexCellIndex);
impl_attrib_index!(EdgeFaceAttrib, EdgeFaceIndex);
impl_attrib_index!(EdgeCellAttrib, EdgeCellIndex);
impl_attrib_index!(FaceCellAttrib, FaceCellIndex);
