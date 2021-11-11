use crate::index::Index;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::ops::{Add, Div, Mul, Rem, Sub};

/// The general structure of all types of mesh topologies is the mapping from one mesh component
/// (like a face) to another (like a vertex). One may implement any data layout convenient for them
/// but the interface shall remain the same.
/// In 3D a mesh can consist of:
///   cells (3D volumes),
///   faces (2D surfaces),
///   edges (1D curves) and
///   vertices (0D points)
/// Topology defines the mapping between each of these components

/// This trait identifies all indices that identify a particular element in a mesh, whether it is a
/// vertex or a triangle, or even a connectivity between triangles and vertices like a
/// triangle-vertex.
pub trait ElementIndex<T>:
    Copy
    + Clone
    + PartialEq
    + PartialOrd
    + Eq
    + Ord
    + From<T>
    + Into<T>
    + Add<Output = Self>
    + Add<T, Output = Self>
    + Sub<Output = Self>
    + Sub<T, Output = Self>
    + Mul<T, Output = Self>
    + Div<T, Output = Self>
    + Rem<T, Output = Self>
{
}

macro_rules! impl_index_type {
    ($index_type:ident) => {
        /// Define index type
        #[derive(Copy, Clone, Debug, PartialEq, PartialOrd, Eq, Ord, Hash)]
        #[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
        pub struct $index_type(usize);

        impl $index_type {
            #[inline]
            pub fn into_inner(self) -> usize {
                self.0
            }
        }

        impl ElementIndex<usize> for $index_type {}

        /*
         * Index arithmetic
         */

        impl Add for $index_type {
            type Output = $index_type;

            #[inline]
            fn add(self, rhs: $index_type) -> $index_type {
                $index_type(self.0 + rhs.0)
            }
        }

        impl Add<$index_type> for usize {
            type Output = $index_type;

            #[inline]
            fn add(self, rhs: $index_type) -> $index_type {
                rhs + self
            }
        }

        impl Add<usize> for $index_type {
            type Output = $index_type;

            #[inline]
            fn add(self, rhs: usize) -> $index_type {
                $index_type(self.0 + rhs)
            }
        }

        impl Sub for $index_type {
            type Output = $index_type;

            #[inline]
            fn sub(self, rhs: $index_type) -> $index_type {
                $index_type(self.0 - rhs.0)
            }
        }

        impl Sub<usize> for $index_type {
            type Output = $index_type;

            #[inline]
            fn sub(self, rhs: usize) -> $index_type {
                $index_type(self.0 - rhs)
            }
        }

        impl Sub<$index_type> for usize {
            type Output = $index_type;

            #[inline]
            fn sub(self, rhs: $index_type) -> $index_type {
                $index_type(self - rhs.0)
            }
        }

        impl Mul<usize> for $index_type {
            type Output = $index_type;

            #[inline]
            fn mul(self, rhs: usize) -> $index_type {
                $index_type(self.0 * rhs)
            }
        }

        impl Mul<$index_type> for usize {
            type Output = $index_type;

            #[inline]
            fn mul(self, rhs: $index_type) -> $index_type {
                rhs * self
            }
        }

        // It often makes sense to divide an index by a non-zero integer
        impl Div<usize> for $index_type {
            type Output = $index_type;

            #[inline]
            fn div(self, rhs: usize) -> $index_type {
                $index_type(self.0 / rhs)
            }
        }

        impl Rem<usize> for $index_type {
            type Output = $index_type;

            #[inline]
            fn rem(self, rhs: usize) -> $index_type {
                $index_type(self.0 % rhs)
            }
        }

        /*
         * Conversion to and from usize and Index (which is a checked index)
         */

        impl From<$index_type> for Index {
            #[inline]
            fn from(i: $index_type) -> Self {
                Index::new(i.0)
            }
        }

        impl From<Option<$index_type>> for Index {
            #[inline]
            fn from(mb_i: Option<$index_type>) -> Self {
                Index::from(mb_i.map(|i| i.0))
            }
        }

        impl From<Index> for $index_type {
            #[inline]
            fn from(i: Index) -> Self {
                $index_type(i.into_inner())
            }
        }

        impl From<$index_type> for usize {
            #[inline]
            fn from(i: $index_type) -> usize {
                i.0
            }
        }

        impl From<usize> for $index_type {
            #[inline]
            fn from(i: usize) -> Self {
                $index_type(i)
            }
        }
    };
}

// Index within a collection of meshes.
impl_index_type!(MeshIndex);

// Indices within a mesh.
impl_index_type!(VertexIndex);
impl_index_type!(EdgeIndex);
impl_index_type!(FaceIndex);
impl_index_type!(CellIndex);

// Indices within a mesh element.
impl_index_type!(EdgeVertexIndex);
impl_index_type!(FaceVertexIndex);
impl_index_type!(FaceEdgeIndex);
impl_index_type!(CellVertexIndex);
impl_index_type!(CellEdgeIndex);
impl_index_type!(CellFaceIndex);

impl_index_type!(VertexEdgeIndex);
impl_index_type!(VertexFaceIndex);
impl_index_type!(VertexCellIndex);
impl_index_type!(EdgeFaceIndex);
impl_index_type!(EdgeCellIndex);
impl_index_type!(FaceCellIndex);

/// TopoIndex is an index for data at a point of connectivity. For instance a `FaceVertexIndex`
/// identifies a specific vertex pointed to by a face. This means it has a source and a
/// destination. This trait defines the indices for this source and destination indices.
pub trait TopoIndex<I>: ElementIndex<I> {
    type SrcIndex: ElementIndex<I>;
    type DstIndex: ElementIndex<I>;
}

macro_rules! impl_topo_index {
    ($topo_index:ident, $from_index:ident, $to_index:ident) => {
        impl TopoIndex<usize> for $topo_index {
            type SrcIndex = $from_index;
            type DstIndex = $to_index;
        }
    };
}

impl_topo_index!(EdgeVertexIndex, EdgeIndex, VertexIndex);
impl_topo_index!(FaceVertexIndex, FaceIndex, VertexIndex);
impl_topo_index!(FaceEdgeIndex, FaceIndex, EdgeIndex);
impl_topo_index!(CellVertexIndex, CellIndex, VertexIndex);
impl_topo_index!(CellEdgeIndex, CellIndex, EdgeIndex);
impl_topo_index!(CellFaceIndex, CellIndex, FaceIndex);

impl_topo_index!(VertexEdgeIndex, VertexIndex, EdgeIndex);
impl_topo_index!(VertexFaceIndex, VertexIndex, FaceIndex);
impl_topo_index!(VertexCellIndex, VertexIndex, CellIndex);
impl_topo_index!(EdgeFaceIndex, EdgeIndex, FaceIndex);
impl_topo_index!(EdgeCellIndex, EdgeIndex, CellIndex);
impl_topo_index!(FaceCellIndex, FaceIndex, CellIndex);

// Simple quantifiers
pub trait NumMeshes {
    fn num_meshes(&self) -> usize;
}
pub trait NumVertices {
    fn num_vertices(&self) -> usize;
}
pub trait NumEdges {
    fn num_edges(&self) -> usize;
}
pub trait NumFaces {
    fn num_faces(&self) -> usize;
}
pub trait NumCells {
    fn num_cells(&self) -> usize;
}

// Topology interfaces

macro_rules! def_topo_trait {
    (__impl,
        $type:ident :
        $topo_fn:ident,
        $dst_fn:ident,
        $transport_fn:ident,
        $num_fn:ident,
        $num_at_fn:ident;
        $from_index:ident,
        $to_index:ident,
        $topo_index:ident
    ) => {
        /// Index of the destination element from the source index.
        fn $transport_fn<I>(&self, i: I, k: usize) -> Option<$to_index>
        where
            I: Copy + Into<$from_index>
        {
            self.$topo_fn(i, k).map(|j| self.$dst_fn(j))
        }

        /// Index of the destination element given the topology index.
        fn $dst_fn<I>(&self, i: I) -> $to_index
        where
            I: Copy + Into<$topo_index>;

        /// Toplogy index: where the data lives in an attribute array.
        fn $topo_fn<I>(&self, i: I, k: usize) -> Option<$topo_index>
        where
            I: Copy + Into<$from_index>;

        /// Topology quantifier. Number of connectors in total.
        fn $num_fn(&self) -> usize;

        /// Topology quantifier. Number of connectors at a particular element.
        fn $num_at_fn<I>(&self, i: I) -> usize
        where
            I: Copy + Into<$from_index>;
    };
    (
        $type:ident :
        $topo_fn:ident,
        $dst_fn:ident,
        $transport_fn:ident,
        $num_fn:ident,
        $num_at_fn:ident;
        $from_index:ident,
        $to_index:ident,
        $topo_index:ident
    ) => {
        pub trait $type {
            def_topo_trait!(__impl, $type: $topo_fn, $dst_fn, $transport_fn, $num_fn, $num_at_fn; $from_index, $to_index, $topo_index);
        }
    };
    (
        $type:ident :
        $topo_fn:ident,
        $dst_fn:ident,
        $transport_fn:ident,
        $num_fn:ident,
        $num_at_fn:ident;
        $from_index:ident,
        $to_index:ident,
        $topo_index:ident,
        $num_from_trait:ident ( $num_from_idx:ident ),
        $num_to_trait:ident ( $num_to_idx:ident )
    ) => {
        pub trait $type {
            def_topo_trait!(__impl, $type: $topo_fn, $dst_fn, $transport_fn, $num_fn, $num_at_fn; $from_index, $to_index, $topo_index);

            /// Generate the reverse topology structure.
            fn reverse_topo(&self) -> (Vec<usize>, Vec<usize>) where Self: $num_from_trait + $num_to_trait {
                let mut indices: Vec<Vec<usize>> = Vec::new();
                indices.resize(self.$num_to_idx(), Vec::new());
                for src_idx in 0..self.$num_from_idx() {
                    let mut which = 0;
                    while let Some(dest_idx) = self.$transport_fn(src_idx, which) {
                        indices[usize::from(dest_idx)].push(src_idx.into());
                        which += 1;
                    }
                }

                let mut offsets = Vec::with_capacity(self.$num_from_idx());
                offsets.push(0);
                for neighbours in indices.iter() {
                    let last = *offsets.last().unwrap();
                    offsets.push(last + neighbours.len());
                }

                (indices
                 .iter()
                 .flat_map(|x| x.iter().cloned())
                 .collect(),
                 offsets)
            }

            /// Generate the reverse topology structure from the destination element to the source
            /// topology element.
            ///
            /// For example the `reverse_source_topo` for `face->vertex` is
            /// `vetex->(face->vertex)`, where `face->vertex` is the 'source' of the original
            /// topology.
            fn reverse_source_topo(&self) -> (Vec<usize>, Vec<usize>) where Self: $num_to_trait {
                let mut indices: Vec<Vec<usize>> = Vec::new();
                indices.resize(self.$num_to_idx(), Vec::new());
                for topo_idx in 0..self.$num_fn() {
                    let dest_idx = self.$dst_fn(topo_idx);
                    indices[usize::from(dest_idx)].push(topo_idx.into());
                }

                let mut offsets = Vec::with_capacity(self.$num_fn());
                offsets.push(0);
                for neighbours in indices.iter() {
                    let last = *offsets.last().unwrap();
                    offsets.push(last + neighbours.len());
                }

                (indices
                 .iter()
                 .flat_map(|x| x.iter().cloned())
                 .collect(),
                 offsets)
            }
        }
    };
}

// Connectivity from Higher to Lower dimensional components (the normal way)
def_topo_trait!(EdgeVertex: edge_vertex, vertex, edge_to_vertex, num_edge_vertices, num_vertices_at_edge;
                EdgeIndex, VertexIndex, EdgeVertexIndex);
def_topo_trait!(FaceVertex: face_vertex, vertex, face_to_vertex, num_face_vertices, num_vertices_at_face;
                FaceIndex, VertexIndex, FaceVertexIndex, NumFaces ( num_faces ), NumVertices ( num_vertices ));
def_topo_trait!(FaceEdge: face_edge, edge, face_to_edge, num_face_edges, num_edges_at_face;
                FaceIndex, EdgeIndex, FaceEdgeIndex);
def_topo_trait!(CellVertex: cell_vertex, vertex, cell_to_vertex, num_cell_vertices, num_vertices_at_cell;
                CellIndex, VertexIndex, CellVertexIndex, NumCells ( num_cells ), NumVertices ( num_vertices ));
def_topo_trait!(CellEdge: cell_edge, edge, cell_to_edge, num_cell_edges, num_edges_at_cell;
                CellIndex, EdgeIndex, CellEdgeIndex);
def_topo_trait!(CellFace: cell_face, face, cell_to_face, num_cell_faces, num_faces_at_cell;
                CellIndex, FaceIndex, CellFaceIndex);

// Connectivity from Lower to Higher dimensional components (the other way)
def_topo_trait!(VertexEdge: vertex_edge, edge, vertex_to_edge, num_vertex_edges, num_edges_at_vertex;
                VertexIndex, EdgeIndex, VertexEdgeIndex);
def_topo_trait!(VertexFace: vertex_face, face, vertex_to_face, num_vertex_faces, num_faces_at_vertex;
                VertexIndex, FaceIndex, VertexFaceIndex);
def_topo_trait!(VertexCell: vertex_cell, cell, vertex_to_cell, num_vertex_cells, num_cells_at_vertex;
                VertexIndex, CellIndex, VertexCellIndex);
def_topo_trait!(EdgeFace: edge_face, face, edge_to_face, num_edge_faces, num_faces_at_edge;
                EdgeIndex, FaceIndex, EdgeFaceIndex);
def_topo_trait!(EdgeCell: edge_cell, cell, edge_to_cell, num_edge_cells, num_cells_at_edge;
                EdgeIndex, CellIndex, EdgeCellIndex);
def_topo_trait!(FaceCell: face_cell, cell, face_to_cell, num_face_cells, num_cells_at_face;
                FaceIndex, CellIndex, FaceCellIndex);
