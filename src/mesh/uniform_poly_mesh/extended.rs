//!
//! Extended uniform poly meshes. This module defines an extension of uniform polygon meshes like
//! TriMesh and QuadMesh that are accompanied by their own dual topologies.
//!

use super::{QuadMesh, TriMesh};

use crate::attrib::*;
use crate::mesh::topology::*;
use crate::mesh::vertex_positions::VertexPositions;
use crate::prim::Triangle;
use crate::Real;
use std::slice::{Iter, IterMut};

/*
 * Commonly used meshes and their implementations.
 */

// Implement indexing for the Vec types used in meshes.
// The macro `impl_index_for` is defined in the `index` module.
impl_index_for!(Vec<usize> by FaceEdgeIndex, FaceVertexIndex, VertexCellIndex, VertexFaceIndex);

macro_rules! impl_uniform_surface_mesh {
    ($mesh_type:ident, $base_type:ident, $verts_per_face:expr) => {
        impl<T: Real> $mesh_type<T> {
            pub fn new(verts: Vec<[T; 3]>, indices: Vec<[usize; $verts_per_face]>) -> $mesh_type<T> {
                let (face_indices, face_offsets) =
                    Self::compute_dual_topology(verts.len(), &indices);
                $mesh_type {
                    base_mesh: $base_type::new(verts, indices),
                    face_indices,
                    face_offsets,
                }
            }

            /// Compute the `face_indices` and `face_offsets` fields of this uniform mesh.
            pub(crate) fn compute_dual_topology(
                num_verts: usize,
                indices: &[[usize; $verts_per_face]],
            ) -> (Vec<usize>, Vec<usize>) {
                let mut face_indices = Vec::new();
                face_indices.resize(num_verts, Vec::new());
                for (fidx, face) in indices.iter().enumerate() {
                    for &vidx in face {
                        face_indices[vidx].push(fidx);
                    }
                }

                let mut face_offsets = Vec::with_capacity(indices.len());
                face_offsets.push(0);
                for neighbours in face_indices.iter() {
                    let last = *face_offsets.last().unwrap();
                    face_offsets.push(last + neighbours.len());
                }

                (
                    face_indices
                        .iter()
                        .flat_map(|x| x.iter().cloned())
                        .collect(),
                    face_offsets,
                )
            }

            /// Iterate over each face.
            pub fn face_iter(&self) -> Iter<[usize; $verts_per_face]> {
                self.base_mesh.face_iter()
            }

            /// Iterate mutably over each face.
            pub fn face_iter_mut(&mut self) -> IterMut<[usize; $verts_per_face]> {
                self.base_mesh.face_iter_mut()
            }

            /// Face accessor. These are vertex indices.
            #[inline]
            pub fn face(&self, fidx: FaceIndex) -> &[usize; $verts_per_face] {
                self.base_mesh.face(fidx)
            }

            /// Return a slice of individual faces.
            #[inline]
            pub fn faces(&self) -> &[[usize; $verts_per_face]] {
                self.base_mesh.faces()
            }

            /// Reverse the order of each polygon in this mesh.
            #[inline]
            pub fn reverse(&mut self) {
                self.base_mesh.reverse()
            }

            /// Reverse the order of each polygon in this mesh. This is the consuming version of the
            /// `reverse` method.
            #[inline]
            pub fn reversed(mut self) -> Self {
                self.reverse();
                self
            }

            /// Sort vertices by the given key values.
            pub fn sort_vertices_by_key<K, F>(&mut self, f: F)
            where
                F: FnMut(usize) -> K,
                K: Ord,
            {
                // Ensure we have at least one vertex.
                if self.num_vertices() == 0 {
                    return;
                }

                let $mesh_type {
                                                                            ref mut base_mesh,
                                                                            ref mut face_indices,
                                                                            ref mut face_offsets,
                                                                            .. // face and face_{vertex,edge} attributes are unchanged
                                                                        } = *self;

                let order = base_mesh.sort_vertices_by_key(f);

                // Can't easily do this in place, so just for simplicity's sake we use extra memory
                // for transferring dual topology.

                let orig_face_indices = face_indices.clone();
                let orig_face_offsets = face_offsets.clone();

                // Note: The first and last offsets don't change.
                let mut prev_off = 0;
                for (idx, off) in face_offsets.iter_mut().skip(1).enumerate() {
                    let orig_idx = order[idx];
                    let prev_orig_off = orig_face_offsets[orig_idx];
                    let orig_off = orig_face_offsets[orig_idx + 1];
                    for (idx, &orig_idx) in face_indices[prev_off..]
                        .iter_mut()
                        .zip(orig_face_indices[prev_orig_off..orig_off].iter())
                    {
                        *idx = orig_idx;
                    }
                    *off = prev_off + orig_off - prev_orig_off;
                    prev_off = *off;
                }
            }
        }

        impl<T: Real> NumVertices for $mesh_type<T> {
            fn num_vertices(&self) -> usize {
                self.base_mesh.num_vertices()
            }
        }

        impl<T: Real> NumFaces for $mesh_type<T> {
            fn num_faces(&self) -> usize {
                self.base_mesh.num_faces()
            }
        }

        impl<T: Real> FaceVertex for $mesh_type<T> {
            #[inline]
            fn vertex<FVI>(&self, fv_idx: FVI) -> VertexIndex
            where
                FVI: Copy + Into<FaceVertexIndex>,
            {
                self.base_mesh.vertex(fv_idx)
            }

            #[inline]
            fn face_vertex<FI>(&self, fidx: FI, which: usize) -> Option<FaceVertexIndex>
            where
                FI: Copy + Into<FaceIndex>,
            {
                self.base_mesh.face_vertex(fidx.into(), which)
            }

            #[inline]
            fn num_face_vertices(&self) -> usize {
                self.base_mesh.num_face_vertices()
            }

            #[inline]
            fn num_vertices_at_face<FI>(&self, fidx: FI) -> usize
            where
                FI: Copy + Into<FaceIndex>,
            {
                self.base_mesh.num_vertices_at_face(fidx.into())
            }
        }

        impl<T: Real> FaceEdge for $mesh_type<T> {
            #[inline]
            fn edge<FEI>(&self, fe_idx: FEI) -> EdgeIndex
            where
                FEI: Copy + Into<FaceEdgeIndex>,
            {
                self.base_mesh.edge(fe_idx)
            }
            #[inline]
            fn face_edge<FI>(&self, fidx: FI, which: usize) -> Option<FaceEdgeIndex>
            where
                FI: Copy + Into<FaceIndex>,
            {
                self.base_mesh.face_edge(fidx.into(), which)
            }
            #[inline]
            fn num_face_edges(&self) -> usize {
                self.base_mesh.num_face_edges()
            }
            #[inline]
            fn num_edges_at_face<FI>(&self, fidx: FI) -> usize
            where
                FI: Copy + Into<FaceIndex>,
            {
                self.base_mesh.num_edges_at_face(fidx.into())
            }
        }

        impl<T: Real> VertexFace for $mesh_type<T> {
            #[inline]
            fn face<VFI>(&self, vf_idx: VFI) -> FaceIndex
            where
                VFI: Copy + Into<VertexFaceIndex>,
            {
                let vf_idx = usize::from(vf_idx.into());
                debug_assert!(vf_idx < self.num_vertex_faces());
                self.face_indices[vf_idx].into()
            }

            #[inline]
            fn vertex_face<VI>(&self, vidx: VI, which: usize) -> Option<VertexFaceIndex>
            where
                VI: Copy + Into<VertexIndex>,
            {
                if which >= self.num_faces_at_vertex(vidx) {
                    return None;
                }

                let vidx = usize::from(vidx.into());
                debug_assert!(vidx < self.num_vertices());

                Some((self.face_offsets[vidx] + which).into())
            }

            #[inline]
            fn num_vertex_faces(&self) -> usize {
                self.face_indices.len()
            }

            #[inline]
            fn num_faces_at_vertex<VI>(&self, vidx: VI) -> usize
            where
                VI: Copy + Into<VertexIndex>,
            {
                let vidx = usize::from(vidx.into());
                self.face_offsets[vidx + 1] - self.face_offsets[vidx]
            }
        }

        impl<T: Real> Default for $mesh_type<T> {
            /// Produce an empty mesh. This is not particularly useful on its own, however it can be
            /// used as a null case for various mesh algorithms.
            fn default() -> Self {
                $mesh_type::new(vec![], vec![])
            }
        }
    };
}

#[derive(Clone, Debug, PartialEq, Attrib, Intrinsic)]
pub struct TriMeshExt<T: Real> {
    /// Vertex positions.
    #[attributes(Vertex, Face, FaceVertex, FaceEdge)]
    #[intrinsics(VertexPositions::vertex_positions)]
    pub base_mesh: TriMesh<T>,
    /// Lists of face indices for each vertex. Since each vertex can have a variable number of face
    /// neighbours, the `face_offsets` field keeps track of where each subarray of indices begins.
    pub face_indices: Vec<usize>,
    /// Offsets into the `face_indices` array, one for each vertex. The last offset is always
    /// equal to the size of `face_indices` for convenience.
    pub face_offsets: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq, Attrib, Intrinsic)]
pub struct QuadMeshExt<T: Real> {
    /// Vertex positions.
    #[attributes(Vertex, Face, FaceVertex, FaceEdge)]
    #[intrinsics(VertexPositions::vertex_positions)]
    pub base_mesh: QuadMesh<T>,
    /// Lists of face indices for each vertex. Since each vertex can have a variable number of face
    /// neighbours, the `face_offsets` field keeps track of where each subarray of indices begins.
    pub face_indices: Vec<usize>,
    /// Offsets into the `face_indices` array, one for each vertex. The last offset is always
    /// equal to the size of `face_indices` for convenience.
    pub face_offsets: Vec<usize>,
}

impl_uniform_surface_mesh!(TriMeshExt, TriMesh, 3);
impl_uniform_surface_mesh!(QuadMeshExt, QuadMesh, 4);

impl<T: Real> TriMeshExt<T> {
    /// Triangle iterator.
    ///
    /// ```
    /// use meshx::mesh::TriMeshExt;
    /// use meshx::prim::Triangle;
    ///
    /// let verts = vec![[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]];
    /// let mesh = TriMeshExt::new(verts.clone(), vec![[0, 1, 2]]);
    /// let tri = Triangle::from_indexed_slice(&[0, 1, 2], verts.as_slice());
    /// assert_eq!(Some(tri), mesh.tri_iter().next());
    /// ```
    #[inline]
    pub fn tri_iter(&self) -> impl Iterator<Item = Triangle<T>> + '_ {
        self.base_mesh.tri_iter()
    }

    /// Get a tetrahedron primitive corresponding to the given vertex indices.
    #[inline]
    pub fn tri_from_indices(&self, indices: &[usize; 3]) -> Triangle<T> {
        self.base_mesh.tri_from_indices(indices)
    }
}

/// Convert a triangle mesh to a polygon mesh.
// TODO: Improve this algorithm with ear clipping:
// https://www.geometrictools.com/Documentation/TriangulationByEarClipping.pdf
impl<T: Real> From<super::PolyMesh<T>> for TriMeshExt<T> {
    fn from(mesh: super::PolyMesh<T>) -> TriMeshExt<T> {
        let base_mesh = TriMesh::from(mesh);

        let (face_indices, face_offsets) = TriMeshExt::<T>::compute_dual_topology(
            base_mesh.vertex_positions.len(),
            base_mesh.indices.as_slice(),
        );

        TriMeshExt {
            base_mesh,
            face_indices,
            face_offsets,
        }
    }
}

macro_rules! impl_mesh_convert {
    ($ext_mesh:ident <-> $base_mesh:ident) => {
        impl<T: Real> From<$base_mesh<T>> for $ext_mesh<T> {
            fn from(base_mesh: $base_mesh<T>) -> $ext_mesh<T> {
                // TODO: Refactor unnecessary unsafe block
                let flat_indices = bytemuck::cast_slice(base_mesh.indices.as_slice());
                let (face_indices, face_offsets) =
                    Self::compute_dual_topology(base_mesh.vertex_positions.len(), flat_indices);

                $ext_mesh {
                    base_mesh,
                    face_indices,
                    face_offsets,
                }
            }
        }

        impl<T: Real> From<$ext_mesh<T>> for $base_mesh<T> {
            fn from(ext: $ext_mesh<T>) -> $base_mesh<T> {
                ext.base_mesh
            }
        }
    };
}

impl_mesh_convert!(TriMeshExt <-> TriMesh);
impl_mesh_convert!(QuadMeshExt <-> QuadMesh);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::Index;

    #[test]
    fn mesh_sort() {
        // Sort -> check for inequality -> sort to original -> check for equality.

        let pts = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
        ];
        let indices = vec![[0, 1, 2], [1, 3, 2], [0, 2, 4]];

        let mut trimesh = TriMeshExt::new(pts, indices);

        let orig_trimesh = trimesh.clone();

        let values = [3, 2, 1, 4, 0];
        trimesh.sort_vertices_by_key(|k| values[k]);

        assert_ne!(trimesh, orig_trimesh);

        let rev_values = [4, 2, 1, 0, 3];
        trimesh.sort_vertices_by_key(|k| rev_values[k]);

        assert_eq!(trimesh, orig_trimesh);

        // Verify exact values.
        trimesh
            .insert_attrib_data::<usize, VertexIndex>("i", vec![0, 1, 2, 3, 4])
            .unwrap();

        trimesh.sort_vertices_by_key(|k| values[k]);

        assert_eq!(
            trimesh.vertex_positions(),
            &[
                [1.0, 1.0, 1.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
            ]
        );

        // `rev_values` actually already corresponds to 0..=4 being sorted by `values`.
        assert_eq!(
            trimesh.attrib_as_slice::<usize, VertexIndex>("i").unwrap(),
            &rev_values[..]
        );
        assert_eq!(
            trimesh.base_mesh.indices.as_slice(),
            &[[3, 2, 1], [2, 4, 1], [3, 1, 0]]
        );
    }

    #[test]
    fn two_triangles_test() {
        let pts = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ];
        let indices = vec![[0, 1, 2], [1, 3, 2]];

        let trimesh = TriMeshExt::new(pts, indices);
        assert_eq!(trimesh.num_vertices(), 4);
        assert_eq!(trimesh.num_faces(), 2);
        assert_eq!(trimesh.num_face_vertices(), 6);
        assert_eq!(trimesh.num_face_edges(), 6);

        assert_eq!(Index::from(trimesh.face_to_vertex(1, 1)), 3);
        assert_eq!(Index::from(trimesh.face_to_vertex(0, 2)), 2);
        assert_eq!(Index::from(trimesh.face_edge(1, 0)), 3);

        let mut face_iter = trimesh.face_iter();
        assert_eq!(face_iter.next(), Some(&[0usize, 1, 2]));
        assert_eq!(face_iter.next(), Some(&[1usize, 3, 2]));

        // Verify dual topology
        let vertex_faces = vec![vec![0], vec![0, 1], vec![0, 1], vec![1]];
        for i in 0..vertex_faces.len() {
            assert_eq!(trimesh.num_faces_at_vertex(i), vertex_faces[i].len());
            let mut local_faces: Vec<usize> = (0..trimesh.num_faces_at_vertex(i))
                .map(|j| trimesh.vertex_to_face(i, j).unwrap().into())
                .collect();
            local_faces.sort();
            assert_eq!(local_faces, vertex_faces[i]);
        }
    }

    /// Test converting from a `PolyMesh` into a `TriMeshExt`, which is a non-trivial operation since
    /// it involves trianguating polygons.
    #[test]
    fn from_polymesh_test() {
        let points = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
        ];
        let faces = vec![
            3, 0, 1, 2, // first triangle
            4, 0, 1, 5, 4, // quadrilateral
            3, 1, 3, 2, // second triangle
        ];

        let polymesh = crate::mesh::PolyMesh::new(points.clone(), &faces);
        let trimesh = TriMeshExt::new(
            points.clone(),
            vec![[0, 1, 2], [0, 1, 5], [0, 5, 4], [1, 3, 2]],
        );
        assert_eq!(trimesh, TriMeshExt::from(polymesh));
    }
}
