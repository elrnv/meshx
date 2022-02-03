mod extended;

pub use extended::*;

use crate::attrib::*;
use crate::mesh::topology::*;
use crate::mesh::vertex_positions::VertexPositions;
use crate::mesh::PolyMesh;
use crate::prim::Triangle;
use crate::utils::slice::*;
use crate::Real;
use std::slice::{Iter, IterMut};

/*
 * Commonly used meshes and their implementations.
 */

macro_rules! impl_uniform_surface_mesh {
    ($mesh_type:ident, $verts_per_face:expr) => {
        impl<T: Real> $mesh_type<T> {
            pub fn new(verts: Vec<[T; 3]>, indices: Vec<[usize; $verts_per_face]>) -> $mesh_type<T> {
                $mesh_type {
                    vertex_positions: IntrinsicAttribute::from_vec(verts),
                    indices: IntrinsicAttribute::from_vec(indices),
                    vertex_attributes: AttribDict::new(),
                    face_attributes: AttribDict::new(),
                    face_vertex_attributes: AttribDict::new(),
                    face_edge_attributes: AttribDict::new(),
                    attribute_value_cache: AttribValueCache::default(),
                }
            }

            /// Iterate over each face.
            pub fn face_iter(&self) -> Iter<[usize; $verts_per_face]> {
                self.indices.iter()
            }

            /// Iterate mutably over each face.
            pub fn face_iter_mut(&mut self) -> IterMut<[usize; $verts_per_face]> {
                self.indices.iter_mut()
            }

            /// Face accessor. These are vertex indices.
            ///
            /// # Panics
            ///
            /// This function panics when the given face index is out of bounds.
            #[inline]
            pub fn face<FI: Into<FaceIndex>>(&self, fidx: FI) -> &[usize; $verts_per_face] {
                &self.indices[fidx.into()]
            }

            /// Return a slice of individual faces.
            #[inline]
            pub fn faces(&self) -> &[[usize; $verts_per_face]] {
                self.indices.as_slice()
            }

            /// Reverse the order of each polygon in this mesh.
            #[inline]
            pub fn reverse(&mut self) {
                for face in self.face_iter_mut() {
                    face.reverse();
                }

                // TODO: Consider doing reversing lazily using a flag field.
                // Since each vertex has an associated face vertex attribute, we must remap those
                // as well.
                // Reverse face vertex attributes
                for (_, attrib) in self.face_vertex_attributes.iter_mut() {
                    let mut data_slice = attrib.data_mut_slice();
                    for mut slice in data_slice.chunks_exact_mut($verts_per_face) {
                        // TODO: implement reverse on SliceDrop
                        let mut i = 0usize;
                        while i < $verts_per_face / 2 {
                            slice.swap(i, $verts_per_face - i - 1);
                            i += 1;
                        }
                    }
                }

                // Reverse face edge attributes
                for (_, attrib) in self.face_edge_attributes.iter_mut() {
                    let mut data_slice = attrib.data_mut_slice();
                    for mut slice in data_slice.chunks_exact_mut($verts_per_face) {
                        let mut i = 0usize;
                        while i < $verts_per_face / 2 {
                            slice.swap(i, $verts_per_face - i - 1);
                            i += 1;
                        }

                        // Orient edges so that sources coincide with the vertices.
                        slice.rotate_left(1);
                    }
                }
            }

            /// Reverse the order of each polygon in this mesh. This is the consuming version of the
            /// `reverse` method.
            #[inline]
            pub fn reversed(mut self) -> Self {
                self.reverse();
                self
            }

            /// Sort vertices by the given key values, and return the reulting order (permutation).
            ///
            /// This function assumes we have at least one vertex.
            pub fn sort_vertices_by_key<K, F>(&mut self, mut f: F) -> Vec<usize>
            where
                F: FnMut(usize) -> K,
                K: Ord,
            {
                // Early exit.
                if self.num_vertices() == 0 {
                    return Vec::new();
                }

                let num = self.attrib_size::<VertexIndex>();
                debug_assert!(num > 0);

                // Original vertex indices.
                let mut order: Vec<usize> = (0..num).collect();

                // Sort vertex indices by the given key.
                order.sort_by_key(|k| f(*k));

                // Now sort all mesh data according to the sorting given by order.

                let $mesh_type {
                    ref mut vertex_positions,
                    ref mut indices,
                    ref mut vertex_attributes,
                    .. // face and face_{vertex,edge} attributes are unchanged
                } = *self;

                let mut seen = vec![false; vertex_positions.len()];

                // Apply the order permutation to vertex_positions in place
                apply_permutation_with_seen(&order, vertex_positions.as_mut_slice(), &mut seen);

                // Apply permutation to each vertex attribute
                for (_, attrib) in vertex_attributes.iter_mut() {
                    let mut data_slice = attrib.data_mut_slice();

                    // Clear seen
                    seen.iter_mut().for_each(|b| *b = false);

                    apply_permutation_with_seen(&order, &mut data_slice, &mut seen);
                }

                // Build a reverse mapping for convenience.
                let mut new_indices = vec![0; order.len()];
                for (new_idx, &old_idx) in order.iter().enumerate() {
                    new_indices[old_idx] = new_idx;
                }

                // Remap face vertices.
                for face in indices.iter_mut() {
                    for vtx_idx in face.iter_mut() {
                        *vtx_idx = new_indices[*vtx_idx];
                    }
                }

                order
            }
        }

        impl<T: Real> NumVertices for $mesh_type<T> {
            fn num_vertices(&self) -> usize {
                self.vertex_positions.len()
            }
        }

        impl<T: Real> NumFaces for $mesh_type<T> {
            fn num_faces(&self) -> usize {
                self.indices.len()
            }
        }

        impl<T: Real> FaceVertex for $mesh_type<T> {
            #[inline]
            fn vertex<FVI>(&self, fv_idx: FVI) -> VertexIndex
            where
                FVI: Copy + Into<FaceVertexIndex>,
            {
                let fv_idx = usize::from(fv_idx.into());
                debug_assert!(fv_idx < self.num_face_vertices());
                self.indices[fv_idx/$verts_per_face][fv_idx%$verts_per_face].into()
            }

            #[inline]
            fn face_vertex<FI>(&self, fidx: FI, which: usize) -> Option<FaceVertexIndex>
            where
                FI: Copy + Into<FaceIndex>,
            {
                if which >= $verts_per_face {
                    None
                } else {
                    let fidx = usize::from(fidx.into());
                    Some(($verts_per_face * fidx + which).into())
                }
            }

            #[inline]
            fn num_face_vertices(&self) -> usize {
                self.indices.len() * $verts_per_face
            }

            #[inline]
            fn num_vertices_at_face<FI>(&self, _: FI) -> usize
            where
                FI: Copy + Into<FaceIndex>,
            {
                $verts_per_face
            }
        }

        impl<T: Real> FaceEdge for $mesh_type<T> {
            #[inline]
            fn edge<FEI>(&self, fe_idx: FEI) -> EdgeIndex
            where
                FEI: Copy + Into<FaceEdgeIndex>,
            {
                // Edges are assumed to be indexed the same as face vertices: the source of each
                // edge is the face vertex with the same index.
                let fe_idx = usize::from(fe_idx.into());
                debug_assert!(fe_idx < self.num_face_vertices());
                self.indices[fe_idx/$verts_per_face][fe_idx%$verts_per_face].into()
            }
            #[inline]
            fn face_edge<FI>(&self, fidx: FI, which: usize) -> Option<FaceEdgeIndex>
            where
                FI: Copy + Into<FaceIndex>,
            {
                // Edges are assumed to be indexed the same as face vertices: the source of each
                // edge is the face vertex with the same index.
                if which >= $verts_per_face {
                    None
                } else {
                    let fidx = usize::from(fidx.into());
                    Some(($verts_per_face * fidx + which).into())
                }
            }
            #[inline]
            fn num_face_edges(&self) -> usize {
                self.indices.len() * $verts_per_face
            }
            #[inline]
            fn num_edges_at_face<FI>(&self, _: FI) -> usize
            where
                FI: Copy + Into<FaceIndex>,
            {
                $verts_per_face
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
pub struct LineMesh<T: Real> {
    /// Vertex positions.
    #[intrinsic(VertexPositions)]
    pub vertex_positions: IntrinsicAttribute<[T; 3], VertexIndex>,
    /// Pairs of indices into `vertices` representing line segments.
    pub indices: IntrinsicAttribute<[usize; 2], FaceIndex>,
    /// Vertex attributes.
    pub vertex_attributes: AttribDict<VertexIndex>,
    /// Line segment attributes.
    pub face_attributes: AttribDict<FaceIndex>,
    /// Line segment vertex attributes.
    pub face_vertex_attributes: AttribDict<FaceVertexIndex>,
    /// Line segment edge attributes.
    ///
    /// A line segment can be seen as having two directed edges.
    pub face_edge_attributes: AttribDict<FaceEdgeIndex>,
    /// Indirect attribute value cache
    pub attribute_value_cache: AttribValueCache,
}

#[derive(Clone, Debug, PartialEq, Attrib, Intrinsic)]
pub struct TriMesh<T: Real> {
    /// Vertex positions.
    #[intrinsic(VertexPositions)]
    pub vertex_positions: IntrinsicAttribute<[T; 3], VertexIndex>,
    /// Triples of indices into `vertices` representing triangles.
    pub indices: IntrinsicAttribute<[usize; 3], FaceIndex>,
    /// Vertex attributes.
    pub vertex_attributes: AttribDict<VertexIndex>,
    /// Triangle attributes.
    pub face_attributes: AttribDict<FaceIndex>,
    /// Triangle vertex attributes.
    pub face_vertex_attributes: AttribDict<FaceVertexIndex>,
    /// Triangle edge attributes.
    pub face_edge_attributes: AttribDict<FaceEdgeIndex>,
    /// Indirect attribute value cache
    pub attribute_value_cache: AttribValueCache,
}

#[derive(Clone, Debug, PartialEq, Attrib, Intrinsic)]
pub struct QuadMesh<T: Real> {
    /// Vertex positions.
    #[intrinsic(VertexPositions)]
    pub vertex_positions: IntrinsicAttribute<[T; 3], VertexIndex>,
    /// Quadruples of indices into `vertices` representing quadrilaterals.
    pub indices: IntrinsicAttribute<[usize; 4], FaceIndex>,
    /// Vertex attributes.
    pub vertex_attributes: AttribDict<VertexIndex>,
    /// Quad attributes.
    pub face_attributes: AttribDict<FaceIndex>,
    /// Quad vertex attributes.
    pub face_vertex_attributes: AttribDict<FaceVertexIndex>,
    /// Quad edge attributes.
    pub face_edge_attributes: AttribDict<FaceEdgeIndex>,
    /// Indirect attribute value cache
    pub attribute_value_cache: AttribValueCache,
}

impl_uniform_surface_mesh!(LineMesh, 2);
impl_uniform_surface_mesh!(TriMesh, 3);
impl_uniform_surface_mesh!(QuadMesh, 4);

impl<T: Real> TriMesh<T> {
    /// Triangle iterator.
    ///
    /// ```
    /// use meshx::mesh::TriMesh;
    /// use meshx::prim::Triangle;
    ///
    /// let verts = vec![[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]];
    /// let mesh = TriMesh::new(verts.clone(), vec![[0, 1, 2]]);
    /// let tri = Triangle::from_indexed_slice(&[0, 1, 2], verts.as_slice());
    /// assert_eq!(Some(tri), mesh.tri_iter().next());
    /// ```
    #[inline]
    pub fn tri_iter(&self) -> impl Iterator<Item = Triangle<T>> + '_ {
        self.face_iter().map(move |tri| self.tri_from_indices(tri))
    }

    /// Get a tetrahedron primitive corresponding to the given vertex indices.
    #[inline]
    pub fn tri_from_indices(&self, indices: &[usize; 3]) -> Triangle<T> {
        Triangle::from_indexed_slice(indices, self.vertex_positions.as_slice())
    }
}

/// Convert a triangle mesh to a polygon mesh.
// TODO: Improve this algorithm with ear clipping:
// https://www.geometrictools.com/Documentation/TriangulationByEarClipping.pdf
//
// Ear Clipping Notes:
// 1. Since Polygons are embedded in 3D instead of 2D, we choose to ignore potential intersections
//    of the ears with other ears. This can be resolved as a post process.
//    This means that criteria for bein an ear is only the angle (no vertex inclusion test).
// 2. Polygons have an orientation. We use this (Right-hand-rule) orientation to determine whether
//    a vertex is convex (<180 degrees) or reflex (>180 degrees).
//
// Ear Clipping Algorithm:
// ...TBD
impl<T: Real> From<PolyMesh<T>> for TriMesh<T> {
    fn from(mesh: PolyMesh<T>) -> TriMesh<T> {
        let mut tri_indices = Vec::with_capacity(mesh.num_faces());
        let mut tri_face_attributes: AttribDict<FaceIndex> = AttribDict::new();
        let mut tri_face_vertex_attributes: AttribDict<FaceVertexIndex> = AttribDict::new();
        let mut tri_face_edge_attributes: AttribDict<FaceEdgeIndex> = AttribDict::new();

        // A mapping back to vertices from the polymesh. This allows us to transfer face vertex
        // attributes.
        let mut poly_face_vert_map: Vec<usize> = Vec::with_capacity(mesh.num_face_vertices());

        // Triangulate
        for (face_idx, face) in mesh.face_iter().enumerate() {
            if face.len() < 3 {
                // Skipping line segments. These cannot be represented by a TriMesh without constructing degenerate triangles.
                continue;
            }

            let mut idx_iter = face.iter();
            let first_idx = idx_iter.next().unwrap();
            let mut second_idx = idx_iter.next().unwrap();
            let mut second = 1;

            for idx in idx_iter {
                tri_indices.push([*first_idx, *second_idx, *idx]);
                poly_face_vert_map.push(mesh.face_vertex(face_idx, 0).unwrap().into());
                poly_face_vert_map.push(mesh.face_vertex(face_idx, second).unwrap().into());
                second += 1;
                poly_face_vert_map.push(mesh.face_vertex(face_idx, second).unwrap().into());
                second_idx = idx;
            }
        }

        // Transfer face vertex attributes
        for (name, attrib) in mesh.attrib_dict::<FaceVertexIndex>().iter() {
            tri_face_vertex_attributes.insert(
                name.to_string(),
                attrib.duplicate_with(|new, old| {
                    for &poly_face_vtx_idx in poly_face_vert_map.iter() {
                        new.push_cloned(old.get(poly_face_vtx_idx));
                    }
                }),
            );
        }

        // Transfer face edge attributes
        // We use the face vertex map here because edges have the same topology as face vertices.
        for (name, attrib) in mesh.attrib_dict::<FaceEdgeIndex>().iter() {
            tri_face_edge_attributes.insert(
                name.to_string(),
                attrib.duplicate_with(|new, old| {
                    for &poly_face_edge_idx in poly_face_vert_map.iter() {
                        new.push_cloned(old.get(poly_face_edge_idx));
                    }
                }),
            );
        }

        // Transfer face attributes
        for (name, attrib) in mesh.attrib_dict::<FaceIndex>().iter() {
            tri_face_attributes.insert(
                name.to_string(),
                attrib.duplicate_with(|new, old| {
                    // Copy the attribute for every triangle originating from this polygon.
                    for (face, elem) in mesh.face_iter().zip(old.iter()) {
                        for _ in 2..face.len() {
                            new.push_cloned(elem.reborrow()).unwrap();
                        }
                    }
                }),
            );
        }

        let PolyMesh {
            vertex_positions,
            vertex_attributes,
            attribute_value_cache,
            ..
        } = mesh;

        TriMesh {
            vertex_positions,
            indices: IntrinsicAttribute::from_vec(tri_indices),
            vertex_attributes,
            face_attributes: tri_face_attributes,
            face_vertex_attributes: tri_face_vertex_attributes,
            face_edge_attributes: tri_face_edge_attributes,
            attribute_value_cache,
        }
    }
}

impl<T: Real> From<PolyMesh<T>> for LineMesh<T> {
    /// Convert a PolyMesh into a LineMesh. This is effectively a wireframe construction.
    ///
    /// Note that this conversion does not merge any attributes, each face will generate its own edge.
    /// This means that two neighbouring faces will generate two overlapping edges.
    /// This is done to preserve all attribute data during the conversion, which means that some of it is duplicated.
    // TODO: Add a specialized method on PolyMesh to generate a "slim" wireframe (with promoted attributes).
    fn from(mesh: PolyMesh<T>) -> LineMesh<T> {
        let mut indices = Vec::with_capacity(mesh.num_faces());
        let mut face_attributes: AttribDict<FaceIndex> = AttribDict::new();
        let mut face_vertex_attributes: AttribDict<FaceVertexIndex> = AttribDict::new();
        let mut face_edge_attributes: AttribDict<FaceEdgeIndex> = AttribDict::new();

        // A mapping back to vertices from the polymesh. This allows us to transfer face vertex
        // attributes.
        let mut poly_face_vert_map: Vec<usize> = Vec::with_capacity(mesh.num_face_vertices());

        // Triangulate
        for (face_idx, face) in mesh.face_iter().enumerate() {
            if face.len() < 2 {
                // Skipping single vertex polys. These cannot be represented by non-degenerate line segments
                // and add nothing to the visuals.
                continue;
            }

            let mut idx_iter = face.iter().enumerate().peekable();
            // We know there are at least 2 vertices (checked above) so the following unwrap will not panic.
            let &(_, &first_idx) = idx_iter.peek().unwrap();

            while let Some((i, idx)) = idx_iter.next() {
                if let Some((next_i, &next_idx)) = idx_iter.peek() {
                    indices.push([*idx, next_idx]);
                    poly_face_vert_map.push(mesh.face_vertex(face_idx, i).unwrap().into());
                    poly_face_vert_map.push(mesh.face_vertex(face_idx, *next_i).unwrap().into());
                } else if face.len() > 2 {
                    // We're at the last vertex. Connect it to the first vertex, but only if the poly has > 2 verts.
                    indices.push([*idx, first_idx]);
                    poly_face_vert_map.push(mesh.face_vertex(face_idx, i).unwrap().into());
                    poly_face_vert_map.push(mesh.face_vertex(face_idx, 0).unwrap().into());
                }
            }
        }

        // Transfer face vertex attributes
        for (name, attrib) in mesh.attrib_dict::<FaceVertexIndex>().iter() {
            face_vertex_attributes.insert(
                name.to_string(),
                attrib.duplicate_with(|new, old| {
                    for &poly_face_vtx_idx in poly_face_vert_map.iter() {
                        new.push_cloned(old.get(poly_face_vtx_idx));
                    }
                }),
            );
        }

        // Transfer face edge attributes
        // We use the face vertex map here because edges have the same topology as face vertices.
        for (name, attrib) in mesh.attrib_dict::<FaceEdgeIndex>().iter() {
            face_edge_attributes.insert(
                name.to_string(),
                attrib.duplicate_with(|new, old| {
                    for &poly_face_edge_idx in poly_face_vert_map.iter() {
                        new.push_cloned(old.get(poly_face_edge_idx));
                    }
                }),
            );
        }

        // Transfer face attributes
        for (name, attrib) in mesh.attrib_dict::<FaceIndex>().iter() {
            face_attributes.insert(
                name.to_string(),
                attrib.duplicate_with(|new, old| {
                    // Copy the attribute for every segment originating from this polygon.
                    for (face, elem) in mesh.face_iter().zip(old.iter()) {
                        if face.len() == 2 {
                            // A polygon with 2 vertices generates a single line segment.
                            new.push_cloned(elem.reborrow()).unwrap();
                        } else {
                            for _ in 0..face.len() {
                                new.push_cloned(elem.reborrow()).unwrap();
                            }
                        }
                    }
                }),
            );
        }

        let PolyMesh {
            vertex_positions,
            vertex_attributes,
            attribute_value_cache,
            ..
        } = mesh;

        LineMesh {
            vertex_positions,
            indices: IntrinsicAttribute::from_vec(indices),
            vertex_attributes,
            face_attributes,
            face_vertex_attributes,
            face_edge_attributes,
            attribute_value_cache,
        }
    }
}

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

        let mut trimesh = TriMesh::new(pts, indices);

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
            trimesh.indices.as_slice(),
            &[[3, 2, 1], [2, 4, 1], [3, 1, 0]]
        );
    }

    #[test]
    fn two_triangles() {
        let pts = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ];
        let indices = vec![[0, 1, 2], [1, 3, 2]];

        let trimesh = TriMesh::new(pts, indices);
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
    }

    /// Test converting from a `PolyMesh` into a `TriMesh`, which is a non-trivial operation since
    /// it involves trianguating polygons.
    #[test]
    fn from_polymesh() {
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
        let trimesh = TriMesh::new(
            points.clone(),
            vec![[0, 1, 2], [0, 1, 5], [0, 5, 4], [1, 3, 2]],
        );
        assert_eq!(trimesh, TriMesh::from(polymesh));
    }

    /// Test converting from a `PolyMesh` into a `TriMesh` with attributes.
    #[test]
    fn trimesh_from_polymesh_with_attrib() -> Result<(), Error> {
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

        let mut polymesh = crate::mesh::PolyMesh::new(points.clone(), &faces);
        polymesh.insert_attrib_data::<u64, VertexIndex>("v", vec![1, 2, 3, 4, 5, 6])?;
        polymesh.insert_attrib_data::<u64, FaceIndex>("f", vec![1, 2, 3])?;
        polymesh.insert_attrib_data::<u64, FaceVertexIndex>(
            "vf",
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        )?;
        polymesh
            .insert_attrib_data::<u64, FaceEdgeIndex>("ve", vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10])?;

        let mut trimesh = TriMesh::new(
            points.clone(),
            vec![[0, 1, 2], [0, 1, 5], [0, 5, 4], [1, 3, 2]],
        );
        trimesh.insert_attrib_data::<u64, VertexIndex>("v", vec![1, 2, 3, 4, 5, 6])?;
        trimesh.insert_attrib_data::<u64, FaceIndex>("f", vec![1, 2, 2, 3])?;
        trimesh.insert_attrib_data::<u64, FaceVertexIndex>(
            "vf",
            vec![1, 2, 3, 4, 5, 6, 4, 6, 7, 8, 9, 10],
        )?;
        trimesh.insert_attrib_data::<u64, FaceEdgeIndex>(
            "ve",
            vec![1, 2, 3, 4, 5, 6, 4, 6, 7, 8, 9, 10],
        )?;

        assert_eq!(trimesh, TriMesh::from(polymesh));
        Ok(())
    }

    /// Test converting from a `PolyMesh` into a `LineMesh` with attributes.
    #[test]
    fn linemesh_from_polymesh_with_attrib() -> Result<(), Error> {
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
            2, 1, 3, // line segment
        ];

        let mut polymesh = crate::mesh::PolyMesh::new(points.clone(), &faces);
        polymesh.insert_attrib_data::<u64, VertexIndex>("v", vec![1, 2, 3, 4, 5, 6])?;
        polymesh.insert_attrib_data::<u64, FaceIndex>("f", vec![1, 2, 3])?;
        polymesh
            .insert_attrib_data::<u64, FaceVertexIndex>("vf", vec![1, 2, 3, 4, 5, 6, 7, 8, 9])?;
        polymesh.insert_attrib_data::<u64, FaceEdgeIndex>("ve", vec![1, 2, 3, 4, 5, 6, 7, 8, 9])?;

        let mut linemesh = LineMesh::new(
            points.clone(),
            vec![
                [0, 1],
                [1, 2],
                [2, 0],
                [0, 1],
                [1, 5],
                [5, 4],
                [4, 0],
                [1, 3],
            ],
        );
        linemesh.insert_attrib_data::<u64, VertexIndex>("v", vec![1, 2, 3, 4, 5, 6])?;
        linemesh.insert_attrib_data::<u64, FaceIndex>("f", vec![1, 1, 1, 2, 2, 2, 2, 3])?;
        linemesh.insert_attrib_data::<u64, FaceVertexIndex>(
            "vf",
            vec![1, 2, 2, 3, 3, 1, 4, 5, 5, 6, 6, 7, 7, 4, 8, 9],
        )?;
        linemesh.insert_attrib_data::<u64, FaceEdgeIndex>(
            "ve",
            vec![1, 2, 2, 3, 3, 1, 4, 5, 5, 6, 6, 7, 7, 4, 8, 9],
        )?;
        assert_eq!(linemesh, LineMesh::from(polymesh));
        Ok(())
    }
}
