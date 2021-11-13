//!
//! Polymesh module. This module defines a mesh data structure cunstructed from unstructured
//! polygons of variable size.
//!

use crate::attrib::*;
use crate::mesh::topology::*;
use crate::mesh::vertex_positions::VertexPositions;
use crate::Real;

/// Mesh with arbitrarily shaped faces. It could have polygons with any number of sides.
/// All faces are assumed to be closed polygons.
#[derive(Clone, Debug, PartialEq, Attrib, Intrinsic)]
pub struct PolyMesh<T: Real> {
    /// Vertex positions intrinsic attribute.
    #[intrinsic(VertexPositions)]
    pub vertex_positions: IntrinsicAttribute<[T; 3], VertexIndex>,
    /// Indices into `vertices` representing face vertices.
    pub indices: Vec<usize>,
    /// Offsets into `indices` representing individual faces. The last element in this `Vec` is
    /// always the length of `indices` for convenience.
    pub offsets: Vec<usize>,
    /// Vertex attributes.
    pub vertex_attributes: AttribDict<VertexIndex>,
    /// Polygon attributes.
    pub face_attributes: AttribDict<FaceIndex>,
    /// Polygon vertex attributes.
    pub face_vertex_attributes: AttribDict<FaceVertexIndex>,
    /// Polygon edge attributes.
    pub face_edge_attributes: AttribDict<FaceEdgeIndex>,
    /// Indirect attribute value cache
    pub attribute_value_cache: AttribValueCache,
}

impl<T: Real> PolyMesh<T> {
    /// Construct a `PolyMesh` from an array of vertices and an array of sizes and indices.
    ///
    /// The `faces` array contains the indices into the vertex array for each face preceeded by the
    /// number of vertices in the corresponding face. I.e. `faces` is expected to be structured as
    /// a contiguous array of a number (corresponding to the number of vertices in the face)
    /// followed by the vertex indices (in the same face):
    /// ```verbatim
    ///     n i_1 i_2 ... i_n m j_1 j_2 ... j_m ...
    /// ```
    ///
    /// # Examples
    /// ```
    /// use meshx::mesh::PolyMesh;
    /// let points = vec![
    ///     [0.0, 0.0, 0.0],
    ///     [1.0, 0.0, 0.0],
    ///     [0.0, 1.0, 0.0],
    ///     [1.0, 1.0, 0.0],
    ///     [0.0, 0.0, 1.0],
    ///     [1.0, 0.0, 1.0]];
    /// let faces = vec![3, 0, 1, 2, // first triangle
    ///                  3, 1, 3, 2, // second triangle
    ///                  4, 0, 1, 5, 4]; // quadrilateral
    ///
    /// let polymesh = PolyMesh::new(points, &faces);
    ///
    /// assert_eq!(polymesh.indices, [0, 1, 2, 1, 3, 2, 0, 1, 5, 4]);
    /// assert_eq!(polymesh.offsets, [0, 3, 6, 10]);
    ///
    /// ```
    pub fn new(verts: Vec<[T; 3]>, faces: &[usize]) -> PolyMesh<T> {
        let mut indices = Vec::new();
        let mut offsets = Vec::new();
        let mut i = 0;
        while i < faces.len() {
            let n = faces[i];
            offsets.push(indices.len());
            i += 1;

            for k in 0..n {
                indices.push(faces[i + k]);
            }

            i += n;
        }

        offsets.push(indices.len());

        PolyMesh {
            vertex_positions: IntrinsicAttribute::from_vec(verts),
            indices,
            offsets,
            vertex_attributes: AttribDict::new(),
            face_attributes: AttribDict::new(),
            face_vertex_attributes: AttribDict::new(),
            face_edge_attributes: AttribDict::new(),
            attribute_value_cache: AttribValueCache::with_hasher(Default::default()),
        }
    }

    pub fn face_iter(&self) -> DynamicIndexSliceIter {
        DynamicIndexSliceIter {
            indices: &self.indices,
            offsets: &self.offsets,
        }
    }

    pub fn face_iter_mut(&mut self) -> DynamicIndexSliceIterMut {
        DynamicIndexSliceIterMut {
            indices: &mut self.indices,
            offsets: &self.offsets,
        }
    }

    /// Reverse the order of each polygon in this mesh.
    #[inline]
    pub fn reverse(&mut self) {
        for face in self.face_iter_mut() {
            face.reverse();
        }

        let num_faces = self.num_faces();

        let Self {
            face_vertex_attributes,
            face_edge_attributes,
            offsets,
            ..
        } = self;

        // TODO: Consider doing reversing lazily using a flag field.
        // Since each vertex has an associated face vertex attribute, we must remap those
        // as well.
        // Reverse face vertex attributes
        for (_, attrib) in face_vertex_attributes.iter_mut() {
            let mut data_slice = attrib.data_mut_slice();
            let mut face_offset = 0;
            for fidx in 0..num_faces {
                let num_verts = offsets[fidx + 1] - offsets[fidx];
                let mut i = 0usize;
                while i < num_verts / 2 {
                    data_slice.swap(face_offset + i, face_offset + num_verts - i - 1);
                    i += 1;
                }
                face_offset += num_verts;
            }
        }

        // Reverse face edge attributes
        for (_, attrib) in face_edge_attributes.iter_mut() {
            let mut data_slice = attrib.data_mut_slice();
            for fidx in 0..num_faces {
                let num_edges = offsets[fidx + 1] - offsets[fidx];
                let mut i = 0usize;
                while i < num_edges / 2 {
                    data_slice.swap(i, num_edges - i - 1);
                    i += 1;
                }

                // Orient edges so that sources coincide with the vertices.
                data_slice.subslice_mut(..num_edges).rotate_left(1);
                //bytes[..elem_size * num_edges].rotate_left(elem_size);

                data_slice = data_slice.into_subslice(num_edges..);
                //bytes = &mut bytes[elem_size * num_edges..];
            }
        }
    }

    /// Reverse the order of each polygon in this mesh. This is the consuming version of the
    /// `reverse` method.
    #[inline]
    pub fn reversed(mut self) -> PolyMesh<T> {
        self.reverse();
        self
    }
}

impl<T: Real> Default for PolyMesh<T> {
    /// Produce an empty mesh. This is not particularly useful on its own, however it can be
    /// used as a null case for various mesh algorithms.
    fn default() -> Self {
        PolyMesh::new(vec![], &[])
    }
}

impl<T: Real> NumVertices for PolyMesh<T> {
    #[inline]
    fn num_vertices(&self) -> usize {
        self.vertex_positions.len()
    }
}

impl<T: Real> NumFaces for PolyMesh<T> {
    #[inline]
    fn num_faces(&self) -> usize {
        self.offsets.len() - 1
    }
}

impl<T: Real> FaceVertex for PolyMesh<T> {
    #[inline]
    fn vertex<FVI>(&self, fv_idx: FVI) -> VertexIndex
    where
        FVI: Copy + Into<FaceVertexIndex>,
    {
        let fv_idx = usize::from(fv_idx.into());
        debug_assert!(fv_idx < self.num_face_vertices());
        self.indices[fv_idx].into()
    }

    #[inline]
    fn face_vertex<FI>(&self, fidx: FI, which: usize) -> Option<FaceVertexIndex>
    where
        FI: Copy + Into<FaceIndex>,
    {
        if which >= self.num_vertices_at_face(fidx) {
            None
        } else {
            let fidx = usize::from(fidx.into());
            debug_assert!(fidx < self.num_faces());

            Some((self.offsets[fidx] + which).into())
        }
    }

    #[inline]
    fn num_face_vertices(&self) -> usize {
        self.indices.len()
    }

    #[inline]
    fn num_vertices_at_face<FI>(&self, fidx: FI) -> usize
    where
        FI: Copy + Into<FaceIndex>,
    {
        let fidx = usize::from(fidx.into());
        self.offsets[fidx + 1] - self.offsets[fidx]
    }
}

impl<T: Real> FaceEdge for PolyMesh<T> {
    #[inline]
    fn edge<FEI>(&self, fe_idx: FEI) -> EdgeIndex
    where
        FEI: Copy + Into<FaceEdgeIndex>,
    {
        let fe_idx = usize::from(fe_idx.into());
        debug_assert!(fe_idx < self.num_face_edges());
        // Edges are assumed to be indexed the same as face vertices: the source of each
        // edge is the face vertex with the same index.
        self.indices[fe_idx].into()
    }

    #[inline]
    fn face_edge<FI>(&self, fidx: FI, which: usize) -> Option<FaceEdgeIndex>
    where
        FI: Copy + Into<FaceIndex>,
    {
        // Edges are assumed to be indexed the same as face vertices: the source of each
        // edge is the face vertex with the same index.
        if which >= self.num_edges_at_face(fidx) {
            None
        } else {
            let fidx = usize::from(fidx.into());
            Some((self.offsets[fidx] + which).into())
        }
    }

    #[inline]
    fn num_face_edges(&self) -> usize {
        self.indices.len()
    }

    #[inline]
    fn num_edges_at_face<FI>(&self, fidx: FI) -> usize
    where
        FI: Copy + Into<FaceIndex>,
    {
        let fidx = usize::from(fidx.into());
        self.offsets[fidx + 1] - self.offsets[fidx]
    }
}

pub struct DynamicIndexSliceIter<'a> {
    /// Vertex indices of a non-uniform mesh.
    indices: &'a [usize],
    /// Face offsets into the `indices` array as in `PolyMesh`.
    offsets: &'a [usize],
}

pub struct DynamicIndexSliceIterMut<'a> {
    /// Vertex indices of a non-uniform mesh.
    indices: &'a mut [usize],
    /// Face offsets into the `indices` array as in `PolyMesh`.
    offsets: &'a [usize],
}

impl<'a> Iterator for DynamicIndexSliceIter<'a> {
    type Item = &'a [usize];

    fn next(&mut self) -> Option<&'a [usize]> {
        match self.offsets.split_first() {
            Some((head, tail)) => {
                if tail.is_empty() {
                    return None;
                }
                self.offsets = tail;
                let n = tail[0] - *head;
                let (l, r) = self.indices.split_at(n);
                self.indices = r;
                Some(l)
            }
            None => {
                debug_assert!(false);
                None
            }
        }
    }
}

impl<'a> Iterator for DynamicIndexSliceIterMut<'a> {
    type Item = &'a mut [usize];

    fn next(&mut self) -> Option<&'a mut [usize]> {
        use std::mem;

        // get a unique mutable reference for indices
        let indices_slice = mem::take(&mut self.indices);
        match self.offsets.split_first() {
            Some((head, tail)) => {
                if tail.is_empty() {
                    return None;
                }
                self.offsets = tail;
                let n = tail[0] - *head;
                let (l, r) = indices_slice.split_at_mut(n);
                self.indices = r;
                Some(l)
            }
            None => {
                debug_assert!(false);
                None
            }
        }
    }
}

/// Convert a triangle mesh into a polygon mesh.
impl<T: Real> From<super::TriMesh<T>> for PolyMesh<T> {
    fn from(mesh: super::TriMesh<T>) -> PolyMesh<T> {
        let super::TriMesh {
            vertex_positions,
            indices,
            vertex_attributes,
            face_attributes,
            face_vertex_attributes,
            face_edge_attributes,
            .. // Missing vertex-face topo attributes
        } = mesh;

        let offsets = (0..=indices.len()).map(|x| 3 * x).collect();

        PolyMesh {
            vertex_positions,
            indices: flatk::Chunked3::from_array_vec(indices.into_vec()).into_inner(),
            offsets,
            vertex_attributes,
            face_attributes,
            face_vertex_attributes,
            face_edge_attributes,
            attribute_value_cache: AttribValueCache::default(),
        }
    }
}

/// Convert a quad mesh into a polygon mesh.
impl<T: Real> From<super::QuadMesh<T>> for PolyMesh<T> {
    fn from(mesh: super::QuadMesh<T>) -> PolyMesh<T> {
        let super::QuadMesh {
            vertex_positions,
            indices,
            vertex_attributes,
            face_attributes,
            face_vertex_attributes,
            face_edge_attributes,
            .. // Missing vertex-face topo attributes
        } = mesh;

        let offsets = (0..=indices.len()).map(|x| 4 * x).collect();

        PolyMesh {
            vertex_positions,
            indices: flatk::Chunked4::from_array_vec(indices.into_vec()).into_inner(),
            offsets,
            vertex_attributes,
            face_attributes,
            face_vertex_attributes,
            face_edge_attributes,
            attribute_value_cache: AttribValueCache::default(),
        }
    }
}

/// Convert a point cloud into a polygon mesh.
impl<T: Real> From<super::PointCloud<T>> for PolyMesh<T> {
    fn from(mesh: super::PointCloud<T>) -> PolyMesh<T> {
        let super::PointCloud {
            vertex_positions,
            vertex_attributes,
        } = mesh;

        PolyMesh {
            vertex_attributes,
            ..PolyMesh::new(vertex_positions.into(), &[])
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::Index;

    #[test]
    fn polymesh_test() {
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
            3, 1, 3, 2, // second triangle
            4, 0, 1, 5, 4, // quadrilateral
        ];

        let mesh = PolyMesh::new(points, &faces);
        assert_eq!(mesh.num_vertices(), 6);
        assert_eq!(mesh.num_faces(), 3);
        assert_eq!(mesh.num_face_vertices(), 10);
        assert_eq!(mesh.num_face_edges(), 10);

        assert_eq!(Index::from(mesh.face_to_vertex(1, 1)), 3);
        assert_eq!(Index::from(mesh.face_to_vertex(0, 2)), 2);
        assert_eq!(Index::from(mesh.face_edge(2, 3)), 9);
    }

    fn sample_points() -> Vec<[f64; 3]> {
        vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ]
    }

    #[test]
    fn from_trimesh_test() {
        use crate::mesh::TriMesh;
        let points = sample_points();
        let tri_faces = vec![
            3, 0, 1, 2, // first triangle
            3, 1, 3, 2, // second triangle
        ];

        let trimesh = TriMesh::new(points.clone(), vec![[0, 1, 2], [1, 3, 2]]);
        let tri_polymesh = PolyMesh::new(points.clone(), &tri_faces);
        assert_eq!(PolyMesh::from(trimesh), tri_polymesh);
    }

    #[test]
    fn from_quadmesh_test() {
        use crate::mesh::QuadMesh;
        let points = sample_points();
        let quad_faces = vec![
            4, 0, 1, 3, 2, // just one quad
        ];

        let quadmesh = QuadMesh::new(points.clone(), vec![[0, 1, 3, 2]]);
        let quad_polymesh = PolyMesh::new(points, &quad_faces);
        assert_eq!(PolyMesh::from(quadmesh), quad_polymesh);
    }

    #[test]
    fn from_pointcloud_test() {
        use crate::mesh::PointCloud;
        let points = sample_points();

        let ptcld = PointCloud::new(points.clone());
        let ptcld_polymesh = PolyMesh::new(points.clone(), &[]);
        assert_eq!(PolyMesh::from(ptcld), ptcld_polymesh);
    }
}
