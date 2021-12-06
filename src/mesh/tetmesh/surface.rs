use crate::attrib::*;
use crate::index::{CheckedIndex, Index};
use crate::mesh::topology::*;
use crate::mesh::*;
use crate::Real;

use super::TetMesh;

type HashMap<K, V> = hashbrown::HashMap<K, V>;

/// A triangle with sorted vertices
#[derive(Copy, Clone, PartialOrd, Ord, PartialEq, Eq, Hash, Debug)]
struct SortedTri {
    pub sorted_indices: [usize; 3],
}

impl SortedTri {
    fn new([a, b, c]: [usize; 3]) -> Self {
        SortedTri {
            sorted_indices: {
                if a <= b {
                    if b <= c {
                        [a, b, c]
                    } else if a <= c {
                        [a, c, b]
                    } else {
                        [c, a, b]
                    }
                } else if a <= c {
                    [b, a, c]
                } else if b <= c {
                    [b, c, a]
                } else {
                    [c, b, a]
                }
            },
        }
    }
}

/// A triangle face of a tetrahedron within a `TetMesh`.
#[derive(Copy, Clone, Eq)]
pub struct TetFace {
    /// Vertex indices in the source tetmesh forming this face.
    pub tri: [usize; 3],
    /// Index of the corresponding tet within the source tetmesh.
    pub tet_index: usize,
    /// Index of the face within the tet betweeen 0 and 4.
    pub face_index: usize,
}

impl TetFace {
    const PERMUTATIONS: [[usize; 3]; 6] = [
        [0, 1, 2],
        [1, 2, 0],
        [2, 0, 1],
        [0, 2, 1],
        [2, 1, 0],
        [1, 0, 2],
    ];
}

/// A utility function to index a slice using three indices, creating a new array of 3
/// corresponding entries of the slice.
fn tri_at<T: Copy>(slice: &[T], tri: &[usize; 3]) -> [T; 3] {
    [slice[tri[0]], slice[tri[1]], slice[tri[2]]]
}

/// Consider any permutation of the triangle to be equivalent to the original.
impl PartialEq for TetFace {
    fn eq(&self, other: &TetFace) -> bool {
        for p in Self::PERMUTATIONS.iter() {
            if tri_at(&other.tri, p) == self.tri {
                return true;
            }
        }

        false
    }
}

impl PartialOrd for TetFace {
    fn partial_cmp(&self, other: &TetFace) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Lexicographic ordering of the sorted indices.
impl Ord for TetFace {
    fn cmp(&self, other: &TetFace) -> std::cmp::Ordering {
        let mut tri = self.tri;
        tri.sort_unstable();
        let mut other_tri = other.tri;
        other_tri.sort_unstable();
        tri.cmp(&other_tri)
    }
}

impl<T: Real> TetMesh<T> {
    /// A helper function to compute surface topology of the tetmesh specified by the given cells.
    ///
    /// The algorithm is to iterate over every tet face and upon seeing a duplicate, remove it from
    /// the list. this will leave only unique faces, which correspond to the surface of the
    /// `TetMesh`.
    ///
    /// This function assumes that the given tetmesh is a manifold.
    fn surface_triangle_set<'a>(
        cells: impl std::iter::ExactSizeIterator<Item = &'a [usize; 4]>,
    ) -> HashMap<SortedTri, TetFace> {
        let mut triangles: HashMap<SortedTri, TetFace> = {
            // This will make surfacing tetmeshes deterministic.
            let hash_builder =
                hashbrown::hash_map::DefaultHashBuilder::with_seeds(7, 47, 2377, 719);
            HashMap::with_capacity_and_hasher(cells.len() * 4, hash_builder)
        };

        let add_tet_faces = |(i, cell): (usize, &[usize; 4])| {
            for (face_idx, tet_face) in Self::TET_FACES.iter().enumerate() {
                let face = TetFace {
                    tri: tri_at(cell, tet_face),
                    tet_index: i,
                    face_index: face_idx,
                };

                let key = SortedTri::new(face.tri);

                if triangles.remove(&key).is_none() {
                    triangles.insert(key, face);
                }
            }
        };

        cells.enumerate().for_each(add_tet_faces);
        triangles
    }

    /// Extracts the surface topology (triangles) of the `TetMesh`.
    ///
    /// This function assumes that the given tetmesh is a manifold.
    pub fn surface_topo(&self) -> Vec<[usize; 3]> {
        Self::surface_topo_from_tets(self.cell_iter())
    }

    /// Extracts the surface topology (triangles) from the given set of tetrahedral cells.
    ///
    /// This function assumes that the given tetrahedron topology is manifold.
    pub fn surface_topo_from_tets<'a>(
        cells: impl std::iter::ExactSizeIterator<Item = &'a [usize; 4]>,
    ) -> Vec<[usize; 3]> {
        Self::surface_triangle_set(cells)
            .into_iter()
            .map(|(_, elem)| elem.tri)
            .collect()
    }

    /// Extract the surface triangle information of the `TetMesh`.
    ///
    /// Only record those faces that are accepted by `filter`.
    ///
    /// This includes the triangle topology, which tet each triangle came from and which face on
    /// the originating tet it belongs to.  The returned vectors have the same size.
    ///
    /// This function assumes that the given tetmesh is a manifold.
    pub fn surface_triangle_data<F>(&self, filter: F) -> (Vec<[usize; 3]>, Vec<usize>, Vec<usize>)
    where
        F: FnMut(&TetFace) -> bool,
    {
        let triangles = Self::surface_triangle_set(self.cell_iter());

        let mut surface_topo = Vec::with_capacity(triangles.len());
        let mut tet_indices = Vec::with_capacity(triangles.len());
        let mut tet_face_indices = Vec::with_capacity(triangles.len());
        for face in triangles.into_iter().map(|(_, face)| face).filter(filter) {
            surface_topo.push(face.tri);
            tet_indices.push(face.tet_index);
            tet_face_indices.push(face.face_index);
        }

        (surface_topo, tet_indices, tet_face_indices)
    }

    /// Extract indices of vertices that are on the surface of this `TetMesh`.
    pub fn surface_vertices(&self) -> Vec<usize> {
        let mut verts = flatk::Chunked3::from_array_vec(self.surface_topo()).into_inner();
        verts.sort_unstable();
        verts.dedup();
        verts
    }

    /// Convert into a mesh of triangles representing the surface of this `TetMesh`.
    pub fn surface_trimesh(&self) -> TriMesh<T> {
        self.surface_trimesh_with_mapping_and_filter(None, None, None, None, |_| true)
    }

    /// Convert into a mesh of triangles representing the surface of this `TetMesh`.
    ///
    /// Additionally this function adds attributes that map the new triangle mesh to the original
    /// tetmesh.
    ///
    /// Note that if the given attribute name coincides with an existing vertex attribute, that
    /// attribute will be replaced with the original tetmesh vertex attribute.
    pub fn surface_trimesh_with_mapping(
        &self,
        original_vertex_index_name: Option<&str>,
        original_tet_index_name: Option<&str>,
        original_tet_vertex_index_name: Option<&str>,
        original_tet_face_index_name: Option<&str>,
    ) -> TriMesh<T> {
        self.surface_trimesh_with_mapping_and_filter(
            original_vertex_index_name,
            original_tet_index_name,
            original_tet_vertex_index_name,
            original_tet_face_index_name,
            |_| true,
        )
    }

    /// Convert into a mesh of triangles representing the surface of this `TetMesh`.
    ///
    /// Filter out surface triangles using the `filter` closure which takes a references to a
    /// `TetFace` representing the triangular face of a tetrahedron from this `TetMesh`.
    ///
    /// Additionally this function adds attributes that map the new triangle mesh to the original
    /// tetmesh.
    ///
    /// Note that if the given attribute name coincides with an existing vertex attribute, that
    /// attribute will be replaced with the original tetmesh vertex attribute.
    pub fn surface_trimesh_with_mapping_and_filter(
        &self,
        original_vertex_index_name: Option<&str>,
        original_tet_index_name: Option<&str>,
        original_tet_vertex_index_name: Option<&str>,
        original_tet_face_index_name: Option<&str>,
        filter: impl FnMut(&TetFace) -> bool,
    ) -> TriMesh<T> {
        // Get the surface topology of this tetmesh.
        let (mut topo, tet_indices, tet_face_indices) = self.surface_triangle_data(filter);

        // Record which vertices we have already handled.
        let mut seen = vec![-1isize; self.num_vertices()];

        // Record the mapping back to tet vertices.
        let mut original_vertex_index = Vec::with_capacity(topo.len());

        // Accumulate surface vertex positions for the new trimesh.
        let mut surf_vert_pos = Vec::with_capacity(topo.len());

        for face in topo.iter_mut() {
            for idx in face.iter_mut() {
                if seen[*idx] == -1 {
                    surf_vert_pos.push(self.vertex_positions[*idx]);
                    original_vertex_index.push(*idx);
                    seen[*idx] = (surf_vert_pos.len() - 1) as isize;
                }
                *idx = seen[*idx] as usize;
            }
        }

        surf_vert_pos.shrink_to_fit();
        original_vertex_index.shrink_to_fit();

        let num_surf_verts = surf_vert_pos.len();

        // Transfer vertex attributes.
        let mut vertex_attributes: AttribDict<VertexIndex> = AttribDict::new();

        for (name, attrib) in self.attrib_dict::<VertexIndex>().iter() {
            let new_attrib = attrib.duplicate_with_len(num_surf_verts, |mut new, old| {
                for (&idx, val) in seen.iter().zip(old.iter()) {
                    if idx != -1 {
                        new.get_mut(idx as usize).clone_from_other(val).unwrap();
                    }
                }
            });
            vertex_attributes.insert(name.to_string(), new_attrib);
        }

        // Transfer triangle attributes from tetrahedron attributes.
        let mut face_attributes: AttribDict<FaceIndex> = AttribDict::new();

        for (name, attrib) in self.attrib_dict::<CellIndex>().iter() {
            face_attributes.insert(
                name.to_string(),
                attrib.promote_with(|new, old| {
                    for &tet_idx in tet_indices.iter() {
                        new.push_cloned(old.get(tet_idx));
                    }
                }),
            );
        }

        // Mapping from face vertex index to its original tet vertex index.
        let mut tet_vertex_index = Vec::new();
        if original_tet_vertex_index_name.is_some() {
            tet_vertex_index.reserve(topo.len() * 3);
            for (&tet_idx, &tet_face_idx) in tet_indices.iter().zip(tet_face_indices.iter()) {
                let tri = &Self::TET_FACES[tet_face_idx];
                for &i in tri.iter() {
                    tet_vertex_index.push(self.cell_vertex(tet_idx, i));
                }
            }
        }

        // Transfer face vertex attributes from tetmesh.
        let mut face_vertex_attributes: AttribDict<FaceVertexIndex> = AttribDict::new();

        for (name, attrib) in self.attrib_dict::<CellVertexIndex>().iter() {
            face_vertex_attributes.insert(
                name.to_string(),
                attrib.promote_with(|new, old| {
                    for (&tet_idx, &tet_face_idx) in tet_indices.iter().zip(tet_face_indices.iter())
                    {
                        for &i in Self::TET_FACES[tet_face_idx].iter() {
                            let tet_vtx_idx = self.cell_vertex(tet_idx, i);
                            new.push_cloned(old.get(Index::from(tet_vtx_idx).unwrap()));
                        }
                    }
                }),
            );
        }

        let mut trimesh = TriMesh {
            vertex_positions: IntrinsicAttribute::from_vec(surf_vert_pos),
            indices: IntrinsicAttribute::from_vec(topo),
            vertex_attributes,
            face_attributes,
            face_vertex_attributes,
            face_edge_attributes: AttribDict::new(), // TetMeshes don't have edge attributes (yet)
            attribute_value_cache: self.attribute_value_cache.clone(),
        };

        // Add the mapping to the original tetmesh. Overwrite any existing attributes.
        if let Some(name) = original_vertex_index_name {
            trimesh
                .set_attrib_data::<_, VertexIndex>(name, original_vertex_index)
                .expect("Failed to add original vertex index attribute.");
        }

        if let Some(name) = original_tet_index_name {
            trimesh
                .set_attrib_data::<_, FaceIndex>(name, tet_indices)
                .expect("Failed to add original tet index attribute.");
        }

        if let Some(name) = original_tet_vertex_index_name {
            trimesh
                .set_attrib_data::<_, FaceVertexIndex>(name, tet_vertex_index)
                .expect("Failed to add original tet vertex index attribute.");
        }

        if let Some(name) = original_tet_face_index_name {
            trimesh
                .set_attrib_data::<_, FaceIndex>(name, tet_face_indices)
                .expect("Failed to add original tet face index attribute.");
        }

        trimesh
    }
}

impl<T: Real> From<TetMesh<T>> for TriMesh<T> {
    fn from(tetmesh: TetMesh<T>) -> TriMesh<T> {
        tetmesh.surface_trimesh()
    }
}

impl<T: Real> From<TetMesh<T>> for PolyMesh<T> {
    fn from(tetmesh: TetMesh<T>) -> PolyMesh<T> {
        PolyMesh::from(tetmesh.surface_trimesh())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tetmesh_surface_topo_test() {
        let points = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
        ];
        let indices = vec![[0, 4, 2, 5], [0, 5, 2, 3], [5, 3, 0, 1]];

        let mesh = TetMesh::new(points, indices);

        let surf_topo = mesh.surface_topo();
        let expected = vec![
            [0, 5, 4],
            [4, 5, 2],
            [0, 4, 2],
            [5, 3, 2],
            [0, 2, 3],
            [5, 1, 3],
            [3, 1, 0],
            [5, 0, 1],
        ];

        assert_eq!(surf_topo.len(), expected.len());
        for tri in surf_topo.iter() {
            assert!(expected.iter().any(|&x| x == *tri));
        }
    }

    #[test]
    fn tetmesh_surface_vertices_test() {
        let points = vec![
            [0.0, 0.0, 0.0],
            [1.0, -1.0, 1.0],
            [-1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [1.0, 1.0, -1.0],
        ];

        // Cube made from tets. There is one vertex in the center
        #[rustfmt::skip]
        let indices = vec![
            [0, 5, 8, 7],
            [6, 1, 0, 8],
            [6, 8, 0, 5],
            [0, 3, 8, 1],
            [0, 7, 8, 3],
            [0, 4, 3, 1],
            [0, 7, 3, 4],
            [0, 1, 2, 4],
            [0, 4, 2, 7],
            [0, 5, 2, 1],
            [0, 7, 2, 5],
            [6, 5, 0, 1],
        ];

        let mesh = TetMesh::new(points, indices);

        let surf_verts = mesh.surface_vertices();
        let expected = vec![1, 2, 3, 4, 5, 6, 7, 8]; // they are all on the surface,
        assert_eq!(surf_verts, expected);
    }

    #[test]
    fn tetmesh_surface_trimesh_test() {
        let points = vec![
            [0.0, 0.0, 0.0],
            [1.0, -1.0, 1.0],
            [-1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [1.0, 1.0, -1.0],
        ];

        // Cube made from tets. There is one vertex in the center
        #[rustfmt::skip]
        let indices = vec![
            [0, 5, 8, 7],
            [6, 1, 0, 8],
            [6, 8, 0, 5],
            [0, 3, 8, 1],
            [0, 7, 8, 3],
            [0, 4, 3, 1],
            [0, 7, 3, 4],
            [0, 1, 2, 4],
            [0, 4, 2, 7],
            [0, 5, 2, 1],
            [0, 7, 2, 5],
            [6, 5, 0, 1],
        ];

        let expected_pos = vec![
            [1.0, -1.0, 1.0],
            [-1.0, 1.0, 1.0],
            [-1.0, -1.0, 1.0],
            [-1.0, -1.0, -1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, -1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
        ];

        let expected_tris = vec![
            [0, 1, 2],
            [3, 0, 2],
            [1, 0, 4],
            [4, 0, 5],
            [6, 0, 3],
            [6, 5, 0],
            [1, 7, 2],
            [7, 3, 2],
            [7, 1, 4],
            [7, 4, 5],
            [6, 3, 5],
            [3, 7, 5],
        ];

        let expected_vtx_attrib = vec![1i32, 4, 2, 5, 3, 8, 6, 7];

        let expected_cell_attrib = vec![7u64, 9, 5, 3, 11, 1, 8, 10, 6, 4, 2, 0];

        let expected_cell_vtx_attrib = vec![
            [29usize, 31, 30],
            [37, 39, 38],
            [21, 23, 22],
            [13, 15, 14],
            [44, 47, 45],
            [4, 7, 5],
            [33, 35, 34],
            [41, 43, 42],
            [25, 27, 26],
            [17, 19, 18],
            [8, 11, 9],
            [1, 3, 2],
        ];

        let mut mesh = TetMesh::new(points, indices);

        let vtx_data = (0i32..mesh.num_vertices() as i32).collect();
        mesh.insert_attrib_data::<_, VertexIndex>("vtx_attrib", vtx_data)
            .unwrap();

        let cell_data = (0u64..mesh.num_cells() as u64).collect();
        mesh.insert_attrib_data::<_, CellIndex>("cell_attrib", cell_data)
            .unwrap();

        let cell_vtx_data = (0usize..mesh.num_cell_vertices() as usize).collect();
        mesh.insert_attrib_data::<_, CellVertexIndex>("cell_vtx_attrib", cell_vtx_data)
            .unwrap();

        let trimesh = mesh.surface_trimesh_with_mapping_and_filter(
            Some("vtx_idx"),
            Some("face_idx"),
            Some("face_vtx_idx"),
            Some("tet_face_idx"),
            |_| true,
        );

        assert_eq!(trimesh.num_vertices(), expected_pos.len());
        assert_eq!(trimesh.num_faces(), expected_tris.len());

        // The following tests are made to be resistant to hash function randomness.
        // Compute permutation for pos

        // Produce a vertex to expected vertex map.
        let mut exp_vtx_idx = vec![None; expected_pos.len()];

        for (ei, exp_pos) in expected_pos.iter().enumerate() {
            for (i, pos) in trimesh.vertex_position_iter().enumerate() {
                if exp_pos == pos {
                    exp_vtx_idx[i] = Some(ei);
                }
            }
        }

        // If the following unwrap fails it means the positions are mismatched.
        let exp_vtx_idx: Vec<_> = exp_vtx_idx.into_iter().map(|i| i.unwrap()).collect();

        // We can use this map to verify vertex attibutes.
        for (i, &val) in trimesh
            .attrib_iter::<i32, VertexIndex>("vtx_attrib")
            .unwrap()
            .enumerate()
        {
            assert_eq!(val, expected_vtx_attrib[exp_vtx_idx[i]]);
        }

        // Produce a face to expected face map.
        let mut exp_tri_idx = vec![None; expected_tris.len()];

        for (ei, exp_tri) in expected_tris.iter().enumerate() {
            for (i, tri) in trimesh.face_iter().enumerate() {
                let real_tri = [
                    exp_vtx_idx[tri[0]],
                    exp_vtx_idx[tri[1]],
                    exp_vtx_idx[tri[2]],
                ];
                if SortedTri::new(*exp_tri) == SortedTri::new(real_tri) {
                    exp_tri_idx[i] = Some(ei);
                }
            }
        }

        // If the following unwrap fails it means the triangles are mismatched.
        let exp_tri_idx: Vec<_> = exp_tri_idx.into_iter().map(|i| i.unwrap()).collect();

        // we can use this map to verify face attributes.
        for (i, &val) in trimesh
            .attrib_iter::<u64, FaceIndex>("cell_attrib")
            .unwrap()
            .enumerate()
        {
            assert_eq!(val, expected_cell_attrib[exp_tri_idx[i]]);
        }

        for (i, val) in trimesh
            .attrib_as_slice::<usize, FaceVertexIndex>("cell_vtx_attrib")
            .unwrap()
            .chunks(3)
            .enumerate()
        {
            let tri = [val[0], val[1], val[2]];
            assert_eq!(
                SortedTri::new(tri),
                SortedTri::new(expected_cell_vtx_attrib[exp_tri_idx[i]])
            );
        }
    }
}
