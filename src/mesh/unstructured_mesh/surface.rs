use crate::{tri_at, CellType, Index, Mesh, Real, SortedTri, TetFace, TriMesh};

use crate::attrib::{Attrib, AttribDict, IntrinsicAttribute};
use crate::topology::{
    CellIndex, CellVertex, CellVertexIndex, FaceIndex, FaceVertexIndex, NumVertices, VertexIndex,
};
use ahash::AHashMap as HashMap;
use ahash::RandomState;
use flatk::ChunkedN;

/// A quad with sorted vertices
#[derive(Copy, Clone, PartialOrd, Ord, PartialEq, Eq, Hash, Debug)]
struct SortedQuad {
    pub sorted_indices: [usize; 4],
}

impl SortedQuad {
    fn new(indices: [usize; 4]) -> Self {
        let mut indices = indices.clone();
        indices.sort();
        SortedQuad {
            sorted_indices: indices,
        }
    }
}

/// A triangle face of a tetrahedron within a `TetMesh`.
#[derive(Copy, Clone, Eq)]
pub struct QuadFace {
    /// Vertex indices in the source mesh forming this face.
    pub quad: [usize; 4],
    /// Index of the corresponding quad within the source mesh.
    pub quad_index: usize,
    /// Index of the face within the cell
    pub face_index: usize,
    pub cell_type: CellType,
}

impl QuadFace {
    #[rustfmt::skip]
    const PERMUTATIONS: [[usize; 4]; 24] = [
        [1, 2, 3, 4], [2, 1, 3, 4], [3, 1, 2, 4], [1, 3, 2, 4],
        [2, 3, 1, 4], [3, 2, 1, 4], [3, 2, 4, 1], [2, 3, 4, 1],
        [4, 3, 2, 1], [3, 4, 2, 1], [2, 4, 3, 1], [4, 2, 3, 1],
        [4, 1, 3, 2], [1, 4, 3, 2], [3, 4, 1, 2], [4, 3, 1, 2],
        [1, 3, 4, 2], [3, 1, 4, 2], [2, 1, 4, 3], [1, 2, 4, 3],
        [4, 2, 1, 3], [2, 4, 1, 3], [1, 4, 2, 3], [4, 1, 2, 3],
    ];
}

/// A utility function to index a slice using four indices, creating a new array of 4
/// corresponding entries of the slice.
fn quad_at<T: Copy>(slice: &[T], quad: &[usize; 4]) -> [T; 4] {
    [
        slice[quad[0]],
        slice[quad[1]],
        slice[quad[2]],
        slice[quad[3]],
    ]
}

/// Consider any permutation of the triangle to be equivalent to the original.
impl PartialEq for QuadFace {
    fn eq(&self, other: &QuadFace) -> bool {
        for p in Self::PERMUTATIONS.iter() {
            if quad_at(&other.quad, p) == self.quad {
                return true;
            }
        }
        false
    }
}

impl PartialOrd for QuadFace {
    fn partial_cmp(&self, other: &QuadFace) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Lexicographic ordering of the sorted indices.
impl Ord for QuadFace {
    fn cmp(&self, other: &QuadFace) -> std::cmp::Ordering {
        let mut quad = self.quad;
        quad.sort_unstable();
        let mut other_quad = other.quad;
        other_quad.sort_unstable();
        quad.cmp(&other_quad)
    }
}

impl<T: Real> Mesh<T> {
    /// A helper function to compute surface topology of a generic mesh specified by the given cells.
    ///
    /// The algorithm is to iterate over every face and upon seeing a duplicate, remove it from
    /// the list. this will leave only unique faces, which correspond to the surface of the
    /// `Mesh`.
    ///
    /// This function assumes that the given Mesh is a manifold.
    fn surface_ngon_set<'a>(
        indices: &flatk::Clumped<Vec<usize>>,
        types: impl std::iter::ExactSizeIterator<Item = &'a CellType> + Clone,
    ) -> (HashMap<SortedTri, TetFace>, HashMap<SortedQuad, QuadFace>) {
        let mut tri_count = 0;
        let mut quad_count = 0;

        for (cells, cell_type) in indices.clump_iter().zip(types.clone()) {
            let cell_count = cells.view().data.len() * cells.view().chunk_size;
            tri_count += cell_type.num_tri_faces() * cell_count;
            quad_count += cell_type.num_quad_faces() * cell_count;
        }

        let mut triangles: HashMap<SortedTri, TetFace> = {
            // This will make surfacing tetmeshes deterministic.
            let hash_builder = RandomState::with_seeds(7, 47, 2377, 719);
            HashMap::with_capacity_and_hasher(tri_count * 3, hash_builder)
        };
        let mut quads: HashMap<SortedQuad, QuadFace> = {
            let hash_builder = RandomState::with_seeds(7, 47, 2377, 719);
            HashMap::with_capacity_and_hasher(quad_count * 4, hash_builder)
        };

        // returns the number of faces used so far: the index the next set of faces should start with.
        let mut add_tri_faces = |cells: &ChunkedN<&[usize]>,
                                 faces: &[[usize; 3]],
                                 starting_idx: usize,
                                 cell_type: CellType|
         -> usize {
            for (i, cell) in cells.iter().enumerate() {
                for (face_idx, tet_face) in faces.iter().enumerate() {
                    let face = TetFace {
                        tri: tri_at(cell, tet_face),
                        tet_index: i,
                        face_index: starting_idx + face_idx,
                        cell_type,
                    };

                    let key = SortedTri::new(face.tri);

                    if triangles.remove(&key).is_none() {
                        triangles.insert(key, face);
                    }
                }
            }
            starting_idx + faces.len()
        };
        let mut add_quad_faces = |cells: &ChunkedN<&[usize]>,
                                  faces: &[[usize; 4]],
                                  starting_idx: usize,
                                  cell_type: CellType|
         -> usize {
            for (i, cell) in cells.iter().enumerate() {
                for (face_idx, tet_face) in faces.iter().enumerate() {
                    let face = QuadFace {
                        quad: quad_at(cell, tet_face),
                        quad_index: i,
                        face_index: starting_idx + face_idx,
                        cell_type,
                    };

                    let key = SortedQuad::new(face.quad);

                    if quads.remove(&key).is_none() {
                        quads.insert(key, face);
                    }
                }
            }
            starting_idx + faces.len()
        };

        for (cells, cell_type) in indices.clump_iter().zip(types) {
            match cell_type {
                CellType::Triangle => {}
                CellType::Quad => {}
                CellType::Tetrahedron => {
                    add_tri_faces(&cells, &CellType::TETRAHEDRON_FACES, 0, *cell_type);
                }
                CellType::Pyramid => {
                    let i = add_tri_faces(&cells, &CellType::PYRAMID_TRIS, 0, *cell_type);
                    add_quad_faces(&cells, &[CellType::PYRAMID_QUAD], i, *cell_type);
                }
                CellType::Hexahedron => {
                    add_quad_faces(&cells, &CellType::HEXAHEDRON_FACES, 0, *cell_type);
                }
                CellType::Wedge => {
                    let i = add_tri_faces(&cells, &CellType::WEDGE_TRIS, 0, *cell_type);
                    add_quad_faces(&cells, &CellType::WEDGE_QUADS, i, *cell_type);
                }
            }
        }

        (triangles, quads)
    }

    /// Extract the surface ngon information of the `Mesh`.
    ///
    /// Only record those faces that are accepted by `filter`.
    ///
    /// This includes the ngon topology, which cell each ngon came from and which face on
    /// the originating cell it belongs to.  The returned vectors have the same size.
    ///
    /// This function assumes that the given mesh is a manifold.
    ///
    /// (triangles, quads, cells, cell_face_indices)
    pub fn surface_ngon_data<F1, F2>(
        &self,
        tri_filter: F1,
        quad_filter: F2,
    ) -> (
        Vec<[usize; 3]>,
        Vec<[usize; 4]>,
        Vec<usize>,
        Vec<usize>,
        Vec<CellType>,
    )
    where
        F1: FnMut(&TetFace) -> bool,
        F2: FnMut(&QuadFace) -> bool,
    {
        let (triangles, quads) = Self::surface_ngon_set(&self.indices, self.types.iter());

        let total = triangles.len() + quads.len();
        let mut surface_tris = Vec::with_capacity(triangles.len());
        let mut surface_quads = Vec::with_capacity(quads.len());
        let mut cell_indices = Vec::with_capacity(total);
        let mut cell_face_indices = Vec::with_capacity(total);
        let mut cell_types = Vec::with_capacity(total);
        for face in triangles
            .into_iter()
            .map(|(_, face)| face)
            .filter(tri_filter)
        {
            surface_tris.push(face.tri);
            cell_indices.push(face.tet_index);
            cell_face_indices.push(face.face_index);
            cell_face_indices.push(face.face_index);
        }
        for face in quads.into_iter().map(|(_, face)| face).filter(quad_filter) {
            surface_quads.push(face.quad);
            cell_indices.push(face.quad_index);
            cell_face_indices.push(face.face_index);
            cell_face_indices.push(face.face_index);
        }

        (
            surface_tris,
            surface_quads,
            cell_indices,
            cell_face_indices,
            cell_types,
        )
    }

    pub fn surface_mesh(&self) -> Mesh<T> {
        self.surface_trimesh_with_mapping_and_filter(None, None, None, None, |_| true, |_| true)
    }

    pub fn surface_trimesh_with_mapping(
        &self,
        original_vertex_index_name: Option<&str>,
        original_tet_index_name: Option<&str>,
        original_tet_vertex_index_name: Option<&str>,
        original_tet_face_index_name: Option<&str>,
    ) -> Mesh<T> {
        self.surface_trimesh_with_mapping_and_filter(
            original_vertex_index_name,
            original_tet_index_name,
            original_tet_vertex_index_name,
            original_tet_face_index_name,
            |_| true,
            |_| true,
        )
    }

    pub fn surface_trimesh_with_mapping_and_filter(
        &self,
        original_vertex_index_name: Option<&str>,
        original_tet_index_name: Option<&str>,
        original_tet_vertex_index_name: Option<&str>,
        original_tet_face_index_name: Option<&str>,
        tri_filter: impl FnMut(&TetFace) -> bool,
        quad_filter: impl FnMut(&QuadFace) -> bool,
    ) -> Mesh<T> {
        // Get the surface topology of this tetmesh.
        let (mut tri_topo, mut quad_topo, cell_indices, cell_face_indices, cell_types) =
            self.surface_ngon_data(tri_filter, quad_filter);

        // Record which vertices we have already handled.
        let mut seen = vec![-1isize; self.num_vertices()];

        // Record the mapping back to tet vertices.
        let mut original_vertex_index = Vec::with_capacity(tri_topo.len());

        // Accumulate surface vertex positions for the new trimesh.
        let mut surf_vert_pos = Vec::with_capacity(tri_topo.len());

        for face in tri_topo.iter_mut() {
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
                    for &tet_idx in cell_indices.iter() {
                        new.push_cloned(old.get(tet_idx));
                    }
                }),
            );
        }

        // Mapping from face vertex index to its original tet vertex index.
        /*let mut tet_vertex_index = Vec::new();
        if original_tet_vertex_index_name.is_some() {
            tet_vertex_index.reserve(topo.len() * 3);
            for (&tet_idx, &tet_face_idx) in cell_indices.iter().zip(cell_face_indices.iter()) {
                let tri = &Self::TET_FACES[tet_face_idx];
                for &i in tri.iter() {
                    tet_vertex_index.push(self.cell_vertex(tet_idx, i));
                }
            }
        }*/

        // Transfer face vertex attributes from tetmesh.
        let mut face_vertex_attributes: AttribDict<FaceVertexIndex> = AttribDict::new();

        for (name, attrib) in self.attrib_dict::<CellVertexIndex>().iter() {
            face_vertex_attributes.insert(
                name.to_string(),
                attrib.promote_with(|new, old| {
                    for (&tet_idx, &tet_face_idx, cell_type) in cell_indices
                        .iter()
                        .zip(cell_face_indices.iter())
                        .zip(cell_types.iter())
                        .map(|((a, b), c)| (a, b, c))
                    {
                        // todo: create an inverse mapping from the face idx.
                        //  the inverse of the mapping created during surface extraction.
                        //  Consider creating utility functions for this on the cell type.

                        // I think we need to know the cell type at this point...
                        /*for &i in Self::TET_FACES[tet_face_idx].iter() {
                            let tet_vtx_idx = self.cell_vertex(tet_idx, i);
                            new.push_cloned(old.get(Index::from(tet_vtx_idx).unwrap()));
                        }*/
                    }
                }),
            );
        }
        todo!();
        let mut trimesh = TriMesh {
            vertex_positions: IntrinsicAttribute::from_vec(surf_vert_pos),
            indices: IntrinsicAttribute::from_vec(tri_topo),
            vertex_attributes,
            face_attributes,
            face_vertex_attributes,
            face_edge_attributes: AttribDict::new(), // TetMeshes don't have edge attributes (yet)
            attribute_value_cache: self.attribute_value_cache.clone(),
        };

        // Add the mapping to the original tetmesh. Overwrite any existing attributes.
        /*if let Some(name) = original_vertex_index_name {
            trimesh
                .set_attrib_data::<_, VertexIndex>(name, original_vertex_index)
                .expect("Failed to add original vertex index attribute.");
        }

        if let Some(name) = original_tet_index_name {
            trimesh
                .set_attrib_data::<_, FaceIndex>(name, cell_indices)
                .expect("Failed to add original tet index attribute.");
        }

        if let Some(name) = original_tet_vertex_index_name {
            trimesh
                .set_attrib_data::<_, FaceVertexIndex>(name, tet_vertex_index)
                .expect("Failed to add original tet vertex index attribute.");
        }

        if let Some(name) = original_tet_face_index_name {
            trimesh
                .set_attrib_data::<_, FaceIndex>(name, cell_face_indices)
                .expect("Failed to add original tet face index attribute.");
        }*/

        trimesh;
    }
}
